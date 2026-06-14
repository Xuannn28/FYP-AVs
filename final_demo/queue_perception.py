# ============================================================
# queue_perception.py — Cooperative Queue Counting for Overtaking Decision
#
# Combines ego camera detections with RSU (Road-Side Unit) JSON detections
# to count the total number of vehicles in a queue ahead and decide
# whether it is safe to overtake.
#
# Pipeline:
#   1. Ego runs YOLO on its own rear-facing image
#   2. RSU detections loaded from JSON (produced by rsu_sender.py)
#   3. Shared anchor vehicle used to compute bbox-corner homography
#      (projects RSU bbox coordinates into ego camera space)
#   4. RSU boxes matched to ego boxes by IoU — matches are WBF-fused,
#      unmatched RSU boxes treated as hidden vehicles
#   5. Total queue count triggers overtaking decision
#
# Outputs saved to the script directory:
#   q01_ego_view.jpg   — Ego-only detection (driver perspective)
#   q02_fused_panel.jpg — 3-panel strip: ego | RSU data | fused result
#
# Usage:
#   py queue_perception.py \
#       --ego ego_queue.jpg --rsu-json rsu_detections.json
#
#   With mirror flag (RSU and ego on opposite sides of the queue):
#   py queue_perception.py \
#       --ego ego_queue.jpg --rsu-json rsu_detections.json --mirror
#
# Arguments:
#   --ego         Ego image — rear-facing driver view of queue   (required)
#   --rsu-json    RSU detections JSON from rsu_sender.py         (required)
#   --target      Comma-separated YOLO classes to count          (default: car,truck)
#   --threshold   Max vehicles allowed before DO NOT OVERTAKE    (default: 3)
#   --mirror      RSU and ego view the anchor from opposite ends
#                 (swaps left/right corner correspondences in projection)
#
# Decision logic:
#   total < threshold  → SAFE TO OVERTAKE
#   total = threshold  → BORDERLINE
#   total > threshold  → DO NOT OVERTAKE
#
# Fusion: Late Fusion + Weighted Box Fusion (WBF)
#   Solovyev et al., "Weighted Boxes Fusion", IVC 2021
#   Xu et al., "OPV2V", ICRA 2022
# ============================================================

import argparse
import json
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from ultralytics import YOLO
from config import MODEL_PATH, FRAME_WIDTH, FRAME_HEIGHT

try:
    from ensemble_boxes import weighted_boxes_fusion
    WBF_AVAILABLE = True
except ImportError:
    WBF_AVAILABLE = False
    print("[Warning] ensemble-boxes not installed. Falling back to highest-confidence selection.")

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

BG      = (28,  28,  36)
CARD    = (42,  42,  54)
WHITE   = (240, 240, 240)
GREY    = (150, 150, 160)
RED_BOX = (65,  80,  185)
GRN_BOX = (55,  155, 65)
CYN_BOX = (165, 148, 55)
COL_OK  = (50,  140, 65)
COL_WARN= (45,  130, 205)
COL_BAD = (65,  60,  175)

ANCHOR_CLASSES = {"truck", "bus", "car"}


# ── Drawing helpers ────────────────────────────────────────

def canvas(w, h):
    return np.full((h, w, 3), BG, dtype=np.uint8)

def txt(img, text, x, y, scale, color, bold=False):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2 if bold else 1, cv2.LINE_AA)

def rect(img, x1, y1, x2, y2, color):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

def bbox(img, x1, y1, x2, y2, color, t=2):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, t)

def dashed_bbox(img, x1, y1, x2, y2, color, dash=12, t=2):
    for ax, ay, bx, by in [(x1,y1,x2,y1),(x2,y1,x2,y2),(x2,y2,x1,y2),(x1,y2,x1,y1)]:
        length = int(np.hypot(bx - ax, by - ay))
        if length == 0:
            continue
        for s in range(0, length, dash * 2):
            e  = min(s + dash, length)
            p1 = (int(ax + (bx-ax)*s/length), int(ay + (by-ay)*s/length))
            p2 = (int(ax + (bx-ax)*e/length), int(ay + (by-ay)*e/length))
            cv2.line(img, p1, p2, color, t)

def badge(img, x1, y1, x2, y2, color, label, fs=0.60):
    rect(img, x1, y1, x2, y2, color)
    tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0]
    cx = x1 + (x2 - x1 - tw) // 2
    cy = y1 + (y2 - y1 + th) // 2
    cv2.putText(img, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, fs, WHITE, 2, cv2.LINE_AA)

def inline_label(img, text, x1, y1, color, fs=0.45, thickness=1):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, thickness)
    rect(img, x1, y1 - th - 6, x1 + tw + 10, y1 + 2, color)
    cv2.putText(img, text, (x1 + 4, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, fs, WHITE, thickness, cv2.LINE_AA)

def divider_line(img, y, color=(60, 60, 72)):
    cv2.line(img, (0, y), (img.shape[1], y), color, 1)


# ── Detection helpers ──────────────────────────────────────

def all_targets(results, model_names, target_classes, clip_floor=False):
    out = []
    for b in results.boxes:
        cls_name = model_names[int(b.cls)]
        if cls_name not in target_classes or float(b.conf) < 0.10:
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        if clip_floor:
            # Rear-facing ego view: cars are wider than tall; cap floor bleed-in
            w = max(x2 - x1, 1)
            y2 = min(y2, y1 + int(w * 1.1), FRAME_HEIGHT - 1)
        if x2 - x1 >= 5 and y2 - y1 >= 5:
            out.append((float(b.conf), (x1, y1, x2, y2), cls_name))
    return out

def find_anchor(results, model_names):
    best, best_area = None, 0
    for box in results.boxes:
        cls = model_names[int(box.cls)]
        if cls in ANCHOR_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                w = max(x2 - x1, 1); h = max(y2 - y1, 1)
                best = {"class": cls, "bbox": (x1, y1, x2, y2),
                        "cx": (x1+x2)/2, "cy": (y1+y2)/2, "w": w, "h": h}
    return best

def find_anchor_from_boxes(boxes):
    best, best_area = None, 0
    for conf, (x1, y1, x2, y2), cls in boxes:
        if cls in ANCHOR_CLASSES:
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                w = max(x2 - x1, 1); h = max(y2 - y1, 1)
                best = {"class": cls, "bbox": (x1, y1, x2, y2),
                        "cx": (x1+x2)/2, "cy": (y1+y2)/2, "w": w, "h": h}
    return best

def load_rsu_json(path):
    with open(path) as f:
        data = json.load(f)
    boxes = []
    for d in data["detections"]:
        x1, y1, x2, y2 = d["bbox"]
        if x2 - x1 >= 5 and y2 - y1 >= 5:
            boxes.append((float(d["conf"]), (x1, y1, x2, y2), d["class"]))
    return boxes


# ── IoU ────────────────────────────────────────────────────

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    aA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    aB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / float(aA + aB - inter)


# ── WBF ────────────────────────────────────────────────────

def run_wbf(ego_hit, rsu_hit):
    boxes_list, scores_list, labels_list = [], [], []

    def norm(bb):
        x1, y1, x2, y2 = bb
        return [x1/FRAME_WIDTH, y1/FRAME_HEIGHT, x2/FRAME_WIDTH, y2/FRAME_HEIGHT]

    if ego_hit:
        boxes_list.append([norm(ego_hit[1])]); scores_list.append([ego_hit[0]]); labels_list.append([0])
    if rsu_hit:
        boxes_list.append([norm(rsu_hit[1])]); scores_list.append([rsu_hit[0]]); labels_list.append([0])
    if not boxes_list:
        return None
    if not WBF_AVAILABLE or len(boxes_list) == 1:
        c = scores_list[0][0]; nb = boxes_list[0][0]
        bb = (int(nb[0]*FRAME_WIDTH), int(nb[1]*FRAME_HEIGHT),
              int(nb[2]*FRAME_WIDTH), int(nb[3]*FRAME_HEIGHT))
        return c, bb
    fused_boxes, fused_scores, _ = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, iou_thr=0.4, skip_box_thr=0.01)
    if len(fused_boxes) == 0:
        return None
    fb = fused_boxes[0]
    bb = (int(fb[0]*FRAME_WIDTH), int(fb[1]*FRAME_HEIGHT),
          int(fb[2]*FRAME_WIDTH), int(fb[3]*FRAME_HEIGHT))
    return float(fused_scores[0]), bb


# ── Homography ─────────────────────────────────────────────

def bbox_corner_homography(anchor_rsu, anchor_ego, global_flipped=False):
    x1v, y1v, x2v, y2v = anchor_rsu["bbox"]
    x1e, y1e, x2e, y2e = anchor_ego["bbox"]
    src = np.float32([[x1v,y1v],[x2v,y1v],[x2v,y2v],[x1v,y2v]])
    if global_flipped:
        dst = np.float32([[x2e,y1e],[x1e,y1e],[x1e,y2e],[x2e,y2e]])
    else:
        dst = np.float32([[x1e,y1e],[x2e,y1e],[x2e,y2e],[x1e,y2e]])
    return cv2.getPerspectiveTransform(src, dst)

def project_bbox(bb, H):
    x1, y1, x2, y2 = bb
    aspect = max(y2 - y1, 1) / max(x2 - x1, 1)

    def _proj(px, py):
        p = np.array([[px],[py],[1.0]], dtype=np.float64)
        r = H @ p
        if abs(float(r[2])) < 1e-6:
            return None
        r /= float(r[2])
        return float(r[0]), float(r[1])

    fl = _proj(x1,y2); fr = _proj(x2,y2); fc = _proj((x1+x2)/2,y2)
    if fl is None or fr is None or fc is None:
        return None
    proj_w = max(abs(fr[0] - fl[0]), 10)
    proj_h = int(proj_w * aspect)
    cx, cy = int(fc[0]), int(fc[1])
    px1 = int(np.clip(cx - proj_w/2, 0, FRAME_WIDTH  - 1))
    px2 = int(np.clip(cx + proj_w/2, 0, FRAME_WIDTH  - 1))
    py2 = int(np.clip(cy,            0, FRAME_HEIGHT - 1))
    py1 = int(np.clip(cy - proj_h,   0, FRAME_HEIGHT - 1))
    if px2 - px1 < 5 or py2 - py1 < 5:
        return None
    return (px1, py1, px2, py2)


# ── Estimate hidden car positions as perspective-nested boxes ─

def estimate_hidden_positions(ego_boxes, hidden):
    """
    Concentric nested boxes inside the anchor bbox — Russian-doll style.
    Closest hidden vehicle = outermost box (drawn first).
    Furthest hidden vehicle = innermost box (drawn last, on top).
    All centered at anchor center.
    Returns positions in draw order: closest first → furthest last.
    """
    if not ego_boxes or not hidden:
        return []

    lead = max(ego_boxes, key=lambda b: (b[1][2]-b[1][0])*(b[1][3]-b[1][1]))
    _, (x1, y1, x2, y2), *_ = lead
    cx       = (x1 + x2) / 2 + (x2 - x1) * 0.12   # slightly right of anchor centre
    cy       = y1 + (y2 - y1) * 0.25               # upper quarter — vehicles ahead appear higher
    anchor_w = x2 - x1
    anchor_h = y2 - y1

    # hidden is sorted furthest-first — draw furthest as outermost (largest box)
    # Fixed proportions: outer 70% → middle 52% → inner 38%
    proportions = [0.70, 0.52, 0.38]
    n = len(hidden)

    result = []
    for i in range(n):
        p  = proportions[i] if i < len(proportions) else proportions[-1] * (0.75 ** (i - len(proportions) + 1))
        bw = max(int(anchor_w * p), 8)
        bh = max(int(anchor_h * p), 6)
        bx1 = int(np.clip(cx - bw / 2, x1 + 1, x2 - 1))
        bx2 = int(np.clip(cx + bw / 2, x1 + 1, x2 - 1))
        by1 = int(np.clip(cy - bh / 2, y1 + 1, y2 - 1))
        by2 = int(np.clip(cy + bh / 2, y1 + 1, y2 - 1))
        if bx2 - bx1 >= 5 and by2 - by1 >= 5:
            result.append((0.5, (bx1, by1, bx2, by2)))

    return result


# ── Match RSU boxes to ego boxes, WBF fuse matches ─────────

def match_and_fuse(ego_boxes, rsu_projected, iou_threshold=0.3):
    fused_visible = []  # (conf, bb, cls, is_wbf_fused)
    hidden        = []
    fusion_log    = []  # (cls, ego_conf, rsu_conf, fused_conf, status)
    matched_ego   = set()

    for rsu_box in rsu_projected:
        best_iou, best_i = 0, -1
        for i, ego_box in enumerate(ego_boxes):
            iou = compute_iou(rsu_box[1], ego_box[1])
            if iou > best_iou:
                best_iou, best_i = iou, i
        if best_iou >= iou_threshold:
            fused    = run_wbf(ego_boxes[best_i], rsu_box)
            ego_cls  = ego_boxes[best_i][2]
            ego_conf = ego_boxes[best_i][0]
            rsu_conf = rsu_box[0]
            if fused:
                fused_visible.append((fused[0], fused[1], ego_cls, True))
                fusion_log.append((ego_cls, ego_conf, rsu_conf, fused[0], "WBF Fused"))
            else:
                c, bb, cls = ego_boxes[best_i]
                fused_visible.append((c, bb, cls, True))
                fusion_log.append((cls, ego_conf, rsu_conf, c, "WBF Fused"))
            matched_ego.add(best_i)
        else:
            hidden.append(rsu_box)
            fusion_log.append((rsu_box[2], None, rsu_box[0], rsu_box[0], "RSU Only"))

    for i, box in enumerate(ego_boxes):
        if i not in matched_ego:
            c, bb, cls = box
            fused_visible.append((c, bb, cls, False))
            fusion_log.append((cls, c, None, c, "Ego Only"))

    hidden.sort(key=lambda h: (h[1][2] - h[1][0]) * (h[1][3] - h[1][1]), reverse=False)
    return fused_visible, hidden, fusion_log


# ── Decision ───────────────────────────────────────────────

def overtake_decision(total, threshold):
    if total < threshold:
        return "SAFE TO OVERTAKE",  COL_OK
    if total == threshold:
        return "BORDERLINE",        COL_WARN
    return "DO NOT OVERTAKE",       COL_BAD


# ── Image 1: Ego view ──────────────────────────────────────

def save_q01(ego_img, ego_boxes, target_class):
    PAD=36; PW=900; PH=675; TITLE_H=80; INFO_H=120
    W = PW + 2*PAD; H = TITLE_H + PH + INFO_H + PAD
    img   = canvas(W, H)
    panel = cv2.resize(ego_img, (PW, PH))
    sx = PW / FRAME_WIDTH; sy = PH / FRAME_HEIGHT

    for conf, bb, cls in ego_boxes:
        bbox(panel, int(bb[0]*sx), int(bb[1]*sy), int(bb[2]*sx), int(bb[3]*sy), RED_BOX, 3)
        inline_label(panel, f"{cls}  {conf:.2f}", int(bb[0]*sx), int(bb[1]*sy), RED_BOX)

    img[TITLE_H:TITLE_H+PH, PAD:PAD+PW] = panel
    txt(img, "Stage 1 - Ego View (Driver Perspective)", PAD, 44, 0.90, WHITE, bold=True)
    txt(img, "Ego can only see the car directly ahead - vehicles beyond are hidden", PAD, 70, 0.52, GREY)
    divider_line(img, TITLE_H - 4)

    cy = TITLE_H + PH + 12
    rect(img, PAD, cy, PAD+PW, cy+INFO_H-16, CARD)
    txt(img, f"Cars visible to ego: {len(ego_boxes)}", PAD+16, cy+44, 0.70, WHITE, bold=True)
    txt(img, "Hidden cars ahead unknown without cooperative partner", PAD+16, cy+80, 0.52, GREY)

    out = os.path.join(OUT_DIR, "q01_ego_view.jpg")
    cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[1/4] Saved -> {out}")


# ── Image 2: 3-panel fused decision ───────────────────────

def save_q02(ego_img, ego_boxes, rsu_boxes,
             fused_visible, hidden, target_class, total, threshold):
    PAD=30; GAP=18; PW=500; PH=400; TITLE_H=86; INFO_H=220
    BADGE_Y1 = 148; BADGE_Y2 = INFO_H - 10
    N = 3
    W = N*PW + (N-1)*GAP + 2*PAD
    PANEL_H = PH + INFO_H
    H = TITLE_H + PANEL_H + PAD

    img = canvas(W, H)
    txt(img, "Stage 2  -  Fused Result", PAD, 48, 0.95, WHITE, bold=True)
    txt(img, "Ego visible objects (solid red)     RSU hidden objects (dashed green, class labelled)",
        PAD, 74, 0.50, GREY)
    divider_line(img, TITLE_H - 4)

    sx = PW / FRAME_WIDTH; sy = PH / FRAME_HEIGHT

    def accent(card, color, h=4):
        cv2.rectangle(card, (0, 0), (PW, h), color, -1)

    # ── Panel A: Ego view ──
    pxA = PAD
    panelA = cv2.resize(ego_img, (PW, PH))
    for conf, bb, cls in ego_boxes:
        bbox(panelA, int(bb[0]*sx), int(bb[1]*sy), int(bb[2]*sx), int(bb[3]*sy), RED_BOX, 3)
        inline_label(panelA, f"{cls}  {conf:.2f}", int(bb[0]*sx), int(bb[1]*sy), RED_BOX)
    cardA = np.full((INFO_H, PW, 3), CARD, dtype=np.uint8)
    accent(cardA, RED_BOX)
    txt(cardA, "Panel A - Ego View",              12,  34, 0.68, WHITE, bold=True)
    txt(cardA, f"Ego sees:  {len(ego_boxes)} car(s)",  12,  68, 0.62, WHITE)
    txt(cardA, "Cars beyond the first are hidden",      12,  96, 0.54, GREY)
    txt(cardA, "No cooperative data yet",               12, 120, 0.54, GREY)
    badge(cardA, 8, BADGE_Y1, PW-8, BADGE_Y2, RED_BOX, "EGO ONLY", 0.60)
    img[TITLE_H:TITLE_H+PANEL_H, pxA:pxA+PW] = np.vstack([panelA, cardA])

    # ── Panel B: RSU JSON detections ──
    pxB = PAD + PW + GAP
    panelB = canvas(PW, PH)
    txt(panelB, "RSU Detections", 20, 44, 0.80, WHITE, bold=True)
    txt(panelB, "(loaded from JSON)", 20, 72, 0.50, GREY)
    for i, (conf, bb, cls) in enumerate(rsu_boxes):
        ry = 100 + i * 72
        if ry + 60 > PH:
            break
        rect(panelB, 12, ry, PW - 12, ry + 62, CARD)
        cv2.rectangle(panelB, (12, ry), (16, ry + 62), GRN_BOX, -1)
        txt(panelB, f"{i+1}.  {cls}", 28, ry + 26, 0.65, GRN_BOX, bold=True)
        txt(panelB, f"conf: {conf:.2f}   bbox: {bb}", 28, ry + 50, 0.46, GREY)
    cardB = np.full((INFO_H, PW, 3), CARD, dtype=np.uint8)
    accent(cardB, GRN_BOX)
    txt(cardB, "Panel B - RSU Data (JSON)",              12,  34, 0.68, WHITE, bold=True)
    txt(cardB, f"RSU sees:  {len(rsu_boxes)} car(s) total",  12,  68, 0.62, WHITE)
    txt(cardB, f"Hidden from ego:  {len(hidden)} car(s)",    12,  98, 0.62, GRN_BOX)
    badge(cardB, 8, BADGE_Y1, PW-8, BADGE_Y2, GRN_BOX, "RSU COOPERATIVE PARTNER", 0.56)
    img[TITLE_H:TITLE_H+PANEL_H, pxB:pxB+PW] = np.vstack([panelB, cardB])

    # ── Panel C: Fused view ──
    pxC = PAD + 2*(PW + GAP)
    panelC = cv2.resize(ego_img, (PW, PH))
    for conf, bb, cls, is_wbf in fused_visible:
        bbox(panelC, int(bb[0]*sx), int(bb[1]*sy), int(bb[2]*sx), int(bb[3]*sy), CYN_BOX, 3)
        inline_label(panelC, f"{cls}  {conf:.2f}", int(bb[0]*sx), int(bb[1]*sy), CYN_BOX, fs=0.44)
    for (_, bb, *_), (h_conf, __, h_cls) in zip(
            estimate_hidden_positions(fused_visible, hidden), hidden):
        dashed_bbox(panelC, int(bb[0]*sx), int(bb[1]*sy), int(bb[2]*sx), int(bb[3]*sy), GRN_BOX)
        inline_label(panelC, f"{h_cls}  {h_conf:.2f}", int(bb[0]*sx), int(bb[1]*sy), (45, 120, 45), fs=0.42)

    visible_cls_str = "  ".join(dict.fromkeys(v[2] for v in fused_visible)) or "none"
    hidden_cls_str  = "  ".join(dict.fromkeys(h[2] for h in hidden))        or "none"
    cardC = np.full((INFO_H, PW, 3), CARD, dtype=np.uint8)
    accent(cardC, CYN_BOX)
    txt(cardC, "Panel C - Fused Result",              12,  34, 0.68, WHITE,   bold=True)
    txt(cardC, f"Visible:  {visible_cls_str}",             12,  68, 0.62, CYN_BOX)
    txt(cardC, f"Hidden:  {hidden_cls_str}",               12,  98, 0.62, GRN_BOX)
    txt(cardC, f"Total detected:  {total} object(s)",      12, 128, 0.66, WHITE,   bold=True)
    badge(cardC, 8, BADGE_Y1, PW-8, BADGE_Y2, CYN_BOX, "EGO + RSU FUSED", 0.60)
    img[TITLE_H:TITLE_H+PANEL_H, pxC:pxC+PW] = np.vstack([panelC, cardC])

    out = os.path.join(OUT_DIR, "q02_fused_panel.jpg")
    cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[2/2] Saved -> {out}")


# ── Main ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Queue counting for overtaking decision")
    parser.add_argument("--ego",       required=True,       help="Ego image (back of queue, driver view)")
    parser.add_argument("--rsu-json",  required=True,       help="RSU detections JSON file")
    parser.add_argument("--target",    default="car,truck", help="Comma-separated YOLO classes to count (default: car,truck)")
    parser.add_argument("--threshold", type=int, default=3, help="Max cars before DO NOT OVERTAKE (default: 3)")
    parser.add_argument("--mirror",    action="store_true", help="RSU and ego view anchor from opposite sides")
    args = parser.parse_args()
    target_classes = {t.strip() for t in args.target.split(",")}

    ego_raw = cv2.imread(args.ego)
    if ego_raw is None: sys.exit(f"[Error] Cannot read: {args.ego}")

    ego_img = cv2.resize(ego_raw, (FRAME_WIDTH, FRAME_HEIGHT))

    print(f"[Queue] Loading model: {MODEL_PATH}")
    model       = YOLO(MODEL_PATH)
    model_names = model.names

    print(f"\n[Ego] Detecting {target_classes}...")
    ego_res   = model(ego_img, conf=0.10, verbose=False)[0]
    ego_boxes = all_targets(ego_res, model_names, target_classes, clip_floor=True)
    print(f"[Ego] Detected: {len(ego_boxes)} vehicle(s)")

    print(f"\n[RSU] Loading detections from: {args.rsu_json}")
    rsu_boxes = load_rsu_json(args.rsu_json)
    rsu_boxes = [(c, b, l) for c, b, l in rsu_boxes if l in target_classes]
    print(f"[RSU] Loaded: {len(rsu_boxes)} vehicle(s)")

    # Project RSU boxes into ego space via shared anchor
    anchor_ego = find_anchor(ego_res, model_names)
    anchor_rsu = find_anchor_from_boxes(rsu_boxes)
    rsu_projected = []

    # Only count the anchor vehicle from ego — it is definitionally the same-lane predecessor
    if anchor_ego:
        ego_boxes_count = [b for b in ego_boxes if b[1] == anchor_ego["bbox"]]
    else:
        ego_boxes_count = ego_boxes
    print(f"[Ego] Same-lane (anchor): {len(ego_boxes_count)} vehicle(s)")

    if anchor_ego and anchor_rsu:
        H = bbox_corner_homography(anchor_rsu, anchor_ego, global_flipped=args.mirror)
        print(f"\n[Proj] Anchor RSU {anchor_rsu['bbox']} -> ego {anchor_ego['bbox']}")
        print(f"[Proj] Mirror: {'MIRRORED' if args.mirror else 'same direction'}")
        # Ego's own car appears rightmost (highest x-centre) in RSU's side view
        non_anchor = [(c, b, l) for c, b, l in rsu_boxes if b != anchor_rsu["bbox"]]
        ego_rsu_bb = max(non_anchor, key=lambda x: (x[1][0]+x[1][2])/2)[1] if non_anchor else None

        for conf, bb, cls in rsu_boxes:
            if ego_rsu_bb is not None and bb == ego_rsu_bb:
                print(f"  {bb} -> skipped (ego's vehicle in RSU side view)")
                continue
            proj = project_bbox(bb, H)
            if proj:
                rsu_projected.append((conf, proj, cls))
                print(f"  {bb} -> {proj}")
            else:
                print(f"  {bb} -> out of frame, skipped")
    else:
        print("\n[Proj] No shared anchor found - using raw RSU coords (no projection)")
        rsu_projected = rsu_boxes

    # Match, fuse, count — using confidence-filtered ego boxes
    fused_visible, hidden, _ = match_and_fuse(ego_boxes_count, rsu_projected)
    total = len(fused_visible) + len(hidden)
    decision_label, _ = overtake_decision(total, args.threshold)

    print(f"\n[Queue] Ego sees     : {len(ego_boxes)} car(s)  ({len(ego_boxes_count)} same-lane anchor)")
    print(f"[Queue] RSU sees     : {len(rsu_boxes)} car(s)")
    print(f"[Queue] Hidden       : {len(hidden)} car(s)")
    print(f"[Queue] Total (fused): {total} car(s)")
    print(f"[Queue] Decision     : {decision_label}  (threshold={args.threshold})")

    print("\n[Queue] Saving output images...")
    save_q01(ego_img, ego_boxes, args.target)
    save_q02(ego_img, ego_boxes, rsu_boxes,
             fused_visible, hidden, args.target, total, args.threshold)

    print(f"\n{'='*52}")
    print(f"  OVERTAKING DECISION SUMMARY")
    print(f"{'='*52}")
    print(f"  Ego sees        : {len(ego_boxes_count)} car(s)  (same-lane anchor)")
    print(f"  RSU adds hidden : {len(hidden)} car(s)")
    print(f"  Total in queue  : {total} car(s)")
    print(f"  Threshold       : <= {args.threshold} cars to overtake")
    print(f"  Decision        : {decision_label}")
    print(f"{'='*52}")
    print("\nDone.")


if __name__ == "__main__":
    main()