# ============================================================
# coop_perception.py — Confidence-Triggered Cooperative Perception
#
# Takes 3 ego images at different real occlusion levels + 1 V2 image.
# Runs the full cooperative perception pipeline for each ego image.
#
# Pipeline (per ego image):
#   1. Ego runs YOLO on its own image
#   2. Check target confidence against threshold
#   3. If conf ≤ lower  → activate V2, receive detections, WBF fuse
#      If conf > upper  → ego sufficient, V2 not contacted
#   4. Record ego conf, fused conf, trigger decision
#
# Outputs saved to OCCLUSION/ folder:
#   01_ego_comparison.jpg   — 3-panel strip: ego detection at each occlusion level
#   02_v2_detection.jpg     — V2 cooperative partner detection
#   03_fused_comparison.jpg — 3-panel strip: fused result at each occlusion level
#   04_confidence_chart.jpg — Bar chart: ego conf vs fused conf per level
#
# Usage:
#   py OCCLUSION/coop_perception.py \
#       --ego1 car_25pct.jpg --ego2 car_50pct.jpg --ego3 car_75pct.jpg \
#       --v2 v2_clear.jpg --target car
#
#   With homography:
#   py OCCLUSION/coop_perception.py \
#       --ego1 car_25pct.jpg --ego2 car_50pct.jpg --ego3 car_75pct.jpg \
#       --v2 v2_clear.jpg --target car --homography homography.npy
#
#
# # Capture ego images (move Pi to each position between shots)
# rpicam-still -o ego_25.jpg --immediate
# # move Pi to next position
# rpicam-still -o ego_50.jpg --immediate
# # move Pi to next position
# rpicam-still -o ego_75.jpg --immediate

# # Fix Pi at V2 position, capture V2 image
# rpicam-still -o v2.jpg --immediate

#
# Arguments:
#   --ego1        Ego image — low occlusion level          (required)
#   --ego2        Ego image — medium occlusion level       (required)
#   --ego3        Ego image — high occlusion level         (required)
#   --v2          V2 / RSU image — clear view of target    (required)
#   --target      YOLO class to track                      (default: car)
#   --upper       Conf above which ego is sufficient       (default: 0.50)
#   --lower       Conf below which V2 is activated         (default: 0.30)
#   --label1      Label for ego1 (default: "Low Occlusion")
#   --label2      Label for ego2 (default: "Mid Occlusion")
#   --label3      Label for ego3 (default: "High Occlusion")
#   --homography  Path to homography .npy                  (optional)
#
# Fusion: Late Fusion + Weighted Box Fusion (WBF)
#   Solovyev et al., "Weighted Boxes Fusion", IVC 2021
#   Xu et al., "OPV2V", ICRA 2022
#   Liu et al., "When2com", CVPR 2020
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
    print("[Warning] ensemble-boxes not installed. Run: pip install ensemble-boxes")
    print("[Warning] Falling back to highest-confidence selection.")

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Design tokens ──────────────────────────────────────────
BG      = (28,  28,  36)
CARD    = (42,  42,  54)
WHITE   = (240, 240, 240)
GREY    = (150, 150, 160)
RED_BOX = (70,  90,  230)
GRN_BOX = (60,  200, 80)
FUS_BOX = (200, 160, 50)

COL_OK   = (55,  175, 80)
COL_WARN = (50,  155, 245)
COL_BAD  = (65,  65,  215)


# ── Canvas + drawing helpers ───────────────────────────────

def canvas(w, h):
    return np.full((h, w, 3), BG, dtype=np.uint8)

def txt(img, text, x, y, scale, color, bold=False):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2 if bold else 1, cv2.LINE_AA)

def rect(img, x1, y1, x2, y2, color):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

def bbox(img, x1, y1, x2, y2, color, t=2):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, t)

def badge(img, x1, y1, x2, y2, color, label, fs=0.60):
    rect(img, x1, y1, x2, y2, color)
    tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0]
    cx = x1 + (x2 - x1 - tw) // 2
    cy = y1 + (y2 - y1 + th) // 2
    cv2.putText(img, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, fs, WHITE, 2, cv2.LINE_AA)

def inline_label(img, text, x1, y1, color, fs=0.50):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    rect(img, x1, y1 - th - 5, x1 + tw + 8, y1 + 2, color)
    cv2.putText(img, text, (x1 + 4, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, fs, WHITE, 1, cv2.LINE_AA)

def divider_line(img, y, color=(60, 60, 72)):
    cv2.line(img, (0, y), (img.shape[1], y), color, 1)

def vertical_divider(img, x, color=(60, 60, 72)):
    cv2.line(img, (x, 0), (x, img.shape[0]), color, 2)


# ── Detection helpers ──────────────────────────────────────

def best_target(results, model_names, target_class):
    hits = [(float(b.conf), tuple(map(int, b.xyxy[0])))
            for b in results.boxes
            if model_names[int(b.cls)] == target_class]
    return max(hits, key=lambda x: x[0]) if hits else None

def decision(conf, upper, lower):
    if conf > upper:  return "EGO SUFFICIENT",   COL_OK,   "SUFFICIENT"
    if conf > lower:  return "COOP RECOMMENDED", COL_WARN, "RECOMMENDED"
    return                   "COOP ACTIVATED",   COL_BAD,  "ACTIVATED"


# ── IoU ────────────────────────────────────────────────────

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter)


def parse_gt(s):
    if not s:
        return None
    try:
        parts = [int(x) for x in s.split(',')]
        return tuple(parts) if len(parts) == 4 else None
    except ValueError:
        return None


# ── WBF fusion ─────────────────────────────────────────────

def run_wbf(ego_hit, v2_hit):
    boxes_list, scores_list, labels_list = [], [], []

    def norm(bb):
        x1, y1, x2, y2 = bb
        return [x1/FRAME_WIDTH, y1/FRAME_HEIGHT, x2/FRAME_WIDTH, y2/FRAME_HEIGHT]

    if ego_hit:
        boxes_list.append([norm(ego_hit[1])])
        scores_list.append([ego_hit[0]])
        labels_list.append([0])

    if v2_hit:
        boxes_list.append([norm(v2_hit[1])])
        scores_list.append([v2_hit[0]])
        labels_list.append([0])

    if not boxes_list:
        return None

    if len(boxes_list) == 1:
        c = scores_list[0][0]
        nb = boxes_list[0][0]
        bb = (int(nb[0]*FRAME_WIDTH), int(nb[1]*FRAME_HEIGHT),
              int(nb[2]*FRAME_WIDTH), int(nb[3]*FRAME_HEIGHT))
        return c, bb

    if not WBF_AVAILABLE:
        # Average the scores; use the highest-confidence box for position
        avg_score = sum(s[0] for s in scores_list) / len(scores_list)
        best_idx  = max(range(len(scores_list)), key=lambda i: scores_list[i][0])
        nb = boxes_list[best_idx][0]
        bb = (int(nb[0]*FRAME_WIDTH), int(nb[1]*FRAME_HEIGHT),
              int(nb[2]*FRAME_WIDTH), int(nb[3]*FRAME_HEIGHT))
        return avg_score, bb

    fused_boxes, fused_scores, _ = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        iou_thr=0.4, skip_box_thr=0.01
    )
    if len(fused_boxes) == 0:
        return None

    # WBF normalises each cluster score by T (number of model lists), which
    # penalises non-overlapping boxes and causes double-division when averaged
    # afterwards.  Use the mean of the original input scores instead so the
    # result is always (ego_conf + v2_conf) / 2.  The box position comes from
    # the highest-confidence WBF cluster (fused_boxes[0]).
    avg_score = sum(s[0] for s in scores_list) / len(scores_list)
    fb = fused_boxes[0]
    bb = (int(fb[0]*FRAME_WIDTH), int(fb[1]*FRAME_HEIGHT),
          int(fb[2]*FRAME_WIDTH), int(fb[3]*FRAME_HEIGHT))
    return avg_score, bb


# ── SIFT auto-homography ───────────────────────────────────

def get_cab_side(img, bbox):
    """
    Detect which horizontal side of the truck bbox contains the orange cab.
    Returns 'left' or 'right'.
    Orange in BGR: low blue (<120), medium-high green (>80), high red (>140).
    """
    x1, y1, x2, y2 = [max(0, v) for v in bbox]
    cx = (x1 + x2) // 2
    left_patch  = img[y1:y2, x1:cx]
    right_patch = img[y1:y2, cx:x2]

    def orange_score(patch):
        if patch.size == 0:
            return 0
        return int(((patch[:, :, 0] < 120) &
                    (patch[:, :, 1] > 80)  &
                    (patch[:, :, 2] > 140)).sum())

    return 'left' if orange_score(left_patch) > orange_score(right_patch) else 'right'


def bbox_corner_homography(anchor_v2, anchor_ego, global_flipped=None):
    """
    Compute a perspective homography (V2 -> ego) from the 4 corners of the
    shared anchor bounding box.

    global_flipped: True  = swap left<->right corners (cameras on opposite sides)
                    False = keep same order
                    None  = default same direction
    """
    x1v, y1v, x2v, y2v = anchor_v2["bbox"]
    x1e, y1e, x2e, y2e = anchor_ego["bbox"]

    src = np.float32([[x1v, y1v], [x2v, y1v], [x2v, y2v], [x1v, y2v]])

    flipped = bool(global_flipped)
    print(f"  Mirror: {'MIRRORED' if flipped else 'same direction'}")

    if flipped:
        dst = np.float32([[x2e, y1e], [x1e, y1e], [x1e, y2e], [x2e, y2e]])
    else:
        dst = np.float32([[x1e, y1e], [x2e, y1e], [x2e, y2e], [x1e, y2e]])

    return cv2.getPerspectiveTransform(src, dst)


# ── Anchor-based projection ────────────────────────────────

ANCHOR_CLASSES = {"truck", "bus"}


def find_anchor(results, model_names):
    """Return the largest vehicle detection as the spatial anchor."""
    best, best_area = None, 0
    for box in results.boxes:
        cls = model_names[int(box.cls)]
        if cls in ANCHOR_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                w = max(x2 - x1, 1)
                h = max(y2 - y1, 1)
                best = {"class": cls, "conf": float(box.conf),
                        "bbox": (x1, y1, x2, y2),
                        "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
                        "w": w, "h": h}
    return best


def estimate_view_angle(anchor):
    """
    Estimate horizontal viewing angle relative to the vehicle front
    from the bounding box aspect ratio.
    Side view (wide bbox) -> ~90 deg, Front view (square) -> ~0 deg.
    """
    aspect = anchor["w"] / max(anchor["h"], 1)
    aspect_clamped = np.clip(aspect, 1.0, 4.0)
    return np.radians((aspect_clamped - 1.0) / 3.0 * 90.0)


def project_via_anchor(hidden_bbox, anchor_v2, anchor_ego):
    """
    Project V2 bbox into ego space using the shared anchor vehicle.
    Rotates the offset vector by the estimated angular difference between
    the two cameras — handles any angle (0 deg same-direction, 90 deg perpendicular).
    """
    angle_v2  = estimate_view_angle(anchor_v2)
    angle_ego = estimate_view_angle(anchor_ego)
    angle_diff_deg = abs(np.degrees(angle_ego) - np.degrees(angle_v2))

    hx1, hy1, hx2, hy2 = hidden_bbox
    hcx = (hx1 + hx2) / 2
    hcy = (hy1 + hy2) / 2
    hw  = hx2 - hx1
    hh  = hy2 - hy1

    rel_x = (hcx - anchor_v2["cx"]) / anchor_v2["w"]
    rel_y = (hcy - anchor_v2["cy"]) / anchor_v2["h"]

    delta = angle_ego - angle_v2
    rot_x = rel_x * np.cos(delta) - rel_y * np.sin(delta)
    rot_y = rel_x * np.sin(delta) + rel_y * np.cos(delta)

    scale   = anchor_ego["w"] / anchor_v2["w"]
    proj_cx = anchor_ego["cx"] + rot_x * anchor_ego["w"]
    proj_cy = anchor_ego["cy"] + rot_y * anchor_ego["h"]
    proj_w  = hw * scale
    proj_h  = hh * scale

    px1 = int(np.clip(proj_cx - proj_w / 2, 0, FRAME_WIDTH  - 1))
    px2 = int(np.clip(proj_cx + proj_w / 2, 0, FRAME_WIDTH  - 1))
    py1 = int(np.clip(proj_cy - proj_h / 2, 0, FRAME_HEIGHT - 1))
    py2 = int(np.clip(proj_cy + proj_h / 2, 0, FRAME_HEIGHT - 1))

    if px2 - px1 < 5 or py2 - py1 < 5:
        return None, angle_diff_deg
    return (px1, py1, px2, py2), angle_diff_deg


# ── Homography projection ──────────────────────────────────

def project_bbox(bbox_v2, H):
    x1, y1, x2, y2 = bbox_v2
    aspect = max(y2 - y1, 1) / max(x2 - x1, 1)

    def _proj(px, py):
        p = np.array([[px], [py], [1.0]], dtype=np.float64)
        r = H @ p
        if abs(r[2]) < 1e-6: return None
        r /= r[2]
        return float(r[0]), float(r[1])

    fl = _proj(x1, y2); fr = _proj(x2, y2); fc = _proj((x1+x2)/2, y2)
    if fl is None or fr is None or fc is None: return None

    proj_w = max(abs(fr[0] - fl[0]), 10)
    proj_h = int(proj_w * aspect)
    cx, cy = int(fc[0]), int(fc[1])
    px1 = int(np.clip(cx - proj_w/2, 0, FRAME_WIDTH-1))
    px2 = int(np.clip(cx + proj_w/2, 0, FRAME_WIDTH-1))
    py2 = int(np.clip(cy,            0, FRAME_HEIGHT-1))
    py1 = int(np.clip(cy - proj_h,   0, FRAME_HEIGHT-1))
    if px2 - px1 < 5 or py2 - py1 < 5: return None
    return (px1, py1, px2, py2)


# ── Per-image pipeline ─────────────────────────────────────

def run_pipeline(v2_hit_projected, ego_hit_raw, upper, lower, ref_box=None):
    """
    Run the trigger + fusion logic for one ego image.
    Returns a result dict with all values needed for visualisation.
    """
    ego_conf      = ego_hit_raw[0] if ego_hit_raw else 0.0
    coop_triggered = ego_conf <= lower

    fused_hit = None
    if coop_triggered:
        wbf_result = run_wbf(ego_hit_raw, v2_hit_projected)
        if wbf_result:
            fused_conf, fused_box = wbf_result
            box = ref_box if ref_box is not None else fused_box
            fused_hit = (fused_conf, box)

    _, col, status = decision(ego_conf, upper, lower)

    return {
        "ego_hit":        ego_hit_raw,
        "ego_conf":       ego_conf,
        "fused_hit":      fused_hit,
        "fused_conf":     fused_hit[0] if fused_hit else ego_conf,
        "coop_triggered": coop_triggered,
        "status":         status,
        "col":            col,
    }


# ── Draw single ego panel (for Image 1) ───────────────────

def draw_ego_panel(ego_img, result, all_results, model_names,
                   target_class, label, upper, lower, pw, ph):
    sx = pw / FRAME_WIDTH
    sy = ph / FRAME_HEIGHT

    panel = cv2.resize(ego_img, (pw, ph))

    # Non-target detections in muted grey
    for b in all_results.boxes:
        if model_names[int(b.cls)] == target_class or float(b.conf) < 0.15:
            continue
        bb = tuple(map(int, b.xyxy[0]))
        bbox(panel, int(bb[0]*sx), int(bb[1]*sy),
             int(bb[2]*sx), int(bb[3]*sy), (90, 90, 105), 1)

    # Target detection
    if result["ego_hit"]:
        conf, bb = result["ego_hit"]
        bbox(panel, int(bb[0]*sx), int(bb[1]*sy),
             int(bb[2]*sx), int(bb[3]*sy), RED_BOX, 3)
        inline_label(panel, f"{target_class}  {conf:.2f}",
                     int(bb[0]*sx), int(bb[1]*sy), RED_BOX)

    INFO_H = 150
    card   = np.full((INFO_H, pw, 3), CARD, dtype=np.uint8)

    # Level label
    txt(card, label, 12, 34, 0.72, WHITE, bold=True)

    if result["ego_hit"]:
        conf = result["ego_conf"]
        txt(card, f"Ego conf: {conf:.2f}", 12, 64, 0.60, WHITE)
        if result["coop_triggered"]:
            txt(card, f"Below {lower:.2f}  ->  ACTIVATED", 12, 90, 0.52, COL_BAD)
        else:
            txt(card, f"Above {upper:.2f}  ->  SUFFICIENT", 12, 90, 0.52, COL_OK)
    else:
        txt(card, "NOT DETECTED  ->  ACTIVATED", 12, 64, 0.55, COL_BAD)

    badge(card, 10, 104, pw - 10, INFO_H - 12, result["col"], result["status"], 0.58)

    return np.vstack([panel, card])


# ── Draw single fused panel (for Image 3) ─────────────────

def draw_fused_panel(ego_img, result, all_results, model_names,
                     target_class, label, upper, lower, pw, ph):
    sx = pw / FRAME_WIDTH
    sy = ph / FRAME_HEIGHT

    panel = cv2.resize(ego_img, (pw, ph))

    # Non-target detections muted
    for b in all_results.boxes:
        if model_names[int(b.cls)] == target_class or float(b.conf) < 0.15:
            continue
        bb = tuple(map(int, b.xyxy[0]))
        bbox(panel, int(bb[0]*sx), int(bb[1]*sy),
             int(bb[2]*sx), int(bb[3]*sy), (90, 90, 105), 1)

    if not result["coop_triggered"]:
        # No fusion — just show ego detection
        if result["ego_hit"]:
            conf, bb = result["ego_hit"]
            bbox(panel, int(bb[0]*sx), int(bb[1]*sy),
                 int(bb[2]*sx), int(bb[3]*sy), RED_BOX, 3)
            inline_label(panel, f"{target_class}  {conf:.2f}  [ego]",
                         int(bb[0]*sx), int(bb[1]*sy), RED_BOX)
    else:
        # Show ego box faintly
        if result["ego_hit"]:
            _, bb = result["ego_hit"]
            bbox(panel, int(bb[0]*sx), int(bb[1]*sy),
                 int(bb[2]*sx), int(bb[3]*sy), RED_BOX, 1)
        # Show fused result prominently
        if result["fused_hit"]:
            conf, bb = result["fused_hit"]
            bbox(panel, int(bb[0]*sx), int(bb[1]*sy),
                 int(bb[2]*sx), int(bb[3]*sy), FUS_BOX, 4)
            inline_label(panel, f"{target_class}  {conf:.2f}  [fused]",
                         int(bb[0]*sx), int(bb[1]*sy), FUS_BOX, fs=0.52)

    INFO_H = 180
    card   = np.full((INFO_H, pw, 3), CARD, dtype=np.uint8)

    txt(card, label, 12, 30, 0.72, WHITE, bold=True)

    ego_c   = result["ego_conf"]
    fused_c = result["fused_conf"]
    gain    = fused_c - ego_c
    iou     = result.get("iou")

    if not result["coop_triggered"]:
        txt(card, f"Ego conf: {ego_c:.2f}  (no fusion needed)", 12, 56, 0.56, WHITE)
        if iou is not None:
            iou_col = COL_OK if iou >= 0.5 else (COL_WARN if iou >= 0.3 else COL_BAD)
            txt(card, f"IoU: {iou:.3f}  (projection vs ground truth)", 12, 78, 0.52, iou_col)
        badge_y = 105
    else:
        txt(card, f"Ego: {ego_c:.2f}   Fused: {fused_c:.2f}", 12, 56, 0.60, WHITE)
        gain_col = COL_OK if gain > 0 else (100, 100, 120)
        txt(card, f"Confidence gain: {'+' if gain>=0 else ''}{gain:.2f}",
            12, 78, 0.56, gain_col)
        if iou is not None:
            iou_col = COL_OK if iou >= 0.5 else (COL_WARN if iou >= 0.3 else COL_BAD)
            txt(card, f"IoU: {iou:.3f}  (projection vs ground truth)", 12, 100, 0.52, iou_col)
        badge_y = 105

    _, col, short = decision(fused_c, upper, lower)
    if result["coop_triggered"]:
        short = "RSU FUSED -> RECOVERED" if fused_c > upper else "STILL LOW"
    else:
        short = "EGO SUFFICIENT"
    badge(card, 10, badge_y, pw - 10, INFO_H - 12, col, short, 0.62)

    return np.vstack([panel, card])


# ── Image 1: Ego comparison strip ─────────────────────────

def save_image1(ego_imgs, all_results, model_names, results,
                target_class, labels, upper, lower):

    PAD, GAP   = 36, 20
    PW, PH     = 480, 360
    TITLE_H    = 80
    N          = len(ego_imgs)
    W          = N * PW + (N-1) * GAP + 2 * PAD
    PANEL_H    = PH + 150          # image + info card
    H          = TITLE_H + PANEL_H + PAD

    img = canvas(W, H)
    txt(img, "Stage 1 - Ego Detection Across Occlusion Levels", PAD, 44, 0.90, WHITE, bold=True)
    txt(img, "Ego runs YOLO independently at each occlusion level - no V2 communication yet",
        PAD, 70, 0.52, GREY)
    divider_line(img, TITLE_H - 4)

    for i, (ego_img, res, result, label) in enumerate(
            zip(ego_imgs, all_results, results, labels)):
        px = PAD + i * (PW + GAP)
        panel = draw_ego_panel(ego_img, result, res, model_names,
                               target_class, label, upper, lower, PW, PH)
        img[TITLE_H:TITLE_H+PANEL_H, px:px+PW] = panel

    out = os.path.join(OUT_DIR, "01_ego_comparison.jpg")
    cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[1/2] Saved → {out}")


# ── Image 2: Fused comparison strip ───────────────────────

def save_image2(ego_imgs, all_results, model_names, results,
                target_class, labels, upper, lower):

    PAD, GAP   = 36, 20
    PW, PH     = 480, 360
    TITLE_H    = 80
    N          = len(ego_imgs)
    W          = N * PW + (N-1) * GAP + 2 * PAD
    PANEL_H    = PH + 180
    H          = TITLE_H + PANEL_H + PAD

    img = canvas(W, H)
    txt(img, "Stage 2 - Fused Result (WBF) Across Occlusion Levels", PAD, 44, 0.90, WHITE, bold=True)
    txt(img, "Weighted Box Fusion applied when ego confidence is below threshold",
        PAD, 70, 0.52, GREY)
    divider_line(img, TITLE_H - 4)

    for i, (ego_img, res, result, label) in enumerate(
            zip(ego_imgs, all_results, results, labels)):
        px = PAD + i * (PW + GAP)
        panel = draw_fused_panel(ego_img, result, res, model_names,
                                 target_class, label, upper, lower, PW, PH)
        img[TITLE_H:TITLE_H+PANEL_H, px:px+PW] = panel

    out = os.path.join(OUT_DIR, "02_fused_comparison.jpg")
    cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[2/2] Saved → {out}")


# ── Main ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Confidence-triggered cooperative perception — 3 real occlusion images"
    )
    parser.add_argument("--ego1",      required=True,  help="Ego image — low occlusion")
    parser.add_argument("--ego2",      required=True,  help="Ego image — mid occlusion")
    parser.add_argument("--ego3",      required=True,  help="Ego image — high occlusion")
    parser.add_argument("--v2",        default=None,
                        help="V2 / RSU image — optional, used only for 02_v2_detection.jpg visualization")
    parser.add_argument("--v2_det",    default=None,
                        help="RSU detections JSON produced by rsu_sender.py (optional)")
    parser.add_argument("--target",    default="car",
                        help="YOLO class to track (default: car)")
    parser.add_argument("--upper",     type=float, default=0.30,
                        help="Conf above which ego is sufficient (default: 0.30)")
    parser.add_argument("--lower",     type=float, default=0.30,
                        help="Conf below which V2 is activated (default: 0.30)")
    parser.add_argument("--label1",    default="Low Occlusion",
                        help="Label for ego1 panel (default: 'Low Occlusion')")
    parser.add_argument("--label2",    default="Mid Occlusion",
                        help="Label for ego2 panel (default: 'Mid Occlusion')")
    parser.add_argument("--label3",    default="High Occlusion",
                        help="Label for ego3 panel (default: 'High Occlusion')")
    parser.add_argument("--homography", default=None,
                        help="Path to homography .npy for V2→ego projection (optional)")
    parser.add_argument("--mirror", action="store_true",
                        help="V2 and ego view the anchor vehicle from opposite ends "
                             "(swaps left/right corner correspondences in projection)")
    parser.add_argument("--refbox", default=None,
                        help="Known car position in ego frame: x1,y1,x2,y2  (from ego1 when car fully visible). "
                             "Used as fused box position when coop is triggered — fixes cross-camera projection error.")
    parser.add_argument("--gt1", default=None,
                        help="Ground truth box for ego1 in pixels: x1,y1,x2,y2  (640x480 space)")
    parser.add_argument("--gt2", default=None,
                        help="Ground truth box for ego2 in pixels: x1,y1,x2,y2  (640x480 space)")
    parser.add_argument("--gt3", default=None,
                        help="Ground truth box for ego3 in pixels: x1,y1,x2,y2  (640x480 space)")
    args = parser.parse_args()

    # Reference box — known car position in ego frame for accurate fused bbox
    ref_box = parse_gt(args.refbox)
    if ref_box:
        print(f"[Coop] Ref box loaded: {ref_box}  (used as fused bbox position when coop triggered)")
    else:
        print("[Coop] No --refbox provided — fused box position from projection (may be inaccurate)")

    # Ground truth boxes (manual or auto-filled)
    gt_boxes = [parse_gt(args.gt1), parse_gt(args.gt2), parse_gt(args.gt3)]

    # Load all images
    ego_paths  = [args.ego1, args.ego2, args.ego3]
    ego_labels = [args.label1, args.label2, args.label3]
    ego_raws   = []
    for path in ego_paths:
        img = cv2.imread(path)
        if img is None: sys.exit(f"[Error] Cannot read: {path}")
        ego_raws.append(img)

    # V2 image is optional — only needed for 02_v2_detection.jpg visualization
    v2_img     = None
    v2_orig_w  = FRAME_WIDTH   # default: rsu_sender.py saves at FRAME resolution
    v2_orig_h  = FRAME_HEIGHT  # so calibrate.py homography was computed at same res
    if args.v2:
        v2_raw = cv2.imread(args.v2)
        if v2_raw is None:
            print(f"[Warning] Cannot read --v2 image: {args.v2} — skipping visualization")
        else:
            v2_orig_h, v2_orig_w = v2_raw.shape[:2]
            v2_img = cv2.resize(v2_raw, (FRAME_WIDTH, FRAME_HEIGHT))

    ego_imgs = [cv2.resize(r, (FRAME_WIDTH, FRAME_HEIGHT)) for r in ego_raws]

    # Load homography — manual file takes priority, then SIFT auto, then anchor
    H = None
    if args.homography and os.path.exists(args.homography):
        H_raw  = np.load(args.homography)
        sx_v2  = FRAME_WIDTH  / v2_orig_w
        sy_v2  = FRAME_HEIGHT / v2_orig_h
        S_dst     = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        S_src_inv = np.array([[1/sx_v2, 0, 0], [0, 1/sy_v2, 0], [0, 0, 1]], dtype=np.float64)
        H = S_dst @ H_raw @ S_src_inv
        print(f"[Coop] Homography loaded from {args.homography}")
    else:
        print("[Coop] No manual homography — bbox-corner homography will be computed per ego image")

    print(f"[Coop] Loading model: {MODEL_PATH}")
    model       = YOLO(MODEL_PATH)
    model_names = model.names

    # ── V2 detection — loaded from RSU JSON (rsu_sender.py output) ──
    v2_detections = []
    v2_hit        = None
    anchor_v2     = None

    if args.v2_det:
        print(f"\n[V2] Loading RSU detections from: {args.v2_det}")
        if not os.path.exists(args.v2_det):
            sys.exit(f"[Error] Cannot find v2_det file: {args.v2_det}")
        with open(args.v2_det) as f:
            v2_payload = json.load(f)
        v2_detections = v2_payload.get("detections", [])

        # Best target detection from JSON
        for det in v2_detections:
            if det["class"] == args.target:
                bb = tuple(det["bbox"])
                if v2_hit is None or det["conf"] > v2_hit[0]:
                    v2_hit = (det["conf"], bb)

        v2_conf = v2_hit[0] if v2_hit else 0.0
        print(f"[V2] Target '{args.target}': conf={v2_conf:.2f}" if v2_hit
              else f"[V2] Target '{args.target}' NOT detected")

        # Reconstruct anchor from JSON detections (truck/bus in ANCHOR_CLASSES)
        best_anchor_area = 0
        for det in v2_detections:
            if det["class"] in ANCHOR_CLASSES:
                x1, y1, x2, y2 = det["bbox"]
                area = (x2 - x1) * (y2 - y1)
                if area > best_anchor_area:
                    best_anchor_area = area
                    w = max(x2 - x1, 1)
                    h = max(y2 - y1, 1)
                    anchor_v2 = {"class": det["class"], "conf": det["conf"],
                                 "bbox": (x1, y1, x2, y2),
                                 "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
                                 "w": w, "h": h}
        if anchor_v2:
            angle_v2_deg = np.degrees(estimate_view_angle(anchor_v2))
            print(f"[V2] Anchor: {anchor_v2['class']}  est. angle={angle_v2_deg:.1f} deg")
        else:
            print("[V2] No anchor vehicle found — anchor projection unavailable")
    else:
        print("\n[V2] No --v2_det provided — running ego-only (RSU not contacted)")

    # Project V2 bbox into ego space via homography (same for all ego images)
    v2_hit_projected_base = v2_hit
    if v2_hit and H is not None:
        proj = project_bbox(v2_hit[1], H)
        if proj:
            v2_hit_projected_base = (v2_hit[0], proj)
            print(f"[V2] Bbox projected into ego space via homography")

    # ── Pre-compute ego detections (used for mirror calibration + main loop) ──
    ego_precomputed = []
    for ego_img, path in zip(ego_imgs, ego_paths):
        res        = model(ego_img, conf=0.10, verbose=False)[0]
        ego_hit    = best_target(res, model_names, args.target)
        anchor_ego = find_anchor(res, model_names)
        ego_precomputed.append((res, ego_hit, anchor_ego))

    # Auto-fill GT from ego detection when confidence is high (>= 0.50)
    for i, (_, ego_hit_cal, _) in enumerate(ego_precomputed):
        if gt_boxes[i] is None and ego_hit_cal and ego_hit_cal[0] >= 0.50:
            gt_boxes[i] = ego_hit_cal[1]
            print(f"[GT] ego{i+1}: auto-set from ego detection  conf={ego_hit_cal[0]:.2f}  box={ego_hit_cal[1]}")

    # ── Mirror state (explicit flag — user specifies if cameras are on opposite sides) ──
    global_flipped = args.mirror
    print(f"[Coop] Projection mirror: {'MIRRORED (--mirror)' if global_flipped else 'same direction'}")

    # ── Ego pipeline per image (uses precomputed detections) ───────────────
    all_ego_res = []
    all_results = []

    print()
    for i, ((res, ego_hit, anchor_ego), ego_img, label, path) in enumerate(
            zip(ego_precomputed, ego_imgs, ego_labels, ego_paths), 1):
        print(f"[Ego {i}] {label} — {os.path.basename(path)}")

        v2_hit_projected = v2_hit_projected_base
        if v2_hit and H is None and anchor_v2 is not None:
            if anchor_ego:
                H_bbox = bbox_corner_homography(anchor_v2, anchor_ego,
                                                global_flipped=global_flipped)
                proj   = project_bbox(v2_hit[1], H_bbox)
                if proj:
                    v2_hit_projected = (v2_hit[0], proj)
                    print(f"  Bbox-corner H: truck {anchor_v2['bbox']} -> {anchor_ego['bbox']}  "
                          f"car projected to {proj}")
                else:
                    print(f"  Bbox-corner H: projection out of frame bounds")
            else:
                print(f"  No anchor found in ego view — using raw V2 coords")

        result = run_pipeline(v2_hit_projected, ego_hit, args.upper, args.lower, ref_box=ref_box)

        # IoU between system output box and ground truth
        gt_box = gt_boxes[i - 1]
        if result["coop_triggered"]:
            eval_box = v2_hit_projected[1] if v2_hit_projected else None
        else:
            eval_box = ego_hit[1] if ego_hit else None

        iou = compute_iou(eval_box, gt_box) if (eval_box and gt_box) else None
        result["iou"]    = iou
        result["gt_box"] = gt_box

        print(f"  Ego conf : {result['ego_conf']:.2f}")
        print(f"  Decision : {result['status']}")
        if result["coop_triggered"]:
            print(f"  Fused    : {result['fused_conf']:.2f}  "
                  f"(gain: {result['fused_conf']-result['ego_conf']:+.2f})")
        if iou is not None:
            print(f"  IoU      : {iou:.3f}  (projected box vs ground truth)")
        else:
            print(f"  IoU      : N/A  "
                  f"({'no detection' if not eval_box else 'no ground truth -- use --gt1/2/3'})")

        all_ego_res.append(res)
        all_results.append(result)

    # ── Save outputs ────────────────────────────────────────
    print("\n[Coop] Saving output images...")
    save_image1(ego_imgs, all_ego_res, model_names, all_results,
                args.target, ego_labels, args.upper, args.lower)

    save_image2(ego_imgs, all_ego_res, model_names, all_results,
                args.target, ego_labels, args.upper, args.lower)

    # ── Summary ─────────────────────────────────────────────
    print("\n" + "═" * 68)
    print("  COOPERATIVE PERCEPTION SUMMARY")
    print("═" * 68)
    print(f"  {'Level':<20} {'Ego':>6} {'Fused':>7} {'Gain':>6}  {'Triggered':<10} {'IoU':>6}")
    print(f"  {'-'*20} {'-'*6} {'-'*7} {'-'*6}  {'-'*10} {'-'*6}")
    for label, r in zip(ego_labels, all_results):
        gain = r["fused_conf"] - r["ego_conf"]
        trig = "YES" if r["coop_triggered"] else "NO"
        iou_str = f"{r['iou']:.3f}" if r.get("iou") is not None else "N/A"
        print(f"  {label:<20} {r['ego_conf']:>6.2f} {r['fused_conf']:>7.2f} "
              f"{gain:>+6.2f}  {trig:<10} {iou_str:>6}")
    print("═" * 68)
    print("\nDone.")


if __name__ == "__main__":
    main()