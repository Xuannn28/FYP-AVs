# ============================================================
# bandwidth_summary.py
# Generates a clean 4-panel result image for selected
# quality levels: Q=5, Q=15, Q=30, Q=100
#
# Run:
#   python3 bandwidth_summary.py --v2 rsu_sample.jpg --target person
# ============================================================

import argparse
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from ultralytics import YOLO
from config import MODEL_PATH, FRAME_WIDTH, FRAME_HEIGHT, CONFIDENCE_THRESHOLD

WHITE      = (255, 255, 255)
LIGHT_GRAY = (210, 210, 210)
BORDER_GREEN = (0,  220,  80)   # panel border when detected
BORDER_RED   = (60,  60, 235)   # panel border when not detected
LABEL_GREEN  = (80, 255, 130)   # text: detected
LABEL_RED    = (100, 100, 255)  # text: not detected
AMBER        = (0,  165, 255)   # non-target bbox
BG_DARK      = (22,  22,  22)   # info bar background
BG_TITLE     = (15,  15,  38)   # title / footer background

SELECTED_QUALITIES = [5, 15, 30, 100]
CHART_QUALITIES    = [100, 80, 60, 30, 15, 5]

PANEL_W  = 380
PANEL_H  = 285
INFO_H   = 105
TITLE_H  = 72
FOOT_H   = 82
CROP_PAD = 80   # padding around bbox for the zoomed panel (higher = less zoomed)


def save_gif(chart_results, ref_bbox, target, out_path, frame_ms=1200):
    GIF_W, GIF_H = 640, 520
    CROP_H       = 340   # image area
    INFO_H       = 180   # info area below

    frames = []
    for r in chart_results:
        canvas = np.full((GIF_H, GIF_W, 3), 18, dtype=np.uint8)

        # ── Cropped image region ──────────────────────
        if ref_bbox:
            zx1, zy1, zx2, zy2 = ref_bbox
            crop = r["image"][zy1:zy2, zx1:zx2]
            offset_dets = [{**d, "bbox": (
                d["bbox"][0] - zx1, d["bbox"][1] - zy1,
                d["bbox"][2] - zx1, d["bbox"][3] - zy1
            )} for d in r["dets"]]
            drawn = draw(crop, offset_dets, target)
        else:
            drawn = draw(r["image"], r["dets"], target)

        img_panel = cv2.resize(drawn, (GIF_W, CROP_H), interpolation=cv2.INTER_LINEAR)

        # Border colour
        border_col = BORDER_GREEN if r["target_found"] else BORDER_RED
        cv2.rectangle(img_panel, (0, 0), (GIF_W - 1, CROP_H - 1), border_col, 6)

        # Quality pill
        q_label = f"Q = {r['quality']}"
        (lw, lh), _ = cv2.getTextSize(q_label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        cv2.rectangle(img_panel, (8, 8), (lw + 24, lh + 22), (0, 0, 0), -1)
        cv2.putText(img_panel, q_label, (14, lh + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 2)

        canvas[0:CROP_H] = img_panel

        # ── Info area ────────────────────────────────
        iy = CROP_H
        cv2.rectangle(canvas, (0, iy), (GIF_W, GIF_H), BG_DARK, -1)

        # File size
        cv2.putText(canvas, f"File size: {r['size_kb']} KB",
                    (24, iy + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, LIGHT_GRAY, 2)

        # Detection result
        if r["target_found"]:
            cv2.putText(canvas, f"{target}: DETECTED",
                        (24, iy + 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, LABEL_GREEN, 2)
            cv2.putText(canvas, f"Confidence: {r['target_conf']:.2f}",
                        (24, iy + 148), cv2.FONT_HERSHEY_SIMPLEX, 0.9, LABEL_GREEN, 2)
        else:
            cv2.putText(canvas, f"{target}: NOT DETECTED",
                        (24, iy + 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, LABEL_RED, 2)
            cv2.putText(canvas, "Confidence: --",
                        (24, iy + 148), cv2.FONT_HERSHEY_SIMPLEX, 0.9, LABEL_RED, 2)

        # Convert BGR → RGB → PIL
        pil_frame = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        frames.append(pil_frame)

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_ms,
        loop=0
    )
    print(f"[GIF]     Saved to {out_path}")


def save_chart(results, target, out_path):
    labels   = [f"Q={r['quality']}\n{r['size_kb']}KB" for r in results]
    confs    = [r["target_conf"] for r in results]
    found    = [r["target_found"] for r in results]
    colors   = ["#22cc66" if f else "#ee4444" for f in found]

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("#12122a")
    ax.set_facecolor("#1a1a2e")

    bars = ax.bar(labels, confs, color=colors, width=0.5,
                  edgecolor="white", linewidth=0.6)

    # Confidence value labels on bars
    for bar, conf, f in zip(bars, confs, found):
        if f:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    conf + 0.02, f"{conf:.2f}",
                    ha="center", va="bottom",
                    color="white", fontsize=13, fontweight="bold")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    0.04, "NOT\nDETECTED",
                    ha="center", va="bottom",
                    color="#ffaaaa", fontsize=10, fontweight="bold")

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("JPEG Quality Level  (lower = smaller file)", color="white", fontsize=12)
    ax.set_ylabel("Detection Confidence", color="white", fontsize=12)
    ax.set_title(f"Compression Quality vs Detection Accuracy  |  Target: '{target}'",
                 color="white", fontsize=13, fontweight="bold", pad=14)

    ax.tick_params(colors="white", labelsize=10)
    plt.xticks(rotation=0, ha="center")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555577")
    ax.yaxis.grid(True, color="#333355", linewidth=0.8)
    ax.set_axisbelow(True)

    detected_patch   = mpatches.Patch(color="#22cc66", label="Detected")
    undetected_patch = mpatches.Patch(color="#ee4444", label="Not Detected")
    ax.legend(handles=[detected_patch, undetected_patch],
              facecolor="#1a1a2e", edgecolor="#555577",
              labelcolor="white", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Chart]   Saved to {out_path}")


def compress(img, quality):
    _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    size_kb = len(enc) / 1024
    return cv2.imdecode(enc, cv2.IMREAD_COLOR), round(size_kb, 2)


def detect(model, img, conf):
    res = model(img, conf=conf, verbose=False)[0]
    return [{"class": model.names[int(b.cls)],
             "conf":  float(b.conf),
             "bbox":  tuple(map(int, b.xyxy[0]))}
            for b in res.boxes]


def draw(img, dets, target):
    out = img.copy()
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        color = BORDER_GREEN if d["class"] == target else AMBER
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(
            d["class"], cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
        cv2.putText(out, f"{d['class']} {d['conf']:.2f}",
                    (x1+3, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2",     required=True)
    parser.add_argument("--target", default="person")
    parser.add_argument("--conf",   type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--out",    default="bandwidth_summary.jpg")
    args = parser.parse_args()

    raw = cv2.imread(args.v2)
    if raw is None:
        sys.exit(f"Cannot read: {args.v2}")
    img = cv2.resize(raw, (FRAME_WIDTH, FRAME_HEIGHT))

    print(f"[Summary] Loading model...")
    model = YOLO(MODEL_PATH)

    # ── Run detection for selected quality levels ──
    results = []
    for q in SELECTED_QUALITIES:
        compressed, size_kb = compress(img, q)
        dets = detect(model, compressed, args.conf)
        target_found = any(d["class"] == args.target for d in dets)
        target_conf  = max((d["conf"] for d in dets
                            if d["class"] == args.target), default=0.0)
        results.append({
            "quality":      q,
            "size_kb":      size_kb,
            "dets":         dets,
            "target_found": target_found,
            "target_conf":  target_conf,
            "image":        compressed,
        })
        status = f"FOUND ({target_conf:.2f})" if target_found else "NOT FOUND"
        print(f"  Q={q:>3}  {size_kb:>7.2f}KB  "
              f"{len(dets)} dets  {args.target}: {status}")

    # ── Get inset crop region from highest-quality detection ──
    ref_bbox = None
    for r in reversed(results):
        target_dets = [d for d in r["dets"] if d["class"] == args.target]
        if target_dets:
            best = max(target_dets, key=lambda d: d["conf"])
            x1, y1, x2, y2 = best["bbox"]
            x1 = max(0, x1 - CROP_PAD)
            y1 = max(0, y1 - CROP_PAD)
            x2 = min(FRAME_WIDTH,  x2 + CROP_PAD)
            y2 = min(FRAME_HEIGHT, y2 + CROP_PAD)
            ref_bbox = (x1, y1, x2, y2)
            break

    # ── Build poster ──────────────────────────────────
    N       = len(SELECTED_QUALITIES)
    DIVW    = 6
    TOTAL_W = PANEL_W * N + DIVW * (N - 1)

    canvas = np.full(
        (TITLE_H + PANEL_H + INFO_H + FOOT_H, TOTAL_W, 3),
        30, dtype=np.uint8)

    # Title bar
    cv2.rectangle(canvas, (0, 0), (TOTAL_W, TITLE_H), BG_TITLE, -1)
    cv2.putText(canvas,
                f"Compression Quality vs Detection Accuracy  |  Target: '{args.target}'",
                (14, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)

    for i, r in enumerate(results):
        x = i * (PANEL_W + DIVW)
        y = TITLE_H

        # ── Crop to zoomed region, stretch to fill panel ──
        if ref_bbox:
            zx1, zy1, zx2, zy2 = ref_bbox
            crop = r["image"][zy1:zy2, zx1:zx2]
            offset_dets = [{**d, "bbox": (
                d["bbox"][0] - zx1, d["bbox"][1] - zy1,
                d["bbox"][2] - zx1, d["bbox"][3] - zy1
            )} for d in r["dets"]]
            base = cv2.resize(draw(crop, offset_dets, args.target),
                              (PANEL_W, PANEL_H),
                              interpolation=cv2.INTER_LINEAR) if crop.size > 0 \
                   else np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
        else:
            base = draw(r["image"], r["dets"], args.target)

        thumb = cv2.resize(base, (PANEL_W, PANEL_H))

        # Outer border: bright green = found, bright red = lost
        border_col = BORDER_GREEN if r["target_found"] else BORDER_RED
        cv2.rectangle(thumb, (0, 0), (PANEL_W-1, PANEL_H-1), border_col, 6)

        # Quality label — dark pill background for contrast
        q_label = f"Q = {r['quality']}"
        (lw, lh), _ = cv2.getTextSize(q_label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(thumb, (6, 6), (lw + 18, lh + 16), (0, 0, 0), -1)
        cv2.putText(thumb, q_label, (12, lh + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)

        canvas[y:y+PANEL_H, x:x+PANEL_W] = thumb

        # Divider
        if i < N - 1:
            canvas[y:y+PANEL_H+INFO_H, x+PANEL_W:x+PANEL_W+DIVW] = 55

        # Info bar — always dark background for legibility
        iy = y + PANEL_H
        cv2.rectangle(canvas, (x, iy), (x+PANEL_W, iy+INFO_H), BG_DARK, -1)

        cv2.putText(canvas, f"Size: {r['size_kb']} KB",
                    (x+12, iy+34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, LIGHT_GRAY, 2)

        if r["target_found"]:
            status_text = f"{args.target}: {r['target_conf']:.2f} conf"
            cv2.putText(canvas, status_text,
                        (x+12, iy+76),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.80, LABEL_GREEN, 2)
        else:
            cv2.putText(canvas, f"{args.target}: NOT DETECTED",
                        (x+12, iy+76),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.80, LABEL_RED, 2)

    # Footer
    fy = TITLE_H + PANEL_H + INFO_H
    cv2.rectangle(canvas, (0, fy), (TOTAL_W, fy+FOOT_H), BG_TITLE, -1)

    stable_q  = next((r["quality"] for r in results if r["target_found"]), None)
    stable_kb = next((r["size_kb"]  for r in results if r["target_found"]), None)
    full_kb   = results[-1]["size_kb"]

    if stable_kb and full_kb:
        saving = round((1 - stable_kb / full_kb) * 100)
        cv2.putText(canvas,
                    f"Stable detection from Q={stable_q} ({stable_kb} KB)  |  "
                    f"{saving}% smaller transmission vs full quality ({full_kb} KB)",
                    (14, fy+48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, LABEL_GREEN, 2)

    cv2.imwrite(args.out, canvas)
    print(f"\n[Summary] Saved to {args.out}")

    # Run detections for extra chart quality levels
    print("\n[Chart]   Running extra quality levels for chart...")
    chart_results = []
    for q in CHART_QUALITIES:
        compressed, size_kb = compress(img, q)
        dets = detect(model, compressed, args.conf)
        target_found = any(d["class"] == args.target for d in dets)
        target_conf  = max((d["conf"] for d in dets
                            if d["class"] == args.target), default=0.0)
        chart_results.append({
            "quality":      q,
            "size_kb":      size_kb,
            "dets":         dets,
            "target_found": target_found,
            "target_conf":  target_conf,
            "image":        compressed,
        })
        status = f"FOUND ({target_conf:.2f})" if target_found else "NOT FOUND"
        print(f"  Q={q:>3}  {size_kb:>7.2f}KB  {args.target}: {status}")

    chart_path = args.out.replace(".jpg", "_chart.jpg")
    save_chart(chart_results, args.target, chart_path)

    # Animated GIF
    gif_path = args.out.replace(".jpg", "_animation.gif")
    save_gif(chart_results, ref_bbox, args.target, gif_path)


if __name__ == "__main__":
    main()
