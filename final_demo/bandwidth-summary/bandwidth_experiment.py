# ============================================================
# bandwidth_experiment.py
#
# Experiments with JPEG compression quality vs detection accuracy.
# Simulates what happens when V2 sends frames at different
# bandwidth levels to the Ego vehicle.
#
# For each quality level it records:
#   - File size in bytes (proxy for bandwidth)
#   - Number of objects detected
#   - Average confidence score
#   - Whether target class is still detected
#
# Outputs:
#   bandwidth_results.jpg  — visual summary poster
#   bandwidth_results.csv  — raw numbers
#
# Run:
#   python3 bandwidth_experiment.py --v2 rsu_sample.jpg
#   python3 bandwidth_experiment.py --v2 rsu_sample.jpg --target person
# ============================================================

import argparse
import os
import sys
import csv
import cv2
import numpy as np
from ultralytics import YOLO
from config import MODEL_PATH, FRAME_WIDTH, FRAME_HEIGHT, CONFIDENCE_THRESHOLD

WHITE = (255, 255, 255)
RED   = (0,   0,   220)
GREEN = (0,   200, 0)
AMBER = (0,   165, 255)
DARK  = (20,  20,  20)

# Quality levels to test
QUALITY_LEVELS = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def compress_image(img, quality):
    """Compress image to JPEG at given quality, decode back to numpy array."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode(".jpg", img, encode_param)
    size_bytes  = len(encoded)
    decoded     = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded, size_bytes


def run_detection(model, img, conf_thresh):
    results = model(img, conf=conf_thresh, verbose=False)[0]
    detections = []
    for box in results.boxes:
        detections.append({
            "class": model.names[int(box.cls)],
            "conf":  float(box.conf),
            "bbox":  tuple(map(int, box.xyxy[0]))
        })
    return detections


def draw_detections(img, detections, target_class):
    out = img.copy()
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        color = GREEN if d["class"] == target_class else AMBER
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(d["class"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, f"{d['class']} {d['conf']:.2f}",
                    (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    return out


def bytes_to_kbps_label(size_bytes):
    kb = size_bytes / 1024
    if kb < 1:
        return f"{size_bytes}B"
    return f"{kb:.1f}KB"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2",     required=True, help="V2 image path")
    parser.add_argument("--target", default="person",
                        help="Key class to track across quality levels (default: person)")
    parser.add_argument("--conf",   type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--out",    default="bandwidth_results.jpg")
    args = parser.parse_args()

    img_raw = cv2.imread(args.v2)
    if img_raw is None:
        sys.exit(f"Cannot read: {args.v2}")

    img = cv2.resize(img_raw, (FRAME_WIDTH, FRAME_HEIGHT))

    print(f"[Experiment] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # ── Run experiment across all quality levels ──────────
    print(f"[Experiment] Testing {len(QUALITY_LEVELS)} quality levels...")
    print(f"[Experiment] Tracking target class: '{args.target}'")
    print()
    print(f"{'Quality':>8} {'Size':>10} {'Detections':>12} {'Avg Conf':>10} {args.target+' found?':>14}")
    print("-" * 60)

    records = []
    for q in QUALITY_LEVELS:
        compressed, size_bytes = compress_image(img, q)
        dets = run_detection(model, compressed, args.conf)

        avg_conf     = sum(d["conf"] for d in dets) / len(dets) if dets else 0.0
        target_found = any(d["class"] == args.target for d in dets)
        target_conf  = max((d["conf"] for d in dets if d["class"] == args.target), default=0.0)

        records.append({
            "quality":      q,
            "size_bytes":   size_bytes,
            "size_kb":      round(size_bytes / 1024, 2),
            "num_dets":     len(dets),
            "avg_conf":     round(avg_conf, 3),
            "target_found": target_found,
            "target_conf":  round(target_conf, 3),
            "detections":   dets,
            "image":        compressed,
        })

        status = "YES" if target_found else "NO  <-- LOST"
        print(f"  Q={q:>3}    {bytes_to_kbps_label(size_bytes):>8}    "
              f"{len(dets):>6} dets    {avg_conf:.2f} conf    {status}")

    print()

    # ── Save CSV ──────────────────────────────────────────
    csv_path = "bandwidth_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "quality", "size_bytes", "size_kb",
            "num_dets", "avg_conf", "target_found", "target_conf"
        ])
        writer.writeheader()
        for r in records:
            writer.writerow({k: r[k] for k in writer.fieldnames})
    print(f"[Experiment] CSV saved to {csv_path}")

    # ════════════════════════════════════════════════════
    # BUILD RESULTS POSTER
    # ════════════════════════════════════════════════════

    THUMB_W = 260
    THUMB_H = 195
    COLS    = 6
    ROWS    = (len(QUALITY_LEVELS) + COLS - 1) // COLS
    PAD     = 8
    INFO_H  = 64   # info bar under each thumbnail
    CELL_W  = THUMB_W + PAD * 2
    CELL_H  = THUMB_H + INFO_H + PAD * 2

    poster_w = CELL_W * COLS
    poster_h = CELL_H * ROWS

    # Find where target class is first lost
    lost_at = None
    for r in records:
        if not r["target_found"]:
            lost_at = r["quality"]
            break

    # ── Grid of thumbnails ────────────────────────────────
    grid = np.full((poster_h, poster_w, 3), 35, dtype=np.uint8)

    for i, r in enumerate(records):
        row = i // COLS
        col = i % COLS
        ox  = col * CELL_W + PAD
        oy  = row * CELL_H + PAD

        # Thumbnail with detections drawn
        thumb = cv2.resize(
            draw_detections(r["image"], r["detections"], args.target),
            (THUMB_W, THUMB_H)
        )

        # Border colour: green = target found, red = lost, amber = first lost
        if r["quality"] == lost_at:
            border_col = AMBER
        elif r["target_found"]:
            border_col = GREEN
        else:
            border_col = RED

        cv2.rectangle(thumb, (0, 0), (THUMB_W-1, THUMB_H-1), border_col, 3)
        grid[oy:oy+THUMB_H, ox:ox+THUMB_W] = thumb

        # Info bar
        iy = oy + THUMB_H
        cv2.rectangle(grid, (ox, iy), (ox+THUMB_W, iy+INFO_H), (45, 45, 45), -1)

        # Quality + size
        cv2.putText(grid, f"Q={r['quality']}  {bytes_to_kbps_label(r['size_bytes'])}",
                    (ox+4, iy+18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, WHITE, 1)

        # Detections
        cv2.putText(grid, f"{r['num_dets']} det  conf:{r['avg_conf']:.2f}",
                    (ox+4, iy+36), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)

        # Target status
        if r["target_found"]:
            cv2.putText(grid, f"{args.target}: {r['target_conf']:.2f}",
                        (ox+4, iy+54), cv2.FONT_HERSHEY_SIMPLEX, 0.50, GREEN, 1)
        else:
            cv2.putText(grid, f"{args.target}: NOT DETECTED",
                        (ox+4, iy+54), cv2.FONT_HERSHEY_SIMPLEX, 0.50, RED, 1)

    # ── Title bar ────────────────────────────────────────
    title = np.full((62, poster_w, 3), 15, dtype=np.uint8)
    cv2.rectangle(title, (0, 0), (poster_w, 62), (25, 25, 55), -1)
    cv2.putText(title,
                f"Bandwidth vs Detection Accuracy  |  Tracking: '{args.target}'",
                (14, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)

    # ── Summary bar ──────────────────────────────────────
    full_q_rec  = records[-1]  # quality=100 as baseline
    full_size   = full_q_rec["size_bytes"]

    summary = np.full((90, poster_w, 3), 18, dtype=np.uint8)
    cv2.rectangle(summary, (0, 0), (poster_w, 90), (18, 18, 40), -1)

    if lost_at:
        # Find last quality where target was still detected
        last_good = max(r["quality"] for r in records if r["target_found"])
        last_good_rec = next(r for r in records if r["quality"] == last_good)
        savings = round((1 - last_good_rec["size_bytes"] / full_size) * 100)
        cv2.putText(summary,
                    f"Target '{args.target}' detected down to Q={last_good}  "
                    f"({bytes_to_kbps_label(last_good_rec['size_bytes'])})  "
                    f"-- {savings}% bandwidth saving vs full quality",
                    (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 255, 120), 2)
        cv2.putText(summary,
                    f"Detection LOST at Q={lost_at}  "
                    f"({bytes_to_kbps_label(next(r['size_bytes'] for r in records if r['quality']==lost_at))})",
                    (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 160, 255), 2)
    else:
        cv2.putText(summary,
                    f"Target '{args.target}' detected at ALL quality levels tested.",
                    (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.75, GREEN, 2)

    # ── Legend ───────────────────────────────────────────
    legend = np.full((52, poster_w, 3), 30, dtype=np.uint8)
    items = [
        (GREEN, f"Green border = '{args.target}' detected"),
        (AMBER, "Amber border = first quality where detection is LOST"),
        (RED,   "Red border   = detection lost at this quality"),
    ]
    lx = 14
    for color, text in items:
        cv2.rectangle(legend, (lx, 14), (lx+24, 40), color, -1)
        cv2.putText(legend, text, (lx+30, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
        lx += poster_w // 3

    final = np.vstack([title, grid, summary, legend])
    cv2.imwrite(args.out, final)
    print(f"[Experiment] Poster saved to {args.out}")


if __name__ == "__main__":
    main()
