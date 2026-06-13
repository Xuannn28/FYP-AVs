# ============================================================
# semantic_experiment.py
#
# Compares three RSU→Ego transmission strategies:
#   1. Raw image       — full JPEG sent, Ego runs YOLO
#   2. Compressed image— Q=30 JPEG sent, Ego runs YOLO
#   3. Semantic only   — RSU runs YOLO locally, sends JSON only
#
# Outputs:
#   semantic_comparison.jpg  — 3-panel visual poster
#   semantic_chart.jpg       — dual bar chart (size + confidence)
#
# Run:
#   python3 semantic_experiment.py --rsu ../images/part_1/rsu_sample.jpg --target person
# ============================================================

import argparse
import sys
import os
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ultralytics import YOLO
from config import MODEL_PATH, FRAME_WIDTH, FRAME_HEIGHT, CONFIDENCE_THRESHOLD

WHITE        = (255, 255, 255)
LIGHT_GRAY   = (210, 210, 210)
BORDER_GREEN = (0,   220,  80)
BORDER_RED   = (60,   60, 235)
LABEL_GREEN  = (80,  255, 130)
LABEL_RED    = (100, 100, 255)
AMBER        = (0,   165, 255)
BG_DARK      = (22,   22,  22)
BG_TITLE     = (15,   15,  38)

PANEL_W = 430
PANEL_H = 310
INFO_H  = 130
TITLE_H = 72
FOOT_H  = 80
DIVW    = 8


# ── Helpers ──────────────────────────────────────────────────

def compress(img, quality):
    _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR), round(len(enc) / 1024, 2)


def detect(model, img, conf):
    res = model(img, conf=conf, verbose=False)[0]
    return [{"class": model.names[int(b.cls)],
             "conf":  float(b.conf),
             "bbox":  tuple(map(int, b.xyxy[0]))}
            for b in res.boxes]


def draw_dets(img, dets, target):
    out = img.copy()
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        color = BORDER_GREEN if d["class"] == target else AMBER
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(d["class"], cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
        cv2.putText(out, f"{d['class']} {d['conf']:.2f}",
                    (x1+3, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
    return out


def make_json_panel(target_dets, target, w, h):
    panel = np.full((h, w, 3), 14, dtype=np.uint8)

    # Draw a subtle code-block background
    cv2.rectangle(panel, (14, 14), (w - 14, h - 14), (28, 28, 48), -1)
    cv2.rectangle(panel, (14, 14), (w - 14, h - 14), (60, 60, 100), 1)

    # Render JSON with colour-coded lines
    json_obj = [{"class": d["class"],
                 "conf":  round(d["conf"], 3),
                 "bbox":  list(d["bbox"])}
                for d in target_dets]
    lines = json.dumps(json_obj, indent=2).split("\n")

    y = 48
    for line in lines:
        if '"class"' in line:
            color = (120, 230, 255)   # cyan
        elif '"conf"' in line:
            color = (100, 255, 140)   # green
        elif '"bbox"' in line:
            color = (255, 200, 100)   # amber
        elif line.strip() in ("{", "}", "[", "]", "[{", "}]"):
            color = (160, 160, 160)
        else:
            color = (200, 200, 200)
        cv2.putText(panel, line, (26, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1)
        y += 30
        if y > h - 24:
            break

    return panel


# ── Chart ────────────────────────────────────────────────────

def save_chart(strategies, target, out_path):
    names  = [s["short_name"] for s in strategies]
    confs  = [s["conf"] for s in strategies]
    sizes  = [s["size"] for s in strategies]
    found  = [s["found"] for s in strategies]
    bar_colors_conf = ["#22cc66" if f else "#ee4444" for f in found]
    bar_colors_size = ["#5577ff", "#44aaee", "#22cc66"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#12122a")
    fig.suptitle(f"Semantic vs Image Transmission  |  Target: '{target}'",
                 color="white", fontsize=13, fontweight="bold", y=1.02)

    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white", labelsize=11)
        for spine in ax.spines.values():
            spine.set_edgecolor("#555577")
        ax.yaxis.grid(True, color="#333355", linewidth=0.8)
        ax.set_axisbelow(True)

    # Left chart — payload size (log scale so semantic bar is visible)
    bars1 = ax1.bar(names, sizes, color=bar_colors_size,
                    width=0.5, edgecolor="white", linewidth=0.6)
    for bar, size in zip(bars1, sizes):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 size * 1.6, f"{size} KB",
                 ha="center", color="white", fontsize=11, fontweight="bold")
    ax1.set_yscale("log")
    ax1.set_title("Transmission Payload Size", color="white",
                  fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("Size (KB)  —  log scale", color="white", fontsize=11)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.2f}" if x < 1 else f"{x:.0f}"))

    # Right chart — detection confidence
    bars2 = ax2.bar(names, confs, color=bar_colors_conf,
                    width=0.5, edgecolor="white", linewidth=0.6)
    for bar, conf, f in zip(bars2, confs, found):
        if f:
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     conf + 0.02, f"{conf:.2f}",
                     ha="center", color="white", fontsize=12, fontweight="bold")
        else:
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     0.04, "NOT\nDETECTED",
                     ha="center", color="#ffaaaa", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Detection Confidence", color="white",
                  fontsize=12, fontweight="bold", pad=10)
    ax2.set_ylabel("Confidence Score", color="white", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[Chart]    Saved to {out_path}")


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rsu",    required=True, help="RSU image path")
    parser.add_argument("--target", default="person")
    parser.add_argument("--conf",   type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--out",    default="semantic_comparison.jpg")
    args = parser.parse_args()

    raw = cv2.imread(args.rsu)
    if raw is None:
        sys.exit(f"Cannot read: {args.rsu}")
    img = cv2.resize(raw, (FRAME_WIDTH, FRAME_HEIGHT))

    print("[Semantic] Loading model...")
    model = YOLO(MODEL_PATH)

    # ── Strategy 1: Raw image ──────────────────────────────
    print("\n[Strategy 1] Raw image (Q=100) — Ego runs YOLO on received image")
    raw_img, raw_size = compress(img, 100)
    raw_dets  = detect(model, raw_img, args.conf)
    raw_found = any(d["class"] == args.target for d in raw_dets)
    raw_conf  = max((d["conf"] for d in raw_dets
                     if d["class"] == args.target), default=0.0)
    print(f"  Payload : {raw_size} KB")
    print(f"  {args.target}: {'FOUND' if raw_found else 'NOT FOUND'} "
          f"(conf {raw_conf:.2f})")

    # ── Strategy 2: Compressed image ──────────────────────
    print("\n[Strategy 2] Compressed image (Q=30) — Ego runs YOLO on received image")
    comp_img, comp_size = compress(img, 30)
    comp_dets  = detect(model, comp_img, args.conf)
    comp_found = any(d["class"] == args.target for d in comp_dets)
    comp_conf  = max((d["conf"] for d in comp_dets
                      if d["class"] == args.target), default=0.0)
    print(f"  Payload : {comp_size} KB")
    print(f"  {args.target}: {'FOUND' if comp_found else 'NOT FOUND'} "
          f"(conf {comp_conf:.2f})")

    # ── Strategy 3: Semantic features only ────────────────
    print("\n[Strategy 3] Semantic only — RSU runs YOLO, sends JSON to Ego")
    full_dets    = detect(model, img, args.conf)
    target_dets  = [d for d in full_dets if d["class"] == args.target]
    payload_obj  = [{"class": d["class"],
                     "conf":  round(d["conf"], 3),
                     "bbox":  list(d["bbox"])}
                    for d in target_dets]
    sem_payload  = json.dumps(payload_obj)
    sem_size     = round(len(sem_payload.encode("utf-8")) / 1024, 4)
    sem_found    = len(target_dets) > 0
    sem_conf     = max((d["conf"] for d in target_dets), default=0.0)
    print(f"  Payload : {sem_size} KB  ({len(sem_payload)} bytes)")
    print(f"  JSON    : {sem_payload}")
    print(f"  {args.target}: {'FOUND' if sem_found else 'NOT FOUND'} "
          f"(conf {sem_conf:.2f})")

    strategies = [
        {"short_name": "Raw Image\n(Q=100)",
         "label": "Strategy 1: Raw Image (Q=100)",
         "img": raw_img, "dets": raw_dets,
         "found": raw_found, "conf": raw_conf, "size": raw_size,
         "semantic": False},
        {"short_name": "Compressed\n(Q=30)",
         "label": "Strategy 2: Compressed Image (Q=30)",
         "img": comp_img, "dets": comp_dets,
         "found": comp_found, "conf": comp_conf, "size": comp_size,
         "semantic": False},
        {"short_name": "Semantic\nOnly",
         "label": "Strategy 3: Semantic Only",
         "img": None, "dets": full_dets,
         "found": sem_found, "conf": sem_conf, "size": sem_size,
         "semantic": True, "target_dets": target_dets},
    ]

    # ── Build poster ──────────────────────────────────────
    N       = len(strategies)
    TOTAL_W = PANEL_W * N + DIVW * (N - 1)

    canvas = np.full(
        (TITLE_H + PANEL_H + INFO_H + FOOT_H, TOTAL_W, 3),
        30, dtype=np.uint8)

    # Title
    cv2.rectangle(canvas, (0, 0), (TOTAL_W, TITLE_H), BG_TITLE, -1)
    cv2.putText(canvas,
                f"RSU Transmission Strategy Comparison  |  Target: '{args.target}'",
                (14, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)

    for i, s in enumerate(strategies):
        x = i * (PANEL_W + DIVW)
        y = TITLE_H

        # Panel content
        if s["semantic"]:
            panel = make_json_panel(s["target_dets"], args.target, PANEL_W, PANEL_H)
        else:
            drawn = draw_dets(s["img"], s["dets"], args.target)
            panel = cv2.resize(drawn, (PANEL_W, PANEL_H))

        border_col = BORDER_GREEN if s["found"] else BORDER_RED
        cv2.rectangle(panel, (0, 0), (PANEL_W-1, PANEL_H-1), border_col, 6)

        # Strategy label pill
        (lw, lh), _ = cv2.getTextSize(
            s["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        cv2.rectangle(panel, (6, 6), (lw + 18, lh + 16), (0, 0, 0), -1)
        cv2.putText(panel, s["label"], (10, lh + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, WHITE, 2)

        canvas[y:y+PANEL_H, x:x+PANEL_W] = panel

        if i < N - 1:
            canvas[y:y+PANEL_H+INFO_H, x+PANEL_W:x+PANEL_W+DIVW] = 55

        # Info bar
        iy = y + PANEL_H
        cv2.rectangle(canvas, (x, iy), (x+PANEL_W, iy+INFO_H), BG_DARK, -1)

        cv2.putText(canvas, f"Payload: {s['size']} KB",
                    (x+12, iy+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80, LIGHT_GRAY, 2)

        if s["found"]:
            cv2.putText(canvas, f"{args.target}: DETECTED",
                        (x+12, iy+82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.84, LABEL_GREEN, 2)
            cv2.putText(canvas, f"Confidence: {s['conf']:.2f}",
                        (x+12, iy+120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.80, LABEL_GREEN, 2)
        else:
            cv2.putText(canvas, f"{args.target}: NOT DETECTED",
                        (x+12, iy+82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.84, LABEL_RED, 2)
            cv2.putText(canvas, "Confidence: --",
                        (x+12, iy+120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.80, LABEL_RED, 2)

    # Footer
    fy = TITLE_H + PANEL_H + INFO_H
    cv2.rectangle(canvas, (0, fy), (TOTAL_W, fy+FOOT_H), BG_TITLE, -1)
    savings = round((1 - sem_size / raw_size) * 100, 1)
    cv2.putText(canvas,
                f"Semantic payload is {savings}% smaller than raw image  |  "
                f"Confidence: {sem_conf:.2f} (semantic) vs {raw_conf:.2f} (raw) vs {comp_conf:.2f} (compressed)",
                (14, fy+48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, LABEL_GREEN, 2)

    # Rebuild with taller footer for two lines
    FOOT_H2 = 90
    canvas2 = np.full(
        (TITLE_H + PANEL_H + INFO_H + FOOT_H2, TOTAL_W, 3),
        30, dtype=np.uint8)
    canvas2[:TITLE_H + PANEL_H + INFO_H] = canvas[:TITLE_H + PANEL_H + INFO_H]
    fy2 = TITLE_H + PANEL_H + INFO_H
    cv2.rectangle(canvas2, (0, fy2), (TOTAL_W, fy2 + FOOT_H2), BG_TITLE, -1)
    savings = round((1 - sem_size / raw_size) * 100, 1)
    cv2.putText(canvas2,
                f"Semantic payload ({sem_size} KB) is {savings}% smaller than raw image ({raw_size} KB)",
                (14, fy2 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, LABEL_GREEN, 2)
    cv2.putText(canvas2,
                f"Confidence  |  Semantic: {sem_conf:.2f}  |  Raw (Q=100): {raw_conf:.2f}  |  Compressed (Q=30): {comp_conf:.2f}",
                (14, fy2 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.68, LIGHT_GRAY, 1)
    canvas = canvas2

    cv2.imwrite(args.out, canvas)
    print(f"\n[Poster]   Saved to {args.out}")

    chart_path = args.out.replace(".jpg", "_chart.jpg")
    save_chart(strategies, args.target, chart_path)


if __name__ == "__main__":
    main()
