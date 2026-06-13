# ============================================================
# ego_check.py — Ego self-assessment: decide if RSU needed
#
# Runs YOLO on ego images and checks confidence against threshold.
# Exits with code 0 (ego sufficient) or 1 (RSU cooperation needed).
# Called by run_demo.sh before deciding whether to SCP from RSU.
#
# Usage:
#   python3 ego_check.py \
#       --ego1 ego_25.jpg --ego2 ego_50.jpg --ego3 ego_75.jpg \
#       --target car --lower 0.30 \
#       --label1 "25% Occluded" --label2 "50% Occluded" --label3 "75% Occluded"
# ============================================================

import argparse
import os
import sys
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
from ultralytics import YOLO
from config import MODEL_PATH, FRAME_WIDTH, FRAME_HEIGHT


def best_conf(results, model_names, target_class):
    hits = [float(b.conf) for b in results.boxes
            if model_names[int(b.cls)] == target_class]
    return max(hits) if hits else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Ego self-assessment — decide if RSU cooperation is needed")
    parser.add_argument("--ego1",   required=True, help="Ego image — low occlusion")
    parser.add_argument("--ego2",   required=True, help="Ego image — mid occlusion")
    parser.add_argument("--ego3",   required=True, help="Ego image — high occlusion")
    parser.add_argument("--target", default="car",  help="YOLO class to check (default: car)")
    parser.add_argument("--lower",  type=float, default=0.30,
                        help="Confidence threshold — below this triggers RSU (default: 0.30)")
    parser.add_argument("--label1", default="Low Occlusion")
    parser.add_argument("--label2", default="Mid Occlusion")
    parser.add_argument("--label3", default="High Occlusion")
    args = parser.parse_args()

    model       = YOLO(MODEL_PATH)
    model_names = model.names

    ego_paths = [args.ego1, args.ego2, args.ego3]
    labels    = [args.label1, args.label2, args.label3]

    print("\n[Ego Check] Running self-assessment...")
    print(f"  {'Frame':<20} {'Conf':>6}  Decision")
    print(f"  {'-'*20} {'-'*6}  {'-'*20}")

    need_rsu = False
    for path, label in zip(ego_paths, labels):
        img = cv2.imread(path)
        if img is None:
            sys.exit(f"[Error] Cannot read: {path}")
        img  = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))
        res  = model(img, conf=0.10, verbose=False)[0]
        conf = best_conf(res, model_names, args.target)

        if conf <= args.lower:
            decision = "RSU NEEDED"
            need_rsu = True
        else:
            decision = "SUFFICIENT"

        print(f"  {label:<20} {conf:>6.2f}  {decision}")

    print()
    if need_rsu:
        print("[Ego Check] RSU cooperation required — low confidence detected.")
        sys.exit(1)
    else:
        print("[Ego Check] Ego sufficient — RSU not contacted.")
        sys.exit(0)


if __name__ == "__main__":
    main()