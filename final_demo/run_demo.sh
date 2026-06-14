#!/bin/bash
# ============================================================
# run_demo.sh — Cooperative Perception Demo Orchestrator
#
# Two-stage pipeline that first checks whether the ego vehicle
# can detect the target on its own, then conditionally pulls
# RSU detections over the network if cooperation is needed.
#
# Flow:
#   1. Ego self-assessment (ego_check.py)
#        — Runs YOLO on all three ego images (25 / 50 / 75 % occlusion)
#        — Exits 0 if ego confidence is sufficient, 1 if RSU needed
#   2a. RSU needed  → SCP the detection JSON from the RSU Raspberry Pi
#        — Uses sshpass for passwordless SCP; installs it if missing
#        — Aborts with an error if the SCP transfer fails
#   2b. Ego sufficient → skip RSU contact entirely (no SCP performed)
#   3.  Run coop_perception.py
#        — With --v2_det if RSU was contacted (cooperative fusion)
#        — Without --v2_det if ego was sufficient (ego-only pipeline)
#
# Prerequisites:
#   python3, sshpass (auto-installed via apt if absent)
#   occlusion-images/ folder containing ego_25.jpg, ego_50.jpg,
#   ego_75.jpg, v2.jpg, and homography.npy
#
# Usage:
#   chmod +x run_demo.sh
#   ./run_demo.sh
#
# Configuration (edit variables below):
#   RSU_USER / RSU_HOST / RSU_PASS  — SSH credentials for the RSU Pi
#   RSU_REMOTE_PATH                 — Path to JSON on the RSU Pi
#   RSU_LOCAL_JSON                  — Local destination for the JSON
# ============================================================

RSU_USER="ops123"
RSU_HOST="172.20.10.4"
RSU_PASS="ops123"
RSU_REMOTE_PATH="~/Documents/YOLO_FYP/occlusion/rsu_detections_occlusion.json"
RSU_LOCAL_JSON="occlusion-images/rsu_detections_occlusion.json"

if ! command -v sshpass &> /dev/null; then
    echo "[Setup] Installing sshpass..."
    sudo apt-get install -y sshpass
fi

echo ""
echo "=================================================="
echo "  COOPERATIVE PERCEPTION DEMO"
echo "=================================================="

echo ""
echo "[Step 1] Ego self-assessment (no RSU contact yet)..."
python3 ego_check.py --ego1 occlusion-images/ego_25.jpg --ego2 occlusion-images/ego_50.jpg --ego3 occlusion-images/ego_75.jpg --target car --label1 "25% Occluded" --label2 "50% Occluded" --label3 "75% Occluded" --lower 0.30

NEED_RSU=$?

if [ $NEED_RSU -eq 1 ]; then
    echo ""
    echo "[Step 2] RSU activation triggered - fetching detections via SCP..."
    sshpass -p "$RSU_PASS" scp $RSU_USER@$RSU_HOST:$RSU_REMOTE_PATH occlusion-images/
    if [ $? -ne 0 ]; then
        echo "[Error] SCP failed - check RSU hostname ($RSU_HOST) and password."
        exit 1
    fi
    echo "[Step 2] RSU detections received: $RSU_LOCAL_JSON"
    echo ""
    echo "[Step 3] Running cooperative fusion pipeline..."
    python3 coop_perception.py --ego1 occlusion-images/ego_25.jpg --ego2 occlusion-images/ego_50.jpg --ego3 occlusion-images/ego_75.jpg --v2 occlusion-images/v2.jpg --v2_det $RSU_LOCAL_JSON --target car --label1 "25% Occluded" --label2 "50% Occluded" --label3 "75% Occluded" --mirror --homography homography.npy --refbox 119,256,248,303
else
    echo ""
    echo "[Step 2] Ego sufficient - RSU not contacted. No SCP performed."
    echo ""
    echo "[Step 3] Running ego-only pipeline..."
    python3 coop_perception.py --ego1 occlusion-images/ego_25.jpg --ego2 occlusion-images/ego_50.jpg --ego3 occlusion-images/ego_75.jpg --v2 occlusion-images/v2.jpg --target car --label1 "25% Occluded" --label2 "50% Occluded" --label3 "75% Occluded" --mirror --homography homography.npy --refbox 119,256,248,303
fi

echo ""
echo "=================================================="
echo "  DONE"
echo "=================================================="