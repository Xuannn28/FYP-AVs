# ============================================================
# config.py — Shared configuration for both Ego and RSU Pis
# ============================================================
# SETUP INSTRUCTIONS:
#   1. Find Pi 1 (Ego) IP:  run `hostname -I` on Pi 1
#   2. Find Pi 2 (RSU) IP:  run `hostname -I` on Pi 2
#   3. Update EGO_IP and RSU_IP below
#   4. Both Pis must be on the same WiFi network
# ============================================================

# --- Network ---
EGO_IP         = "172.20.10.2"   # Pi 1 (Ego) — RSU sends everything HERE
RSU_IP         = "172.20.10.4"   # Pi 2 (RSU) — for reference only
UDP_PORT       = 5005             # RSU → Ego: detection JSON
RSU_FRAME_PORT = 5006             # RSU → Ego: compressed camera frame (JPEG)

# --- Model ---
MODEL_PATH = "yolov8n.pt"

# --- Camera ---
CAMERA_INDEX  = 0
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480

# --- Detection ---
CONFIDENCE_THRESHOLD = 0.40

# --- Frame streaming (RSU → Ego) ---
JPEG_QUALITY = 40

# --- Homography calibration ---
HOMOGRAPHY_FILE = "homography.npy"

# --- Display (Ego main monitor) ---
DIVIDER_WIDTH  = 6
PANEL_WIDTH    = FRAME_WIDTH
DISPLAY_WIDTH  = PANEL_WIDTH * 3 + DIVIDER_WIDTH * 2
DISPLAY_HEIGHT = FRAME_HEIGHT

# RSU detection is considered stale after this many seconds
RSU_TIMEOUT_SEC = 2.0

# Classes that trigger a critical flashing warning when hidden from ego
CRITICAL_CLASSES = {"person", "bicycle", "motorcycle"}

# --- Lane classification (RSU frame x-coordinates) ---
SAME_LANE_X_MIN = 180
SAME_LANE_X_MAX = 460
