# ============================================================
# ego_v2v.py — Run this on Pi 1 (Ego Vehicle) — V2V mode
#
# V2V (Vehicle-to-Vehicle) version of ego.py.
# Pi 2 is mounted in a SECOND CAR driving nearby.
#
# 3-panel split screen on HDMI monitor:
#   Panel A (left)  : Ego camera  + ego YOLO detections   (red boxes)
#   Panel B (centre): Vehicle 2 camera + V2 YOLO detections (green boxes)
#   Panel C (right) : Fused view  — ego camera + projected V2 detections (dashed green)
#
# NOTE on homography for V2V:
#   Both cameras are at similar heights (both in cars).
#   Calibrate as normal using calibrate.py — place 4 markers on the
#   ground between the two cars, click them in both camera views.
#   The ground-plane homography works the same as V2I.
#
# Run:  python ego_v2v.py
# ============================================================

import socket
import json
import threading
import time
import os
import cv2
import numpy as np
from ultralytics import YOLO
from config import (
    UDP_PORT, RSU_FRAME_PORT, RSU_TIMEOUT_SEC,
    MODEL_PATH, CAMERA_INDEX,
    FRAME_WIDTH, FRAME_HEIGHT,
    PANEL_WIDTH, DISPLAY_WIDTH, DISPLAY_HEIGHT, DIVIDER_WIDTH,
    CONFIDENCE_THRESHOLD, CRITICAL_CLASSES,
    HOMOGRAPHY_FILE
)

# ──────────────────────────────────────────────────────────────────────
# Shared state (written by background threads, read by main loop)
# ──────────────────────────────────────────────────────────────────────

_v2_detections: list = []
_v2_last_seen:  float = 0.0
_v2_lock = threading.Lock()

_v2_frame      = None
_v2_frame_lock = threading.Lock()

# ──────────────────────────────────────────────────────────────────────
# Background threads
# ──────────────────────────────────────────────────────────────────────

def _listen_v2_detections():
    """Receive detection JSON from Vehicle 2 over UDP port 5005."""
    global _v2_detections, _v2_last_seen

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    sock.settimeout(1.0)
    print(f"[Ego] Listening for Vehicle 2 detections on :{UDP_PORT}")

    while True:
        try:
            data, _ = sock.recvfrom(65535)
            payload = json.loads(data.decode())
            with _v2_lock:
                _v2_detections = payload.get("detections", [])
                _v2_last_seen  = time.time()
        except socket.timeout:
            if time.time() - _v2_last_seen > RSU_TIMEOUT_SEC:
                with _v2_lock:
                    _v2_detections = []
        except Exception as e:
            print(f"[Ego] Detection receive error: {e}")


def _listen_v2_frame():
    """Receive JPEG-compressed Vehicle 2 camera frame over UDP port 5006."""
    global _v2_frame

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", RSU_FRAME_PORT))
    sock.settimeout(1.0)
    print(f"[Ego] Listening for Vehicle 2 camera frame on :{RSU_FRAME_PORT}")

    while True:
        try:
            data, _ = sock.recvfrom(65535)
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                with _v2_frame_lock:
                    _v2_frame = frame
        except socket.timeout:
            pass
        except Exception as e:
            print(f"[Ego] Frame receive error: {e}")


def _v2_active():
    """Returns True if Vehicle 2 sent data within the timeout window."""
    return (time.time() - _v2_last_seen) < RSU_TIMEOUT_SEC


# ──────────────────────────────────────────────────────────────────────
# Homography — maps Vehicle 2 image coords → Ego image coords
# ──────────────────────────────────────────────────────────────────────

def load_homography():
    if os.path.exists(HOMOGRAPHY_FILE):
        H = np.load(HOMOGRAPHY_FILE)
        print(f"[Ego] Homography loaded from {HOMOGRAPHY_FILE}")
        return H
    print(f"[Ego] WARNING: {HOMOGRAPHY_FILE} not found.")
    print("[Ego]          Run calibrate.py first — place 4 ground markers visible")
    print("[Ego]          from BOTH car cameras, click them in the same order.")
    print("[Ego]          Falling back to alert-bar mode for now.")
    return None


def project_v2_bbox_to_ego(bbox_v2, H):
    """
    Transform a Vehicle 2 bounding box into ego camera pixel coordinates.
    Uses ground-plane homography — foot point (bottom-centre of bbox)
    is projected from V2 image space into ego image space.
    Works the same as V2I because both cars share the same ground plane.
    """
    x1, y1, x2, y2 = bbox_v2

    foot = np.array([[(x1 + x2) / 2.0],
                     [float(y2)],
                     [1.0]], dtype=np.float64)

    projected = H @ foot

    if abs(projected[2]) < 1e-6:
        return None

    projected /= projected[2]

    ex = int(projected[0])
    ey = int(projected[1])

    ex = max(10, min(FRAME_WIDTH  - 10, ex))
    ey = max(10, min(FRAME_HEIGHT - 10, ey))

    scale = max(0.15, ey / FRAME_HEIGHT)
    box_h = int(FRAME_HEIGHT * 0.38 * scale)
    box_w = int(box_h * 0.45)

    return (ex - box_w // 2, ey - box_h, ex + box_w // 2, ey)


# ──────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────

RED   = (0,   0,   220)
GREEN = (0,   210, 0)
WHITE = (255, 255, 255)
DARK  = (30,  30,  30)
AMBER = (0,   165, 255)


def _draw_solid_box(frame, x1, y1, x2, y2, label, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)


def _draw_dashed_box(frame, x1, y1, x2, y2, color, dash=12):
    sides = [(x1, y1, x2, y1), (x2, y1, x2, y2),
             (x2, y2, x1, y2), (x1, y2, x1, y1)]
    for (ax, ay, bx, by) in sides:
        dx     = bx - ax
        dy     = by - ay
        length = max(abs(dx), abs(dy), 1)
        steps  = max(1, length // (dash * 2))
        for i in range(steps + 1):
            t0 = min(i * 2 * dash / length, 1.0)
            t1 = min((i * 2 * dash + dash) / length, 1.0)
            sx, sy = int(ax + t0 * dx), int(ay + t0 * dy)
            ex, ey = int(ax + t1 * dx), int(ay + t1 * dy)
            cv2.line(frame, (sx, sy), (ex, ey), color, 2)


def _header_bar(panel, text, color):
    cv2.rectangle(panel, (0, 0), (FRAME_WIDTH, 40), color, -1)
    cv2.putText(panel, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2)


def _footer_bar(panel, text, color):
    cv2.rectangle(panel, (0, FRAME_HEIGHT - 30), (FRAME_WIDTH, FRAME_HEIGHT), DARK, -1)
    cv2.putText(panel, text, (8, FRAME_HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)


# ──────────────────────────────────────────────────────────────────────
# Panel A — Ego camera with ego-only detections
# ──────────────────────────────────────────────────────────────────────

def draw_ego_panel(frame, results, model_names):
    panel     = frame.copy()
    ego_count = len(results.boxes)

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls  = model_names[int(box.cls)]
        conf = float(box.conf)
        _draw_solid_box(panel, x1, y1, x2, y2, f"{cls} {conf:.2f}", RED)

    with _v2_lock:
        v2_classes = {d["class"] for d in _v2_detections}
    ego_classes = {model_names[int(b.cls)] for b in results.boxes}
    hidden      = v2_classes - ego_classes

    _header_bar(panel, "Panel A — Ego Camera (Single Vehicle)", RED)
    _footer_bar(panel, f"Ego only  |  Objects detected: {ego_count}", (180, 180, 180))

    if hidden and _v2_active():
        cv2.rectangle(panel, (0, FRAME_HEIGHT - 64), (FRAME_WIDTH, FRAME_HEIGHT - 30), (100, 0, 0), -1)
        cv2.putText(panel, f"  BLIND SPOT: {', '.join(hidden).upper()} NOT SEEN",
                    (8, FRAME_HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 120, 255), 2)

    return panel, ego_count


# ──────────────────────────────────────────────────────────────────────
# Panel B — Vehicle 2 camera (streamed from Pi 2, bboxes pre-drawn)
# ──────────────────────────────────────────────────────────────────────

def draw_v2_panel():
    with _v2_frame_lock:
        frame = _v2_frame.copy() if _v2_frame is not None else None

    with _v2_lock:
        v2_count  = len(_v2_detections)
        v2_online = _v2_active()

    if frame is None:
        panel = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 30, dtype=np.uint8)
        msg = "Waiting for Vehicle 2 feed..." if v2_online else "VEHICLE 2 OFFLINE"
        col = (180, 180, 180) if v2_online else AMBER
        cv2.putText(panel, msg, (30, FRAME_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, col, 2)
    else:
        panel = frame

    dot_col = GREEN if v2_online else AMBER
    dot_lbl = "V2 ONLINE" if v2_online else "V2 OFFLINE"
    _header_bar(panel, "Panel B — Vehicle 2 Camera (Cooperative Partner)", (0, 120, 0))
    cv2.circle(panel, (FRAME_WIDTH - 90, 22), 7, dot_col, -1)
    cv2.putText(panel, dot_lbl, (FRAME_WIDTH - 78, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, dot_col, 1)
    _footer_bar(panel, f"V2 detections: {v2_count}  (green = Vehicle 2 YOLO bboxes)", GREEN)

    return panel


# ──────────────────────────────────────────────────────────────────────
# Panel C — Fused ego view with projected Vehicle 2 detections
# ──────────────────────────────────────────────────────────────────────

def draw_fused_panel(frame, results, model_names, H):
    panel = frame.copy()

    ego_classes = {model_names[int(b.cls)] for b in results.boxes}

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls  = model_names[int(box.cls)]
        conf = float(box.conf)
        _draw_solid_box(panel, x1, y1, x2, y2, f"{cls} {conf:.2f}", RED)

    with _v2_lock:
        v2_dets   = list(_v2_detections)
        v2_online = _v2_active()

    hidden_objs = []
    alert_y     = 48

    for det in v2_dets:
        cls  = det["class"]
        conf = det["conf"]
        bbox = det["bbox"]

        if cls in ego_classes:
            continue

        hidden_objs.append(det)

        if H is not None:
            projected = project_v2_bbox_to_ego(bbox, H)
            if projected:
                px1, py1, px2, py2 = projected

                _draw_dashed_box(panel, px1, py1, px2, py2, GREEN)

                label = f"[V2] {cls} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(panel, (px1, py1 - th - 6), (px1 + tw + 4, py1), GREEN, -1)
                cv2.putText(panel, label, (px1 + 2, py1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

                foot_x = (px1 + px2) // 2
                cv2.arrowedLine(panel, (foot_x, py2), (foot_x, min(py2 + 22, FRAME_HEIGHT - 5)),
                                GREEN, 2, tipLength=0.4)
        else:
            bar = f"  [V2]  {cls.upper()}  detected  ({conf:.0%} conf)"
            cv2.rectangle(panel, (0, alert_y), (FRAME_WIDTH, alert_y + 30), (0, 140, 0), -1)
            cv2.putText(panel, bar, (6, alert_y + 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.57, WHITE, 2)
            alert_y += 34

    total_count = len(results.boxes) + len(hidden_objs)
    mode_note   = "dashed = V2 projected" if H is not None else "run calibrate.py for projection"

    _header_bar(panel, "Panel C — Fused View (Ego + Vehicle 2)", (0, 140, 0))
    _footer_bar(panel, f"Fused  |  Objects: {total_count}  |  {mode_note}", GREEN)

    dot_col = GREEN if v2_online else AMBER
    dot_lbl = "V2 ONLINE" if v2_online else "V2 OFFLINE"
    cv2.circle(panel, (FRAME_WIDTH - 90, 22), 7, dot_col, -1)
    cv2.putText(panel, dot_lbl, (FRAME_WIDTH - 78, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, dot_col, 1)

    critical = {d["class"] for d in hidden_objs} & CRITICAL_CLASSES
    if critical:
        flash = int(time.time() * 2) % 2 == 0
        bcol  = (0, 180, 0) if flash else (0, 90, 0)
        names = " + ".join(c.upper() for c in critical)
        cv2.rectangle(panel, (0, FRAME_HEIGHT - 64), (FRAME_WIDTH, FRAME_HEIGHT - 30), bcol, -1)
        cv2.putText(panel, f"  !! {names} AHEAD  (V2 WARNING) !!",
                    (8, FRAME_HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.58, WHITE, 2)

    return panel, total_count


# ──────────────────────────────────────────────────────────────────────
# Top metrics bar
# ──────────────────────────────────────────────────────────────────────

def draw_metrics_bar(single_count, coop_count):
    bar = np.zeros((32, DISPLAY_WIDTH, 3), dtype=np.uint8)
    bar[:] = (20, 20, 20)
    gain = coop_count - single_count
    sign = "+" if gain >= 0 else ""
    cv2.putText(bar,
                f"Single (A): {single_count} obj   |   Cooperative (C): {coop_count} obj   |   "
                f"V2 revealed: {sign}{gain} hidden object(s)",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
    return bar


# ──────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────

def main():
    threading.Thread(target=_listen_v2_detections, daemon=True).start()
    threading.Thread(target=_listen_v2_frame,      daemon=True).start()

    H = load_homography()

    print(f"[Ego] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Check CAMERA_INDEX in config.py.")

    win_name = "V2V Cooperative Perception Demo — FYP"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, DISPLAY_WIDTH, DISPLAY_HEIGHT + 32)

    print("[Ego] V2V Display running. Press Q to quit.")

    divider = np.full((FRAME_HEIGHT, DIVIDER_WIDTH, 3), 160, dtype=np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Ego] Camera read failed — retrying...")
            time.sleep(0.1)
            continue

        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

        panel_a, single_count = draw_ego_panel(frame,   results, model.names)
        panel_b               = draw_v2_panel()
        panel_c, coop_count   = draw_fused_panel(frame, results, model.names, H)

        split   = np.hstack([panel_a, divider, panel_b, divider, panel_c])
        metrics = draw_metrics_bar(single_count, coop_count)
        display = np.vstack([metrics, split])

        cv2.imshow(win_name, display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[Ego] Stopped.")


if __name__ == "__main__":
    main()
