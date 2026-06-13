# ============================================================
# ego.py — Run this on Pi 1 (Ego Vehicle)
#
# 3-panel split screen on HDMI monitor:
#   Panel A (left)  : Ego camera  + ego YOLO detections   (red boxes)
#   Panel B (centre): RSU camera  + RSU YOLO detections   (green boxes) ← streamed from Pi 2
#   Panel C (right) : Fused view  — ego camera + projected RSU detections (dashed green)
#                     RSU detections are transformed into ego camera space
#                     using a pre-computed homography matrix (see calibrate.py)
#
# Run:  python ego.py
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

# the latest list of objects RSU can see
_rsu_detections: list = []

# timestamp of when RSU last sent data
_rsu_last_seen: float = 0.0

# prevents both threads reading/writing RSU data at the same time
# prevents corrupting shared data
_rsu_lock = threading.Lock()

# latest annotated frame received from RSU camera (JPEG decoded)
_rsu_frame      = None
_rsu_frame_lock = threading.Lock()

# ──────────────────────────────────────────────────────────────────────
# Background threads
# ──────────────────────────────────────────────────────────────────────

def _listen_rsu_detections():
    """Receive detection JSON from RSU over UDP port 5005."""
    global _rsu_detections, _rsu_last_seen

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # listen on all network interfaces, wait for incoming RSU packets
    sock.bind(("0.0.0.0", UDP_PORT))
    # if packet not received within 1s, raise timeout exception
    sock.settimeout(1.0)
    print(f"[Ego] Listening for RSU detections on :{UDP_PORT}")

    while True:
        try:
            # wait for UDP packet (data = raw bytes, max 65535)
            data, _ = sock.recvfrom(65535)
            # convert bytes to string, parse JSON to dict
            payload = json.loads(data.decode())
            # replace current detections with new ones, update timestamp
            with _rsu_lock:
                _rsu_detections = payload.get("detections", [])
                _rsu_last_seen  = time.time()
        except socket.timeout:
            # clear detections if RSU has gone silent — prevent stale data
            if time.time() - _rsu_last_seen > RSU_TIMEOUT_SEC:
                with _rsu_lock:
                    _rsu_detections = []
        except Exception as e:
            print(f"[Ego] Detection receive error: {e}")


def _listen_rsu_frame():
    """
    Receive JPEG-compressed RSU camera frame over UDP port 5006.
    RSU sends its annotated frame (with bboxes already drawn by results.plot()).
    We decode it and store it for Panel B display.
    """
    global _rsu_frame

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", RSU_FRAME_PORT))
    sock.settimeout(1.0)
    print(f"[Ego] Listening for RSU camera frame on :{RSU_FRAME_PORT}")

    while True:
        try:
            data, _ = sock.recvfrom(65535)
            # decode JPEG bytes back into a numpy image array
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                with _rsu_frame_lock:
                    _rsu_frame = frame
        except socket.timeout:
            pass
        except Exception as e:
            print(f"[Ego] Frame receive error: {e}")


def _rsu_active():
    """Returns True if RSU sent data within the timeout window."""
    return (time.time() - _rsu_last_seen) < RSU_TIMEOUT_SEC


# ──────────────────────────────────────────────────────────────────────
# Homography — maps RSU image coords → Ego image coords
# ──────────────────────────────────────────────────────────────────────

def load_homography():
    """
    Load the 3x3 homography matrix produced by calibrate.py.
    This matrix transforms a point in RSU image space to the
    corresponding point in ego image space (via the ground plane).
    Returns None if the file doesn't exist yet.
    """
    if os.path.exists(HOMOGRAPHY_FILE):
        H = np.load(HOMOGRAPHY_FILE)
        print(f"[Ego] Homography loaded from {HOMOGRAPHY_FILE}")
        return H
    print(f"[Ego] WARNING: {HOMOGRAPHY_FILE} not found.")
    print("[Ego]          Run calibrate.py first for geometric projection.")
    print("[Ego]          Falling back to alert-bar mode for now.")
    return None


def project_rsu_bbox_to_ego(bbox_rsu, H):
    """
    Transform an RSU bounding box into ego camera pixel coordinates.

    Key idea — ground plane homography:
      A homography H maps any point on a flat surface (the table/ground)
      from one camera's image to another camera's image.
      We use the *bottom-centre* of the bbox as the 'foot point' —
      the point where the object contacts the ground — because that
      point lies on the ground plane and is well-modelled by H.

    Steps:
      1. Take foot point = bottom-centre of RSU bbox
      2. Apply H  →  projected foot point in ego image
      3. Estimate a display box around the projected foot point,
         scaled by its vertical position (lower in ego frame = closer = bigger)

    Returns (x1, y1, x2, y2) in ego pixel coordinates, or None on failure.
    """
    x1, y1, x2, y2 = bbox_rsu

    # foot point in homogeneous coordinates [x, y, 1]
    foot = np.array([[(x1 + x2) / 2.0],
                     [float(y2)],
                     [1.0]], dtype=np.float64)

    # apply homography: projected = H @ foot
    projected = H @ foot

    # guard against degenerate (near-zero) w component
    if abs(projected[2]) < 1e-6:
        return None

    # convert from homogeneous to Cartesian: divide by w
    projected /= projected[2]

    ex = int(projected[0])   # ego x coordinate
    ey = int(projected[1])   # ego y coordinate (foot, ground level)

    # clamp to frame bounds so box is always visible
    ex = max(10, min(FRAME_WIDTH  - 10, ex))
    ey = max(10, min(FRAME_HEIGHT - 10, ey))

    # estimate display box size:
    # objects lower in the ego frame (larger ey) are closer → appear bigger
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
    """Solid bounding box with filled label background — used for direct detections."""
    # draw rectangle border
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    # measure label text size to size the background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    # filled label background
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    # label text e.g. "car 0.87"
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)


def _draw_dashed_box(frame, x1, y1, x2, y2, color, dash=12):
    """
    Dashed bounding box — visually signals that this object is NOT directly
    visible in this camera but is known via the RSU through cooperative perception.
    The dashed style distinguishes it clearly from solid ego detections.
    """
    # draw all 4 sides as dashes
    sides = [(x1, y1, x2, y1),   # top edge
             (x2, y1, x2, y2),   # right edge
             (x2, y2, x1, y2),   # bottom edge
             (x1, y2, x1, y1)]   # left edge
    for (ax, ay, bx, by) in sides:
        dx     = bx - ax
        dy     = by - ay
        length = max(abs(dx), abs(dy), 1)
        steps  = max(1, length // (dash * 2))
        for i in range(steps + 1):
            # each dash spans from t0 to t1 along the edge
            t0 = min(i * 2 * dash / length, 1.0)
            t1 = min((i * 2 * dash + dash) / length, 1.0)
            sx, sy = int(ax + t0 * dx), int(ay + t0 * dy)
            ex, ey = int(ax + t1 * dx), int(ay + t1 * dy)
            cv2.line(frame, (sx, sy), (ex, ey), color, 2)


def _header_bar(panel, text, color):
    """40px coloured title bar at top of panel."""
    cv2.rectangle(panel, (0, 0), (FRAME_WIDTH, 40), color, -1)
    cv2.putText(panel, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2)


def _footer_bar(panel, text, color):
    """30px dark footer bar at bottom of panel."""
    cv2.rectangle(panel, (0, FRAME_HEIGHT - 30), (FRAME_WIDTH, FRAME_HEIGHT), DARK, -1)
    cv2.putText(panel, text, (8, FRAME_HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)


# ──────────────────────────────────────────────────────────────────────
# Panel A — Ego camera with ego-only detections
# ──────────────────────────────────────────────────────────────────────

def draw_ego_panel(frame, results, model_names):
    """
    LEFT panel — what the ego vehicle's camera sees alone (no cooperation).
    Simulates a single autonomous vehicle with no external sensor sharing.
    Red boxes = objects ego YOLO detected.
    """
    panel     = frame.copy()
    ego_count = len(results.boxes)

    # draw red bounding boxes for each ego detection
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls  = model_names[int(box.cls)]
        conf = float(box.conf)
        _draw_solid_box(panel, x1, y1, x2, y2, f"{cls} {conf:.2f}", RED)

    # find classes ego cannot see but RSU can (the blind spot)
    with _rsu_lock:
        rsu_classes = {d["class"] for d in _rsu_detections}
    ego_classes = {model_names[int(b.cls)] for b in results.boxes}
    hidden      = rsu_classes - ego_classes   # set difference = blind spot

    _header_bar(panel, "Panel A — Ego Camera (Single Perception)", RED)
    _footer_bar(panel, f"Ego only  |  Objects detected: {ego_count}", (180, 180, 180))

    # show blind-spot warning bar if RSU is active and ego is missing objects
    if hidden and _rsu_active():
        cv2.rectangle(panel, (0, FRAME_HEIGHT - 64), (FRAME_WIDTH, FRAME_HEIGHT - 30), (100, 0, 0), -1)
        cv2.putText(panel, f"  BLIND SPOT: {', '.join(hidden).upper()} NOT SEEN",
                    (8, FRAME_HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 120, 255), 2)

    return panel, ego_count


# ──────────────────────────────────────────────────────────────────────
# Panel B — RSU camera (streamed from Pi 2, bboxes pre-drawn)
# ──────────────────────────────────────────────────────────────────────

def draw_rsu_panel():
    """
    CENTRE panel — what the RSU camera sees, received live from Pi 2.
    The frame already has YOLO bboxes drawn by results.plot() in rsu.py.
    We just add the header/footer HUD on top.
    """
    with _rsu_frame_lock:
        frame = _rsu_frame.copy() if _rsu_frame is not None else None

    with _rsu_lock:
        rsu_count  = len(_rsu_detections)
        rsu_online = _rsu_active()

    if frame is None:
        # placeholder until first frame arrives
        panel = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 30, dtype=np.uint8)
        msg = "Waiting for RSU camera feed..." if rsu_online else "RSU OFFLINE"
        col = (180, 180, 180) if rsu_online else AMBER
        cv2.putText(panel, msg, (30, FRAME_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, col, 2)
    else:
        panel = frame   # already annotated with bboxes from rsu.py

    # RSU online/offline status dot (top-right)
    dot_col = GREEN if rsu_online else AMBER
    dot_lbl = "RSU ONLINE" if rsu_online else "RSU OFFLINE"
    _header_bar(panel, "Panel B — RSU Camera (Roadside View)", (0, 120, 0))
    cv2.circle(panel, (FRAME_WIDTH - 90, 22), 7, dot_col, -1)
    cv2.putText(panel, dot_lbl, (FRAME_WIDTH - 78, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, dot_col, 1)
    _footer_bar(panel, f"RSU detections: {rsu_count}  (green = RSU YOLO bboxes)", GREEN)

    return panel


# ──────────────────────────────────────────────────────────────────────
# Panel C — Fused ego view with geometrically projected RSU detections
# ──────────────────────────────────────────────────────────────────────

def draw_fused_panel(frame, results, model_names, H):
    """
    RIGHT panel — ego camera frame with two layers:
      1. Ego YOLO detections  (solid red boxes)  — same as Panel A
      2. RSU-only detections  (dashed green boxes) — geometrically projected
         from RSU image space into ego image space using homography H.

    The dashed green box appears at the correct position BEHIND the truck
    from the ego camera's point of view — proving the system understands
    WHERE the hidden object is in ego's frame of reference.
    """
    panel = frame.copy()

    # build ego class set (to find what ego is MISSING)
    ego_classes = {model_names[int(b.cls)] for b in results.boxes}

    # draw ego detections — solid red (identical to Panel A for fair comparison)
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls  = model_names[int(box.cls)]
        conf = float(box.conf)
        _draw_solid_box(panel, x1, y1, x2, y2, f"{cls} {conf:.2f}", RED)

    # get latest RSU detections safely
    with _rsu_lock:
        rsu_dets   = list(_rsu_detections)
        rsu_online = _rsu_active()

    hidden_objs = []
    alert_y     = 48   # fallback alert bar y-position (used only if H is None)

    for det in rsu_dets:
        cls  = det["class"]
        conf = det["conf"]
        bbox = det["bbox"]   # [x1, y1, x2, y2] in RSU image pixel coordinates

        if cls in ego_classes:
            continue   # ego already sees this — no need to project

        hidden_objs.append(det)

        if H is not None:
            # ── Geometric projection via homography ───────────────────
            # Transform RSU foot-point → ego image coordinates
            projected = project_rsu_bbox_to_ego(bbox, H)
            if projected:
                px1, py1, px2, py2 = projected

                # dashed green box = detected by RSU, projected into ego frame
                _draw_dashed_box(panel, px1, py1, px2, py2, GREEN)

                # label: "[RSU] person 0.91"
                label = f"[RSU] {cls} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(panel, (px1, py1 - th - 6), (px1 + tw + 4, py1), GREEN, -1)
                cv2.putText(panel, label, (px1 + 2, py1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

                # small downward arrow at the foot point (ground contact)
                foot_x = (px1 + px2) // 2
                cv2.arrowedLine(panel, (foot_x, py2), (foot_x, min(py2 + 22, FRAME_HEIGHT - 5)),
                                GREEN, 2, tipLength=0.4)
        else:
            # ── Fallback: text alert bar (no homography file) ─────────
            bar = f"  [RSU]  {cls.upper()}  detected  ({conf:.0%} conf)"
            cv2.rectangle(panel, (0, alert_y), (FRAME_WIDTH, alert_y + 30), (0, 140, 0), -1)
            cv2.putText(panel, bar, (6, alert_y + 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.57, WHITE, 2)
            alert_y += 34

    total_count = len(results.boxes) + len(hidden_objs)
    mode_note   = "dashed = RSU projected" if H is not None else "run calibrate.py for projection"

    _header_bar(panel, "Panel C — Fused View (Ego Perspective + RSU)", (0, 140, 0))
    _footer_bar(panel, f"Fused  |  Objects: {total_count}  |  {mode_note}", GREEN)

    # RSU status dot
    dot_col = GREEN if rsu_online else AMBER
    dot_lbl = "RSU ONLINE" if rsu_online else "RSU OFFLINE"
    cv2.circle(panel, (FRAME_WIDTH - 90, 22), 7, dot_col, -1)
    cv2.putText(panel, dot_lbl, (FRAME_WIDTH - 78, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, dot_col, 1)

    # flashing critical warning banner (person / bicycle / motorcycle)
    critical = {d["class"] for d in hidden_objs} & CRITICAL_CLASSES
    if critical:
        flash = int(time.time() * 2) % 2 == 0   # alternates every 0.5s
        bcol  = (0, 180, 0) if flash else (0, 90, 0)
        names = " + ".join(c.upper() for c in critical)
        cv2.rectangle(panel, (0, FRAME_HEIGHT - 64), (FRAME_WIDTH, FRAME_HEIGHT - 30), bcol, -1)
        cv2.putText(panel, f"  !! {names} AHEAD  (RSU WARNING) !!",
                    (8, FRAME_HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.58, WHITE, 2)

    return panel, total_count


# ──────────────────────────────────────────────────────────────────────
# Top metrics bar (spans full display width)
# ──────────────────────────────────────────────────────────────────────

def draw_metrics_bar(single_count, coop_count):
    """
    Thin bar above all three panels showing the key quantitative comparison:
    single-agent object count vs cooperative object count.
    """
    bar = np.zeros((32, DISPLAY_WIDTH, 3), dtype=np.uint8)
    bar[:] = (20, 20, 20)
    gain = coop_count - single_count
    sign = "+" if gain >= 0 else ""
    cv2.putText(bar,
                f"Single (A): {single_count} obj   |   Cooperative (C): {coop_count} obj   |   "
                f"RSU revealed: {sign}{gain} hidden object(s)",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
    return bar


# ──────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────

def main():
    # start background threads (daemon = stop when main process exits)
    threading.Thread(target=_listen_rsu_detections, daemon=True).start()
    threading.Thread(target=_listen_rsu_frame,      daemon=True).start()

    # load homography — optional, graceful fallback to alert bars if missing
    H = load_homography()

    print(f"[Ego] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Check CAMERA_INDEX in config.py.")

    win_name = "Cooperative Perception Demo — FYP"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    # initial window size — user can maximise on the demo monitor
    cv2.resizeWindow(win_name, DISPLAY_WIDTH, DISPLAY_HEIGHT + 32)

    print("[Ego] Display running. Press Q to quit.")

    divider = np.full((FRAME_HEIGHT, DIVIDER_WIDTH, 3), 160, dtype=np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Ego] Camera read failed — retrying...")
            time.sleep(0.1)
            continue

        # run ego YOLO on this frame
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

        # build all three panels
        panel_a, single_count = draw_ego_panel(frame,   results, model.names)
        panel_b               = draw_rsu_panel()
        panel_c, coop_count   = draw_fused_panel(frame, results, model.names, H)

        # compose: [A | divider | B | divider | C]
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
