# ============================================================
# rsu.py — Run this on Pi 2 (RSU / Roadside Unit)
#
# Sends two streams to Ego Pi over UDP:
#   Port 5005 → detection JSON  (lightweight, every frame)
#   Port 5006 → annotated JPEG frame (compressed, every frame)
#
# Run: python rsu.py
# ============================================================

import socket
import json
import time
import cv2
import numpy as np
from ultralytics import YOLO
from config import (
    EGO_IP, UDP_PORT, RSU_FRAME_PORT,
    MODEL_PATH, CAMERA_INDEX,
    FRAME_WIDTH, FRAME_HEIGHT,
    CONFIDENCE_THRESHOLD, JPEG_QUALITY
)


def build_payload(results, model_names):
    """Serialize YOLO detections to a JSON-safe dict."""
    detections = []
    for box in results.boxes:
        detections.append({
            "class": model_names[int(box.cls)],
            "conf":  round(float(box.conf), 3),
            "bbox":  [round(v, 1) for v in box.xyxy[0].tolist()]
        })
    return {"agent": "RSU", "ts": time.time(), "detections": detections}


def encode_frame(frame):
    """JPEG-compress frame for UDP transmission. Returns bytes or None if too large."""
    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    data = jpeg.tobytes()
    # UDP max payload = 65507 bytes — skip frame if too large
    if len(data) > 65000:
        return None
    return data


def draw_rsu_overlay(annotated, n_detections, bytes_sent):
    """Draw HUD on RSU local preview window."""
    # Header bar
    cv2.rectangle(annotated, (0, 0), (FRAME_WIDTH, 40), (0, 80, 0), -1)
    cv2.putText(annotated, "RSU (Roadside Unit) Camera",
                (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    # Status dot (green = sending)
    cv2.circle(annotated, (FRAME_WIDTH - 20, 20), 8, (0, 255, 0), -1)
    # Footer
    cv2.rectangle(annotated, (0, FRAME_HEIGHT - 28), (FRAME_WIDTH, FRAME_HEIGHT), (20, 20, 20), -1)
    cv2.putText(annotated, f"Detections: {n_detections}   |   Frame sent: {bytes_sent} B",
                (8, FRAME_HEIGHT - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 255, 180), 1)
    return annotated


def main():
    print(f"[RSU] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Check CAMERA_INDEX in config.py.")

    det_sock   = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)   # detections
    frame_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)   # camera frame

    print(f"[RSU] Sending detections  → {EGO_IP}:{UDP_PORT}")
    print(f"[RSU] Sending camera frame → {EGO_IP}:{RSU_FRAME_PORT}")
    print("[RSU] Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[RSU] Camera read failed — retrying...")
            time.sleep(0.1)
            continue

        # --- Detection ---
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

        # --- Send detection JSON ---
        payload = build_payload(results, model.names)
        det_sock.sendto(json.dumps(payload).encode(), (EGO_IP, UDP_PORT))

        # --- Send annotated frame (JPEG) ---
        annotated = results.plot()   # draws bboxes on the frame
        jpeg_data  = encode_frame(annotated)
        bytes_label = 0
        if jpeg_data:
            frame_sock.sendto(jpeg_data, (EGO_IP, RSU_FRAME_PORT))
            bytes_label = len(jpeg_data)

        # --- Local preview ---
        preview = draw_rsu_overlay(annotated, len(results.boxes), bytes_label)
        cv2.imshow("RSU View", preview)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    det_sock.close()
    frame_sock.close()
    print("[RSU] Stopped.")


if __name__ == "__main__":
    main()
