# ============================================================
# calibrate.py — One-time homography calibration
#
# What it produces:
#   homography.npy — a 3x3 matrix H that maps any point on the
#   table surface from RSU camera coordinates to Ego camera
#   coordinates. ego.py loads this to project RSU detections
#   (e.g. a hidden pedestrian) into the ego camera's frame.
#
# How to run:
#   Step 1 — Place 4 markers on the table (e.g. coloured tape crosses)
#            at the corners of the visible area.
#   Step 2 — Take one clear photo from each camera and save them:
#               ego_sample.jpg   (ego vehicle camera perspective)
#               rsu_sample.jpg   (RSU camera perspective)
#            OR provide live camera indices (see --live flag below).
#   Step 3 — Run:
#               python calibrate.py                          (uses saved images)
#               python calibrate.py --live 0 1              (live cameras)
#   Step 4 — Click the SAME 4 markers in BOTH images (order matters!)
#   Step 5 — homography.npy is saved → copy to ego Pi if needed.
#
# NOTE: Run this on whichever machine has access to both images/cameras.
#       Copy homography.npy to Pi 1 (Ego) before running ego.py.
# ============================================================

import cv2
import numpy as np
import sys
import os
from config import HOMOGRAPHY_FILE, FRAME_WIDTH, FRAME_HEIGHT

# Collected click points
_clicked_points = []
_current_label  = ""


def _mouse_callback(event, x, y, flags, param):
    """Record up to 4 clicks per image."""
    if event == cv2.EVENT_LBUTTONDOWN and len(_clicked_points) < 4:
        _clicked_points.append((x, y))
        print(f"  [{_current_label}] Point {len(_clicked_points)}: ({x}, {y})")


def collect_points(window_name, image, label):
    """
    Show image, let user click 4 reference points.
    Returns list of 4 (x, y) tuples.
    """
    global _clicked_points, _current_label
    _clicked_points = []
    _current_label  = label

    display = image.copy()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, _mouse_callback)

    print(f"\n[Calibrate] Click the 4 reference markers on the {label} image.")
    print("            Click them in the SAME ORDER as you did for the other camera.")
    print("            Press SPACE to confirm when 4 points are selected.")

    while True:
        frame = display.copy()

        # draw clicked points so far
        for i, (px, py) in enumerate(_clicked_points):
            cv2.circle(frame, (px, py), 8, (0, 255, 0), -1)
            cv2.putText(frame, str(i + 1), (px + 10, py - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # instruction overlay
        msg = f"{label}: {len(_clicked_points)}/4 points  |  SPACE to confirm"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (20, 20, 20), -1)
        cv2.putText(frame, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord(' ') and len(_clicked_points) == 4:
            break
        if key == ord('q'):
            print("[Calibrate] Cancelled.")
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyWindow(window_name)
    return list(_clicked_points)


def load_image_or_camera(source):
    """
    Load an image from a file path or from a live camera index.
    Returns a BGR numpy array.
    """
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        print(f"[Calibrate] Capturing from camera {source} — press SPACE to grab frame.")
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Cannot read camera {source}")
            cv2.imshow(f"Camera {source} — press SPACE to grab", frame)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                cv2.destroyAllWindows()
                cap.release()
                return frame
    else:
        if not os.path.exists(source):
            raise FileNotFoundError(f"Image not found: {source}")
        return cv2.imread(source)


def main():
    # Parse arguments
    if "--live" in sys.argv:
        idx = sys.argv.index("--live")
        ego_source = int(sys.argv[idx + 1])
        rsu_source = int(sys.argv[idx + 2])
        print(f"[Calibrate] Live mode — ego camera: {ego_source}, RSU camera: {rsu_source}")
    else:
        ego_source = "ego_sample.jpg"
        rsu_source = "rsu_sample.jpg"
        print(f"[Calibrate] Image mode — ego: {ego_source}, RSU: {rsu_source}")
        print("            To use live cameras: python calibrate.py --live 0 1")

    ego_img = load_image_or_camera(ego_source)
    rsu_img = load_image_or_camera(rsu_source)

    print("\n[Calibrate] ── Step 1: Click 4 points on the EGO image ──")
    print("            These should be the same physical markers on your table.")
    ego_pts = collect_points("EGO image — click 4 markers", ego_img, "EGO")

    print("\n[Calibrate] ── Step 2: Click the SAME 4 points on the RSU image ──")
    print("            Click them in the EXACT SAME ORDER as the ego image!")
    rsu_pts = collect_points("RSU image — click 4 markers", rsu_img, "RSU")

    # Convert to numpy arrays for OpenCV
    ego_pts_np = np.array(ego_pts, dtype=np.float32)
    rsu_pts_np = np.array(rsu_pts, dtype=np.float32)

    # Compute homography: maps RSU points → Ego points
    # (src = RSU, dst = Ego — so H transforms RSU coords into Ego coords)
    H, mask = cv2.findHomography(rsu_pts_np, ego_pts_np, cv2.RANSAC)

    if H is None:
        print("[Calibrate] ERROR: Could not compute homography. Try again with clearer points.")
        sys.exit(1)

    np.save(HOMOGRAPHY_FILE, H)
    print(f"\n[Calibrate] Homography saved to: {HOMOGRAPHY_FILE}")
    print("            Copy this file to Pi 1 (Ego) if calibrated on a different machine:")
    print(f"              scp {HOMOGRAPHY_FILE} pi@<EGO_PI_IP>:~/YOLO/")
    print("\n[Calibrate] Matrix H (RSU → Ego):")
    print(H)

    # Verification — warp RSU image onto ego image to check alignment
    print("\n[Calibrate] Showing verification overlay (press any key to close)...")
    warped = cv2.warpPerspective(rsu_img, H, (FRAME_WIDTH, FRAME_HEIGHT))
    blend  = cv2.addWeighted(ego_img, 0.6, warped, 0.4, 0)
    cv2.putText(blend, "Verification: RSU warped onto Ego frame (check alignment)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Calibration Verification", blend)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
