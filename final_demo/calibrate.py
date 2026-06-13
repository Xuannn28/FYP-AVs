# ============================================================
# calibrate.py — One-time SIFT homography calibration
#
# Computes a homography matrix (V2 -> ego) using SIFT feature
# matching between the V2 image and one ego image.
# Saves the result as homography.npy for use with coop_perception.py
#
# Usage:
#   python calibrate.py --v2 v2.jpg --ego ego_25.jpg
#   python calibrate.py --v2 v2.jpg --ego ego_25.jpg --output homography.npy
# ============================================================

import argparse
import sys
import cv2
import numpy as np


def compute_homography_sift(v2_img, ego_img, min_matches=10, min_inliers=6):
    gray_v2  = cv2.cvtColor(v2_img,  cv2.COLOR_BGR2GRAY)
    gray_ego = cv2.cvtColor(ego_img, cv2.COLOR_BGR2GRAY)

    sift    = cv2.SIFT_create(nfeatures=4000)
    kp_v2,  des_v2  = sift.detectAndCompute(gray_v2,  None)
    kp_ego, des_ego = sift.detectAndCompute(gray_ego, None)

    print(f"  SIFT keypoints — V2: {len(kp_v2)}   Ego: {len(kp_ego)}")

    if des_v2 is None or des_ego is None or len(kp_v2) < min_matches or len(kp_ego) < min_matches:
        print("[Error] Not enough keypoints detected.")
        return None, None

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw     = matcher.knnMatch(des_v2, des_ego, k=2)

    # Lowe's ratio test — relaxed to 0.80 for large viewpoint differences
    good = [m for m, n in raw if m.distance < 0.80 * n.distance]
    print(f"  Good matches after ratio test: {len(good)}")

    if len(good) < min_matches:
        print(f"[Error] Only {len(good)} good matches — need at least {min_matches}.")
        print("  Try using an image with more texture/features visible in both cameras.")
        return None, None

    src_pts = np.float32([kp_v2[m.queryIdx].pt  for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_ego[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Reprojection threshold relaxed to 8 px to handle large viewpoint changes
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=8.0)
    inliers = int(mask.sum()) if mask is not None else 0
    print(f"  RANSAC inliers: {inliers} / {len(good)}")

    if H is None or inliers < min_inliers:
        print("[Error] Homography could not be computed reliably.")
        return None, None

    return H, mask


def draw_matches_preview(v2_img, ego_img, H, output_path):
    h_ego, w_ego = ego_img.shape[:2]
    warped = cv2.warpPerspective(v2_img, H, (w_ego, h_ego))

    # Side-by-side: warped V2 | ego
    preview = np.hstack([warped, ego_img])
    cv2.putText(preview, "V2 warped into ego frame",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2, cv2.LINE_AA)
    cv2.putText(preview, "Ego image",
                (w_ego + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
    cv2.imwrite(output_path, preview, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  Preview saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SIFT homography calibration — V2 to ego")
    parser.add_argument("--v2",     required=True, help="V2 image path")
    parser.add_argument("--ego",    required=True, help="Ego reference image (use least-occluded, e.g. ego_25.jpg)")
    parser.add_argument("--output", default="homography.npy", help="Output .npy path (default: homography.npy)")
    args = parser.parse_args()

    v2_img  = cv2.imread(args.v2)
    ego_img = cv2.imread(args.ego)

    if v2_img is None:  sys.exit(f"[Error] Cannot read: {args.v2}")
    if ego_img is None: sys.exit(f"[Error] Cannot read: {args.ego}")

    print(f"\n[Calibrate] V2  image : {args.v2}   ({v2_img.shape[1]}x{v2_img.shape[0]})")
    print(f"[Calibrate] Ego image : {args.ego}  ({ego_img.shape[1]}x{ego_img.shape[0]})")
    print(f"[Calibrate] Running SIFT matching...")

    H, mask = compute_homography_sift(v2_img, ego_img)

    if H is None:
        sys.exit("[Calibrate] Failed — homography not saved.")

    np.save(args.output, H)
    print(f"\n[Calibrate] Homography saved → {args.output}")

    preview_path = args.output.replace(".npy", "_preview.jpg")
    draw_matches_preview(v2_img, ego_img, H, preview_path)

    print("\n[Calibrate] Done. Use with coop_perception.py:")
    print(f"  --homography {args.output}")
    print("\n  Check the preview image — V2 warped side should roughly align with ego.")


if __name__ == "__main__":
    main()
