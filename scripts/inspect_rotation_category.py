"""Inspect SIFT performance across CoMoFoD rotation dataset (images 041-050).

Analyzes ground truth keypoint density, inlier counts, transformation parameters
(rotation angle and scale), and compares true forgeries against architectural repetitions.
"""
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detector.sift import detect_and_compute, match_within_image, verify_matches_ransac
from detector.evaluate import scores

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "comofod")
ROT_DIR = os.path.join(DATA_DIR, "rotation")


def extract_transform_params(M):
    """Extract rotation angle (degrees) and scale factor from a 2x3 affine matrix."""
    if M is None:
        return 0.0, 1.0
    a, b = M[0, 0], M[0, 1]
    scale = np.sqrt(a**2 + b**2)
    angle = np.degrees(np.arctan2(b, a))
    return angle, scale


def main():
    print(f"{'Img':>5} | {'GT Px':>8} | {'Total KP':>8} | {'GT KP':>6} | {'Raw Match':>9} | {'Inliers':>8} | {'Angle (deg)':>11} | {'Scale':>6} | {'Status'}")
    print("-" * 90)

    for i in range(41, 51):
        tag = f"{i:03d}"
        f_path = os.path.join(ROT_DIR, f"{tag}_F.png")
        m_path = os.path.join(ROT_DIR, f"{tag}_M.png")

        if not os.path.exists(f_path):
            continue

        gray = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(m_path) else None
        gt_bin = (gt > 10).astype(np.uint8) if gt is not None else None

        kps, descs = detect_and_compute(gray)

        # Count keypoints landing inside the ground truth forged region
        gt_kps = 0
        if gt_bin is not None:
            for kp in kps:
                x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
                if 0 <= y < gt_bin.shape[0] and 0 <= x < gt_bin.shape[1]:
                    if gt_bin[y, x] > 0:
                        gt_kps += 1

        pts1, pts2, _ = match_within_image(kps, descs, ratio_threshold=0.6, min_dist=30.0)
        inliers1, inliers2, transforms = verify_matches_ransac(
            pts1, pts2, ransac_threshold=3.0, min_inliers=4, cluster_threshold=50.0
        )

        n_inliers = len(inliers1)
        if len(transforms) > 0:
            angle, scale = extract_transform_params(transforms[0])
        else:
            angle, scale = 0.0, 1.0

        status = "DETECTED" if n_inliers >= 4 else "MISSED"
        gt_px = int(gt_bin.sum()) if gt_bin is not None else 0

        print(
            f"{tag:>5} | {gt_px:8d} | {len(kps):8d} | {gt_kps:6d} | {len(pts1):9d} | {n_inliers:8d} | {angle:11.1f} | {scale:6.2f} | {status}"
        )

    print("-" * 90)


if __name__ == "__main__":
    main()
