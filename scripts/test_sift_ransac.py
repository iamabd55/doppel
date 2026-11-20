"""Test SIFT geometric verification with RANSAC affine estimation.

Demonstrates how RANSAC filters out spurious architectural matches (outliers)
while keeping true copy-move clones (inliers) across translation, rotation, and scaling.
"""
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detector.sift import (
    detect_and_compute,
    match_within_image,
    verify_matches_ransac,
    draw_matches,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "comofod")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "phase2_sift")
os.makedirs(OUT_DIR, exist_ok=True)


def load_category_image(category, filename):
    # Try specified folder, then search other categories
    search_dirs = [category, "translation", "rotation", "scaling", "distortion", "combination", ""]
    for d in search_dirs:
        p = os.path.join(DATA_DIR, d, filename)
        if os.path.exists(p):
            return cv2.imread(p, cv2.IMREAD_GRAYSCALE), cv2.imread(p, cv2.IMREAD_COLOR)
    return None, None


def main():
    test_cases = [
        ("translation", "001_F.png", "Forged (Translation)"),
        ("rotation", "062_F.png", "Forged (Rotation)"),
        ("scaling", "082_F.png", "Forged (Scaling)"),
        ("original", "001_O.png", "Original Untouched (False Alarm Test)"),
    ]

    print(f"{'Category':>12} | {'Image':>9} | {'Raw Matches':>12} | {'RANSAC Inliers':>15} | {'Angle (deg)':>12} | {'Status':>10}")
    print("-" * 85)

    for cat, filename, desc in test_cases:
        gray, color = load_category_image(cat, filename)
        if gray is None:
            print(f"{cat:>12} | {filename:>9} | NOT FOUND")
            continue

        kps, descs = detect_and_compute(gray)
        pts1, pts2, _ = match_within_image(kps, descs, ratio_threshold=0.6, min_dist=30.0)

        inliers1, inliers2, transforms = verify_matches_ransac(
            pts1, pts2, ransac_threshold=3.0, min_inliers=4, cluster_threshold=50.0
        )

        n_raw = len(pts1)
        n_inliers = len(inliers1)
        status = "FLAGGED" if n_inliers >= 4 else "CLEAN"

        angle_str = "-"
        if len(transforms) > 0:
            M = transforms[0]
            deg = float(np.degrees(np.arctan2(M[0, 1], M[0, 0])))
            angle_str = f"{deg:8.1f}°"

        # Save inlier visualization
        vis = draw_matches(color, inliers1, inliers2, thickness=2)
        out_name = f"ransac_{cat}_{filename.replace('.png', '')}.png"
        cv2.imwrite(os.path.join(OUT_DIR, out_name), vis)

        print(
            f"{cat:>12} | {filename:>9} | {n_raw:12d} | {n_inliers:15d} | {angle_str:>12} | {status:>10}"
        )

    print("-" * 85)
    print("Geometric verification complete. Inlier overlays saved in results/phase2_sift/.")


if __name__ == "__main__":
    main()
