"""Test within-image SIFT matching for copy-move candidate detection.

Matches descriptors within the same image using kNN (k=3), removes self-matches,
applies Lowe's ratio test and minimum distance thresholding, and visualizes match lines.
"""
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detector.sift import detect_and_compute, match_within_image, draw_matches

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "comofod")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "phase2_sift")
os.makedirs(OUT_DIR, exist_ok=True)


def load_category_image(category, filename):
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
        ("scaling", "081_F.png", "Forged (Scaling)"),
        ("original", "001_O.png", "Original (Untouched)"),
    ]

    print(f"{'Category':>12} | {'Image':>9} | {'Keypoints':>10} | {'Matches (r=0.6)':>16} | {'Matches (r=0.5)':>16} | {'Output Visualization'}")
    print("-" * 105)

    for cat, filename, desc in test_cases:
        gray, color = load_category_image(cat, filename)
        if gray is None:
            print(f"{cat:>12} | {filename:>9} | NOT FOUND")
            continue

        kps, descs = detect_and_compute(gray)

        # Match with default ratio 0.6
        pts1_06, pts2_06, ratios_06 = match_within_image(
            kps, descs, ratio_threshold=0.6, min_dist=30.0
        )

        # Match with stricter ratio 0.5
        pts1_05, pts2_05, _ = match_within_image(
            kps, descs, ratio_threshold=0.5, min_dist=30.0
        )

        # Save visualization of matches
        vis = draw_matches(color, pts1_06, pts2_06, thickness=2)
        out_name = f"matches_{cat}_{filename.replace('.png', '')}.png"
        out_path = os.path.join(OUT_DIR, out_name)
        cv2.imwrite(out_path, vis)

        print(
            f"{cat:>12} | {filename:>9} | {len(kps):10d} | {len(pts1_06):16d} | {len(pts1_05):16d} | {os.path.relpath(out_path)}"
        )

    print("-" * 105)
    print("Within-image matching test complete. Visualizations saved in results/phase2_sift/.")


if __name__ == "__main__":
    main()
