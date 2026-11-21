"""Test SIFT localization (Step 2.5).

Evaluates the generated binary forgery masks against ground truth masks
using detector/evaluate.py (precision, recall, F1) and saves visual overlays.
"""
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detector.sift import detect_sift
from detector.evaluate import scores

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "comofod")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "phase2_sift")
os.makedirs(OUT_DIR, exist_ok=True)

CATEGORIES = ["translation", "rotation", "scaling", "distortion", "combination"]


def load_sample(category, filename):
    for d in [category, "translation", "rotation", "scaling", "distortion", "combination", ""]:
        p = os.path.join(DATA_DIR, d, filename)
        if os.path.exists(p):
            gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            color = cv2.imread(p, cv2.IMREAD_COLOR)

            tag = filename[:3]
            m_path = os.path.join(DATA_DIR, d, f"{tag}_M.png")
            gt = None
            if os.path.exists(m_path):
                raw_m = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
                gt = (raw_m > 10).astype(np.uint8) * 255
            return gray, color, gt
    return None, None, None


def make_overlay(color, mask):
    """Create a semi-transparent red overlay on top of detected regions."""
    overlay = color.copy()
    red = np.zeros_like(color)
    red[:, :] = (0, 0, 255)
    flagged = mask > 0
    overlay[flagged] = cv2.addWeighted(color[flagged], 0.5, red[flagged], 0.5, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    return overlay


def main():
    test_cases = [
        ("translation", "001_F.png"),
        ("rotation", "062_F.png"),
        ("scaling", "082_F.png"),
        ("original", "001_O.png"),
    ]

    print(f"{'Category':>12} | {'Image':>9} | {'Precision':>9} | {'Recall':>8} | {'F1':>8} | {'Flagged Px':>11} | {'Output Visual'}")
    print("-" * 95)

    for cat, filename in test_cases:
        gray, color, gt = load_sample(cat, filename)
        if gray is None:
            print(f"{cat:>12} | {filename:>9} | NOT FOUND")
            continue

        mask, inliers1, inliers2, transforms = detect_sift(
            gray,
            ratio_threshold=0.6,
            min_dist=30.0,
            ransac_threshold=3.0,
            min_inliers=4,
            cluster_threshold=50.0,
            mask_radius=20,
            use_convex_hull=True,
        )

        flagged_px = int((mask > 0).sum())

        if gt is not None:
            p, r, f1 = scores(mask, gt)
            gt_panel = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
        else:
            p, r, f1 = 0.0, 0.0, 0.0
            gt_panel = np.zeros_like(color)

        overlay = make_overlay(color, mask)
        mask_panel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        comparison = np.hstack([overlay, mask_panel, gt_panel])

        out_name = f"localization_{cat}_{filename.replace('.png', '')}.png"
        out_path = os.path.join(OUT_DIR, out_name)
        cv2.imwrite(out_path, comparison)

        print(
            f"{cat:>12} | {filename:>9} | {p:9.3f} | {r:8.3f} | {f1:8.3f} | {flagged_px:11d} | {os.path.relpath(out_path)}"
        )

    print("-" * 95)
    print("Localization testing complete. Tri-panel comparisons saved in results/phase2_sift/.")


if __name__ == "__main__":
    main()
