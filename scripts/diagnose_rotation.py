"""Diagnose SIFT detection on CoMoFoD rotation samples.

Inspects keypoint matches against ground truth mask, checks angle differences,
and diagnoses why rotation matches were missed or overshadowed by architectural repetitions.
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


def main():
    tag = "045"
    img_path = os.path.join(DATA_DIR, "rotation", f"{tag}_F.png")
    mask_path = os.path.join(DATA_DIR, "rotation", f"{tag}_M.png")

    if not os.path.exists(img_path):
        # search anywhere in comofod
        for cat in ["translation", "rotation", "scaling", "distortion", "combination"]:
            p = os.path.join(DATA_DIR, cat, f"{tag}_F.png")
            if os.path.exists(p):
                img_path = p
                mask_path = os.path.join(DATA_DIR, cat, f"{tag}_M.png")
                break

    print(f"Testing image: {img_path}")
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    color = cv2.imread(img_path, cv2.IMREAD_COLOR)
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else None

    if gt_mask is not None:
        gt_bin = (gt_mask > 10).astype(np.uint8)
        print(f"Ground truth forged pixels: {gt_bin.sum()} (covers source and target)")

    # 1. Detect and compute with default vs relaxed ratio
    kps, descs = detect_and_compute(gray)
    print(f"Total keypoints detected: {len(kps)}")

    for ratio in [0.6, 0.7, 0.75]:
        pts1, pts2, ratios = match_within_image(kps, descs, ratio_threshold=ratio, min_dist=30.0)
        print(f"\n--- Ratio threshold: {ratio} ---")
        print(f"Raw matches found: {len(pts1)}")

        if gt_mask is not None and len(pts1) > 0:
            # Check how many points land inside the ground truth forged regions
            in_gt1 = [gt_bin[int(round(p[1])), int(round(p[0]))] > 0 for p in pts1]
            in_gt2 = [gt_bin[int(round(p[1])), int(round(p[0]))] > 0 for p in pts2]
            both_in_gt = sum([g1 and g2 for g1, g2 in zip(in_gt1, in_gt2)])
            print(f"Matches where BOTH points are inside ground truth: {both_in_gt}/{len(pts1)}")
            print(f"Matches where points are OUTSIDE ground truth (architectural): {len(pts1) - both_in_gt}/{len(pts1)}")

    # Check keypoint angle differences (rotation angles)
    # Match without cluster restriction to inspect raw geometry
    pts1_7, pts2_7, _ = match_within_image(kps, descs, ratio_threshold=0.75, min_dist=30.0)
    vis = draw_matches(color, pts1_7, pts2_7, thickness=2)
    cv2.imwrite(os.path.join(OUT_DIR, "diagnose_045_matches.png"), vis)
    print(f"\nSaved raw match visualization to results/phase2_sift/diagnose_045_matches.png")


if __name__ == "__main__":
    main()
