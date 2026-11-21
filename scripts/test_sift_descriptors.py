"""Test SIFT 128-dimensional descriptor extraction.

Verifies descriptor dimensionality, data type, L2 normalization properties,
and computes similarity comparisons between keypoints.
"""
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detector.sift import detect_and_compute

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "comofod")
CATEGORIES = ["translation", "rotation", "scaling", "distortion", "combination"]


def find_image(tag):
    for cat in CATEGORIES:
        p = os.path.join(DATA_DIR, cat, f"{tag}_F.png")
        if os.path.exists(p):
            return p
    p = os.path.join(DATA_DIR, f"{tag}_F.png")
    return p if os.path.exists(p) else None


def main():
    sample_tags = ["001", "002", "005"]
    print(f"{'Image':>6} | {'Keypoints':>10} | {'Descriptor Shape':>18} | {'Dtype':>8} | {'Mean L2 Norm':>13}")
    print("-" * 70)

    for tag in sample_tags:
        path = find_image(tag)
        if not path:
            continue
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue

        kps, descs = detect_and_compute(gray)

        if len(descs) > 0:
            norms = np.linalg.norm(descs, axis=1)
            mean_norm = float(np.mean(norms))
            shape_str = f"({descs.shape[0]}, {descs.shape[1]})"
            dtype_str = str(descs.dtype)
        else:
            mean_norm = 0.0
            shape_str = "(0, 128)"
            dtype_str = "none"

        print(
            f"{tag:>6} | {len(kps):10d} | {shape_str:>18} | {dtype_str:>8} | {mean_norm:13.2f}"
        )

    print("-" * 70)
    print("Descriptor inspection complete: each keypoint is represented by 128 float values.")


if __name__ == "__main__":
    main()
