"""Test SIFT keypoint detection on CoMoFoD samples.

Demonstrates keypoint extraction, visualizes keypoints with scale and orientation,
and compares keypoint density across textured vs flat regions.
"""
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detector.sift import extract_keypoints, draw_keypoints

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "comofod")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "phase2_sift")
os.makedirs(OUT_DIR, exist_ok=True)

CATEGORIES = ["translation", "rotation", "scaling", "distortion", "combination"]


def find_image(tag):
    for cat in CATEGORIES:
        p = os.path.join(DATA_DIR, cat, f"{tag}_F.png")
        if os.path.exists(p):
            return p
    p = os.path.join(DATA_DIR, f"{tag}_F.png")
    return p if os.path.exists(p) else None


def analyze_texture_distribution(gray, keypoints, block_size=16, min_std=4.0):
    """Count keypoints located in textured vs flat image patches."""
    h, w = gray.shape
    textured_kps = 0
    flat_kps = 0

    for kp in keypoints:
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        r0 = max(0, y - block_size // 2)
        r1 = min(h, y + block_size // 2)
        c0 = max(0, x - block_size // 2)
        c1 = min(w, x + block_size // 2)

        patch = gray[r0:r1, c0:c1]
        std = np.std(patch)
        if std >= min_std:
            textured_kps += 1
        else:
            flat_kps += 1

    return textured_kps, flat_kps


def main():
    sample_tags = ["001", "002", "005"]
    print(f"{'Image':>6} | {'Total KPs':>10} | {'Textured (std>=4)':>18} | {'Flat (std<4)':>13} | {'Saved To'}")
    print("-" * 80)

    for tag in sample_tags:
        path = find_image(tag)
        if not path:
            print(f"{tag:>6} | NOT FOUND")
            continue

        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"{tag:>6} | FAILED TO READ")
            continue

        keypoints = extract_keypoints(gray)
        tex_cnt, flat_cnt = analyze_texture_distribution(gray, keypoints)

        vis = draw_keypoints(gray, keypoints, rich=True)
        out_path = os.path.join(OUT_DIR, f"keypoints_{tag}.png")
        cv2.imwrite(out_path, vis)

        print(
            f"{tag:>6} | {len(keypoints):10d} | {tex_cnt:18d} | {flat_cnt:13d} | {os.path.relpath(out_path)}"
        )

    print("-" * 80)
    print("Done. Visualizations saved in results/phase2_sift/")


if __name__ == "__main__":
    main()
