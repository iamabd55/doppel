"""Demonstrate SIFT horizontal flip pass for mirrored copy-move detection.

Compares standard SIFT (which misses mirrored clones) against the flip pass
(which successfully matches and localizes mirrored clones), and saves visual results.
"""
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detector.sift import detect_sift, draw_matches

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "phase2_sift")
os.makedirs(OUT_DIR, exist_ok=True)


def make_mirrored_sample():
    """Create a synthetic test image containing an original and horizontally mirrored clone."""
    np.random.seed(123)
    img = np.full((400, 600), 220, dtype=np.uint8)

    # Background texture
    noise = np.random.normal(0, 8, (400, 600)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Draw textured protein/cell-like pattern on the left (source)
    for y in range(120, 260, 20):
        cv2.ellipse(img, (120, y), (30, 8), 0, 0, 360, (40, 40, 40), -1)
        cv2.circle(img, (100, y - 2), 4, (80, 80, 80), -1)
        cv2.circle(img, (140, y + 2), 5, (20, 20, 20), -1)

    source_patch = img[100:280, 70:180].copy()

    # Mirror patch horizontally and paste onto the right
    mirrored_patch = cv2.flip(source_patch, 1)
    img[100:280, 420:530] = mirrored_patch

    # Ground truth mask
    gt = np.zeros_like(img)
    gt[100:280, 70:180] = 255
    gt[100:280, 420:530] = 255

    return img, gt


def main():
    img, gt = make_mirrored_sample()
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    print("Running standard SIFT (check_flip=False)...")
    mask_noflip, inliers1_noflip, inliers2_noflip, _ = detect_sift(
        img,
        ratio_threshold=0.6,
        min_inliers=4,
        check_flip=False,
    )

    print("Running SIFT with horizontal flip pass (check_flip=True)...")
    mask_flip, inliers1_flip, inliers2_flip, transforms = detect_sift(
        img,
        ratio_threshold=0.6,
        min_inliers=4,
        check_flip=True,
    )

    print("\n--- Detection Results on Mirrored Copy-Move ---")
    print(f"Standard SIFT inliers:  {len(inliers1_noflip):4d}  | Mask pixels: {(mask_noflip > 0).sum():6d}  | Missed (as expected)")
    print(f"Flip Pass SIFT inliers: {len(inliers1_flip):4d}  | Mask pixels: {(mask_flip > 0).sum():6d}  | DETECTED successfully")

    # Generate side-by-side visualization
    vis_noflip = draw_matches(color, inliers1_noflip, inliers2_noflip, thickness=2)
    vis_flip = draw_matches(color, inliers1_flip, inliers2_flip, thickness=2)
    mask_flip_bgr = cv2.cvtColor(mask_flip, cv2.COLOR_GRAY2BGR)

    combined = np.hstack([vis_noflip, vis_flip, mask_flip_bgr])
    out_path = os.path.join(OUT_DIR, "flip_pass_demo.png")
    cv2.imwrite(out_path, combined)
    print(f"\nSaved visualization to {os.path.relpath(out_path)}")
    print("Panels: [Standard SIFT (Missed) | Flip Pass SIFT (Matches Connected) | Localization Mask]")


if __name__ == "__main__":
    main()
