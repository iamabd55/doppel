"""Unit tests for SIFT horizontal flip pass and coordinate mapping.

Verifies:
1. Coordinate mapping round-trip and boundary conditions (x_orig = W - 1 - x_flip).
2. Synthetic mirrored copy-move detection: confirming that check_flip=False misses
   mirrored clones while check_flip=True successfully detects them.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import pytest
from detector.sift import map_flip_coordinates, detect_sift


def test_map_flip_coordinates_roundtrip():
    """Mapping back and forth must be completely lossless."""
    width = 640
    pts = np.array([
        [10.0, 20.0],
        [320.0, 150.0],
        [639.0, 400.0],
        [0.0, 0.0],
    ], dtype=np.float32)

    mapped_once = map_flip_coordinates(pts, width)
    mapped_twice = map_flip_coordinates(mapped_once, width)

    np.testing.assert_allclose(pts, mapped_twice, atol=1e-5)


def test_map_flip_coordinates_boundaries():
    """Boundary coordinates must map to exact opposite margins."""
    width = 512
    # Left edge maps to right edge
    left_pt = np.array([[0.0, 100.0]], dtype=np.float32)
    right_mapped = map_flip_coordinates(left_pt, width)
    assert right_mapped[0, 0] == float(width - 1)
    assert right_mapped[0, 1] == 100.0

    # Right edge maps to left edge
    right_pt = np.array([[float(width - 1), 250.0]], dtype=np.float32)
    left_mapped = map_flip_coordinates(right_pt, width)
    assert left_mapped[0, 0] == 0.0
    assert left_mapped[0, 1] == 250.0

    # Center axis maps to itself
    center_x = float(width - 1) / 2.0
    center_pt = np.array([[center_x, 300.0]], dtype=np.float32)
    center_mapped = map_flip_coordinates(center_pt, width)
    assert abs(center_mapped[0, 0] - center_x) < 1e-5


def test_mirrored_copy_detection():
    """Mirrored copy must be missed by standard SIFT but caught by the flip pass."""
    # Create a 400x600 synthetic image with a textured patch
    np.random.seed(42)
    img = np.full((400, 600), 200, dtype=np.uint8)

    # Add random noise and geometric shapes to create rich SIFT keypoints
    noise = np.random.normal(0, 10, (400, 600)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Draw distinctive asymmetric textured objects at source location (x: 50..180, y: 100..260)
    for r in range(120, 240, 30):
        # Asymmetric triangles
        pts = np.array([[70, r], [160, r + 8], [80, r + 22]], dtype=np.int32)
        cv2.fillPoly(img, [pts], (30, 30, 30))
        # Asymmetric text character
        cv2.putText(img, "R", (105, r + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    source_patch = img[100:260, 50:180].copy()

    # Horizontally mirror the patch and paste it onto target location (x: 400..530, y: 100..260)
    mirrored_patch = cv2.flip(source_patch, 1)
    img[100:260, 400:530] = mirrored_patch

    # Count how many inlier pairs connect source region (x < 250) to mirrored target (x > 350)
    def count_cross_matches(p1_arr, p2_arr):
        if len(p1_arr) == 0:
            return 0
        cnt = 0
        for p1, p2 in zip(p1_arr, p2_arr):
            is_cross = (p1[0] < 250 and p2[0] > 350) or (p1[0] > 350 and p2[0] < 250)
            if is_cross:
                cnt += 1
        return cnt

    # 1. Standard SIFT without flip pass must find ZERO matches connecting source to mirrored target
    mask_noflip, inl1_noflip, inl2_noflip, _ = detect_sift(
        img,
        ratio_threshold=0.6,
        min_inliers=4,
        check_flip=False,
    )
    cross_noflip = count_cross_matches(inl1_noflip, inl2_noflip)
    assert cross_noflip == 0, f"Standard SIFT unexpectedly connected {cross_noflip} pairs to mirrored clone!"

    # 2. SIFT with flip pass must successfully connect source to mirrored target
    mask_flip, inl1_flip, inl2_flip, transforms = detect_sift(
        img,
        ratio_threshold=0.6,
        min_inliers=4,
        check_flip=True,
    )
    cross_flip = count_cross_matches(inl1_flip, inl2_flip)
    assert cross_flip >= 4, f"Flip pass only found {cross_flip} cross matches (expected >= 4)!"
    assert len(transforms) >= 1
    assert (mask_flip > 0).sum() > 0


if __name__ == "__main__":
    test_map_flip_coordinates_roundtrip()
    test_map_flip_coordinates_boundaries()
    test_mirrored_copy_detection()
    print("All flip pass unit tests passed successfully!")
