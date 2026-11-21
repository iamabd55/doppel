"""SIFT keypoint and descriptor feature extraction.

Reference:
    Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints.
    International Journal of Computer Vision, 60(2), 91-110.
"""
from typing import List, Tuple
import cv2
import numpy as np


def extract_keypoints(
    gray: np.ndarray,
    nfeatures: int = 0,
    contrast_threshold: float = 0.04,
    edge_threshold: float = 10.0,
    sigma: float = 1.6,
) -> List[cv2.KeyPoint]:
    """Detect SIFT keypoints on a grayscale image."""
    if gray.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale array.")

    sift = cv2.SIFT_create(
        nfeatures=nfeatures,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
        sigma=sigma,
    )
    keypoints = sift.detect(gray, None)
    return list(keypoints)


def extract_descriptors(
    gray: np.ndarray,
    keypoints: List[cv2.KeyPoint],
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """Compute 128-dimensional SIFT descriptors for existing keypoints."""
    if gray.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale array.")

    sift = cv2.SIFT_create()
    kps, descs = sift.compute(gray, keypoints)
    if descs is None:
        descs = np.empty((0, 128), dtype=np.float32)
    return list(kps), descs


def detect_and_compute(
    gray: np.ndarray,
    nfeatures: int = 0,
    contrast_threshold: float = 0.04,
    edge_threshold: float = 10.0,
    sigma: float = 1.6,
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """Detect keypoints and compute 128-d descriptors in a single pass."""
    if gray.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale array.")

    sift = cv2.SIFT_create(
        nfeatures=nfeatures,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
        sigma=sigma,
    )
    kps, descs = sift.detectAndCompute(gray, None)
    if descs is None:
        descs = np.empty((0, 128), dtype=np.float32)
    return list(kps), descs


def draw_keypoints(
    image: np.ndarray,
    keypoints: List[cv2.KeyPoint],
    rich: bool = True,
) -> np.ndarray:
    """Draw keypoints with scale and orientation circles."""
    flags = (
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        if rich
        else cv2.DRAW_MATCHES_FLAGS_DEFAULT
    )
    if image.ndim == 2:
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        base = image.copy()

    return cv2.drawKeypoints(base, keypoints, None, color=(0, 255, 0), flags=flags)
