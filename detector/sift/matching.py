"""SIFT descriptor matching for within-image and horizontal flip passes.

Reference:
    Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. IJCV.
"""
from typing import List, Tuple
import cv2
import numpy as np


def match_within_image(
    keypoints: List[cv2.KeyPoint],
    descriptors: np.ndarray,
    ratio_threshold: float = 0.6,
    min_dist: float = 30.0,
    return_angles: bool = False,
) -> Tuple:
    """Match descriptors within the same image using kNN (k=3) and Lowe's ratio test."""
    empty = np.empty((0, 2), dtype=np.float32)
    empty_ang = np.empty((0,), dtype=np.float32)
    if descriptors is None or len(descriptors) < 3:
        if return_angles:
            return empty, empty, [], empty_ang, empty_ang
        return empty, empty, []

    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = bf.knnMatch(descriptors, descriptors, k=3)

    pts1_list = []
    pts2_list = []
    ang1_list = []
    ang2_list = []
    ratios_list = []
    seen_pairs = set()

    for i, matches in enumerate(knn_matches):
        non_self = [m for m in matches if m.trainIdx != i]
        if len(non_self) < 2:
            continue

        best = non_self[0]
        runner_up = non_self[1]

        if runner_up.distance <= 0:
            continue
        ratio = best.distance / runner_up.distance
        if ratio >= ratio_threshold:
            continue

        j = best.trainIdx
        pt1 = keypoints[i].pt
        pt2 = keypoints[j].pt

        spatial_dist = np.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])
        if spatial_dist < min_dist:
            continue

        pair_key = (min(i, j), max(i, j))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        pts1_list.append(pt1)
        pts2_list.append(pt2)
        ang1_list.append(keypoints[i].angle)
        ang2_list.append(keypoints[j].angle)
        ratios_list.append(float(ratio))

    if len(pts1_list) == 0:
        if return_angles:
            return empty, empty, [], empty_ang, empty_ang
        return empty, empty, []

    pts1 = np.array(pts1_list, dtype=np.float32)
    pts2 = np.array(pts2_list, dtype=np.float32)
    if return_angles:
        return pts1, pts2, ratios_list, np.array(ang1_list, dtype=np.float32), np.array(ang2_list, dtype=np.float32)
    return pts1, pts2, ratios_list


def map_flip_coordinates(pts: np.ndarray, width: int) -> np.ndarray:
    """Map coordinates from a horizontally flipped image back to original image space."""
    if pts is None or len(pts) == 0:
        return np.empty((0, 2), dtype=np.float32)

    mapped = pts.copy().astype(np.float32)
    mapped[:, 0] = float(width - 1) - mapped[:, 0]
    return mapped


def match_flipped(
    kps_orig: List[cv2.KeyPoint],
    descs_orig: np.ndarray,
    kps_flip: List[cv2.KeyPoint],
    descs_flip: np.ndarray,
    width: int,
    ratio_threshold: float = 0.6,
    min_dist: float = 30.0,
    return_angles: bool = False,
) -> Tuple:
    """Match original image descriptors against horizontally flipped descriptors."""
    empty = np.empty((0, 2), dtype=np.float32)
    empty_ang = np.empty((0,), dtype=np.float32)
    if (
        descs_orig is None
        or len(descs_orig) < 2
        or descs_flip is None
        or len(descs_flip) < 2
    ):
        if return_angles:
            return empty, empty, [], empty_ang, empty_ang
        return empty, empty, []

    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = bf.knnMatch(descs_orig, descs_flip, k=2)

    pts1_list = []
    pts2_list = []
    ang1_list = []
    ang2_list = []
    ratios_list = []

    for matches in knn_matches:
        if len(matches) < 2:
            continue
        m1, m2 = matches[0], matches[1]
        if m2.distance <= 0:
            continue

        ratio = m1.distance / m2.distance
        if ratio >= ratio_threshold:
            continue

        pt_orig = kps_orig[m1.queryIdx].pt
        pt_flip = kps_flip[m1.trainIdx].pt

        pt_back_x = float(width - 1) - pt_flip[0]
        pt_back_y = pt_flip[1]
        pt_back = (pt_back_x, pt_back_y)

        dist = np.hypot(pt_orig[0] - pt_back[0], pt_orig[1] - pt_back[1])
        if dist < min_dist:
            continue

        pts1_list.append(pt_orig)
        pts2_list.append(pt_back)
        ang1_list.append(kps_orig[m1.queryIdx].angle)
        # Flip inverts horizontal direction: angle theta becomes (180 - theta) mod 360
        ang2_list.append((180.0 - kps_flip[m1.trainIdx].angle) % 360.0)
        ratios_list.append(float(ratio))

    if len(pts1_list) == 0:
        if return_angles:
            return empty, empty, [], empty_ang, empty_ang
        return empty, empty, []

    pts1 = np.array(pts1_list, dtype=np.float32)
    pts2 = np.array(pts2_list, dtype=np.float32)
    if return_angles:
        return (
            pts1,
            pts2,
            ratios_list,
            np.array(ang1_list, dtype=np.float32),
            np.array(ang2_list, dtype=np.float32),
        )
    return pts1, pts2, ratios_list


def filter_by_orientation(
    pts1: np.ndarray,
    pts2: np.ndarray,
    ang1: np.ndarray,
    ang2: np.ndarray,
    tolerance: float = 20.0,
    min_support: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter candidate matches by requiring consistent orientation difference.

    True copy-move regions undergo a shared rigid or affine rotation. Matches
    whose relative angles diverge from peer consensus are discarded.

    Parameters:
        pts1: (N, 2) array of source coordinates.
        pts2: (N, 2) array of target coordinates.
        ang1: (N,) array of source keypoint angles in degrees.
        ang2: (N,) array of target keypoint angles in degrees.
        tolerance: Angular difference threshold in degrees.
        min_support: Minimum cluster size required to keep a match group.

    Returns:
        Tuple of (filtered_pts1, filtered_pts2).
    """
    empty = np.empty((0, 2), dtype=np.float32)
    if pts1 is None or len(pts1) == 0 or len(pts1) < min_support:
        return empty, empty

    delta = (ang2 - ang1) % 360.0
    d1 = delta[:, None]
    d2 = delta[None, :]
    diff = np.abs(d1 - d2) % 360.0
    circ_dist = np.minimum(diff, 360.0 - diff)
    support = np.sum(circ_dist <= tolerance, axis=1)

    keep = support >= min_support
    if not np.any(keep):
        return empty, empty
    return pts1[keep], pts2[keep]


def draw_matches(
    image: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    thickness: int = 1,
) -> np.ndarray:
    """Draw match lines connecting cloned keypoints on a single image."""
    if image.ndim == 2:
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        out = image.copy()

    palette = [
        (0, 255, 255),
        (255, 0, 255),
        (0, 255, 0),
        (255, 255, 0),
        (0, 165, 255),
        (200, 100, 255),
    ]

    for idx, (p1, p2) in enumerate(zip(pts1, pts2)):
        c = palette[idx % len(palette)]
        pt1 = (int(round(p1[0])), int(round(p1[1])))
        pt2 = (int(round(p2[0])), int(round(p2[1])))
        cv2.line(out, pt1, pt2, c, thickness, cv2.LINE_AA)
        cv2.circle(out, pt1, 3, c, -1, cv2.LINE_AA)
        cv2.circle(out, pt2, 3, c, -1, cv2.LINE_AA)

    return out
