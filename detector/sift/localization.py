"""Mask localization and end-to-end SIFT copy-move detection pipeline."""
from typing import List, Tuple
import cv2
import numpy as np

from detector.sift.features import detect_and_compute
from detector.sift.matching import match_within_image, match_flipped
from detector.sift.verification import verify_matches_ransac


def create_sift_mask(
    shape: Tuple[int, int],
    pts1: np.ndarray,
    pts2: np.ndarray,
    radius: int = 20,
    use_convex_hull: bool = True,
    kernel_size: int = 9,
) -> np.ndarray:
    """Turn verified inlier coordinates into a binary forgery mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    if pts1 is None or len(pts1) == 0:
        return mask

    all_points = np.vstack([pts1, pts2])
    for pt in all_points:
        center = (int(round(pt[0])), int(round(pt[1])))
        cv2.circle(mask, center, radius, 255, -1)

    if use_convex_hull:
        for pts in [pts1, pts2]:
            if len(pts) >= 3:
                int_pts = np.round(pts).astype(np.int32)
                hull = cv2.convexHull(int_pts)
                cv2.fillConvexPoly(mask, hull, 255)

    if kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def detect_sift(
    gray: np.ndarray,
    ratio_threshold: float = 0.6,
    min_dist: float = 30.0,
    ransac_threshold: float = 3.0,
    min_inliers: int = 4,
    cluster_threshold: float = 50.0,
    mask_radius: int = 20,
    use_convex_hull: bool = True,
    check_flip: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    """Complete SIFT copy-move detection pipeline returning a binary mask."""
    h, w = gray.shape

    # 1. Standard within-image pass (translation, rotation, scaling)
    kps, descs = detect_and_compute(gray)
    pts1_std, pts2_std, _ = match_within_image(
        kps, descs, ratio_threshold=ratio_threshold, min_dist=min_dist
    )
    inl1_std, inl2_std, tf_std = verify_matches_ransac(
        pts1_std,
        pts2_std,
        ransac_threshold=ransac_threshold,
        min_inliers=min_inliers,
        cluster_threshold=cluster_threshold,
    )

    all_inl1 = [inl1_std] if len(inl1_std) > 0 else []
    all_inl2 = [inl2_std] if len(inl2_std) > 0 else []
    all_transforms = list(tf_std)

    # 2. Horizontal flip pass (mirrored / reflected clones)
    if check_flip:
        gray_flip = cv2.flip(gray, 1)
        kps_flip, descs_flip = detect_and_compute(gray_flip)
        pts1_flip, pts2_flip, _ = match_flipped(
            kps,
            descs,
            kps_flip,
            descs_flip,
            width=w,
            ratio_threshold=ratio_threshold,
            min_dist=min_dist,
        )
        inl1_flip, inl2_flip, tf_flip = verify_matches_ransac(
            pts1_flip,
            pts2_flip,
            ransac_threshold=ransac_threshold,
            min_inliers=min_inliers,
            cluster_threshold=cluster_threshold,
        )
        if len(inl1_flip) > 0:
            all_inl1.append(inl1_flip)
            all_inl2.append(inl2_flip)
            all_transforms.extend(tf_flip)

    if len(all_inl1) > 0:
        combined_inl1 = np.vstack(all_inl1)
        combined_inl2 = np.vstack(all_inl2)
    else:
        empty = np.empty((0, 2), dtype=np.float32)
        combined_inl1, combined_inl2 = empty, empty

    mask = create_sift_mask(
        gray.shape,
        combined_inl1,
        combined_inl2,
        radius=mask_radius,
        use_convex_hull=use_convex_hull,
    )
    return mask, combined_inl1, combined_inl2, all_transforms
