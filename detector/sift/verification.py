"""Geometric verification using RANSAC affine transform estimation and spatial clustering.

References:
    Fischler, M. A., & Bolles, R. C. (1981). Random sample consensus (RANSAC). CACM.
    Amerini, I., et al. (2011). A SIFT-based forensic method for copy-move attack
    detection and transformation recovery. IEEE TIFS.
"""
from typing import List, Tuple
import cv2
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage


def verify_matches_ransac(
    pts1: np.ndarray,
    pts2: np.ndarray,
    ransac_threshold: float = 3.0,
    min_inliers: int = 4,
    cluster_threshold: float = 100.0,
    min_inlier_ratio: float = 0.0,
    return_info: bool = False,
) -> Tuple:
    """Filter candidate matches using hierarchical clustering and RANSAC affine estimation.

    Parameters:
        pts1: (M, 2) float32 array of source coordinates.
        pts2: (M, 2) float32 array of target coordinates.
        ransac_threshold: Maximum reprojection error in pixels.
        min_inliers: Minimum number of inliers required to accept a transform.
        cluster_threshold: Spatial distance threshold for agglomerative clustering.
        min_inlier_ratio: Minimum ratio of inliers / candidate matches per cluster.
        return_info: If True, returns an additional metadata dictionary.

    Returns:
        Tuple of (verified_pts1, verified_pts2, transforms) or
        (verified_pts1, verified_pts2, transforms, info) if return_info is True.
    """
    empty = np.empty((0, 2), dtype=np.float32)
    info = {"candidates": 0, "inliers": 0, "inlier_ratio": 0.0}

    if pts1 is None or len(pts1) < min_inliers:
        if return_info:
            return empty, empty, [], info
        return empty, empty, []

    all_pts = np.vstack([pts1, pts2])
    n_pairs = len(pts1)
    info["candidates"] = n_pairs

    # 1. Spatial clustering of keypoints
    try:
        Z = linkage(all_pts, method="ward")
        labels = fcluster(Z, t=cluster_threshold, criterion="distance")
        labels1 = labels[:n_pairs]
        labels2 = labels[n_pairs:]
    except Exception:
        labels1 = np.zeros(n_pairs, dtype=int)
        labels2 = np.ones(n_pairs, dtype=int)

    # 2. Group matches by cluster pairs (c_src, c_dst) with consistent direction
    cluster_groups = {}
    for idx in range(n_pairs):
        c1 = labels1[idx]
        c2 = labels2[idx]
        if c1 == c2:
            continue

        pair_key = (min(c1, c2), max(c1, c2))
        if pair_key not in cluster_groups:
            cluster_groups[pair_key] = {"src": [], "dst": []}

        if c1 < c2:
            cluster_groups[pair_key]["src"].append(pts1[idx])
            cluster_groups[pair_key]["dst"].append(pts2[idx])
        else:
            cluster_groups[pair_key]["src"].append(pts2[idx])
            cluster_groups[pair_key]["dst"].append(pts1[idx])

    verified_src = []
    verified_dst = []
    transforms = []

    # 3. Fit an affine transform using RANSAC for each cluster pair
    for pair_key, group in cluster_groups.items():
        n_cluster_candidates = len(group["src"])
        if n_cluster_candidates < min_inliers:
            continue

        p_src = np.array(group["src"], dtype=np.float32)
        p_dst = np.array(group["dst"], dtype=np.float32)

        M, inlier_mask = cv2.estimateAffine2D(
            p_src,
            p_dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            maxIters=2000,
            confidence=0.99,
        )

        if M is not None and inlier_mask is not None:
            mask = inlier_mask.ravel().astype(bool)
            n_inl = int(mask.sum())
            cluster_ratio = n_inl / max(1, n_cluster_candidates)

            if n_inl >= min_inliers and cluster_ratio >= min_inlier_ratio:
                verified_src.append(p_src[mask])
                verified_dst.append(p_dst[mask])
                transforms.append(M)

    # 4. Compare with global direct RANSAC
    if n_pairs >= min_inliers:
        M_all, inlier_all = cv2.estimateAffine2D(
            pts1,
            pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            maxIters=2000,
            confidence=0.99,
        )
        if M_all is not None and inlier_all is not None:
            mask_all = inlier_all.ravel().astype(bool)
            direct_count = int(mask_all.sum())
            direct_ratio = direct_count / max(1, n_pairs)

            if direct_count >= min_inliers and direct_ratio >= min_inlier_ratio:
                cluster_count = sum(len(s) for s in verified_src)
                if direct_count > cluster_count:
                    verified_src = [pts1[mask_all]]
                    verified_dst = [pts2[mask_all]]
                    transforms = [M_all]

    if len(verified_src) == 0:
        if return_info:
            return empty, empty, [], info
        return empty, empty, []

    res_src = np.vstack(verified_src)
    res_dst = np.vstack(verified_dst)
    total_inliers = len(res_src)
    info["inliers"] = total_inliers
    info["inlier_ratio"] = round(total_inliers / max(1, n_pairs), 3)

    if return_info:
        return res_src, res_dst, transforms, info
    return res_src, res_dst, transforms
