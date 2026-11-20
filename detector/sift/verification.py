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
    cluster_threshold: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Filter candidate matches using hierarchical clustering and RANSAC affine estimation."""
    empty = np.empty((0, 2), dtype=np.float32)
    if pts1 is None or len(pts1) < min_inliers:
        return empty, empty, []

    # 1. Cluster all matched keypoints spatially into distinct object regions
    all_pts = np.vstack([pts1, pts2])
    n_pairs = len(pts1)

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
        if len(group["src"]) < min_inliers:
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
            if mask.sum() >= min_inliers:
                verified_src.append(p_src[mask])
                verified_dst.append(p_dst[mask])
                transforms.append(M)

    # 4. Fallback: if clustering did not isolate clusters but enough matches exist
    if len(verified_src) == 0 and n_pairs >= min_inliers:
        M, inlier_mask = cv2.estimateAffine2D(
            pts1,
            pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            maxIters=2000,
            confidence=0.99,
        )
        if M is not None and inlier_mask is not None:
            mask = inlier_mask.ravel().astype(bool)
            if mask.sum() >= min_inliers:
                verified_src.append(pts1[mask])
                verified_dst.append(pts2[mask])
                transforms.append(M)

    if len(verified_src) == 0:
        return empty, empty, []

    return (
        np.vstack(verified_src),
        np.vstack(verified_dst),
        transforms,
    )
