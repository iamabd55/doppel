"""SIFT copy-move forgery detection package.

Reference:
    Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. IJCV.
    Amerini, I., et al. (2011). A SIFT-based forensic method for copy-move attack detection. IEEE TIFS.
"""
from detector.sift.features import (
    extract_keypoints,
    extract_descriptors,
    detect_and_compute,
    draw_keypoints,
)
from detector.sift.matching import (
    match_within_image,
    map_flip_coordinates,
    match_flipped,
    draw_matches,
)
from detector.sift.verification import (
    verify_matches_ransac,
)
from detector.sift.localization import (
    create_sift_mask,
    detect_sift,
)

__all__ = [
    "extract_keypoints",
    "extract_descriptors",
    "detect_and_compute",
    "draw_keypoints",
    "match_within_image",
    "map_flip_coordinates",
    "match_flipped",
    "draw_matches",
    "verify_matches_ransac",
    "create_sift_mask",
    "detect_sift",
]
