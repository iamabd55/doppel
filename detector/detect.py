import cv2
import numpy as np
from skimage.morphology import remove_small_objects
from detector.features import extract_features
from detector.matching import find_matches


def detect(gray, B=8, K=16, Q=4, min_std=4.0, window=5, min_dist=30,
           min_votes=50, kernel=9, min_size=500):
    # stages 1-2: fingerprints, then sort and match
    feats, pos = extract_features(gray, B=B, K=K, Q=Q, min_std=min_std)
    a, b, shift = find_matches(feats, pos, window=window, min_dist=min_dist)

    mask = np.zeros(gray.shape, np.uint8)
    if len(shift) == 0:
        return mask

    # stage 3: voting, keep pairs whose shift got enough votes
    vectors, inverse, counts = np.unique(shift, axis=0,
                                         return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    keep = counts[inverse] >= min_votes
    a, b = a[keep], b[keep]

    # stage 4: paint surviving blocks, then clean up
    for r, c in np.concatenate([a, b]):
        mask[r:r+B, c:c+B] = 255

    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kernel, kernel), np.uint8))
    clean = remove_small_objects(closed > 0, max_size=min_size-1)
    return (clean * 255).astype(np.uint8)