"""Check the five fusion rules used by the natural-image API."""

from itertools import product

import numpy as np
import pytest

from detector.natural import Detection, fuse


@pytest.mark.parametrize("votes", list(product((False, True), repeat=3)))
def test_natural_fusion_votes_and_masks(votes):
    patterns = np.array([
        [[1, 1, 0], [0, 0, 0]],
        [[0, 1, 1], [0, 0, 0]],
        [[1, 0, 1], [0, 1, 0]],
    ], dtype=bool)
    dct, sift, doagan = votes
    result = fuse(*(Detection(flagged, mask) for flagged, mask in zip(votes, patterns, strict=True)))
    dct_mask, sift_mask, doagan_mask = (mask & flagged for flagged, mask in zip(votes, patterns, strict=True))
    majority = sum(votes) >= 2
    classical_union = dct_mask | sift_mask
    union = classical_union | doagan_mask
    expected = {
        "F1": (dct or sift, classical_union),
        "F2": (dct and sift, dct_mask & sift_mask),
        "F3": (any(votes), union),
        "F4": (majority, union & majority),
        "F5": ((dct or sift) and doagan, classical_union & doagan_mask),
    }
    assert result.keys() == expected.keys()
    for rule, (flagged, mask) in expected.items():
        assert result[rule].flagged is flagged
        np.testing.assert_array_equal(result[rule].mask, mask)
