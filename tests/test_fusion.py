"""Unit tests for multi-detector fusion module (detector/fusion.py)."""

import numpy as np
import pytest

from detector.fusion import fuse, FusionResult, extract_regions_from_mask


def test_fuse_high_all_three():
    """High tier when all three detectors flag."""
    res = fuse(dct_out=True, sift_out=True, cnn_out=True)
    assert isinstance(res, FusionResult)
    assert res.tier == "high"
    assert res.detectors_fired == ["adapted_dct", "frozen_sift", "cnn"]


def test_fuse_high_two_detectors():
    """High tier when any 2 of the 3 detectors flag."""
    # Adapted DCT and SIFT
    r1 = fuse(dct_out=True, sift_out=True, cnn_out=False)
    assert r1.tier == "high"
    assert r1.detectors_fired == ["adapted_dct", "frozen_sift"]

    # Adapted DCT and CNN
    r2 = fuse(dct_out=True, sift_out=False, cnn_out=True)
    assert r2.tier == "high"
    assert r2.detectors_fired == ["adapted_dct", "cnn"]

    # SIFT and CNN
    r3 = fuse(dct_out=False, sift_out=True, cnn_out=True)
    assert r3.tier == "high"
    assert r3.detectors_fired == ["frozen_sift", "cnn"]


def test_fuse_medium_cnn_only():
    """Medium tier when CNN flags and DCT/SIFT are silent."""
    res = fuse(dct_out=False, sift_out=False, cnn_out=True)
    assert res.tier == "medium"
    assert res.detectors_fired == ["cnn"]


def test_fuse_low_single_classical():
    """Low tier when exactly one of DCT or SIFT flags and CNN is silent."""
    # DCT only
    r_dct = fuse(dct_out=True, sift_out=False, cnn_out=False)
    assert r_dct.tier == "low"
    assert r_dct.detectors_fired == ["adapted_dct"]

    # SIFT only
    r_sift = fuse(dct_out=False, sift_out=True, cnn_out=False)
    assert r_sift.tier == "low"
    assert r_sift.detectors_fired == ["frozen_sift"]


def test_fuse_none_no_detections():
    """None tier when no detector flags."""
    res = fuse(dct_out=False, sift_out=False, cnn_out=False)
    assert res.tier == "none"
    assert res.detectors_fired == []


def test_fuse_unpacking():
    """Verify namedtuple unpacking into tier, detectors_fired, regions."""
    tier, fired, regions = fuse(dct_out=True, sift_out=True, cnn_out=False)
    assert tier == "high"
    assert fired == ["adapted_dct", "frozen_sift"]
    assert isinstance(regions, dict)
    assert "adapted_dct" in regions
    assert "frozen_sift" in regions
    assert "cnn" in regions


def test_fuse_with_complex_outputs():
    """Verify fuse handles masks, dicts, and tuples properly."""
    # Synthetic 50x50 mask with 10x10 square at (15, 15)
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[15:25, 15:25] = 255

    dct_out = mask
    sift_out = {"flagged": False, "mask": None, "regions": []}
    cnn_out = (mask, 0.98, [(15, 15, 10, 10)])

    res = fuse(dct_out=dct_out, sift_out=sift_out, cnn_out=cnn_out)
    assert res.tier == "high"
    assert res.detectors_fired == ["adapted_dct", "cnn"]

    # Regions check
    assert len(res.regions_per_detector["adapted_dct"]) == 1
    assert res.regions_per_detector["adapted_dct"][0] == (15, 15, 10, 10)
    assert len(res.regions_per_detector["cnn"]) == 1
    assert res.regions_per_detector["cnn"][0] == (15, 15, 10, 10)
    assert len(res.regions_per_detector["frozen_sift"]) == 0


def test_extract_regions_from_mask():
    """Verify connected component region bounding boxes."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 20:35] = 255
    mask[50:80, 60:90] = 255

    regions = extract_regions_from_mask(mask)
    assert len(regions) == 2
    # Check bounding box components
    assert (20, 10, 15, 10) in regions
    assert (60, 50, 30, 30) in regions

    # Empty mask
    empty = extract_regions_from_mask(np.zeros((50, 50), dtype=np.uint8))
    assert empty == []

    # None mask
    assert extract_regions_from_mask(None) == []
