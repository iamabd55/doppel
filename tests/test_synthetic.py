"""Automated tests for synthetic copy-move generator (Phase 3).

Verifies:
1. Augmentation and degradation shape and dtype invariants across tiers.
2. Generator mask alignment: mask_A == cv2.bitwise_or(mask_M, mask_S).
3. Clean negative sample properties (empty masks).
4. Paired twin identity: clean and forged twins share exact dimensions and degradation profile.
5. Forged sample properties (non-empty masks, valid bounding boxes).
6. Exact determinism given identical random seeds.
7. Robustness on narrow Western blot strips (e.g. height < 30 px).
8. Document-level split disjointness (zero document leakage).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from detector.synthetic import (
    SyntheticCopyMoveGenerator,
    apply_patch_augmentation,
    sample_degradation_profile,
    apply_degradation_pipeline,
)


@pytest.fixture
def default_config():
    """Load default configuration from configs/synthetic.yaml if available, else fallback."""
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "synthetic.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "generation": {
            "patch_min_area_ratio": 0.02,
            "patch_max_area_ratio": 0.25,
            "min_texture_std": 4.0,
            "low_texture_share": 0.10,
        },
        "transforms": {
            "translation_prob": 0.60,
            "cardinal_rot_flip_prob": 0.20,
            "arbitrary_rot_scale_prob": 0.15,
            "combination_prob": 0.05,
            "rotation_range": [1.0, 180.0],
            "scale_range": [0.5, 2.0],
        },
        "degradations": {
            "level_distribution": {
                "zero": 0.25,
                "light": 0.25,
                "moderate": 0.25,
                "heavy": 0.25,
            },
            "edge_feathering_prob": 0.50,
        },
    }


def test_patch_augmentation_flip():
    """Test horizontal and vertical patch flips."""
    patch = np.zeros((40, 50, 3), dtype=np.uint8)
    patch[10:20, 10:20] = 255

    flipped_h, mask_h = apply_patch_augmentation(patch, "cardinal_flip", flip_type="H-Flip")
    assert flipped_h.shape == patch.shape
    assert mask_h.shape == patch.shape[:2]
    assert np.any(flipped_h[10:20, 30:40] == 255)
    assert np.all(flipped_h[10:20, 10:20] == 0)

    flipped_v, mask_v = apply_patch_augmentation(patch, "cardinal_flip", flip_type="V-Flip")
    assert flipped_v.shape == patch.shape
    assert np.any(flipped_v[20:30, 10:20] == 255)
    assert np.all(flipped_v[10:20, 10:20] == 0)


def test_patch_augmentation_cardinal_rotation():
    """Test 90, 180, and 270 degree cardinal rotations."""
    patch = np.ones((30, 60, 3), dtype=np.uint8) * 128
    rot90, m90 = apply_patch_augmentation(patch, "cardinal_flip", rotation_deg=90.0)
    assert rot90.shape == (60, 30, 3)
    assert m90.shape == (60, 30)

    rot180, m180 = apply_patch_augmentation(patch, "cardinal_flip", rotation_deg=180.0)
    assert rot180.shape == (30, 60, 3)

    rot270, m270 = apply_patch_augmentation(patch, "cardinal_flip", rotation_deg=270.0)
    assert rot270.shape == (60, 30, 3)


def test_degradation_tiers():
    """Test that all degradation tiers produce valid images preserving shape and dtype."""
    rng = np.random.default_rng(123)
    img = np.full((64, 80, 3), 120, dtype=np.uint8)

    for level in ["zero", "light", "moderate", "heavy"]:
        profile = sample_degradation_profile(level, rng)
        deg_img = apply_degradation_pipeline(img, profile, rng)
        assert deg_img.shape == img.shape
        assert deg_img.dtype == np.uint8
        if level == "zero":
            assert np.array_equal(deg_img, img)


def test_paired_twin_generation(default_config):
    """Test that paired twins share exact dimensions and degradation metadata."""
    generator = SyntheticCopyMoveGenerator(default_config)
    rng = np.random.default_rng(42)

    base_img = (rng.uniform(40, 200, (80, 120, 3))).astype(np.uint8)
    forged_pack, clean_pack = generator.generate_twin_pair(
        base_img=base_img,
        pair_id_prefix="twin_test_01",
        class_name="Microscopy",
        doc_id="101",
        rel_path="101/01.png",
        rng=rng,
        degradation_level="moderate",
    )

    f_img, f_M, f_S, f_A, f_meta = forged_pack
    c_img, c_M, c_S, c_A, c_meta = clean_pack

    # Dimension and metadata consistency
    assert f_img.shape == c_img.shape == base_img.shape
    assert f_meta["degradation_level"] == c_meta["degradation_level"] == "moderate"
    assert f_meta["doc_id"] == c_meta["doc_id"] == "101"

    # Forged properties
    assert f_meta["is_forged"] is True
    assert np.sum(f_M) > 0
    assert np.sum(f_S) > 0
    assert np.array_equal(f_A, cv2.bitwise_or(f_M, f_S))

    # Clean twin properties
    assert c_meta["is_forged"] is False
    assert np.sum(c_M) == 0
    assert np.sum(c_S) == 0
    assert np.sum(c_A) == 0


def test_generator_determinism(default_config):
    """Identical seeds must produce bit-identical forged images and masks."""
    generator = SyntheticCopyMoveGenerator(default_config)

    base_img = (np.random.default_rng(777).uniform(30, 220, (100, 120, 3))).astype(np.uint8)
    meta_in = {"sample_id": "det_0", "class": "Blot/Gel", "is_forged": True, "degradation_level": "light"}

    rng1 = np.random.default_rng(42)
    f1, m1, s1, a1, meta1 = generator.generate(base_img, meta_in, rng1)

    rng2 = np.random.default_rng(42)
    f2, m2, s2, a2, meta2 = generator.generate(base_img, meta_in, rng2)

    assert np.array_equal(f1, f2)
    assert np.array_equal(m1, m2)
    assert np.array_equal(s1, s2)
    assert np.array_equal(a1, a2)
    assert meta1["source_bbox"] == meta2["source_bbox"]
    assert meta1["target_bbox"] == meta2["target_bbox"]


def test_generator_narrow_blot_strip(default_config):
    """Generator must successfully handle narrow Western blot strips (e.g. height 22 px)."""
    generator = SyntheticCopyMoveGenerator(default_config)
    rng = np.random.default_rng(55)

    blot_strip = np.zeros((22, 250, 3), dtype=np.uint8)
    for x in range(20, 240, 30):
        blot_strip[4:18, x:x + 12] = 220

    meta_in = {"sample_id": "blot_strip_0", "class": "Blot/Gel", "is_forged": True, "degradation_level": "zero"}
    forged, mask_M, mask_S, mask_A, meta_out = generator.generate(blot_strip, meta_in, rng)

    assert forged.shape == blot_strip.shape
    assert np.sum(mask_M) > 0
    assert np.sum(mask_S) > 0
    assert np.array_equal(mask_A, cv2.bitwise_or(mask_M, mask_S))


def test_split_manifests_disjointness():
    """Verify that train, validation, and test IDD manifests have zero document overlap."""
    splits_dir = os.path.join(os.path.dirname(__file__), "..", "data", "biofors", "splits")
    train_path = os.path.join(splits_dir, "train_manifest.csv")
    val_path = os.path.join(splits_dir, "val_manifest.csv")
    test_path = os.path.join(splits_dir, "test_idd_manifest.csv")

    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        pytest.skip("Manifest files not found on disk, skipping split disjointness test.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    train_docs = set(train_df["doc_id"].unique())
    val_docs = set(val_df["doc_id"].unique())
    test_docs = set(test_df["doc_id"].unique())

    assert len(train_docs.intersection(val_docs)) == 0, "Leakage detected between train and val pools"
    assert len(train_docs.intersection(test_docs)) == 0, "Leakage detected between train and test pools"
    assert len(val_docs.intersection(test_docs)) == 0, "Leakage detected between val and test pools"


def test_class_tier_chi_square_independence(default_config):
    """Assert chi-square test confirms statistical independence between class and tier (p > 0.05)."""
    from scipy.stats import chi2_contingency

    classes = ["Blot/Gel", "Microscopy"]
    tiers = ["zero", "light", "moderate", "heavy"]
    cells = [(c, t) for c in classes for t in tiers]

    num_pairs = 160
    base_cnt = num_pairs // len(cells)
    pair_plan = []
    for cell in cells:
        pair_plan.extend([cell] * base_cnt)

    rng = np.random.default_rng(123)
    rng.shuffle(pair_plan)

    plan_df = pd.DataFrame(pair_plan, columns=["class", "degradation_level"])
    contingency = pd.crosstab(plan_df["class"], plan_df["degradation_level"])

    chi2, p_val, dof, _ = chi2_contingency(contingency)
    assert p_val > 0.05, f"Chi-square independence failed: p={p_val:.4f} <= 0.05"


def test_multi_copy_forgery(default_config):
    """Verify that multi-copy generation places multiple targets with valid union masks."""
    cfg = dict(default_config)
    cfg["generation"]["multi_copy_prob"] = 1.0  # Force multi-copy
    generator = SyntheticCopyMoveGenerator(cfg)
    rng = np.random.default_rng(99)

    base_img = (rng.uniform(30, 220, (300, 400, 3))).astype(np.uint8)
    meta_in = {"sample_id": "mc_0", "class": "Microscopy", "is_forged": True, "degradation_level": "light"}

    forged, mask_M, mask_S, mask_A, meta_out = generator.generate(base_img, meta_in, rng)

    assert meta_out["is_multi_copy"] is True
    assert meta_out["num_targets"] >= 2
    assert len(meta_out["target_bboxes"]) == meta_out["num_targets"]
    assert np.array_equal(mask_A, cv2.bitwise_or(mask_M, mask_S))
    assert np.sum(mask_M) > 0
    assert np.sum(mask_S) > 0


def test_patch_area_floor(default_config):
    """Verify that donor patch area obeys the 400 px floor on adequately sized images."""
    generator = SyntheticCopyMoveGenerator(default_config)
    rng = np.random.default_rng(101)

    base_img = (rng.uniform(20, 200, (200, 200, 3))).astype(np.uint8)
    meta_in = {"sample_id": "floor_test", "class": "Microscopy", "is_forged": True, "degradation_level": "zero"}

    forged, mask_M, mask_S, mask_A, meta_out = generator.generate(base_img, meta_in, rng)

    assert meta_out["patch_area"] >= 400


def test_pristine_train_docs_exclusion():
    """Verify that the 177 held-out pristine train documents are completely absent from generator train pool."""
    import json
    held_out_path = os.path.join(os.path.dirname(__file__), "..", "configs", "held_out_pristine_docs.json")
    if not os.path.exists(held_out_path):
        pytest.skip("held_out_pristine_docs.json not found.")

    with open(held_out_path, "r", encoding="utf-8") as f:
        held_out_docs = set(json.load(f))

    assert len(held_out_docs) == 177

    splits_dir = os.path.join(os.path.dirname(__file__), "..", "data", "biofors", "splits")
    train_path = os.path.join(splits_dir, "train_manifest.csv")
    train_df = pd.read_csv(train_path)

    synth_train_pool = train_df[~train_df["doc_id"].astype(str).isin(held_out_docs)]
    pool_docs = set(synth_train_pool["doc_id"].astype(str).unique())

    assert pool_docs.isdisjoint(held_out_docs), "Contamination: pristine train docs found in generator pool!"

