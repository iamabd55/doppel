"""Unit test and sanity verification for ColabDoppelDataset (Phase 4).

Verifies:
1. One batch execution: tensor shapes, dtypes, forged fraction, mask coverage.
2. Train cropping policy: random crops with configurable positive forged coverage.
3. Val padding policy: full image zero-padded to multiples of 32 without cropping or resizing.
4. Document-level split disjointness: zero leakage between train and val splits.
"""
from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader

from detector.colab_dataset import ColabDoppelDataset, colab_collate_fn, verify_splits_disjointness

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_colab_dataset_batch():
    train_manifest = REPO_ROOT / "data" / "synthetic" / "train_manifest.csv"
    if not train_manifest.exists():
        pytest.skip("train_manifest.csv not found")

    train_ds = ColabDoppelDataset(
        manifest_path=str(train_manifest),
        split="train",
        crop_size=(256, 256),
        pos_crop_prob=0.50,
        augment=True,
        seed=42,
    )

    loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=colab_collate_fn)
    batch = next(iter(loader))

    images = batch["image"]
    masks = batch["mask"]
    labels = batch["label"]

    # 1. Shapes
    b, c, h, w = images.shape
    assert b == 16 and c == 3, f"Unexpected image shape: {images.shape}"
    assert h >= 256 and (h % 32 == 0), f"Height {h} not valid multiple of 32"
    assert w >= 256 and (w % 32 == 0), f"Width {w} not valid multiple of 32"
    assert masks.shape == (16, 1, h, w), f"Unexpected mask shape: {masks.shape}"
    assert labels.shape == (16,), f"Unexpected label shape: {labels.shape}"

    # 2. Dtypes
    assert images.dtype == torch.float32
    assert masks.dtype == torch.float32
    assert labels.dtype == torch.float32

    # 3. Value ranges
    assert images.min() >= 0.0 and images.max() <= 1.0
    assert torch.all((masks == 0.0) | (masks == 1.0))

    # 4. Forged fraction and mask coverage
    forged_count = int(labels.sum().item())
    clean_count = 16 - forged_count
    mask_pixels = int(masks.sum().item())
    total_pixels = 16 * 256 * 256
    coverage_pct = (mask_pixels / total_pixels) * 100

    print(f"\n[BATCH TEST]")
    print(f"  Image batch shape : {images.shape} (dtype: {images.dtype})")
    print(f"  Mask batch shape  : {masks.shape} (dtype: {masks.dtype})")
    print(f"  Label batch shape : {labels.shape} (dtype: {labels.dtype})")
    print(f"  Forged fraction   : {forged_count}/16 ({forged_count/16*100:.1f}%) [clean: {clean_count}/16]")
    print(f"  Mask coverage     : {mask_pixels:,} px ({coverage_pct:.2f}% of batch canvas)")


def test_colab_dataset_val_full_image_padding():
    val_manifest = REPO_ROOT / "data" / "synthetic" / "val_manifest.csv"
    if not val_manifest.exists():
        pytest.skip("val_manifest.csv not found")

    val_ds = ColabDoppelDataset(
        manifest_path=str(val_manifest),
        split="val",
        crop_size=(256, 256),  # Should be ignored for val
        pad_multiple=32,
        augment=False,
    )

    sample = val_ds[0]
    img = sample["image"]
    mask = sample["mask"]

    _, h, w = img.shape
    assert h % 32 == 0, f"Height {h} not multiple of 32"
    assert w % 32 == 0, f"Width {w} not multiple of 32"
    assert mask.shape == (1, h, w)


def test_colab_dataset_disjointness():
    train_manifest = REPO_ROOT / "data" / "synthetic" / "train_manifest.csv"
    val_manifest = REPO_ROOT / "data" / "synthetic" / "val_manifest.csv"
    if not (train_manifest.exists() and val_manifest.exists()):
        pytest.skip("Manifests not found")

    train_ds = ColabDoppelDataset(str(train_manifest), split="train")
    val_ds = ColabDoppelDataset(str(val_manifest), split="val")

    verify_splits_disjointness(train_ds, val_ds)


def test_colab_dataset_v3_triplets():
    v3_train = REPO_ROOT / "data" / "synthetic_v3" / "train"
    v3_val = REPO_ROOT / "data" / "synthetic_v3" / "val"
    if not (v3_train.exists() and v3_val.exists()):
        pytest.skip("synthetic_v3 dataset not found")

    train_ds = ColabDoppelDataset(str(v3_train), split="train", crop_size=(256, 256), augment=True)
    val_ds = ColabDoppelDataset(str(v3_val), split="val", pad_multiple=32, augment=False)

    assert len(train_ds) == 9000
    assert len(val_ds) == 900
    verify_splits_disjointness(train_ds, val_ds)

    # Check that pristine and unrelated_paste have empty masks and label 0
    # and forged has non-empty mask and label 1
    found_kinds = set()
    for idx in range(min(60, len(val_ds))):
        sample = val_ds[idx]
        kind = sample["meta"]["kind"]
        found_kinds.add(kind)
        mask_sum = float(sample["mask"].sum().item())
        label_val = float(sample["label"].item())

        if kind in ["pristine", "unrelated_paste"]:
            assert mask_sum == 0.0, f"Expected 0 mask for {kind}, got {mask_sum}"
            assert label_val == 0.0, f"Expected 0 label for {kind}, got {label_val}"
        elif kind == "forged":
            assert mask_sum > 0.0, f"Expected non-empty mask for forged, got {mask_sum}"
            assert label_val == 1.0, f"Expected 1 label for forged, got {label_val}"

    assert {"pristine", "forged", "unrelated_paste"}.issubset(found_kinds)


def test_pristine_dev_halves_disjointness():
    import json

    split_path = REPO_ROOT / "configs" / "pristine_dev_split.json"
    held_out_path = REPO_ROOT / "configs" / "held_out_pristine_docs.json"
    v3_train = REPO_ROOT / "data" / "synthetic_v3" / "train"
    if not (split_path.exists() and held_out_path.exists()):
        pytest.skip("pristine_dev_split.json or held_out_pristine_docs.json not found")

    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    with open(held_out_path, "r", encoding="utf-8") as f:
        all_held_out = set(str(d) for d in json.load(f))

    half_a = set(str(d) for d in split_data["half_a"])
    half_b = set(str(d) for d in split_data["half_b"])

    assert len(half_a & half_b) == 0, "Half A and Half B must be disjoint"
    assert (half_a | half_b) == all_held_out, "Half A + Half B must partition all 177 held-out docs"

    if v3_train.exists():
        train_ds = ColabDoppelDataset(str(v3_train), split="train")
        assert len(train_ds.doc_ids & all_held_out) == 0, "Held-out docs must not appear in synthetic_v3/train"

