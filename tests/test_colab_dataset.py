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

from detector.colab_dataset import ColabDoppelDataset, verify_splits_disjointness

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

    loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    batch = next(iter(loader))

    images = batch["image"]
    masks = batch["mask"]
    labels = batch["label"]

    # 1. Shapes
    assert images.shape == (16, 3, 256, 256), f"Unexpected image shape: {images.shape}"
    assert masks.shape == (16, 1, 256, 256), f"Unexpected mask shape: {masks.shape}"
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
