"""Unit tests for CNN inference pipeline and standardized prediction interface."""

from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

from detector.cnn import (
    load_cnn_model,
    load_cnn_thresholds,
    predict,
    predict_cnn,
)


def test_load_cnn_thresholds_defaults():
    """Verify threshold lookup for Blot/Gel and Microscopy across budgets."""
    # Budget 1.0% (primary)
    blot_01 = load_cnn_thresholds("configs/cnn_threshold.yaml", image_class="Blot/Gel", budget=0.01)
    assert blot_01["image_threshold"] == 0.95
    assert round(blot_01["pixel_threshold"], 2) == 0.30
    assert blot_01["min_area_pixels"] == 25

    mic_01 = load_cnn_thresholds("configs/cnn_threshold.yaml", image_class="Microscopy", budget=0.01)
    assert mic_01["image_threshold"] == 0.99
    assert round(mic_01["pixel_threshold"], 2) == 0.30
    assert mic_01["min_area_pixels"] == 1600

    # Budget 2.0%
    blot_02 = load_cnn_thresholds("configs/cnn_threshold.yaml", image_class="Blot/Gel", budget=0.02)
    assert blot_02["image_threshold"] == 0.50
    assert round(blot_02["pixel_threshold"], 2) == 0.70
    assert blot_02["min_area_pixels"] == 150

    mic_02 = load_cnn_thresholds("configs/cnn_threshold.yaml", image_class="Microscopy", budget=0.02)
    assert mic_02["image_threshold"] == 0.99
    assert round(mic_02["pixel_threshold"], 2) == 0.70
    assert mic_02["min_area_pixels"] == 200


def test_load_cnn_thresholds_fallback():
    """Non-existent config file returns safe fallback thresholds without crashing."""
    res = load_cnn_thresholds("configs/non_existent.yaml", image_class="Blot/Gel", budget=0.01)
    assert "image_threshold" in res
    assert "pixel_threshold" in res
    assert "min_area_pixels" in res


def test_load_cnn_model_caching():
    """Verify model loading and memory caching."""
    ckpt_path = Path("checkpoints/corr_v2/last.pt")
    if not ckpt_path.exists():
        pytest.skip("Checkpoint last.pt not available for model load test.")

    model1 = load_cnn_model(checkpoint_path=ckpt_path, device="cpu")
    model2 = load_cnn_model(checkpoint_path=ckpt_path, device="cpu")
    assert model1 is model2
    assert not model1.training


class MockCNNModel(nn.Module):
    """Mock CNN model for deterministic unit tests."""

    def __init__(self, high_image_score: bool = True, put_blob: bool = True, pad_leak: bool = False):
        super().__init__()
        self.high_image_score = high_image_score
        self.put_blob = put_blob
        self.pad_leak = pad_leak

    def forward(self, x, valid_mask=None, ablate_corr=False):
        batch, _, h, w = x.shape
        logits = torch.full((batch, 1, h, w), -10.0, dtype=torch.float32)

        if self.put_blob:
            # Place a 10x10 patch in the center of the valid region
            logits[:, :, 50:60, 50:60] = 10.0

        if self.pad_leak:
            # Place a patch only at the bottom border where padding would be
            logits[:, :, -5:, -5:] = 10.0

        img_logit_val = 5.0 if self.high_image_score else -5.0
        img_logit = torch.tensor([img_logit_val], dtype=torch.float32)
        return logits, img_logit


def test_predict_cnn_positive_detection():
    """Verify that positive detection returns 255 mask, score, and region boxes."""
    model = MockCNNModel(high_image_score=True, put_blob=True)
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    mask, score, regions = predict_cnn(
        img,
        image_class="Blot/Gel",
        budget=0.01,
        model=model,
        device="cpu",
    )

    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})
    assert np.sum(mask == 255) == 100  # 10x10 blob = 100 px
    assert score > 0.90
    assert len(regions) == 1
    rx, ry, rw, rh = regions[0]
    assert (rx, ry, rw, rh) == (36, 36, 10, 10)


def test_predict_cnn_gatekeeper_suppression():
    """When image head is negative, any blob is gatekeeper-suppressed to 0."""
    model = MockCNNModel(high_image_score=False, put_blob=True)
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    mask, score, regions = predict_cnn(
        img,
        image_class="Blot/Gel",
        budget=0.01,
        model=model,
        device="cpu",
    )

    assert np.sum(mask) == 0
    assert len(regions) == 0
    assert score < 0.10


def test_predict_cnn_arbitrary_dimensions_and_padding():
    """Verify arbitrary non-multiple-of-32 image shapes are cropped back exactly."""
    model = MockCNNModel(high_image_score=True, put_blob=False, pad_leak=True)
    # Image shape not divisible by 32: 135 x 217
    img = np.zeros((135, 217, 3), dtype=np.uint8)

    mask, score, regions = predict_cnn(
        img,
        image_class="Blot/Gel",
        budget=0.01,
        model=model,
        device="cpu",
        pad_multiple=32,
    )

    assert mask.shape == (135, 217)
    # The pad leak occurred in the padded strip, so cropped mask must be completely clean
    assert np.sum(mask) == 0
    assert len(regions) == 0


def test_predict_alias_and_return_dict():
    """Verify convenience alias and return_dict option."""
    model = MockCNNModel(high_image_score=True, put_blob=True)
    img = np.zeros((64, 64), dtype=np.uint8)  # 2D grayscale input

    # predict alias
    mask1, score1, reg1 = predict(img, model=model, device="cpu")
    assert mask1.shape == (64, 64)

    # return_dict
    res = predict_cnn(img, model=model, device="cpu", return_dict=True)
    assert isinstance(res, dict)
    assert "mask" in res
    assert "image_score" in res
    assert "regions" in res
    assert "is_detected" in res
    assert "prob_map" in res
    assert res["prob_map"].shape == (64, 64)
