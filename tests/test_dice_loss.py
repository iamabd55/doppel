"""Unit tests for soft Dice loss and clean twin masking."""

import pytest
import torch
from detector.cnn.loss import CombinedBCEDiceLoss, soft_dice_loss


def test_mixed_batch_perfect_predictions():
    """(a) Mixed batch with perfect predictions gives Dice loss near 0."""
    b, h, w = 4, 32, 32
    # Two forged samples (indices 0 and 2), two clean samples (indices 1 and 3)
    targets = torch.zeros(b, 1, h, w, dtype=torch.float32)
    targets[0, 0, 8:24, 8:24] = 1.0
    targets[2, 0, 4:16, 4:16] = 1.0

    # Confident, perfect predictions: +20 for positive pixels, -20 for negative pixels
    logits = torch.full((b, 1, h, w), -20.0, dtype=torch.float32)
    logits[0, 0, 8:24, 8:24] = 20.0
    logits[2, 0, 4:16, 4:16] = 20.0

    dice = soft_dice_loss(logits, targets, smooth=1.0)
    assert dice.item() < 1e-3, f"Expected near-zero Dice loss, got {dice.item()}"


def test_mixed_batch_all_zero_predictions():
    """(b) Same batch with all-zero predictions gives high Dice loss (> 0.95)."""
    b, h, w = 4, 32, 32
    targets = torch.zeros(b, 1, h, w, dtype=torch.float32)
    targets[0, 0, 8:24, 8:24] = 1.0
    targets[2, 0, 4:16, 4:16] = 1.0

    # All-zero predictions across all samples
    logits = torch.full((b, 1, h, w), -20.0, dtype=torch.float32)

    dice = soft_dice_loss(logits, targets, smooth=1.0)
    assert dice.item() > 0.95, f"Expected high Dice loss (>0.95), got {dice.item()}"


def test_all_clean_batch_zero_dice():
    """(c) Batch of only clean samples gives Dice term exactly 0."""
    b, h, w = 4, 32, 32
    targets = torch.zeros(b, 1, h, w, dtype=torch.float32)
    logits = torch.randn(b, 1, h, w, dtype=torch.float32)

    dice = soft_dice_loss(logits, targets, smooth=1.0)
    assert dice.item() == 0.0, f"Expected Dice term to be 0.0, got {dice.item()}"


def test_clean_samples_never_add_gradient_to_dice():
    """Verify clean samples never add gradient to the Dice term in mixed or clean batches."""
    b, h, w = 4, 32, 32
    targets = torch.zeros(b, 1, h, w, dtype=torch.float32)
    # Samples 0 and 1 are forged, samples 2 and 3 are clean
    targets[0, 0, 8:16, 8:16] = 1.0
    targets[1, 0, 12:20, 12:20] = 1.0

    logits = torch.randn(b, 1, h, w, dtype=torch.float32, requires_grad=True)

    # 1. Check Dice loss gradient in mixed batch
    dice = soft_dice_loss(logits, targets, smooth=1.0)
    dice.backward()

    assert logits.grad is not None
    # Forged samples must receive gradients
    assert not torch.all(logits.grad[0] == 0.0)
    assert not torch.all(logits.grad[1] == 0.0)
    # Clean samples must have strictly zero gradient from Dice
    assert torch.all(logits.grad[2] == 0.0)
    assert torch.all(logits.grad[3] == 0.0)

    # 2. Check all-clean batch
    logits_clean = torch.randn(2, 1, h, w, dtype=torch.float32, requires_grad=True)
    targets_clean = torch.zeros(2, 1, h, w, dtype=torch.float32)
    dice_clean = soft_dice_loss(logits_clean, targets_clean, smooth=1.0)
    assert dice_clean.item() == 0.0
    assert dice_clean.grad_fn is None


def test_combined_loss_backward():
    """Verify CombinedBCEDiceLoss runs forward and backward cleanly."""
    b, h, w = 2, 16, 16
    targets = torch.zeros(b, 1, h, w, dtype=torch.float32)
    targets[0, 0, 4:8, 4:8] = 1.0
    logits = torch.randn(b, 1, h, w, dtype=torch.float32, requires_grad=True)

    criterion = CombinedBCEDiceLoss(pos_weight=2.0, bce_weight=1.0, dice_weight=1.0)
    total, bce, dice = criterion(logits, targets)

    total.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
