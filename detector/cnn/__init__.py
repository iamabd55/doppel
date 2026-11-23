"""CNN detection module for Doppel (Phase 4).

Models:
- ResNetUNet: Plain U-Net baseline with ResNet-18 encoder.
- ResNetCorrNet: ResNet-18 encoder with stride-8 self-correlation and top-4 similarity map.
"""
from detector.cnn.unet import ResNetUNet
from detector.cnn.corr_net import ResNetCorrNet
from detector.cnn.loss import soft_dice_loss, CombinedBCEDiceLoss
from detector.cnn.selection import is_better_checkpoint, load_checkpoint, sweep_thresholds, wilson_ci
from detector.cnn.infer import (
    crop_to_original,
    filter_min_area,
    pad_to_multiple,
    predict_mask,
    predict_from_probs,
)

__all__ = [
    "ResNetUNet",
    "ResNetCorrNet",
    "soft_dice_loss",
    "CombinedBCEDiceLoss",
    "is_better_checkpoint",
    "load_checkpoint",
    "sweep_thresholds",
    "wilson_ci",
    "crop_to_original",
    "filter_min_area",
    "pad_to_multiple",
    "predict_mask",
    "predict_from_probs",
]
