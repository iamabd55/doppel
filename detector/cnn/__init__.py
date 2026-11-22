"""CNN detection module for Doppel (Phase 4).

Models:
- ResNetUNet: Plain U-Net baseline with ResNet-18 encoder.
- ResNetCorrNet: ResNet-18 encoder with stride-8 self-correlation and top-4 similarity map.
"""
from detector.cnn.unet import ResNetUNet
from detector.cnn.corr_net import ResNetCorrNet

__all__ = ["ResNetUNet", "ResNetCorrNet"]
