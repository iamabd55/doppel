"""Plain U-Net baseline with ResNet-18 encoder (Phase 4).

Interface:
    forward(image, valid_mask=None) -> 1-channel mask logits at input resolution.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class ConvBlock(nn.Module):
    """Two 3x3 convs with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResNetUNet(nn.Module):
    """ResNet-18 based U-Net for binary copy-move forgery segmentation."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet18(weights=weights)

        # Standard ImageNet normalization buffer
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # Encoder stages
        self.conv1 = base.conv1  # Stride 2, out 64
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool  # Stride 4, out 64

        self.layer1 = base.layer1  # Stride 4, out 64
        self.layer2 = base.layer2  # Stride 8, out 128
        self.layer3 = base.layer3  # Stride 16, out 256
        self.layer4 = base.layer4  # Stride 32, out 512

        # Decoder stages (upsample to exact spatial sizes of skip features)
        self.dec4 = ConvBlock(512 + 256, 256)  # Layer4 + Layer3
        self.dec3 = ConvBlock(256 + 128, 128)  # Up + Layer2
        self.dec2 = ConvBlock(128 + 64, 64)    # Up + Layer1
        self.dec1 = ConvBlock(64 + 64, 32)     # Up + conv1 output (stride 2)

        self.final_up = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(
        self,
        image: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            image: [B, 3, H, W] tensor in range [0, 1].
            valid_mask: [B, 1, H, W] optional binary mask of valid (non-padded) pixels.

        Returns:
            [B, 1, H, W] logits mask at input resolution.
        """
        orig_h, orig_w = image.shape[2], image.shape[3]

        # Normalize with ImageNet statistics
        x = (image - self.mean) / self.std

        # Encoder
        x0 = self.relu(self.bn1(self.conv1(x)))  # Stride 2, 64 ch
        x1 = self.maxpool(x0)                    # Stride 4, 64 ch
        x1 = self.layer1(x1)                     # Stride 4, 64 ch
        x2 = self.layer2(x1)                     # Stride 8, 128 ch
        x3 = self.layer3(x2)                     # Stride 16, 256 ch
        x4 = self.layer4(x3)                     # Stride 32, 512 ch

        # Decoder with explicit target spatial size interpolation to handle any dimensions
        # Stride 32 -> Stride 16
        up4 = F.interpolate(
            x4, size=x3.shape[2:], mode="bilinear", align_corners=False
        )
        d4 = self.dec4(torch.cat([up4, x3], dim=1))

        # Stride 16 -> Stride 8
        up3 = F.interpolate(
            d4, size=x2.shape[2:], mode="bilinear", align_corners=False
        )
        d3 = self.dec3(torch.cat([up3, x2], dim=1))

        # Stride 8 -> Stride 4
        up2 = F.interpolate(
            d3, size=x1.shape[2:], mode="bilinear", align_corners=False
        )
        d2 = self.dec2(torch.cat([up2, x1], dim=1))

        # Stride 4 -> Stride 2
        up1 = F.interpolate(
            d2, size=x0.shape[2:], mode="bilinear", align_corners=False
        )
        d1 = self.dec1(torch.cat([up1, x0], dim=1))

        # Stride 2 -> Full resolution (H, W)
        up0 = F.interpolate(
            d1, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )
        logits = self.final_up(up0)

        return logits
