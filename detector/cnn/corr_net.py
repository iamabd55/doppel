"""Correlation Network (ResNetCorrNet) for Copy-Move Forgery Detection (Phase 4).

Architecture:
- ResNet-18 backbone encoder extracting stride 2, 4, 8, 16 feature maps.
- Stride-8 self-correlation module:
  * Project layer2 features to 128-d, L2 normalize.
  * Compute full cosine similarity matrix (N x N, where N = H/8 * W/8).
  * Mask out self-match neighborhood (spatial Chebyshev distance <= 1 cell in grid).
  * Mask out padded positions via valid_mask (padded cells match nothing).
  * Extract top-4 similarity values per position -> [B, 4, H/8, W/8] map.
- Decoder:
  * Concatenate top-4 correlation map with layer2 features (128 + 4 = 132 ch).
  * Skip connections from stride 16, stride 4, and stride 2.
  * Explicit size interpolation to support narrow panels (e.g. 32xW and 40xW).
  * Output: 1-channel mask logits at full input resolution.
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


class ResNetCorrNet(nn.Module):
    """Self-correlation copy-move detector with ResNet-18 backbone."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet18(weights=weights)

        # ImageNet normalization constants
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

        # Correlation projection at stride 8
        self.proj = nn.Conv2d(128, 128, kernel_size=1, bias=False)

        # Decoder stages
        # Stride 16 branch
        self.dec16 = ConvBlock(256, 128)

        # Stride 8 fusion: up from 16 (128) + layer2 (128) + top4 corr (4) = 260
        self.dec8 = ConvBlock(128 + 128 + 4, 128)

        # Stride 4 fusion: up from 8 (128) + layer1 (64) = 192
        self.dec4 = ConvBlock(128 + 64, 64)

        # Stride 2 fusion: up from 4 (64) + conv1 (64) = 128
        self.dec2 = ConvBlock(64 + 64, 32)

        # Full resolution output
        self.final_up = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        self.ablate_corr = False

    def _compute_correlation(
        self,
        feat: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute self-correlation at stride 8 with spatial and padding masking.

        Args:
            feat: [B, 128, H8, W8] feature map.
            valid_mask: [B, 1, H, W] optional binary mask of valid pixels.

        Returns:
            [B, 4, H8, W8] top-4 similarity feature map.
        """
        B, C, H8, W8 = feat.shape
        N = H8 * W8

        # Project and L2 normalize
        p = self.proj(feat)
        p = F.normalize(p, p=2, dim=1, eps=1e-6)

        # Reshape to [B, C, N]
        p_flat = p.view(B, C, N)

        # Cosine similarity matrix: [B, N, N]
        sim = torch.bmm(p_flat.transpose(1, 2), p_flat)

        # 1. Spatial self-match neighborhood mask: spatial distance <= 1 cell (cached by grid shape)
        grid_key = (H8, W8, str(feat.device))
        if not hasattr(self, "_spatial_mask_cache"):
            self._spatial_mask_cache = {}
        if grid_key not in self._spatial_mask_cache:
            ys = torch.arange(H8, device=feat.device)
            xs = torch.arange(W8, device=feat.device)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            grid_y = grid_y.reshape(N)
            grid_x = grid_x.reshape(N)

            dy = torch.abs(grid_y.unsqueeze(0) - grid_y.unsqueeze(1))
            dx = torch.abs(grid_x.unsqueeze(0) - grid_x.unsqueeze(1))
            self._spatial_mask_cache[grid_key] = ((dy <= 1) & (dx <= 1)).unsqueeze(0)

        spatial_self_mask = self._spatial_mask_cache[grid_key]

        # 2. Valid-pixel mask at stride 8
        if valid_mask is not None:
            # Interpolate valid mask to stride 8
            vm8 = F.interpolate(
                valid_mask.float(), size=(H8, W8), mode="nearest"
            )[:, 0]
            vm8_flat = vm8.view(B, N) > 0.5  # [B, N] boolean
            # A pair is valid only if both positions are valid
            valid_pair = vm8_flat.unsqueeze(2) & vm8_flat.unsqueeze(1)  # [B, N, N]
            invalid_pair = ~valid_pair
        else:
            invalid_pair = torch.zeros(
                (B, N, N), dtype=torch.bool, device=feat.device
            )

        # Combined mask of forbidden match locations
        forbidden = spatial_self_mask | invalid_pair

        # fp16-safe masking: set forbidden locations to -1e4
        sim = sim.masked_fill(forbidden, -1e4)

        # Extract top-4 similarity values per position
        k = min(4, N)
        topk_vals, _ = torch.topk(sim, k=k, dim=-1)  # [B, N, k]

        # Clamp to [0, 1] so masked and padded positions become 0.0
        topk_vals = torch.clamp(topk_vals, min=0.0, max=1.0)

        # If N < 4 (rare edge case), pad with zeros to 4 channels
        if k < 4:
            topk_vals = F.pad(topk_vals, (0, 4 - k), value=0.0)

        # Reshape to [B, 4, H8, W8]
        topk_map = topk_vals.transpose(1, 2).view(B, 4, H8, W8)
        return topk_map

    def forward(
        self,
        image: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        ablate_corr: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            image: [B, 3, H, W] tensor in range [0, 1].
            valid_mask: [B, 1, H, W] optional binary mask of valid (non-padded) pixels.
            ablate_corr: If True, zeroes correlation channels at decoder input.

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

        # Correlation at stride 8
        corr_map = self._compute_correlation(x2, valid_mask=valid_mask)
        if ablate_corr or getattr(self, "ablate_corr", False):
            corr_map = torch.zeros_like(corr_map)

        # Decoder with explicit target spatial size interpolation
        # Stride 16 -> Stride 8
        d16 = self.dec16(x3)
        up16 = F.interpolate(
            d16, size=x2.shape[2:], mode="bilinear", align_corners=False
        )

        # Stride 8 fusion
        d8_in = torch.cat([up16, x2, corr_map], dim=1)  # 128 + 128 + 4 = 260 ch
        d8 = self.dec8(d8_in)

        # Stride 8 -> Stride 4
        up8 = F.interpolate(
            d8, size=x1.shape[2:], mode="bilinear", align_corners=False
        )
        d4_in = torch.cat([up8, x1], dim=1)  # 128 + 64 = 192 ch
        d4 = self.dec4(d4_in)

        # Stride 4 -> Stride 2
        up4 = F.interpolate(
            d4, size=x0.shape[2:], mode="bilinear", align_corners=False
        )
        d2_in = torch.cat([up4, x0], dim=1)  # 64 + 64 = 128 ch
        d2 = self.dec2(d2_in)

        # Stride 2 -> Full resolution (H, W)
        up0 = F.interpolate(
            d2, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )
        logits = self.final_up(up0)

        return logits
