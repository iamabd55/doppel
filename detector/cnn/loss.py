"""Loss functions for CNN copy-move forgery detection."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Compute soft Dice loss only over samples containing forged pixels.

    Clean samples (target mask all zeros) are excluded from the Dice computation
    so they do not artificially penalize the model or dilute the Dice metric.
    If no samples in the batch contain forged pixels, returns 0.0 with no gradient.

    Args:
        logits: Raw predictions of shape [B, 1, H, W] or [B, H, W].
        targets: Binary ground truth mask in {0.0, 1.0} of same spatial shape.
        smooth: Smoothing term for numerator and denominator.

    Returns:
        Scalar Dice loss tensor in [0, 1].
    """
    probs = torch.sigmoid(logits)
    b = logits.shape[0]
    probs_flat = probs.view(b, -1)
    targets_flat = targets.view(b, -1)

    # Detect forged samples (mask has at least one positive pixel)
    has_forged = targets_flat.sum(dim=1) > 0.5
    if not has_forged.any():
        return logits.new_tensor(0.0)

    forged_probs = probs_flat[has_forged]
    forged_targets = targets_flat[has_forged]

    intersection = (forged_probs * forged_targets).sum(dim=1)
    cardinality = forged_probs.sum(dim=1) + forged_targets.sum(dim=1)

    dice_per_sample = 1.0 - (2.0 * intersection + smooth) / (cardinality + smooth)
    return dice_per_sample.mean()


class CombinedBCEDiceLoss(nn.Module):
    """Combined BCE and soft Dice loss for copy-move segmentation.

    BCE is computed over valid pixels across all samples (both forged and clean).
    Dice is computed only over samples with forged pixels.
    """

    def __init__(
        self,
        pos_weight: float = 1.0,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.pos_weight_val = pos_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute total loss, BCE loss, and soft Dice loss.

        Args:
            logits: Predicted logits [B, 1, H, W]
            targets: Target binary masks [B, 1, H, W]
            valid_mask: Optional binary mask of valid pixels [B, 1, H, W]

        Returns:
            Tuple of (total_loss, bce_loss, dice_loss)
        """
        pos_weight_tensor = logits.new_tensor([self.pos_weight_val])
        if valid_mask is not None:
            bce_unreduced = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=pos_weight_tensor,
                reduction="none",
            )
            bce = (bce_unreduced * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
        else:
            bce = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=pos_weight_tensor,
            )

        dice = soft_dice_loss(logits, targets, smooth=self.smooth)
        total = self.bce_weight * bce + self.dice_weight * dice
        return total, bce, dice
