"""Inference utilities for Phase 4 CNN copy-move detection models.

Ensures all probability maps and masks are strictly cropped back to the
original unpadded image dimensions before computing image-level decisions,
connected components, or pixel metrics.
"""
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn


def filter_min_area(binary_mask: np.ndarray, min_area: int = 50) -> np.ndarray:
    """Remove connected components smaller than min_area."""
    if min_area <= 0 or np.sum(binary_mask) == 0:
        return binary_mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8
    )
    filtered = np.zeros_like(binary_mask, dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 1
    return filtered


def pad_to_multiple(
    img: np.ndarray, multiple: int = 32
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """Zero-pad image to next multiple of multiple (e.g. 32). Never resizes.

    Returns:
        padded_img: Array padded with zeros.
        valid_mask: Binary mask (1 for original image pixels, 0 for pad strip).
        pad_box: Tuple of (top, bottom, left, right) pad widths.
    """
    h, w = img.shape[:2]
    target_h = int(np.ceil(h / multiple) * multiple)
    target_w = int(np.ceil(w / multiple) * multiple)

    pad_h = target_h - h
    pad_w = target_w - w

    if pad_h == 0 and pad_w == 0:
        valid_mask = np.ones((h, w), dtype=np.uint8)
        return img, valid_mask, (0, 0, 0, 0)

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if img.ndim == 3:
        padded_img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
    else:
        padded_img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
        )

    valid_mask = np.zeros((target_h, target_w), dtype=np.uint8)
    valid_mask[top : top + h, left : left + w] = 1

    return padded_img, valid_mask, (top, bottom, left, right)


def crop_to_original(
    arr: np.ndarray,
    pad_box: Tuple[int, int, int, int],
    orig_shape: Tuple[int, int],
) -> np.ndarray:
    """Crop padded array back to original unpadded (orig_h, orig_w) dimensions.

    Args:
        arr: 2D array of shape (padded_h, padded_w) or higher-dim array with last dims (H, W).
        pad_box: Tuple of (top, bottom, left, right).
        orig_shape: Tuple of (orig_h, orig_w).

    Returns:
        Cropped array strictly matching orig_shape.
    """
    top, bottom, left, right = pad_box
    orig_h, orig_w = orig_shape

    end_y = top + orig_h
    end_x = left + orig_w

    cropped = arr[..., top:end_y, left:end_x]
    assert cropped.shape[-2:] == (orig_h, orig_w), (
        f"Crop shape mismatch: expected {orig_shape}, got {cropped.shape[-2:]}"
    )
    return cropped


def compute_top_k_mean(prob_map: np.ndarray, k: int = 100) -> float:
    """Compute the mean probability of the top-k pixels in a 2D probability map."""
    total_pixels = prob_map.size
    if total_pixels == 0:
        return 0.0
    k_actual = min(k, total_pixels)
    top_k_values = np.partition(prob_map.ravel(), -k_actual)[-k_actual:]
    return float(np.mean(top_k_values))


def predict_from_probs(
    probs_padded: np.ndarray,
    pad_box: Tuple[int, int, int, int],
    orig_shape: Tuple[int, int],
    threshold: float = 0.5,
    min_area: int = 50,
    image_prob: Optional[float] = None,
    image_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Process padded probability map into unpadded image decision and filtered mask.

    Crops predictions back to original dimensions before any thresholding,
    connected component analysis, or image-level decision.
    Decision rule:
      If image_prob is provided:
        image positive if image_prob >= image_threshold AND predicted mask area >= min_area.
      Else:
        image positive if predicted mask area >= min_area.
    """
    probs_cropped = crop_to_original(probs_padded, pad_box, orig_shape)

    max_prob = float(probs_cropped.max()) if probs_cropped.size > 0 else 0.0
    top100_mean = compute_top_k_mean(probs_cropped, k=100)

    pred_bin = (probs_cropped >= threshold).astype(np.uint8)
    pred_mask = filter_min_area(pred_bin, min_area=min_area)
    has_mask_hit = bool(np.sum(pred_mask) > 0)

    if image_prob is not None:
        is_detected = bool((image_prob >= image_threshold) and has_mask_hit)
    else:
        is_detected = has_mask_hit

    return {
        "is_detected": is_detected,
        "pred_mask": pred_mask,
        "prob_map": probs_cropped,
        "max_prob": max_prob,
        "top100_mean": top100_mean,
        "image_prob": image_prob,
    }


def predict_mask(
    model: nn.Module,
    image: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 50,
    image_threshold: float = 0.5,
    device: Union[str, torch.device] = "cpu",
    pad_multiple: int = 32,
    ablate_corr: bool = False,
) -> Dict[str, Any]:
    """Run end-to-end model inference on an unpadded image.

    Pads image to multiple of pad_multiple, runs forward pass with valid_mask,
    and strictly crops output back to original unpadded resolution before
    thresholding and connected component filtering.

    Args:
        model: PyTorch model (ResNetUNet or ResNetCorrNet).
        image: RGB or grayscale image as uint8 numpy array [H, W, 3] or [H, W].
        threshold: Decision threshold tau.
        min_area: Minimum connected component area in pixels.
        image_threshold: Decision threshold s for image head (if present).
        device: Compute device.
        pad_multiple: Multiple to pad spatial dimensions to.
        ablate_corr: If True, ablate correlation module in ResNetCorrNet.

    Returns:
        Dict with keys:
          'is_detected': bool indicating whether sample meets compound criteria.
          'pred_mask': uint8 binary mask of shape (orig_h, orig_w).
          'prob_map': float32 probability map of shape (orig_h, orig_w).
          'max_prob': float maximum probability inside unpadded region.
          'top100_mean': float mean probability of top-100 pixels.
          'image_prob': float probability from image head (or None).
    """
    orig_h, orig_w = image.shape[:2]
    padded_img, valid_mask, pad_box = pad_to_multiple(image, multiple=pad_multiple)

    # Convert to RGB tensor [1, 3, H, W] in [0, 1]
    if padded_img.ndim == 2:
        padded_img = cv2.cvtColor(padded_img, cv2.COLOR_GRAY2RGB)

    img_tensor = (
        torch.from_numpy(padded_img.transpose(2, 0, 1).astype(np.float32) / 255.0)
        .unsqueeze(0)
        .to(device)
    )
    vm_tensor = (
        torch.from_numpy(valid_mask.astype(np.float32))
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    model.eval()
    with torch.no_grad():
        if hasattr(model, "forward") and "ablate_corr" in model.forward.__code__.co_varnames:
            out = model(img_tensor, valid_mask=vm_tensor, ablate_corr=ablate_corr)
        else:
            out = model(img_tensor, valid_mask=vm_tensor)

        if isinstance(out, tuple):
            logits, img_logit = out
            image_prob = float(torch.sigmoid(img_logit)[0].item())
        else:
            logits = out
            image_prob = None

        probs_padded = torch.sigmoid(logits)[0, 0].cpu().numpy()

    return predict_from_probs(
        probs_padded,
        pad_box=pad_box,
        orig_shape=(orig_h, orig_w),
        threshold=threshold,
        min_area=min_area,
        image_prob=image_prob,
        image_threshold=image_threshold,
    )
