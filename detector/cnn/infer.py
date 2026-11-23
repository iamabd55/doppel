"""Inference utilities for Phase 4 CNN copy-move detection models.

Ensures all probability maps and masks are strictly cropped back to the
original unpadded image dimensions before computing image-level decisions,
connected components, or pixel metrics.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml



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


_MODEL_CACHE: Dict[Tuple[str, str], nn.Module] = {}


def load_cnn_model(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Union[str, torch.device] = "cpu",
    model_type: Optional[str] = None,
    image_head: Optional[bool] = None,
) -> nn.Module:
    """Load and cache CNN detection model on specified device.

    Args:
        checkpoint_path: Path to checkpoint .pt file. Defaults to 'checkpoints/corr_v2/last.pt'.
        device: Target execution device ('cpu' or 'cuda').
        model_type: Optional override ('corr' or 'unet'). Inferred if None.
        image_head: Optional override for image classification head. Inferred if None.

    Returns:
        Loaded PyTorch model in eval mode.
    """
    if checkpoint_path is None:
        default_ckpt = Path("checkpoints/corr_v2/last.pt")
        if default_ckpt.exists():
            checkpoint_path = default_ckpt
        else:
            raise FileNotFoundError(
                f"Default checkpoint not found at {default_ckpt}. Please specify checkpoint_path."
            )

    ckpt_path = Path(checkpoint_path).resolve()
    dev_str = str(device)
    cache_key = (str(ckpt_path), dev_str)

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    from detector.cnn.corr_net import ResNetCorrNet
    from detector.cnn.selection import load_checkpoint
    from detector.cnn.unet import ResNetUNet

    ckpt = load_checkpoint(ckpt_path, map_location=device)
    config = ckpt.get("config", {})

    m_type = model_type or config.get("model")
    if m_type is None:
        state_keys = ckpt["model_state_dict"].keys()
        if any("corr" in k for k in state_keys) or any("dec8.conv1.0.weight" in k for k in state_keys):
            m_type = "corr"
        else:
            m_type = "unet"

    has_img_head = image_head
    if has_img_head is None:
        has_img_head = config.get("image_head", False) or any(
            "image_head_fc" in k for k in ckpt.get("model_state_dict", {}).keys()
        )

    if m_type == "unet":
        model = ResNetUNet(pretrained=False)
    else:
        model = ResNetCorrNet(pretrained=False, image_head=has_img_head)

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    _MODEL_CACHE[cache_key] = model
    return model


def load_cnn_thresholds(
    config_path: Union[str, Path] = "configs/cnn_threshold.yaml",
    image_class: Optional[str] = "Blot/Gel",
    budget: float = 0.01,
) -> Dict[str, Any]:
    """Load locked operating thresholds for a specific image class and budget.

    Args:
        config_path: Path to configs/cnn_threshold.yaml.
        image_class: Panel domain ('Blot/Gel' or 'Microscopy').
        budget: False alarm rate budget (default: 0.01 = 1.0%).

    Returns:
        Dict with keys: 'image_threshold', 'pixel_threshold', 'min_area_pixels', 'target_far', 'class_key'.
    """
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        # Fallback defaults for primary budget
        return {
            "image_threshold": 0.95,
            "pixel_threshold": 0.30,
            "min_area_pixels": 25,
            "target_far": budget,
            "class_key": "blot_gel",
        }

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Normalize class key
    cls_str = (image_class or "overall").lower()
    if "blot" in cls_str or "gel" in cls_str:
        cls_key = "blot_gel"
    elif "micro" in cls_str:
        cls_key = "microscopy"
    else:
        cls_key = "overall"

    budgets_dict = cfg.get("budgets", {})
    selected_budget_dict = None

    # Try direct key match: e.g. budget_1_0pct or budget_1_5pct
    b_tag = f"budget_{str(round(budget * 100, 2)).replace('.', '_')}pct"
    if b_tag in budgets_dict:
        selected_budget_dict = budgets_dict[b_tag]
    elif budgets_dict:
        # Search by closest numerical target_far
        best_diff = float("inf")
        for k, v in budgets_dict.items():
            t_far = float(v.get("target_far", 0.01))
            diff = abs(t_far - budget)
            if diff < best_diff:
                best_diff = diff
                selected_budget_dict = v

    if selected_budget_dict is not None and cls_key in selected_budget_dict:
        entry = selected_budget_dict[cls_key]
    elif selected_budget_dict is not None and "overall" in selected_budget_dict:
        entry = selected_budget_dict["overall"]
    elif "per_class" in cfg and cls_key in cfg["per_class"]:
        entry = cfg["per_class"][cls_key]
    elif "overall" in cfg:
        entry = cfg["overall"]
    else:
        entry = cfg

    image_thresh = float(entry.get("image_threshold", cfg.get("image_threshold", 0.99)))
    pixel_thresh = float(entry.get("pixel_threshold", cfg.get("pixel_threshold", 0.70)))
    min_area = int(entry.get("min_area_pixels", cfg.get("min_area_pixels", 25)))

    return {
        "image_threshold": image_thresh,
        "pixel_threshold": pixel_thresh,
        "min_area_pixels": min_area,
        "target_far": budget,
        "class_key": cls_key,
    }


def predict_cnn(
    image: np.ndarray,
    image_class: Optional[str] = "Blot/Gel",
    config_path: Union[str, Path] = "configs/cnn_threshold.yaml",
    checkpoint_path: Optional[Union[str, Path]] = None,
    budget: float = 0.01,
    device: str = "cpu",
    model: Optional[nn.Module] = None,
    pad_multiple: int = 32,
    return_dict: bool = False,
) -> Union[Tuple[np.ndarray, float, List[Tuple[int, int, int, int]]], Dict[str, Any]]:
    """Standardized prediction interface for Phase 4 CNN copy-move detection.

    Matches the predict interface conventions of the DCT and SIFT detectors.
    Accepts arbitrary image dimensions, pads to multiples of pad_multiple,
    and strictly crops predictions back to native image dimensions.

    Args:
        image: Input image as RGB or grayscale uint8 numpy array of shape (H, W, 3) or (H, W).
        image_class: Panel domain ('Blot/Gel' or 'Microscopy').
        config_path: Path to YAML configuration with locked thresholds.
        checkpoint_path: Path to model checkpoint (.pt). Defaults to 'checkpoints/corr_v2/last.pt'.
        budget: False alarm rate budget (default: 0.01 = 1.0%).
        device: Execution device ('cpu' or 'cuda'). Defaults to 'cpu'.
        model: Optional pre-loaded PyTorch model. If None, loaded from checkpoint_path.
        pad_multiple: Multiple to zero-pad spatial dimensions to (default: 32).
        return_dict: If True, return dictionary with diagnostic maps.

    Returns:
        If return_dict is False (default):
            mask: uint8 binary mask (0 or 255) of shape (H, W).
            image_score: float confidence score in [0, 1].
            regions: list of bounding boxes [(x, y, w, h), ...] for detected components.
        If return_dict is True:
            dict containing 'mask', 'image_score', 'regions', 'is_detected',
            'prob_map', 'image_prob', 'thresholds', etc.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected image as numpy ndarray, got {type(image)}")

    orig_shape = image.shape[:2]

    # Load thresholds from YAML
    thresh_dict = load_cnn_thresholds(
        config_path=config_path,
        image_class=image_class,
        budget=budget,
    )
    img_thresh = thresh_dict["image_threshold"]
    pix_thresh = thresh_dict["pixel_threshold"]
    m_area = thresh_dict["min_area_pixels"]

    # Load model if not provided
    if model is None:
        model = load_cnn_model(checkpoint_path=checkpoint_path, device=device)

    # Run inference with zero-pad and native unpad
    raw_res = predict_mask(
        model=model,
        image=image,
        threshold=pix_thresh,
        min_area=m_area,
        image_threshold=img_thresh,
        device=device,
        pad_multiple=pad_multiple,
    )

    is_detected = raw_res["is_detected"]
    pred_mask = raw_res["pred_mask"]
    image_prob = raw_res["image_prob"]
    max_prob = raw_res["max_prob"]

    image_score = float(image_prob) if image_prob is not None else float(max_prob)

    # If image decision is negative, gatekeeper suppresses all pixels to zero
    if is_detected:
        final_mask = (pred_mask > 0).astype(np.uint8) * 255
    else:
        final_mask = np.zeros(orig_shape, dtype=np.uint8)

    # Extract region bounding boxes for surviving components
    regions: List[Tuple[int, int, int, int]] = []
    if np.sum(final_mask) > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (final_mask > 0).astype(np.uint8), connectivity=8
        )
        for i in range(1, num_labels):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            regions.append((x, y, w, h))

    if return_dict:
        return {
            "mask": final_mask,
            "image_score": image_score,
            "regions": regions,
            "is_detected": is_detected,
            "prob_map": raw_res["prob_map"],
            "max_prob": max_prob,
            "image_prob": image_prob,
            "thresholds": thresh_dict,
        }

    return final_mask, image_score, regions


# Convenience alias matching detect and detect_sift
predict = predict_cnn

