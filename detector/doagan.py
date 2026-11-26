"""Pretrained DOA-GAN inference with the authors' folder preprocessing."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import skimage
import skimage.io
import torch
import torchvision

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "doagan_clean"
WEIGHTS = ROOT / "doa-gan_weights"
SIZE = (320, 320)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TO_TENSOR = torchvision.transforms.ToTensor()


def load_model(weights=None, *, channels=1, device="cpu"):
    """Load the corresponding upstream network and all checkpoint weights."""
    if weights is None:
        weights = WEIGHTS / ("single_channel.pkl" if channels == 1 else "three_channel.pkl")
    spec = importlib.util.spec_from_file_location("upstream_doagan", UPSTREAM / "models.py")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    # The strict checkpoint supplies VGG weights. Override the upstream module's
    # factory locally so it never needs to download the initial VGG weights.
    upstream.torchvision = SimpleNamespace(models=SimpleNamespace(
        vgg19=lambda **_: torchvision.models.vgg19(weights=None),
    ))
    model = upstream.DOA(out_channel=channels)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval()


def preprocess(rgb):
    """Resize the RGB array before converting to float and normalizing."""
    resized = cv2.resize(rgb[..., :3], SIZE, interpolation=cv2.INTER_LINEAR)
    image = skimage.img_as_float32(skimage.img_as_float(resized))
    return TO_TENSOR((image - MEAN) / STD).float()


def read_rgb(path):
    image = skimage.io.imread(str(path))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Unsupported image shape {image.shape} for {path}")
    return image[..., :3]


@torch.no_grad()
def localize(model, tensors):
    """Return one model-resolution foreground probability map per image."""
    mask_logits, _ = model(tensors.to(next(model.parameters()).device))
    return mask_logits.sigmoid()[:, 0].cpu().numpy()


def summarize(heatmap, image_shape):
    """Return the area score at model resolution and the mask at image resolution."""
    score = float(np.count_nonzero(heatmap > 0.5) / heatmap.size)
    height, width = image_shape[:2]
    native_heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
    return score, native_heatmap > 0.5
