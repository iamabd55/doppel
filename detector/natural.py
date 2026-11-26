"""Run the natural-image detectors and combine their decisions and masks."""

from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import yaml

from detector.dct import detect
from detector.sift.localization import detect_sift

ROOT = Path(__file__).resolve().parents[1]
CLASSICAL_CONFIG = ROOT / "configs" / "natural_classical_calibrated.yaml"
DOAGAN_CONFIG = ROOT / "configs" / "natural_doagan_calibrated.yaml"

FROZEN_DCT = dict(
    B=8, K=16, Q=4, min_std=4.0, window=5, min_dist=30,
    min_votes=50, kernel=9, min_size=500,
)
FROZEN_SIFT = dict(
    ratio_threshold=0.70, min_dist=30.0, ransac_threshold=3.0,
    min_inliers=6, cluster_threshold=100.0, mask_radius=12,
    use_convex_hull=True, check_flip=True, min_inlier_ratio=0.25,
    min_inliers_flip=10, filter_orientation=True, orientation_tolerance=20.0,
)


class Detection(NamedTuple):
    flagged: bool
    mask: np.ndarray


class NaturalResult(NamedTuple):
    detectors: dict[str, Detection]
    fusion: dict[str, Detection]
    doagan_score: float


def classical_parameters(config_path=None):
    if config_path is None:
        return FROZEN_DCT.copy(), FROZEN_SIFT.copy()
    config = yaml.safe_load(Path(config_path).read_text())
    dct = FROZEN_DCT | {name: config["dct"][name] for name in ("min_votes", "min_size")}
    sift = FROZEN_SIFT | {"min_inliers": config["sift"]["min_inliers"]}
    return dct, sift


def classical_masks(gray, dct_params, sift_params):
    dct = detect(gray, **dct_params) > 0
    sift = detect_sift(gray, **sift_params)[0] > 0
    return dct, sift


def from_mask(mask):
    return Detection(bool(mask.any()), mask)


def fuse(dct, sift, doagan):
    """Apply F1 through F5 to three aligned binary detections."""
    shape = dct.mask.shape
    if sift.mask.shape != shape or doagan.mask.shape != shape:
        raise ValueError("All detector masks must have the same image dimensions")

    empty = np.zeros(shape, dtype=bool)
    dct_mask = dct.mask if dct.flagged else empty
    sift_mask = sift.mask if sift.flagged else empty
    doagan_mask = doagan.mask if doagan.flagged else empty
    classical_or = dct.flagged or sift.flagged
    union = dct_mask | sift_mask
    all_masks = union | doagan_mask
    majority = sum((dct.flagged, sift.flagged, doagan.flagged)) >= 2
    return {
        "F1": Detection(classical_or, union),
        "F2": Detection(dct.flagged and sift.flagged, dct_mask & sift_mask),
        "F3": Detection(classical_or or doagan.flagged, all_masks),
        "F4": Detection(majority, all_masks if majority else empty),
        "F5": Detection(classical_or and doagan.flagged, union & doagan_mask),
    }


class NaturalPipeline:
    """Load the pretrained model once and run the complete pipeline per RGB image."""

    def __init__(self, *, classical_config=CLASSICAL_CONFIG, doagan_config=DOAGAN_CONFIG,
                 doagan_threshold=None, weights=None, device="cpu"):
        from detector.doagan import load_model

        self.dct_params, self.sift_params = classical_parameters(classical_config)
        if doagan_threshold is None:
            config = yaml.safe_load(Path(doagan_config).read_text())
            doagan_threshold = config["budgets"]["2%"]["threshold"]
        self.threshold = float(doagan_threshold)
        if not 0 <= self.threshold <= np.nextafter(1.0, np.inf):
            raise ValueError("Invalid DOA-GAN area threshold")
        self.model = load_model(weights, device=device)

    def predict(self, rgb):
        from skimage import img_as_ubyte
        from detector.doagan import localize, preprocess, summarize

        gray = cv2.cvtColor(img_as_ubyte(rgb), cv2.COLOR_RGB2GRAY)
        dct, sift = (from_mask(mask) for mask in classical_masks(gray, self.dct_params, self.sift_params))
        heatmap = localize(self.model, preprocess(rgb).unsqueeze(0))[0]
        score, mask = summarize(heatmap, rgb.shape)
        positive = score >= self.threshold
        doagan = Detection(positive, mask if positive else np.zeros_like(mask))
        detectors = {"dct": dct, "sift": sift, "doagan": doagan}
        return NaturalResult(detectors, fuse(dct, sift, doagan), score)
