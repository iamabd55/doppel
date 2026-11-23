"""Hybrid detection fusion module for Doppel copy-move forgery detection.

Combines decisions from Adapted DCT, Frozen SIFT, and CorrNet v2 CNN detectors
into standardized confidence tiers:
  - High confidence: 2 or more of (adapted DCT, frozen SIFT, CNN) flag the panel.
  - Medium confidence: CNN only flags the panel (DCT and SIFT silent).
  - Low confidence: Exactly one of DCT or SIFT flags the panel and not CNN.
  - None: No detector flags the panel.
"""

from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import cv2
import numpy as np


class FusionResult(NamedTuple):
    """Structured result of fusing multi-detector decisions."""
    tier: str
    detectors_fired: List[str]
    regions_per_detector: Dict[str, List[Tuple[int, int, int, int]]]


def extract_regions_from_mask(mask: Optional[np.ndarray]) -> List[Tuple[int, int, int, int]]:
    """Extract bounding box regions (x, y, w, h) from connected components in a binary mask."""
    if mask is None or not isinstance(mask, np.ndarray) or not (mask > 0).any():
        return []
    bin_mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    regions: List[Tuple[int, int, int, int]] = []
    for i in range(1, num_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        regions.append((x, y, w, h))
    return regions


def _parse_detector_output(out: Any) -> Tuple[bool, List[Tuple[int, int, int, int]]]:
    """Parse detector output into (flagged, regions) tuple.

    Supports:
      - bool: True/False
      - dict: with keys 'flagged' or 'is_detected', and optional 'regions' or 'mask'
      - tuple/list: (mask, ...) e.g. (mask, score, regions) from predict_cnn
      - np.ndarray: binary prediction mask
      - None: False, []
    """
    if out is None:
        return False, []

    if isinstance(out, bool):
        return out, []

    if isinstance(out, dict):
        flagged = bool(out.get("flagged", out.get("is_detected", False)))
        mask = out.get("mask", None)
        if mask is not None and not flagged:
            flagged = bool((mask > 0).any())
        regions = out.get("regions", [])
        if not regions and mask is not None and flagged:
            regions = extract_regions_from_mask(mask)
        return flagged, list(regions)

    if isinstance(out, (tuple, list)):
        if len(out) == 0:
            return False, []
        mask = out[0] if isinstance(out[0], np.ndarray) else None
        flagged = bool((mask > 0).any()) if mask is not None else False
        regions: List[Tuple[int, int, int, int]] = []
        # Check if any element in tuple is a list of 4-tuples (bounding boxes)
        for elem in out[1:]:
            if isinstance(elem, list) and len(elem) > 0:
                if isinstance(elem[0], (tuple, list)) and len(elem[0]) == 4:
                    regions = [tuple(int(v) for v in b) for b in elem]
                    break
        if not regions and mask is not None and flagged:
            regions = extract_regions_from_mask(mask)
        return flagged, regions

    if isinstance(out, np.ndarray):
        flagged = bool((out > 0).any())
        regions = extract_regions_from_mask(out) if flagged else []
        return flagged, regions

    return bool(out), []


def fuse(
    dct_out: Any,
    sift_out: Any,
    cnn_out: Any,
) -> FusionResult:
    """Fuse decisions from Adapted DCT, Frozen SIFT, and CNN detectors.

    Decision rules:
      - High confidence: 2 or more of (adapted DCT, frozen SIFT, CNN) flag the panel.
      - Medium confidence: CNN only flags the panel (DCT and SIFT silent).
      - Low confidence: Exactly one of DCT or SIFT flags the panel and not CNN.
      - None: No detector flags the panel.

    Args:
        dct_out: Output from Adapted DCT detector (mask, dict, tuple, or bool).
        sift_out: Output from Frozen SIFT detector (mask, dict, tuple, or bool).
        cnn_out: Output from CNN detector (mask, dict, tuple, or bool).

    Returns:
        FusionResult namedtuple with fields (tier, detectors_fired, regions_per_detector).
    """
    dct_flagged, dct_regions = _parse_detector_output(dct_out)
    sift_flagged, sift_regions = _parse_detector_output(sift_out)
    cnn_flagged, cnn_regions = _parse_detector_output(cnn_out)

    detectors_fired: List[str] = []
    if dct_flagged:
        detectors_fired.append("adapted_dct")
    if sift_flagged:
        detectors_fired.append("frozen_sift")
    if cnn_flagged:
        detectors_fired.append("cnn")

    count = len(detectors_fired)

    if count >= 2:
        tier = "high"
    elif cnn_flagged and count == 1:
        tier = "medium"
    elif (dct_flagged or sift_flagged) and not cnn_flagged and count == 1:
        tier = "low"
    else:
        tier = "none"

    regions_per_detector: Dict[str, List[Tuple[int, int, int, int]]] = {
        "adapted_dct": dct_regions,
        "frozen_sift": sift_regions,
        "cnn": cnn_regions,
        "dct": dct_regions,
        "sift": sift_regions,
    }

    return FusionResult(
        tier=tier,
        detectors_fired=detectors_fired,
        regions_per_detector=regions_per_detector,
    )
