"""Checkpoint selection and threshold sweep utilities for Phase 4 CNN detection."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


DEFAULT_THRESHOLDS = [0.3, 0.5, 0.7, 0.9]
DEFAULT_MIN_AREAS = [0, 25, 50, 100, 200, 400, 800]
DEFAULT_IMAGE_THRESHOLDS = [0.0, 0.3, 0.5, 0.7, 0.9]


def load_checkpoint(checkpoint_path: Union[str, Path], map_location: Any = "cpu") -> Dict[str, Any]:
    """Load model checkpoint handling PyTorch 2.6+ weights_only safely."""
    import torch
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)


def sweep_thresholds(
    forged_results: List[Dict[str, Any]],
    synth_clean_results: List[Dict[str, Any]],
    pristine_results: List[Dict[str, Any]],
    thresholds: Optional[List[float]] = None,
    min_areas: Optional[List[int]] = None,
    image_thresholds: Optional[List[float]] = None,
    min_area: Optional[int] = None,
    max_far: float = 0.01,
) -> Dict[str, Any]:
    """Sweep (s, tau, min_area) grid to find operating point maximizing recall within FAR budget.

    Evaluates grid over:
      - s (image probability threshold)
      - tau (pixel probability threshold)
      - min_area (connected component area threshold in px)

    If no configuration meets FAR <= max_far on both sets, selects the configuration
    achieving the lowest max-FAR: min(max(synth_far, pristine_far)).

    Returns:
        Dict containing selection outcome, best thresholds, achieved recall, FARs, and grid stats.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    if min_areas is None:
        if min_area is not None:
            min_areas = [min_area]
        else:
            min_areas = DEFAULT_MIN_AREAS

    has_img_prob = any(
        "image_prob" in r and (r["image_prob"] is not None)
        for r in (forged_results + synth_clean_results + pristine_results)
    )

    if image_thresholds is None:
        image_thresholds = DEFAULT_IMAGE_THRESHOLDS if has_img_prob else [0.0]

    n_forged = len(forged_results)
    n_clean = len(synth_clean_results)
    n_pristine = len(pristine_results)

    best_s = image_thresholds[0]
    best_thresh = thresholds[0]
    best_area = min_areas[0]
    best_recall = 0.0
    best_synth_far = 1.0
    best_pristine_far = 1.0
    budget_met = False

    # Fallback tracker: configuration with lowest max(synth_far, pristine_far)
    lowest_max_far = 2.0
    fallback_s = best_s
    fallback_thresh = best_thresh
    fallback_area = best_area
    fallback_rec = 0.0
    fallback_s_far = 1.0
    fallback_p_far = 1.0

    per_config_stats = []

    for s in image_thresholds:
        for tau in thresholds:
            for area in min_areas:
                # 1. Synthetic clean twin FAR
                if n_clean > 0:
                    clean_flags = 0
                    for r in synth_clean_results:
                        img_ok = True if s == 0.0 else (float(r.get("image_prob", 1.0)) >= s)
                        flag_area = bool(r.get(f"flag_{tau}_{area}", r.get(f"flagged_{tau}_{area}", r.get(f"flagged_{tau}", False))))
                        if img_ok and flag_area:
                            clean_flags += 1
                    synth_far = clean_flags / n_clean
                else:
                    synth_far = 0.0

                # 2. Pristine panels FAR
                if n_pristine > 0:
                    pristine_flags = 0
                    for r in pristine_results:
                        img_ok = True if s == 0.0 else (float(r.get("image_prob", 1.0)) >= s)
                        flag_area = bool(r.get(f"flag_{tau}_{area}", r.get(f"flagged_{tau}_{area}", r.get(f"flagged_{tau}", False))))
                        if img_ok and flag_area:
                            pristine_flags += 1
                    pristine_far = pristine_flags / n_pristine
                else:
                    pristine_far = 0.0

                # 3. Forged image recall (overlap rule)
                if n_forged > 0:
                    forged_hits = 0
                    for r in forged_results:
                        img_ok = True if s == 0.0 else (float(r.get("image_prob", 1.0)) >= s)
                        hit_area = bool(r.get(f"hit_{tau}_{area}", r.get(f"detected_{tau}_{area}", r.get(f"detected_{tau}", False))))
                        if img_ok and hit_area:
                            forged_hits += 1
                    recall = forged_hits / n_forged
                else:
                    recall = 0.0

                meets = (synth_far <= max_far) and (pristine_far <= max_far)
                cur_max_far = max(synth_far, pristine_far)

                per_config_stats.append({
                    "image_threshold": s,
                    "threshold": tau,
                    "min_area": area,
                    "recall": recall,
                    "synth_far": synth_far,
                    "pristine_far": pristine_far,
                    "meets_budget": meets,
                })

                if meets:
                    if (not budget_met) or (recall > best_recall + 1e-5):
                        budget_met = True
                        best_s = s
                        best_thresh = tau
                        best_area = area
                        best_recall = recall
                        best_synth_far = synth_far
                        best_pristine_far = pristine_far
                    elif abs(recall - best_recall) <= 1e-5:
                        if (synth_far + pristine_far) < (best_synth_far + best_pristine_far):
                            best_s = s
                            best_thresh = tau
                            best_area = area
                            best_synth_far = synth_far
                            best_pristine_far = pristine_far
                else:
                    if cur_max_far < lowest_max_far - 1e-5:
                        lowest_max_far = cur_max_far
                        fallback_s = s
                        fallback_thresh = tau
                        fallback_area = area
                        fallback_rec = recall
                        fallback_s_far = synth_far
                        fallback_p_far = pristine_far
                    elif abs(cur_max_far - lowest_max_far) <= 1e-5:
                        if (synth_far + pristine_far) < (fallback_s_far + fallback_p_far):
                            fallback_s = s
                            fallback_thresh = tau
                            fallback_area = area
                            fallback_rec = recall
                            fallback_s_far = synth_far
                            fallback_p_far = pristine_far

    if not budget_met:
        best_s = fallback_s
        best_thresh = fallback_thresh
        best_area = fallback_area
        best_recall = fallback_rec
        best_synth_far = fallback_s_far
        best_pristine_far = fallback_p_far

    return {
        "meets_budget": budget_met,
        "best_image_threshold": best_s,
        "best_threshold": best_thresh,
        "best_min_area": best_area,
        "best_recall": best_recall,
        "best_synth_far": best_synth_far,
        "best_pristine_far": best_pristine_far,
        "stats": per_config_stats,
    }


def is_better_checkpoint(
    current: Dict[str, Any],
    best: Optional[Dict[str, Any]],
) -> bool:
    """Determine if current epoch is strictly better than the current best.

    Protocol:
    1. An epoch meeting FAR <= 1% on both sets ALWAYS beats an epoch that failed.
    2. Among epochs meeting the budget, higher recall wins. If recall is tied, higher fallback score wins.
    3. Among epochs failing the budget, lowest max-FAR wins: min(max(synth_far, pristine_far)).
       If max-FAR is tied, higher fallback score wins.

    Args:
        current: Dict with keys 'meets_budget', 'recall', 'synth_far', 'pristine_far', etc.
        best: Dict with keys 'meets_budget', 'recall', 'synth_far', 'pristine_far', etc., or None.

    Returns:
        True if current is strictly superior to best.
    """
    if best is None:
        return True

    curr_meets = current.get("meets_budget", False)
    best_meets = best.get("meets_budget", False)

    if curr_meets and not best_meets:
        return True
    if not curr_meets and best_meets:
        return False

    curr_score = current.get("fallback_score", current.get("roc_auc", 0.0))
    best_score = best.get("fallback_score", best.get("roc_auc", 0.0))

    if curr_meets and best_meets:
        curr_rec = current.get("recall", 0.0)
        best_rec = best.get("recall", 0.0)
        if curr_rec > best_rec + 1e-4:
            return True
        elif abs(curr_rec - best_rec) <= 1e-4:
            return curr_score > best_score
        return False
    else:
        # Neither meets budget: pick epoch with LOWEST max-FAR
        curr_s_far = current.get("synth_far", current.get("best_synth_far", 1.0))
        curr_p_far = current.get("pristine_far", current.get("best_pristine_far", 1.0))
        best_s_far = best.get("synth_far", best.get("best_synth_far", 1.0))
        best_p_far = best.get("pristine_far", best.get("best_pristine_far", 1.0))

        curr_max_far = max(curr_s_far, curr_p_far)
        best_max_far = max(best_s_far, best_p_far)

        if curr_max_far < best_max_far - 1e-4:
            return True
        elif abs(curr_max_far - best_max_far) <= 1e-4:
            return curr_score > best_score
        return False
