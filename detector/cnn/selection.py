"""Checkpoint selection and threshold sweep utilities for Phase 4 CNN detection."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


DEFAULT_THRESHOLDS = [0.3, 0.5, 0.7, 0.9]
DEFAULT_MIN_AREAS = [0, 25, 50, 100, 200, 400, 800]


def sweep_thresholds(
    forged_results: List[Dict[str, Any]],
    synth_clean_results: List[Dict[str, Any]],
    pristine_results: List[Dict[str, Any]],
    thresholds: Optional[List[float]] = None,
    min_areas: Optional[List[int]] = None,
    min_area: Optional[int] = None,
    max_far: float = 0.01,
) -> Dict[str, Any]:
    """Sweep (tau, min_area) grid to find operating point maximizing recall within FAR budget.

    Evaluates grid over tau in {0.3, 0.5, 0.7, 0.9} x min_area in {0, 25, 50, 100, 200, 400, 800}.
    Maintains backward compatibility with legacy 1D records (flagged_{tau}).

    Args:
        forged_results: List of dicts for forged validation images.
        synth_clean_results: List of dicts for clean twin validation images.
        pristine_results: List of dicts for real pristine held-out panels.
        thresholds: List of candidate thresholds. Defaults to [0.3, 0.5, 0.7, 0.9].
        min_areas: List of candidate min_area values. Defaults to [0, 25, 50, 100, 200, 400, 800].
        min_area: Legacy single min_area argument. If min_areas is None and min_area is given, [min_area].
        max_far: Maximum acceptable false alarm rate (default 0.01 = 1.0%).

    Returns:
        Dict containing selection outcome, best_threshold, best_min_area, achieved recall, FARs, and grid stats.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    if min_areas is None:
        if min_area is not None:
            min_areas = [min_area]
        else:
            min_areas = DEFAULT_MIN_AREAS

    n_forged = len(forged_results)
    n_clean = len(synth_clean_results)
    n_pristine = len(pristine_results)

    best_thresh = thresholds[0]
    best_area = min_areas[0]
    best_recall = 0.0
    best_synth_far = 1.0
    best_pristine_far = 1.0
    budget_met = False

    per_config_stats = []

    for tau in thresholds:
        for area in min_areas:
            # 1. Synthetic clean twin FAR
            if n_clean > 0:
                clean_flags = sum(
                    1
                    for r in synth_clean_results
                    if r.get(f"flagged_{tau}_{area}", r.get(f"flagged_{tau}", False))
                )
                synth_far = clean_flags / n_clean
            else:
                synth_far = 0.0

            # 2. Pristine panels FAR
            if n_pristine > 0:
                pristine_flags = sum(
                    1
                    for r in pristine_results
                    if r.get(f"flagged_{tau}_{area}", r.get(f"flagged_{tau}", False))
                )
                pristine_far = pristine_flags / n_pristine
            else:
                pristine_far = 0.0

            # 3. Forged image recall (overlap rule)
            if n_forged > 0:
                forged_hits = sum(
                    1
                    for r in forged_results
                    if r.get(f"detected_{tau}_{area}", r.get(f"detected_{tau}", False))
                )
                recall = forged_hits / n_forged
            else:
                recall = 0.0

            meets = (synth_far <= max_far) and (pristine_far <= max_far)

            per_config_stats.append({
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
                    best_thresh = tau
                    best_area = area
                    best_recall = recall
                    best_synth_far = synth_far
                    best_pristine_far = pristine_far
                elif abs(recall - best_recall) <= 1e-5:
                    # Tie-break: prefer lower combined FAR
                    if (synth_far + pristine_far) < (best_synth_far + best_pristine_far):
                        best_thresh = tau
                        best_area = area
                        best_synth_far = synth_far
                        best_pristine_far = pristine_far

    return {
        "meets_budget": budget_met,
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
    3. Among epochs failing the budget, higher fallback score (top-100 pixel mean ROC AUC) wins.

    Args:
        current: Dict with keys 'meets_budget', 'recall', and 'roc_auc' or 'fallback_score'.
        best: Dict with keys 'meets_budget', 'recall', and 'roc_auc' or 'fallback_score', or None for initial epoch.

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
        return curr_score > best_score
