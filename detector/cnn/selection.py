"""Checkpoint selection and threshold sweep utilities for Phase 4 CNN detection."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def sweep_thresholds(
    forged_results: List[Dict[str, Any]],
    synth_clean_results: List[Dict[str, Any]],
    pristine_results: List[Dict[str, Any]],
    thresholds: Optional[List[float]] = None,
    min_area: int = 50,
    max_far: float = 0.01,
) -> Dict[str, Any]:
    """Sweep pixel thresholds to find the threshold maximizing recall within the FAR budget.

    Each item in forged_results is expected to contain:
      - 'max_prob': float
      - 'probs': np.ndarray (or precomputed pixel counts per threshold)
      - 'mask': np.ndarray (ground truth binary mask)
    Or precomputed summary flags.

    Args:
        forged_results: List of dicts for forged validation images.
        synth_clean_results: List of dicts for clean twin validation images.
        pristine_results: List of dicts for real pristine held-out panels.
        thresholds: List of candidate thresholds to sweep. Defaults to 0.05 to 0.95 in 0.05 steps.
        min_area: Minimum connected component area in pixels.
        max_far: Maximum acceptable false alarm rate (default 0.01 = 1.0%).

    Returns:
        Dict containing selection outcome, best threshold, achieved recall, FARs, and metric key.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.linspace(0.05, 0.95, 19).tolist()]

    n_forged = len(forged_results)
    n_clean = len(synth_clean_results)
    n_pristine = len(pristine_results)

    best_thresh = 0.5
    best_recall = 0.0
    best_synth_far = 1.0
    best_pristine_far = 1.0
    budget_met = False

    per_thresh_stats = []

    for tau in thresholds:
        # 1. Synthetic clean twin FAR
        if n_clean > 0:
            clean_flags = sum(1 for r in synth_clean_results if r.get(f"flagged_{tau}", False))
            synth_far = clean_flags / n_clean
        else:
            synth_far = 0.0

        # 2. Pristine panels FAR
        if n_pristine > 0:
            pristine_flags = sum(1 for r in pristine_results if r.get(f"flagged_{tau}", False))
            pristine_far = pristine_flags / n_pristine
        else:
            pristine_far = 0.0

        # 3. Forged image recall (overlap rule: flagged and overlaps ground truth)
        if n_forged > 0:
            forged_hits = sum(1 for r in forged_results if r.get(f"detected_{tau}", False))
            recall = forged_hits / n_forged
        else:
            recall = 0.0

        meets = (synth_far <= max_far) and (pristine_far <= max_far)

        per_thresh_stats.append({
            "threshold": tau,
            "recall": recall,
            "synth_far": synth_far,
            "pristine_far": pristine_far,
            "meets_budget": meets,
        })

        if meets:
            if (not budget_met) or (recall > best_recall):
                budget_met = True
                best_thresh = tau
                best_recall = recall
                best_synth_far = synth_far
                best_pristine_far = pristine_far

    return {
        "meets_budget": budget_met,
        "best_threshold": best_thresh,
        "best_recall": best_recall,
        "best_synth_far": best_synth_far,
        "best_pristine_far": best_pristine_far,
        "stats": per_thresh_stats,
    }


def is_better_checkpoint(
    current: Dict[str, Any],
    best: Optional[Dict[str, Any]],
) -> bool:
    """Determine if current epoch is strictly better than the current best.

    Protocol:
    1. An epoch meeting FAR <= 1% on both sets ALWAYS beats an epoch that failed.
    2. Among epochs meeting the budget, higher recall wins. If recall is tied, higher AUC wins.
    3. Among epochs failing the budget, higher ROC AUC wins.

    Args:
        current: Dict with keys 'meets_budget', 'recall', 'roc_auc'.
        best: Dict with keys 'meets_budget', 'recall', 'roc_auc', or None for initial epoch.

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

    if curr_meets and best_meets:
        curr_rec = current.get("recall", 0.0)
        best_rec = best.get("recall", 0.0)
        if curr_rec > best_rec + 1e-4:
            return True
        elif abs(curr_rec - best_rec) <= 1e-4:
            return current.get("roc_auc", 0.0) > best.get("roc_auc", 0.0)
        return False
    else:
        return current.get("roc_auc", 0.0) > best.get("roc_auc", 0.0)
