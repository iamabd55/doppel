"""Unit tests for checkpoint selection and validation threshold sweep."""

import pytest
from detector.cnn.selection import is_better_checkpoint, sweep_thresholds


def test_is_better_checkpoint_cold_start():
    """Initial epoch with no previous best is always accepted."""
    curr = {"meets_budget": False, "recall": 0.0, "roc_auc": 0.55}
    assert is_better_checkpoint(curr, None) is True


def test_is_better_checkpoint_both_fail_budget():
    """When neither epoch meets the 1% FAR budget, higher ROC AUC wins."""
    best = {"meets_budget": False, "recall": 0.10, "roc_auc": 0.65}
    better = {"meets_budget": False, "recall": 0.05, "roc_auc": 0.72}
    worse = {"meets_budget": False, "recall": 0.20, "roc_auc": 0.60}

    assert is_better_checkpoint(better, best) is True
    assert is_better_checkpoint(worse, best) is False


def test_is_better_checkpoint_budget_transition():
    """An epoch meeting the FAR budget ALWAYS beats an epoch that failed, even with lower AUC."""
    best = {"meets_budget": False, "recall": 0.0, "roc_auc": 0.88}
    curr = {"meets_budget": True, "recall": 0.32, "roc_auc": 0.75}

    assert is_better_checkpoint(curr, best) is True


def test_is_better_checkpoint_reject_non_budget():
    """An epoch failing the budget never overwrites an existing budget-compliant checkpoint."""
    best = {"meets_budget": True, "recall": 0.32, "roc_auc": 0.75}
    curr = {"meets_budget": False, "recall": 0.80, "roc_auc": 0.99}

    assert is_better_checkpoint(curr, best) is False


def test_is_better_checkpoint_both_meet_budget():
    """When both meet the FAR budget, higher recall wins."""
    best = {"meets_budget": True, "recall": 0.35, "roc_auc": 0.80}
    better = {"meets_budget": True, "recall": 0.42, "roc_auc": 0.78}
    worse = {"meets_budget": True, "recall": 0.30, "roc_auc": 0.85}

    assert is_better_checkpoint(better, best) is True
    assert is_better_checkpoint(worse, best) is False


def test_sweep_thresholds_selection():
    """Verify threshold sweep identifies the threshold maximizing recall within 1% FAR."""
    thresholds = [0.2, 0.4, 0.6, 0.8]

    # 100 clean synth, 100 pristine panels, 100 forged
    # At tau=0.2: clean has 5 flags (5% FAR > 1%), recall=0.90 -> rejects (FAR > 1%)
    # At tau=0.4: clean has 1 flag (1% FAR <= 1%), pristine has 1 flag (1% FAR <= 1%), recall=0.75 -> meets!
    # At tau=0.6: clean has 0 flags (0% FAR <= 1%), pristine has 0 flags (0% FAR <= 1%), recall=0.60 -> meets!
    # At tau=0.8: clean has 0 flags, pristine has 0 flags, recall=0.30 -> meets!

    synth_clean = []
    for i in range(100):
        synth_clean.append({
            "flagged_0.2": i < 5,
            "flagged_0.4": i < 1,
            "flagged_0.6": False,
            "flagged_0.8": False,
        })

    pristine = []
    for i in range(100):
        pristine.append({
            "flagged_0.2": i < 3,
            "flagged_0.4": i < 1,
            "flagged_0.6": False,
            "flagged_0.8": False,
        })

    forged = []
    for i in range(100):
        forged.append({
            "detected_0.2": i < 90,
            "detected_0.4": i < 75,
            "detected_0.6": i < 60,
            "detected_0.8": i < 30,
        })

    res = sweep_thresholds(forged, synth_clean, pristine, thresholds=thresholds, max_far=0.01)

    assert res["meets_budget"] is True
    assert res["best_threshold"] == 0.4
    assert res["best_recall"] == 0.75
    assert res["best_synth_far"] == 0.01
    assert res["best_pristine_far"] == 0.01


def test_sweep_thresholds_2d_grid():
    """Verify 2D grid sweep finds optimal (tau, min_area) operating point within budget."""
    thresholds = [0.3, 0.5, 0.7]
    min_areas = [0, 50, 100]

    synth_clean = []
    pristine = []
    forged = []

    # Suppose at min_area=0, tau=0.5 has 5/100 FAR (fails).
    # But at min_area=50, tau=0.5 drops to 1/100 FAR (passes) and yields recall 0.70.
    for i in range(100):
        c_rec = {}
        p_rec = {}
        f_rec = {}
        for tau in thresholds:
            for area in min_areas:
                if area == 0:
                    c_rec[f"flagged_{tau}_{area}"] = (i < 5) if tau <= 0.5 else (i < 2)
                    p_rec[f"flagged_{tau}_{area}"] = (i < 4) if tau <= 0.5 else (i < 2)
                else:
                    c_rec[f"flagged_{tau}_{area}"] = (i < 1) if tau >= 0.5 else (i < 3)
                    p_rec[f"flagged_{tau}_{area}"] = (i < 1) if tau >= 0.5 else (i < 3)

                if tau == 0.5 and area == 50:
                    f_rec[f"detected_{tau}_{area}"] = (i < 70)
                elif tau == 0.7 and area == 50:
                    f_rec[f"detected_{tau}_{area}"] = (i < 50)
                else:
                    f_rec[f"detected_{tau}_{area}"] = (i < 40)

        synth_clean.append(c_rec)
        pristine.append(p_rec)
        forged.append(f_rec)

    res = sweep_thresholds(
        forged,
        synth_clean,
        pristine,
        thresholds=thresholds,
        min_areas=min_areas,
        max_far=0.01,
    )

    assert res["meets_budget"] is True
    assert res["best_threshold"] == 0.5
    assert res["best_min_area"] == 50
    assert res["best_recall"] == 0.70
    assert res["best_synth_far"] <= 0.01
    assert res["best_pristine_far"] <= 0.01

