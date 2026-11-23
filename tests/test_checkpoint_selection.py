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


def test_is_better_checkpoint_lowest_max_far_fallback():
    """When no epoch passes the FAR budget, lowest max-FAR wins."""
    # Epoch 1: synth_far 0.15, pristine_far 0.10 -> max_far = 0.15, ROC AUC 0.85
    ckpt1 = {
        "meets_budget": False,
        "best_synth_far": 0.15,
        "best_pristine_far": 0.10,
        "roc_auc": 0.85,
    }
    # Epoch 2: synth_far 0.08, pristine_far 0.09 -> max_far = 0.09, ROC AUC 0.82
    ckpt2 = {
        "meets_budget": False,
        "best_synth_far": 0.08,
        "best_pristine_far": 0.09,
        "roc_auc": 0.82,
    }

    # Ckpt2 has lower max-FAR (0.09 < 0.15), so it must beat Ckpt1 even though Ckpt1 had higher AUC
    assert is_better_checkpoint(ckpt2, ckpt1) is True
    assert is_better_checkpoint(ckpt1, ckpt2) is False


def test_sweep_thresholds_3d_compound():
    """Verify 3D grid sweep with image head probability s."""
    # 100 clean, 100 pristine, 100 forged
    # Suppose pixel alone has 10/100 FAR (fails).
    # But filtering by image_prob >= 0.7 cuts FAR to 1/100 while keeping recall 0.65.
    thresholds = [0.5]
    min_areas = [50]
    image_thresholds = [0.0, 0.7]

    synth_clean = []
    pristine = []
    forged = []

    for i in range(100):
        # Clean: 10 images have high pixel activations, but only 1 has image_prob >= 0.7
        c_prob = 0.85 if i == 0 else (0.4 if i < 10 else 0.1)
        synth_clean.append({
            "image_prob": c_prob,
            "flag_0.5_50": i < 10,
        })

        p_prob = 0.80 if i == 0 else (0.3 if i < 8 else 0.05)
        pristine.append({
            "image_prob": p_prob,
            "flag_0.5_50": i < 8,
        })

        # Forged: 65 have image_prob >= 0.7 and hit_0.5_50
        f_prob = 0.90 if i < 65 else 0.2
        forged.append({
            "image_prob": f_prob,
            "hit_0.5_50": i < 70,
        })

    res = sweep_thresholds(
        forged,
        synth_clean,
        pristine,
        thresholds=thresholds,
        min_areas=min_areas,
        image_thresholds=image_thresholds,
        max_far=0.01,
    )

    assert res["meets_budget"] is True
    assert res["best_image_threshold"] == 0.7
    assert res["best_threshold"] == 0.5
    assert res["best_min_area"] == 50
    assert res["best_recall"] == 0.65
    assert res["best_synth_far"] == 0.01
    assert res["best_pristine_far"] == 0.01


def test_zero_recall_reproduced_and_resolved():
    """Reproduce zero-recall bug (AUC ~0.8) and prove nonzero-recall point is found with extended grid."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    np.random.seed(42)
    # Synthetic scores yielding AUC around 0.80
    f_scores = np.clip(np.random.normal(0.65, 0.20, 100), 0, 1)
    c_scores = np.clip(np.random.normal(0.42, 0.20, 100), 0, 1)
    p_scores = np.clip(np.random.normal(0.40, 0.20, 100), 0, 1)

    auc = float(roc_auc_score([1] * 100 + [0] * 100, list(f_scores) + list(c_scores)))
    assert 0.75 <= auc <= 0.85, f"Expected AUC around 0.80, got {auc:.3f}"

    # Build records where:
    # 1. At coarse grid points (s in [0.0, 0.7], tau=0.5, area=50):
    #    clean has 2 flags (2% FAR > 1%), pristine has 2 flags (2% FAR > 1%) -> fails budget
    # 2. At coarse point (s=0.90, tau=0.70, area=50):
    #    clean has 0 flags, pristine has 0 flags, but forged has 0 hits -> recall = 0.0!
    # 3. At extended point (s=0.95, tau=0.50, area=50):
    #    clean has 1 flag (1% FAR <= 1%), pristine has 1 flag (1% FAR <= 1%), forged has 50 hits -> recall = 0.50!

    synth_clean = []
    pristine = []
    forged = []

    for i in range(100):
        c_prob = float(c_scores[i])
        p_prob = float(p_scores[i])
        f_prob = float(f_scores[i])

        # Clean: flags 2 images at s <= 0.90, but only 1 image at s >= 0.95
        c_rec = {
            "image_prob": c_prob,
            "flagged_0.5_50": i < 2,
            "flagged_0.7_50": False,  # 0 flags at tau=0.7
        }
        # Pristine: flags 2 images at s <= 0.90, but only 1 at s >= 0.95
        p_rec = {
            "image_prob": p_prob,
            "flagged_0.5_50": i < 2,
            "flagged_0.7_50": False,  # 0 flags at tau=0.7
        }
        # Forged: 50 forged images have hits at tau=0.5, 0 hits at tau=0.7
        f_rec = {
            "image_prob": f_prob,
            "hit_0.5_50": i < 50,
            "hit_0.7_50": False,      # 0 hits at tau=0.7
        }

        # Ensure image_prob condition: image 0 has prob 0.96 (so it passes s=0.95)
        # image 1 has prob 0.92 (so it passes s=0.90 but fails s=0.95)
        if i == 0:
            c_rec["image_prob"] = 0.96
            p_rec["image_prob"] = 0.96
        elif i == 1:
            c_rec["image_prob"] = 0.92
            p_rec["image_prob"] = 0.92

        # Forged items: 50 items have image_prob >= 0.95 and hit_0.5_50
        if i < 50:
            f_rec["image_prob"] = 0.96
            f_rec["hit_0.5_50"] = True

        synth_clean.append(c_rec)
        pristine.append(p_rec)
        forged.append(f_rec)

    # Test 1: If restricted to coarse grid (s up to 0.90 only), tau=0.7 area=50 has 0 FAR but 0 recall.
    # The new sweep MUST NOT claim meets_budget=True for 0 recall!
    res_coarse = sweep_thresholds(
        forged,
        synth_clean,
        pristine,
        thresholds=[0.5, 0.7],
        min_areas=[50],
        image_thresholds=[0.0, 0.5, 0.7, 0.9],
        max_far=0.01,
    )
    assert res_coarse["meets_budget"] is False, "A zero-recall point must NEVER be flagged as meets_budget=True"

    # Test 2: On extended grid (including s=0.95), the operating point with 50% recall and <= 1% FAR is found!
    res_extended = sweep_thresholds(
        forged,
        synth_clean,
        pristine,
        thresholds=[0.5, 0.7],
        min_areas=[50],
        image_thresholds=[0.0, 0.5, 0.7, 0.9, 0.95],
        max_far=0.01,
    )
    assert res_extended["meets_budget"] is True
    assert res_extended["best_image_threshold"] == 0.95
    assert res_extended["best_threshold"] == 0.5
    assert res_extended["best_min_area"] == 50
    assert res_extended["best_recall"] == 0.50
    assert res_extended["best_synth_far"] <= 0.01
    assert res_extended["best_pristine_far"] <= 0.01

    # Test 3: is_better_checkpoint rejects a zero-recall checkpoint over a nonzero-recall one
    zero_recall_ckpt = {"meets_budget": False, "recall": 0.0, "roc_auc": 0.82, "best_synth_far": 0.0, "best_pristine_far": 0.0}
    nonzero_recall_ckpt = {"meets_budget": True, "recall": 0.50, "roc_auc": 0.80, "best_synth_far": 0.01, "best_pristine_far": 0.01}
    assert is_better_checkpoint(nonzero_recall_ckpt, zero_recall_ckpt) is True
    assert is_better_checkpoint(zero_recall_ckpt, nonzero_recall_ckpt) is False


def test_wilson_ci_properties():
    """Verify Wilson 95% score interval properties and boundaries."""
    from detector.cnn.selection import wilson_ci

    # Zero sample size
    assert wilson_ci(0, 0) == (0.0, 0.0)

    # 0 positive out of 100
    lo, hi = wilson_ci(0, 100)
    assert lo == 0.0
    assert 0.03 < hi < 0.05

    # 100 positive out of 100
    lo, hi = wilson_ci(100, 100)
    assert 0.95 < lo < 0.97
    assert hi == 1.0

    # 50 positive out of 100 (symmetric around 0.5)
    lo, hi = wilson_ci(50, 100)
    assert abs((lo + hi) / 2.0 - 0.5) < 1e-5
    assert 0.39 < lo < 0.41
    assert 0.59 < hi < 0.61


def test_sweep_with_different_budgets():
    """Verify sweep_thresholds obeys different max_far budgets (1%, 2%, 3%)."""
    # 100 clean, 100 pristine, 100 forged
    # Configuration A: tau=0.5, FAR=1.5% on pristine, 0.8% on synth, recall=60%
    # Configuration B: tau=0.7, FAR=0.5% on pristine, 0.5% on synth, recall=40%
    synth_clean = [{"flag_0.5_50": i < 1, "flag_0.7_50": False} for i in range(100)]
    pristine = [{"flag_0.5_50": i < 2, "flag_0.7_50": False} for i in range(100)]
    forged = [{"hit_0.5_50": i < 60, "hit_0.7_50": i < 40} for i in range(100)]

    # At 1% budget: Config A has pristine FAR 2% > 1%, so Config B (recall 40%) is chosen
    res_1pct = sweep_thresholds(
        forged,
        synth_clean,
        pristine,
        thresholds=[0.5, 0.7],
        min_areas=[50],
        image_thresholds=[0.0],
        max_far=0.01,
    )
    assert res_1pct["meets_budget"] is True
    assert res_1pct["best_threshold"] == 0.7
    assert res_1pct["best_recall"] == 0.40

    # At 2% budget: Config A has pristine FAR 2% <= 2% and synth FAR 1% <= 2%, so Config A (recall 60%) is chosen
    res_2pct = sweep_thresholds(
        forged,
        synth_clean,
        pristine,
        thresholds=[0.5, 0.7],
        min_areas=[50],
        image_thresholds=[0.0],
        max_far=0.02,
    )
    assert res_2pct["meets_budget"] is True
    assert res_2pct["best_threshold"] == 0.5
    assert res_2pct["best_recall"] == 0.60


def test_cache_provenance_verification(tmp_path):
    """Verify that verify_cache_provenance correctly validates provenance keys and rejects mismatches."""
    from scripts.select_threshold import verify_cache_provenance

    # Create dummy files
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"dummy_weights" * 100)

    val_manifest = tmp_path / "val_manifest.csv"
    val_manifest.write_text("image_path,label\n")

    pristine_zip = tmp_path / "pristine.zip"
    pristine_zip.write_bytes(b"PK0304" * 10)

    taus = [0.3, 0.5, 0.7]
    areas = [0, 50, 100]

    valid_cache = {
        "ckpt_name": ckpt.name,
        "ckpt_size": ckpt.stat().st_size,
        "ckpt_mtime": ckpt.stat().st_mtime,
        "val_manifest_size": val_manifest.stat().st_size,
        "val_manifest_mtime": val_manifest.stat().st_mtime,
        "pristine_zip_size": pristine_zip.stat().st_size,
        "pristine_zip_mtime": pristine_zip.stat().st_mtime,
        "val_subset": -1,
        "pristine_subset": -1,
        "has_image_head": True,
        "model_type": "corr",
        "candidate_thresholds": taus,
        "candidate_areas": areas,
    }

    # 1. Matching provenance passes
    ok, msg = verify_cache_provenance(
        cache=valid_cache,
        ckpt_path=ckpt,
        val_manifest=val_manifest,
        pristine_zip=pristine_zip,
        val_subset=None,
        pristine_subset=None,
        model_type="corr",
        has_image_head=True,
        candidate_thresholds=taus,
        candidate_areas=areas,
    )
    assert ok is True
    assert "verified" in msg

    # 2. Checkpoint size mismatch fails
    bad_ckpt_cache = dict(valid_cache, ckpt_size=999999)
    ok, msg = verify_cache_provenance(
        cache=bad_ckpt_cache,
        ckpt_path=ckpt,
        val_manifest=val_manifest,
        pristine_zip=pristine_zip,
        val_subset=None,
        pristine_subset=None,
        model_type="corr",
        has_image_head=True,
        candidate_thresholds=taus,
        candidate_areas=areas,
    )
    assert ok is False
    assert "size mismatch" in msg

    # 3. Validation subset mismatch fails
    bad_sub_cache = dict(valid_cache, val_subset=50)
    ok, msg = verify_cache_provenance(
        cache=bad_sub_cache,
        ckpt_path=ckpt,
        val_manifest=val_manifest,
        pristine_zip=pristine_zip,
        val_subset=None,
        pristine_subset=None,
        model_type="corr",
        has_image_head=True,
        candidate_thresholds=taus,
        candidate_areas=areas,
    )
    assert ok is False
    assert "subset mismatch" in msg

    # 4. Architecture / image-head mismatch fails
    bad_head_cache = dict(valid_cache, has_image_head=False)
    ok, msg = verify_cache_provenance(
        cache=bad_head_cache,
        ckpt_path=ckpt,
        val_manifest=val_manifest,
        pristine_zip=pristine_zip,
        val_subset=None,
        pristine_subset=None,
        model_type="corr",
        has_image_head=True,
        candidate_thresholds=taus,
        candidate_areas=areas,
    )
    assert ok is False
    assert "Image head mismatch" in msg

    # 5. Missing provenance key fails (legacy cache)
    incomplete_cache = dict(valid_cache)
    del incomplete_cache["ckpt_size"]
    ok, msg = verify_cache_provenance(
        cache=incomplete_cache,
        ckpt_path=ckpt,
        val_manifest=val_manifest,
        pristine_zip=pristine_zip,
        val_subset=None,
        pristine_subset=None,
        model_type="corr",
        has_image_head=True,
        candidate_thresholds=taus,
        candidate_areas=areas,
    )
    assert ok is False
    assert "Missing provenance key" in msg



