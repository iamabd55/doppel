"""Pre-Step-C Diagnostic Checks (No Tuning, Report Only).

Computes:
1. SIFT tier anomaly breakdown on synthetic test split:
   - Per degradation tier: mean patch area, class mix (% Blot/Gel vs % Microscopy),
     transform mix, clean-twin FAR for Frozen DCT, Adapted DCT, Frozen SIFT.
2. Patch-size realism:
   - Synthetic train patch area distribution vs Real BioFors IDD box areas.
   - Median, quartiles (Q1, Q3), min, max, coarse ranges.
   - Fraction of synthetic patches smaller than real Q1.
3. Fusion on Real BioFors IDD test set:
   - OR and AND combinations of Frozen DCT, Adapted DCT, and Frozen SIFT.
   - Recall (overlap rule) and Clean False Alarm Rate with Wilson 95% CIs.
"""
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = 1.95996
    p_hat = k / n
    denominator = 1.0 + (z**2) / n
    centre_adjusted_probability = p_hat + (z**2) / (2 * n)
    adjusted_standard_deviation = math.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * n)) / n)
    lower = max(0.0, (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator)
    upper = min(1.0, (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator)
    return round(lower, 4), round(upper, 4)


def check_1_sift_tier_anomaly():
    test_manifest_path = REPO_ROOT / "data" / "synthetic" / "test_manifest.csv"
    bench_path = REPO_ROOT / "results" / "phase3_synthetic" / "synthetic_benchmark.csv"

    if not test_manifest_path.exists() or not bench_path.exists():
        print("Missing synthetic test data.")
        return

    m_df = pd.read_csv(test_manifest_path)
    b_df = pd.read_csv(bench_path)

    # Merge by sample_id
    df = pd.merge(m_df, b_df[["sample_id", "dct_flagged", "dct_pix_tp", "ad_dct_flagged", "ad_dct_pix_tp", "sift_flagged", "sift_pix_tp"]], on="sample_id")

    tiers = ["zero", "light", "moderate", "heavy"]

    print("=" * 80)
    print("CHECK 1: SIFT TIER ANOMALY AUDIT (SYNTHETIC TEST SPLIT)")
    print("=" * 80)

    for tier in tiers:
        t_sub = df[df["degradation_level"] == tier]
        f_sub = t_sub[t_sub["is_forged"]]
        c_sub = t_sub[~t_sub["is_forged"]]

        n_forged = len(f_sub)
        n_clean = len(c_sub)

        mean_area = f_sub["patch_area"].mean()
        median_area = f_sub["patch_area"].median()

        blot_cnt = (f_sub["class"] == "Blot/Gel").sum()
        micro_cnt = (f_sub["class"] == "Microscopy").sum()
        blot_pct = blot_cnt / n_forged * 100 if n_forged > 0 else 0
        micro_pct = micro_cnt / n_forged * 100 if n_forged > 0 else 0

        tf_counts = f_sub["transform_type"].value_counts().to_dict()

        # Detector Recalls (Overlap rule: pix_tp > 0)
        dct_rec = (f_sub["dct_pix_tp"] > 0).sum()
        ad_rec = (f_sub["ad_dct_pix_tp"] > 0).sum()
        sift_rec = (f_sub["sift_pix_tp"] > 0).sum()

        # Detector Clean Twin False Alarms
        dct_fa = c_sub["dct_flagged"].sum()
        ad_fa = c_sub["ad_dct_flagged"].sum()
        sift_fa = c_sub["sift_flagged"].sum()

        dct_far = dct_fa / n_clean * 100 if n_clean > 0 else 0
        ad_far = ad_fa / n_clean * 100 if n_clean > 0 else 0
        sift_far = sift_fa / n_clean * 100 if n_clean > 0 else 0

        print(f"\n--- Degradation Tier: {tier.upper()} (N={n_forged} forged, N={n_clean} clean twins) ---")
        print(f"  Mean Patch Area   : {mean_area:.1f} px (Median: {median_area:.1f} px)")
        print(f"  Class Mix         : Blot/Gel = {blot_cnt} ({blot_pct:.1f}%) | Microscopy = {micro_cnt} ({micro_pct:.1f}%)")
        print(f"  Transform Mix     : {tf_counts}")
        print(f"  Forged Recall     : Frozen DCT = {dct_rec}/{n_forged} ({dct_rec/n_forged*100:.1f}%) | Adapted DCT = {ad_rec}/{n_forged} ({ad_rec/n_forged*100:.1f}%) | SIFT = {sift_rec}/{n_forged} ({sift_rec/n_forged*100:.1f}%)")
        print(f"  Clean Twin FAR    : Frozen DCT = {dct_fa}/{n_clean} ({dct_far:.1f}%) | Adapted DCT = {ad_fa}/{n_clean} ({ad_far:.1f}%) | SIFT = {sift_fa}/{n_clean} ({sift_far:.1f}%)")


def check_2_patch_size_realism():
    print("\n" + "=" * 80)
    print("CHECK 2: PATCH-SIZE REALISM (SYNTHETIC TRAIN VS REAL BIOFORS IDD)")
    print("=" * 80)

    # 1. Synthetic Train patch areas
    synth_train_manifest = REPO_ROOT / "data" / "synthetic" / "train_manifest.csv"
    st_df = pd.read_csv(synth_train_manifest)
    synth_forged = st_df[st_df["is_forged"]].copy()
    synth_areas = synth_forged["patch_area"].values

    # 2. Real BioFors IDD box areas
    gt_path = REPO_ROOT / "data" / "biofors" / "annotations" / "idd_gt.json"
    with open(gt_path, "r", encoding="utf-8") as f:
        idd_gt = json.load(f)

    # Filter to Blot/Gel and Microscopy documents in test_idd
    test_manifest_path = REPO_ROOT / "data" / "biofors" / "splits" / "test_idd_manifest.csv"
    test_df = pd.read_csv(test_manifest_path)
    real_target_files = set(zip(test_df["doc_id"].astype(str), test_df["filename"].astype(str)))

    real_areas = []
    for doc_id, doc_files in idd_gt.items():
        for filename, boxes in doc_files.items():
            if (str(doc_id), str(filename)) in real_target_files:
                for b in boxes:
                    x1, y1, x2, y2 = b
                    area = (x2 - x1) * (y2 - y1)
                    real_areas.append(area)

    real_areas = np.array(real_areas)

    def get_stats(arr):
        return {
            "count": len(arr),
            "min": int(np.min(arr)),
            "q25": int(np.percentile(arr, 25)),
            "median": int(np.median(arr)),
            "q75": int(np.percentile(arr, 75)),
            "max": int(np.max(arr)),
            "mean": int(np.mean(arr)),
        }

    s_stats = get_stats(synth_areas)
    r_stats = get_stats(real_areas)

    print(f"Real BioFors IDD Boxes (N={r_stats['count']} boxes across 91 manipulated panels):")
    print(f"  Min: {r_stats['min']:,} px | Q1 (25%): {r_stats['q25']:,} px | Median: {r_stats['median']:,} px | Q3 (75%): {r_stats['q75']:,} px | Max: {r_stats['max']:,} px | Mean: {r_stats['mean']:,} px")

    print(f"\nSynthetic Train Patches (N={s_stats['count']} forged samples):")
    print(f"  Min: {s_stats['min']:,} px | Q1 (25%): {s_stats['q25']:,} px | Median: {s_stats['median']:,} px | Q3 (75%): {s_stats['q75']:,} px | Max: {s_stats['max']:,} px | Mean: {s_stats['mean']:,} px")

    # Fraction of synthetic patches smaller than real Q1
    real_q1 = r_stats["q25"]
    pct_below_real_q1 = float(np.mean(synth_areas < real_q1)) * 100
    print(f"\nSynthetic patches smaller than Real Q1 ({real_q1} px): {pct_below_real_q1:.1f}%")
    if pct_below_real_q1 > 25.0:
        print(f"  [FLAG] More than 25% ({pct_below_real_q1:.1f}%) of synthetic patches are smaller than real Q1.")
    else:
        print(f"  [PASS] Synthetic patch area distribution aligns with real quartiles.")


def check_3_fusion():
    print("\n" + "=" * 80)
    print("CHECK 3: FUSION PERFORMANCE ON REAL BIOFORS IDD TEST SET (2,286 PANELS)")
    print("=" * 80)

    bench_path = REPO_ROOT / "results" / "phase3_synthetic" / "real_biofors_benchmark.csv"
    if not bench_path.exists():
        print("Missing real BioFors benchmark results.")
        return

    df = pd.read_csv(bench_path)
    f_df = df[df["is_forged"]].copy()
    c_df = df[~df["is_forged"]].copy()

    n_forged = len(f_df)
    n_clean = len(c_df)

    # Boolean detection flags (overlap rule: pix_tp > 0)
    f_df["hit_df"] = (f_df["dct_pix_tp"] > 0)
    f_df["hit_da"] = (f_df["ad_dct_pix_tp"] > 0)
    f_df["hit_s"] = (f_df["sift_pix_tp"] > 0)

    c_df["fa_df"] = c_df["dct_flagged"]
    c_df["fa_da"] = c_df["ad_dct_flagged"]
    c_df["fa_s"] = c_df["sift_flagged"]

    combos = [
        ("Frozen DCT Only", f_df["hit_df"], c_df["fa_df"]),
        ("Adapted DCT Only", f_df["hit_da"], c_df["fa_da"]),
        ("Frozen SIFT Only", f_df["hit_s"], c_df["fa_s"]),
        ("Frozen DCT OR SIFT", f_df["hit_df"] | f_df["hit_s"], c_df["fa_df"] | c_df["fa_s"]),
        ("Frozen DCT AND SIFT", f_df["hit_df"] & f_df["hit_s"], c_df["fa_df"] & c_df["fa_s"]),
        ("Adapted DCT OR SIFT", f_df["hit_da"] | f_df["hit_s"], c_df["fa_da"] | c_df["fa_s"]),
        ("Adapted DCT AND SIFT", f_df["hit_da"] & f_df["hit_s"], c_df["fa_da"] & c_df["fa_s"]),
    ]

    print(f"{'Method / Fusion':<25} | {'Forged Recall (Overlap)':<30} | {'Clean False Alarm Rate':<30}")
    print("-" * 90)

    for name, hit_series, fa_series in combos:
        n_hits = hit_series.sum()
        n_fa = fa_series.sum()

        rec = n_hits / n_forged
        far = n_fa / n_clean

        rec_ci = wilson_ci(n_hits, n_forged)
        far_ci = wilson_ci(n_fa, n_clean)

        print(f"{name:<25} | {n_hits:2d}/{n_forged:2d} ({rec*100:5.1f}%) [{rec_ci[0]*100:4.1f}%, {rec_ci[1]*100:4.1f}%] | {n_fa:3d}/{n_clean:4d} ({far*100:4.2f}%) [{far_ci[0]*100:4.2f}%, {far_ci[1]*100:4.2f}%]")


if __name__ == "__main__":
    check_1_sift_tier_anomaly()
    check_2_patch_size_realism()
    check_3_fusion()
