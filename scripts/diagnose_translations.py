"""Diagnostic script: Investigating Translation Recall and Synthetic Dataset Integrity.

Addresses reviewer critique:
1. Zero-degradation test: 100 pure-translation samples (50 Blot/Gel, 50 Microscopy)
   with zero blur, zero noise, zero gamma shift, zero JPEG compression, and zero feathering.
2. Parameter sensitivity: compares frozen DCT (min_votes=150, min_size=500) against
   patch-size-aware DCT on small scientific strips.
3. Patch size and texture breakdown: measures recall for patch area < 500 px vs >= 500 px.
4. Dataset shortcut check: verifies whether clean negatives and forged samples differ
   in global image statistics (mean, std, Laplacian blur variance).
"""
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from detector.detect import detect
from detector.sift import detect_sift
from detector.synthetic import SyntheticCopyMoveGenerator


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score 95% confidence interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% confidence
    p_hat = k / n
    denominator = 1.0 + (z**2) / n
    centre_adjusted_probability = p_hat + (z**2) / (2 * n)
    adjusted_standard_deviation = math.sqrt(
        (p_hat * (1 - p_hat) + (z**2) / (4 * n)) / n
    )
    lower = max(0.0, (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator)
    upper = min(1.0, (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator)
    return round(lower, 4), round(upper, 4)


def run_translation_diagnostic(num_samples: int = 100):
    splits_dir = REPO_ROOT / "data" / "biofors" / "splits"
    images_root = REPO_ROOT / "data" / "biofors" / "images"
    val_manifest = splits_dir / "val_manifest.csv"

    if not val_manifest.exists():
        print("ERROR: val_manifest.csv not found.")
        return

    val_df = pd.read_csv(val_manifest)
    rng = np.random.default_rng(2026)

    # 1. Config with ZERO degradations and 100% pure translation
    zero_deg_cfg = {
        "generation": {
            "patch_min_area_ratio": 0.02,
            "patch_max_area_ratio": 0.25,
            "min_texture_std": 4.0,
            "low_texture_share": 0.0,
        },
        "transforms": {
            "translation_prob": 1.0,
            "cardinal_rot_flip_prob": 0.0,
            "arbitrary_rot_scale_prob": 0.0,
            "combination_prob": 0.0,
        },
        "degradations": {
            "blur_prob": 0.0,
            "noise_prob": 0.0,
            "brightness_gamma_prob": 0.0,
            "jpeg_recompression_prob": 0.0,
            "edge_feathering_prob": 0.0,
        },
    }

    generator = SyntheticCopyMoveGenerator(zero_deg_cfg)

    blot_pool = val_df[val_df["class"] == "Blot/Gel"].reset_index(drop=True)
    micro_pool = val_df[val_df["class"] == "Microscopy"].reset_index(drop=True)

    records = []
    print(f"\nGenerating {num_samples} zero-degradation pure-translation samples...")

    for i in range(num_samples):
        target_class = "Blot/Gel" if (i % 2 == 0) else "Microscopy"
        pool = blot_pool if target_class == "Blot/Gel" else micro_pool
        row = pool.iloc[int(rng.integers(0, len(pool)))]

        img_path = images_root / row["rel_path"]
        base_img = cv2.imread(str(img_path))
        if base_img is None:
            continue

        sample_meta = {
            "sample_id": f"diag_{i:04d}",
            "class": target_class,
            "is_forged": True,
        }

        forged_img, mask_M, mask_S, mask_A, meta = generator.generate(base_img, sample_meta, rng)
        gray = cv2.cvtColor(forged_img, cv2.COLOR_BGR2GRAY)

        sbox = meta["source_bbox"]
        tbox = meta["target_bbox"]
        pw = sbox[2] - sbox[0]
        ph = sbox[3] - sbox[1]
        patch_area = pw * ph
        patch_std = float(np.std(gray[sbox[1]:sbox[3], sbox[0]:sbox[2]]))

        # A. Frozen DCT Baseline (Q=8, K=6, min_votes=150, min_size=500)
        dct_frozen_mask = detect(gray, Q=8, K=6, min_votes=150, min_size=500)
        dct_frozen_hit = bool((dct_frozen_mask > 0).any() and ((dct_frozen_mask > 0) & (mask_A > 0)).any())

        # B. Small-Patch Adapted DCT (min_votes=30, min_size=64)
        dct_adapted_mask = detect(gray, Q=8, K=6, min_votes=30, min_size=64)
        dct_adapted_hit = bool((dct_adapted_mask > 0).any() and ((dct_adapted_mask > 0) & (mask_A > 0)).any())

        # C. Frozen SIFT (cfg8_best_combo)
        sift_mask, _, _, _ = detect_sift(
            gray,
            ratio_threshold=0.70,
            min_inliers=6,
            min_inlier_ratio=0.25,
            min_inliers_flip=10,
            filter_orientation=True,
            orientation_tolerance=20.0,
            use_convex_hull=True,
            mask_radius=12,
            check_flip=True,
        )
        sift_hit = bool((sift_mask > 0).any() and ((sift_mask > 0) & (mask_A > 0)).any())

        records.append({
            "sample_id": meta["sample_id"],
            "class": target_class,
            "img_h": forged_img.shape[0],
            "img_w": forged_img.shape[1],
            "patch_w": pw,
            "patch_h": ph,
            "patch_area": patch_area,
            "patch_std": round(patch_std, 2),
            "dct_frozen_hit": dct_frozen_hit,
            "dct_adapted_hit": dct_adapted_hit,
            "sift_hit": sift_hit,
        })

    df = pd.DataFrame(records)

    print("\n" + "=" * 70)
    print("DIAGNOSTIC TEST 1: ZERO-DEGRADATION TRANSLATION RESULTS")
    print("=" * 70)
    n = len(df)
    n_frozen = df["dct_frozen_hit"].sum()
    n_adapted = df["dct_adapted_hit"].sum()
    n_sift = df["sift_hit"].sum()

    f_ci = wilson_ci(n_frozen, n)
    a_ci = wilson_ci(n_adapted, n)
    s_ci = wilson_ci(n_sift, n)

    print(f"Overall Clean Translation Recall (N={n}):")
    print(f"  Frozen DCT (min_votes=150, min_size=500): {n_frozen}/{n} ({n_frozen/n*100:.1f}%) [95% CI: {f_ci[0]*100:.1f}% - {f_ci[1]*100:.1f}%]")
    print(f"  Adapted DCT (min_votes=30, min_size=64) : {n_adapted}/{n} ({n_adapted/n*100:.1f}%) [95% CI: {a_ci[0]*100:.1f}% - {a_ci[1]*100:.1f}%]")
    print(f"  Frozen SIFT (cfg8_best_combo)           : {n_sift}/{n} ({n_sift/n*100:.1f}%) [95% CI: {s_ci[0]*100:.1f}% - {s_ci[1]*100:.1f}%]")

    # Breakdown by Patch Area: < 500 px vs >= 500 px
    small_df = df[df["patch_area"] < 500]
    large_df = df[df["patch_area"] >= 500]

    print(f"\nBreakdown by Patch Size:")
    print(f"  Small patches (< 500 px, N={len(small_df)}):")
    print(f"    Frozen DCT  : {small_df['dct_frozen_hit'].sum()}/{len(small_df)} ({small_df['dct_frozen_hit'].mean()*100:.1f}%)")
    print(f"    Adapted DCT : {small_df['dct_adapted_hit'].sum()}/{len(small_df)} ({small_df['dct_adapted_hit'].mean()*100:.1f}%)")
    print(f"    Frozen SIFT : {small_df['sift_hit'].sum()}/{len(small_df)} ({small_df['sift_hit'].mean()*100:.1f}%)")

    print(f"  Large patches (>= 500 px, N={len(large_df)}):")
    print(f"    Frozen DCT  : {large_df['dct_frozen_hit'].sum()}/{len(large_df)} ({large_df['dct_frozen_hit'].mean()*100:.1f}%)")
    print(f"    Adapted DCT : {large_df['dct_adapted_hit'].sum()}/{len(large_df)} ({large_df['dct_adapted_hit'].mean()*100:.1f}%)")
    print(f"    Frozen SIFT : {large_df['sift_hit'].sum()}/{len(large_df)} ({large_df['sift_hit'].mean()*100:.1f}%)")

    # Breakdown by Class: Blot/Gel vs Microscopy
    print(f"\nBreakdown by Scientific Class:")
    for cname in ["Blot/Gel", "Microscopy"]:
        cdf = df[df["class"] == cname]
        print(f"  {cname} (N={len(cdf)}):")
        print(f"    Frozen DCT  : {cdf['dct_frozen_hit'].sum()}/{len(cdf)} ({cdf['dct_frozen_hit'].mean()*100:.1f}%)")
        print(f"    Adapted DCT : {cdf['dct_adapted_hit'].sum()}/{len(cdf)} ({cdf['dct_adapted_hit'].mean()*100:.1f}%)")
        print(f"    Frozen SIFT : {cdf['sift_hit'].sum()}/{len(cdf)} ({cdf['sift_hit'].mean()*100:.1f}%)")


def run_shortcut_and_leakage_check():
    """Verify whether clean vs forged images exhibit artificial statistical shortcuts."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC TEST 2: DATASET SHORTCUT & LEAKAGE CHECK")
    print("=" * 70)

    train_manifest = REPO_ROOT / "data" / "synthetic" / "train_manifest.csv"
    if not train_manifest.exists():
        print("ERROR: train_manifest.csv not found.")
        return

    df = pd.read_csv(train_manifest)
    data_dir = REPO_ROOT / "data"

    clean_samples = df[~df["is_forged"]].head(200)
    forged_samples = df[df["is_forged"]].head(200)

    def get_stats(sub_df):
        means, stds, laplacians = [], [], []
        for _, row in sub_df.iterrows():
            img_path = data_dir / row["path_F"]
            im = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if im is not None:
                means.append(float(np.mean(im)))
                stds.append(float(np.std(im)))
                laplacians.append(float(cv2.Laplacian(im, cv2.CV_64F).var()))
        return np.mean(means), np.mean(stds), np.mean(laplacians)

    c_mean, c_std, c_lap = get_stats(clean_samples)
    f_mean, f_std, f_lap = get_stats(forged_samples)

    print(f"Comparison of Global Image Statistics (200 Clean vs 200 Forged):")
    print(f"  Mean Intensity  : Clean = {c_mean:.2f} | Forged = {f_mean:.2f} (Delta = {abs(c_mean - f_mean):.2f})")
    print(f"  Std Deviation   : Clean = {c_std:.2f} | Forged = {f_std:.2f} (Delta = {abs(c_std - f_std):.2f})")
    print(f"  Laplacian Var   : Clean = {c_lap:.2f} | Forged = {f_lap:.2f} (Delta = {abs(c_lap - f_lap):.2f})")

    # Document Leakage verification
    train_docs = set(df[df["split"] == "train"]["doc_id"].unique())
    val_docs = set(df[df["split"] == "val"]["doc_id"].unique())
    test_manifest = REPO_ROOT / "data" / "synthetic" / "test_manifest.csv"
    test_df = pd.read_csv(test_manifest)
    test_docs = set(test_df["doc_id"].unique())

    idd_path = REPO_ROOT / "data" / "biofors" / "splits" / "test_idd_manifest.csv"
    idd_df = pd.read_csv(idd_path)
    real_test_docs = set(idd_df["doc_id"].unique())

    print(f"\nDocument Leakage Audit:")
    print(f"  Train / Val Synthetic Overlap        : {len(train_docs.intersection(val_docs))} docs")
    print(f"  Train / Test Synthetic Overlap       : {len(train_docs.intersection(test_docs))} docs")
    print(f"  Synthetic (All) / Real BioFors IDD   : {len((train_docs | val_docs | test_docs).intersection(real_test_docs))} docs")
    print("=" * 70)


if __name__ == "__main__":
    run_translation_diagnostic(100)
    run_shortcut_and_leakage_check()
