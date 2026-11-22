"""Tuning Adapted DCT Thresholds on Synthetic Validation Set Only.

HARD RULE: Never use BioFors test or synthetic test for tuning.
All parameter search happens strictly on data/synthetic/val_manifest.csv.

Explores:
- min_votes: [20, 30, 50, 75, 100, 150]
- min_size: [36, 64, 100, 200, 500]

Measures:
- Forged Recall on 150 synthetic validation forgeries.
- Clean False Alarm Rate (FAR) on 150 synthetic validation clean controls.
- Selects optimal adapted operating point that boosts small-patch recall while keeping FAR <= 2.0%.
"""
import argparse
import itertools
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from detector.detect import detect


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune adapted DCT thresholds on synthetic validation set.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/synthetic/val_manifest.csv",
        help="Path to synthetic validation manifest CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/phase3_synthetic",
        help="Directory to save tuning results",
    )
    return parser.parse_args()


def evaluate_config(
    images_pack: List[Tuple[np.ndarray, np.ndarray, bool, int]],
    min_votes: int,
    min_size: int,
) -> Dict[str, Any]:
    tp_hits = 0
    fp_hits = 0
    total_forged = 0
    total_clean = 0

    for gray, gt_mask, is_forged, patch_area in images_pack:
        mask = detect(gray, Q=8, K=6, min_votes=min_votes, min_size=min_size)
        flagged = bool((mask > 0).any())

        if is_forged:
            total_forged += 1
            # Require ground-truth overlap
            if flagged and ((mask > 0) & (gt_mask > 0)).any():
                tp_hits += 1
        else:
            total_clean += 1
            if flagged:
                fp_hits += 1

    recall = tp_hits / total_forged if total_forged > 0 else 0.0
    far = fp_hits / total_clean if total_clean > 0 else 0.0

    return {
        "min_votes": min_votes,
        "min_size": min_size,
        "tp_hits": tp_hits,
        "total_forged": total_forged,
        "val_recall": round(recall, 4),
        "fp_hits": fp_hits,
        "total_clean": total_clean,
        "val_clean_far": round(far, 4),
    }


def main():
    args = parse_args()
    manifest_path = REPO_ROOT / args.manifest
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        print(f"ERROR: Validation manifest {manifest_path} not found.")
        sys.exit(1)

    df = pd.read_csv(manifest_path)
    data_dir = REPO_ROOT / "data"

    print("=" * 70)
    print("STEP 3.3.5: ADAPTED DCT TUNING ON SYNTHETIC VAL SET ONLY")
    print("=" * 70)
    print(f"Validation Manifest : {manifest_path}")
    print(f"Validation Samples  : {len(df)} ({sum(df['is_forged'])} forged, {sum(~df['is_forged'])} clean)")

    # Preload images and ground truth masks into memory for speed
    print("\nPreloading validation images...")
    images_pack = []
    for _, row in df.iterrows():
        f_path = data_dir / row["path_F"]
        a_path = data_dir / row["path_A"]
        img = cv2.imread(str(f_path), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(a_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if gt is None:
            gt = np.zeros(img.shape, dtype=np.uint8)
        images_pack.append((img, gt, bool(row["is_forged"]), int(row.get("patch_area", 0))))

    print(f"Loaded {len(images_pack)} validation images.")

    # Parameter grid
    min_votes_list = [20, 30, 50, 75, 100, 150]
    min_size_list = [36, 64, 100, 200, 500]

    results = []
    start_time = time.time()

    print("\nEvaluating parameter grid on validation set...")
    for votes, size in itertools.product(min_votes_list, min_size_list):
        res = evaluate_config(images_pack, votes, size)
        results.append(res)
        print(f"  votes={votes:3d}, size={size:3d} => Val Recall: {res['tp_hits']}/{res['total_forged']} ({res['val_recall']*100:5.1f}%) | Clean FAR: {res['fp_hits']}/{res['total_clean']} ({res['val_clean_far']*100:4.1f}%)")

    res_df = pd.DataFrame(results)

    # Save CSV
    csv_path = out_dir / "adapted_dct_val_tuning.csv"
    res_df.to_csv(csv_path, index=False)
    print(f"\nWrote validation tuning results to: {csv_path}")

    # Baseline vs Optimal Adapted
    frozen_row = res_df[(res_df["min_votes"] == 150) & (res_df["min_size"] == 500)].iloc[0]

    # Select candidates with FAR <= 2.0%, then highest recall
    safe_candidates = res_df[res_df["val_clean_far"] <= 0.02].sort_values(by=["val_recall", "val_clean_far"], ascending=[False, True])
    if len(safe_candidates) > 0:
        best_adapted = safe_candidates.iloc[0]
    else:
        best_adapted = res_df.sort_values(by="val_clean_far").iloc[0]

    # Write Markdown Report
    md = []
    md.append("# Adapted DCT Threshold Tuning on Synthetic Validation Split")
    md.append("")
    md.append("Evaluated parameter grid strictly on `data/synthetic/val_manifest.csv` (150 forged, 150 clean paired twins).")
    md.append("HARD RULE: Held-out BioFors test set and synthetic test set were untouched.")
    md.append("")
    md.append("## 1. Frozen Baseline vs Selected Adapted DCT")
    md.append("")
    md.append("| Setting | Parameters | Val Forged Recall | Val Clean False Alarms | Status |")
    md.append("|---|---|---|---|---|")
    md.append(f"| Frozen Baseline | min_votes=150, min_size=500 | {frozen_row['tp_hits']}/{frozen_row['total_forged']} ({frozen_row['val_recall']*100:.1f}%) | {frozen_row['fp_hits']}/{frozen_row['total_clean']} ({frozen_row['val_clean_far']*100:.1f}%) | Frozen Phase 1 |")
    md.append(f"| **Selected Adapted** | **min_votes={int(best_adapted['min_votes'])}, min_size={int(best_adapted['min_size'])}** | **{best_adapted['tp_hits']}/{best_adapted['total_forged']} ({best_adapted['val_recall']*100:.1f}%)** | **{best_adapted['fp_hits']}/{best_adapted['total_clean']} ({best_adapted['val_clean_far']*100:.1f}%)** | **Optimal on Val** |")
    md.append("")
    md.append("## 2. Complete Tuning Grid on Synthetic Validation")
    md.append("")
    md.append("| min_votes | min_size | Val Forged Recall | Val Clean False Alarm Rate | Clean FP / Total |")
    md.append("|---|---|---|---|---|")
    for _, r in res_df.sort_values(by=["min_votes", "min_size"]).iterrows():
        md.append(f"| {int(r['min_votes'])} | {int(r['min_size'])} | {r['val_recall']*100:.1f}% ({int(r['tp_hits'])}/{int(r['total_forged'])}) | {r['val_clean_far']*100:.1f}% | {int(r['fp_hits'])}/{int(r['total_clean'])} |")

    md_path = out_dir / "adapted_dct_val_tuning.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"Saved tuning summary to: {md_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
