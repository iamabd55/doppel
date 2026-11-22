"""Scientific Benchmark Runner for Phase 3 (Rigorous Evaluation Protocol).

Evaluates three detectors:
1. Frozen DCT baseline (min_votes=150, min_size=500)
2. Adapted DCT (min_votes=75, min_size=64, locked on synthetic VAL only)
3. Frozen SIFT baseline (cfg8_best_combo)

Evaluates on:
1. Synthetic Scientific Test Split (300 paired-twin images: 150 forged, 150 clean negatives)
2. Real BioFors IDD Test Split (2,286 panels: 91 forged, 2,195 clean negatives; Blot/Gel and Microscopy separately)
3. Real Pristine Train Check (sample of up to 1,000 panels from documents unused by generator)

Reporting standards:
- Image recall with overlap rule (pix_tp > 0) and any-flagged rule (pred_mask > 0) side-by-side.
- Wilson score 95% confidence intervals on every recall and false alarm rate.
- Stratified breakdowns on synthetic test:
  * Degradation tier (zero, light, moderate, heavy)
  * Patch area bucket (<500 px, 500-1500 px, >1500 px)
  * Transformation type (translation, 90/180 rotation, flip, arbitrary rotation/scale, combination)
- Low confidence warning (* N < 30) on small sample sizes.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from detector.detect import detect
from detector.sift import detect_sift


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DCT and SIFT on scientific datasets.")
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "biofors", "pristine_train", "both", "summary_only"],
        default="both",
        help="Which dataset to benchmark (default: both)",
    )
    parser.add_argument(
        "--detector",
        choices=["dct", "sift", "both"],
        default="both",
        help="Which detector to evaluate (default: both)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of samples",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/synthetic.yaml",
        help="Path to synthetic configuration file",
    )
    parser.add_argument(
        "--adapted-config",
        type=str,
        default="configs/dct_adapted.yaml",
        help="Path to adapted DCT configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/phase3_synthetic",
        help="Directory to save benchmark results",
    )
    return parser.parse_args()


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score 95% confidence interval for proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    z = 1.95996
    p_hat = k / n
    denominator = 1.0 + (z**2) / n
    centre_adjusted_probability = p_hat + (z**2) / (2 * n)
    adjusted_standard_deviation = math.sqrt(
        (p_hat * (1 - p_hat) + (z**2) / (4 * n)) / n
    )
    lower = max(0.0, (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator)
    upper = min(1.0, (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator)
    return round(lower, 4), round(upper, 4)


def compute_mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    """Compute Matthews Correlation Coefficient safely."""
    if (tp + fp == 0) or (tp + fn == 0) or (tn + fp == 0) or (tn + fn == 0):
        return 0.0
    num = (tp * tn) - (fp * fn)
    if num == 0:
        return 0.0
    sign = 1.0 if num > 0 else -1.0
    try:
        den = 0.5 * (
            math.log(tp + fp) + math.log(tp + fn) + math.log(tn + fp) + math.log(tn + fn)
        )
        log_abs_mcc = math.log(abs(num)) - den
        return sign * math.exp(log_abs_mcc)
    except (ValueError, OverflowError):
        den_direct = math.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
        return float(num) / den_direct if den_direct > 0 else 0.0


def compute_metrics(records: Any, prefix: str) -> Dict[str, Any]:
    """Compute pooled pixel and image metrics reporting overlap and any-flagged rules."""
    if isinstance(records, pd.DataFrame):
        records = records.to_dict("records")

    img_tp_overlap = 0
    img_tp_any = 0
    img_fn_overlap = 0
    img_fn_any = 0
    img_fp = 0
    img_tn = 0

    pix_tp = 0
    pix_fp = 0
    pix_tn = 0
    pix_fn = 0

    for r in records:
        gt_flag = bool(r["is_forged"])
        flagged = bool(r.get(f"{prefix}_flagged", False))
        p_tp = int(r.get(f"{prefix}_pix_tp", 0))

        is_overlap_hit = flagged and (p_tp > 0)
        is_any_hit = flagged

        if gt_flag:
            if is_overlap_hit:
                img_tp_overlap += 1
            else:
                img_fn_overlap += 1

            if is_any_hit:
                img_tp_any += 1
            else:
                img_fn_any += 1
        else:
            if flagged:
                img_fp += 1
            else:
                img_tn += 1

        pix_tp += int(r.get(f"{prefix}_pix_tp", 0))
        pix_fp += int(r.get(f"{prefix}_pix_fp", 0))
        pix_tn += int(r.get(f"{prefix}_pix_tn", 0))
        pix_fn += int(r.get(f"{prefix}_pix_fn", 0))

    total_forged = img_tp_overlap + img_fn_overlap
    total_clean = img_fp + img_tn

    rec_overlap = img_tp_overlap / total_forged if total_forged > 0 else 0.0
    rec_any = img_tp_any / total_forged if total_forged > 0 else 0.0
    far = img_fp / total_clean if total_clean > 0 else 0.0

    ci_rec_overlap = wilson_ci(img_tp_overlap, total_forged)
    ci_rec_any = wilson_ci(img_tp_any, total_forged)
    ci_far = wilson_ci(img_fp, total_clean)

    pix_prec = pix_tp / (pix_tp + pix_fp) if (pix_tp + pix_fp) > 0 else 0.0
    pix_rec = pix_tp / (pix_tp + pix_fn) if (pix_tp + pix_fn) > 0 else 0.0
    pix_f1 = (2 * pix_prec * pix_rec / (pix_prec + pix_rec)) if (pix_prec + pix_rec) > 0 else 0.0
    pix_mcc = compute_mcc(pix_tp, pix_tn, pix_fp, pix_fn)

    return {
        "total_forged": total_forged,
        "total_clean": total_clean,
        "tp_overlap": img_tp_overlap,
        "tp_any": img_tp_any,
        "fp": img_fp,
        "rec_overlap": round(rec_overlap, 4),
        "ci_rec_overlap": ci_rec_overlap,
        "rec_any": round(rec_any, 4),
        "ci_rec_any": ci_rec_any,
        "far": round(far, 4),
        "ci_far": ci_far,
        "pix_f1": round(pix_f1, 4),
        "pix_mcc": round(pix_mcc, 4),
    }


def evaluate_pair(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[int, int, int, int, float, float, float]:
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    tp = int(((pred_bin == 1) & (gt_bin == 1)).sum())
    fp = int(((pred_bin == 1) & (gt_bin == 0)).sum())
    fn = int(((pred_bin == 0) & (gt_bin == 1)).sum())
    tn = int(((pred_bin == 0) & (gt_bin == 0)).sum())

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return tp, fp, fn, tn, prec, rec, f1


def categorize_transformation(row: pd.Series) -> str:
    tf = str(row.get("transform_type", "none"))
    flip = str(row.get("flip_type", "none"))
    rot = float(row.get("rotation_deg", 0.0))

    if tf == "translation":
        return "translation"
    elif tf == "cardinal_flip":
        if flip in ["H-Flip", "V-Flip"]:
            return "flip"
        elif round(rot) in [90, 180, 270]:
            return "90/180 rotation"
        return "flip"
    elif tf == "arbitrary":
        return "arbitrary rotation/scale"
    elif tf == "combination":
        return "combination"
    return "other"


def run_benchmark_synthetic(
    manifest_path: Path,
    sift_params: Dict[str, Any],
    eval_dct: bool,
    eval_sift: bool,
    adapted_dct_params: Dict[str, Any],
    limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    print("\n" + "=" * 70)
    print("BENCHMARK 1: SYNTHETIC TEST SET (PAIRED TWINS)")
    print("=" * 70)

    df = pd.read_csv(manifest_path)
    if limit:
        df = df.head(limit)

    ad_votes = adapted_dct_params.get("min_votes", 75)
    ad_size = adapted_dct_params.get("min_size", 64)

    records = []
    start_time = time.time()

    for idx, row in df.iterrows():
        f_path = REPO_ROOT / "data" / row["path_F"]
        a_path = REPO_ROOT / "data" / row["path_A"]

        img_bgr = cv2.imread(str(f_path))
        if img_bgr is None:
            continue
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gt_mask = cv2.imread(str(a_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            gt_mask = np.zeros(gray.shape, dtype=np.uint8)

        record = {
            "sample_id": row["sample_id"],
            "class": row["class"],
            "is_forged": bool(row["is_forged"]),
            "degradation_level": str(row.get("degradation_level", "unknown")),
            "transform_category": categorize_transformation(row),
            "patch_area": int(row.get("patch_area", 0)),
            "low_texture": bool(row.get("low_texture", False)),
        }

        # 1. Frozen DCT (votes=150, size=500)
        if eval_dct:
            t0 = time.time()
            dct_mask = detect(gray, Q=8, K=6, min_votes=150, min_size=500)
            tp, fp, fn, tn, p, r, f1 = evaluate_pair(dct_mask, gt_mask)
            record.update({
                "dct_flagged": bool((dct_mask > 0).any()),
                "dct_pix_tp": tp,
                "dct_pix_fp": fp,
                "dct_pix_fn": fn,
                "dct_pix_tn": tn,
                "dct_f1": round(f1, 4),
            })

            # 2. Adapted DCT (votes=75, size=64)
            ad_mask = detect(gray, Q=8, K=6, min_votes=ad_votes, min_size=ad_size)
            tp, fp, fn, tn, p, r, f1 = evaluate_pair(ad_mask, gt_mask)
            record.update({
                "ad_dct_flagged": bool((ad_mask > 0).any()),
                "ad_dct_pix_tp": tp,
                "ad_dct_pix_fp": fp,
                "ad_dct_pix_fn": fn,
                "ad_dct_pix_tn": tn,
                "ad_dct_f1": round(f1, 4),
            })

        # 3. Frozen SIFT (cfg8_best_combo)
        if eval_sift:
            sift_mask, _, _, _ = detect_sift(
                gray,
                ratio_threshold=sift_params.get("ratio_threshold", 0.70),
                min_inliers=sift_params.get("min_inliers", 6),
                min_inlier_ratio=sift_params.get("min_inlier_ratio", 0.25),
                min_inliers_flip=sift_params.get("min_inliers_flip", 10),
                filter_orientation=sift_params.get("filter_orientation", True),
                orientation_tolerance=sift_params.get("orientation_tolerance", 20.0),
                use_convex_hull=sift_params.get("use_convex_hull", True),
                mask_radius=sift_params.get("mask_radius", 12),
                check_flip=True,
            )
            tp, fp, fn, tn, p, r, f1 = evaluate_pair(sift_mask, gt_mask)
            record.update({
                "sift_flagged": bool((sift_mask > 0).any()),
                "sift_pix_tp": tp,
                "sift_pix_fp": fp,
                "sift_pix_fn": fn,
                "sift_pix_tn": tn,
                "sift_f1": round(f1, 4),
            })

        records.append(record)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(df):
            elapsed = time.time() - start_time
            print(f"  Processed [{idx + 1}/{len(df)}] synthetic samples ({elapsed:.1f}s)")

    results_df = pd.DataFrame(records)
    dct_metrics = compute_metrics(records, "dct") if eval_dct else {}
    ad_dct_metrics = compute_metrics(records, "ad_dct") if eval_dct else {}
    sift_metrics = compute_metrics(records, "sift") if eval_sift else {}

    return results_df, dct_metrics, ad_dct_metrics, sift_metrics


def run_benchmark_biofors(
    manifest_path: Path,
    images_dir: Path,
    idd_gt_path: Path,
    sift_params: Dict[str, Any],
    eval_dct: bool,
    eval_sift: bool,
    adapted_dct_params: Dict[str, Any],
    limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    print("\n" + "=" * 70)
    print("BENCHMARK 2: REAL BIOFORS IDD TEST SET (2,286 PANELS)")
    print("=" * 70)

    manifest_df = pd.read_csv(manifest_path)
    if limit:
        manifest_df = manifest_df.head(limit)

    with open(idd_gt_path, "r", encoding="utf-8") as f:
        idd_gt = json.load(f)

    ad_votes = adapted_dct_params.get("min_votes", 75)
    ad_size = adapted_dct_params.get("min_size", 64)

    records = []
    start_time = time.time()

    for idx, row in manifest_df.iterrows():
        doc_id = str(row["doc_id"])
        filename = str(row["filename"])
        rel_path = str(row["rel_path"])
        cls_name = str(row["class"])
        is_manipulated = bool(row["is_manipulated"])

        img_path = images_dir / rel_path
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        gt_mask = np.zeros((h, w), dtype=np.uint8)
        if doc_id in idd_gt and filename in idd_gt[doc_id]:
            boxes = idd_gt[doc_id][filename]
            for b in boxes:
                x1, y1, x2, y2 = b
                gt_mask[y1:y2, x1:x2] = 255

        record = {
            "doc_id": doc_id,
            "filename": filename,
            "class": cls_name,
            "is_forged": is_manipulated,
            "box_count": int(row.get("box_count", 0)),
        }

        # 1. Frozen DCT (votes=150, size=500)
        if eval_dct:
            dct_mask = detect(gray, Q=8, K=6, min_votes=150, min_size=500)
            tp, fp, fn, tn, p, r, f1 = evaluate_pair(dct_mask, gt_mask)
            record.update({
                "dct_flagged": bool((dct_mask > 0).any()),
                "dct_pix_tp": tp,
                "dct_pix_fp": fp,
                "dct_pix_fn": fn,
                "dct_pix_tn": tn,
                "dct_f1": round(f1, 4),
            })

            # 2. Adapted DCT (votes=75, size=64)
            ad_mask = detect(gray, Q=8, K=6, min_votes=ad_votes, min_size=ad_size)
            tp, fp, fn, tn, p, r, f1 = evaluate_pair(ad_mask, gt_mask)
            record.update({
                "ad_dct_flagged": bool((ad_mask > 0).any()),
                "ad_dct_pix_tp": tp,
                "ad_dct_pix_fp": fp,
                "ad_dct_pix_fn": fn,
                "ad_dct_pix_tn": tn,
                "ad_dct_f1": round(f1, 4),
            })

        # 3. Frozen SIFT (cfg8_best_combo)
        if eval_sift:
            sift_mask, _, _, _ = detect_sift(
                gray,
                ratio_threshold=sift_params.get("ratio_threshold", 0.70),
                min_inliers=sift_params.get("min_inliers", 6),
                min_inlier_ratio=sift_params.get("min_inlier_ratio", 0.25),
                min_inliers_flip=sift_params.get("min_inliers_flip", 10),
                filter_orientation=sift_params.get("filter_orientation", True),
                orientation_tolerance=sift_params.get("orientation_tolerance", 20.0),
                use_convex_hull=sift_params.get("use_convex_hull", True),
                mask_radius=sift_params.get("mask_radius", 12),
                check_flip=True,
            )
            tp, fp, fn, tn, p, r, f1 = evaluate_pair(sift_mask, gt_mask)
            record.update({
                "sift_flagged": bool((sift_mask > 0).any()),
                "sift_pix_tp": tp,
                "sift_pix_fp": fp,
                "sift_pix_fn": fn,
                "sift_pix_tn": tn,
                "sift_f1": round(f1, 4),
            })

        records.append(record)

        if (idx + 1) % 250 == 0 or (idx + 1) == len(manifest_df):
            elapsed = time.time() - start_time
            rate = (idx + 1) / max(0.01, elapsed)
            print(f"  Processed [{idx + 1}/{len(manifest_df)}] real panels ({rate:.1f} panels/s)")

    results_df = pd.DataFrame(records)
    dct_metrics = compute_metrics(records, "dct") if eval_dct else {}
    ad_dct_metrics = compute_metrics(records, "ad_dct") if eval_dct else {}
    sift_metrics = compute_metrics(records, "sift") if eval_sift else {}

    return results_df, dct_metrics, ad_dct_metrics, sift_metrics


def run_pristine_train_audit(
    images_dir: Path,
    sift_params: Dict[str, Any],
    eval_dct: bool,
    eval_sift: bool,
    adapted_dct_params: Dict[str, Any],
    sample_size: int = 1000,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    print("\n" + "=" * 70)
    print("BENCHMARK 3: REAL PRISTINE TRAIN CHECK (UNUSED DOCUMENTS)")
    print("=" * 70)

    train_manifest_path = REPO_ROOT / "data" / "biofors" / "splits" / "train_manifest.csv"
    synthetic_train_manifest_path = REPO_ROOT / "data" / "synthetic" / "train_manifest.csv"

    if not train_manifest_path.exists() or not synthetic_train_manifest_path.exists():
        print("Skipping pristine train audit: manifests not found.")
        return pd.DataFrame(), {}

    train_df = pd.read_csv(train_manifest_path)
    synth_df = pd.read_csv(synthetic_train_manifest_path)

    used_docs = set(synth_df["doc_id"].astype(str).unique())
    unused_train_df = train_df[~train_df["doc_id"].astype(str).isin(used_docs)].reset_index(drop=True)

    print(f"Total train panels in unused documents : {len(unused_train_df)} across {unused_train_df['doc_id'].nunique()} docs")

    # Sample up to 1000 panels balanced across Blot/Gel and Microscopy
    blots = unused_train_df[unused_train_df["class"] == "Blot/Gel"]
    micros = unused_train_df[unused_train_df["class"] == "Microscopy"]

    n_half = sample_size // 2
    sampled_blots = blots.sample(min(n_half, len(blots)), random_state=42)
    sampled_micros = micros.sample(min(n_half, len(micros)), random_state=42)
    sample_df = pd.concat([sampled_blots, sampled_micros]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"Sampled {len(sample_df)} pristine panels ({len(sampled_blots)} Blot/Gel, {len(sampled_micros)} Microscopy)")

    ad_votes = adapted_dct_params.get("min_votes", 75)
    ad_size = adapted_dct_params.get("min_size", 64)

    records = []
    start_time = time.time()

    for idx, row in sample_df.iterrows():
        img_path = images_dir / row["rel_path"]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        record = {
            "doc_id": row["doc_id"],
            "filename": row["filename"],
            "class": row["class"],
            "is_forged": False,
        }

        if eval_dct:
            dct_mask = detect(gray, Q=8, K=6, min_votes=150, min_size=500)
            ad_mask = detect(gray, Q=8, K=6, min_votes=ad_votes, min_size=ad_size)
            record["dct_flagged"] = bool((dct_mask > 0).any())
            record["ad_dct_flagged"] = bool((ad_mask > 0).any())

        if eval_sift:
            sift_mask, _, _, _ = detect_sift(
                gray,
                ratio_threshold=sift_params.get("ratio_threshold", 0.70),
                min_inliers=sift_params.get("min_inliers", 6),
                min_inlier_ratio=sift_params.get("min_inlier_ratio", 0.25),
                min_inliers_flip=sift_params.get("min_inliers_flip", 10),
                filter_orientation=sift_params.get("filter_orientation", True),
                orientation_tolerance=sift_params.get("orientation_tolerance", 20.0),
                use_convex_hull=sift_params.get("use_convex_hull", True),
                mask_radius=sift_params.get("mask_radius", 12),
                check_flip=True,
            )
            record["sift_flagged"] = bool((sift_mask > 0).any())

        records.append(record)

        if (idx + 1) % 250 == 0 or (idx + 1) == len(sample_df):
            elapsed = time.time() - start_time
            rate = (idx + 1) / max(0.01, elapsed)
            print(f"  Processed [{idx + 1}/{len(sample_df)}] pristine train panels ({rate:.1f} panels/s)")

    results_df = pd.DataFrame(records)

    pristine_stats = {}
    for prefix, name in [("dct", "Frozen DCT"), ("ad_dct", "Adapted DCT"), ("sift", "Frozen SIFT")]:
        if f"{prefix}_flagged" in results_df:
            fp = results_df[f"{prefix}_flagged"].sum()
            n_tot = len(results_df)
            far = fp / n_tot if n_tot > 0 else 0.0
            ci = wilson_ci(fp, n_tot)
            pristine_stats[prefix] = {
                "name": name,
                "fp": int(fp),
                "total": n_tot,
                "far": round(far, 4),
                "ci": ci,
            }

    return results_df, pristine_stats


def generate_summary_markdown(
    synth_dct: Dict[str, Any],
    synth_ad_dct: Dict[str, Any],
    synth_sift: Dict[str, Any],
    real_dct: Dict[str, Any],
    real_ad_dct: Dict[str, Any],
    real_sift: Dict[str, Any],
    synth_df: pd.DataFrame,
    real_df: pd.DataFrame,
    pristine_stats: Dict[str, Any],
    out_path: Path,
):
    md = []
    md.append("# Phase 3: Scientific Benchmark Summary")
    md.append("")
    md.append("Comparative evaluation of Frozen DCT (votes=150, min_size=500), Adapted DCT (votes=75, min_size=64, tuned strictly on synthetic VAL only), and Frozen SIFT (cfg8_best_combo).")
    md.append("")
    md.append("**Evaluation Rules:**")
    md.append("- Overlap Rule: Image detection strictly requires true overlap with ground truth (`pix_tp > 0`).")
    md.append("- Any-Flagged Rule: Traditional rule counting an image as detected if any pixel is flagged (`pred_mask > 0`).")
    md.append("- Confidence Intervals: 95% Wilson score confidence interval `[lower, upper]` reported for all rates.")
    md.append("- Note: Cells marked with `*` indicate sample size N < 30 (low statistical confidence).")
    md.append("")

    # 1. Executive Comparison Table
    md.append("## 1. Executive Comparison Table")
    md.append("")
    md.append("| Benchmark Dataset | Detector | Image Recall (Overlap: pix_tp > 0) [95% CI] | Image Recall (Any Flagged) [95% CI] | Pixel F1 | Pixel MCC | Clean False Alarm Rate [95% CI] |")
    md.append("|---|---|---|---|---|---|---|")

    def format_row(dname, mname, m):
        if not m:
            return
        r_ci = m["ci_rec_overlap"]
        a_ci = m["ci_rec_any"]
        f_ci = m["ci_far"]
        md.append(
            f"| {dname} | {mname} | "
            f"{m['tp_overlap']}/{m['total_forged']} ({m['rec_overlap']*100:.1f}%) [{r_ci[0]*100:.1f}%, {r_ci[1]*100:.1f}%] | "
            f"{m['tp_any']}/{m['total_forged']} ({m['rec_any']*100:.1f}%) [{a_ci[0]*100:.1f}%, {a_ci[1]*100:.1f}%] | "
            f"{m['pix_f1']:.4f} | {m['pix_mcc']:.4f} | "
            f"{m['fp']}/{m['total_clean']} ({m['far']*100:.1f}%) [{f_ci[0]*100:.1f}%, {f_ci[1]*100:.1f}%] |"
        )

    format_row("Synthetic Test (300)", "Frozen DCT", synth_dct)
    format_row("Synthetic Test (300)", "Adapted DCT (val-tuned)", synth_ad_dct)
    format_row("Synthetic Test (300)", "Frozen SIFT", synth_sift)
    format_row("Real BioFors IDD (2,286)", "Frozen DCT", real_dct)
    format_row("Real BioFors IDD (2,286)", "Adapted DCT (val-tuned)", real_ad_dct)
    format_row("Real BioFors IDD (2,286)", "Frozen SIFT", real_sift)

    # 2. Real BioFors IDD Test (Blot/Gel vs Microscopy)
    md.append("")
    md.append("## 2. Real BioFors IDD Test (Blot/Gel and Microscopy Separately)")
    md.append("")
    md.append("| Class | Manipulated Panels | Clean Panels | Detector | Image Recall (Overlap) [95% CI] | Image Recall (Any Flagged) [95% CI] | Clean False Alarm Rate [95% CI] |")
    md.append("|---|---|---|---|---|---|---|")

    if not real_df.empty:
        for cname in ["Blot/Gel", "Microscopy"]:
            cdf = real_df[real_df["class"] == cname]
            f_sub = cdf[cdf["is_forged"]]
            c_sub = cdf[~cdf["is_forged"]]
            nf = len(f_sub)
            nc = len(c_sub)
            low_conf = " *" if nf < 30 else ""

            for pfx, dlabel in [("dct", "Frozen DCT"), ("ad_dct", "Adapted DCT"), ("sift", "Frozen SIFT")]:
                if f"{pfx}_flagged" in f_sub:
                    tp_ov = (f_sub[f"{pfx}_pix_tp"] > 0).sum()
                    tp_any = f_sub[f"{pfx}_flagged"].sum()
                    fp = c_sub[f"{pfx}_flagged"].sum()

                    r_ov = tp_ov / nf if nf > 0 else 0.0
                    r_any = tp_any / nf if nf > 0 else 0.0
                    far = fp / nc if nc > 0 else 0.0

                    r_ov_ci = wilson_ci(tp_ov, nf)
                    r_any_ci = wilson_ci(tp_any, nf)
                    far_ci = wilson_ci(fp, nc)

                    md.append(
                        f"| {cname}{low_conf} | {nf} | {nc} | {dlabel} | "
                        f"{tp_ov}/{nf} ({r_ov*100:.1f}%) [{r_ov_ci[0]*100:.1f}%, {r_ov_ci[1]*100:.1f}%] | "
                        f"{tp_any}/{nf} ({r_any*100:.1f}%) [{r_any_ci[0]*100:.1f}%, {r_any_ci[1]*100:.1f}%] | "
                        f"{fp}/{nc} ({far*100:.1f}%) [{far_ci[0]*100:.1f}%, {far_ci[1]*100:.1f}%] |"
                    )

    # 3. Synthetic Test Recall by Class and Degradation Severity Tier
    md.append("")
    md.append("## 3. Synthetic Test Recall by Class and Degradation Severity Tier (Class x Tier Table, Overlap Rule)")
    md.append("")
    md.append("| Degradation Tier | Image Class | Forged Samples | Frozen DCT Recall [95% CI] | Adapted DCT Recall [95% CI] | Frozen SIFT Recall [95% CI] |")
    md.append("|---|---|---|---|---|---|")

    if not synth_df.empty:
        f_synth = synth_df[synth_df["is_forged"]]
        for deg in ["zero", "light", "moderate", "heavy"]:
            deg_sub = f_synth[f_synth["degradation_level"] == deg]
            n_tot = len(deg_sub)
            if n_tot == 0:
                continue

            def get_stat(df_slice, n_slice, pfx):
                k = (df_slice[f"{pfx}_pix_tp"] > 0).sum() if f"{pfx}_pix_tp" in df_slice else 0
                ci = wilson_ci(k, n_slice)
                return f"{k}/{n_slice} ({k/n_slice*100:.1f}%) [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]"

            low_tot = " *" if n_tot < 30 else ""
            md.append(f"| **{deg.capitalize()} (All)**{low_tot} | **Combined** | **{n_tot}** | {get_stat(deg_sub, n_tot, 'dct')} | {get_stat(deg_sub, n_tot, 'ad_dct')} | {get_stat(deg_sub, n_tot, 'sift')} |")

            for cname in ["Blot/Gel", "Microscopy"]:
                c_sub = deg_sub[deg_sub["class"] == cname]
                nc = len(c_sub)
                if nc == 0:
                    continue
                low_c = " *" if nc < 30 else ""
                md.append(f"| {deg.capitalize()} | {cname}{low_c} | {nc} | {get_stat(c_sub, nc, 'dct')} | {get_stat(c_sub, nc, 'ad_dct')} | {get_stat(c_sub, nc, 'sift')} |")

    # 4. Synthetic Test Recall by Patch Area Bucket
    md.append("")
    md.append("## 4. Synthetic Test Recall by Patch Area Bucket (Overlap Rule)")
    md.append("")
    md.append("| Patch Area Bucket | Forged Samples | Frozen DCT Recall [95% CI] | Adapted DCT Recall [95% CI] | Frozen SIFT Recall [95% CI] |")
    md.append("|---|---|---|---|---|")

    if not synth_df.empty:
        f_synth = synth_df[synth_df["is_forged"]]
        buckets = [
            ("Small (< 500 px)", f_synth["patch_area"] < 500),
            ("Medium (500 - 1500 px)", (f_synth["patch_area"] >= 500) & (f_synth["patch_area"] <= 1500)),
            ("Large (> 1500 px)", f_synth["patch_area"] > 1500),
        ]
        for bname, mask in buckets:
            sub = f_synth[mask]
            n = len(sub)
            if n == 0:
                continue
            low_conf = " *" if n < 30 else ""

            def get_stat(pfx):
                k = (sub[f"{pfx}_pix_tp"] > 0).sum() if f"{pfx}_pix_tp" in sub else 0
                ci = wilson_ci(k, n)
                return f"{k}/{n} ({k/n*100:.1f}%) [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]"

            md.append(f"| {bname}{low_conf} | {n} | {get_stat('dct')} | {get_stat('ad_dct')} | {get_stat('sift')} |")

    # 5. Synthetic Test Recall by Transform Category
    md.append("")
    md.append("## 5. Synthetic Test Recall by Transformation Category (Overlap Rule)")
    md.append("")
    md.append("| Transformation Category | Forged Samples | Frozen DCT Recall [95% CI] | Adapted DCT Recall [95% CI] | Frozen SIFT Recall [95% CI] |")
    md.append("|---|---|---|---|---|")

    if not synth_df.empty:
        f_synth = synth_df[synth_df["is_forged"]]
        for tf_cat in ["translation", "90/180 rotation", "flip", "arbitrary rotation/scale", "combination"]:
            sub = f_synth[f_synth["transform_category"] == tf_cat]
            n = len(sub)
            if n == 0:
                continue
            low_conf = " *" if n < 30 else ""

            def get_stat(pfx):
                k = (sub[f"{pfx}_pix_tp"] > 0).sum() if f"{pfx}_pix_tp" in sub else 0
                ci = wilson_ci(k, n)
                return f"{k}/{n} ({k/n*100:.1f}%) [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]"

            md.append(f"| {tf_cat}{low_conf} | {n} | {get_stat('dct')} | {get_stat('ad_dct')} | {get_stat('sift')} |")

    # 6. Real BioFors IDD Fusion & Agreement Analysis
    if not real_df.empty:
        md.append("")
        md.append("## 6. Real BioFors IDD Fusion and Agreement Analysis")
        md.append("")
        md.append("Evaluates detector agreement and decision combinations on the real BioFors IDD test set (91 manipulated panels, 2,195 clean negative panels).")
        md.append("Note: These metrics were computed on the held-out real BioFors IDD test set for evaluation and reporting purposes only. The final fusion decision rule will be formally selected in Phase 5 using validation data only.")
        md.append("")
        md.append("| Decision Tier / Combination | Rule Definition | Forged Recall (Overlap) [95% CI] | Clean False Alarm Rate [95% CI] | Detection Precision [95% CI] |")
        md.append("|---|---|---|---|---|")

        f_real = real_df[real_df["is_forged"]].copy()
        c_real = real_df[~real_df["is_forged"]].copy()
        nf_real = len(f_real)
        nc_real = len(c_real)

        hit_df = (f_real["dct_pix_tp"] > 0)
        hit_da = (f_real["ad_dct_pix_tp"] > 0)
        hit_s = (f_real["sift_pix_tp"] > 0)

        fa_df = c_real["dct_flagged"]
        fa_da = c_real["ad_dct_flagged"]
        fa_s = c_real["sift_flagged"]

        fusion_combos = [
            ("High Confidence (Both Agree)", "Adapted DCT AND Frozen SIFT", hit_da & hit_s, fa_da & fa_s),
            ("Low Confidence (Adapted DCT Only)", "Adapted DCT Flags, SIFT Silent", hit_da & ~hit_s, fa_da & ~fa_s),
            ("Low Confidence (Frozen SIFT Only)", "Frozen SIFT Flags, DCT Silent", hit_s & ~hit_da, fa_s & ~fa_da),
            ("Combined OR Fusion (Either Flags)", "Adapted DCT OR Frozen SIFT", hit_da | hit_s, fa_da | fa_s),
            ("Baseline: Frozen DCT OR SIFT", "Frozen DCT OR Frozen SIFT", hit_df | hit_s, fa_df | fa_s),
            ("Baseline: Frozen DCT AND SIFT", "Frozen DCT AND Frozen SIFT", hit_df & hit_s, fa_df & fa_s),
        ]

        for title, rule_desc, f_hits, c_fas in fusion_combos:
            tp_cnt = int(f_hits.sum())
            fp_cnt = int(c_fas.sum())
            rec = tp_cnt / nf_real if nf_real > 0 else 0.0
            far = fp_cnt / nc_real if nc_real > 0 else 0.0
            prec = tp_cnt / (tp_cnt + fp_cnt) if (tp_cnt + fp_cnt) > 0 else 0.0

            ci_rec = wilson_ci(tp_cnt, nf_real)
            ci_far = wilson_ci(fp_cnt, nc_real)
            ci_prec = wilson_ci(tp_cnt, tp_cnt + fp_cnt) if (tp_cnt + fp_cnt) > 0 else (0.0, 0.0)

            md.append(
                f"| **{title}** | {rule_desc} | "
                f"{tp_cnt}/{nf_real} ({rec*100:.1f}%) [{ci_rec[0]*100:.1f}%, {ci_rec[1]*100:.1f}%] | "
                f"{fp_cnt}/{nc_real} ({far*100:.2f}%) [{ci_far[0]*100:.2f}%, {ci_far[1]*100:.2f}%] | "
                f"{tp_cnt}/{tp_cnt+fp_cnt} ({prec*100:.1f}%) [{ci_prec[0]*100:.1f}%, {ci_prec[1]*100:.1f}%] |"
            )

    # 7. Real Pristine Train Check
    if pristine_stats:
        md.append("")
        md.append("## 7. Real Pristine Train Check (Unused Train Documents, N=1,000)")
        md.append("")
        md.append("Evaluates false alarm rates on pristine panels sampled from BioFors TRAIN documents not used by the synthetic generator (report only, no tuning).")
        md.append("")
        md.append("| Detector | Clean Panels Evaluated | False Alarms | Clean False Alarm Rate [95% CI] |")
        md.append("|---|---|---|---|")
        for k in ["dct", "ad_dct", "sift"]:
            if k in pristine_stats:
                st = pristine_stats[k]
                ci = st["ci"]
                md.append(f"| {st['name']} | {st['total']} | {st['fp']} | {st['far']*100:.1f}% [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%] |")

    md.append("")
    md.append("## 8. Notes and Observations")
    md.append("")
    md.append("1. **Adapted DCT Validation Locking:** Parameters (min_votes=75, min_size=64) were locked strictly on synthetic VAL (`val_manifest.csv`). Synthetic test and BioFors IDD test splits were strictly held-out.")
    md.append("2. **Overlap Rule vs Any-Flagged Rule:** The overlap rule strictly requires `pix_tp > 0`. On clean backgrounds, any spurious flag is recorded as a false alarm rather than an accidental hit.")
    md.append("3. **Statistical Uncertainty:** Subsets with N < 30 (notably real microscopy forgeries N=17, and combination synthetic transforms) have wide confidence intervals and must be interpreted cautiously.")
    md.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"\nSaved benchmark markdown summary to: {out_path}")


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    sift_params = cfg.get("frozen_sift", {})

    adapted_cfg = load_yaml(args.adapted_config) if os.path.exists(args.adapted_config) else {"min_votes": 75, "min_size": 64}

    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_dct = args.detector in ["dct", "both"]
    eval_sift = args.detector in ["sift", "both"]

    synth_df = pd.DataFrame()
    synth_dct_metrics, synth_ad_dct_metrics, synth_sift_metrics = {}, {}, {}

    real_df = pd.DataFrame()
    real_dct_metrics, real_ad_dct_metrics, real_sift_metrics = {}, {}, {}

    pristine_df = pd.DataFrame()
    pristine_stats = {}

    images_dir = REPO_ROOT / "data" / "biofors" / "images"

    # 1. Benchmark on Synthetic Test Set
    if args.dataset in ["synthetic", "both"]:
        synth_manifest = REPO_ROOT / "data" / "synthetic" / "test_manifest.csv"
        if synth_manifest.exists():
            synth_df, synth_dct_metrics, synth_ad_dct_metrics, synth_sift_metrics = run_benchmark_synthetic(
                synth_manifest, sift_params, eval_dct, eval_sift, adapted_cfg, limit=args.limit
            )
            synth_df.to_csv(out_dir / "synthetic_benchmark.csv", index=False)

    # 2. Benchmark on Real BioFors IDD Test Set
    if args.dataset in ["biofors", "both"]:
        real_manifest = REPO_ROOT / "data" / "biofors" / "splits" / "test_idd_manifest.csv"
        idd_gt_path = REPO_ROOT / "data" / "biofors" / "annotations" / "idd_gt.json"

        if real_manifest.exists() and idd_gt_path.exists():
            real_df, real_dct_metrics, real_ad_dct_metrics, real_sift_metrics = run_benchmark_biofors(
                real_manifest, images_dir, idd_gt_path, sift_params, eval_dct, eval_sift, adapted_cfg, limit=args.limit
            )
            real_df.to_csv(out_dir / "real_biofors_benchmark.csv", index=False)

    # 3. Real Pristine Train Audit Check
    if args.dataset in ["pristine_train", "both"]:
        pristine_df, pristine_stats = run_pristine_train_audit(
            images_dir, sift_params, eval_dct, eval_sift, adapted_cfg, sample_size=1000
        )
        if not pristine_df.empty:
            pristine_df.to_csv(out_dir / "pristine_train_audit.csv", index=False)

    # 4. Fallback loading if cached results exist on disk
    if synth_df.empty and (out_dir / "synthetic_benchmark.csv").exists():
        synth_df = pd.read_csv(out_dir / "synthetic_benchmark.csv")
        synth_dct_metrics = compute_metrics(synth_df, "dct")
        synth_ad_dct_metrics = compute_metrics(synth_df, "ad_dct")
        synth_sift_metrics = compute_metrics(synth_df, "sift")

    if real_df.empty and (out_dir / "real_biofors_benchmark.csv").exists():
        real_df = pd.read_csv(out_dir / "real_biofors_benchmark.csv")
        real_dct_metrics = compute_metrics(real_df, "dct")
        real_ad_dct_metrics = compute_metrics(real_df, "ad_dct")
        real_sift_metrics = compute_metrics(real_df, "sift")

    if not pristine_stats and (out_dir / "pristine_train_audit.csv").exists():
        pdf = pd.read_csv(out_dir / "pristine_train_audit.csv")
        for pfx, dlabel in [("dct", "Frozen DCT"), ("ad_dct", "Adapted DCT"), ("sift", "Frozen SIFT")]:
            if f"{pfx}_flagged" in pdf:
                fp = int(pdf[f"{pfx}_flagged"].sum())
                tot = len(pdf)
                pristine_stats[pfx] = {
                    "name": dlabel,
                    "fp": fp,
                    "total": tot,
                    "far": fp / tot if tot > 0 else 0.0,
                    "ci": wilson_ci(fp, tot),
                }

    # 5. Generate Consolidated Summary Markdown
    summary_md_path = out_dir / "benchmark_summary.md"
    generate_summary_markdown(
        synth_dct_metrics,
        synth_ad_dct_metrics,
        synth_sift_metrics,
        real_dct_metrics,
        real_ad_dct_metrics,
        real_sift_metrics,
        synth_df,
        real_df,
        pristine_stats,
        summary_md_path,
    )


if __name__ == "__main__":
    main()
