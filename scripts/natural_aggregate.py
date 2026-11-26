"""Aggregate cached classical detector results and report metrics.

What this script does, step by step:
1. Load all three manifests to get ground-truth labels and splits.
2. Scan results/natural/cache/<detector>/<dataset>/<split>/ for .json files.
3. Join each cached result back to its manifest row using the image_path as key.
4. CHECK: if any row belongs to a test split (casia2 test or coverage test) AND
   --final was NOT passed, print an error and exit with code 1. The test numbers
   are never printed without --final.
5. For each (detector, dataset, split) group, compute:
      recall = TP / N_forged
      FAR    = FP / N_authentic
   and Wilson 95% CIs on both.
6. If --calibrate is passed, run the grid search defined in the protocol over
   the CASIA calibration split and save the best config to
   configs/natural_classical_calibrated.yaml (with a SHA-256 hash of the file).
7. Write docs/natural_baselines_dev.md with all results for calibration and
   dev splits (never test).

Usage:
  .venv/bin/python scripts/natural_aggregate.py --detector dct
  .venv/bin/python scripts/natural_aggregate.py --detector sift
  .venv/bin/python scripts/natural_aggregate.py --detector dct --calibrate
  .venv/bin/python -u scripts/natural_aggregate.py --final  # full held-out evaluation
  .venv/bin/python scripts/natural_aggregate.py --all   # both detectors, calibration + baselines doc
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = REPO_ROOT / "data" / "natural" / "manifests"
CACHE_ROOT = REPO_ROOT / "results" / "natural" / "cache"
CONFIG_OUT = REPO_ROOT / "configs" / "natural_classical_calibrated.yaml"
DOC_OUT = REPO_ROOT / "docs" / "natural_baselines_dev.md"

# Test-only splits: blocked without --final
TEST_SPLITS = {("casia2", "test"), ("coverage", "test")}

# Wilson CI z-value for 95%
Z95 = 1.96

# Protocol calibration grids (Section 2.2 and 2.4)
DCT_GRID = {
    "min_votes": [25, 50, 75, 100, 125, 150],
    "min_size": [100, 250, 500, 750, 1000],
}
SIFT_GRID = {
    "min_inliers": [3, 4, 5, 6, 8, 10, 12, 15],
}

# FAR budget for calibration (protocol: 2%)
FAR_BUDGET = 0.02

# ---------------------------------------------------------------------------
# Wilson CI
# ---------------------------------------------------------------------------

def wilson_ci(successes: int, n: int, z: float = Z95):
    """Return the observed proportion and its Wilson confidence limits."""
    if n == 0:
        return (float("nan"),) * 3
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def ci_str(successes: int, n: int) -> str:
    """Format as 'value% [lo%, hi%]' (Wilson 95% CI)."""
    if n == 0:
        return "N/A"
    c, lo, hi = wilson_ci(successes, n)
    return f"{c*100:.1f}% [{lo*100:.1f}%, {hi*100:.1f}%]"


# ---------------------------------------------------------------------------
# Load manifests
# ---------------------------------------------------------------------------

def load_manifests() -> pd.DataFrame:
    frames = []
    for name in ("casia2.csv", "comofod.csv", "coverage.csv"):
        p = MANIFEST_DIR / name
        if p.exists():
            df = pd.read_csv(p)
            if len(df) > 0:
                frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No manifests in {MANIFEST_DIR}")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Load cache for one detector
# ---------------------------------------------------------------------------

def load_cache(detector: str) -> pd.DataFrame:
    """Return a DataFrame with columns: image_path, decision, score, runtime_sec."""
    cache_det = CACHE_ROOT / detector
    rows = []
    if not cache_det.exists():
        return pd.DataFrame(columns=["image_path", "decision", "score", "runtime_sec"])
    for json_file in cache_det.rglob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
        # image_path is stored in the JSON (added in natural_run_classical.py).
        if "image_path" not in data:
            continue  # Skip legacy cache files that predate this field.
        rows.append({
            "image_path": data["image_path"],
            "decision": data["decision"],
            "score": data["score"],
            "runtime_sec": data["runtime_sec"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Join cache to manifest
# ---------------------------------------------------------------------------

def join_results(manifest: pd.DataFrame, cache_df: pd.DataFrame,
                 detector: str) -> pd.DataFrame:
    """Return manifest rows with decision/score columns added, joined on image_path."""
    if cache_df.empty or "image_path" not in cache_df.columns:
        return pd.DataFrame()

    merged = manifest.merge(
        cache_df[["image_path", "decision", "score", "runtime_sec"]],
        on="image_path",
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()

    return merged[[
        "dataset", "split", "label", "attack_type",
        "image_path", "group_id", "decision", "score", "runtime_sec",
    ]]


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame, group_cols=("dataset", "split")):
    """Compute recall and FAR per group. Returns list of dicts."""
    rows = []
    for keys, grp in df.groupby(list(group_cols)):
        if isinstance(keys, str):
            keys = (keys,)
        forged = grp[grp["label"] == "forged"]
        authentic = grp[grp["label"] == "authentic"]
        n_f = len(forged)
        n_a = len(authentic)
        tp = int(forged["decision"].sum()) if n_f > 0 else 0
        fp = int(authentic["decision"].sum()) if n_a > 0 else 0
        rec_c, rec_lo, rec_hi = wilson_ci(tp, n_f)
        far_c, far_lo, far_hi = wilson_ci(fp, n_a)
        entry = dict(zip(group_cols, keys))
        entry.update({
            "n_forged": n_f, "n_authentic": n_a,
            "TP": tp, "FP": fp,
            "recall": rec_c, "recall_lo": rec_lo, "recall_hi": rec_hi,
            "FAR": far_c, "FAR_lo": far_lo, "FAR_hi": far_hi,
        })
        rows.append(entry)
    return rows


# ---------------------------------------------------------------------------
# Test-split guard
# ---------------------------------------------------------------------------

def check_test_guard(results_df: pd.DataFrame, final: bool):
    """Raise SystemExit if test data is present but --final was not passed."""
    for _, row in results_df.iterrows():
        if (row["dataset"], row["split"]) in TEST_SPLITS:
            if not final:
                print(
                    "\n[GUARD] ERROR: Results include test-split data "
                    f"({row['dataset']}/{row['split']}).\n"
                    "       Pass --final to unlock test metrics.\n"
                    "       Protocol rule: test splits are evaluated in ONE final pass.\n",
                    file=sys.stderr,
                )
                sys.exit(1)


# ---------------------------------------------------------------------------
# Calibration grid search
# ---------------------------------------------------------------------------

def run_calibration_grid(detector: str, manifest: pd.DataFrame,
                         cache_df: pd.DataFrame) -> dict:
    """
    Search the protocol grid over CASIA calibration split.
    Returns the best config as a dict.

    For DCT: grid over min_votes x min_size (30 points).
    For SIFT: grid over min_inliers (8 points).

    The score stored in the cache is used to replay decisions at different
    thresholds WITHOUT re-running the detector:
      - DCT: score = foreground pixel fraction. A result is "positive" if
        score > 0 AND the number of foreground pixels exceeds min_size (we
        store this as we cannot replay the exact vote filter, so we use a
        separate re-run approach via the mask).
      - SIFT: score = inlier count. Decision at threshold t: score >= t.

    For DCT we cannot replay different min_votes/min_size from the pixel score
    alone (the score is a post-filter fraction). We load the saved mask and
    recompute component sizes to replay min_size, but min_votes cannot be
    replayed from the mask -- it was applied before painting. Therefore:
      - DCT calibration re-runs the detector for each grid point on the
        calibration split (this is the only correct approach).
      - SIFT calibration uses the cached inlier count (score field) to replay
        all 8 thresholds without re-running.
    """
    cal_manifest = manifest[
        (manifest["dataset"] == "casia2") &
        (manifest["split"] == "calibration")
    ].copy()

    if detector == "sift":
        return _calibrate_sift_from_cache(cal_manifest, cache_df)
    else:
        return _calibrate_dct_rerun(cal_manifest)


def _eval_sift_single_worker(args_tuple):
    """Run detect_sift with a specific min_inliers and return only (label, bool)."""
    label, image_path, min_inliers = args_tuple
    sys.path.insert(0, str(REPO_ROOT))
    import cv2 as _cv2
    from detector.sift.localization import detect_sift

    img = _cv2.imread(str(REPO_ROOT / image_path))
    if img is None:
        raise OSError(f"Could not read manifest image {image_path}")
    gray = _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)
    mask, _, _, _ = detect_sift(
        gray,
        ratio_threshold=0.70, min_dist=30.0, ransac_threshold=3.0,
        min_inliers=min_inliers, cluster_threshold=100.0, mask_radius=12,
        use_convex_hull=True, check_flip=True, min_inlier_ratio=0.25,
        min_inliers_flip=10, filter_orientation=True, orientation_tolerance=20.0,
    )
    return label, bool((mask > 0).any())


def _extract_sift_matches_worker(args_tuple):
    """Extract raw SIFT matches once per image for fast exact grid search."""
    idx, label, image_path = args_tuple
    sys.path.insert(0, str(REPO_ROOT))
    import cv2 as _cv2
    from detector.sift.features import detect_and_compute
    from detector.sift.matching import match_within_image, match_flipped

    img = _cv2.imread(image_path)
    if img is None:
        return idx, label, None
    gray = _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)
    _, w = gray.shape
    kps, descs = detect_and_compute(gray)
    pts1_std, pts2_std, _, ang1_std, ang2_std = match_within_image(
        kps, descs, ratio_threshold=0.70, min_dist=30.0, return_angles=True
    )
    gray_flip = _cv2.flip(gray, 1)
    kps_flip, descs_flip = detect_and_compute(gray_flip)
    pts1_flip, pts2_flip, _, ang1_flip, ang2_flip = match_flipped(
        kps, descs, kps_flip, descs_flip,
        width=w, ratio_threshold=0.70, min_dist=30.0, return_angles=True
    )
    return idx, label, (
        pts1_std, pts2_std, ang1_std, ang2_std,
        pts1_flip, pts2_flip, ang1_flip, ang2_flip,
    )


def _eval_sift_from_matches(cached_matches, min_inliers: int) -> bool:
    """Run orientation filter + RANSAC verification on precomputed SIFT matches."""
    sys.path.insert(0, str(REPO_ROOT))
    from detector.sift.matching import filter_by_orientation
    from detector.sift.verification import verify_matches_ransac

    (pts1_std, pts2_std, ang1_std, ang2_std,
     pts1_flip, pts2_flip, ang1_flip, ang2_flip) = cached_matches

    f1, f2 = filter_by_orientation(
        pts1_std, pts2_std, ang1_std, ang2_std,
        tolerance=20.0, min_support=min_inliers,
    )
    inl1_std, _, _ = verify_matches_ransac(
        f1, f2, ransac_threshold=3.0, min_inliers=min_inliers,
        cluster_threshold=100.0, min_inlier_ratio=0.25,
    )
    if len(inl1_std) > 0:
        return True

    ff1, ff2 = filter_by_orientation(
        pts1_flip, pts2_flip, ang1_flip, ang2_flip,
        tolerance=20.0, min_support=10,
    )
    inl1_flip, _, _ = verify_matches_ransac(
        ff1, ff2, ransac_threshold=3.0, min_inliers=10,
        cluster_threshold=100.0, min_inlier_ratio=0.25,
    )
    return len(inl1_flip) > 0


def _calibrate_sift_from_cache(cal_manifest: pd.DataFrame,
                                cache_df: pd.DataFrame) -> dict:
    """Evaluate all 8 min_inliers values on CASIA calibration split exactly."""
    from concurrent.futures import ProcessPoolExecutor

    forged_rows = cal_manifest[cal_manifest["label"] == "forged"]
    authentic_rows = cal_manifest[cal_manifest["label"] == "authentic"]
    n_f, n_a = len(forged_rows), len(authentic_rows)
    if n_f == 0 or n_a == 0:
        raise RuntimeError("Calibration split has no forged or authentic rows.")

    work = [
        (i, row["label"], row["image_path"])
        for i, (_, row) in enumerate(cal_manifest.iterrows())
    ]
    print(f"  [calibrate-sift] Precomputing SIFT matches for {len(work)} calibration images...")
    precomputed_forged = []
    precomputed_auth = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for _, label, data in ex.map(_extract_sift_matches_worker, work, chunksize=32):
            if data is None:
                continue
            if label == "forged":
                precomputed_forged.append(data)
            else:
                precomputed_auth.append(data)

    best = None
    best_recall = -1.0
    best_far = 1.0
    lowest_far_entry = None

    for min_inliers in SIFT_GRID["min_inliers"]:
        tp = sum(1 for d in precomputed_forged if _eval_sift_from_matches(d, min_inliers))
        fp = sum(1 for d in precomputed_auth if _eval_sift_from_matches(d, min_inliers))
        recall = tp / n_f
        far = fp / n_a
        print(f"  min_inliers={min_inliers:2d} recall={recall:.3f} far={far:.3f}")
        entry = {
            "min_inliers": min_inliers,
            "tp": tp,
            "fp": fp,
            "calibration_recall": round(recall, 4),
            "calibration_far": round(far, 4),
        }
        if lowest_far_entry is None or far < lowest_far_entry["calibration_far"]:
            lowest_far_entry = entry
        if far <= FAR_BUDGET:
            if recall > best_recall or (recall == best_recall and far < best_far):
                best_recall = recall
                best_far = far
                best = entry

    if best is None:
        best = lowest_far_entry
        print("[calibrate-sift] WARNING: no grid point meets 2% FAR budget; "
              "using lowest-FAR point.", file=sys.stderr)

    print(f"[calibrate-sift] best min_inliers={best['min_inliers']} "
          f"recall={best['calibration_recall']:.3f} "
          f"far={best['calibration_far']:.3f}")
    return best


def _eval_dct_single_worker(args_tuple):
    """Run detect() with specific min_votes/min_size and return only (label, bool)."""
    label, image_path, min_votes, min_size = args_tuple
    sys.path.insert(0, str(REPO_ROOT))
    import cv2 as _cv2
    from detector.detect import detect

    img = _cv2.imread(str(REPO_ROOT / image_path))
    if img is None:
        raise OSError(f"Could not read manifest image {image_path}")
    gray = _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)
    mask = detect(
        gray, B=8, K=16, Q=4, min_std=4.0, window=5, min_dist=30,
        min_votes=min_votes, kernel=9, min_size=min_size,
    )
    return label, bool((mask > 0).any())


def _extract_dct_matches_worker(args_tuple):
    """Extract (a, b, shift, shape) once per image for fast DCT grid search."""
    idx, label, image_path = args_tuple
    sys.path.insert(0, str(REPO_ROOT))
    import cv2 as _cv2
    from detector.features import extract_features
    from detector.matching import find_matches

    img = _cv2.imread(image_path)
    if img is None:
        return idx, label, None
    gray = _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)
    feats, pos = extract_features(gray, B=8, K=16, Q=4, min_std=4.0)
    a, b, shift = find_matches(feats, pos, window=5, min_dist=30)
    return idx, label, (a, b, shift, gray.shape)


def _eval_dct_stage34(cached_matches, min_votes: int, min_size: int) -> bool:
    """Run stages 3-4 of detector/detect.py on precomputed matches."""
    import cv2 as _cv2
    from skimage.morphology import remove_small_objects

    a, b, shift, shape = cached_matches
    if len(shift) == 0:
        return False
    _, inverse, counts = np.unique(
        shift, axis=0, return_inverse=True, return_counts=True
    )
    inverse = inverse.reshape(-1)
    keep = counts[inverse] >= min_votes
    a_keep, b_keep = a[keep], b[keep]
    if len(a_keep) == 0:
        return False

    mask = np.zeros(shape, np.uint8)
    for r, c in np.concatenate([a_keep, b_keep]):
        mask[r:r + 8, c:c + 8] = 255
    closed = _cv2.morphologyEx(mask, _cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    clean = remove_small_objects(closed > 0, max_size=min_size - 1)
    return bool(clean.any())


def _calibrate_dct_rerun(cal_manifest: pd.DataFrame) -> dict:
    """
    Calibrate DCT on the calibration split.
    Precomputes DCT features and block matches once per image in parallel,
    then evaluates all 30 (min_votes, min_size) grid points exactly.
    """
    from concurrent.futures import ProcessPoolExecutor

    forged_rows = cal_manifest[cal_manifest["label"] == "forged"]
    authentic_rows = cal_manifest[cal_manifest["label"] == "authentic"]
    n_f = len(forged_rows)
    n_a = len(authentic_rows)

    if n_f == 0 or n_a == 0:
        raise RuntimeError("Calibration split has no forged or authentic rows.")

    work = [
        (i, row["label"], row["image_path"])
        for i, (_, row) in enumerate(cal_manifest.iterrows())
    ]
    print(f"  [calibrate-dct] Precomputing DCT matches for {len(work)} calibration images...")
    precomputed_forged = []
    precomputed_auth = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for _, label, data in ex.map(_extract_dct_matches_worker, work, chunksize=32):
            if data is None:
                continue
            if label == "forged":
                precomputed_forged.append(data)
            else:
                precomputed_auth.append(data)

    best = None
    best_recall = -1.0
    best_far = 1.0
    lowest_far_entry = None

    total_points = len(DCT_GRID["min_votes"]) * len(DCT_GRID["min_size"])
    point = 0
    for min_votes in DCT_GRID["min_votes"]:
        for min_size in DCT_GRID["min_size"]:
            point += 1
            tp = sum(1 for d in precomputed_forged if _eval_dct_stage34(d, min_votes, min_size))
            fp = sum(1 for d in precomputed_auth if _eval_dct_stage34(d, min_votes, min_size))
            recall = tp / n_f
            far = fp / n_a
            print(f"  [{point}/{total_points}] min_votes={min_votes} min_size={min_size} "
                  f"recall={recall:.3f} far={far:.3f}")
            entry = {
                "min_votes": min_votes,
                "min_size": min_size,
                "tp": tp,
                "fp": fp,
                "calibration_recall": round(recall, 4),
                "calibration_far": round(far, 4),
            }
            if lowest_far_entry is None or far < lowest_far_entry["calibration_far"]:
                lowest_far_entry = entry
            if far <= FAR_BUDGET:
                if recall > best_recall or (recall == best_recall and far < best_far):
                    best_recall = recall
                    best_far = far
                    best = entry

    if best is None:
        print("[calibrate-dct] WARNING: no grid point meets 2% FAR budget; "
              "using lowest-FAR point.", file=sys.stderr)
        best = lowest_far_entry

    print(f"[calibrate-dct] best min_votes={best['min_votes']} "
          f"min_size={best['min_size']} "
          f"recall={best['calibration_recall']:.3f} "
          f"far={best['calibration_far']:.3f}")
    return best


def save_calibrated_config(dct_best: dict, sift_best: dict):
    """Write configs/natural_classical_calibrated.yaml and store its SHA-256."""
    config = {
        "description": (
            "Calibrated knobs for DCT and SIFT classical detectors. "
            "Selected on CASIA2 calibration split, seed=0, FAR budget 2%. "
            "Frozen after generation. See natural_protocol.md Section 2."
        ),
        "dct": {
            "min_votes": dct_best["min_votes"],
            "min_size": dct_best["min_size"],
            "calibration_recall": dct_best["calibration_recall"],
            "calibration_far": dct_best["calibration_far"],
        },
        "sift": {
            "min_inliers": sift_best["min_inliers"],
            "calibration_recall": sift_best["calibration_recall"],
            "calibration_far": sift_best["calibration_far"],
        },
    }
    CONFIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_OUT, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    # Compute and store a SHA-256 hash of the config file for integrity
    sha = hashlib.sha256(CONFIG_OUT.read_bytes()).hexdigest()
    hash_path = CONFIG_OUT.with_suffix(".yaml.sha256")
    hash_path.write_text(sha + "\n")
    print(f"[calibrate] Config written to {CONFIG_OUT}")
    print(f"[calibrate] SHA-256: {sha}")
    return sha


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def format_table_row(label, n_f, n_a, tp, fp):
    rec_str = ci_str(tp, n_f)
    far_str = ci_str(fp, n_a)
    return f"| {label} | {n_f} | {n_a} | {tp} | {fp} | {rec_str} | {far_str} |"


def write_baselines_doc(all_results: list, calibrated_config: dict | None):
    """Write docs/natural_baselines_dev.md."""
    lines = []
    lines.append("# Natural-Image Classical Baselines (Calibration and Dev Splits)")
    lines.append("")
    lines.append("Generated by `scripts/natural_aggregate.py`. "
                 "No test-split results appear here.")
    lines.append("")
    lines.append("> [!IMPORTANT]")
    lines.append("> CoMoFoD results are labeled **optimistic** for DCT and SIFT "
                 "because those configs were tuned on CoMoFoD. CASIA2 calibration "
                 "results are used for threshold selection only.")
    lines.append("")

    # Frozen results
    for detector in ("dct", "sift"):
        det_results = [r for r in all_results
                       if r["detector"] == detector and r["variant"] == "frozen"
                       and "attack_type" not in r]
        if not det_results:
            continue
        lines.append(f"## Frozen {detector.upper()}")
        lines.append("")
        lines.append("| Dataset / Split | N forged | N authentic | TP | FP | "
                     "Recall (Wilson 95% CI) | FAR (Wilson 95% CI) |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in sorted(det_results, key=lambda x: (x["dataset"], x["split"])):
            row = format_table_row(
                f"{r['dataset']} / {r['split']}",
                r["n_forged"], r["n_authentic"], r["TP"], r["FP"]
            )
            lines.append(row)
        lines.append("")

    # Calibrated results
    for detector in ("dct", "sift"):
        det_results = [r for r in all_results
                       if r["detector"] == detector and r["variant"] == "calibrated"]
        if not det_results:
            continue
        lines.append(f"## Calibrated {detector.upper()}")
        lines.append("")
        if calibrated_config:
            if detector == "dct":
                mv = calibrated_config.get("dct", {}).get("min_votes", "?")
                ms = calibrated_config.get("dct", {}).get("min_size", "?")
                lines.append(f"Calibrated on CASIA2 calibration split at FAR <= 2%. "
                             f"Best: `min_votes={mv}`, `min_size={ms}`.")
            else:
                mi = calibrated_config.get("sift", {}).get("min_inliers", "?")
                lines.append(f"Calibrated on CASIA2 calibration split at FAR <= 2%. "
                             f"Best: `min_inliers={mi}`.")
        lines.append("")
        lines.append("| Dataset / Split | N forged | N authentic | TP | FP | "
                     "Recall (Wilson 95% CI) | FAR (Wilson 95% CI) |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in sorted(det_results, key=lambda x: (x["dataset"], x["split"])):
            row = format_table_row(
                f"{r['dataset']} / {r['split']}",
                r["n_forged"], r["n_authentic"], r["TP"], r["FP"]
            )
            lines.append(row)
        lines.append("")

    # CoMoFoD per attack category (frozen DCT and SIFT)
    per_cat = [r for r in all_results if r.get("attack_type") and r["split"] == "dev"]
    if per_cat:
        lines.append("## CoMoFoD per Attack Category (Frozen Detectors)")
        lines.append("")
        lines.append("| Detector | Attack Type | N forged | N authentic | TP | FP | "
                     "Recall (Wilson 95% CI) | FAR (Wilson 95% CI) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in sorted(per_cat, key=lambda x: (x["detector"], x.get("attack_type", ""))):
            if r["variant"] != "frozen":
                continue
            row_label = f"{r['detector'].upper()} | {r.get('attack_type','')}"
            row = format_table_row(
                row_label, r["n_forged"], r["n_authentic"], r["TP"], r["FP"]
            )
            lines.append(row)
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Protocol: `docs/natural_protocol.md` (frozen, do not edit).*")
    lines.append("")

    DOC_OUT.parent.mkdir(parents=True, exist_ok=True)
    DOC_OUT.write_text("\n".join(lines))
    print(f"[aggregate] Wrote {DOC_OUT}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=["dct", "sift"],
                        help="Detector to aggregate. Omit with --all.")
    parser.add_argument("--all", action="store_true",
                        help="Aggregate both detectors.")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run calibration grid search and save config.")
    parser.add_argument("--apply-calibrated", nargs=3, type=int,
                        metavar=("MIN_VOTES", "MIN_SIZE", "MIN_INLIERS"),
                        help="Evaluate and save already-selected calibrated parameters.")
    parser.add_argument("--split", choices=["cal_dev", "test", "all"], default="cal_dev",
                        help="Which splits to report (test/all require --final).")
    parser.add_argument("--final", action="store_true",
                        help="Unlock test-split metrics (ONE final pass only).")
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--doc-out", type=Path, default=DOC_OUT)
    args = parser.parse_args()

    if args.final:
        if args.calibrate or args.apply_calibrated:
            parser.error("The final pass uses the locked thresholds")
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from natural_final import run_final

        run_final()
        return

    if not args.detector and not args.all:
        parser.error("Specify --detector dct|sift or --all.")

    cv2.setNumThreads(1)
    detectors = ["dct", "sift"] if args.all else [args.detector]

    # Print git commit
    try:
        import subprocess
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, text=True).strip()
    except Exception:
        commit = "unknown"
    print(f"[natural_aggregate] git={commit} detectors={detectors} "
          f"final={args.final} calibrate={args.calibrate}")

    manifest = load_manifests()

    all_result_rows = []
    calibrated_config_loaded = None
    best_by_det = {}

    # Load calibrated config if it exists (for reporting)
    if CONFIG_OUT.exists():
        with open(CONFIG_OUT) as f:
            calibrated_config_loaded = yaml.safe_load(f)

    for detector in detectors:
        print(f"\n--- {detector.upper()} ---")
        cache_df = load_cache(detector)
        print(f"  cache entries: {len(cache_df)}")

        joined = join_results(manifest, cache_df, detector)
        if joined.empty:
            print(f"  No cached results found for {detector}. "
                  "Run natural_run_classical.py first.")
            continue

        if args.split in ("test", "all") and not args.final:
            check_test_guard(joined, args.final)

        # Restrict to non-test splits unless --final and --split test/all
        if args.split == "cal_dev" or not args.final:
            joined = joined[~joined.apply(
                lambda r: (r["dataset"], r["split"]) in TEST_SPLITS, axis=1
            )]
        elif args.split == "test":
            joined = joined[joined.apply(
                lambda r: (r["dataset"], r["split"]) in TEST_SPLITS, axis=1
            )]

        # Compute frozen metrics
        metrics = compute_metrics(joined, group_cols=("dataset", "split"))
        for m in metrics:
            m["detector"] = detector
            m["variant"] = "frozen"
            all_result_rows.append(m)
            print(f"  frozen {m['dataset']}/{m['split']}: "
                  f"recall={m['recall']*100:.1f}% "
                  f"[{m['recall_lo']*100:.1f},{m['recall_hi']*100:.1f}] "
                  f"FAR={m['FAR']*100:.1f}% "
                  f"[{m['FAR_lo']*100:.1f},{m['FAR_hi']*100:.1f}]")

        # CoMoFoD per attack category (frozen)
        comofod_joined = joined[joined["dataset"] == "comofod"].copy()
        if len(comofod_joined) > 0:
            cat_metrics = compute_metrics(
                comofod_joined, group_cols=("dataset", "split", "attack_type")
            )
            for m in cat_metrics:
                m["detector"] = detector
                m["variant"] = "frozen"
                all_result_rows.append(m)

        # Calibration grid search or apply pre-selected calibrated params
        if args.calibrate or args.apply_calibrated:
            if args.apply_calibrated:
                mv_p, ms_p, mi_p = args.apply_calibrated
                if detector == "dct":
                    from concurrent.futures import ProcessPoolExecutor
                    cal_m = manifest[(manifest["dataset"] == "casia2") & (manifest["split"] == "calibration")]
                    n_f_c = int((cal_m["label"] == "forged").sum())
                    n_a_c = int((cal_m["label"] == "authentic").sum())
                    work_c = [(r["label"], r["image_path"], mv_p, ms_p) for _, r in cal_m.iterrows()]
                    print(f"  Evaluating calibrated DCT on {len(work_c)} CASIA2 calibration images...")
                    tp_c, fp_c = 0, 0
                    with ProcessPoolExecutor(max_workers=6) as ex:
                        decisions = ex.map(_eval_dct_single_worker, work_c, chunksize=8)
                        for idx, (lbl, dec) in enumerate(decisions, 1):
                            if idx % 100 == 0:
                                print(f"    [dct-calibration] {idx}/{len(work_c)}", flush=True)
                            if dec:
                                if lbl == "forged":
                                    tp_c += 1
                                else:
                                    fp_c += 1
                    best = {
                        "min_votes": mv_p, "min_size": ms_p,
                        "tp": tp_c, "fp": fp_c,
                        "calibration_recall": round(tp_c / n_f_c, 4),
                        "calibration_far": round(fp_c / n_a_c, 4),
                    }
                else:
                    from concurrent.futures import ProcessPoolExecutor
                    cal_m = manifest[(manifest["dataset"] == "casia2") & (manifest["split"] == "calibration")]
                    n_f_c = int((cal_m["label"] == "forged").sum())
                    n_a_c = int((cal_m["label"] == "authentic").sum())
                    work_c = [(r["label"], r["image_path"], mi_p) for _, r in cal_m.iterrows()]
                    print(f"  Evaluating calibrated SIFT on {len(work_c)} CASIA2 calibration images...")
                    tp_c, fp_c = 0, 0
                    with ProcessPoolExecutor(max_workers=6) as ex:
                        decisions = ex.map(_eval_sift_single_worker, work_c, chunksize=8)
                        for idx, (lbl, dec) in enumerate(decisions, 1):
                            if idx % 100 == 0:
                                print(f"    [sift-calibration] {idx}/{len(work_c)}", flush=True)
                            if dec:
                                if lbl == "forged":
                                    tp_c += 1
                                else:
                                    fp_c += 1
                    best = {
                        "min_inliers": mi_p,
                        "tp": tp_c, "fp": fp_c,
                        "calibration_recall": round(tp_c / n_f_c, 4),
                        "calibration_far": round(fp_c / n_a_c, 4),
                    }
            else:
                print(f"\n  Running calibration grid for {detector}...")
                cal_cache_df = cache_df  # already loaded
                best = run_calibration_grid(detector, manifest, cal_cache_df)
            best_by_det[detector] = best

            # Compute calibrated metrics using the best config
            if detector == "sift":
                min_inliers_best = best["min_inliers"]
                n_f = len(manifest[(manifest["dataset"] == "casia2") &
                                   (manifest["split"] == "calibration") &
                                   (manifest["label"] == "forged")])
                n_a = len(manifest[(manifest["dataset"] == "casia2") &
                                   (manifest["split"] == "calibration") &
                                   (manifest["label"] == "authentic")])
                tp_cal = int(best["tp"])
                fp_cal = int(best["fp"])
                rec_c, rec_lo, rec_hi = wilson_ci(tp_cal, n_f)
                far_c, far_lo, far_hi = wilson_ci(fp_cal, n_a)
                all_result_rows.append({
                    "detector": "sift", "variant": "calibrated",
                    "dataset": "casia2", "split": "calibration",
                    "n_forged": n_f, "n_authentic": n_a,
                    "TP": tp_cal, "FP": fp_cal,
                    "recall": rec_c, "recall_lo": rec_lo, "recall_hi": rec_hi,
                    "FAR": far_c, "FAR_lo": far_lo, "FAR_hi": far_hi,
                })
                print(f"  calibrated casia2/calibration: recall={rec_c*100:.1f}% FAR={far_c*100:.1f}%")

                comofod_m = manifest[
                    (manifest["dataset"] == "comofod") & (manifest["split"] == "dev")
                ]
                if len(comofod_m) > 0:
                    if min_inliers_best == 6:
                        for r in all_result_rows:
                            if (r["detector"] == "sift" and r["variant"] == "frozen"
                                    and r["dataset"] == "comofod" and not r.get("attack_type")):
                                copy_r = dict(r)
                                copy_r["variant"] = "calibrated"
                                all_result_rows.append(copy_r)
                                break
                    else:
                        from concurrent.futures import ProcessPoolExecutor
                        print(f"  Evaluating calibrated SIFT (min_inliers={min_inliers_best}) on CoMoFoD dev...")
                        work_dev = [
                            (row["label"], row["image_path"], min_inliers_best)
                            for _, row in comofod_m.iterrows()
                        ]
                        tp_dev, fp_dev = 0, 0
                        n_f_dev = int((comofod_m["label"] == "forged").sum())
                        n_a_dev = int((comofod_m["label"] == "authentic").sum())
                        with ProcessPoolExecutor(max_workers=6) as ex:
                            for idx_s, (lbl, dec) in enumerate(ex.map(_eval_sift_single_worker, work_dev, chunksize=16), 1):
                                if dec:
                                    if lbl == "forged":
                                        tp_dev += 1
                                    else:
                                        fp_dev += 1
                                if idx_s % 100 == 0:
                                    print(f"    [sift-dev] {idx_s}/{len(work_dev)}", flush=True)
                        rc, rlo, rhi = wilson_ci(tp_dev, n_f_dev)
                        fc, flo, fhi = wilson_ci(fp_dev, n_a_dev)
                        all_result_rows.append({
                            "detector": "sift", "variant": "calibrated",
                            "dataset": "comofod", "split": "dev",
                            "n_forged": n_f_dev, "n_authentic": n_a_dev,
                            "TP": tp_dev, "FP": fp_dev,
                            "recall": rc, "recall_lo": rlo, "recall_hi": rhi,
                            "FAR": fc, "FAR_lo": flo, "FAR_hi": fhi,
                        })
                        print(f"  calibrated comofod/dev: recall={rc*100:.1f}% FAR={fc*100:.1f}%")

            if detector == "dct":
                print(f"  [dct-calibrated] recall={best['calibration_recall']:.3f} "
                      f"far={best['calibration_far']:.3f} on CASIA cal only")
                n_f = len(manifest[(manifest["dataset"]=="casia2") &
                                   (manifest["split"]=="calibration") &
                                   (manifest["label"]=="forged")])
                n_a = len(manifest[(manifest["dataset"]=="casia2") &
                                   (manifest["split"]=="calibration") &
                                   (manifest["label"]=="authentic")])
                tp_cal = int(best["tp"])
                fp_cal = int(best["fp"])
                rec_c, rec_lo, rec_hi = wilson_ci(tp_cal, n_f)
                far_c, far_lo, far_hi = wilson_ci(fp_cal, n_a)
                all_result_rows.append({
                    "detector": "dct", "variant": "calibrated",
                    "dataset": "casia2", "split": "calibration",
                    "n_forged": n_f, "n_authentic": n_a,
                    "TP": tp_cal, "FP": fp_cal,
                    "recall": rec_c, "recall_lo": rec_lo, "recall_hi": rec_hi,
                    "FAR": far_c, "FAR_lo": far_lo, "FAR_hi": far_hi,
                })

                # Also evaluate calibrated DCT on CoMoFoD dev split
                comofod_m = manifest[
                    (manifest["dataset"] == "comofod") & (manifest["split"] == "dev")
                ]
                if len(comofod_m) > 0:
                    mv_best = best["min_votes"]
                    ms_best = best["min_size"]
                    if mv_best == 50 and ms_best == 500:
                        # Identical to frozen DCT; reuse frozen CoMoFoD metrics
                        for r in all_result_rows:
                            if (r["detector"] == "dct" and r["variant"] == "frozen"
                                    and r["dataset"] == "comofod" and not r.get("attack_type")):
                                copy_r = dict(r)
                                copy_r["variant"] = "calibrated"
                                all_result_rows.append(copy_r)
                                break
                    else:
                        from concurrent.futures import ProcessPoolExecutor
                        print(f"  Evaluating calibrated DCT (min_votes={mv_best}, min_size={ms_best}) on CoMoFoD dev...")
                        work_dev = [
                            (row["label"], row["image_path"], mv_best, ms_best)
                            for _, row in comofod_m.iterrows()
                        ]
                        tp_dev, fp_dev = 0, 0
                        n_f_dev = int((comofod_m["label"] == "forged").sum())
                        n_a_dev = int((comofod_m["label"] == "authentic").sum())
                        with ProcessPoolExecutor(max_workers=6) as ex:
                            for idx_d, (lbl, dec) in enumerate(ex.map(_eval_dct_single_worker, work_dev, chunksize=16), 1):
                                if dec:
                                    if lbl == "forged":
                                        tp_dev += 1
                                    else:
                                        fp_dev += 1
                                if idx_d % 100 == 0:
                                    print(f"    [dct-dev] {idx_d}/{len(work_dev)}", flush=True)
                        rc, rlo, rhi = wilson_ci(tp_dev, n_f_dev)
                        fc, flo, fhi = wilson_ci(fp_dev, n_a_dev)
                        all_result_rows.append({
                            "detector": "dct", "variant": "calibrated",
                            "dataset": "comofod", "split": "dev",
                            "n_forged": n_f_dev, "n_authentic": n_a_dev,
                            "TP": tp_dev, "FP": fp_dev,
                            "recall": rc, "recall_lo": rlo, "recall_hi": rhi,
                            "FAR": fc, "FAR_lo": flo, "FAR_hi": fhi,
                        })
                        print(f"  calibrated comofod/dev: recall={rc*100:.1f}% FAR={fc*100:.1f}%")

    # Save calibrated config if both detectors were calibrated
    if (args.calibrate or args.apply_calibrated) and "dct" in best_by_det and "sift" in best_by_det:
        save_calibrated_config(best_by_det["dct"], best_by_det["sift"])
        with open(CONFIG_OUT) as f:
            calibrated_config_loaded = yaml.safe_load(f)

    # Write doc
    if all_result_rows:
        write_baselines_doc(all_result_rows, calibrated_config_loaded)
    else:
        print("[aggregate] No results to report.")


if __name__ == "__main__":
    main()
