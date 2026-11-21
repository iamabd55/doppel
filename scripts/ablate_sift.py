"""Ablation and calibration study for SIFT copy-move detector on CoMoFoD splits.

Evaluates configurations across the 50/50 tuning split (first 20 images of each
category + 20 clean originals) to determine optimal operating thresholds without
data leakage onto the held-out evaluation set.

Can also evaluate the selected configuration on the held-out split with --held-out.

Usage:
    uv run scripts/ablate_sift.py                    # Run all configs on tuning split
    uv run scripts/ablate_sift.py --config cfg0_baseline
    uv run scripts/ablate_sift.py --held-out --config cfg8_best_combo
"""
import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Tuple
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detector.sift import detect_sift
from detector.evaluate import scores

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "comofod")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "phase2_sift")
os.makedirs(OUT_DIR, exist_ok=True)

TUNING_CATEGORIES = {
    "translation": (1, 20),
    "rotation": (41, 60),
    "scaling": (81, 100),
    "distortion": (121, 140),
    "combination": (161, 180),
}

HELDOUT_CATEGORIES = {
    "translation": (21, 40),
    "rotation": (61, 80),
    "scaling": (101, 120),
    "distortion": (141, 160),
    "combination": (181, 200),
}

CONFIGURATIONS = {
    "cfg0_baseline": {
        "name": "Config 0: Baseline (Current default)",
        "params": {
            "ratio_threshold": 0.70,
            "min_inliers": 4,
            "min_inlier_ratio": 0.0,
            "min_inliers_flip": 4,
            "check_flip": True,
            "use_convex_hull": True,
            "mask_radius": 12,
            "filter_orientation": False,
        },
    },
    "cfg1_inliers6": {
        "name": "Config 1: min_inliers=6",
        "params": {
            "ratio_threshold": 0.70,
            "min_inliers": 6,
            "min_inlier_ratio": 0.0,
            "min_inliers_flip": 6,
            "check_flip": True,
            "use_convex_hull": True,
            "mask_radius": 12,
            "filter_orientation": False,
        },
    },
    "cfg2_inliers8": {
        "name": "Config 2: min_inliers=8",
        "params": {
            "ratio_threshold": 0.70,
            "min_inliers": 8,
            "min_inlier_ratio": 0.0,
            "min_inliers_flip": 8,
            "check_flip": True,
            "use_convex_hull": True,
            "mask_radius": 12,
            "filter_orientation": False,
        },
    },
    "cfg3_inlier_ratio25": {
        "name": "Config 3: min_inliers=6, inlier_ratio>=0.25",
        "params": {
            "ratio_threshold": 0.70,
            "min_inliers": 6,
            "min_inlier_ratio": 0.25,
            "min_inliers_flip": 6,
            "check_flip": True,
            "use_convex_hull": True,
            "mask_radius": 12,
            "filter_orientation": False,
        },
    },
    "cfg4_strict_flip10": {
        "name": "Config 4: min_inliers=6, strict flip (flip_inliers=10)",
        "params": {
            "ratio_threshold": 0.70,
            "min_inliers": 6,
            "min_inlier_ratio": 0.0,
            "min_inliers_flip": 10,
            "check_flip": True,
            "use_convex_hull": True,
            "mask_radius": 12,
            "filter_orientation": False,
        },
    },
    "cfg5_no_flip": {
        "name": "Config 5: min_inliers=6, flip pass disabled",
        "params": {
            "ratio_threshold": 0.70,
            "min_inliers": 6,
            "min_inlier_ratio": 0.0,
            "min_inliers_flip": None,
            "check_flip": False,
            "use_convex_hull": True,
            "mask_radius": 12,
            "filter_orientation": False,
        },
    },
    "cfg6_orientation_filter": {
        "name": "Config 6: min_inliers=6, orientation consistency filter",
        "params": {
            "ratio_threshold": 0.70,
            "min_inliers": 6,
            "min_inlier_ratio": 0.0,
            "min_inliers_flip": 6,
            "check_flip": True,
            "use_convex_hull": True,
            "mask_radius": 12,
            "filter_orientation": True,
            "orientation_tolerance": 20.0,
        },
    },
    "cfg7_conservative_mask": {
        "name": "Config 7: min_inliers=6, conservative mask (no hull, radius 15)",
        "params": {
            "ratio_threshold": 0.70,
            "min_inliers": 6,
            "min_inlier_ratio": 0.0,
            "min_inliers_flip": 6,
            "check_flip": True,
            "use_convex_hull": False,
            "mask_radius": 15,
            "filter_orientation": False,
        },
    },
    "cfg8_best_combo": {
        "name": "Config 8: Candidate Best Combo (inliers=6, ratio=0.25, strict flip=10, orientation=True)",
        "params": {
            "ratio_threshold": 0.70,
            "min_inliers": 6,
            "min_inlier_ratio": 0.25,
            "min_inliers_flip": 10,
            "check_flip": True,
            "use_convex_hull": True,
            "mask_radius": 12,
            "filter_orientation": True,
            "orientation_tolerance": 20.0,
        },
    },
}


def find_category_path(category: str, filename: str) -> str:
    p = os.path.join(DATA_DIR, category, filename)
    if os.path.exists(p):
        return p
    p = os.path.join(DATA_DIR, filename)
    return p if os.path.exists(p) else None


def evaluate_single_config(
    cfg_key: str,
    cfg_info: dict,
    cat_ranges: dict,
    orig_range: Tuple[int, int],
    split_name: str,
) -> dict:
    params = cfg_info["params"]
    print(f"\n{'='*70}")
    print(f"RUNNING: {cfg_info['name']} on [{split_name.upper()} SPLIT]")
    print(f"Params: {params}")
    print(f"{'='*70}")

    summary = {
        "cfg_key": cfg_key,
        "cfg_name": cfg_info["name"],
        "split": split_name,
        "categories": {},
        "forged_detected": 0,
        "forged_total": 0,
        "forged_mean_f1": 0.0,
        "orig_flagged": 0,
        "orig_total": 0,
        "orig_mean_fp_area": 0.0,
        "orig_median_fp_area": 0.0,
        "orig_max_fp_area": 0,
        "orig_mean_pixel_far": 0.0,
    }

    total_f1_list = []
    total_det_count = 0
    total_img_count = 0

    # 1. Evaluate forged categories
    for cat_name, (s_idx, e_idx) in cat_ranges.items():
        cat_p_list = []
        cat_r_list = []
        cat_f1_list = []
        cat_det = 0
        cat_total = e_idx - s_idx + 1

        for i in range(s_idx, e_idx + 1):
            tag = f"{i:03d}"
            f_path = find_category_path(cat_name, f"{tag}_F.png")
            if not f_path:
                continue

            gray = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue

            m_path = find_category_path(cat_name, f"{tag}_M.png")
            gt = None
            if m_path:
                raw_m = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
                if raw_m is not None:
                    gt = (raw_m > 10).astype(np.uint8) * 255

            mask, inl1, inl2, _ = detect_sift(gray, **params)

            if gt is not None:
                p, r, f1 = scores(mask, gt)
                cat_p_list.append(p)
                cat_r_list.append(r)
                cat_f1_list.append(f1)
                if f1 > 0.3:
                    cat_det += 1
            else:
                cat_p_list.append(0.0)
                cat_r_list.append(0.0)
                cat_f1_list.append(0.0)

        mean_p = float(np.mean(cat_p_list)) if cat_p_list else 0.0
        mean_r = float(np.mean(cat_r_list)) if cat_r_list else 0.0
        mean_f1 = float(np.mean(cat_f1_list)) if cat_f1_list else 0.0

        summary["categories"][cat_name] = {
            "detected": cat_det,
            "total": cat_total,
            "det_rate": round(cat_det / max(1, cat_total) * 100.0, 1),
            "precision": round(mean_p, 3),
            "recall": round(mean_r, 3),
            "f1": round(mean_f1, 3),
        }

        total_det_count += cat_det
        total_img_count += cat_total
        total_f1_list.extend(cat_f1_list)

        print(
            f"  {cat_name.title():<12}: Det {cat_det:2d}/{cat_total:2d} "
            f"({cat_det/cat_total*100.0:5.1f}%) | "
            f"P={mean_p:.3f} R={mean_r:.3f} F1={mean_f1:.3f}"
        )

    summary["forged_detected"] = total_det_count
    summary["forged_total"] = total_img_count
    summary["forged_mean_f1"] = round(float(np.mean(total_f1_list)), 3) if total_f1_list else 0.0

    # 2. Evaluate clean original images for false alarms
    orig_s, orig_e = orig_range
    orig_total = orig_e - orig_s + 1
    orig_flagged = 0
    fp_areas = []
    pixel_far_rates = []

    for i in range(orig_s, orig_e + 1):
        tag = f"{i:03d}"
        o_path = find_category_path("translation", f"{tag}_O.png")
        if not o_path:
            o_path = find_category_path("originals", f"{tag}_O.png")
        if not o_path:
            continue

        gray = cv2.imread(o_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue

        mask, inl1, inl2, _ = detect_sift(gray, **params)
        n_flagged = int(np.sum(mask > 0))
        h, w = gray.shape
        far_pct = (n_flagged / (h * w)) * 100.0

        fp_areas.append(n_flagged)
        pixel_far_rates.append(far_pct)
        if n_flagged > 0:
            orig_flagged += 1

    summary["orig_flagged"] = orig_flagged
    summary["orig_total"] = orig_total
    summary["orig_mean_fp_area"] = round(float(np.mean(fp_areas)), 1) if fp_areas else 0.0
    summary["orig_median_fp_area"] = round(float(np.median(fp_areas)), 1) if fp_areas else 0.0
    summary["orig_max_fp_area"] = int(np.max(fp_areas)) if fp_areas else 0
    summary["orig_mean_pixel_far"] = round(float(np.mean(pixel_far_rates)), 4) if pixel_far_rates else 0.0

    print(f"\n  ORIGINALS FALSE ALARMS: {orig_flagged}/{orig_total} flagged ({orig_flagged/max(1,orig_total)*100:.1f}%)")
    print(f"  FP Area: Mean={summary['orig_mean_fp_area']} px, Median={summary['orig_median_fp_area']} px, Max={summary['orig_max_fp_area']} px")
    print(f"  Pixel False Alarm Rate: {summary['orig_mean_pixel_far']:.4f}%")
    print(f"  Overall Forged Detection: {total_det_count}/{total_img_count} ({total_det_count/max(1,total_img_count)*100:.1f}%), Mean F1: {summary['forged_mean_f1']:.3f}")

    return summary


def write_ablation_markdown(summaries: List[dict], split_name: str, filepath: str):
    with open(filepath, "w") as f:
        f.write(f"# SIFT Calibration and Ablation Study: {split_name.title()} Split\n\n")
        f.write(f"Protocol: 50/50 partition. {split_name.title()} split evaluates 20 images per category (100 forged images) and 20 clean originals.\n\n")

        # Master comparison table
        f.write("## Overall Comparison Table\n\n")
        f.write("| Configuration | Overall Det (100) | Overall F1 | Clean Flagged (20) | Clean Max FP Area | Mean Pixel FAR |\n")
        f.write("|---|---|---|---|---|---|\n")
        for s in summaries:
            det_str = f"{s['forged_detected']}/{s['forged_total']} ({s['forged_detected']/max(1, s['forged_total'])*100:.1f}%)"
            clean_str = f"{s['orig_flagged']}/{s['orig_total']} ({s['orig_flagged']/max(1, s['orig_total'])*100:.1f}%)"
            f.write(
                f"| **{s['cfg_name']}** | {det_str} | {s['forged_mean_f1']:.3f} | {clean_str} | {s['orig_max_fp_area']} px | {s['orig_mean_pixel_far']:.4f}% |\n"
            )

        f.write("\n## Per-Category Breakdown (Detection / F1)\n\n")
        f.write("| Configuration | Translation | Rotation | Scaling | Distortion | Combination |\n")
        f.write("|---|---|---|---|---|---|\n")
        for s in summaries:
            cats = s["categories"]
            tr = f"{cats.get('translation', {}).get('detected', 0)}/20 (F1 {cats.get('translation', {}).get('f1', 0.0):.3f})"
            ro = f"{cats.get('rotation', {}).get('detected', 0)}/20 (F1 {cats.get('rotation', {}).get('f1', 0.0):.3f})"
            sc = f"{cats.get('scaling', {}).get('detected', 0)}/20 (F1 {cats.get('scaling', {}).get('f1', 0.0):.3f})"
            di = f"{cats.get('distortion', {}).get('detected', 0)}/20 (F1 {cats.get('distortion', {}).get('f1', 0.0):.3f})"
            co = f"{cats.get('combination', {}).get('detected', 0)}/20 (F1 {cats.get('combination', {}).get('f1', 0.0):.3f})"
            f.write(f"| **{s['cfg_name']}** | {tr} | {ro} | {sc} | {di} | {co} |\n")

        f.write("\n## False Positive Area Distribution on Clean Images\n\n")
        f.write("| Configuration | Images Flagged (20) | Mean FP Area | Median FP Area | Max FP Area | Pixel FAR |\n")
        f.write("|---|---|---|---|---|---|\n")
        for s in summaries:
            f.write(
                f"| **{s['cfg_name']}** | {s['orig_flagged']}/20 | {s['orig_mean_fp_area']} px | {s['orig_median_fp_area']} px | {s['orig_max_fp_area']} px | {s['orig_mean_pixel_far']:.4f}% |\n"
            )


def write_ablation_csv(summaries: List[dict], filepath: str):
    fieldnames = [
        "cfg_key",
        "cfg_name",
        "split",
        "forged_detected",
        "forged_total",
        "forged_det_rate",
        "forged_mean_f1",
        "trans_det",
        "trans_f1",
        "rot_det",
        "rot_f1",
        "scale_det",
        "scale_f1",
        "dist_det",
        "dist_f1",
        "comb_det",
        "comb_f1",
        "orig_flagged",
        "orig_total",
        "orig_mean_fp_area",
        "orig_median_fp_area",
        "orig_max_fp_area",
        "orig_mean_pixel_far",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            cats = s["categories"]
            row = {
                "cfg_key": s["cfg_key"],
                "cfg_name": s["cfg_name"],
                "split": s["split"],
                "forged_detected": s["forged_detected"],
                "forged_total": s["forged_total"],
                "forged_det_rate": round(s["forged_detected"] / max(1, s["forged_total"]) * 100.0, 1),
                "forged_mean_f1": s["forged_mean_f1"],
                "trans_det": cats.get("translation", {}).get("detected", 0),
                "trans_f1": cats.get("translation", {}).get("f1", 0.0),
                "rot_det": cats.get("rotation", {}).get("detected", 0),
                "rot_f1": cats.get("rotation", {}).get("f1", 0.0),
                "scale_det": cats.get("scaling", {}).get("detected", 0),
                "scale_f1": cats.get("scaling", {}).get("f1", 0.0),
                "dist_det": cats.get("distortion", {}).get("detected", 0),
                "dist_f1": cats.get("distortion", {}).get("f1", 0.0),
                "comb_det": cats.get("combination", {}).get("detected", 0),
                "comb_f1": cats.get("combination", {}).get("f1", 0.0),
                "orig_flagged": s["orig_flagged"],
                "orig_total": s["orig_total"],
                "orig_mean_fp_area": s["orig_mean_fp_area"],
                "orig_median_fp_area": s["orig_median_fp_area"],
                "orig_max_fp_area": s["orig_max_fp_area"],
                "orig_mean_pixel_far": s["orig_mean_pixel_far"],
            }
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="SIFT Calibration and Ablation Study")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Run a specific configuration key (e.g. cfg0_baseline, cfg8_best_combo) or omit to run all.",
    )
    parser.add_argument(
        "--held-out",
        action="store_true",
        help="Evaluate on the held-out split (images 21-40) instead of the tuning split (images 1-20).",
    )
    args = parser.parse_args()

    split_name = "held_out" if args.held_out else "tuning"
    cat_ranges = HELDOUT_CATEGORIES if args.held_out else TUNING_CATEGORIES
    orig_range = (21, 40) if args.held_out else (1, 20)

    if args.config:
        if args.config not in CONFIGURATIONS:
            print(f"Error: Unknown configuration '{args.config}'. Available: {list(CONFIGURATIONS.keys())}")
            sys.exit(1)
        configs_to_run = {args.config: CONFIGURATIONS[args.config]}
    else:
        configs_to_run = CONFIGURATIONS

    summaries = []
    t_start = time.time()
    for key, info in configs_to_run.items():
        summary = evaluate_single_config(key, info, cat_ranges, orig_range, split_name)
        summaries.append(summary)

    total_dur = round(time.time() - t_start, 1)
    print(f"\nAll configurations evaluated in {total_dur}s.")

    md_path = os.path.join(OUT_DIR, f"ablation_{split_name}_split.md")
    csv_path = os.path.join(OUT_DIR, f"ablation_{split_name}_split.csv")

    write_ablation_markdown(summaries, split_name, md_path)
    write_ablation_csv(summaries, csv_path)

    print(f"\nAblation markdown report saved to: {md_path}")
    print(f"Ablation CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
