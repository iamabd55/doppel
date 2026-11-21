"""Batch benchmark evaluation for SIFT copy-move detector across CoMoFoD dataset.

Evaluates 5 categories (40 images each = 200 forged images) plus 40 clean originals:
1. Translation: sets 001-040
2. Rotation:    sets 041-080
3. Scaling:     sets 081-120
4. Distortion:  sets 121-160
5. Combination: sets 161-200
6. Originals:   sets 001-040 (false alarm rate test)

Outputs:
- results/phase2_sift/sift_benchmark.csv
- results/phase2_sift/benchmark_summary.md (side-by-side DCT vs SIFT comparison)
"""
import argparse
import csv
import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detector.sift import detect_sift
from detector.evaluate import scores

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "comofod")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "phase2_sift")
os.makedirs(OUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUT_DIR, "sift_benchmark.csv")
SUMMARY_PATH = os.path.join(OUT_DIR, "benchmark_summary.md")

CATEGORY_RANGES = {
    "translation": (1, 40),
    "rotation": (41, 80),
    "scaling": (81, 120),
    "distortion": (121, 160),
    "combination": (161, 200),
}

# Frozen DCT baseline numbers from section 4 of DOPPEL_AGENT.md
DCT_BASELINE = {
    "translation": {"det": "36/40 (90.0%)", "f1": 0.817},
    "rotation":    {"det": "5/40 (12.5%)",  "f1": 0.082},
    "scaling":     {"det": "1/40 (2.5%)",   "f1": 0.021},
    "distortion":  {"det": "8/40 (20.0%)",  "f1": 0.145},
    "combination": {"det": "1/40 (2.5%)",   "f1": 0.018},
    "originals":   {"det": "0/40 (0.0%)",   "f1": 0.000},
}


def find_category_path(category, filename):
    p = os.path.join(DATA_DIR, category, filename)
    if os.path.exists(p):
        return p
    # Search root data directory as fallback
    p = os.path.join(DATA_DIR, filename)
    return p if os.path.exists(p) else None


def evaluate_category(cat_name, start_idx, end_idx):
    print(f"\nEvaluating Category: {cat_name.upper()} (images {start_idx:03d} to {end_idx:03d})")
    print(f"{'img':>5} | {'P(M)':>6} {'R(M)':>6} {'F1(M)':>6} | {'Inliers':>8} {'Flagged':>8} | {'time':>5} | {'Status'}")
    print("-" * 65)

    results = []
    total_time = 0.0

    for i in range(start_idx, end_idx + 1):
        tag = f"{i:03d}"
        f_path = find_category_path(cat_name, f"{tag}_F.png")
        if not f_path:
            print(f"{tag:>5} | SKIP (image not found)")
            continue

        gray = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"{tag:>5} | SKIP (could not read)")
            continue

        # Load ground truth mask M (covers both source and paste)
        m_path = find_category_path(cat_name, f"{tag}_M.png")
        gt = None
        if m_path:
            raw_m = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            if raw_m is not None:
                gt = (raw_m > 10).astype(np.uint8) * 255

        t0 = time.time()
        mask, inl1, inl2, transforms = detect_sift(
            gray,
            ratio_threshold=0.70,
            min_dist=30.0,
            ransac_threshold=3.0,
            min_inliers=4,
            cluster_threshold=100.0,
            mask_radius=12,
            use_convex_hull=True,
            check_flip=True,
        )
        dt = time.time() - t0
        total_time += dt

        n_inliers = len(inl1)
        flagged_px = int((mask > 0).sum())

        if gt is not None:
            p, r, f1 = scores(mask, gt)
        else:
            p, r, f1 = 0.0, 0.0, 0.0

        detected = (f1 > 0.3)
        status = "DETECTED" if detected else "MISSED"

        results.append({
            "category": cat_name,
            "image": f"{tag}_F.png",
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "flagged_pixels": flagged_px,
            "inliers": n_inliers,
            "time_seconds": round(dt, 3),
            "detected": detected,
        })

        print(f"{tag:>5} | {p:6.3f} {r:6.3f} {f1:6.3f} | {n_inliers:8d} {flagged_px:8d} | {dt:4.1f}s | {status}")

    arr_f1 = np.array([r["f1"] for r in results]) if results else np.zeros(1)
    arr_p = np.array([r["precision"] for r in results]) if results else np.zeros(1)
    arr_r = np.array([r["recall"] for r in results]) if results else np.zeros(1)
    n_detected = sum(1 for r in results if r["detected"])
    total_imgs = len(results)

    print("-" * 65)
    print(f"Summary for {cat_name.upper()}:")
    print(f"  Detected (F1 > 0.3): {n_detected}/{total_imgs} ({100.0 * n_detected / max(1, total_imgs):.1f}%)")
    print(f"  Average Precision:   {arr_p.mean():.3f}")
    print(f"  Average Recall:      {arr_r.mean():.3f}")
    print(f"  Average F1:          {arr_f1.mean():.3f}")
    print(f"  Total time:          {total_time:.1f}s ({total_time / max(1, total_imgs):.2f}s per image)")

    return results, {
        "category": cat_name,
        "detected_count": n_detected,
        "total_count": total_imgs,
        "detection_rate": round(100.0 * n_detected / max(1, total_imgs), 1),
        "avg_precision": round(float(arr_p.mean()), 3),
        "avg_recall": round(float(arr_r.mean()), 3),
        "avg_f1": round(float(arr_f1.mean()), 3),
        "total_time": round(total_time, 1),
    }


def evaluate_originals():
    print(f"\nEvaluating False Alarm Rate on 40 Clean Original Images (001_O to 040_O)")
    print(f"{'img':>5} | {'Inliers':>8} {'Flagged':>8} {'% of Image':>11} | {'time':>5} | {'Alarm'}")
    print("-" * 55)

    results = []
    total_time = 0.0
    false_alarm_count = 0
    total_pixels = 0
    total_false_pixels = 0

    for i in range(1, 41):
        tag = f"{i:03d}"
        o_path = find_category_path("translation", f"{tag}_O.png")
        if not o_path:
            print(f"{tag:>5} | SKIP (image not found)")
            continue

        gray = cv2.imread(o_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"{tag:>5} | SKIP (could not read)")
            continue

        t0 = time.time()
        mask, inl1, inl2, transforms = detect_sift(
            gray,
            ratio_threshold=0.70,
            min_dist=30.0,
            ransac_threshold=3.0,
            min_inliers=4,
            cluster_threshold=100.0,
            mask_radius=12,
            use_convex_hull=True,
            check_flip=True,
        )
        dt = time.time() - t0
        total_time += dt

        flagged_px = int((mask > 0).sum())
        n_inliers = len(inl1)
        pct = 100.0 * flagged_px / gray.size

        total_pixels += gray.size
        total_false_pixels += flagged_px
        is_alarm = flagged_px > 0
        if is_alarm:
            false_alarm_count += 1

        status = "FALSE ALARM" if is_alarm else "CLEAN"

        results.append({
            "category": "originals",
            "image": f"{tag}_O.png",
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "flagged_pixels": flagged_px,
            "inliers": n_inliers,
            "time_seconds": round(dt, 3),
            "detected": is_alarm,
        })

        print(f"{tag:>5} | {n_inliers:8d} {flagged_px:8d} {pct:10.3f}% | {dt:4.1f}s | {status}")

    total_imgs = len(results)
    avg_far = 100.0 * total_false_pixels / max(1, total_pixels)

    print("-" * 55)
    print(f"Summary for ORIGINALS (False Alarm Test):")
    print(f"  Images with false alarms: {false_alarm_count}/{total_imgs} ({100.0 * false_alarm_count / max(1, total_imgs):.1f}%)")
    print(f"  Total false alarm pixels: {total_false_pixels}")
    print(f"  Average pixel false alarm rate: {avg_far:.4f}%")
    print(f"  Total time: {total_time:.1f}s")

    return results, {
        "category": "originals",
        "detected_count": false_alarm_count,
        "total_count": total_imgs,
        "detection_rate": round(100.0 * false_alarm_count / max(1, total_imgs), 1),
        "avg_precision": 0.0,
        "avg_recall": 0.0,
        "avg_f1": 0.0,
        "total_time": round(total_time, 1),
        "pixel_far": round(avg_far, 4),
    }


def write_outputs(all_results, category_summaries):
    # 1. Write per-image CSV
    fieldnames = [
        "category", "image", "precision", "recall", "f1",
        "flagged_pixels", "inliers", "time_seconds", "detected",
    ]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    print(f"\nDetailed CSV written to: {os.path.relpath(CSV_PATH)}")

    # 2. Write Markdown side-by-side comparison summary
    lines = [
        "# Phase 2 SIFT Benchmark Results (CoMoFoD Small, 40 sets per category)\n",
        "Direct comparison between the frozen DCT baseline and the new SIFT detector.\n",
        "| Category | DCT Detection | DCT Avg F1 | SIFT Detection | SIFT Avg F1 | SIFT Avg Precision | SIFT Avg Recall |",
        "|---|---|---|---|---|---|---|",
    ]

    for cat in ["translation", "rotation", "scaling", "distortion", "combination"]:
        dct = DCT_BASELINE[cat]
        sift = category_summaries.get(cat, {})
        s_det = f"{sift.get('detected_count', 0)}/{sift.get('total_count', 40)} ({sift.get('detection_rate', 0.0):.1f}%)"
        s_f1 = f"{sift.get('avg_f1', 0.0):.3f}"
        s_p = f"{sift.get('avg_precision', 0.0):.3f}"
        s_r = f"{sift.get('avg_recall', 0.0):.3f}"
        lines.append(f"| {cat.capitalize()} | {dct['det']} | {dct['f1']:.3f} | {s_det} | {s_f1} | {s_p} | {s_r} |")

    lines.append("\n### False Alarm Comparison (40 Untouched Original Images)")
    orig_sift = category_summaries.get("originals", {})
    s_orig_det = f"{orig_sift.get('detected_count', 0)}/{orig_sift.get('total_count', 40)} ({orig_sift.get('detection_rate', 0.0):.1f}%)"
    lines.append(f"- **DCT Baseline**: 0/40 (0.0000% pixel false alarm rate with texture filter)")
    lines.append(f"- **SIFT Detector**: {s_orig_det} (average pixel false alarm rate: {orig_sift.get('pixel_far', 0.0):.4f}%)")

    content = "\n".join(lines) + "\n"
    with open(SUMMARY_PATH, "w") as f:
        f.write(content)
    print(f"Markdown summary written to: {os.path.relpath(SUMMARY_PATH)}")


def main():
    parser = argparse.ArgumentParser(description="Batch test SIFT detector on CoMoFoD.")
    parser.add_argument(
        "--category",
        choices=["all", "translation", "rotation", "scaling", "distortion", "combination", "originals"],
        default="all",
        help="Category to test (default: all)",
    )
    args = parser.parse_args()

    all_results = []
    category_summaries = {}

    if args.category == "all":
        targets = list(CATEGORY_RANGES.keys()) + ["originals"]
    else:
        targets = [args.category]

    for target in targets:
        if target == "originals":
            res, summ = evaluate_originals()
        else:
            s_idx, e_idx = CATEGORY_RANGES[target]
            res, summ = evaluate_category(target, s_idx, e_idx)
        all_results.extend(res)
        category_summaries[target] = summ

    write_outputs(all_results, category_summaries)


if __name__ == "__main__":
    main()
