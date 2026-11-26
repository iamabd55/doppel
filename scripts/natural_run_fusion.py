"""Fit DOA-GAN on CASIA2 calibration, evaluate fusion, or predict one image."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector.natural import (
    CLASSICAL_CONFIG, DOAGAN_CONFIG, Detection, NaturalPipeline,
    classical_masks, classical_parameters, from_mask, fuse,
)
from natural_aggregate import ci_str
from natural_run_classical import cache_paths
from natural_run_doagan import CACHE_ROOT, MANIFESTS, SPLITS, result_path

BUDGETS = {"1%": 0.01, "2%": 0.02, "5%": 0.05}


def manifest_rows(split):
    dataset = SPLITS[split]
    manifest = pd.read_csv(MANIFESTS / f"{dataset}.csv")
    rows = manifest[manifest["split"].eq(split)].to_dict("records")
    if not rows:
        raise ValueError(f"No images for {dataset}/{split}")
    return rows


def doagan_cache(row):
    path = result_path(row["dataset"], row["split"], row["image_path"])
    with np.load(path) as data:
        return float(data["score"]), data["mask"].astype(bool, copy=False)


def fit_thresholds(scored_labels):
    positives = sum(forged for _, forged in scored_labels)
    negatives = len(scored_labels) - positives
    if not positives or not negatives:
        raise ValueError("Threshold calibration requires forged and authentic images")
    grid = {}
    for score, forged in scored_labels:
        if not 0 <= score <= 1:
            raise ValueError(f"Invalid DOA-GAN score {score}")
        counts = grid.setdefault(score, [0, 0])
        counts[0 if forged else 1] += 1

    above_one = float(np.nextafter(1.0, np.inf))
    best = {name: (0, 0, above_one) for name in BUDGETS}
    tp = fp = 0
    for threshold, (new_tp, new_fp) in sorted(grid.items(), reverse=True):
        tp += new_tp
        fp += new_fp
        for name, budget in BUDGETS.items():
            result = (tp, -fp, threshold)
            if fp / negatives <= budget and result > best[name]:
                best[name] = result
    return {
        name: {"threshold": threshold, "tp": tp, "fp": -negative_fp,
               "recall": tp / positives, "far": -negative_fp / negatives}
        for name, (tp, negative_fp, threshold) in best.items()
    }


def require_doagan_cache(rows):
    missing = sum(not result_path(row["dataset"], row["split"], row["image_path"]).exists()
                  for row in rows)
    if missing:
        raise ValueError(f"{missing}/{len(rows)} DOA-GAN images still need inference")


def calibrate():
    rows = manifest_rows("calibration")
    require_doagan_cache(rows)
    scored_labels = [(doagan_cache(row)[0], row["label"] == "forged") for row in rows]
    config = {
        "checkpoint": "doa-gan_weights/single_channel.pkl",
        "calibration_split": "casia2/calibration",
        "image_score": "fraction above 0.5 at model resolution",
        "budgets": fit_thresholds(scored_labels),
    }
    DOAGAN_CONFIG.write_text(yaml.safe_dump(config, sort_keys=False))
    for name, result in config["budgets"].items():
        print(f"[DOA-GAN] {name} budget: threshold={result['threshold']:.8f}, "
              f"recall={result['recall']:.2%}, FAR={result['far']:.2%}")


def frozen_detection(row, detector):
    json_path, mask_path = cache_paths(
        CACHE_ROOT, detector, row["dataset"], row["split"], row["image_path"],
    )
    cached = json.loads(json_path.read_text())
    if cached["image_path"] != row["image_path"]:
        raise ValueError(f"Cache key collision at {json_path}")
    with np.load(mask_path) as data:
        return Detection(bool(cached["decision"]), data["mask"] > 0)


def classical_detections(row, config):
    if config is None:
        return frozen_detection(row, "dct"), frozen_detection(row, "sift")
    image = cv2.imread(str(ROOT / row["image_path"]), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Could not read {row['image_path']}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dct, sift = (from_mask(mask) for mask in classical_masks(gray, *config))
    return dct, sift


def doagan_detections(score, mask, thresholds):
    empty = np.zeros_like(mask)
    return {
        budget: Detection(score >= threshold, mask if score >= threshold else empty)
        for budget, threshold in thresholds.items()
    }


def ground_truth(row, shape):
    if row["label"] == "authentic":
        return None
    gt = cv2.imread(str(ROOT / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
    if gt is None or gt.shape != shape:
        raise ValueError(f"Missing or misaligned ground truth for {row['image_path']}")
    gt = gt > 0
    if not gt.any():
        raise ValueError(f"Empty forged ground truth for {row['image_path']}")
    return gt


def score_detections(detections, gt):
    if gt is None:
        return {name: (int(det.flagged), 0, 0.0, 0.0) for name, det in detections.items()}

    gt_count = np.count_nonzero(gt)
    metrics = {}
    for name, det in detections.items():
        tp = np.count_nonzero(gt & det.mask)
        predicted = np.count_nonzero(det.mask)
        metrics[name] = (
            int(det.flagged), int(tp > 0), 2 * tp / (gt_count + predicted),
            tp / (gt_count + predicted - tp),
        )
    return metrics


def evaluate_one(row, *, config, thresholds):
    dct, sift = classical_detections(row, config)
    score, mask = doagan_cache(row)
    doagan = doagan_detections(score, mask, thresholds)
    detections = (
        {"DCT": dct, "SIFT": sift}
        | {f"DOA-GAN ({budget} budget)": result for budget, result in doagan.items()}
        | fuse(dct, sift, doagan["2%"])
    )
    gt = ground_truth(row, mask.shape)
    return gt is not None, score_detections(detections, gt)


def evaluate(split, calibrated, workers):
    rows = manifest_rows(split)
    require_doagan_cache(rows)
    budgets = yaml.safe_load(DOAGAN_CONFIG.read_text())["budgets"]
    thresholds = {budget: budgets[budget]["threshold"] for budget in BUDGETS}
    parameters = classical_parameters(CLASSICAL_CONFIG) if calibrated else None
    variant = "calibrated" if calibrated else "frozen"
    run_one = partial(evaluate_one, config=parameters, thresholds=thresholds)
    cv2.setNumThreads(1)
    totals = {}
    n_forged = 0
    print(f"[fusion] {SPLITS[split]}/{split}, {variant} classical detectors, {len(rows)} images")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = pool.map(run_one, rows, chunksize=8)
        for index, (forged, metrics) in enumerate(results, 1):
            n_forged += forged
            for name, (flagged, overlap, f1, iou) in metrics.items():
                total = totals.setdefault(name, np.zeros(5))
                total += (flagged if forged else 0, flagged if not forged else 0, overlap, f1, iou)
            if index % 100 == 0 or index == len(rows):
                print(f"[fusion] {index}/{len(rows)} processed")

    n_authentic = len(rows) - n_forged
    table = [
        f"# Fusion with {variant} classical detectors on {SPLITS[split]} {split}",
        "",
        f"Fusion uses the DOA-GAN threshold set at the CASIA2 2% budget ({thresholds['2%']:.8f}).",
        ("CASIA2 localization uses target-only ground truth." if SPLITS[split] == "casia2"
         else "CoMoFoD localization uses combined source and target ground truth."),
        f"Pixel F1 and IoU are macro-averages over {n_forged} forged images. "
        f"FAR uses {n_authentic} authentic images.",
        "",
        "| Method | Recall (95% CI) | FAR (95% CI) | Overlap recall (95% CI) | Pixel F1 | IoU |",
        "|---|---|---|---|---:|---:|",
    ]
    for name, (tp, fp, overlap, f1, iou) in totals.items():
        table.append(f"| {name} | {ci_str(int(tp), n_forged)} | {ci_str(int(fp), n_authentic)} | "
                     f"{ci_str(int(overlap), n_forged)} | {f1 / n_forged:.4f} | {iou / n_forged:.4f} |")
    report = ROOT / "docs" / f"natural_fusion_{split}_{variant}.md"
    report.write_text("\n".join(table) + "\n")
    print("\n".join(table))
    print(f"[fusion] Wrote {report}")


def predict(path, frozen, masks_out, smoke_threshold):
    import torch
    from detector.doagan import read_rgb

    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    pipeline = NaturalPipeline(classical_config=None if frozen else CLASSICAL_CONFIG,
                               doagan_threshold=smoke_threshold)
    result = pipeline.predict(read_rgb(path))
    if masks_out:
        np.savez_compressed(masks_out, **{name: detection.mask for name, detection in
                                       (result.detectors | result.fusion).items()})
    print(json.dumps({
        "doagan_score": result.doagan_score,
        "doagan_threshold": pipeline.threshold,
        "threshold_source": "CASIA2 calibration" if smoke_threshold is None else "smoke override",
        "detectors": {name: value.flagged for name, value in result.detectors.items()},
        "fusion": {name: value.flagged for name, value in result.fusion.items()},
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("calibrate")
    evaluation = subcommands.add_parser("evaluate")
    evaluation.add_argument("--split", choices=SPLITS, default="calibration")
    evaluation.add_argument("--final", action="store_true", help="Unlock the reserved test pass")
    evaluation.add_argument("--calibrated", action="store_true")
    evaluation.add_argument("--workers", type=int, default=6)
    prediction = subcommands.add_parser("predict")
    prediction.add_argument("image", type=Path)
    prediction.add_argument("--frozen", action="store_true")
    prediction.add_argument("--smoke-threshold", type=float,
                            help="Use an explicit threshold for an exploratory single-image smoke run")
    prediction.add_argument("--masks-out", type=Path)
    args = parser.parse_args()
    if args.command == "calibrate":
        calibrate()
    elif args.command == "evaluate":
        if args.split == "test" and not args.final:
            parser.error("The test split requires --final")
        if args.workers < 1:
            parser.error("workers must be positive")
        evaluate(args.split, args.calibrated, args.workers)
    else:
        predict(args.image, args.frozen, args.masks_out, args.smoke_threshold)


if __name__ == "__main__":
    main()
