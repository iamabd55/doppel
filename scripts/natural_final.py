"""Build the locked natural-image evaluation report and held-out montages."""

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

import cv2
import numpy as np
import yaml

from detector.doagan import UPSTREAM, WEIGHTS
from detector.natural import (
    CLASSICAL_CONFIG, DOAGAN_CONFIG, ROOT, Detection, classical_parameters, fuse,
)
from natural_aggregate import wilson_ci
from natural_run_doagan import load_comofod_subset
from natural_run_fusion import (
    classical_detections, doagan_cache, doagan_detections, frozen_detection,
    ground_truth, manifest_rows, require_doagan_cache, score_detections,
)

RESULTS = ROOT / "results" / "natural"
REPORT = ROOT / "docs" / "natural_results.md"
FILE_HASHES = {
    CLASSICAL_CONFIG: "e338ee56b65037e349b41f3a00935806653ce533651bf5b78e883ab7a3f7c51e",
    DOAGAN_CONFIG: "563b25ef232d4ee5dd5c7d5a5774a57f894671af337bd5658792a0a2047b5e99",
    WEIGHTS / "single_channel.pkl": "ffa14ea9a47c9779d9a013a150521247b3006ed5dddee900d18b4bc420f0edb5",
}
DEFAULT_HASHES = (
    "eadb5a4b30db7d9b30dec8f695c809e79fbf20fed91f193325ebec26276de0d6",
    "9b55dcdc1228237a6fa44c888ec7d48d5f4af393b3411705d847bdadb4973c29",
)
UPSTREAM_COMMIT = "081cc88aa1b67c250290311e78a7c4f101fc28d7"
SIZE_BINS = ("Small (<400 px)", "Medium (400–800 px)", "Large (>800 px)")
FUSION_RULES = {
    "F1": "DCT OR SIFT",
    "F2": "DCT AND SIFT",
    "F3": "DCT OR SIFT OR DOA-GAN",
    "F4": "Three-detector majority",
    "F5": "(DCT OR SIFT) AND DOA-GAN",
}
MONTAGE_DETECTORS = {"dct": "DCT (frozen)", "sift": "SIFT (frozen)", "doagan": "DOA-GAN (2%)"}


def verify_settings():
    for path, expected in FILE_HASHES.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Locked artifact hash changed: {path}")
    for params, expected in zip(classical_parameters(), DEFAULT_HASHES, strict=True):
        encoded = json.dumps(params, separators=(",", ":"), sort_keys=True).encode()
        if hashlib.sha256(encoded).hexdigest() != expected:
            raise ValueError("A frozen classical parameter set changed")
    revision = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    changed = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "status", "--short", "--", "models.py"], text=True,
    )
    if revision != UPSTREAM_COMMIT or changed:
        raise ValueError("DOA-GAN model source differs from the locked upstream revision")
    config = yaml.safe_load(DOAGAN_CONFIG.read_text())
    return {name: result["threshold"] for name, result in config["budgets"].items()}


def infer_doagan(split):
    command = [sys.executable, "-u", str(ROOT / "scripts" / "natural_run_doagan.py"), "--split", split]
    if split == "test":
        command.append("--final")
    subprocess.run(command, cwd=ROOT, check=True)


def score_image(item, *, thresholds, calibrated):
    row, include_doagan = item
    frozen = classical_detections(row, None)
    shape = frozen[0].mask.shape
    truth = ground_truth(row, shape)
    detections = {}
    primary = Detection(False, np.zeros(shape, dtype=bool))
    if include_doagan:
        score, mask = doagan_cache(row)
        doagan = doagan_detections(score, mask, thresholds)
        detections.update({f"DOA-GAN ({budget})": detection for budget, detection in doagan.items()})
        primary = doagan["2%"]
    calibrated_pair = classical_detections(row, calibrated)
    for variant, (dct, sift) in (("frozen", frozen), ("calibrated", calibrated_pair)):
        detections[f"DCT ({variant})"] = dct
        detections[f"SIFT ({variant})"] = sift
        detections.update({f"{rule} ({variant})": result
                           for rule, result in fuse(dct, sift, primary).items()
                           if include_doagan or rule in ("F1", "F2")})
    shorter = min(shape)
    size = 0 if shorter < 400 else 2 if shorter > 800 else 1
    return size, score_detections(detections, truth)


class Summary:
    def __init__(self):
        self.forged = 0
        self.authentic = 0
        self.totals = {}

    def add(self, forged, metrics):
        self.forged += forged
        self.authentic += not forged
        for method, (flagged, overlap, f1, iou) in metrics.items():
            total = self.totals.setdefault(method, np.zeros(5))
            total += (flagged and forged, flagged and not forged, overlap, f1, iou)


def analyze_split(split, rows, thresholds, calibrated, subset=None):
    items = [(row, subset is None or row["image_path"] in subset) for row in rows]
    require_doagan_cache([row for row, included in items if included])
    overall = Summary()
    sizes = [Summary() for _ in SIZE_BINS]
    attacks = defaultdict(Summary)
    examples = defaultdict(list)
    subset_overall = Summary()
    subset_sizes = [Summary() for _ in SIZE_BINS]
    subset_attacks = defaultdict(Summary)
    classical = set(classical_method_order())
    worker = partial(score_image, thresholds=thresholds, calibrated=calibrated)
    print(f"[final] Measuring {split} on {len(rows)} images")
    with ProcessPoolExecutor(max_workers=6) as pool:
        scored = pool.map(worker, items, chunksize=8)
        for index, ((row, included), (size, metrics)) in enumerate(zip(items, scored, strict=True), 1):
            forged = row["label"] == "forged"
            main_metrics = metrics
            if subset is not None:
                if included:
                    subset_overall.add(forged, metrics)
                    subset_sizes[size].add(forged, metrics)
                    subset_attacks[row["attack_type"]].add(forged, metrics)
                main_metrics = {name: values for name, values in metrics.items() if name in classical}
            overall.add(forged, main_metrics)
            sizes[size].add(forged, main_metrics)
            if split == "dev":
                attacks[row["attack_type"]].add(forged, main_metrics)
            elif split == "test" and forged:
                code = Path(row["image_path"]).stem.split("_")[2]
                attacks[code].add(forged, metrics)
                for detector, method in MONTAGE_DETECTORS.items():
                    outcome = "successes" if metrics[method][0] else "failures"
                    examples[(detector, outcome)].append(row)
            if index % 100 == 0 or index == len(rows):
                print(f"[final] {split}: {index}/{len(rows)} scored")
    return (overall, sizes, attacks, examples), (subset_overall, subset_sizes, subset_attacks, {})


def percent_ci(count, total):
    if not total:
        return "N/A"
    rate, low, high = wilson_ci(int(count), total)
    return f"{100*rate:.2f}% [{100*low:.2f}, {100*high:.2f}]"


def method_order():
    return (
        [f"{name} ({variant})" for name in ("DCT", "SIFT") for variant in ("frozen", "calibrated")]
        + [f"DOA-GAN ({budget})" for budget in ("1%", "2%", "5%")]
        + [f"{rule} ({variant})" for rule in FUSION_RULES for variant in ("frozen", "calibrated")]
    )


def classical_method_order():
    return [name for name in method_order() if name.split()[0] in ("DCT", "SIFT", "F1", "F2")]


def ablation_table(summary, method_names):
    lines = [
        "| Method | Forged | Authentic | Recall (95% CI) | FAR (95% CI) | Overlap recall (95% CI) | Pixel F1 | Pixel IoU |",
        "|---|---:|---:|---|---|---|---:|---:|",
    ]
    for name in method_names:
        tp, fp, overlap, f1, iou = summary.totals[name]
        lines.append(
            f"| {name} | {summary.forged} | {summary.authentic} | {percent_ci(tp, summary.forged)} "
            f"| {percent_ci(fp, summary.authentic)} | {percent_ci(overlap, summary.forged)} "
            f"| {f1/summary.forged:.4f} | {iou/summary.forged:.4f} |"
        )
    return lines


def write_montages(examples, thresholds):
    RESULTS.mkdir(parents=True, exist_ok=True)
    paths = []
    for detector in MONTAGE_DETECTORS:
        for outcome in ("successes", "failures"):
            candidates = examples[(detector, outcome)]
            images = random.Random(0).sample(candidates, min(8, len(candidates)))
            if not images:
                continue
            tiles = []
            for row in images:
                image = cv2.imread(str(ROOT / row["image_path"]), cv2.IMREAD_COLOR)
                if image is None:
                    raise OSError(f"Could not read montage image {row['image_path']}")
                if detector == "doagan":
                    score, mask = doagan_cache(row)
                    detection = doagan_detections(score, mask, thresholds)["2%"]
                else:
                    detection = frozen_detection(row, detector)
                truth = ground_truth(row, detection.mask.shape)
                if outcome == "successes":
                    image[detection.mask] = (image[detection.mask] * 0.55 + np.array([0, 115, 0])).astype(np.uint8)
                contours, _ = cv2.findContours(truth.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
                height, width = image.shape[:2]
                scale = min(300 / width, 222 / height)
                resized = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                                     interpolation=cv2.INTER_AREA)
                tile = np.full((248, 300, 3), 25, dtype=np.uint8)
                dx, dy = (300 - resized.shape[1]) // 2, (222 - resized.shape[0]) // 2
                tile[dy:dy + resized.shape[0], dx:dx + resized.shape[1]] = resized
                cv2.putText(tile, Path(row["image_path"]).stem[-42:], (6, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (240, 240, 240), 1, cv2.LINE_AA)
                tiles.append(tile)
            while len(tiles) < 8:
                tiles.append(np.zeros_like(tiles[0]))
            mosaic = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
            path = RESULTS / f"montage_{detector}_{outcome}.png"
            if not cv2.imwrite(str(path), mosaic):
                raise OSError(f"Could not write {path}")
            paths.append((detector, outcome, len(images), path))
    return paths


def build_report(results, montage_paths, revision):
    methods = method_order()
    classical = classical_method_order()
    test, calibration, dev, dev_subset = (
        results[split][0] for split in ("test", "calibration", "dev", "dev_subset")
    )
    lines = [
        "# Doppel natural-image evaluation",
        "",
        f"Evaluation code revision `{revision}`. All settings were locked before the held-out pass.",
        "CASIA2 calibration results are in-sample. CASIA2 test is the held-out estimate of performance.",
        "CoMoFoD dev is optimistic for DCT and SIFT because their frozen settings were originally tuned on it.",
        "CoMoFoD DOA-GAN rows and three-detector fusion use a locked stratified subset. "
        "The subset table includes the classical controls evaluated on those same images.",
        "COVERAGE was skipped because its local folder contains no dataset images and its manifest contains zero rows.",
        "",
        "Pixel F1 and IoU are macro-averaged over forged images, including missed detections.",
        "CASIA2 uses target-only ground truth. CoMoFoD uses combined source and target. "
        "All proportion intervals are Wilson 95% confidence intervals.",
        "",
        "## CASIA2 held-out test ablation",
        "",
        *ablation_table(test, methods),
        "",
        "## CASIA2 calibration ablation (in-sample)",
        "",
        *ablation_table(calibration, methods),
        "",
        "## CoMoFoD dev ablation, full 6,400 images (optimistic for DCT and SIFT)",
        "",
        *ablation_table(dev, classical),
        "",
        "## CoMoFoD dev ablation, matched 800 images (optimistic for DCT and SIFT)",
        "",
        *ablation_table(dev_subset, methods),
        "",
        "## Recall change from calibration to held-out test",
        "",
        "| Method | Calibration recall | Held-out recall | Change (percentage points) |",
        "|---|---:|---:|---:|",
    ]
    dropped = []
    for method in methods:
        before = calibration.totals[method][0] / calibration.forged
        after = test.totals[method][0] / test.forged
        diff = 100 * (after - before)
        if diff < 0:
            dropped.append((diff, method, before, after))
        lines.append(f"| {method} | {before:.2%} | {after:.2%} | {diff:+.2f} |")
    if dropped:
        delta, method, before, after = min(dropped)
        lines += ["", f"Held-out recall dropped for {len(dropped)} of {len(methods)} methods. "
                  f"The largest drop was {method}, from {before:.2%} to {after:.2%} ({delta:+.2f} points)."]
    else:
        lines += ["", "Held-out recall was at least as high as calibration recall for every reported method."]

    lines += ["", "## Failure analysis by shorter image side", ""]
    names = {
        "test": "CASIA2 held-out test",
        "calibration": "CASIA2 calibration (in-sample)",
        "dev": "CoMoFoD dev, full 6,400 images (optimistic for DCT and SIFT)",
        "dev_subset": "CoMoFoD dev, matched 800 images (optimistic for DCT and SIFT)",
    }
    for split, split_name in names.items():
        applicable = classical if split == "dev" else methods
        for label, summary in zip(SIZE_BINS, results[split][1], strict=True):
            lines += [f"### {split_name} · {label}", ""]
            if not summary.totals:
                lines += ["No images in this bin.", ""]
                continue
            lines += ["| Method | Forged | Authentic | Recall (95% CI) | FAR (95% CI) |",
                      "|---|---:|---:|---|---|"]
            for method in applicable:
                tp, fp = summary.totals[method][:2]
                lines.append(f"| {method} | {summary.forged} | {summary.authentic} | "
                             f"{percent_ci(tp, summary.forged)} | {percent_ci(fp, summary.authentic)} |")
            if summary.forged:
                weakest = min(applicable, key=lambda method: summary.totals[method][0])
                lines += ["", f"Lowest recall in this bin was {weakest} at "
                          f"{summary.totals[weakest][0] / summary.forged:.1%}."]
            lines.append("")

    lines += ["## CASIA2 held-out failure analysis by manipulation code", "",
              "Recall is shown for filename codes with at least 20 forged images. Authentic images have no manipulation code.", "",
              "| Code | Forged | " + " | ".join(methods) + " |",
              "|---|---:|" + "---|" * len(methods)]
    eligible = [(code, group) for code, group in sorted(results["test"][2].items()) if group.forged >= 20]
    for code, group in eligible:
        rates = [percent_ci(group.totals[method][0], group.forged) for method in methods]
        lines.append(f"| {code} | {group.forged} | " + " | ".join(rates) + " |")
    if eligible:
        lines.extend(["", "Lowest recall by code for selected methods:"])
        for method in ("DCT (frozen)", "SIFT (frozen)", "DOA-GAN (2%)", "F4 (calibrated)"):
            code, group = min(eligible, key=lambda item: item[1].totals[method][0] / item[1].forged)
            lines.append(f"- {method}: {code}, {group.totals[method][0] / group.forged:.1%} recall (n={group.forged}).")

    for split, applicable, count in (("dev", classical, "full 6,400 images"),
                                     ("dev_subset", methods, "matched 800 images")):
        lines += ["", f"## CoMoFoD dev by attack category, {count} (optimistic for DCT and SIFT)", ""]
        for attack, group in sorted(results[split][2].items()):
            lines += [f"### {attack.title()} · {count} · optimistic for DCT and SIFT",
                      "", *ablation_table(group, applicable), ""]
    lines += ["## Held-out montage artifacts", "",
              "Seed 0 selects eight forged cases per detector and outcome where available. "
              "Green shows predicted regions on successes. Red outlines the ground-truth boundary.", "",
              "| Detector | Outcome | Images | Local artifact |", "|---|---|---:|---|"]
    for detector, outcome, count, path in montage_paths:
        relpath = path.relative_to(ROOT)
        lines.append(f"| {MONTAGE_DETECTORS[detector]} | {outcome} | {count} | [{path.name}](../{relpath}) |")
    lines += ["", "## Fusion rules", ""]
    lines += [f"- **{rule}** {label}" for rule, label in FUSION_RULES.items()]
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(ablation_table(test, methods)))
    print(f"[final] Wrote {REPORT}")


def run_final():
    if REPORT.exists():
        raise RuntimeError(f"Final report already exists at {REPORT}")
    thresholds = verify_settings()
    calibrated = classical_parameters(CLASSICAL_CONFIG)
    cv2.setNumThreads(1)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    splits = {
        "calibration": {"forged": 964, "authentic": 2247},
        "dev": {"forged": 3200, "authentic": 3200},
        "test": {"forged": 2320, "authentic": 5244},
    }
    rows = {split: manifest_rows(split) for split in splits}
    for split, expected in splits.items():
        if Counter(row["label"] for row in rows[split]) != expected:
            raise ValueError(f"Unexpected label counts for {split}")
    subset = load_comofod_subset().to_dict("records")
    dev_lookup = {row["image_path"]: (row["label"], row["attack_type"]) for row in rows["dev"]}
    subset_lookup = {row["image_path"]: (row["label"], row["attack_type"]) for row in subset}
    groups = Counter(subset_lookup.values())
    if (len(subset_lookup) != 800 or len(groups) != 10 or set(groups.values()) != {80}
            or any(dev_lookup.get(path) != identity for path, identity in subset_lookup.items())):
        raise ValueError("The locked subset does not match the CoMoFoD dev manifest and strata")
    print(f"[final] git={revision} manifests={ROOT / 'data' / 'natural' / 'manifests'}")
    print("[final] Skipping COVERAGE. No dataset images or manifest records.")
    infer_doagan("test")
    infer_doagan("dev")
    results = {}
    for split in ("test", "calibration", "dev"):
        selection = set(subset_lookup) if split == "dev" else None
        results[split], matched = analyze_split(split, rows[split], thresholds, calibrated, selection)
        if split == "dev":
            results["dev_subset"] = matched
    paths = write_montages(results["test"][3], thresholds)
    build_report(results, paths, revision)
