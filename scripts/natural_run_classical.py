"""Run frozen DCT or frozen SIFT on every image in the natural manifests.

What this script does, step by step:
1. Read the three manifests (casia2.csv, comofod.csv, coverage.csv).
2. For each row (one image), build a cache path:
       results/natural/cache/<detector>/<dataset>/<split>/<image_stem>.json
       results/natural/cache/<detector>/<dataset>/<split>/<image_stem>_mask.npz
3. If both files exist, skip the image (resumable).
4. Otherwise: load the image as grayscale, call the detector, save the JSON
   (decision, score, runtime_sec) and the compressed mask.
5. Print a summary line for each processed image.

Flags:
  --detector dct|sift     which detector to run (required)
  --smoke                 run only 10 forged + 10 authentic from CASIA calibration
  --workers N             number of parallel worker processes (default 1)
  --manifest-dir PATH     override default data/natural/manifests/

The frozen configs are hard-coded here from the protocol. No YAML is read at
runtime so there is no risk of a stale config file changing the results.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detector.natural import FROZEN_DCT, FROZEN_SIFT


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def cache_paths(cache_root: Path, detector: str, dataset: str,
                split: str, image_path: str):
    """Return (json_path, mask_path) for one image."""
    # Use a hash of the full image path as the filename to avoid collisions
    # from images with the same stem in different directories.
    stem = Path(image_path).stem
    # Replace spaces (CASIA2 paths have "CASIA 2 Groundtruth") with underscores.
    safe = stem.replace(" ", "_")
    # Also include last two parts of the path to disambiguate within a dataset.
    parts = Path(image_path).parts
    prefix = "_".join(parts[-2:]).replace(" ", "_").replace("/", "_")
    key = prefix + "__" + safe
    # Truncate to 180 chars to stay safe on all filesystems.
    key = key[:180]
    base = cache_root / detector / dataset / split / key
    return base.with_suffix(".json"), base.with_name(key + "_mask.npz")


def is_cached(json_path: Path, mask_path: Path) -> bool:
    return json_path.exists() and mask_path.exists()


def save_cache(json_path: Path, mask_path: Path,
               decision: bool, score: float, runtime_sec: float,
               mask: np.ndarray, image_path: str):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump({"image_path": image_path,
                   "decision": bool(decision), "score": float(score),
                   "runtime_sec": float(runtime_sec)}, f)
    np.savez_compressed(str(mask_path), mask=mask)


# ---------------------------------------------------------------------------
# Worker function (runs in a subprocess when workers > 1)
# ---------------------------------------------------------------------------

def process_one(args_tuple):
    """Process one image. Returns a status dict."""
    (detector, image_path, label, dataset, split,
     json_path_str, mask_path_str) = args_tuple

    json_path = Path(json_path_str)
    mask_path = Path(mask_path_str)

    if is_cached(json_path, mask_path):
        return {"status": "skip", "image": image_path, "runtime_sec": 0.0}

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return {"status": "error", "image": image_path,
                "reason": "cv2.imread returned None"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t0 = time.perf_counter()

    if detector == "dct":
        # Import here so the worker subprocess can find the module.
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from detector.dct import detect
        mask = detect(gray, **FROZEN_DCT)
        # DCT has no continuous score; use the fraction of foreground pixels
        # as a proxy score (0 = no forgery signal, 1 = fully flagged).
        total_px = mask.size
        score = float((mask > 0).sum()) / total_px if total_px > 0 else 0.0
        decision = bool((mask > 0).any())

    elif detector == "sift":
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from detector.sift.localization import detect_sift
        mask, pts1, pts2, _ = detect_sift(gray, **FROZEN_SIFT)
        # SIFT score = inlier count (protocol Section 2.3)
        score = float(len(pts1))
        decision = bool((mask > 0).any())

    else:
        return {"status": "error", "image": image_path,
                "reason": f"unknown detector {detector!r}"}

    runtime_sec = time.perf_counter() - t0
    save_cache(json_path, mask_path, decision, score, runtime_sec, mask, image_path)

    return {
        "status": "done",
        "image": image_path,
        "label": label,
        "split": split,
        "decision": decision,
        "score": score,
        "runtime_sec": runtime_sec,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_manifests(manifest_dir: Path):
    frames = []
    for name in ("casia2.csv", "comofod.csv", "coverage.csv"):
        p = manifest_dir / name
        if p.exists():
            df = pd.read_csv(p)
            if len(df) > 0:
                frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No non-empty manifests found in {manifest_dir}")
    return pd.concat(frames, ignore_index=True)


def select_smoke(df: pd.DataFrame) -> pd.DataFrame:
    """Return 10 forged + 10 authentic from CASIA calibration."""
    cal = df[(df["dataset"] == "casia2") & (df["split"] == "calibration")]
    forged = cal[cal["label"] == "forged"].head(10)
    authentic = cal[cal["label"] == "authentic"].head(10)
    return pd.concat([forged, authentic], ignore_index=True)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    default_manifest_dir = repo_root / "data" / "natural" / "manifests"
    default_cache_root = repo_root / "results" / "natural" / "cache"

    parser = argparse.ArgumentParser(description="Run frozen classical detector on natural manifests.")
    parser.add_argument("--detector", required=True, choices=["dct", "sift"],
                        help="Which detector to run.")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 10 forged + 10 authentic from CASIA calibration only.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes.")
    parser.add_argument("--manifest-dir", type=Path, default=default_manifest_dir)
    parser.add_argument("--cache-root", type=Path, default=default_cache_root)
    args = parser.parse_args()

    # Print git commit for reproducibility
    try:
        import subprocess
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root, text=True).strip()
    except Exception:
        commit = "unknown"
    print(f"[natural_run_classical] git={commit} detector={args.detector} "
          f"smoke={args.smoke} workers={args.workers}")
    print(f"  manifests: {args.manifest_dir}")
    print(f"  cache:     {args.cache_root}")

    df = load_manifests(args.manifest_dir)
    print(f"  total rows loaded: {len(df)}")

    if args.smoke:
        df = select_smoke(df)
        print(f"  smoke mode: {len(df)} images")

    # Build work list
    work = []
    for _, row in df.iterrows():
        json_path, mask_path = cache_paths(
            args.cache_root, args.detector,
            row["dataset"], row["split"], row["image_path"]
        )
        work.append((
            args.detector,
            row["image_path"],
            row["label"],
            row["dataset"],
            row["split"],
            str(json_path),
            str(mask_path),
        ))

    # Count already cached
    n_cached = sum(
        1 for (_, img, lbl, ds, sp, jp, mp) in work
        if is_cached(Path(jp), Path(mp))
    )
    print(f"  {n_cached}/{len(work)} already cached, {len(work)-n_cached} to run")

    # Run
    done_count = 0
    skip_count = 0
    error_count = 0
    runtimes = []

    if args.workers <= 1:
        for item in work:
            result = process_one(item)
            if result["status"] == "skip":
                skip_count += 1
            elif result["status"] == "done":
                done_count += 1
                runtimes.append(result["runtime_sec"])
                print(f"  [{done_count+skip_count}/{len(work)}] "
                      f"{result['status']:4s} {result['label']:9s} "
                      f"{result['split']:11s} "
                      f"dec={result['decision']} score={result['score']:.3f} "
                      f"t={result['runtime_sec']:.2f}s  "
                      f"{Path(result['image']).name}")
            else:
                error_count += 1
                print(f"  ERROR: {result['reason']}  {result['image']}", file=sys.stderr)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process_one, item): item for item in work}
            processed = 0
            for future in as_completed(futures):
                processed += 1
                result = future.result()
                if result["status"] == "skip":
                    skip_count += 1
                elif result["status"] == "done":
                    done_count += 1
                    runtimes.append(result["runtime_sec"])
                    print(f"  [{processed}/{len(work)}] "
                          f"dec={result['decision']} score={result['score']:.3f} "
                          f"t={result['runtime_sec']:.2f}s  "
                          f"{Path(result['image']).name}")
                else:
                    error_count += 1
                    print(f"  ERROR: {result['reason']}  {result['image']}", file=sys.stderr)

    print(f"\nSummary: done={done_count}, skipped={skip_count}, errors={error_count}")
    if runtimes:
        print(f"  runtime: mean={sum(runtimes)/len(runtimes):.2f}s "
              f"min={min(runtimes):.2f}s max={max(runtimes):.2f}s")


if __name__ == "__main__":
    main()
