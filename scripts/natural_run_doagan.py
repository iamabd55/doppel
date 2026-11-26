"""Cache single-channel DOA-GAN area scores and native masks on calibration or dev."""

import argparse
import hashlib
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector.doagan import load_model, localize, preprocess, read_rgb, summarize
from natural_run_classical import cache_paths

MANIFESTS = ROOT / "data" / "natural" / "manifests"
CACHE_ROOT = ROOT / "results" / "natural" / "cache"
CACHE_VERSION = "doagan_single_area_v1"
SPLITS = {"calibration": "casia2", "dev": "comofod", "test": "casia2"}
COMOFOD_SUBSET = ROOT / "configs" / "natural_comofod_doagan_subset.csv"
COMOFOD_SUBSET_SHA256 = "8ea0e168262a1920d3a6e49f51ee680c1313e0746703b71ea51ed4d544aeabad"


def load_comofod_subset():
    if hashlib.sha256(COMOFOD_SUBSET.read_bytes()).hexdigest() != COMOFOD_SUBSET_SHA256:
        raise ValueError("The locked CoMoFoD DOA-GAN subset changed")
    return pd.read_csv(COMOFOD_SUBSET)


def result_path(dataset, split, image_path):
    _, path = cache_paths(CACHE_ROOT, CACHE_VERSION, dataset, split, image_path)
    return path


def write_result(path, score, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            np.savez_compressed(temporary, score=score, mask=mask)
            temporary.flush()
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, default="calibration")
    parser.add_argument("--final", action="store_true", help="Unlock the reserved test pass")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch size and thread count must be positive")
    if args.split == "test" and not args.final:
        parser.error("The test split requires --final")

    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    dataset = SPLITS[args.split]
    manifest = load_comofod_subset() if args.split == "dev" else pd.read_csv(MANIFESTS / f"{dataset}.csv")
    rows = list(manifest[manifest["split"].eq(args.split)].itertuples(index=False))
    if not rows:
        raise ValueError(f"No {dataset}/{args.split} images")
    todo = [(row, result_path(row.dataset, row.split, row.image_path)) for row in rows]
    todo = [(row, path) for row, path in todo if not path.is_file()]
    print(f"[DOA-GAN] {dataset}/{args.split}: {len(rows)} images, {len(todo)} pending")
    if not todo:
        return

    model = load_model(device=args.device)
    for start in range(0, len(todo), args.batch_size):
        batch = todo[start:start + args.batch_size]
        images = [read_rgb(ROOT / row.image_path) for row, _ in batch]
        heatmaps = localize(model, torch.stack([preprocess(image) for image in images]))
        for (_, path), image, heatmap in zip(batch, images, heatmaps, strict=True):
            score, mask = summarize(heatmap, image.shape)
            write_result(path, score, mask)
        done = start + len(batch)
        if start == 0 or done % 50 < len(batch) or done == len(todo):
            print(f"[DOA-GAN] {done}/{len(todo)} pending images processed")


if __name__ == "__main__":
    main()
