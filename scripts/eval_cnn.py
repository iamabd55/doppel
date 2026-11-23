#!/usr/bin/env python3
"""Evaluation and ablation script for Phase 4 CNN copy-move detection.

Evaluates a trained checkpoint on the validation split with fine-grained breakdowns:
  - Class (Blot/Gel, Microscopy)
  - Degradation tier (zero, light, moderate, heavy)
  - Transform type (translation, 90/180 rotation, flip, arbitrary, combination)
  - Multi-copy status (single-copy vs multi-copy)
  - Patch-area bucket (<500, 500-1500, >1500)

Supports --ablate-corr to measure performance when correlation channels are zeroed out.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from detector.cnn import ResNetCorrNet, ResNetUNet, load_checkpoint
from detector.colab_dataset import ColabDoppelDataset


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute Wilson score 95% confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.95996
    p = k / n
    denominator = 1.0 + (z**2) / n
    centre = p + (z**2) / (2 * n)
    spread = z * np.sqrt((p * (1.0 - p) + (z**2) / (4 * n)) / n)
    lower = max(0.0, (centre - spread) / denominator)
    upper = min(1.0, (centre + spread) / denominator)
    return (float(lower), float(upper))


def filter_min_area(binary_mask: np.ndarray, min_area: int = 50) -> np.ndarray:
    """Remove connected components smaller than min_area."""
    if np.sum(binary_mask) == 0:
        return binary_mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    filtered = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 1
    return filtered


def categorize_transform(row: Dict[str, Any]) -> str:
    """Map manifest transform properties to fine-grained transform category."""
    ttype = str(row.get("transform_type", "translation")).lower()
    flip = str(row.get("flip_type", "none")).lower()
    try:
        rot = float(row.get("rotation_deg", 0.0))
    except (ValueError, TypeError):
        rot = 0.0

    if ttype == "combination":
        return "combination"
    if ttype == "arbitrary":
        return "arbitrary"
    if ttype == "cardinal_flip":
        if flip != "none" and rot == 0.0:
            return "flip"
        elif rot in [90.0, 180.0, 270.0]:
            return "90/180 rotation"
        else:
            return "cardinal/flip"
    if ttype == "translation":
        return "translation"
    return ttype


def get_area_bucket(area: float) -> str:
    """Map patch area to size bucket."""
    if area < 500:
        return "<500"
    elif area <= 1500:
        return "500-1500"
    else:
        return ">1500"


def evaluate_slice(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute performance metrics for a slice of validation samples."""
    forged = [r for r in records if r["is_forged"]]
    clean = [r for r in records if not r["is_forged"]]

    n_forged = len(forged)
    n_clean = len(clean)

    if n_forged > 0:
        n_overlap = sum(1 for r in forged if r["detected_overlap"])
        n_any = sum(1 for r in forged if r["detected_any"])

        rec_overlap = n_overlap / n_forged
        ci_overlap = wilson_ci(n_overlap, n_forged)

        rec_any = n_any / n_forged
        ci_any = wilson_ci(n_any, n_forged)

        tot_tp = sum(r["tp"] for r in forged)
        tot_fp = sum(r["fp"] for r in forged)
        tot_fn = sum(r["fn"] for r in forged)
        tot_tn = sum(r["tn"] for r in forged)

        denom_f1 = 2 * tot_tp + tot_fp + tot_fn
        pixel_f1 = (2 * tot_tp / denom_f1) if denom_f1 > 0 else 0.0

        mcc_num = float(tot_tp * tot_tn - tot_fp * tot_fn)
        mcc_den = np.sqrt(
            float(tot_tp + tot_fp)
            * float(tot_tp + tot_fn)
            * float(tot_tn + tot_fp)
            * float(tot_tn + tot_fn)
        )
        pixel_mcc = (mcc_num / mcc_den) if mcc_den > 0 else 0.0
    else:
        n_overlap = 0
        n_any = 0
        rec_overlap = 0.0
        ci_overlap = (0.0, 0.0)
        rec_any = 0.0
        ci_any = (0.0, 0.0)
        pixel_f1 = 0.0
        pixel_mcc = 0.0

    if n_clean > 0:
        clean_flags = sum(1 for r in clean if r["detected_any"])
        clean_far = clean_flags / n_clean
        ci_clean = wilson_ci(clean_flags, n_clean)
    else:
        clean_flags = 0
        clean_far = 0.0
        ci_clean = (0.0, 0.0)

    return {
        "n_forged": n_forged,
        "n_overlap": n_overlap,
        "recall_overlap": rec_overlap,
        "ci_overlap": ci_overlap,
        "n_any": n_any,
        "recall_any": rec_any,
        "ci_any": ci_any,
        "pixel_f1": pixel_f1,
        "pixel_mcc": pixel_mcc,
        "n_clean": n_clean,
        "clean_flags": clean_flags,
        "clean_far": clean_far,
        "ci_clean": ci_clean,
    }


def parse_args(args: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Evaluate CNN copy-move detector across validation slices.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--model", type=str, choices=["unet", "corr"], default=None, help="Model type (auto-detected if omitted).")
    parser.add_argument("--val-manifest", type=str, default="data/synthetic/val_manifest.csv", help="Path to val_manifest.csv.")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory.")
    parser.add_argument("--pristine-zip", type=str, default="data/pristine_heldout.zip", help="Path to pristine held-out zip.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Pixel detection threshold.")
    parser.add_argument("--min-area", type=int, default=50, help="Minimum connected component area.")
    parser.add_argument("--image-head", action="store_true", help="Force image head for CorrNet.")
    parser.add_argument("--ablate-corr", action="store_true", help="Zero out correlation channels at inference.")
    parser.add_argument("--out-csv", type=str, default=None, help="Optional output CSV path for slice results.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device.")
    return parser.parse_args(args)


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = load_checkpoint(ckpt_path, map_location=device)
    config = ckpt.get("config", {})

    # Detect model type if not explicitly supplied
    model_type = args.model or config.get("model")
    if model_type is None:
        state_keys = ckpt["model_state_dict"].keys()
        if any("corr" in k for k in state_keys) or any("dec8.conv1.0.weight" in k for k in state_keys):
            model_type = "corr"
        else:
            model_type = "unet"

    has_image_head = args.image_head or config.get("image_head", False) or any(
        "image_head_fc" in k for k in ckpt.get("model_state_dict", {}).keys()
    )

    print(f"[INFO] Evaluating model: {model_type} from {ckpt_path.name}")
    if args.ablate_corr and model_type == "corr":
        print("[INFO] Ablation active: correlation channels are zeroed at inference.")

    if model_type == "unet":
        model = ResNetUNet(pretrained=False)
    else:
        model = ResNetCorrNet(pretrained=False, image_head=has_image_head)
        if args.ablate_corr:
            model.ablate_corr = True

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    val_manifest = Path(args.val_manifest)
    if not val_manifest.exists():
        # Fallback check
        alt_manifest = Path(args.data_dir) / "synthetic" / "val_manifest.csv"
        if alt_manifest.exists():
            val_manifest = alt_manifest
        else:
            raise FileNotFoundError(f"Missing val manifest: {val_manifest}")

    val_dataset = ColabDoppelDataset(
        manifest_path=str(val_manifest),
        images_dir=args.data_dir,
        split="val",
        pad_multiple=32,
        augment=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print(f"[INFO] Loaded {len(val_dataset)} validation samples from {val_manifest}")

    # Process all validation samples
    records: List[Dict[str, Any]] = []

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            img = batch["image"].to(device)
            mask_gt = batch["mask"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            is_forged = bool(batch["label"].item() > 0.5)
            meta = {k: v[0] if isinstance(v, list) else v for k, v in batch["meta"].items()}

            if model_type == "corr":
                out = model(img, valid_mask=valid_mask, ablate_corr=args.ablate_corr)
            else:
                out = model(img, valid_mask=valid_mask)

            if isinstance(out, tuple):
                logits, img_logit = out
                img_prob = float(torch.sigmoid(img_logit)[0].item())
            else:
                logits = out
                img_prob = None

            probs = torch.sigmoid(logits)

            vm_bool = (valid_mask > 0.5)[0, 0].cpu().numpy()
            probs_np = probs[0, 0].cpu().numpy()
            mask_gt_np = (mask_gt[0, 0].cpu().numpy() > 0.5).astype(np.uint8)

            # Strict crop back to unpadded original resolution
            ys, xs = np.where(vm_bool)
            if len(ys) > 0 and len(xs) > 0:
                top_y, bot_y = int(ys.min()), int(ys.max() + 1)
                left_x, right_x = int(xs.min()), int(xs.max() + 1)
            else:
                top_y, bot_y = 0, probs_np.shape[0]
                left_x, right_x = 0, probs_np.shape[1]

            probs_cropped = probs_np[top_y:bot_y, left_x:right_x]
            mask_gt_cropped = mask_gt_np[top_y:bot_y, left_x:right_x]

            pred_bin = (probs_cropped >= args.threshold).astype(np.uint8)
            pred_filt = filter_min_area(pred_bin, min_area=args.min_area)

            detected_any = bool(np.sum(pred_filt) > 0)
            detected_overlap = (
                bool(np.sum((pred_filt > 0) & (mask_gt_cropped > 0)) > 0)
                if is_forged
                else False
            )

            tp = int(np.sum((pred_filt == 1) & (mask_gt_cropped == 1)))
            fp = int(np.sum((pred_filt == 1) & (mask_gt_cropped == 0)))
            fn = int(np.sum((pred_filt == 0) & (mask_gt_cropped == 1)))
            tn = int(np.sum((pred_filt == 0) & (mask_gt_cropped == 0)))

            try:
                patch_area = float(meta.get("patch_area", 0.0))
            except (ValueError, TypeError):
                patch_area = 0.0

            try:
                num_targets = int(meta.get("num_targets", 1))
            except (ValueError, TypeError):
                num_targets = 1

            records.append({
                "idx": idx,
                "is_forged": is_forged,
                "class": str(meta.get("class", "Unknown")),
                "tier": str(meta.get("degradation_level", "unknown")),
                "transform": categorize_transform(meta),
                "is_multi_copy": (num_targets > 1) or bool(meta.get("is_multi_copy", False)),
                "patch_area": patch_area,
                "area_bucket": get_area_bucket(patch_area),
                "detected_any": detected_any,
                "detected_overlap": detected_overlap,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            })

    # Group into evaluation slices
    slices: List[Tuple[str, str, List[Dict[str, Any]]]] = []

    # 1. Overall
    slices.append(("Overall", "All Val", records))

    # 2. Class
    for cls_name in ["Blot/Gel", "Microscopy"]:
        slices.append(("Class", cls_name, [r for r in records if r["class"] == cls_name]))

    # 3. Degradation Tier
    for tier in ["zero", "light", "moderate", "heavy"]:
        slices.append(("Degradation Tier", tier, [r for r in records if r["tier"] == tier]))

    # 4. Transform Type
    for tform in ["translation", "90/180 rotation", "flip", "arbitrary", "combination"]:
        slices.append(("Transform Type", tform, [r for r in records if r["transform"] == tform]))

    # 5. Multi-copy
    slices.append(("Multi-Copy", "Single-Copy", [r for r in records if not r["is_multi_copy"]]))
    slices.append(("Multi-Copy", "Multi-Copy", [r for r in records if r["is_multi_copy"]]))

    # 6. Patch Area Bucket
    for bucket in ["<500", "500-1500", ">1500"]:
        slices.append(("Patch Area", bucket, [r for r in records if r["area_bucket"] == bucket]))

    # Compute slice metrics and print table
    output_rows = []
    print("\n" + "=" * 105)
    header = (
        f"{'Category':<18} | {'Slice':<16} | {'N (Forged)':<10} | {'Recall (Overlap)':<22} | "
        f"{'Recall (Any)':<16} | {'Pixel F1':<8} | {'Pixel MCC':<9} | {'Clean FAR':<10}"
    )
    print(header)
    print("-" * 105)

    has_low_n = False

    for category, slice_name, slice_records in slices:
        stats = evaluate_slice(slice_records)
        n = stats["n_forged"]
        low_n_flag = "*" if n < 30 else ""
        if n < 30:
            has_low_n = True

        n_str = f"{n}{low_n_flag}"
        rec_ov_str = f"{stats['recall_overlap']*100:.1f}% [{stats['ci_overlap'][0]*100:.1f}, {stats['ci_overlap'][1]*100:.1f}]"
        rec_any_str = f"{stats['recall_any']*100:.1f}%"
        f1_str = f"{stats['pixel_f1']:.4f}"
        mcc_str = f"{stats['pixel_mcc']:.4f}"
        far_str = f"{stats['clean_far']*100:.2f}%"

        print(
            f"{category:<18} | {slice_name:<16} | {n_str:<10} | {rec_ov_str:<22} | "
            f"{rec_any_str:<16} | {f1_str:<8} | {mcc_str:<9} | {far_str:<10}"
        )

        output_rows.append({
            "category": category,
            "slice": slice_name,
            "n_forged": n,
            "low_power_flag": bool(n < 30),
            "recall_overlap": stats["recall_overlap"],
            "ci_overlap_low": stats["ci_overlap"][0],
            "ci_overlap_high": stats["ci_overlap"][1],
            "recall_any": stats["recall_any"],
            "ci_any_low": stats["ci_any"][0],
            "ci_any_high": stats["ci_any"][1],
            "pixel_f1": stats["pixel_f1"],
            "pixel_mcc": stats["pixel_mcc"],
            "clean_far": stats["clean_far"],
        })

    print("-" * 105)
    if has_low_n:
        print("* Note: N < 30 indicates reduced statistical power (interpret with caution).")
    print("=" * 105 + "\n")

    if args.out_csv:
        out_csv_path = Path(args.out_csv)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
            writer.writeheader()
            writer.writerows(output_rows)
        print(f"[INFO] Wrote slice evaluation CSV to {out_csv_path}")


if __name__ == "__main__":
    main()
