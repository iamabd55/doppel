#!/usr/bin/env python3
"""Diagnostic script analyzing CNN false alarms, probability distributions, and edge artifacts.

Evaluates a given checkpoint on:
  1. Synthetic validation clean twins (500 images)
  2. Fixed seeded subset of real pristine held-out panels (1,000 panels)
  3. Synthetic validation forged images (500 images)

Outputs:
  - Checkpoint training configuration and loss settings (pos_weight, etc.) from config.json
  - Max probability, predicted area at tau in {0.5, 0.7, 0.9}, and distance to image edge
  - False alarm rate grid over tau {0.3, 0.5, 0.7, 0.9, 0.95, 0.98} x min_area {0, 50, 100, 200, 400, 800, 1600} with Wilson 95% CIs
  - Recall grid on forged images over the same tau x min_area grid
  - False alarm breakdowns: synthetic clean by class and degradation tier, pristine by class
  - Recall breakdowns: synthetic forged by class, degradation tier, transform type, and multi-copy at tau 0.5 and 0.9
  - False alarm reduction when ignoring pixels within 8 px of the image edge
  - Overlay figure of the 12 worst clean panels (image + probability map) saved to disk
"""
import argparse
import csv
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from detector.cnn import ResNetCorrNet, ResNetUNet, load_checkpoint
from detector.cnn.infer import compute_top_k_mean, crop_to_original, filter_min_area, pad_to_multiple
from detector.colab_dataset import ColabDoppelDataset


def wilson_ci(k: int, n: int) -> Tuple[float, float]:
    """Compute 95% Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.95996
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    margin = (z * np.sqrt((p * (1 - p) + (z**2) / (4 * n)) / n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def load_model(checkpoint_path: str, model_type: Optional[str] = None, force_image_head: bool = False, device: str = "cpu") -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    """Load model and full config dictionary from checkpoint and config.json."""
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    cfg = ckpt.get("config", {})

    # Check for accompanying config.json in checkpoint directory
    cfg_file = Path(checkpoint_path).parent / "config.json"
    if cfg_file.exists():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                disk_cfg = json.load(f)
                disk_cfg.update(cfg)
                cfg = disk_cfg
        except Exception as e:
            print(f"[WARNING] Could not read {cfg_file}: {e}")

    if model_type is None:
        model_type = cfg.get("model")
        if model_type is None:
            if any("corr_proj" in k for k in state_dict.keys()) or any("proj.weight" in k for k in state_dict.keys()):
                model_type = "corr"
            else:
                model_type = "unet"

    has_image_head = bool(force_image_head or cfg.get("image_head", False) or any("image_head_fc" in k for k in state_dict.keys()))

    if model_type == "corr":
        model = ResNetCorrNet(pretrained=False, image_head=has_image_head)
    else:
        model = ResNetUNet(pretrained=False)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[INFO] Loaded {model_type.upper()} model (image_head={has_image_head}) from {checkpoint_path}")
    return model, cfg, ckpt


def print_training_config(cfg: Dict[str, Any], ckpt: Dict[str, Any], checkpoint_path: str):
    """Print loss settings and training configuration."""
    print("\n" + "=" * 90)
    print("CHECKPOINT TRAINING CONFIGURATION & LOSS SETTINGS")
    print("=" * 90)
    print(f"Checkpoint File:       {Path(checkpoint_path).name}")
    print(f"Git Commit Hash:       {cfg.get('git_commit', 'unknown')}")
    print(f"Model Architecture:    {cfg.get('model', 'unknown')}")
    print(f"Image Head Enabled:    {cfg.get('image_head', False)}")
    print(f"Loss pos_weight:       {cfg.get('pos_weight', 1.0)}")
    print(f"BCE Weight:            {cfg.get('bce_weight', 1.0)}")
    print(f"Dice Weight:           {cfg.get('dice_weight', 1.0)}")
    print(f"Peak Learning Rate:    {cfg.get('lr', 'not specified')}")
    print(f"Weight Decay:          {cfg.get('weight_decay', 'not specified')}")
    print(f"Epochs Configured:     {cfg.get('epochs', 'not specified')}")
    print(f"Checkpoint Epoch:      {ckpt.get('epoch', 'unknown')}")
    if "best_checkpoint_eval" in ckpt and ckpt["best_checkpoint_eval"] is not None:
        bce_info = ckpt["best_checkpoint_eval"]
        print(f"Best Eval at Save:     recall={bce_info.get('recall', 0.0):.4f}, meets_budget={bce_info.get('meets_budget', False)}")
    print("=" * 90 + "\n")


def load_pristine_panels(pristine_zip_path: str, max_count: int = 1000, seed: int = 42) -> List[Dict[str, Any]]:
    """Load seeded pristine panels from pristine_heldout.zip."""
    if not os.path.exists(pristine_zip_path):
        print(f"[WARNING] Pristine zip not found: {pristine_zip_path}")
        return []

    panels = []
    with zipfile.ZipFile(pristine_zip_path, "r") as zf:
        manifest_names = [n for n in zf.namelist() if n.endswith("pristine_manifest.csv")]
        if not manifest_names:
            return []
        manifest_bytes = zf.read(manifest_names[0]).decode("utf-8")
        reader = csv.DictReader(manifest_bytes.splitlines())
        rows = list(reader)

        if max_count > 0 and len(rows) > max_count:
            rng = np.random.default_rng(seed)
            sampled_indices = rng.choice(len(rows), size=max_count, replace=False)
            sampled_indices.sort()
            rows = [rows[i] for i in sampled_indices]

        for r in rows:
            arc_name = f"pristine_heldout/{r['path']}"
            if arc_name in zf.namelist():
                img_bytes = zf.read(arc_name)
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    panels.append({
                        "image": img,
                        "class": r["class"],
                        "doc_id": r.get("doc_id", "unknown"),
                        "path": r["path"],
                    })
    print(f"[INFO] Loaded {len(panels)} pristine panels from {pristine_zip_path}")
    return panels


def parse_args(args: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Diagnose CNN false alarms and boundary effects.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt).")
    parser.add_argument("--model", type=str, choices=["unet", "corr"], default=None, help="Model type.")
    parser.add_argument("--image-head", action="store_true", help="Force image head on for CorrNet.")
    parser.add_argument("--val-manifest", type=str, default="data/synthetic/val_manifest.csv", help="Path to val manifest.")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory.")
    parser.add_argument("--val-subset", type=int, default=None, help="Optional subset of validation samples to evaluate.")
    parser.add_argument("--pristine-zip", type=str, default="data/pristine_heldout.zip", help="Path to pristine zip.")
    parser.add_argument("--pristine-subset", type=int, default=1000, help="Number of seeded pristine panels (default 1000).")
    parser.add_argument("--out-dir", type=str, default="results/phase4_cnn/diagnostics", help="Output directory.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device.")
    return parser.parse_args(args)


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, cfg, ckpt = load_model(args.checkpoint, model_type=args.model, force_image_head=args.image_head, device=args.device)
    print_training_config(cfg, ckpt, args.checkpoint)

    # 1. Load Synthetic Validation dataset
    val_dataset = ColabDoppelDataset(
        manifest_path=args.val_manifest,
        images_dir=args.data_dir,
        split="val",
        pad_multiple=32,
        augment=False,
    )
    if args.val_subset is not None and args.val_subset > 0:
        from torch.utils.data import Subset
        val_dataset = Subset(val_dataset, list(range(min(args.val_subset, len(val_dataset)))))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print(f"[INFO] Loaded {len(val_dataset)} synthetic validation images.")

    # 2. Load Pristine Panels
    pristine_panels = load_pristine_panels(args.pristine_zip, max_count=args.pristine_subset, seed=42)

    taus = [0.3, 0.5, 0.7, 0.9, 0.95, 0.98]
    min_areas = [0, 50, 100, 200, 400, 800, 1600]

    synth_clean_records: List[Dict[str, Any]] = []
    synth_forged_records: List[Dict[str, Any]] = []

    print("[INFO] Evaluating synthetic validation split...")
    with torch.no_grad():
        for batch in val_loader:
            image = batch["image"].to(device)
            mask_gt = batch["mask"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            is_forged = bool(batch["label"].item() > 0.5)
            meta = {k: v[0] if isinstance(v, list) else v for k, v in batch["meta"].items()}

            out = model(image, valid_mask=valid_mask)
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

            # Strict crop back to unpadded dimensions
            ys, xs = np.where(vm_bool)
            if len(ys) > 0 and len(xs) > 0:
                top_y, bot_y = int(ys.min()), int(ys.max() + 1)
                left_x, right_x = int(xs.min()), int(xs.max() + 1)
            else:
                top_y, bot_y = 0, probs_np.shape[0]
                left_x, right_x = 0, probs_np.shape[1]

            probs_cropped = probs_np[top_y:bot_y, left_x:right_x]
            mask_gt_cropped = mask_gt_np[top_y:bot_y, left_x:right_x]
            h, w = probs_cropped.shape[:2]

            max_prob = float(probs_cropped.max()) if probs_cropped.size > 0 else 0.0
            if probs_cropped.size > 0:
                y_max, x_max = np.unravel_index(np.argmax(probs_cropped), probs_cropped.shape)
                dist_edge = int(min(y_max, h - 1 - y_max, x_max, w - 1 - x_max))
            else:
                dist_edge = 0

            area_05 = int(np.sum(probs_cropped >= 0.5))
            area_07 = int(np.sum(probs_cropped >= 0.7))
            area_09 = int(np.sum(probs_cropped >= 0.9))

            # Inner probability map (>8 px from edge)
            inner_probs = probs_cropped.copy()
            if h > 16 and w > 16:
                inner_probs[:8, :] = 0.0
                inner_probs[-8:, :] = 0.0
                inner_probs[:, :8] = 0.0
                inner_probs[:, -8:] = 0.0
            else:
                inner_probs[:] = 0.0

            rec: Dict[str, Any] = {
                "doc_id": str(meta.get("doc_id", "unknown")),
                "class": str(meta.get("class", "unknown")),
                "tier": str(meta.get("degradation_level", "zero")),
                "transform_type": str(meta.get("transform_type", "none")),
                "is_multi_copy": bool(meta.get("is_multi_copy", False) or int(meta.get("num_targets", 1)) > 1),
                "is_forged": is_forged,
                "image_prob": img_prob,
                "max_prob": max_prob,
                "dist_edge": dist_edge,
                "area_05": area_05,
                "area_07": area_07,
                "area_09": area_09,
                "probs": probs_cropped,
                "image_rgb": (
                    (image[0].permute(1, 2, 0).cpu().numpy()[top_y:bot_y, left_x:right_x] * 255.0)
                    .clip(0, 255)
                    .astype(np.uint8)
                ),
            }

            # Evaluate grid over tau and min_area
            for tau in taus:
                bin_mask = (probs_cropped >= tau).astype(np.uint8)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
                areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([], dtype=int)

                # Inner mask (>8 px)
                inner_bin = (inner_probs >= tau).astype(np.uint8)
                i_num_labels, _, i_stats, _ = cv2.connectedComponentsWithStats(inner_bin, connectivity=8)
                i_areas = i_stats[1:, cv2.CC_STAT_AREA] if i_num_labels > 1 else np.array([], dtype=int)

                for area in min_areas:
                    if is_forged:
                        if area == 0:
                            hit = bool(np.sum((bin_mask > 0) & (mask_gt_cropped > 0)) > 0)
                        else:
                            hit = False
                            for lab_idx in range(1, num_labels):
                                if stats[lab_idx, cv2.CC_STAT_AREA] >= area:
                                    if np.sum((labels == lab_idx) & (mask_gt_cropped > 0)) > 0:
                                        hit = True
                                        break
                        rec[f"hit_{tau}_{area}"] = hit
                    else:
                        if area == 0:
                            flag = bool(np.sum(bin_mask) > 0)
                            flag_inner = bool(np.sum(inner_bin) > 0)
                        else:
                            flag = bool(len(areas) > 0 and np.any(areas >= area))
                            flag_inner = bool(len(i_areas) > 0 and np.any(i_areas >= area))
                        rec[f"flag_{tau}_{area}"] = flag
                        rec[f"flag_inner_{tau}_{area}"] = flag_inner

            if is_forged:
                synth_forged_records.append(rec)
            else:
                synth_clean_records.append(rec)

    # 3. Evaluate Pristine Panels
    pristine_records: List[Dict[str, Any]] = []
    print("[INFO] Evaluating pristine held-out panels...")
    with torch.no_grad():
        for p_info in pristine_panels:
            p_img = p_info["image"]
            h, w = p_img.shape[:2]
            padded_img, valid_mask, pad_box = pad_to_multiple(p_img, multiple=32)

            t_img = torch.from_numpy(padded_img.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0).to(device)
            t_vm = torch.from_numpy(valid_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

            out = model(t_img, valid_mask=t_vm)
            if isinstance(out, tuple):
                logits, img_logit = out
                p_img_prob = float(torch.sigmoid(img_logit)[0].item())
            else:
                logits = out
                p_img_prob = None

            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

            probs_cropped = crop_to_original(probs, pad_box=pad_box, orig_shape=(h, w))
            max_prob = float(probs_cropped.max()) if probs_cropped.size > 0 else 0.0

            if probs_cropped.size > 0:
                y_max, x_max = np.unravel_index(np.argmax(probs_cropped), probs_cropped.shape)
                dist_edge = int(min(y_max, h - 1 - y_max, x_max, w - 1 - x_max))
            else:
                dist_edge = 0

            area_05 = int(np.sum(probs_cropped >= 0.5))
            area_07 = int(np.sum(probs_cropped >= 0.7))
            area_09 = int(np.sum(probs_cropped >= 0.9))

            inner_probs = probs_cropped.copy()
            if h > 16 and w > 16:
                inner_probs[:8, :] = 0.0
                inner_probs[-8:, :] = 0.0
                inner_probs[:, :8] = 0.0
                inner_probs[:, -8:] = 0.0
            else:
                inner_probs[:] = 0.0

            p_rec: Dict[str, Any] = {
                "doc_id": p_info.get("doc_id", "unknown"),
                "class": p_info.get("class", "unknown"),
                "is_forged": False,
                "image_prob": p_img_prob,
                "max_prob": max_prob,
                "dist_edge": dist_edge,
                "area_05": area_05,
                "area_07": area_07,
                "area_09": area_09,
                "probs": probs_cropped,
                "image_rgb": p_img,
            }

            for tau in taus:
                bin_mask = (probs_cropped >= tau).astype(np.uint8)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
                areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([], dtype=int)

                inner_bin = (inner_probs >= tau).astype(np.uint8)
                i_num_labels, _, i_stats, _ = cv2.connectedComponentsWithStats(inner_bin, connectivity=8)
                i_areas = i_stats[1:, cv2.CC_STAT_AREA] if i_num_labels > 1 else np.array([], dtype=int)

                for area in min_areas:
                    if area == 0:
                        flag = bool(np.sum(bin_mask) > 0)
                        flag_inner = bool(np.sum(inner_bin) > 0)
                    else:
                        flag = bool(len(areas) > 0 and np.any(areas >= area))
                        flag_inner = bool(len(i_areas) > 0 and np.any(i_areas >= area))
                    p_rec[f"flag_{tau}_{area}"] = flag
                    p_rec[f"flag_inner_{tau}_{area}"] = flag_inner

            pristine_records.append(p_rec)

    # 4. Print Edge Distance Distribution & Summary
    all_clean = synth_clean_records + pristine_records
    n_clean_total = len(all_clean)
    print("\n" + "=" * 90)
    print("MAX-PROBABILITY PIXEL DISTANCE TO IMAGE EDGE (CLEAN SAMPLES)")
    print("=" * 90)
    if n_clean_total > 0:
        edge_dists = [r["dist_edge"] for r in all_clean]
        within_1 = sum(1 for d in edge_dists if d <= 1)
        within_4 = sum(1 for d in edge_dists if d <= 4)
        within_8 = sum(1 for d in edge_dists if d <= 8)
        print(f"Total clean images evaluated: {n_clean_total} ({len(synth_clean_records)} synth clean, {len(pristine_records)} pristine)")
        print(f"Mean distance to edge: {np.mean(edge_dists):.2f} px | Median: {np.median(edge_dists):.1f} px")
        print(f"  Max-prob pixel <= 1 px from edge: {within_1:4d} / {n_clean_total} ({within_1/n_clean_total*100:5.1f}%)")
        print(f"  Max-prob pixel <= 4 px from edge: {within_4:4d} / {n_clean_total} ({within_4/n_clean_total*100:5.1f}%)")
        print(f"  Max-prob pixel <= 8 px from edge: {within_8:4d} / {n_clean_total} ({within_8/n_clean_total*100:5.1f}%)")

    # 5. Print False Alarm Rate Grid (Synth Clean Twins)
    n_synth_clean = len(synth_clean_records)
    print("\n" + "=" * 105)
    print(f"FALSE ALARM RATE GRID: SYNTHETIC CLEAN TWINS (N = {n_synth_clean})")
    print("Format: FAR % [Wilson 95% CI]")
    print("=" * 105)
    header = f"{'tau':<6} | " + " | ".join(f"area>={a:<4}" for a in min_areas)
    print(header)
    print("-" * len(header))
    for tau in taus:
        row_str = f"{tau:<6.2f} | "
        cols = []
        for area in min_areas:
            count = sum(1 for r in synth_clean_records if r[f"flag_{tau}_{area}"])
            rate = count / max(1, n_synth_clean) * 100.0
            lo, hi = wilson_ci(count, n_synth_clean)
            cols.append(f"{rate:4.1f}% [{lo*100:4.1f}-{hi*100:4.1f}]")
        print(row_str + " | ".join(cols))

    # 6. Print False Alarm Rate Grid (Real Pristine Panels)
    n_pristine = len(pristine_records)
    print("\n" + "=" * 105)
    print(f"FALSE ALARM RATE GRID: REAL PRISTINE PANELS (N = {n_pristine})")
    print("Format: FAR % [Wilson 95% CI]")
    print("=" * 105)
    print(header)
    print("-" * len(header))
    for tau in taus:
        row_str = f"{tau:<6.2f} | "
        cols = []
        for area in min_areas:
            count = sum(1 for r in pristine_records if r[f"flag_{tau}_{area}"])
            rate = count / max(1, n_pristine) * 100.0
            lo, hi = wilson_ci(count, n_pristine)
            cols.append(f"{rate:4.1f}% [{lo*100:4.1f}-{hi*100:4.1f}]")
        print(row_str + " | ".join(cols))

    # 7. Print Recall Grid (Synthetic Forged Validation)
    n_forged = len(synth_forged_records)
    print("\n" + "=" * 105)
    print(f"RECALL GRID: SYNTHETIC FORGED IMAGES (N = {n_forged}, OVERLAP RULE)")
    print("Format: Recall % [Wilson 95% CI]")
    print("=" * 105)
    print(header)
    print("-" * len(header))
    for tau in taus:
        row_str = f"{tau:<6.2f} | "
        cols = []
        for area in min_areas:
            count = sum(1 for r in synth_forged_records if r[f"hit_{tau}_{area}"])
            rate = count / max(1, n_forged) * 100.0
            lo, hi = wilson_ci(count, n_forged)
            cols.append(f"{rate:4.1f}% [{lo*100:4.1f}-{hi*100:4.1f}]")
        print(row_str + " | ".join(cols))

    # 8. FALSE ALARM BREAKDOWNS (by class, degradation tier)
    print("\n" + "=" * 105)
    print("FALSE ALARM BREAKDOWNS (SYNTHETIC CLEAN & PRISTINE)")
    print("=" * 105)
    print(f"{'Dataset':<16} | {'Subgroup':<24} | {'N':<6} | {'FAR@0.5 (a=0)':<16} | {'FAR@0.5 (a=50)':<16} | {'FAR@0.9 (a=50)':<16}")
    print("-" * 105)

    # Synthetic clean by class
    for cls in sorted(list(set(r["class"] for r in synth_clean_records))):
        sub = [r for r in synth_clean_records if r["class"] == cls]
        n_sub = len(sub)
        if n_sub > 0:
            c_05_0 = sum(1 for r in sub if r.get("flag_0.5_0", False))
            c_05_50 = sum(1 for r in sub if r.get("flag_0.5_50", False))
            c_09_50 = sum(1 for r in sub if r.get("flag_0.9_50", False))
            print(f"{'Synth Clean':<16} | {('Class: ' + cls):<24} | {n_sub:<6d} | {c_05_0/n_sub*100:>5.1f}% [{wilson_ci(c_05_0, n_sub)[0]*100:.1f}-{wilson_ci(c_05_0, n_sub)[1]*100:.1f}] | {c_05_50/n_sub*100:>5.1f}% [{wilson_ci(c_05_50, n_sub)[0]*100:.1f}-{wilson_ci(c_05_50, n_sub)[1]*100:.1f}] | {c_09_50/n_sub*100:>5.1f}% [{wilson_ci(c_09_50, n_sub)[0]*100:.1f}-{wilson_ci(c_09_50, n_sub)[1]*100:.1f}]")

    # Synthetic clean by degradation tier
    for tier in ["zero", "light", "medium", "heavy"]:
        sub = [r for r in synth_clean_records if r.get("tier") == tier]
        n_sub = len(sub)
        if n_sub > 0:
            c_05_0 = sum(1 for r in sub if r.get("flag_0.5_0", False))
            c_05_50 = sum(1 for r in sub if r.get("flag_0.5_50", False))
            c_09_50 = sum(1 for r in sub if r.get("flag_0.9_50", False))
            print(f"{'Synth Clean':<16} | {('Tier: ' + tier):<24} | {n_sub:<6d} | {c_05_0/n_sub*100:>5.1f}% [{wilson_ci(c_05_0, n_sub)[0]*100:.1f}-{wilson_ci(c_05_0, n_sub)[1]*100:.1f}] | {c_05_50/n_sub*100:>5.1f}% [{wilson_ci(c_05_50, n_sub)[0]*100:.1f}-{wilson_ci(c_05_50, n_sub)[1]*100:.1f}] | {c_09_50/n_sub*100:>5.1f}% [{wilson_ci(c_09_50, n_sub)[0]*100:.1f}-{wilson_ci(c_09_50, n_sub)[1]*100:.1f}]")

    # Pristine by class
    for cls in sorted(list(set(r["class"] for r in pristine_records))):
        sub = [r for r in pristine_records if r["class"] == cls]
        n_sub = len(sub)
        if n_sub > 0:
            c_05_0 = sum(1 for r in sub if r.get("flag_0.5_0", False))
            c_05_50 = sum(1 for r in sub if r.get("flag_0.5_50", False))
            c_09_50 = sum(1 for r in sub if r.get("flag_0.9_50", False))
            print(f"{'Real Pristine':<16} | {('Class: ' + cls):<24} | {n_sub:<6d} | {c_05_0/n_sub*100:>5.1f}% [{wilson_ci(c_05_0, n_sub)[0]*100:.1f}-{wilson_ci(c_05_0, n_sub)[1]*100:.1f}] | {c_05_50/n_sub*100:>5.1f}% [{wilson_ci(c_05_50, n_sub)[0]*100:.1f}-{wilson_ci(c_05_50, n_sub)[1]*100:.1f}] | {c_09_50/n_sub*100:>5.1f}% [{wilson_ci(c_09_50, n_sub)[0]*100:.1f}-{wilson_ci(c_09_50, n_sub)[1]*100:.1f}]")

    # 9. RECALL BREAKDOWNS (by class, tier, transform type, multi-copy at tau 0.5 and 0.9)
    print("\n" + "=" * 105)
    print("RECALL BREAKDOWNS ON SYNTHETIC FORGED IMAGES (OVERLAP RULE, min_area=50)")
    print("=" * 105)
    print(f"{'Dimension':<18} | {'Slice':<22} | {'N':<6} | {'Recall @ tau=0.5':<22} | {'Recall @ tau=0.9':<22}")
    print("-" * 105)

    def print_recall_slice(dim_name: str, slice_name: str, sub_records: List[Dict[str, Any]]):
        n_s = len(sub_records)
        if n_s == 0:
            return
        h_05 = sum(1 for r in sub_records if r.get("hit_0.5_50", False))
        h_09 = sum(1 for r in sub_records if r.get("hit_0.9_50", False))
        lo5, hi5 = wilson_ci(h_05, n_s)
        lo9, hi9 = wilson_ci(h_09, n_s)
        n_str = f"{n_s}{'*' if n_s < 30 else ''}"
        print(f"{dim_name:<18} | {slice_name:<22} | {n_str:<6} | {h_05/n_s*100:>5.1f}% [{lo5*100:4.1f}-{hi5*100:4.1f}]      | {h_09/n_s*100:>5.1f}% [{lo9*100:4.1f}-{hi9*100:4.1f}]")

    # Class
    for cls in sorted(list(set(r["class"] for r in synth_forged_records))):
        print_recall_slice("Class", cls, [r for r in synth_forged_records if r["class"] == cls])

    # Degradation tier
    for tier in ["zero", "light", "medium", "heavy"]:
        print_recall_slice("Tier", tier, [r for r in synth_forged_records if r.get("tier") == tier])

    # Transform type
    for tt in sorted(list(set(r["transform_type"] for r in synth_forged_records))):
        print_recall_slice("Transform", tt, [r for r in synth_forged_records if r.get("transform_type") == tt])

    # Multi-copy
    print_recall_slice("Multi-Copy", "single", [r for r in synth_forged_records if not r.get("is_multi_copy", False)])
    print_recall_slice("Multi-Copy", "multi-copy", [r for r in synth_forged_records if r.get("is_multi_copy", False)])

    print("-" * 105)
    print("* N < 30 indicates small sample size.")

    # 10. Print Edge Margin Analysis: False Alarms Disappearing with >8 px Margin
    print("\n" + "=" * 90)
    print("FALSE ALARMS DISAPPEARING WHEN EXCLUDING PIXELS WITHIN 8 PX OF IMAGE EDGE")
    print("Evaluated on all clean samples (Synth Clean + Pristine, N = %d)" % n_clean_total)
    print("=" * 90)
    margin_header = f"{'tau':<5} | {'min_area':<8} | {'Standard FA':<14} | {'Inner (>8px) FA':<16} | {'Disappeared':<12} | {'Reduction %':<12}"
    print(margin_header)
    print("-" * len(margin_header))
    for tau in [0.3, 0.5, 0.7, 0.9]:
        for area in [0, 50, 100, 200, 400]:
            std_fa = sum(1 for r in all_clean if r[f"flag_{tau}_{area}"])
            inn_fa = sum(1 for r in all_clean if r[f"flag_inner_{tau}_{area}"])
            disappeared = std_fa - inn_fa
            pct = (disappeared / std_fa * 100.0) if std_fa > 0 else 0.0
            print(f"{tau:<5.1f} | {area:<8d} | {std_fa:<14d} | {inn_fa:<16d} | {disappeared:<12d} | {pct:>10.1f}%")

    # 11. Generate Overlay Figure of 12 Worst Clean Panels
    print("\n[INFO] Generating overlay figure of 12 worst clean panels...")
    worst_clean = sorted(all_clean, key=lambda r: r["max_prob"], reverse=True)[:12]

    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    for idx, r in enumerate(worst_clean):
        ax = axes[idx // 4, idx % 4]
        img_rgb = r["image_rgb"]
        p_map = r["probs"]

        heat = cv2.applyColorMap((p_map * 255.0).clip(0, 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

        if img_rgb.shape[:2] == heat.shape[:2]:
            blended = cv2.addWeighted(img_rgb, 0.45, heat, 0.55, 0)
        else:
            blended = heat

        img_p_str = f" | img_p: {r['image_prob']:.3f}" if r.get("image_prob") is not None else ""
        ax.imshow(blended)
        ax.set_title(
            f"#{idx+1}: {r['class']} (doc: {r['doc_id'][:8]})\nmax_p: {r['max_prob']:.3f}{img_p_str} | edge_dist: {r['dist_edge']}px\narea@0.5: {r['area_05']}px",
            fontsize=9,
        )
        ax.axis("off")

    plt.tight_layout()
    fig_path = out_dir / "worst_clean_panels.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved worst clean overlay figure to {fig_path}")


if __name__ == "__main__":
    main()
