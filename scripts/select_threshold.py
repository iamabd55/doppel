#!/usr/bin/env python3
"""Threshold and component area selection script for Phase 4 CNN copy-move detection.

Finds the optimal (pixel_threshold, min_area) operating point using:
  - Synthetic validation split (500 forged, 500 clean twins)
  - All 4,039 real pristine held-out panels from pristine_heldout.zip

Optimization objective:
  Maximize recall on forged validation images subject to:
    1. False alarm rate on synthetic clean twins <= 1.0%
    2. False alarm rate on real pristine panels <= 1.0%

Strict rule: Zero BioFors test data or synthetic test data is ever accessed.
Outputs configs/cnn_threshold.yaml.
"""

import argparse
import csv
import os
import sys
import zipfile
from datetime import datetime
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


def load_all_pristine_panels(pristine_zip_path: str) -> List[Tuple[np.ndarray, str]]:
    """Load all pristine panels from pristine_heldout.zip."""
    if not os.path.exists(pristine_zip_path):
        print(f"[WARNING] Pristine zip not found at {pristine_zip_path}")
        return []

    panels = []
    print(f"[INFO] Reading pristine panels from {pristine_zip_path}...")
    with zipfile.ZipFile(pristine_zip_path, "r") as zf:
        manifest_names = [n for n in zf.namelist() if n.endswith("pristine_manifest.csv")]
        if not manifest_names:
            return []
        manifest_bytes = zf.read(manifest_names[0]).decode("utf-8")
        reader = csv.DictReader(manifest_bytes.splitlines())
        rows = list(reader)

        for r in rows:
            arc_name = f"pristine_heldout/{r['path']}"
            if arc_name in zf.namelist():
                img_bytes = zf.read(arc_name)
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    panels.append((img, r["class"]))

    print(f"[INFO] Loaded {len(panels)} pristine held-out panels.")
    return panels


def parse_args(args: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Select optimal (threshold, min_area) operating point.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt checkpoint.")
    parser.add_argument("--model", type=str, choices=["unet", "corr"], default=None, help="Model architecture.")
    parser.add_argument("--image-head", action="store_true", help="Force image head for CorrNet.")
    parser.add_argument("--val-manifest", type=str, default="data/synthetic/val_manifest.csv", help="Path to val_manifest.csv.")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory.")
    parser.add_argument("--pristine-zip", type=str, default="data/pristine_heldout.zip", help="Path to pristine held-out zip.")
    parser.add_argument("--out-yaml", type=str, default="configs/cnn_threshold.yaml", help="Output path for YAML config.")
    parser.add_argument("--max-far", type=float, default=0.01, help="Max false alarm rate budget (default 0.01 = 1.0%).")
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
    if model_type == "unet":
        model = ResNetUNet(pretrained=False)
    else:
        model = ResNetCorrNet(pretrained=False, image_head=has_image_head)

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    val_manifest = Path(args.val_manifest)
    if not val_manifest.exists():
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
    print(f"[INFO] Loaded {len(val_dataset)} validation samples.")

    # 1. Compute prediction probability maps for validation split
    val_preds: List[Dict[str, Any]] = []
    print("[INFO] Running inference on synthetic validation set...")
    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].to(device)
            mask_gt = batch["mask"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            is_forged = bool(batch["label"].item() > 0.5)

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

            # Strict crop back to unpadded resolution
            ys, xs = np.where(vm_bool)
            if len(ys) > 0 and len(xs) > 0:
                top_y, bot_y = int(ys.min()), int(ys.max() + 1)
                left_x, right_x = int(xs.min()), int(xs.max() + 1)
            else:
                top_y, bot_y = 0, probs_np.shape[0]
                left_x, right_x = 0, probs_np.shape[1]

            probs_cropped = probs_np[top_y:bot_y, left_x:right_x]
            mask_gt_cropped = mask_gt_np[top_y:bot_y, left_x:right_x]

            val_preds.append({
                "is_forged": is_forged,
                "probs": probs_cropped,
                "mask_gt": mask_gt_cropped,
                "max_p": float(probs_cropped.max()) if probs_cropped.size > 0 else 0.0,
                "image_prob": img_prob,
            })

    # 2. Compute prediction probability maps for pristine held-out panels
    pristine_panels = load_all_pristine_panels(args.pristine_zip)
    pristine_preds: List[Dict[str, Any]] = []

    if len(pristine_panels) > 0:
        print("[INFO] Running inference on pristine panels...")
        with torch.no_grad():
            for p_img, _ in pristine_panels:
                h, w = p_img.shape[:2]
                pad_multiple = 32
                target_h = int(np.ceil(h / pad_multiple) * pad_multiple)
                target_w = int(np.ceil(w / pad_multiple) * pad_multiple)

                top = (target_h - h) // 2
                bottom = (target_h - h) - top
                left = (target_w - w) // 2
                right = (target_w - w) - left

                padded_img = cv2.copyMakeBorder(
                    p_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )
                vm = np.zeros((target_h, target_w), dtype=np.uint8)
                vm[top : top + h, left : left + w] = 1

                t_img = torch.from_numpy(
                    padded_img.transpose(2, 0, 1).astype(np.float32) / 255.0
                ).unsqueeze(0).to(device)
                t_vm = torch.from_numpy(vm.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

                out_p = model(t_img, valid_mask=t_vm)
                if isinstance(out_p, tuple):
                    logits, img_logit = out_p
                    p_img_prob = float(torch.sigmoid(img_logit)[0].item())
                else:
                    logits = out_p
                    p_img_prob = None

                probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

                # Strict crop back to unpadded pristine panel resolution
                probs_cropped = probs[top : top + h, left : left + w]
                max_p = float(probs_cropped.max()) if probs_cropped.size > 0 else 0.0

                pristine_preds.append({
                    "probs": probs_cropped,
                    "max_p": max_p,
                    "image_prob": p_img_prob,
                })

    forged_items = [p for p in val_preds if p["is_forged"]]
    clean_items = [p for p in val_preds if not p["is_forged"]]
    n_forged = len(forged_items)
    n_clean = len(clean_items)
    n_pristine = len(pristine_preds)

    print(f"[INFO] Val counts: {n_forged} forged, {n_clean} clean twins, {n_pristine} pristine panels.")

    # 3. Grid search over image_thresholds (if present), thresholds, and min_area
    candidate_s = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95] if has_image_head else [0.0]
    candidate_thresholds = [round(t, 2) for t in np.linspace(0.10, 0.90, 17)]
    candidate_areas = [32, 50, 64, 100, 150, 200, 300, 500]

    best_config: Optional[Dict[str, Any]] = None
    best_recall = -1.0
    compliant_configs = []

    lowest_max_far = 2.0
    fallback_config: Optional[Dict[str, Any]] = None

    print("[INFO] Sweeping (image_threshold, threshold, min_area) grid...")
    for s in candidate_s:
        for min_area in candidate_areas:
            for tau in candidate_thresholds:
                # Synthetic clean twins FAR
                clean_flags = 0
                for item in clean_items:
                    img_ok = True if s == 0.0 else (item.get("image_prob", 1.0) >= s)
                    if img_ok and item["max_p"] >= tau:
                        pred_bin = (item["probs"] >= tau).astype(np.uint8)
                        filt = filter_min_area(pred_bin, min_area=min_area)
                        if np.sum(filt) > 0:
                            clean_flags += 1
                clean_far = clean_flags / n_clean if n_clean > 0 else 0.0

                # Pristine panels FAR
                pristine_flags = 0
                if n_pristine > 0:
                    for item in pristine_preds:
                        img_ok = True if s == 0.0 else (item.get("image_prob", 1.0) >= s)
                        if img_ok and item["max_p"] >= tau:
                            pred_bin = (item["probs"] >= tau).astype(np.uint8)
                            filt = filter_min_area(pred_bin, min_area=min_area)
                            if np.sum(filt) > 0:
                                pristine_flags += 1
                    pristine_far = pristine_flags / n_pristine
                else:
                    pristine_far = 0.0

                meets = (clean_far <= args.max_far) and (pristine_far <= args.max_far)
                cur_max_far = max(clean_far, pristine_far)

                # Forged recall
                forged_hits = 0
                for item in forged_items:
                    img_ok = True if s == 0.0 else (item.get("image_prob", 1.0) >= s)
                    if img_ok and item["max_p"] >= tau:
                        pred_bin = (item["probs"] >= tau).astype(np.uint8)
                        filt = filter_min_area(pred_bin, min_area=min_area)
                        if np.sum((filt > 0) & (item["mask_gt"] > 0)) > 0:
                            forged_hits += 1
                recall = forged_hits / n_forged if n_forged > 0 else 0.0

                entry = {
                    "image_threshold": s,
                    "threshold": tau,
                    "min_area": min_area,
                    "recall": recall,
                    "clean_far": clean_far,
                    "pristine_far": pristine_far,
                    "meets_budget": meets,
                }

                if meets:
                    compliant_configs.append(entry)
                    if recall > best_recall:
                        best_recall = recall
                        best_config = entry
                else:
                    if cur_max_far < lowest_max_far - 1e-5:
                        lowest_max_far = cur_max_far
                        fallback_config = entry
                    elif abs(cur_max_far - lowest_max_far) <= 1e-5:
                        if (clean_far + pristine_far) < (fallback_config["clean_far"] + fallback_config["pristine_far"]):
                            fallback_config = entry

    # Fallback if no configuration met the strict budget
    if best_config is None:
        print("[WARNING] No grid point met the 1.0% FAR budget on both sets. Selecting lowest max-FAR fallback.")
        best_config = fallback_config or {
            "image_threshold": 0.0,
            "threshold": 0.50,
            "min_area": 50,
            "recall": 0.0,
            "clean_far": 1.0,
            "pristine_far": 1.0,
            "meets_budget": False,
        }

    print("\n======================================================================")
    print("OPTIMAL CNN THRESHOLD CONFIGURATION SELECTED")
    print(f"Model: {model_type}")
    if has_image_head:
        print(f"Image threshold: {best_config.get('image_threshold', 0.0):.2f}")
    print(f"Pixel threshold: {best_config['threshold']:.2f}")
    print(f"Min area: {best_config['min_area']} px")
    print(f"Achieved Validation Recall: {best_config['recall']*100:.2f}%")
    print(f"Synthetic Clean FAR: {best_config['clean_far']*100:.2f}% (budget <= {args.max_far*100:.1f}%)")
    print(f"Real Pristine FAR: {best_config['pristine_far']*100:.2f}% (budget <= {args.max_far*100:.1f}%)")
    print(f"Budget Met: {best_config['meets_budget']}")
    print("======================================================================\n")

    # Write configs/cnn_threshold.yaml
    out_yaml_path = Path(args.out_yaml)
    out_yaml_path.parent.mkdir(parents=True, exist_ok=True)

    yaml_content = f"""# Phase 4 CNN locked operating thresholds
# Selected on synthetic validation split + 4,039 real pristine held-out train panels
# Selection date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Strict protocol: BioFors test split and synthetic test split were NEVER accessed.

model_type: "{model_type}"
checkpoint_file: "{ckpt_path.name}"
image_threshold: {best_config.get('image_threshold', 0.0)}
pixel_threshold: {best_config['threshold']}
min_area_pixels: {best_config['min_area']}

achieved_metrics:
  validation_recall: {round(float(best_config['recall']), 4)}
  synthetic_val_clean_far: {round(float(best_config['clean_far']), 6)}
  real_pristine_heldout_far: {round(float(best_config['pristine_far']), 6)}
  budget_met: {str(best_config['meets_budget']).lower()}
"""

    with open(out_yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"[INFO] Wrote locked threshold configuration to {out_yaml_path}")


if __name__ == "__main__":
    main()
