#!/usr/bin/env python3
"""Threshold and component area selection script for Phase 4 CNN copy-move detection.

Finds optimal (image_threshold, pixel_threshold, min_area) operating points using:
  - Synthetic validation split (500 forged, 500 clean twins)
  - All 4,039 real pristine held-out panels from pristine_heldout.zip

Optimization objective:
  Maximize recall on forged validation images subject to:
    1. False alarm rate on synthetic clean twins <= budget
    2. False alarm rate on real pristine panels <= budget

Evaluates multiple false alarm budgets (default: 0.01, 0.02, 0.03) and provides
both overall and per-class operating points (Blot/Gel vs Microscopy).

Strict protocol:
  - Zero BioFors test data or synthetic test data is ever accessed.
  - Operating points never accept zero recall.

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

from detector.cnn import ResNetCorrNet, ResNetUNet, load_checkpoint, sweep_thresholds, wilson_ci
from detector.colab_dataset import ColabDoppelDataset


def load_all_pristine_panels(pristine_zip_path: str, max_count: Optional[int] = None, seed: int = 42) -> List[Tuple[np.ndarray, str, str]]:
    """Load pristine panels from pristine_heldout.zip with doc_id and class."""
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

        if max_count is not None and max_count > 0 and len(rows) > max_count:
            rng = np.random.RandomState(seed)
            chosen_indices = rng.choice(len(rows), size=max_count, replace=False)
            rows = [rows[i] for i in chosen_indices]

        for r in rows:
            arc_name = f"pristine_heldout/{r['path']}"
            if arc_name in zf.namelist():
                img_bytes = zf.read(arc_name)
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    panels.append((img, r.get("doc_id", "unknown"), r.get("class", "unknown")))

    print(f"[INFO] Loaded {len(panels)} pristine held-out panels.")
    return panels


def parse_args(args: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Select optimal (image_threshold, threshold, min_area) operating points.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt).")
    parser.add_argument("--model", type=str, choices=["unet", "corr"], default=None, help="Model architecture.")
    parser.add_argument("--image-head", action="store_true", help="Force image head for CorrNet.")
    parser.add_argument("--val-manifest", type=str, default="data/synthetic/val_manifest.csv", help="Path to val_manifest.csv.")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory.")
    parser.add_argument("--val-subset", type=int, default=None, help="Optional subset of validation samples.")
    parser.add_argument("--pristine-zip", type=str, default="data/pristine_heldout.zip", help="Path to pristine held-out zip.")
    parser.add_argument("--pristine-subset", type=int, default=None, help="Optional max pristine panels (default None = all 4,039 panels).")
    parser.add_argument("--out-yaml", type=str, default="configs/cnn_threshold.yaml", help="Output path for YAML config.")
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.01, 0.02, 0.03], help="FAR budgets (default: 0.01 0.02 0.03).")
    parser.add_argument("--max-far", type=float, default=None, help="Legacy alias for max false alarm rate budget.")
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

    print(f"[INFO] Evaluating model: {model_type} (image_head={has_image_head}) from {ckpt_path.name}")
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
    if args.val_subset is not None and args.val_subset > 0:
        from torch.utils.data import Subset
        val_dataset = Subset(val_dataset, list(range(min(args.val_subset, len(val_dataset)))))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print(f"[INFO] Loaded {len(val_dataset)} validation samples.")

    # Sweep grid
    candidate_thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    candidate_areas = [0, 25, 32, 50, 64, 100, 150, 200, 300, 400, 500, 800, 1200, 1600]
    candidate_s = [0.0, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99] if has_image_head else [0.0]

    budgets = sorted(list(set(args.budgets + ([args.max_far] if args.max_far is not None else []))))
    print(f"[INFO] Evaluating budgets: {[f'{b*100:.1f}%' for b in budgets]}")

    # 1. Compute prediction probability maps and precalculate CC flags for validation split
    forged_records: List[Dict[str, Any]] = []
    clean_records: List[Dict[str, Any]] = []

    print("[INFO] Running inference on synthetic validation set...")
    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].to(device)
            mask_gt = batch["mask"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            is_forged = bool(batch["label"].item() > 0.5)

            meta_cls = batch["meta"].get("class", "unknown")
            cls_name = meta_cls[0] if isinstance(meta_cls, list) else meta_cls

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

            rec: Dict[str, Any] = {
                "image_prob": img_prob,
                "class": cls_name,
                "is_forged": is_forged,
            }

            for tau in candidate_thresholds:
                pred_bin = (probs_cropped >= tau).astype(np.uint8)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_bin, connectivity=8)
                areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([], dtype=int)

                for area in candidate_areas:
                    if is_forged:
                        if area == 0:
                            hit = bool(np.sum((pred_bin > 0) & (mask_gt_cropped > 0)) > 0)
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
                            flagged = bool(np.sum(pred_bin) > 0)
                        else:
                            flagged = bool(len(areas) > 0 and np.any(areas >= area))
                        rec[f"flag_{tau}_{area}"] = flagged

            if is_forged:
                forged_records.append(rec)
            else:
                clean_records.append(rec)

    # 2. Compute prediction probability maps and precalculate CC flags for pristine held-out panels
    pristine_panels = load_all_pristine_panels(args.pristine_zip, max_count=args.pristine_subset)
    pristine_records: List[Dict[str, Any]] = []

    if len(pristine_panels) > 0:
        print(f"[INFO] Running inference on {len(pristine_panels)} pristine panels...")
        with torch.no_grad():
            for p_img, _, p_cls in pristine_panels:
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
                probs_cropped = probs[top : top + h, left : left + w]

                p_rec: Dict[str, Any] = {
                    "image_prob": p_img_prob,
                    "class": p_cls,
                }

                for tau in candidate_thresholds:
                    pred_bin = (probs_cropped >= tau).astype(np.uint8)
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_bin, connectivity=8)
                    areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([], dtype=int)
                    for area in candidate_areas:
                        if area == 0:
                            flagged = bool(np.sum(pred_bin) > 0)
                        else:
                            flagged = bool(len(areas) > 0 and np.any(areas >= area))
                        p_rec[f"flag_{tau}_{area}"] = flagged

                pristine_records.append(p_rec)

    n_forged = len(forged_records)
    n_clean = len(clean_records)
    n_pristine = len(pristine_records)
    print(f"[INFO] Evaluated: {n_forged} forged, {n_clean} clean twins, {n_pristine} pristine panels.")

    # Slices for per-class operating points
    f_blot = [r for r in forged_records if r.get("class") == "Blot/Gel"]
    c_blot = [r for r in clean_records if r.get("class") == "Blot/Gel"]
    p_blot = [r for r in pristine_records if r.get("class") == "Blot/Gel"]

    f_mic = [r for r in forged_records if r.get("class") == "Microscopy"]
    c_mic = [r for r in clean_records if r.get("class") == "Microscopy"]
    p_mic = [r for r in pristine_records if r.get("class") == "Microscopy"]

    print(f"[INFO] Blot/Gel: {len(f_blot)} forged, {len(c_blot)} clean, {len(p_blot)} pristine.")
    print(f"[INFO] Microscopy: {len(f_mic)} forged, {len(c_mic)} clean, {len(p_mic)} pristine.")

    # 3. Multi-budget sweep across Overall, Blot/Gel, and Microscopy
    results_by_budget: Dict[float, Dict[str, Any]] = {}

    for b in budgets:
        sw_overall = sweep_thresholds(
            forged_records,
            clean_records,
            pristine_records,
            thresholds=candidate_thresholds,
            min_areas=candidate_areas,
            image_thresholds=candidate_s,
            max_far=b,
        )

        sw_blot = sweep_thresholds(
            f_blot,
            c_blot,
            p_blot,
            thresholds=candidate_thresholds,
            min_areas=candidate_areas,
            image_thresholds=candidate_s,
            max_far=b,
        ) if (len(f_blot) > 0 and len(c_blot) > 0 and len(p_blot) > 0) else sw_overall

        sw_mic = sweep_thresholds(
            f_mic,
            c_mic,
            p_mic,
            thresholds=candidate_thresholds,
            min_areas=candidate_areas,
            image_thresholds=candidate_s,
            max_far=b,
        ) if (len(f_mic) > 0 and len(c_mic) > 0 and len(p_mic) > 0) else sw_overall

        results_by_budget[b] = {
            "overall": sw_overall,
            "blot_gel": sw_blot,
            "microscopy": sw_mic,
        }

    # 4. Print side-by-side comparison tables
    for b in budgets:
        ov = results_by_budget[b]["overall"]
        blot = results_by_budget[b]["blot_gel"]
        mic = results_by_budget[b]["microscopy"]

        print("\n" + "=" * 115)
        print(f"OPERATING POINT COMPARISON: BUDGET <= {b*100:.1f}% FALSE ALARM RATE")
        print("Note: Image class is assumed known (BioFors metadata; classifier in product).")
        print("Format: Metric / Rate [Wilson 95% Confidence Interval]")
        print("=" * 115)
        col_w = 28
        fmt_row = lambda label, c1, c2, c3: f"{label:<25} | {c1:<{col_w}} | {c2:<{col_w}} | {c3:<{col_w}}"

        print(fmt_row("Scope", "Overall (Pooled)", "Blot/Gel", "Microscopy"))
        print("-" * 115)
        print(fmt_row("Meets Budget", str(ov["meets_budget"]), str(blot["meets_budget"]), str(mic["meets_budget"])))
        print(fmt_row("Image Threshold (s)", f"{ov['best_image_threshold']:.2f}", f"{blot['best_image_threshold']:.2f}", f"{mic['best_image_threshold']:.2f}"))
        print(fmt_row("Pixel Threshold (tau)", f"{ov['best_threshold']:.2f}", f"{blot['best_threshold']:.2f}", f"{mic['best_threshold']:.2f}"))
        print(fmt_row("Min Area (pixels)", f"{ov['best_min_area']} px", f"{blot['best_min_area']} px", f"{mic['best_min_area']} px"))

        ov_r = f"{ov['best_recall']*100:.1f}% [{ov['recall_ci'][0]*100:.1f}-{ov['recall_ci'][1]*100:.1f}]"
        bl_r = f"{blot['best_recall']*100:.1f}% [{blot['recall_ci'][0]*100:.1f}-{blot['recall_ci'][1]*100:.1f}]"
        mi_r = f"{mic['best_recall']*100:.1f}% [{mic['recall_ci'][0]*100:.1f}-{mic['recall_ci'][1]*100:.1f}]"
        print(fmt_row("Validation Recall", ov_r, bl_r, mi_r))

        ov_sf = f"{ov['best_synth_far']*100:.2f}% [{ov['synth_far_ci'][0]*100:.2f}-{ov['synth_far_ci'][1]*100:.2f}]"
        bl_sf = f"{blot['best_synth_far']*100:.2f}% [{blot['synth_far_ci'][0]*100:.2f}-{blot['synth_far_ci'][1]*100:.2f}]"
        mi_sf = f"{mic['best_synth_far']*100:.2f}% [{mic['synth_far_ci'][0]*100:.2f}-{mic['synth_far_ci'][1]*100:.2f}]"
        print(fmt_row("Synthetic Clean FAR", ov_sf, bl_sf, mi_sf))

        ov_pf = f"{ov['best_pristine_far']*100:.2f}% [{ov['pristine_far_ci'][0]*100:.2f}-{ov['pristine_far_ci'][1]*100:.2f}]"
        bl_pf = f"{blot['best_pristine_far']*100:.2f}% [{blot['pristine_far_ci'][0]*100:.2f}-{blot['pristine_far_ci'][1]*100:.2f}]"
        mi_pf = f"{mic['best_pristine_far']*100:.2f}% [{mic['pristine_far_ci'][0]*100:.2f}-{mic['pristine_far_ci'][1]*100:.2f}]"
        print(fmt_row("Real Pristine FAR", ov_pf, bl_pf, mi_pf))
        print("=" * 115)

    # 5. Write configs/cnn_threshold.yaml
    primary_budget = budgets[0]
    primary_ov = results_by_budget[primary_budget]["overall"]
    primary_blot = results_by_budget[primary_budget]["blot_gel"]
    primary_mic = results_by_budget[primary_budget]["microscopy"]

    out_yaml_path = Path(args.out_yaml)
    out_yaml_path.parent.mkdir(parents=True, exist_ok=True)

    yaml_lines = [
        "# Phase 4 CNN locked operating thresholds",
        "# Selected on synthetic validation split + 4,039 real pristine held-out train panels",
        f"# Selection date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "# Strict protocol: BioFors test split and synthetic test split were NEVER accessed.",
        "# Note: Image class (Blot/Gel vs Microscopy) is assumed known at test time.",
        "# BioFors provides this metadata; in the production pipeline, an upstream panel",
        "# classifier supplies the domain tag before routing to detector thresholds.",
        "",
        f'model_type: "{model_type}"',
        f'checkpoint_file: "{ckpt_path.name}"',
        f"primary_budget: {primary_budget}",
        "",
        "# Root-level keys represent overall operating point at primary budget for backward compatibility",
        f"image_threshold: {primary_ov['best_image_threshold']}",
        f"pixel_threshold: {primary_ov['best_threshold']}",
        f"min_area_pixels: {primary_ov['best_min_area']}",
        "",
        "overall:",
        f"  image_threshold: {primary_ov['best_image_threshold']}",
        f"  pixel_threshold: {primary_ov['best_threshold']}",
        f"  min_area_pixels: {primary_ov['best_min_area']}",
        "  metrics:",
        f"    validation_recall: {round(float(primary_ov['best_recall']), 4)}",
        f"    validation_recall_ci: [{round(primary_ov['recall_ci'][0], 4)}, {round(primary_ov['recall_ci'][1], 4)}]",
        f"    synthetic_val_clean_far: {round(float(primary_ov['best_synth_far']), 6)}",
        f"    synthetic_val_clean_far_ci: [{round(primary_ov['synth_far_ci'][0], 6)}, {round(primary_ov['synth_far_ci'][1], 6)}]",
        f"    real_pristine_heldout_far: {round(float(primary_ov['best_pristine_far']), 6)}",
        f"    real_pristine_heldout_far_ci: [{round(primary_ov['pristine_far_ci'][0], 6)}, {round(primary_ov['pristine_far_ci'][1], 6)}]",
        f"    budget_met: {str(primary_ov['meets_budget']).lower()}",
        "",
        "per_class:",
        "  blot_gel:",
        f"    image_threshold: {primary_blot['best_image_threshold']}",
        f"    pixel_threshold: {primary_blot['best_threshold']}",
        f"    min_area_pixels: {primary_blot['best_min_area']}",
        "    metrics:",
        f"      validation_recall: {round(float(primary_blot['best_recall']), 4)}",
        f"      validation_recall_ci: [{round(primary_blot['recall_ci'][0], 4)}, {round(primary_blot['recall_ci'][1], 4)}]",
        f"      synthetic_val_clean_far: {round(float(primary_blot['best_synth_far']), 6)}",
        f"      synthetic_val_clean_far_ci: [{round(primary_blot['synth_far_ci'][0], 6)}, {round(primary_blot['synth_far_ci'][1], 6)}]",
        f"      real_pristine_heldout_far: {round(float(primary_blot['best_pristine_far']), 6)}",
        f"      real_pristine_heldout_far_ci: [{round(primary_blot['pristine_far_ci'][0], 6)}, {round(primary_blot['pristine_far_ci'][1], 6)}]",
        f"      budget_met: {str(primary_blot['meets_budget']).lower()}",
        "  microscopy:",
        f"    image_threshold: {primary_mic['best_image_threshold']}",
        f"    pixel_threshold: {primary_mic['best_threshold']}",
        f"    min_area_pixels: {primary_mic['best_min_area']}",
        "    metrics:",
        f"      validation_recall: {round(float(primary_mic['best_recall']), 4)}",
        f"      validation_recall_ci: [{round(primary_mic['recall_ci'][0], 4)}, {round(primary_mic['recall_ci'][1], 4)}]",
        f"      synthetic_val_clean_far: {round(float(primary_mic['best_synth_far']), 6)}",
        f"      synthetic_val_clean_far_ci: [{round(primary_mic['synth_far_ci'][0], 6)}, {round(primary_mic['synth_far_ci'][1], 6)}]",
        f"      real_pristine_heldout_far: {round(float(primary_mic['best_pristine_far']), 6)}",
        f"      real_pristine_heldout_far_ci: [{round(primary_mic['pristine_far_ci'][0], 6)}, {round(primary_mic['pristine_far_ci'][1], 6)}]",
        f"      budget_met: {str(primary_mic['meets_budget']).lower()}",
        "",
        "budgets:",
    ]

    for b in budgets:
        b_ov = results_by_budget[b]["overall"]
        b_bl = results_by_budget[b]["blot_gel"]
        b_mi = results_by_budget[b]["microscopy"]
        yaml_lines.extend([
            f"  budget_{int(round(b*100))}pct:",
            f"    target_far: {b}",
            "    overall:",
            f"      image_threshold: {b_ov['best_image_threshold']}",
            f"      pixel_threshold: {b_ov['best_threshold']}",
            f"      min_area_pixels: {b_ov['best_min_area']}",
            f"      validation_recall: {round(float(b_ov['best_recall']), 4)}",
            f"      synthetic_val_clean_far: {round(float(b_ov['best_synth_far']), 6)}",
            f"      real_pristine_heldout_far: {round(float(b_ov['best_pristine_far']), 6)}",
            f"      budget_met: {str(b_ov['meets_budget']).lower()}",
            "    blot_gel:",
            f"      image_threshold: {b_bl['best_image_threshold']}",
            f"      pixel_threshold: {b_bl['best_threshold']}",
            f"      min_area_pixels: {b_bl['best_min_area']}",
            f"      validation_recall: {round(float(b_bl['best_recall']), 4)}",
            f"      synthetic_val_clean_far: {round(float(b_bl['best_synth_far']), 6)}",
            f"      real_pristine_heldout_far: {round(float(b_bl['best_pristine_far']), 6)}",
            f"      budget_met: {str(b_bl['meets_budget']).lower()}",
            "    microscopy:",
            f"      image_threshold: {b_mi['best_image_threshold']}",
            f"      pixel_threshold: {b_mi['best_threshold']}",
            f"      min_area_pixels: {b_mi['best_min_area']}",
            f"      validation_recall: {round(float(b_mi['best_recall']), 4)}",
            f"      synthetic_val_clean_far: {round(float(b_mi['best_synth_far']), 6)}",
            f"      real_pristine_heldout_far: {round(float(b_mi['best_pristine_far']), 6)}",
            f"      budget_met: {str(b_mi['meets_budget']).lower()}",
        ])

    with open(out_yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines) + "\n")

    print(f"[INFO] Wrote locked threshold configuration to {out_yaml_path}")


if __name__ == "__main__":
    main()
