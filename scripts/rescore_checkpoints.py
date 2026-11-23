#!/usr/bin/env python3
"""Rescore checkpoints in a training run folder using the full (s, tau, min_area) grid sweep.

Evaluates all epoch_XX.pt and last.pt checkpoints on synthetic validation data
and pristine held-out panels subject to FAR <= 1.0% on both sets.

Strict protocol:
  - BioFors test split and synthetic test split are NEVER accessed.
  - meets_budget is True ONLY IF recall > 0 and FAR <= 1.0% on both sets.
"""

import argparse
import csv
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from detector.cnn import ResNetCorrNet, ResNetUNet, load_checkpoint, sweep_thresholds
from detector.cnn.infer import compute_top_k_mean
from detector.colab_dataset import ColabDoppelDataset


def load_pristine_panels(pristine_zip_path: str, max_count: int = 1000, seed: int = 42) -> List[Tuple[np.ndarray, str, str]]:
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

        rng = np.random.RandomState(seed)
        if len(rows) > max_count:
            chosen_indices = rng.choice(len(rows), size=max_count, replace=False)
            chosen_rows = [rows[i] for i in chosen_indices]
        else:
            chosen_rows = rows

        for row in chosen_rows:
            p_rel = row["path"]
            matching_files = [n for n in zf.namelist() if n.endswith(p_rel)]
            if matching_files:
                img_bytes = zf.read(matching_files[0])
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    panels.append((img, row.get("doc_id", "unknown"), row.get("class", "unknown")))
    print(f"[INFO] Loaded {len(panels)} pristine panels from {pristine_zip_path}")
    return panels


def parse_args(args: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Rescore CNN checkpoints across validation sets.")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to run directory containing checkpoints.")
    parser.add_argument("--val-manifest", type=str, default="data/synthetic/val_manifest.csv", help="Path to val_manifest.csv.")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory.")
    parser.add_argument("--val-subset", type=int, default=None, help="Optional subset of validation samples.")
    parser.add_argument("--pristine-zip", type=str, default="data/pristine_heldout.zip", help="Path to pristine held-out zip.")
    parser.add_argument("--pristine-subset", type=int, default=1000, help="Max pristine panels (default 1000).")
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.01, 0.02, 0.03], help="False alarm rate budgets (default: 0.01 0.02 0.03).")
    parser.add_argument("--max-far", type=float, default=None, help="Legacy alias for max false alarm rate budget.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device.")
    parser.add_argument("--out-csv", type=str, default=None, help="Optional output CSV path for rescore summary.")
    return parser.parse_args(args)


def evaluate_checkpoint(
    model: nn.Module,
    val_loader: DataLoader,
    pristine_panels: List[Tuple[np.ndarray, str, str]],
    thresholds: List[float],
    min_areas: List[int],
    image_thresholds: List[float],
    device: torch.device,
    budgets: List[float],
) -> Dict[str, Any]:
    """Run validation inference and grid sweep for a single model checkpoint across budgets."""
    model.eval()

    all_scores: List[float] = []
    all_labels: List[int] = []

    forged_records: List[Dict[str, Any]] = []
    clean_records: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in val_loader:
            image = batch["image"].to(device)
            mask_gt = batch["mask"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            is_forged = bool(batch["label"].item() > 0.5)

            meta_cls = batch["meta"].get("class", "unknown")
            cls_name = meta_cls[0] if isinstance(meta_cls, list) else meta_cls

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

            ys, xs = np.where(vm_bool)
            if len(ys) > 0 and len(xs) > 0:
                top_y, bot_y = int(ys.min()), int(ys.max() + 1)
                left_x, right_x = int(xs.min()), int(xs.max() + 1)
            else:
                top_y, bot_y = 0, probs_np.shape[0]
                left_x, right_x = 0, probs_np.shape[1]

            probs_cropped = probs_np[top_y:bot_y, left_x:right_x]
            mask_gt_cropped = mask_gt_np[top_y:bot_y, left_x:right_x]

            top100_mean = compute_top_k_mean(probs_cropped, k=100)
            all_scores.append(top100_mean)
            all_labels.append(1 if is_forged else 0)

            rec: Dict[str, Any] = {
                "top100_mean": top100_mean,
                "image_prob": img_prob,
                "class": cls_name,
            }

            for tau in thresholds:
                pred_bin = (probs_cropped >= tau).astype(np.uint8)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_bin, connectivity=8)
                areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([], dtype=int)

                for area in min_areas:
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
                        rec[f"detected_{tau}_{area}"] = hit
                    else:
                        if area == 0:
                            flagged = bool(np.sum(pred_bin) > 0)
                        else:
                            flagged = bool(len(areas) > 0 and np.any(areas >= area))
                        rec[f"flagged_{tau}_{area}"] = flagged

            if is_forged:
                forged_records.append(rec)
            else:
                clean_records.append(rec)

    # Pristine evaluation
    pristine_records: List[Dict[str, Any]] = []
    if len(pristine_panels) > 0:
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

                padded_img = cv2.copyMakeBorder(p_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                vm = np.zeros((target_h, target_w), dtype=np.uint8)
                vm[top : top + h, left : left + w] = 1

                t_img = torch.from_numpy(padded_img.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0).to(device)
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
                for tau in thresholds:
                    pred_bin = (probs_cropped >= tau).astype(np.uint8)
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_bin, connectivity=8)
                    areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([], dtype=int)
                    for area in min_areas:
                        if area == 0:
                            flagged = bool(np.sum(pred_bin) > 0)
                        else:
                            flagged = bool(len(areas) > 0 and np.any(areas >= area))
                        p_rec[f"flagged_{tau}_{area}"] = flagged
                pristine_records.append(p_rec)

    # ROC AUC
    try:
        from sklearn.metrics import roc_auc_score
        val_auc = float(roc_auc_score(all_labels, all_scores))
    except Exception:
        val_auc = 0.50

    # Multi-budget sweeps: Overall, Blot/Gel, and Microscopy
    budget_results: Dict[float, Dict[str, Any]] = {}
    f_blot = [r for r in forged_records if r.get("class") == "Blot/Gel"]
    c_blot = [r for r in clean_records if r.get("class") == "Blot/Gel"]
    p_blot = [r for r in pristine_records if r.get("class") == "Blot/Gel"]

    f_mic = [r for r in forged_records if r.get("class") == "Microscopy"]
    c_mic = [r for r in clean_records if r.get("class") == "Microscopy"]
    p_mic = [r for r in pristine_records if r.get("class") == "Microscopy"]

    for b in budgets:
        sw_overall = sweep_thresholds(
            forged_records,
            clean_records,
            pristine_records,
            thresholds=thresholds,
            min_areas=min_areas,
            image_thresholds=image_thresholds,
            max_far=b,
        )
        sw_blot = sweep_thresholds(
            f_blot,
            c_blot,
            p_blot,
            thresholds=thresholds,
            min_areas=min_areas,
            image_thresholds=image_thresholds,
            max_far=b,
        ) if (len(f_blot) > 0 and len(c_blot) > 0 and len(p_blot) > 0) else sw_overall

        sw_mic = sweep_thresholds(
            f_mic,
            c_mic,
            p_mic,
            thresholds=thresholds,
            min_areas=min_areas,
            image_thresholds=image_thresholds,
            max_far=b,
        ) if (len(f_mic) > 0 and len(c_mic) > 0 and len(p_mic) > 0) else sw_overall

        budget_results[b] = {
            "overall": sw_overall,
            "blot_gel": sw_blot,
            "microscopy": sw_mic,
        }

    return {
        "roc_auc": val_auc,
        "budgets": budget_results,
    }


def main():
    args = parse_args()
    device = torch.device(args.device)

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Discover checkpoints
    ckpt_files = sorted(list(run_dir.glob("*.pt")))
    if not ckpt_files:
        raise FileNotFoundError(f"No .pt checkpoints found in {run_dir}")

    def _sort_key(p: Path) -> int:
        match = re.search(r"epoch_(\d+)", p.name)
        if match:
            return int(match.group(1))
        if "best" in p.name:
            return 9998
        if "last" in p.name:
            return 9999
        return 5000

    ckpt_files.sort(key=_sort_key)
    print(f"[INFO] Found {len(ckpt_files)} checkpoints to rescore in {run_dir}: {[f.name for f in ckpt_files]}")

    # Load validation dataset
    val_manifest = Path(args.val_manifest)
    if not val_manifest.exists():
        alt_manifest = Path(args.data_dir) / "synthetic" / "val_manifest.csv"
        if alt_manifest.exists():
            val_manifest = alt_manifest
        else:
            raise FileNotFoundError(f"Val manifest not found: {val_manifest}")

    val_dataset = ColabDoppelDataset(
        manifest_path=str(val_manifest),
        images_dir=args.data_dir,
        split="val",
        crop_size=None,
        pad_multiple=32,
        augment=False,
    )
    if args.val_subset is not None and args.val_subset > 0:
        n_sub = min(args.val_subset, len(val_dataset))
        val_dataset = Subset(val_dataset, list(range(n_sub)))
        print(f"[INFO] Using validation subset: {len(val_dataset)} images")

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Load pristine panels
    pristine_panels = load_pristine_panels(args.pristine_zip, max_count=args.pristine_subset)

    # Extended sweep grid
    thresholds = [0.3, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    min_areas = [0, 25, 50, 100, 200, 400, 800, 1200, 1600]

    budgets = sorted(list(set(args.budgets + ([args.max_far] if args.max_far is not None else []))))
    print(f"[INFO] Evaluating FAR budgets: {[f'{b*100:.1f}%' for b in budgets]}")

    rows = []

    for ckpt_path in ckpt_files:
        print(f"\n[INFO] Rescoring {ckpt_path.name}...")
        ckpt = load_checkpoint(ckpt_path, map_location=device)
        cfg = ckpt.get("config", {})
        state_dict = ckpt.get("model_state_dict", ckpt)

        model_type = cfg.get("model")
        if model_type is None:
            if any("corr_proj" in k or "proj.weight" in k for k in state_dict.keys()):
                model_type = "corr"
            else:
                model_type = "unet"

        has_image_head = bool(cfg.get("image_head", False) or any("image_head_fc" in k for k in state_dict.keys()))

        if model_type == "corr":
            model = ResNetCorrNet(pretrained=False, image_head=has_image_head)
        else:
            model = ResNetUNet(pretrained=False)

        model.load_state_dict(state_dict)
        model.to(device)

        image_thresholds = [0.0, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99] if has_image_head else [0.0]

        res = evaluate_checkpoint(
            model=model,
            val_loader=val_loader,
            pristine_panels=pristine_panels,
            thresholds=thresholds,
            min_areas=min_areas,
            image_thresholds=image_thresholds,
            device=device,
            budgets=budgets,
        )

        epoch_val = ckpt.get("epoch", _sort_key(ckpt_path))
        for b in budgets:
            b_info = res["budgets"][b]
            ov = b_info["overall"]
            blot = b_info["blot_gel"]
            mic = b_info["microscopy"]

            rows.append({
                "checkpoint": ckpt_path.name,
                "epoch": epoch_val,
                "budget": b,
                "meets_budget": ov["meets_budget"],
                "recall": ov["best_recall"],
                "s": ov["best_image_threshold"],
                "tau": ov["best_threshold"],
                "min_area": ov["best_min_area"],
                "synth_far": ov["best_synth_far"],
                "pristine_far": ov["best_pristine_far"],
                "roc_auc": res["roc_auc"],
                "blot_meets": blot["meets_budget"],
                "blot_recall": blot["best_recall"],
                "blot_s": blot["best_image_threshold"],
                "blot_tau": blot["best_threshold"],
                "blot_area": blot["best_min_area"],
                "blot_synth_far": blot["best_synth_far"],
                "blot_pristine_far": blot["best_pristine_far"],
                "mic_meets": mic["meets_budget"],
                "mic_recall": mic["best_recall"],
                "mic_s": mic["best_image_threshold"],
                "mic_tau": mic["best_threshold"],
                "mic_area": mic["best_min_area"],
                "mic_synth_far": mic["best_synth_far"],
                "mic_pristine_far": mic["best_pristine_far"],
            })

    # Print summary table
    print("\n" + "=" * 125)
    print("CHECKPOINT RESCORE SUMMARY (OVERALL BY BUDGET)")
    print(f"Run directory: {run_dir}")
    print("=" * 125)
    header = f"{'Checkpoint':<16} | {'Epoch':<5} | {'Budget':<7} | {'Meets Budget':<12} | {'Rec@Budget':<10} | {'s':<5} | {'tau':<5} | {'Area':<6} | {'Synth FAR':<9} | {'Pristine FAR':<12} | {'AUC':<6}"
    print(header)
    print("-" * 125)

    for r in rows:
        b_str = f"{r['budget']*100:.1f}%"
        meets_str = "True" if r["meets_budget"] else "False"
        rec_str = f"{r['recall']*100:.1f}%"
        s_str = f"{r['s']:.2f}"
        tau_str = f"{r['tau']:.2f}"
        area_str = f"{r['min_area']}"
        s_far_str = f"{r['synth_far']*100:.2f}%"
        p_far_str = f"{r['pristine_far']*100:.2f}%"
        auc_str = f"{r['roc_auc']:.4f}"
        print(f"{r['checkpoint']:<16} | {r['epoch']:<5} | {b_str:<7} | {meets_str:<12} | {rec_str:<10} | {s_str:<5} | {tau_str:<5} | {area_str:<6} | {s_far_str:<9} | {p_far_str:<12} | {auc_str:<6}")

    print("=" * 125 + "\n")

    # Print per-class side-by-side summary
    print("=" * 135)
    print("PER-CLASS OPERATING POINTS SIDE-BY-SIDE")
    print("Note: Class (Blot/Gel vs Microscopy) is assumed known from metadata or upstream classifier.")
    print("=" * 135)
    pc_header = f"{'Checkpoint':<16} | {'Budget':<7} | {'Scope':<12} | {'Meets':<6} | {'Recall':<8} | {'s':<5} | {'tau':<5} | {'Area':<6} | {'Synth FAR':<10} | {'Pristine FAR':<12}"
    print(pc_header)
    print("-" * 135)

    for r in rows:
        b_str = f"{r['budget']*100:.1f}%"
        # Overall
        print(f"{r['checkpoint']:<16} | {b_str:<7} | {'Overall':<12} | {str(r['meets_budget']):<6} | {r['recall']*100:>6.1f}% | {r['s']:.2f} | {r['tau']:.2f} | {r['min_area']:<6} | {r['synth_far']*100:>8.2f}% | {r['pristine_far']*100:>10.2f}%")
        # Blot/Gel
        print(f"{'':<16} | {'':<7} | {'Blot/Gel':<12} | {str(r['blot_meets']):<6} | {r['blot_recall']*100:>6.1f}% | {r['blot_s']:.2f} | {r['blot_tau']:.2f} | {r['blot_area']:<6} | {r['blot_synth_far']*100:>8.2f}% | {r['blot_pristine_far']*100:>10.2f}%")
        # Microscopy
        print(f"{'':<16} | {'':<7} | {'Microscopy':<12} | {str(r['mic_meets']):<6} | {r['mic_recall']*100:>6.1f}% | {r['mic_s']:.2f} | {r['mic_tau']:.2f} | {r['mic_area']:<6} | {r['mic_synth_far']*100:>8.2f}% | {r['mic_pristine_far']*100:>10.2f}%")
        print("-" * 135)

    print("=" * 135 + "\n")

    if args.out_csv is not None:
        out_csv_path = Path(args.out_csv)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "checkpoint", "epoch", "budget", "meets_budget", "recall", "s", "tau", "min_area", "synth_far", "pristine_far", "roc_auc",
                "blot_meets", "blot_recall", "blot_s", "blot_tau", "blot_area", "blot_synth_far", "blot_pristine_far",
                "mic_meets", "mic_recall", "mic_s", "mic_tau", "mic_area", "mic_synth_far", "mic_pristine_far",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"[INFO] Wrote rescore table to {out_csv_path}")


if __name__ == "__main__":
    main()
