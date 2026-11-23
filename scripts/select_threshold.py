#!/usr/bin/env python3
"""Threshold and component area selection script for Phase 4 CNN copy-move detection.

Finds optimal (image_threshold, pixel_threshold, min_area) operating points using:
  - Synthetic validation split (500 forged, 500 clean twins)
  - All 4,039 real pristine held-out panels from pristine_heldout.zip

Optimization objective:
  Maximize recall on forged validation images subject to:
    1. False alarm rate on synthetic clean twins <= budget
    2. False alarm rate on real pristine panels <= budget

Evaluates multiple false alarm budgets (default: 0.01, 0.015, 0.02, 0.03) and provides
both overall and per-class operating points (Blot/Gel vs Microscopy).

Features:
  - Caches per-image model predictions (max prob, image-head prob, predicted areas)
    to an .npz file so subsequent sweeps and re-runs execute instantaneously.
  - Generates comprehensive side-by-side comparison tables with 95% Wilson CIs.
  - Writes configs/cnn_threshold.yaml documenting locked thresholds.

Strict protocol:
  - Zero BioFors test data or synthetic test data is ever accessed.
  - Operating points never accept zero recall.
"""

import argparse
import csv
import os
import sys
import time
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

from detector.cnn import ResNetCorrNet, ResNetUNet, load_checkpoint, wilson_ci
from detector.colab_dataset import ColabDoppelDataset


def load_all_pristine_panels(
    pristine_zip_path: str, max_count: Optional[int] = None, seed: int = 42
) -> List[Tuple[np.ndarray, str, str]]:
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
    parser = argparse.ArgumentParser(
        description="Select optimal (image_threshold, threshold, min_area) operating points."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt).")
    parser.add_argument("--model", type=str, choices=["unet", "corr"], default=None, help="Model architecture.")
    parser.add_argument("--image-head", action="store_true", help="Force image head for CorrNet.")
    parser.add_argument(
        "--val-manifest", type=str, default="data/synthetic/val_manifest.csv", help="Path to val_manifest.csv."
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory.")
    parser.add_argument("--val-subset", type=int, default=None, help="Optional subset of validation samples.")
    parser.add_argument(
        "--pristine-zip", type=str, default="data/pristine_heldout.zip", help="Path to pristine held-out zip."
    )
    parser.add_argument(
        "--pristine-subset",
        type=int,
        default=None,
        help="Optional max pristine panels (default None = all 4,039 panels).",
    )
    parser.add_argument(
        "--out-yaml", type=str, default="configs/cnn_threshold.yaml", help="Output path for YAML config."
    )
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=[0.01, 0.015, 0.02, 0.03],
        help="FAR budgets (default: 0.01 0.015 0.02 0.03).",
    )
    parser.add_argument("--max-far", type=float, default=None, help="Legacy alias for max false alarm rate budget.")
    parser.add_argument(
        "--cache-npz",
        type=str,
        default="checkpoints/corr_v2/val_pristine_preds.npz",
        help="Path to .npz file for caching per-image predictions.",
    )
    parser.add_argument("--force-compute", action="store_true", help="Force recomputation even if cache exists.")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device."
    )
    return parser.parse_args(args)


def fast_slice_sweep(
    f_mask: np.ndarray,
    c_mask: np.ndarray,
    p_mask: np.ndarray,
    val_image_prob: np.ndarray,
    val_forged_max_overlap_area: np.ndarray,
    val_clean_max_area: np.ndarray,
    pristine_image_prob: np.ndarray,
    pristine_max_area: np.ndarray,
    candidate_s: List[float],
    candidate_thresholds: List[float],
    candidate_areas: List[int],
    budget: float,
) -> Dict[str, Any]:
    """Execute grid sweep on slice using precalculated numpy arrays."""
    f_idx = np.where(f_mask)[0]
    c_idx = np.where(c_mask)[0]
    p_idx = np.where(p_mask)[0]

    n_f = len(f_idx)
    n_c = len(c_idx)
    n_p = len(p_idx)

    sub_f_img_prob = val_image_prob[f_idx]
    sub_c_img_prob = val_image_prob[c_idx]
    sub_p_img_prob = pristine_image_prob[p_idx]

    sub_f_overlap = val_forged_max_overlap_area[f_idx]
    sub_c_max_area = val_clean_max_area[c_idx]
    sub_p_max_area = pristine_max_area[p_idx]

    best_s = candidate_s[0]
    best_t = candidate_thresholds[0]
    best_a = candidate_areas[0]
    best_rec = 0.0
    best_c_far = 1.0
    best_p_far = 1.0
    budget_met = False

    lowest_max_far = 2.0
    fallback_s = best_s
    fallback_t = best_t
    fallback_a = best_a
    fallback_rec = 0.0
    fallback_c_far = 1.0
    fallback_p_far = 1.0

    for s in candidate_s:
        if s <= 0.0:
            f_img_ok = np.ones(n_f, dtype=bool)
            c_img_ok = np.ones(n_c, dtype=bool)
            p_img_ok = np.ones(n_p, dtype=bool)
        else:
            f_img_ok = (sub_f_img_prob >= s) & (~np.isnan(sub_f_img_prob))
            c_img_ok = (sub_c_img_prob >= s) & (~np.isnan(sub_c_img_prob))
            p_img_ok = (sub_p_img_prob >= s) & (~np.isnan(sub_p_img_prob))

        for t_idx, tau in enumerate(candidate_thresholds):
            f_t_areas = sub_f_overlap[:, t_idx]
            c_t_areas = sub_c_max_area[:, t_idx]
            p_t_areas = sub_p_max_area[:, t_idx]

            for area in candidate_areas:
                if area == 0:
                    f_hits = int(np.sum(f_img_ok & (f_t_areas > 0))) if n_f > 0 else 0
                    c_flags = int(np.sum(c_img_ok & (c_t_areas > 0))) if n_c > 0 else 0
                    p_flags = int(np.sum(p_img_ok & (p_t_areas > 0))) if n_p > 0 else 0
                else:
                    f_hits = int(np.sum(f_img_ok & (f_t_areas >= area))) if n_f > 0 else 0
                    c_flags = int(np.sum(c_img_ok & (c_t_areas >= area))) if n_c > 0 else 0
                    p_flags = int(np.sum(p_img_ok & (p_t_areas >= area))) if n_p > 0 else 0

                rec = f_hits / n_f if n_f > 0 else 0.0
                c_far = c_flags / n_c if n_c > 0 else 0.0
                p_far = p_flags / n_p if n_p > 0 else 0.0

                meets = (c_far <= budget) and (p_far <= budget) and (rec > 1e-5)
                cur_max_far = max(c_far, p_far)

                if meets:
                    if (not budget_met) or (rec > best_rec + 1e-5):
                        budget_met = True
                        best_s = s
                        best_t = tau
                        best_a = area
                        best_rec = rec
                        best_c_far = c_far
                        best_p_far = p_far
                    elif abs(rec - best_rec) <= 1e-5:
                        if (c_far + p_far) < (best_c_far + best_p_far):
                            best_s = s
                            best_t = tau
                            best_a = area
                            best_c_far = c_far
                            best_p_far = p_far
                else:
                    if rec > 1e-5:
                        if cur_max_far < lowest_max_far - 1e-5:
                            lowest_max_far = cur_max_far
                            fallback_s = s
                            fallback_t = tau
                            fallback_a = area
                            fallback_rec = rec
                            fallback_c_far = c_far
                            fallback_p_far = p_far
                        elif abs(cur_max_far - lowest_max_far) <= 1e-5:
                            if (rec > fallback_rec + 1e-5) or (
                                abs(rec - fallback_rec) <= 1e-5
                                and (c_far + p_far) < (fallback_c_far + fallback_p_far)
                            ):
                                fallback_s = s
                                fallback_t = tau
                                fallback_a = area
                                fallback_rec = rec
                                fallback_c_far = c_far
                                fallback_p_far = p_far

    if not budget_met:
        best_s = fallback_s
        best_t = fallback_t
        best_a = fallback_a
        best_rec = fallback_rec
        best_c_far = fallback_c_far
        best_p_far = fallback_p_far

    f_hits_final = int(round(best_rec * n_f))
    c_flags_final = int(round(best_c_far * n_c))
    p_flags_final = int(round(best_p_far * n_p))

    return {
        "meets_budget": budget_met,
        "best_image_threshold": float(best_s),
        "best_threshold": float(best_t),
        "best_min_area": int(best_a),
        "best_recall": float(best_rec),
        "best_synth_far": float(best_c_far),
        "best_pristine_far": float(best_p_far),
        "n_forged": n_f,
        "n_clean": n_c,
        "n_pristine": n_p,
        "forged_hits": f_hits_final,
        "clean_flags": c_flags_final,
        "pristine_flags": p_flags_final,
        "recall_ci": wilson_ci(f_hits_final, n_f),
        "synth_far_ci": wilson_ci(c_flags_final, n_c),
        "pristine_far_ci": wilson_ci(p_flags_final, n_p),
    }


def verify_cache_provenance(
    cache: Any,
    ckpt_path: Path,
    val_manifest: Path,
    pristine_zip: Path,
    val_subset: Optional[int],
    pristine_subset: Optional[int],
    model_type: Optional[str],
    has_image_head: bool,
    candidate_thresholds: List[float],
    candidate_areas: List[int],
) -> Tuple[bool, str]:
    """Verify that cached per-image outputs match current checkpoint, inputs, and options."""
    req_keys = [
        "ckpt_name",
        "ckpt_size",
        "ckpt_mtime",
        "val_manifest_size",
        "val_manifest_mtime",
        "pristine_zip_size",
        "pristine_zip_mtime",
        "val_subset",
        "pristine_subset",
        "has_image_head",
    ]
    for k in req_keys:
        if k not in cache:
            return False, f"Missing provenance key '{k}' in cache (legacy or incomplete cache)."

    # Checkpoint verification
    if ckpt_path.exists():
        cur_size = ckpt_path.stat().st_size
        cached_size = int(cache["ckpt_size"])
        if cur_size != cached_size:
            return False, f"Checkpoint file size mismatch ({cur_size} bytes vs cached {cached_size} bytes)."
        cur_mtime = ckpt_path.stat().st_mtime
        cached_mtime = float(cache["ckpt_mtime"])
        if cur_mtime > cached_mtime + 1.0:
            return False, f"Checkpoint was modified after cache was generated (mtime {cur_mtime} > {cached_mtime})."
        cached_name = str(cache["ckpt_name"])
        if cached_name and cached_name != ckpt_path.name:
            return False, f"Checkpoint name mismatch ({ckpt_path.name} vs cached {cached_name})."
    else:
        return False, f"Checkpoint {ckpt_path} not found on disk."

    # Validation manifest verification
    if val_manifest.exists():
        cur_size = val_manifest.stat().st_size
        cached_size = int(cache["val_manifest_size"])
        if cur_size != cached_size:
            return False, f"Validation manifest size mismatch ({cur_size} vs cached {cached_size})."
        cur_mtime = val_manifest.stat().st_mtime
        cached_mtime = float(cache["val_manifest_mtime"])
        if cur_mtime > cached_mtime + 1.0:
            return False, "Validation manifest was modified after cache generation."
    else:
        return False, f"Validation manifest {val_manifest} not found on disk."

    # Pristine archive verification
    if pristine_zip.exists():
        cur_size = pristine_zip.stat().st_size
        cached_size = int(cache["pristine_zip_size"])
        if cur_size != cached_size:
            return False, f"Pristine zip size mismatch ({cur_size} vs cached {cached_size})."
        cur_mtime = pristine_zip.stat().st_mtime
        cached_mtime = float(cache["pristine_zip_mtime"])
        if cur_mtime > cached_mtime + 1.0:
            return False, "Pristine zip was modified after cache generation."
    else:
        return False, f"Pristine zip {pristine_zip} not found on disk."

    # Subset options verification
    cached_val_sub = int(cache["val_subset"])
    req_val_sub = val_subset if val_subset is not None else -1
    if cached_val_sub != req_val_sub:
        return False, f"Validation subset mismatch (requested {req_val_sub} vs cached {cached_val_sub})."

    cached_pris_sub = int(cache["pristine_subset"])
    req_pris_sub = pristine_subset if pristine_subset is not None else -1
    if cached_pris_sub != req_pris_sub:
        return False, f"Pristine subset mismatch (requested {req_pris_sub} vs cached {cached_pris_sub})."

    # Architecture and image head verification
    cached_head = bool(cache["has_image_head"])
    if cached_head != has_image_head:
        return False, f"Image head mismatch (requested {has_image_head} vs cached {cached_head})."

    if model_type is not None:
        cached_model = str(cache.get("model_type", ""))
        if cached_model and cached_model != model_type:
            return False, f"Model architecture mismatch ({model_type} vs cached {cached_model})."

    # Candidate grids verification
    cached_taus = [round(float(t), 4) for t in cache["candidate_thresholds"]]
    req_taus = [round(float(t), 4) for t in candidate_thresholds]
    if cached_taus != req_taus:
        return False, "Candidate thresholds mismatch."

    cached_areas = [int(a) for a in cache["candidate_areas"]]
    req_areas = [int(a) for a in candidate_areas]
    if cached_areas != req_areas:
        return False, "Candidate areas mismatch."

    return True, "Cache provenance verified."


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Multi-threading on CPU
    if device.type == "cpu":
        num_cores = os.cpu_count() or 4
        torch.set_num_threads(num_cores)

    candidate_thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    candidate_areas = [0, 25, 32, 50, 64, 100, 150, 200, 300, 400, 500, 800, 1200, 1600]

    cache_path = Path(args.cache_npz)
    loaded_from_cache = False

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    val_manifest = Path(args.val_manifest)
    if not val_manifest.exists():
        alt_manifest = Path(args.data_dir) / "synthetic" / "val_manifest.csv"
        if alt_manifest.exists():
            val_manifest = alt_manifest
        else:
            raise FileNotFoundError(f"Missing val manifest: {val_manifest}")

    pristine_zip = Path(args.pristine_zip)
    if not pristine_zip.exists():
        raise FileNotFoundError(f"Missing pristine zip: {pristine_zip}")

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

    if cache_path.exists() and not args.force_compute:
        print(f"[INFO] Inspecting prediction cache at {cache_path}...")
        cache = np.load(cache_path, allow_pickle=True)
        valid, reason = verify_cache_provenance(
            cache=cache,
            ckpt_path=ckpt_path,
            val_manifest=val_manifest,
            pristine_zip=pristine_zip,
            val_subset=args.val_subset,
            pristine_subset=args.pristine_subset,
            model_type=model_type,
            has_image_head=has_image_head,
            candidate_thresholds=candidate_thresholds,
            candidate_areas=candidate_areas,
        )
        if valid:
            print(f"[SUCCESS] Cache provenance verified for {ckpt_path.name} (inputs match). Loading per-image outputs...")
            val_is_forged = cache["val_is_forged"]
            val_class = cache["val_class"]
            val_image_prob = cache["val_image_prob"]
            val_max_prob = cache["val_max_prob"]
            val_forged_max_overlap_area = cache["val_forged_max_overlap_area"]
            val_clean_max_area = cache["val_clean_max_area"]
            pristine_class = cache["pristine_class"]
            pristine_image_prob = cache["pristine_image_prob"]
            pristine_max_prob = cache["pristine_max_prob"]
            pristine_max_area = cache["pristine_max_area"]
            candidate_thresholds = [float(t) for t in cache["candidate_thresholds"]]
            candidate_areas = [int(a) for a in cache["candidate_areas"]]
            has_image_head = bool(cache.get("has_image_head", True))
            model_type = str(cache.get("model_type", "corr"))
            loaded_from_cache = True
            print(f"[SUCCESS] Loaded {len(val_is_forged)} val samples and {len(pristine_class)} pristine panels from cache.")
        else:
            print(f"[WARNING] Cache mismatch: {reason}. Recomputing per-image predictions...")

    if not loaded_from_cache:
        # Run inference and populate cache
        print(f"[INFO] Running inference with {model_type} (image_head={has_image_head}) from {ckpt_path.name}")
        if model_type == "unet":
            model = ResNetUNet(pretrained=False)
        else:
            model = ResNetCorrNet(pretrained=False, image_head=has_image_head)

        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

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

        # 1. Validation split inference
        val_is_forged_list: List[bool] = []
        val_class_list: List[str] = []
        val_image_prob_list: List[float] = []
        val_max_prob_list: List[float] = []
        val_forged_overlap_list: List[List[int]] = []
        val_clean_area_list: List[List[int]] = []

        t0_val = time.time()
        print("[INFO] Computing validation probability maps and connected components...")
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
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
                    img_prob = np.nan

                probs = torch.sigmoid(logits)

                vm_bool = (valid_mask > 0.5)[0, 0].cpu().numpy()
                probs_np = probs[0, 0].cpu().numpy()
                mask_gt_np = (mask_gt[0, 0].cpu().numpy() > 0.5).astype(np.uint8)

                # Crop back to unpadded dimensions
                ys, xs = np.where(vm_bool)
                if len(ys) > 0 and len(xs) > 0:
                    top_y, bot_y = int(ys.min()), int(ys.max() + 1)
                    left_x, right_x = int(xs.min()), int(xs.max() + 1)
                else:
                    top_y, bot_y = 0, probs_np.shape[0]
                    left_x, right_x = 0, probs_np.shape[1]

                probs_cropped = probs_np[top_y:bot_y, left_x:right_x]
                mask_gt_cropped = mask_gt_np[top_y:bot_y, left_x:right_x]
                max_p = float(probs_cropped.max()) if probs_cropped.size > 0 else 0.0

                f_overlap_row = []
                c_area_row = []

                for tau in candidate_thresholds:
                    pred_bin = (probs_cropped >= tau).astype(np.uint8)
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_bin, connectivity=8)

                    if is_forged:
                        max_ov = 0
                        if num_labels > 1:
                            for lab_idx in range(1, num_labels):
                                if np.sum((labels == lab_idx) & (mask_gt_cropped > 0)) > 0:
                                    comp_a = int(stats[lab_idx, cv2.CC_STAT_AREA])
                                    if comp_a > max_ov:
                                        max_ov = comp_a
                        elif num_labels == 1 and np.sum((pred_bin > 0) & (mask_gt_cropped > 0)) > 0:
                            max_ov = int(np.sum((pred_bin > 0) & (mask_gt_cropped > 0)))
                        f_overlap_row.append(max_ov)
                        c_area_row.append(0)
                    else:
                        max_ca = int(np.max(stats[1:, cv2.CC_STAT_AREA])) if num_labels > 1 else 0
                        c_area_row.append(max_ca)
                        f_overlap_row.append(0)

                val_is_forged_list.append(is_forged)
                val_class_list.append(cls_name)
                val_image_prob_list.append(img_prob)
                val_max_prob_list.append(max_p)
                val_forged_overlap_list.append(f_overlap_row)
                val_clean_area_list.append(c_area_row)

                if (idx + 1) % 250 == 0 or (idx + 1) == len(val_dataset):
                    print(f"  Processed {idx + 1}/{len(val_dataset)} validation images ({time.time() - t0_val:.1f}s)")

        # 2. Pristine held-out inference
        pristine_panels = load_all_pristine_panels(args.pristine_zip, max_count=args.pristine_subset)
        pristine_class_list: List[str] = []
        pristine_image_prob_list: List[float] = []
        pristine_max_prob_list: List[float] = []
        pristine_max_area_list: List[List[int]] = []

        if len(pristine_panels) > 0:
            t0_p = time.time()
            print(f"[INFO] Computing pristine probability maps on {len(pristine_panels)} panels...")
            with torch.no_grad():
                for idx, (p_img, _, p_cls) in enumerate(pristine_panels):
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
                        p_img_prob = np.nan

                    probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
                    probs_cropped = probs[top : top + h, left : left + w]
                    p_max_p = float(probs_cropped.max()) if probs_cropped.size > 0 else 0.0

                    p_area_row = []
                    for tau in candidate_thresholds:
                        pred_bin = (probs_cropped >= tau).astype(np.uint8)
                        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_bin, connectivity=8)
                        max_ca = int(np.max(stats[1:, cv2.CC_STAT_AREA])) if num_labels > 1 else 0
                        p_area_row.append(max_ca)

                    pristine_class_list.append(p_cls)
                    pristine_image_prob_list.append(p_img_prob)
                    pristine_max_prob_list.append(p_max_p)
                    pristine_max_area_list.append(p_area_row)

                    if (idx + 1) % 500 == 0 or (idx + 1) == len(pristine_panels):
                        print(f"  Processed {idx + 1}/{len(pristine_panels)} pristine panels ({time.time() - t0_p:.1f}s)")

        # Convert to numpy arrays
        val_is_forged = np.array(val_is_forged_list, dtype=bool)
        val_class = np.array(val_class_list, dtype=object)
        val_image_prob = np.array(val_image_prob_list, dtype=np.float32)
        val_max_prob = np.array(val_max_prob_list, dtype=np.float32)
        val_forged_max_overlap_area = np.array(val_forged_overlap_list, dtype=np.int32)
        val_clean_max_area = np.array(val_clean_area_list, dtype=np.int32)

        pristine_class = np.array(pristine_class_list, dtype=object)
        pristine_image_prob = np.array(pristine_image_prob_list, dtype=np.float32)
        pristine_max_prob = np.array(pristine_max_prob_list, dtype=np.float32)
        pristine_max_area = np.array(pristine_max_area_list, dtype=np.int32)

        # Save to cache file
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Caching per-image predictions to {cache_path}...")
        np.savez_compressed(
            cache_path,
            val_is_forged=val_is_forged,
            val_class=val_class,
            val_image_prob=val_image_prob,
            val_max_prob=val_max_prob,
            val_forged_max_overlap_area=val_forged_max_overlap_area,
            val_clean_max_area=val_clean_max_area,
            pristine_class=pristine_class,
            pristine_image_prob=pristine_image_prob,
            pristine_max_prob=pristine_max_prob,
            pristine_max_area=pristine_max_area,
            candidate_thresholds=np.array(candidate_thresholds, dtype=np.float32),
            candidate_areas=np.array(candidate_areas, dtype=np.int32),
            has_image_head=has_image_head,
            model_type=model_type,
            # Provenance metadata
            ckpt_name=str(ckpt_path.name),
            ckpt_size=ckpt_path.stat().st_size if ckpt_path.exists() else 0,
            ckpt_mtime=ckpt_path.stat().st_mtime if ckpt_path.exists() else 0.0,
            val_manifest_name=str(val_manifest.name),
            val_manifest_size=val_manifest.stat().st_size if val_manifest.exists() else 0,
            val_manifest_mtime=val_manifest.stat().st_mtime if val_manifest.exists() else 0.0,
            pristine_zip_name=str(pristine_zip.name),
            pristine_zip_size=pristine_zip.stat().st_size if pristine_zip.exists() else 0,
            pristine_zip_mtime=pristine_zip.stat().st_mtime if pristine_zip.exists() else 0.0,
            val_subset=args.val_subset if args.val_subset is not None else -1,
            pristine_subset=args.pristine_subset if args.pristine_subset is not None else -1,
        )
        print(f"[SUCCESS] Prediction cache saved ({os.path.getsize(cache_path) / 1024:.1f} KB).")

    # Grid search candidate values
    candidate_s = (
        [0.0, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99] if has_image_head else [0.0]
    )

    budgets = sorted(list(set(args.budgets + ([args.max_far] if args.max_far is not None else []))))
    print(f"\n[INFO] Sweeping grid across FAR budgets: {[f'{b*100:.2f}%' for b in budgets]}")

    # Masks for scopes
    f_all = val_is_forged
    c_all = ~val_is_forged
    p_all = np.ones(len(pristine_class), dtype=bool)

    f_blot = val_is_forged & (val_class == "Blot/Gel")
    c_blot = (~val_is_forged) & (val_class == "Blot/Gel")
    p_blot = pristine_class == "Blot/Gel"

    f_mic = val_is_forged & (val_class == "Microscopy")
    c_mic = (~val_is_forged) & (val_class == "Microscopy")
    p_mic = pristine_class == "Microscopy"

    print(f"[INFO] Counts Overall:    {np.sum(f_all)} forged, {np.sum(c_all)} clean twins, {np.sum(p_all)} pristine.")
    print(f"[INFO] Counts Blot/Gel:   {np.sum(f_blot)} forged, {np.sum(c_blot)} clean twins, {np.sum(p_blot)} pristine.")
    print(f"[INFO] Counts Microscopy: {np.sum(f_mic)} forged, {np.sum(c_mic)} clean twins, {np.sum(p_mic)} pristine.")

    results_by_budget: Dict[float, Dict[str, Any]] = {}
    table_rows = []

    for b in budgets:
        sw_ov = fast_slice_sweep(
            f_all,
            c_all,
            p_all,
            val_image_prob,
            val_forged_max_overlap_area,
            val_clean_max_area,
            pristine_image_prob,
            pristine_max_area,
            candidate_s,
            candidate_thresholds,
            candidate_areas,
            budget=b,
        )

        sw_bl = fast_slice_sweep(
            f_blot,
            c_blot,
            p_blot,
            val_image_prob,
            val_forged_max_overlap_area,
            val_clean_max_area,
            pristine_image_prob,
            pristine_max_area,
            candidate_s,
            candidate_thresholds,
            candidate_areas,
            budget=b,
        )

        sw_mi = fast_slice_sweep(
            f_mic,
            c_mic,
            p_mic,
            val_image_prob,
            val_forged_max_overlap_area,
            val_clean_max_area,
            pristine_image_prob,
            pristine_max_area,
            candidate_s,
            candidate_thresholds,
            candidate_areas,
            budget=b,
        )

        results_by_budget[b] = {
            "overall": sw_ov,
            "blot_gel": sw_bl,
            "microscopy": sw_mi,
        }

        table_rows.append({"budget": b, "scope": "Overall", **sw_ov})
        table_rows.append({"budget": b, "scope": "Blot/Gel", **sw_bl})
        table_rows.append({"budget": b, "scope": "Microscopy", **sw_mi})

    # Print unified summary table
    print("\n" + "=" * 140)
    print("PHASE 4 CNN OPERATING THRESHOLDS SUMMARY (STEP 1 WRAP-UP)")
    print(f"Model: {model_type.upper()} (image_head={has_image_head}) | Checkpoint: {Path(args.checkpoint).name}")
    print("Note: Image class (Blot/Gel vs Microscopy) is assumed known from metadata or upstream classifier.")
    print("=" * 140)
    header = (
        f"{'Budget':<7} | {'Scope':<11} | {'Meets':<6} | {'Recall [95% CI]':<22} | "
        f"{'s':<5} | {'tau':<5} | {'Area':<6} | {'Synth Clean FAR [CI]':<26} | {'Real Pristine FAR [CI]':<26}"
    )
    print(header)
    print("-" * 140)

    cur_b = None
    for r in table_rows:
        b_val = r["budget"]
        if cur_b is not None and b_val != cur_b:
            print("-" * 140)
        cur_b = b_val

        b_str = f"{b_val*100:.1f}%"
        meets_str = "True" if r["meets_budget"] else "False"
        rec_str = f"{r['best_recall']*100:.1f}% [{r['recall_ci'][0]*100:.1f}-{r['recall_ci'][1]*100:.1f}]"
        s_str = f"{r['best_image_threshold']:.2f}"
        tau_str = f"{r['best_threshold']:.2f}"
        area_str = f"{r['best_min_area']} px"
        sf_str = f"{r['best_synth_far']*100:.2f}% [{r['synth_far_ci'][0]*100:.2f}-{r['synth_far_ci'][1]*100:.2f}]"
        pf_str = f"{r['best_pristine_far']*100:.2f}% [{r['pristine_far_ci'][0]*100:.2f}-{r['pristine_far_ci'][1]*100:.2f}]"

        print(
            f"{b_str:<7} | {r['scope']:<11} | {meets_str:<6} | {rec_str:<22} | "
            f"{s_str:<5} | {tau_str:<5} | {area_str:<6} | {sf_str:<26} | {pf_str:<26}"
        )

    print("=" * 140 + "\n")

    # Write configs/cnn_threshold.yaml
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
        f'checkpoint_file: "{Path(args.checkpoint).name}"',
        f"primary_budget: {primary_budget}",
        "",
        "# Root-level keys represent overall operating point at 1.0% budget for backward compatibility",
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
        b_pct_tag = f"budget_{str(round(b*100, 2)).replace('.', '_')}pct"
        b_ov = results_by_budget[b]["overall"]
        b_bl = results_by_budget[b]["blot_gel"]
        b_mi = results_by_budget[b]["microscopy"]
        yaml_lines.extend([
            f"  {b_pct_tag}:",
            f"    target_far: {b}",
            "    overall:",
            f"      image_threshold: {b_ov['best_image_threshold']}",
            f"      pixel_threshold: {b_ov['best_threshold']}",
            f"      min_area_pixels: {b_ov['best_min_area']}",
            f"      validation_recall: {round(float(b_ov['best_recall']), 4)}",
            f"      validation_recall_ci: [{round(b_ov['recall_ci'][0], 4)}, {round(b_ov['recall_ci'][1], 4)}]",
            f"      synthetic_val_clean_far: {round(float(b_ov['best_synth_far']), 6)}",
            f"      synthetic_val_clean_far_ci: [{round(b_ov['synth_far_ci'][0], 6)}, {round(b_ov['synth_far_ci'][1], 6)}]",
            f"      real_pristine_heldout_far: {round(float(b_ov['best_pristine_far']), 6)}",
            f"      real_pristine_heldout_far_ci: [{round(b_ov['pristine_far_ci'][0], 6)}, {round(b_ov['pristine_far_ci'][1], 6)}]",
            f"      budget_met: {str(b_ov['meets_budget']).lower()}",
            "    blot_gel:",
            f"      image_threshold: {b_bl['best_image_threshold']}",
            f"      pixel_threshold: {b_bl['best_threshold']}",
            f"      min_area_pixels: {b_bl['best_min_area']}",
            f"      validation_recall: {round(float(b_bl['best_recall']), 4)}",
            f"      validation_recall_ci: [{round(b_bl['recall_ci'][0], 4)}, {round(b_bl['recall_ci'][1], 4)}]",
            f"      synthetic_val_clean_far: {round(float(b_bl['best_synth_far']), 6)}",
            f"      synthetic_val_clean_far_ci: [{round(b_bl['synth_far_ci'][0], 6)}, {round(b_bl['synth_far_ci'][1], 6)}]",
            f"      real_pristine_heldout_far: {round(float(b_bl['best_pristine_far']), 6)}",
            f"      real_pristine_heldout_far_ci: [{round(b_bl['pristine_far_ci'][0], 6)}, {round(b_bl['pristine_far_ci'][1], 6)}]",
            f"      budget_met: {str(b_bl['meets_budget']).lower()}",
            "    microscopy:",
            f"      image_threshold: {b_mi['best_image_threshold']}",
            f"      pixel_threshold: {b_mi['best_threshold']}",
            f"      min_area_pixels: {b_mi['best_min_area']}",
            f"      validation_recall: {round(float(b_mi['best_recall']), 4)}",
            f"      validation_recall_ci: [{round(b_mi['recall_ci'][0], 4)}, {round(b_mi['recall_ci'][1], 4)}]",
            f"      synthetic_val_clean_far: {round(float(b_mi['best_synth_far']), 6)}",
            f"      synthetic_val_clean_far_ci: [{round(b_mi['synth_far_ci'][0], 6)}, {round(b_mi['synth_far_ci'][1], 6)}]",
            f"      real_pristine_heldout_far: {round(float(b_mi['best_pristine_far']), 6)}",
            f"      real_pristine_heldout_far_ci: [{round(b_mi['pristine_far_ci'][0], 6)}, {round(b_mi['pristine_far_ci'][1], 6)}]",
            f"      budget_met: {str(b_mi['meets_budget']).lower()}",
        ])

    with open(out_yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines) + "\n")

    print(f"[INFO] Wrote locked threshold configuration to {out_yaml_path}")


if __name__ == "__main__":
    main()
