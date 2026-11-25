#!/usr/bin/env python3
"""
eval_phase5.py: Evaluate a trained Phase 5 CNN checkpoint against Targets 1, 2, and 3.

Why this exists:
Evaluates the newly trained Phase 5 model and its selected operating threshold on:
1. Real Pristine Dev Set (4,039 panels from 177 held-out documents) -> Target 1.
2. Synthetic v3 Test Set (300 pristine, 300 unrelated_paste, 300 forged) -> Targets 2 & 3.

Never touches the locked real BioFors IDD test split.
"""
import argparse
import csv
import json
import math
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from detector.cnn import load_checkpoint
from detector.cnn.infer import load_cnn_model, load_cnn_thresholds, predict_mask


def wilson_ci(k: int, n: int) -> Tuple[float, float, float]:
    """Compute proportion and 95% Wilson score confidence interval."""
    if n == 0:
        return 0.0, 0.0, 0.0
    z = 1.95996
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    margin = (z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))) / denom
    return p, max(0.0, center - margin), min(1.0, center + margin)


def parse_args(args: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Evaluate Phase 5 CNN checkpoint against pre-registered targets.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained Phase 5 checkpoint (.pt).")
    parser.add_argument("--config", type=str, default=None, help="Path to calibrated threshold YAML (optional).")
    parser.add_argument("--budget", type=float, default=0.01, help="Operating FAR budget to evaluate (default 0.01).")
    parser.add_argument("--test-dir", type=str, default="data/synthetic_v3/test", help="Path to synthetic_v3/test split.")
    parser.add_argument("--pristine-zip", type=str, default="data/pristine_heldout.zip", help="Path to pristine_heldout.zip.")
    parser.add_argument("--class-map", type=str, default="configs/v3_panel_classes.json", help="Optional JSON class map.")
    parser.add_argument("--out-dir", type=str, default="results/phase5", help="Directory to save CSV and summary JSON.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device.")
    return parser.parse_args(args)


def get_thresholds_for_class(
    cls_name: str,
    config_path: Optional[Path],
    ckpt_eval: Dict[str, Any],
    budget: float,
) -> Tuple[float, float, int]:
    """Resolve (image_threshold, pixel_threshold, min_area) from config YAML or checkpoint eval."""
    if config_path is not None and config_path.exists():
        cfg = load_cnn_thresholds(config_path=config_path, image_class=cls_name, budget=budget)
        return float(cfg["image_threshold"]), float(cfg["pixel_threshold"]), int(cfg["min_area_pixels"])

    img_t = float(ckpt_eval.get("best_image_threshold", 0.5))
    pix_t = float(ckpt_eval.get("best_threshold", 0.5))
    min_a = int(ckpt_eval.get("best_min_area", 50))
    return img_t, pix_t, min_a


def evaluate_real_pristine_dev(
    model: torch.nn.Module,
    pristine_zip: Path,
    config_path: Optional[Path],
    ckpt_eval: Dict[str, Any],
    budget: float,
    device: str,
) -> Dict[str, Any]:
    """Evaluate false alarm rate on the 4,039 real pristine held-out dev panels."""
    if not pristine_zip.exists():
        raise FileNotFoundError(f"Pristine dev archive not found: {pristine_zip}")

    val_s = float(ckpt_eval.get("best_image_threshold", 0.0))
    val_tau = float(ckpt_eval.get("best_threshold", 0.5))
    val_area = int(ckpt_eval.get("best_min_area", 50))

    total = 0
    fa_cal = 0
    fa_val_only = 0
    by_class: Dict[str, Dict[str, int]] = {
        "Blot/Gel": {"total": 0, "fa_cal": 0, "fa_val": 0},
        "Microscopy": {"total": 0, "fa_cal": 0, "fa_val": 0},
    }

    t0 = time.time()
    with zipfile.ZipFile(pristine_zip, "r") as zf:
        manifest_names = [n for n in zf.namelist() if n.endswith("pristine_manifest.csv")]
        if not manifest_names:
            raise FileNotFoundError(f"pristine_manifest.csv not found in {pristine_zip}")
        manifest_bytes = zf.read(manifest_names[0]).decode("utf-8")
        rows = list(csv.DictReader(manifest_bytes.splitlines()))

        for idx, r in enumerate(rows):
            arc_name = f"pristine_heldout/{r['path']}"
            if arc_name not in zf.namelist():
                continue
            arr = np.frombuffer(zf.read(arc_name), dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            cls_name = r.get("class", "Blot/Gel")

            cal_s, cal_tau, cal_area = get_thresholds_for_class(cls_name, config_path, ckpt_eval, budget)

            # Single forward pass via calibrated threshold, plus check val-only threshold on prob_map
            res_cal = predict_mask(
                model=model,
                image=img_rgb,
                threshold=cal_tau,
                min_area=cal_area,
                image_threshold=cal_s,
                device=device,
            )
            det_cal = bool(res_cal["is_detected"])

            # Check val-only operating point directly using the returned prob_map and image_prob
            prob_map = res_cal["prob_map"]
            img_prob = res_cal["image_prob"]
            img_ok_val = (val_s <= 0.0) or (img_prob is not None and float(img_prob) >= val_s)
            det_val = False
            if img_ok_val:
                pred_bin_val = (prob_map >= val_tau).astype(np.uint8)
                if val_area <= 0:
                    det_val = bool(np.sum(pred_bin_val) > 0)
                else:
                    n_lbl, _, stats, _ = cv2.connectedComponentsWithStats(pred_bin_val, connectivity=8)
                    det_val = bool(n_lbl > 1 and np.any(stats[1:, cv2.CC_STAT_AREA] >= val_area))

            total += 1
            if det_cal:
                fa_cal += 1
            if det_val:
                fa_val_only += 1
            if cls_name in by_class:
                by_class[cls_name]["total"] += 1
                by_class[cls_name]["fa_cal"] += int(det_cal)
                by_class[cls_name]["fa_val"] += int(det_val)

            if (idx + 1) % 1000 == 0 or (idx + 1) == len(rows):
                print(f"[Real Dev] {idx + 1}/{len(rows)} panels ({time.time() - t0:.1f}s)")

    far_cal, cal_lo, cal_hi = wilson_ci(fa_cal, total)
    far_val, val_lo, val_hi = wilson_ci(fa_val_only, total)
    return {
        "total": total,
        "fa_calibrated": fa_cal,
        "far_calibrated": far_cal,
        "far_calibrated_ci": [cal_lo, cal_hi],
        "fa_val_only": fa_val_only,
        "far_val_only": far_val,
        "far_val_only_ci": [val_lo, val_hi],
        "val_only_thresholds": {"s": val_s, "tau": val_tau, "min_area": val_area},
        "by_class": by_class,
        "target1_met": bool(far_cal <= 0.02),
    }


def evaluate_synthetic_test(
    model: torch.nn.Module,
    test_dir: Path,
    config_path: Optional[Path],
    ckpt_eval: Dict[str, Any],
    budget: float,
    class_map: Dict[str, str],
    device: str,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Evaluate seam parity (Target 2) and forged recall (Target 3) on synthetic_v3/test."""
    meta_path = test_dir / "meta.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"Test metadata not found: {meta_path}")

    rows = [json.loads(line) for line in open(meta_path, "r", encoding="utf-8") if line.strip()]
    records: List[Dict[str, Any]] = []

    tp_pix = 0
    fp_pix = 0
    fn_pix = 0
    tn_pix = 0

    t0 = time.time()
    for idx, item in enumerate(rows):
        img_id = item["id"]
        kind = item["kind"]
        base_rel = str(item.get("base", ""))
        cls_name = class_map.get(base_rel, "overall")

        img_path = test_dir / "images" / f"{img_id}.png"
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Test image missing: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        cal_s, cal_tau, cal_area = get_thresholds_for_class(cls_name, config_path, ckpt_eval, budget)
        res = predict_mask(
            model=model,
            image=img_rgb,
            threshold=cal_tau,
            min_area=cal_area,
            image_threshold=cal_s,
            device=device,
        )
        is_det = bool(res["is_detected"])
        pred_mask = (res["pred_mask"] > 0).astype(np.uint8) if is_det else np.zeros(img_rgb.shape[:2], dtype=np.uint8)

        if kind == "forged":
            mask_path = test_dir / "mask" / f"{img_id}.png"
            gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            gt_bin = (gt_mask > 127).astype(np.uint8) if gt_mask is not None else np.zeros(img_rgb.shape[:2], dtype=np.uint8)
            tp_pix += int(np.sum((pred_mask == 1) & (gt_bin == 1)))
            fp_pix += int(np.sum((pred_mask == 1) & (gt_bin == 0)))
            fn_pix += int(np.sum((pred_mask == 0) & (gt_bin == 1)))
            tn_pix += int(np.sum((pred_mask == 0) & (gt_bin == 0)))

        records.append({
            "id": img_id,
            "kind": kind,
            "doc": str(item.get("doc", "")),
            "base": base_rel,
            "class": cls_name,
            "is_detected": is_det,
            "image_prob": float(res["image_prob"]) if res["image_prob"] is not None else np.nan,
            "max_prob": float(res["max_prob"]),
            "mask_pixels": int(np.sum(pred_mask)),
        })

        if (idx + 1) % 300 == 0 or (idx + 1) == len(rows):
            print(f"[Synth Test] {idx + 1}/{len(rows)} panels ({time.time() - t0:.1f}s)")

    df = pd.DataFrame(records)
    pristine_df = df[df["kind"] == "pristine"]
    unrelated_df = df[df["kind"] == "unrelated_paste"]
    forged_df = df[df["kind"] == "forged"]

    k_p, n_p = int(pristine_df["is_detected"].sum()), len(pristine_df)
    k_u, n_u = int(unrelated_df["is_detected"].sum()), len(unrelated_df)
    k_f, n_f = int(forged_df["is_detected"].sum()), len(forged_df)

    fpr_p, p_lo, p_hi = wilson_ci(k_p, n_p)
    fpr_u, u_lo, u_hi = wilson_ci(k_u, n_u)
    rec_f, f_lo, f_hi = wilson_ci(k_f, n_f)

    ratio = (fpr_u / fpr_p) if fpr_p > 0 else (float("inf") if fpr_u > 0 else 1.0)
    denom_f1 = 2 * tp_pix + fp_pix + fn_pix
    pix_f1 = (2 * tp_pix / denom_f1) if denom_f1 > 0 else 0.0
    mcc_den = math.sqrt(float(tp_pix + fp_pix) * float(tp_pix + fn_pix) * float(tn_pix + fp_pix) * float(tn_pix + fn_pix))
    pix_mcc = ((tp_pix * tn_pix - fp_pix * fn_pix) / mcc_den) if mcc_den > 0 else 0.0

    # Target 2: unrelated_paste FPR <= 1.5x pristine FPR AND <= 3.0% absolute
    t2_ratio_ok = (ratio <= 1.5)
    t2_abs_ok = (fpr_u <= 0.03)
    target2_met = bool(t2_ratio_ok and t2_abs_ok)
    target3_met = bool(rec_f >= 0.30)

    summary = {
        "n_pristine": n_p,
        "hits_pristine": k_p,
        "fpr_pristine": fpr_p,
        "fpr_pristine_ci": [p_lo, p_hi],
        "n_unrelated_paste": n_u,
        "hits_unrelated_paste": k_u,
        "fpr_unrelated_paste": fpr_u,
        "fpr_unrelated_paste_ci": [u_lo, u_hi],
        "seam_ratio": ratio,
        "n_forged": n_f,
        "hits_forged": k_f,
        "forged_recall": rec_f,
        "forged_recall_ci": [f_lo, f_hi],
        "pixel_f1": pix_f1,
        "pixel_mcc": pix_mcc,
        "target2_ratio_met": t2_ratio_ok,
        "target2_abs_met": t2_abs_ok,
        "target2_met": target2_met,
        "target3_met": target3_met,
    }
    return summary, df


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    cfg_path = Path(args.config) if args.config else None
    test_dir = Path(args.test_dir)
    pristine_zip = Path(args.pristine_zip)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_map: Dict[str, str] = {}
    cm_path = Path(args.class_map) if args.class_map else None
    if cm_path and cm_path.exists():
        with open(cm_path, "r", encoding="utf-8") as f:
            class_map = json.load(f)

    ckpt = load_checkpoint(ckpt_path, map_location=args.device)
    ckpt_eval = ckpt.get("best_checkpoint_eval", {})
    model = load_cnn_model(checkpoint_path=ckpt_path, device=args.device)

    print("\n================ PHASE 5 TARGET EVALUATION ================")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Threshold  : {cfg_path if cfg_path and cfg_path.exists() else 'best_checkpoint_eval (val-only)'}")
    print(f"Budget     : {args.budget * 100:.1f}%")

    dev_res = evaluate_real_pristine_dev(
        model=model,
        pristine_zip=pristine_zip,
        config_path=cfg_path,
        ckpt_eval=ckpt_eval,
        budget=args.budget,
        device=args.device,
    )
    test_res, df_test = evaluate_synthetic_test(
        model=model,
        test_dir=test_dir,
        config_path=cfg_path,
        ckpt_eval=ckpt_eval,
        budget=args.budget,
        class_map=class_map,
        device=args.device,
    )

    df_test.to_csv(out_dir / "phase5_synthetic_v3_test.csv", index=False)
    report = {"real_pristine_dev": dev_res, "synthetic_v3_test": test_res}
    with open(out_dir / "phase5_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n--- TARGET 1: Real Held-Out Pristine Dev Set (177 Docs) ---")
    print(f"Calibrated FAR : {dev_res['fa_calibrated']}/{dev_res['total']} = {dev_res['far_calibrated']*100:.2f}% "
          f"[{dev_res['far_calibrated_ci'][0]*100:.2f}%, {dev_res['far_calibrated_ci'][1]*100:.2f}%] (Target <= 2.0%)")
    print(f"Val-only FAR   : {dev_res['fa_val_only']}/{dev_res['total']} = {dev_res['far_val_only']*100:.2f}% "
          f"[{dev_res['far_val_only_ci'][0]*100:.2f}%, {dev_res['far_val_only_ci'][1]*100:.2f}%]")
    print(f"Target 1 Met   : {dev_res['target1_met']}")

    print("\n--- TARGET 2 & 3: Synthetic v3 Test Set ---")
    print(f"Pristine FPR        : {test_res['hits_pristine']}/{test_res['n_pristine']} = {test_res['fpr_pristine']*100:.2f}% "
          f"[{test_res['fpr_pristine_ci'][0]*100:.2f}%, {test_res['fpr_pristine_ci'][1]*100:.2f}%]")
    print(f"Unrelated Paste FPR : {test_res['hits_unrelated_paste']}/{test_res['n_unrelated_paste']} = {test_res['fpr_unrelated_paste']*100:.2f}% "
          f"[{test_res['fpr_unrelated_paste_ci'][0]*100:.2f}%, {test_res['fpr_unrelated_paste_ci'][1]*100:.2f}%] (Target <= 3.0%)")
    print(f"Seam Ratio (U/P)    : {test_res['seam_ratio']:.2f}x (Target <= 1.50x)")
    print(f"Forged Recall       : {test_res['hits_forged']}/{test_res['n_forged']} = {test_res['forged_recall']*100:.2f}% "
          f"[{test_res['forged_recall_ci'][0]*100:.2f}%, {test_res['forged_recall_ci'][1]*100:.2f}%] (Target >= 30.0%)")
    print(f"Pixel F1 / MCC      : {test_res['pixel_f1']:.4f} / {test_res['pixel_mcc']:.4f}")
    print(f"Target 2 Met        : {test_res['target2_met']}")
    print(f"Target 3 Met        : {test_res['target3_met']}")
    print("\nNote: Locked real BioFors IDD test split was NOT accessed (exploratory only, run once at end).")


if __name__ == "__main__":
    main()
