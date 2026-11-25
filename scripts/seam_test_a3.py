#!/usr/bin/env python3
"""
seam_test_a3.py: Run seam test on the current CorrNet v2 model using synthetic_v3 test split.

Why this exists:
Task A3 tests whether CorrNet v2 learned boundary seams / generator artifacts rather
than true duplication. An unrelated_paste image has identical boundary seams and blending
as a forged image, but contains no intra-image duplication. If the false positive rate
on unrelated_paste is clearly higher than pristine, the seam shortcut hypothesis is confirmed.

What it does:
1. Loads data/synthetic_v3/test/meta.jsonl (300 pristine, 300 forged, 300 unrelated_paste).
2. Looks up panel class (Blot/Gel vs Microscopy) from data/biofors/annotations/classification.json.
3. Loads checkpoints/corr_v2/last.pt and locked thresholds from configs/cnn_threshold.yaml (budget 0.01).
4. Runs predict_cnn on each of the 900 images using the existing infer.py pipeline.
5. Computes image-level FPR on pristine, FPR on unrelated_paste, and recall on forged.
6. Reports overall and per-class tables with 95% Wilson confidence intervals.
7. Saves per-image results to results/phase5_seam_test_a3.csv.
"""
import json
import math
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
# Simple progress logger instead of tqdm
def progress_iter(iterable, desc="Progress", step=100):
    total = len(iterable)
    for idx, item in enumerate(iterable):
        if idx % step == 0 or idx == total - 1:
            print(f"[{desc}] {idx + 1}/{total} ({(idx + 1)/total*100:.1f}%)")
        yield item

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from detector.cnn.infer import load_cnn_model, predict_cnn


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple:
    if n == 0:
        return 0.0, 0.0, 0.0
    z = 1.95996
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    margin = (z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return p, lower, upper


def main():
    meta_path = REPO_ROOT / "data" / "synthetic_v3" / "test" / "meta.jsonl"
    images_dir = REPO_ROOT / "data" / "synthetic_v3" / "test" / "images"
    ann_path = REPO_ROOT / "data" / "biofors" / "annotations" / "classification.json"
    ckpt_path = REPO_ROOT / "checkpoints" / "corr_v2" / "last.pt"
    cfg_path = REPO_ROOT / "configs" / "cnn_threshold.yaml"
    out_csv = REPO_ROOT / "results" / "phase5_seam_test_a3.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print("=== TASK A3: SEAM TEST ON CURRENT CORRNET V2 ===")
    print(f"Metadata file   : {meta_path}")
    print(f"Checkpoint      : {ckpt_path}")
    print(f"Threshold config: {cfg_path}")

    # Load classification mapping for class lookup
    with open(ann_path, "r") as f:
        classification = json.load(f)

    # Load metadata rows
    rows = []
    with open(meta_path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    print(f"Total samples to evaluate: {len(rows)}")

    # Load CNN model
    device = "cpu"
    print(f"Loading model on {device}...")
    model = load_cnn_model(checkpoint_path=ckpt_path, device=device)

    records = []
    for item in progress_iter(rows, desc="Evaluating synthetic_v3 test", step=100):
        img_id = item["id"]
        kind = item["kind"]
        doc = item["doc"]
        base_fn = Path(item["base"]).name
        img_cls = classification.get(str(doc), {}).get(base_fn, "Blot/Gel")

        img_path = images_dir / f"{img_id}.png"
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Run inference using locked per-class thresholds at 1.0% budget
        res = predict_cnn(
            image=img,
            image_class=img_cls,
            config_path=cfg_path,
            checkpoint_path=ckpt_path,
            budget=0.01,
            device=device,
            model=model,
            return_dict=True,
        )

        is_detected = bool(res["is_detected"])
        img_prob = float(res.get("image_prob") or 0.0)
        max_prob = float(res.get("max_prob") or 0.0)
        mask_pixels = int(np.sum(res["mask"] > 0)) if "mask" in res else 0

        records.append({
            "id": img_id,
            "kind": kind,
            "doc": doc,
            "base": item["base"],
            "class": img_cls,
            "is_detected": is_detected,
            "image_prob": img_prob,
            "max_prob": max_prob,
            "mask_pixels": mask_pixels,
        })

    df_res = pd.DataFrame(records)
    df_res.to_csv(out_csv, index=False)
    print(f"Saved evaluation results to {out_csv}")

    # Compute statistics
    print("\n" + "=" * 80)
    print("TASK A3 SEAM TEST RESULTS SUMMARY")
    print("=" * 80)

    kinds = ["pristine", "unrelated_paste", "forged"]
    class_subsets = [
        ("Overall", df_res),
        ("Blot/Gel", df_res[df_res["class"] == "Blot/Gel"]),
        ("Microscopy", df_res[df_res["class"] == "Microscopy"]),
    ]

    summary_rows = []
    for cls_name, sub_df in class_subsets:
        n_pristine = len(sub_df[sub_df["kind"] == "pristine"])
        n_unrelated = len(sub_df[sub_df["kind"] == "unrelated_paste"])
        n_forged = len(sub_df[sub_df["kind"] == "forged"])

        hits_pristine = int(sub_df[sub_df["kind"] == "pristine"]["is_detected"].sum())
        hits_unrelated = int(sub_df[sub_df["kind"] == "unrelated_paste"]["is_detected"].sum())
        hits_forged = int(sub_df[sub_df["kind"] == "forged"]["is_detected"].sum())

        fpr_p, p_l, p_u = wilson_ci(hits_pristine, n_pristine)
        fpr_u, u_l, u_u = wilson_ci(hits_unrelated, n_unrelated)
        rec_f, f_l, f_u = wilson_ci(hits_forged, n_forged)

        ratio = (fpr_u / fpr_p) if fpr_p > 0 else (float("inf") if fpr_u > 0 else 1.0)

        summary_rows.append({
            "class": cls_name,
            "pristine_hits": f"{hits_pristine}/{n_pristine}",
            "pristine_fpr": f"{fpr_p*100:.2f}% [{p_l*100:.2f}%, {p_u*100:.2f}%]",
            "unrelated_hits": f"{hits_unrelated}/{n_unrelated}",
            "unrelated_fpr": f"{fpr_u*100:.2f}% [{u_l*100:.2f}%, {u_u*100:.2f}%]",
            "ratio": f"{ratio:.2f}x" if ratio != float("inf") else "inf",
            "forged_hits": f"{hits_forged}/{n_forged}",
            "forged_recall": f"{rec_f*100:.2f}% [{f_l*100:.2f}%, {f_u*100:.2f}%]",
        })

    print(f"{'Class':12s} | {'Pristine FPR (Clean)':28s} | {'Unrelated Paste FPR (Clean)':28s} | {'FPR Ratio':10s} | {'Forged Recall':28s}")
    print("-" * 115)
    for r in summary_rows:
        print(f"{r['class']:12s} | {r['pristine_hits']:7s} ({r['pristine_fpr']}) | {r['unrelated_hits']:7s} ({r['unrelated_fpr']}) | {r['ratio']:10s} | {r['forged_hits']:7s} ({r['forged_recall']})")
    print("-" * 115)

    # Hypothesis Verdict
    ov = summary_rows[0]
    hits_p = int(ov["pristine_hits"].split("/")[0])
    hits_u = int(ov["unrelated_hits"].split("/")[0])
    print("\n=== HYPOTHESIS VERDICT ===")
    if hits_u > hits_p * 1.5:
        print(f"CONFIRMED: False positive rate on unrelated_paste ({ov['unrelated_fpr']}) is CLEARLY HIGHER than on pristine ({ov['pristine_fpr']}).")
        print("CorrNet v2 is firing on paste boundary seams / generator artifacts, not true semantic duplication.")
    elif hits_u == hits_p:
        print("REJECTED / EQUAL: False positive rate on unrelated_paste is EQUAL to pristine.")
    else:
        print(f"INCONCLUSIVE / OTHER: Unrelated paste hits = {hits_u}, Pristine hits = {hits_p}.")


if __name__ == "__main__":
    main()
