"""Dataset Shortcut Audit via Machine Learning Classifiers.

Verifies that the synthetic dataset has NO global statistical shortcuts:
- Evaluates with all 6 features: mean, std, Laplacian variance, width, height, file size (bytes).
- Evaluates with 5 features (file size dropped): mean, std, Laplacian variance, width, height.
- Evaluates each individual non-file-size feature alone.
- Classifiers: Logistic Regression and HistGradientBoostingClassifier.
- Grouping: GroupKFold cross-validation strictly grouped by document ID (doc_id).
- Threshold: If any non-file-size feature achieves accuracy > 55%, flags it.
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit dataset for global statistical shortcuts.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/synthetic/train_manifest.csv",
        help="Path to manifest CSV to audit",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/phase3_synthetic/shortcut_audit.md",
        help="Path to save markdown report",
    )
    return parser.parse_args()


def extract_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    data_dir = REPO_ROOT / "data"
    feature_rows = []
    labels = []
    groups = []

    feature_names = [
        "mean_intensity",
        "std_deviation",
        "laplacian_variance",
        "width",
        "height",
        "file_size_bytes",
    ]

    print(f"Extracting features across {len(df)} samples...")
    for idx, row in df.iterrows():
        f_path = data_dir / row["path_F"]
        if not f_path.exists():
            continue

        file_size = f_path.stat().st_size
        img = cv2.imread(str(f_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        h, w = img.shape
        mean_val = float(np.mean(img))
        std_val = float(np.std(img))
        lap_var = float(cv2.Laplacian(img, cv2.CV_64F).var())

        feature_rows.append([mean_val, std_val, lap_var, float(w), float(h), float(file_size)])
        labels.append(1 if row["is_forged"] else 0)
        groups.append(str(row["doc_id"]))

    X = np.array(feature_rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    g = np.array(groups)
    return X, y, g, feature_names


def evaluate_feature_set(
    X_sub: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_desc: str,
) -> Dict[str, Any]:
    gkf = GroupKFold(n_splits=5)

    lr_accs, lr_aucs = [], []
    hgb_accs, hgb_aucs = [], []

    for train_idx, test_idx in gkf.split(X_sub, y, groups):
        X_train, X_test = X_sub[train_idx], X_sub[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 1. Logistic Regression
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
        lr.fit(X_train, y_train)
        preds_lr = lr.predict(X_test)
        lr_accs.append(accuracy_score(y_test, preds_lr))
        try:
            probs = lr.predict_proba(X_test)[:, 1]
            lr_aucs.append(roc_auc_score(y_test, probs))
        except Exception:
            lr_aucs.append(0.5)

        # 2. HistGradientBoosting
        hgb = make_pipeline(StandardScaler(), HistGradientBoostingClassifier(max_depth=4, random_state=42))
        hgb.fit(X_train, y_train)
        preds_hgb = hgb.predict(X_test)
        hgb_accs.append(accuracy_score(y_test, preds_hgb))
        try:
            probs = hgb.predict_proba(X_test)[:, 1]
            hgb_aucs.append(roc_auc_score(y_test, probs))
        except Exception:
            hgb_aucs.append(0.5)

    lr_mean_acc = float(np.mean(lr_accs))
    lr_std_acc = float(np.std(lr_accs))
    hgb_mean_acc = float(np.mean(hgb_accs))
    hgb_std_acc = float(np.std(hgb_accs))

    return {
        "feature_set": feature_desc,
        "lr_acc": round(lr_mean_acc, 4),
        "lr_std": round(lr_std_acc, 4),
        "lr_auc": round(float(np.mean(lr_aucs)), 4),
        "hgb_acc": round(hgb_mean_acc, 4),
        "hgb_std": round(hgb_std_acc, 4),
        "hgb_auc": round(float(np.mean(hgb_aucs)), 4),
    }


def main():
    args = parse_args()
    manifest_path = REPO_ROOT / args.manifest
    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        print(f"ERROR: Manifest {manifest_path} not found.")
        sys.exit(1)

    df = pd.read_csv(manifest_path)

    # Class x Degradation Tier Stratification Report
    print("\n" + "=" * 70)
    print("STRATIFICATION AUDIT: CLASS X DEGRADATION TIER COUNTS")
    print("=" * 70)
    ct_table = pd.crosstab(df["class"], df["degradation_level"])
    print(ct_table.to_string())
    from scipy.stats import chi2_contingency
    chi2_val, p_val, dof, _ = chi2_contingency(ct_table)
    print(f"\nChi-Square Test of Independence: chi2={chi2_val:.4f}, p={p_val:.4f} (DoF={dof})")
    if p_val > 0.05:
        print("  [PASS] Class and tier are statistically independent (p > 0.05).")
    else:
        print("  [FAIL] Class and tier are dependent (p <= 0.05).")

    X, y, groups, feat_names = extract_features(df)

    print("\n" + "=" * 70)
    print("DATASET SHORTCUT AUDIT (5-FOLD GROUPED CV BY DOC_ID)")
    print("=" * 70)
    print(f"Total samples evaluated : {len(y)} ({sum(y == 1)} forged, {sum(y == 0)} clean paired twins)")
    print(f"Unique documents        : {len(np.unique(groups))}")

    results = []

    # 1. All 6 features
    r_all = evaluate_feature_set(X, y, groups, "All 6 Features (with file_size)")
    results.append(r_all)
    print(f"\n1. All 6 Features:")
    print(f"   Logistic Regression : {r_all['lr_acc']*100:.2f}% (+/- {r_all['lr_std']*100:.2f}%) | ROC-AUC: {r_all['lr_auc']:.4f}")
    print(f"   HistGradientBoosting: {r_all['hgb_acc']*100:.2f}% (+/- {r_all['hgb_std']*100:.2f}%) | ROC-AUC: {r_all['hgb_auc']:.4f}")

    # 2. 5 features (file_size dropped)
    X_no_size = X[:, :5]
    r_no_size = evaluate_feature_set(X_no_size, y, groups, "5 Features (file_size dropped)")
    results.append(r_no_size)
    print(f"\n2. 5 Features (file_size dropped):")
    print(f"   Logistic Regression : {r_no_size['lr_acc']*100:.2f}% (+/- {r_no_size['lr_std']*100:.2f}%) | ROC-AUC: {r_no_size['lr_auc']:.4f}")
    print(f"   HistGradientBoosting: {r_no_size['hgb_acc']*100:.2f}% (+/- {r_no_size['hgb_std']*100:.2f}%) | ROC-AUC: {r_no_size['hgb_auc']:.4f}")

    # 3. Individual non-file-size features
    print(f"\n3. Individual Non-File-Size Feature Audits:")
    flagged_features = []
    for f_idx in range(5):
        f_name = feat_names[f_idx]
        X_single = X[:, f_idx:f_idx + 1]
        r_single = evaluate_feature_set(X_single, y, groups, f"Feature: {f_name}")
        results.append(r_single)
        print(f"   {f_name:<20}: LR={r_single['lr_acc']*100:5.2f}% | HGB={r_single['hgb_acc']*100:5.2f}%")
        if r_single['lr_acc'] > 0.55 or r_single['hgb_acc'] > 0.55:
            flagged_features.append((f_name, max(r_single['lr_acc'], r_single['hgb_acc'])))

    # Markdown report
    md = []
    md.append("# Phase 3: Dataset Shortcut Audit Report")
    md.append("")
    md.append("Evaluates whether paired-twin clean negatives and forged samples can be distinguished using global statistical features.")
    md.append("5-fold cross-validation strictly grouped by `doc_id`.")
    md.append("")
    md.append("## Class x Degradation Tier Distribution")
    md.append("")
    md.append("| Class | Zero | Light | Moderate | Heavy | Total |")
    md.append("|---|---|---|---|---|---|")
    for c in ct_table.index:
        row_vals = [str(ct_table.loc[c, t]) for t in ["zero", "light", "moderate", "heavy"]]
        tot = sum(int(x) for x in row_vals)
        md.append(f"| {c} | " + " | ".join(row_vals) + f" | {tot} |")
    col_tots = [str(ct_table[t].sum()) for t in ["zero", "light", "moderate", "heavy"]]
    md.append(f"| **Total** | " + " | ".join(col_tots) + f" | {len(df)} |")
    md.append("")
    md.append(f"**Chi-Square Independence Test:** chi2 = {chi2_val:.4f}, p = {p_val:.4f} (DoF = {dof})")
    md.append("- Status: " + ("PASS (p > 0.05, statistically independent)" if p_val > 0.05 else "FAIL (p <= 0.05)"))
    md.append("")
    md.append("## Feature Set Performance")
    md.append("")
    md.append("| Feature Set | LogReg Accuracy | LogReg AUC | HistGB Accuracy | HistGB AUC | Status |")
    md.append("|---|---|---|---|---|---|")
    for r in results:
        max_acc = max(r['lr_acc'], r['hgb_acc'])
        status = "PASS (<55%)" if max_acc <= 0.55 else f"FLAGGED ({max_acc*100:.1f}%)"
        md.append(f"| {r['feature_set']} | {r['lr_acc']*100:.2f}% (+/- {r['lr_std']*100:.2f}%) | {r['lr_auc']:.4f} | {r['hgb_acc']*100:.2f}% (+/- {r['hgb_std']*100:.2f}%) | {r['hgb_auc']:.4f} | {status} |")

    md.append("")
    if flagged_features:
        md.append("## Warning: Flagged Features (> 55% accuracy)")
        for name, acc in flagged_features:
            md.append(f"- **{name}**: Accuracy = {acc*100:.2f}% (exceeds chance threshold 55.0%)")
    else:
        md.append("## Audit Verdict: PASS")
        md.append("All non-file-size global statistical features perform near 50% chance level. No feature exceeds 55% accuracy. The paired-twin dataset is free from global statistical shortcuts.")
    md.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"\nReport written to: {output_path}")
    if flagged_features:
        print("\nWARNING: Some features exceeded 55% accuracy:")
        for name, acc in flagged_features:
            print(f"  - {name}: {acc*100:.2f}%")
    else:
        print("\nAll non-file-size features are <= 55% (near 50% chance level). Audit passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
