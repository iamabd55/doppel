#!/usr/bin/env python3
"""
audit_train_forgeries.py: Audit BioFors TRAIN split for real forged panels.

Why this exists:
Task A1 requires determining if real forged panels exist in the BioFors TRAIN split
(not the locked test split), how many there are per class (Blot/Gel vs Microscopy),
and whether they have usable region masks. This decides whether real positive panels
can be incorporated into CNN training.

What it does:
1. Loads train_manifest.csv to get all document IDs and panels in the train split.
2. Cross-references the official ground-truth annotation files:
   - idd_gt.json (intra-document duplication / copy-move bounding boxes)
   - cstd_gt.json (cross-scale/transform duplication bounding boxes)
   - edd_gt.json (extra-document duplication bounding boxes)
3. Checks if any document in the train split has ground-truth manipulation annotations.
4. Outputs a structured count table per class.
"""
import json
import os
import pickle
import pandas as pd

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    splits_dir = os.path.join(repo_root, "data", "biofors", "splits")
    ann_dir = os.path.join(repo_root, "data", "biofors", "annotations")

    # 1. Load train manifest
    train_manifest_path = os.path.join(splits_dir, "train_manifest.csv")
    df_train = pd.read_csv(train_manifest_path)
    train_docs = set(df_train["doc_id"].astype(str).unique())
    print(f"Total panels in train_manifest.csv: {len(df_train)}")
    print(f"Total unique documents in train_manifest.csv: {len(train_docs)}")

    # 2. Load official annotation files
    with open(os.path.join(ann_dir, "classification.json"), "r") as f:
        classification = json.load(f)

    with open(os.path.join(ann_dir, "idd_gt.json"), "r") as f:
        idd_gt = json.load(f)

    with open(os.path.join(ann_dir, "cstd_gt.json"), "r") as f:
        cstd_gt = json.load(f)

    with open(os.path.join(ann_dir, "edd_gt.json"), "r") as f:
        edd_gt = json.load(f)

    # 3. Check for IDD (copy-move) forgeries in train
    idd_train_docs = set(idd_gt.keys()).intersection(train_docs)
    cstd_train_docs = set(cstd_gt.keys()).intersection(train_docs)
    edd_train_docs = set(edd_gt.keys()).intersection(train_docs)

    print("\n--- Document Overlap with Ground Truth Annotations ---")
    print(f"IDD ground-truth documents in train: {len(idd_train_docs)} / {len(idd_gt)}")
    print(f"CSTD ground-truth documents in train: {len(cstd_train_docs)} / {len(cstd_gt)}")
    print(f"EDD ground-truth documents in train: {len(edd_train_docs)} / {len(edd_gt)}")

    # Check panel-level matches in train_manifest for IDD
    idd_panels_found = []
    for doc_id, files in idd_gt.items():
        if str(doc_id) in train_docs:
            for fn, boxes in files.items():
                cls = classification.get(str(doc_id), {}).get(fn, "Unknown")
                idd_panels_found.append({
                    "doc_id": doc_id,
                    "filename": fn,
                    "class": cls,
                    "num_boxes": len(boxes),
                    "boxes": boxes
                })

    print(f"\nTotal real forged IDD panels found in BioFors train split: {len(idd_panels_found)}")

    # Class breakdown in train manifest
    class_counts = df_train["class"].value_counts().to_dict()
    print("\n--- Train Manifest Total Panels by Class ---")
    for cls, cnt in class_counts.items():
        print(f"  {cls}: {cnt} panels")

    # Table report
    print("\n=== A1 RESULTS TABLE ===")
    print(f"{'Split':10s} | {'Class':12s} | {'Total Panels':14s} | {'Forged Panels':14s} | {'Usable Masks':14s}")
    print("-" * 75)
    for cls in ["Blot/Gel", "Microscopy"]:
        total_p = sum(df_train["class"] == cls)
        forged_p = sum(p["class"] == cls for p in idd_panels_found)
        usable_m = sum(p["class"] == cls and p["num_boxes"] > 0 for p in idd_panels_found)
        print(f"{'TRAIN':10s} | {cls:12s} | {total_p:14d} | {forged_p:14d} | {usable_m:14d}")
    print("-" * 75)

if __name__ == "__main__":
    main()
