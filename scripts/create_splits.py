"""Create document-level train/val/test split manifests for BioFors.

Partitions the 696 BioFors train documents into Train (85%) and Validation (15%)
sets by document ID, ensuring zero data leakage.

Generates:
- data/biofors/splits/train_manifest.csv
- data/biofors/splits/val_manifest.csv
- data/biofors/splits/test_idd_manifest.csv

Usage:
    uv run scripts/create_splits.py
"""
import csv
import json
import os
import pickle
import random
import sys
from collections import Counter
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "synthetic.yaml")


def load_config():
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: Config not found at {CONFIG_PATH}")
        sys.exit(1)
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    seed = cfg.get("seed", 42)
    paths = cfg["paths"]
    splits_cfg = cfg["splits"]

    project_root = os.path.join(os.path.dirname(__file__), "..")
    ann_dir = os.path.normpath(os.path.join(project_root, paths["annotations_dir"]))
    img_dir = os.path.normpath(os.path.join(project_root, paths["images_dir"]))
    out_dir = os.path.normpath(os.path.join(project_root, paths["splits_dir"]))
    os.makedirs(out_dir, exist_ok=True)

    target_classes = set(splits_cfg.get("target_classes", ["Blot/Gel", "Microscopy"]))
    train_ratio = splits_cfg.get("train_doc_ratio", 0.85)

    print("=" * 70)
    print("STEP 1: BIOFORS DOCUMENT-LEVEL SPLIT CREATION")
    print("=" * 70)
    print(f"Random seed       : {seed}")
    print(f"Target classes    : {list(target_classes)}")
    print(f"Train/Val ratio   : {train_ratio:.2f} / {1.0 - train_ratio:.2f} by Document")

    # 1. Load official annotations
    with open(os.path.join(ann_dir, "train_pdfs.pkl"), "rb") as f:
        train_pdfs = sorted([str(x) for x in pickle.load(f)])

    with open(os.path.join(ann_dir, "test_pdfs.pkl"), "rb") as f:
        test_pdfs = sorted([str(x) for x in pickle.load(f)])

    with open(os.path.join(ann_dir, "idd_pdfs.pkl"), "rb") as f:
        idd_pdfs = sorted([str(x) for x in pickle.load(f)])

    with open(os.path.join(ann_dir, "classification.json"), "r") as f:
        classification = json.load(f)

    with open(os.path.join(ann_dir, "idd_gt.json"), "r") as f:
        idd_gt = json.load(f)

    # 2. Partition 696 train documents by document ID
    rng = random.Random(seed)
    shuffled_train_pdfs = list(train_pdfs)
    rng.shuffle(shuffled_train_pdfs)

    n_train = int(round(len(shuffled_train_pdfs) * train_ratio))
    train_doc_ids = sorted(shuffled_train_pdfs[:n_train])
    val_doc_ids = sorted(shuffled_train_pdfs[n_train:])

    # 3. Assert zero document overlap
    s_train = set(train_doc_ids)
    s_val = set(val_doc_ids)
    s_test = set(test_pdfs)
    s_idd = set(idd_pdfs)

    assert len(s_train.intersection(s_val)) == 0, "Error: Overlap between Train and Val documents!"
    assert len(s_train.intersection(s_test)) == 0, "Error: Overlap between Train and Test documents!"
    assert len(s_val.intersection(s_test)) == 0, "Error: Overlap between Val and Test documents!"
    assert s_idd.issubset(s_test), "Error: IDD test documents must belong to the test split!"

    print("\n--- Document Counts ---")
    print(f"  Train pool documents : {len(train_doc_ids)}")
    print(f"  Val pool documents   : {len(val_doc_ids)}")
    print(f"  Test IDD documents   : {len(idd_pdfs)} (strictly held-out)")
    print("  Document Overlap     : 0 (Zero leakage verified)")

    # 4. Build manifests
    def build_rows(doc_list, split_name):
        rows = []
        for doc_id in doc_list:
            panels = classification.get(doc_id, {})
            for panel_file, cls_name in sorted(panels.items()):
                if cls_name not in target_classes:
                    continue
                rel_path = f"{doc_id}/{panel_file}"
                rows.append({
                    "doc_id": doc_id,
                    "filename": panel_file,
                    "rel_path": rel_path,
                    "class": cls_name,
                    "split": split_name,
                })
        return rows

    train_rows = build_rows(train_doc_ids, "train")
    val_rows = build_rows(val_doc_ids, "val")

    # Build Test IDD rows with ground truth manipulation info
    test_idd_rows = []
    for doc_id in idd_pdfs:
        panels = classification.get(doc_id, {})
        doc_gt = idd_gt.get(doc_id, {})
        for panel_file, cls_name in sorted(panels.items()):
            if cls_name not in target_classes:
                continue
            is_manip = panel_file in doc_gt
            boxes = doc_gt.get(panel_file, [])
            rel_path = f"{doc_id}/{panel_file}"
            test_idd_rows.append({
                "doc_id": doc_id,
                "filename": panel_file,
                "rel_path": rel_path,
                "class": cls_name,
                "split": "test_idd",
                "is_manipulated": is_manip,
                "box_count": len(boxes),
            })

    # 5. Write CSVs
    train_csv = os.path.join(out_dir, "train_manifest.csv")
    val_csv = os.path.join(out_dir, "val_manifest.csv")
    test_csv = os.path.join(out_dir, "test_idd_manifest.csv")

    fieldnames_pool = ["doc_id", "filename", "rel_path", "class", "split"]
    with open(train_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_pool)
        writer.writeheader()
        writer.writerows(train_rows)

    with open(val_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_pool)
        writer.writeheader()
        writer.writerows(val_rows)

    fieldnames_test = ["doc_id", "filename", "rel_path", "class", "split", "is_manipulated", "box_count"]
    with open(test_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_test)
        writer.writeheader()
        writer.writerows(test_idd_rows)

    # 6. Report summary
    def print_class_summary(rows, name):
        counts = Counter(r["class"] for r in rows)
        print(f"\n{name} ({len(rows)} panels across {len(set(r['doc_id'] for r in rows))} documents):")
        for cls_name, cnt in counts.most_common():
            print(f"    {cls_name:<15}: {cnt:5d} panels ({cnt/len(rows)*100:.1f}%)")

    print_class_summary(train_rows, "Train Pool Manifest")
    print_class_summary(val_rows, "Validation Pool Manifest")

    test_counts = Counter(r["class"] for r in test_idd_rows)
    test_forged = sum(1 for r in test_idd_rows if r["is_manipulated"])
    test_pristine = sum(1 for r in test_idd_rows if not r["is_manipulated"])
    print(f"\nTest IDD Manifest ({len(test_idd_rows)} panels across {len(idd_pdfs)} documents):")
    print(f"    Manipulated (Forged) : {test_forged:5d} panels")
    print(f"    Pristine (Clean)     : {test_pristine:5d} panels")
    for cls_name, cnt in test_counts.most_common():
        print(f"    {cls_name:<15}: {cnt:5d} panels")

    print("\nManifest files generated:")
    print(f"  1. {train_csv}")
    print(f"  2. {val_csv}")
    print(f"  3. {test_csv}")
    print("\n" + "=" * 70)
    print("STEP 1 COMPLETE: Document-level manifests successfully written.")
    print("=" * 70)


if __name__ == "__main__":
    main()
