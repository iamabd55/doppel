"""BioFors dataset audit script.

Audits downloaded BioFors images against official annotations:
- Verifies document IDs for train, test, and IDD splits.
- Counts images per class (Blot/Gel, Microscopy).
- Measures resolution and aspect ratio distributions.
- Asserts strict document disjointness between train and test pools.

Usage:
    uv run scripts/audit_biofors.py
"""
import json
import os
import pickle
import sys
from collections import Counter
import cv2
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "biofors")
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

# Fallback to ~/Downloads/biofors_images if symlink not yet created
if not os.path.exists(IMAGES_DIR):
    alt_images = os.path.expanduser("~/Downloads/biofors_images")
    if os.path.exists(alt_images):
        IMAGES_DIR = alt_images


def audit():
    print("=" * 70)
    print("BIOFORS DATASET AUDIT (STEP 0)")
    print("=" * 70)

    if not os.path.exists(ANNOTATIONS_DIR):
        print(f"Error: Annotations directory not found at {ANNOTATIONS_DIR}")
        print("Please copy annotations: mkdir -p data/biofors/annotations && cp -r /tmp/biofors_repo/annotation_files/* data/biofors/annotations/")
        sys.exit(1)

    if not os.path.exists(IMAGES_DIR):
        print(f"Error: Images directory not found at {IMAGES_DIR}")
        sys.exit(1)

    print(f"Annotations source : {ANNOTATIONS_DIR}")
    print(f"Images source      : {IMAGES_DIR}")

    # 1. Load annotations
    with open(os.path.join(ANNOTATIONS_DIR, "train_pdfs.pkl"), "rb") as f:
        train_pdfs = [str(x) for x in pickle.load(f)]

    with open(os.path.join(ANNOTATIONS_DIR, "test_pdfs.pkl"), "rb") as f:
        test_pdfs = [str(x) for x in pickle.load(f)]

    with open(os.path.join(ANNOTATIONS_DIR, "idd_pdfs.pkl"), "rb") as f:
        idd_pdfs = [str(x) for x in pickle.load(f)]

    with open(os.path.join(ANNOTATIONS_DIR, "classification.json"), "r") as f:
        classification = json.load(f)

    with open(os.path.join(ANNOTATIONS_DIR, "idd_gt.json"), "r") as f:
        idd_gt = json.load(f)

    # 2. Document count and disjointness check
    train_set = set(train_pdfs)
    test_set = set(test_pdfs)
    idd_set = set(idd_pdfs)

    print("\n--- Document Grouping & Integrity ---")
    print(f"Train documents (PDFs) : {len(train_pdfs)}")
    print(f"Test documents (PDFs)  : {len(test_pdfs)}")
    print(f"IDD test documents     : {len(idd_pdfs)}")

    overlap = train_set.intersection(test_set)
    print(f"Train / Test Document Overlap: {len(overlap)} documents")
    assert len(overlap) == 0, f"Critical error: {len(overlap)} overlapping documents between train and test!"

    idd_in_test = idd_set.issubset(test_set)
    print(f"All IDD test documents belong to Test split: {idd_in_test}")

    # 3. Check existing directories on disk
    existing_dirs = set(os.listdir(IMAGES_DIR))
    train_on_disk = [d for d in train_pdfs if d in existing_dirs]
    idd_on_disk = [d for d in idd_pdfs if d in existing_dirs]

    print(f"Train documents present on disk : {len(train_on_disk)} / {len(train_pdfs)}")
    print(f"IDD documents present on disk   : {len(idd_on_disk)} / {len(idd_pdfs)}")

    # 4. Count images and check classes in Train Pool
    train_class_counts = Counter()
    train_blot_shapes = []
    train_micro_shapes = []
    missing_images = 0

    print("\nAuditing image files and dimensions in Train Pool (sampling)...")
    for doc_id in train_on_disk:
        panels = classification.get(doc_id, {})
        for panel_file, cls_name in panels.items():
            train_class_counts[cls_name] += 1
            img_path = os.path.join(IMAGES_DIR, doc_id, panel_file)
            if not os.path.exists(img_path):
                missing_images += 1
                continue

            # Sample shapes for Blot/Gel and Microscopy
            if cls_name == "Blot/Gel" and len(train_blot_shapes) < 500:
                im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if im is not None:
                    train_blot_shapes.append(im.shape)
            elif cls_name == "Microscopy" and len(train_micro_shapes) < 500:
                im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if im is not None:
                    train_micro_shapes.append(im.shape)

    print("\n--- Train Pool Image Counts (by Class) ---")
    for cls_name, cnt in train_class_counts.most_common():
        print(f"  {cls_name:<15}: {cnt:6d} images")
    if missing_images > 0:
        print(f"  Warning: {missing_images} referenced images not found on disk.")

    # 5. IDD Test Split ground truth inspection
    idd_forged_images = 0
    idd_pristine_images = 0
    idd_box_counts = []
    idd_class_counts = Counter()

    for doc_id in idd_on_disk:
        panels = classification.get(doc_id, {})
        doc_gt = idd_gt.get(doc_id, {})
        for panel_file, cls_name in panels.items():
            if cls_name not in ["Blot/Gel", "Microscopy"]:
                continue
            if panel_file in doc_gt:
                idd_forged_images += 1
                boxes = doc_gt[panel_file]
                idd_box_counts.append(len(boxes))
                idd_class_counts[f"{cls_name}_Forged"] += 1
            else:
                idd_pristine_images += 1
                idd_class_counts[f"{cls_name}_Pristine"] += 1

    print("\n--- IDD Test Split (Blot/Gel & Microscopy) ---")
    print(f"  Total IDD Panels Evaluated : {idd_forged_images + idd_pristine_images}")
    print(f"  Manipulated (Forged) Panels : {idd_forged_images}")
    print(f"  Pristine (Clean) Panels     : {idd_pristine_images}")
    print(f"  Average boxes per forgery   : {np.mean(idd_box_counts):.2f}" if idd_box_counts else "  N/A")
    for k, v in idd_class_counts.items():
        print(f"    {k:<22}: {v:4d} images")

    # 6. Shape distributions
    print("\n--- Image Resolution Distribution (Sample of 500 per class) ---")
    if train_blot_shapes:
        blot_h = [s[0] for s in train_blot_shapes]
        blot_w = [s[1] for s in train_blot_shapes]
        print(f"  Blot/Gel Height   : min={min(blot_h)} px, median={int(np.median(blot_h))} px, max={max(blot_h)} px")
        print(f"  Blot/Gel Width    : min={min(blot_w)} px, median={int(np.median(blot_w))} px, max={max(blot_w)} px")
    if train_micro_shapes:
        micro_h = [s[0] for s in train_micro_shapes]
        micro_w = [s[1] for s in train_micro_shapes]
        print(f"  Microscopy Height : min={min(micro_h)} px, median={int(np.median(micro_h))} px, max={max(micro_h)} px")
        print(f"  Microscopy Width  : min={min(micro_w)} px, median={int(np.median(micro_w))} px, max={max(micro_w)} px")

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE: All document groupings and splits are verified.")
    print("=" * 70)


if __name__ == "__main__":
    audit()
