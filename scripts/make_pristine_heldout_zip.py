#!/usr/bin/env python3
"""Build data/pristine_heldout.zip: pristine Blot/Gel and Microscopy panels
from the 177 held-out train documents. No BioFors test data included.

Output zip contents:
  pristine_heldout/images/<doc_id>/<filename>
  pristine_heldout/pristine_manifest.csv  (path, doc_id, class, width, height)
"""
import csv
import json
import os
import sys
import zipfile

import cv2

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO_ROOT)


def main():
    held_out_path = os.path.join(REPO_ROOT, "configs", "held_out_pristine_docs.json")
    train_manifest_path = os.path.join(REPO_ROOT, "data", "biofors", "splits", "train_manifest.csv")
    images_dir = os.path.join(REPO_ROOT, "data", "biofors", "images")
    out_zip = os.path.join(REPO_ROOT, "data", "pristine_heldout.zip")

    with open(held_out_path, encoding="utf-8") as f:
        held_out_docs = set(json.load(f))

    print(f"Held-out pristine document IDs: {len(held_out_docs)}")

    # Read the train manifest and filter to held-out docs + target classes
    rows = []
    with open(train_manifest_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_str = str(row["doc_id"]).strip()
            cls = row["class"].strip()
            if doc_str in held_out_docs and cls in ("Blot/Gel", "Microscopy"):
                rows.append(row)

    print(f"Pristine panels to package: {len(rows)}")

    # Verify none come from the test split
    test_manifest_path = os.path.join(REPO_ROOT, "data", "biofors", "splits", "test_idd_manifest.csv")
    if os.path.exists(test_manifest_path):
        test_docs = set()
        with open(test_manifest_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                test_docs.add(str(row["doc_id"]).strip())
        overlap = held_out_docs.intersection(test_docs)
        assert len(overlap) == 0, f"FATAL: {len(overlap)} held-out docs overlap with BioFors test!"
        print(f"[OK] Zero overlap with BioFors IDD test docs ({len(test_docs)} test docs checked)")

    manifest_rows = []
    missing = 0

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for row in rows:
            doc_id = str(row["doc_id"]).strip()
            rel_path = row["rel_path"].strip()
            cls = row["class"].strip()

            img_path = os.path.join(images_dir, rel_path)
            if not os.path.exists(img_path):
                missing += 1
                continue

            img = cv2.imread(img_path)
            if img is None:
                missing += 1
                continue

            h, w = img.shape[:2]
            arc_name = f"pristine_heldout/images/{rel_path}"
            zf.write(img_path, arc_name)

            manifest_rows.append({
                "path": f"images/{rel_path}",
                "doc_id": doc_id,
                "class": cls,
                "width": w,
                "height": h,
            })

        # Write manifest CSV inside the zip
        csv_lines = ["path,doc_id,class,width,height"]
        for mr in manifest_rows:
            csv_lines.append(f"{mr['path']},{mr['doc_id']},{mr['class']},{mr['width']},{mr['height']}")
        csv_content = "\n".join(csv_lines) + "\n"
        zf.writestr("pristine_heldout/pristine_manifest.csv", csv_content)

    if missing > 0:
        print(f"[WARN] {missing} panels could not be read (missing or unreadable)")

    size_mb = os.path.getsize(out_zip) / (1024 * 1024)
    print(f"\nCreated {out_zip}")
    print(f"  Panels packaged: {len(manifest_rows)}")
    print(f"  Size: {size_mb:.1f} MB")

    # Summary by class
    from collections import Counter
    cls_counts = Counter(mr["class"] for mr in manifest_rows)
    for cls, count in sorted(cls_counts.items()):
        print(f"  {cls}: {count} panels")


if __name__ == "__main__":
    main()
