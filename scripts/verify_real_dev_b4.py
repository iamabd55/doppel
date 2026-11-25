#!/usr/bin/env python3
"""
verify_real_dev_b4.py: Verify real dev set of pristine panels from 177 held-out documents.

Why this exists:
Task B4 establishes the real dev set of pristine panels from the 177 held-out train documents
(configs/held_out_pristine_docs.json). We must programmatically verify that ZERO of these documents
appear in data/synthetic_v3/train (neither as base images nor as foreign patch donors).

What it does:
1. Unpacks data/pristine_heldout.zip into data/pristine_heldout/ if not already unpacked.
2. Loads the 177 held-out document IDs from configs/held_out_pristine_docs.json.
3. Loads all document IDs from data/synthetic_v3/train/meta.jsonl (base docs and foreign docs).
4. Verifies strictly that held_out_docs intersection with train_docs is EMPTY.
5. Verifies split disjointness with val and test splits.
6. Reports the exact count of real pristine dev panels by modality (Blot/Gel vs Microscopy).
"""
import json
import os
import sys
import zipfile
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    held_out_path = REPO_ROOT / "configs" / "held_out_pristine_docs.json"
    synth_train_meta = REPO_ROOT / "data" / "synthetic_v3" / "train" / "meta.jsonl"
    synth_val_meta = REPO_ROOT / "data" / "synthetic_v3" / "val" / "meta.jsonl"
    synth_test_meta = REPO_ROOT / "data" / "synthetic_v3" / "test" / "meta.jsonl"
    zip_path = REPO_ROOT / "data" / "pristine_heldout.zip"
    dev_dir = REPO_ROOT / "data" / "pristine_heldout"

    print("=== TASK B4: REAL DEV SET PRISTINE VERIFICATION ===")

    # 1. Unpack pristine_heldout.zip if needed
    if not (dev_dir / "pristine_manifest.csv").exists():
        print(f"Unpacking {zip_path} into {REPO_ROOT / 'data'}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(REPO_ROOT / "data")
        print("Unpack complete.")
    else:
        print(f"Found existing unpacked pristine dev set at {dev_dir}")

    # 2. Load held-out document IDs
    with open(held_out_path, "r", encoding="utf-8") as f:
        held_out_docs = set(str(x) for x in json.load(f))
    print(f"Total held-out pristine documents in config: {len(held_out_docs)}")

    # 3. Load synthetic v3 documents
    def get_synth_docs(meta_file: Path):
        base_docs = set()
        foreign_docs = set()
        with open(meta_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                base_docs.add(str(item["doc"]))
                if item.get("foreign_doc"):
                    foreign_docs.add(str(item["foreign_doc"]))
        return base_docs, foreign_docs

    train_base, train_foreign = get_synth_docs(synth_train_meta)
    val_base, val_foreign = get_synth_docs(synth_val_meta)
    test_base, test_foreign = get_synth_docs(synth_test_meta)

    all_train_docs = train_base.union(train_foreign)
    all_val_docs = val_base.union(val_foreign)
    all_test_docs = test_base.union(test_foreign)

    print("\n--- Synthetic v3 Document Counts ---")
    print(f"  Train: {len(train_base)} base docs, {len(train_foreign)} foreign docs (total {len(all_train_docs)} unique)")
    print(f"  Val  : {len(val_base)} base docs, {len(val_foreign)} foreign docs (total {len(all_val_docs)} unique)")
    print(f"  Test : {len(test_base)} base docs, {len(test_foreign)} foreign docs (total {len(all_test_docs)} unique)")

    # 4. Programmatic leakage assertions
    leakage_train_base = held_out_docs.intersection(train_base)
    leakage_train_foreign = held_out_docs.intersection(train_foreign)
    leakage_val = held_out_docs.intersection(all_val_docs)
    leakage_test = held_out_docs.intersection(all_test_docs)

    print("\n--- Leakage Verification with Held-Out Pristine Docs ---")
    print(f"  Overlap with synthetic train base docs   : {len(leakage_train_base)}")
    print(f"  Overlap with synthetic train foreign docs: {len(leakage_train_foreign)}")
    print(f"  Overlap with synthetic val docs          : {len(leakage_val)}")
    print(f"  Overlap with synthetic test docs         : {len(leakage_test)}")

    assert len(leakage_train_base) == 0, f"LEAKAGE: {leakage_train_base} in synthetic train base docs!"
    assert len(leakage_train_foreign) == 0, f"LEAKAGE: {leakage_train_foreign} in synthetic train foreign docs!"
    assert len(leakage_val) == 0, f"LEAKAGE: {leakage_val} in synthetic val docs!"
    assert len(leakage_test) == 0, f"LEAKAGE: {leakage_test} in synthetic test docs!"
    print("\n[VERIFIED] Zero document leakage: 177 held-out pristine docs NEVER appear in synthetic_v3!")

    # 5. Check panels in pristine_manifest.csv
    manifest_csv = dev_dir / "pristine_manifest.csv"
    df_manifest = pd.read_csv(manifest_csv)
    manifest_docs = set(df_manifest["doc_id"].astype(str).unique())
    print(f"\n--- Real Pristine Dev Manifest Statistics ---")
    print(f"  Manifest path: {manifest_csv}")
    print(f"  Total panels : {len(df_manifest)}")
    print(f"  Total docs   : {len(manifest_docs)}")

    # Verify all manifest docs belong to held_out_docs
    assert manifest_docs == held_out_docs, "Error: manifest docs do not exactly match held_out_pristine_docs.json!"

    class_counts = df_manifest["class"].value_counts().to_dict()
    print("\n=== REAL PRISTINE DEV SET BREAKDOWN ===")
    print(f"{'Class / Modality':20s} | {'Pristine Panels':16s} | {'Unique Documents':16s}")
    print("-" * 60)
    for cls in ["Blot/Gel", "Microscopy"]:
        cnt = class_counts.get(cls, 0)
        docs_cnt = len(df_manifest[df_manifest["class"] == cls]["doc_id"].unique())
        print(f"{cls:20s} | {cnt:16d} | {docs_cnt:16d}")
    print("-" * 60)
    print(f"{'Total Real Dev':20s} | {len(df_manifest):16d} | {len(manifest_docs):16d}")
    print("-" * 60)


if __name__ == "__main__":
    main()
