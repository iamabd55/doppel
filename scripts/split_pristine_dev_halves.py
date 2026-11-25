#!/usr/bin/env python3
"""
split_pristine_dev_halves.py: Split 177 held-out pristine docs into two disjoint halves.

Why this exists:
Avoids circular evaluation on the real pristine dev set. Half A is used solely
during threshold calibration (select_threshold.py), and Half B is used solely
for held-out Target 1 evaluation (eval_phase5.py).

What it does:
1. Reads the 177 document IDs from configs/held_out_pristine_docs.json.
2. Reads panel counts and domain classes (Blot/Gel vs Microscopy) from
   data/pristine_heldout/pristine_manifest.csv (falling back to data/pristine_heldout.zip).
3. Performs a fixed-seed (seed=42) document-level split stratified by Microscopy
   presence so Half A (89 docs) and Half B (88 docs) are balanced and disjoint.
4. Asserts zero document overlap and exact 177-doc coverage.
5. Writes configs/pristine_dev_split.json.
"""
import argparse
import csv
import json
import random
import zipfile
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_manifest_rows(manifest_csv: Path, pristine_zip: Path) -> List[Dict[str, str]]:
    if manifest_csv.exists():
        with open(manifest_csv, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    if pristine_zip.exists():
        with zipfile.ZipFile(pristine_zip, "r") as zf:
            names = [n for n in zf.namelist() if n.endswith("pristine_manifest.csv")]
            if names:
                text = zf.read(names[0]).decode("utf-8")
                return list(csv.DictReader(text.splitlines()))
    raise FileNotFoundError(f"Could not find {manifest_csv} or {pristine_zip}")


def main():
    parser = argparse.ArgumentParser(description="Split 177 held-out pristine docs into disjoint Half A and Half B.")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed for document split.")
    parser.add_argument(
        "--out",
        type=str,
        default=str(REPO_ROOT / "configs" / "pristine_dev_split.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    held_out_path = REPO_ROOT / "configs" / "held_out_pristine_docs.json"
    manifest_csv = REPO_ROOT / "data" / "pristine_heldout" / "pristine_manifest.csv"
    pristine_zip = REPO_ROOT / "data" / "pristine_heldout.zip"

    with open(held_out_path, "r", encoding="utf-8") as f:
        held_out_docs = sorted(str(x) for x in json.load(f))
    assert len(held_out_docs) == 177, f"Expected 177 held-out docs, got {len(held_out_docs)}"

    rows = load_manifest_rows(manifest_csv, pristine_zip)
    doc_stats: Dict[str, Dict[str, int]] = {
        d: {"total": 0, "blot_gel": 0, "microscopy": 0} for d in held_out_docs
    }
    for r in rows:
        d = str(r["doc_id"])
        assert d in doc_stats, f"Unexpected doc_id {d} in pristine manifest"
        doc_stats[d]["total"] += 1
        if r["class"] == "Microscopy":
            doc_stats[d]["microscopy"] += 1
        else:
            doc_stats[d]["blot_gel"] += 1

    # Stratify by whether document has Microscopy panels to balance both halves
    with_mic = sorted(d for d in held_out_docs if doc_stats[d]["microscopy"] > 0)
    blot_only = sorted(d for d in held_out_docs if doc_stats[d]["microscopy"] == 0)

    rng = random.Random(args.seed)
    rng.shuffle(with_mic)
    rng.shuffle(blot_only)

    half_a: List[str] = []
    half_b: List[str] = []

    for idx, d in enumerate(with_mic):
        if idx % 2 == 0:
            half_a.append(d)
        else:
            half_b.append(d)

    # Alternate starting side for blot_only so total doc counts stay 89 vs 88
    for idx, d in enumerate(blot_only):
        if idx % 2 == 0:
            half_b.append(d)
        else:
            half_a.append(d)

    half_a = sorted(half_a)
    half_b = sorted(half_b)

    # Strict disjointness assertions
    set_a = set(half_a)
    set_b = set(half_b)
    set_all = set(held_out_docs)
    assert len(set_a & set_b) == 0, f"Overlap between Half A and Half B: {set_a & set_b}"
    assert (set_a | set_b) == set_all, "Half A + Half B do not cover all 177 held-out docs"

    doc_to_half = {}
    for d in half_a:
        doc_to_half[d] = "A"
    for d in half_b:
        doc_to_half[d] = "B"

    def summarize(doc_list: List[str]) -> Dict[str, int]:
        return {
            "docs": len(doc_list),
            "panels": sum(doc_stats[d]["total"] for d in doc_list),
            "blot_gel_panels": sum(doc_stats[d]["blot_gel"] for d in doc_list),
            "microscopy_panels": sum(doc_stats[d]["microscopy"] for d in doc_list),
        }

    payload = {
        "seed": args.seed,
        "total_docs": len(held_out_docs),
        "total_panels": len(rows),
        "half_a_role": "threshold_calibration_only",
        "half_b_role": "target1_heldout_eval_only",
        "summary": {
            "half_a": summarize(half_a),
            "half_b": summarize(half_b),
        },
        "half_a": half_a,
        "half_b": half_b,
        "doc_to_half": doc_to_half,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[VERIFIED] Zero overlap between Half A ({len(half_a)} docs) and Half B ({len(half_b)} docs).")
    print(f"Half A (Calibration) : {payload['summary']['half_a']}")
    print(f"Half B (Target 1)    : {payload['summary']['half_b']}")
    print(f"Wrote split config to {out_path}")


if __name__ == "__main__":
    main()
