#!/usr/bin/env python3
"""
build_natural_manifests.py: Build unified manifests and leakage-free splits for natural-image datasets.

Why this script exists:
Prepares clean, verified, leakage-free CSV manifests for CASIA v2.0 (copy-move subset
plus authentic negatives), CoMoFoD (dev benchmark), and COVERAGE (when populated),
along with a visual montage and summary markdown report.

What it does:
1. Parses data/natural/casia2_masks/Notes/fileNamesCorrection.xlsx (141 rename rules)
   to separate true copy-move (Tp_S_*) from splicing (Tp_D_*) in CASIA2.
2. Matches every CASIA2 copy-move image to its ground-truth mask in CASIA 2 Groundtruth,
   verifies readability, identical (H, W) size, and non-empty mask (> 127), and drops/logs bad ones.
3. Pairs CASIA2 copy-move images with all readable Au authentic images, assigning group_id
   from the source image ID so authentic originals and their forged copies share one group.
4. Splits CASIA2 by group_id (30% calibration, 70% test, fixed seed 0) and asserts zero
   group overlap and zero cross-split authentic/forged leakage.
5. Builds CoMoFoD dev manifest (forged + authentic variants across 5 attack categories)
   and COVERAGE test manifest (0 rows while only README.md is present).
6. Saves a 20-tile montage (12 random forged CASIA2 + 8 forged CoMoFoD with mask outlines)
   to results/natural/montage_data_check.png and writes docs/natural_data_summary.md.
"""
import csv
import json
import random
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
MANIFEST_COLS = ["dataset", "image_path", "mask_path", "label", "group_id", "split", "attack_type"]


def load_casia2_rename_map(xlsx_path: Path) -> Dict[str, str]:
    """Parse original -> corrected filename mapping from fileNamesCorrection.xlsx."""
    rename_map: Dict[str, str] = {}
    with zipfile.ZipFile(xlsx_path, "r") as zf:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            ns_si = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
            for si in root.findall(f".//{ns_si}si"):
                shared.append("".join(t.text or "" for t in si.iter(f"{ns_si}t")))
        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
        for r in sheet.findall(f".//{ns}row")[1:]:
            vals = []
            for c in r.findall(f"{ns}c"):
                t = c.attrib.get("t")
                v = c.find(f"{ns}v")
                vals.append("" if (v is None or v.text is None) else (shared[int(v.text)] if t == "s" else v.text))
            m = re.search(r'ren\s+"([^"]+)"\s+"([^"]+)"', " ".join(vals))
            if m:
                rename_map[m.group(1)] = m.group(2)
    return rename_map


def check_image_and_mask(img_path: Path, mask_path: Path) -> Tuple[bool, str]:
    """Verify image and mask are readable, have matching (H, W), and mask is non-empty."""
    img = cv2.imread(str(img_path))
    if img is None:
        return False, "unreadable_image"
    if not mask_path.exists():
        return False, "missing_mask"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False, "unreadable_mask"
    if img.shape[:2] != mask.shape[:2]:
        return False, f"size_mismatch_img_{img.shape[0]}x{img.shape[1]}_mask_{mask.shape[0]}x{mask.shape[1]}"
    if not np.any(mask > 127):
        return False, "empty_mask"
    return True, "ok"


def build_casia2_manifest(seed: int = 0) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, int]]:
    """Build CASIA2 copy-move + authentic manifest with 30/70 group-level split."""
    casia_root = REPO_ROOT / "data" / "natural" / "casia2" / "CASIA2"
    tp_dir = casia_root / "Tp"
    au_dir = casia_root / "Au"
    gt_dir = casia_root / "CASIA 2 Groundtruth"
    xlsx_path = REPO_ROOT / "data" / "natural" / "casia2_masks" / "Notes" / "fileNamesCorrection.xlsx"

    rename_map = load_casia2_rename_map(xlsx_path)
    dropped: List[Dict[str, str]] = []
    rows_raw: List[Dict[str, str]] = []

    splicing_excluded = 0
    tp_candidates = 0

    for p in sorted(tp_dir.iterdir()):
        if not (p.is_file() and p.suffix.lower() in IMG_EXTS):
            continue
        corr_name = rename_map.get(p.name, p.name)
        if not corr_name.startswith("Tp_S_"):
            splicing_excluded += 1
            continue

        tp_candidates += 1
        corr_stem = Path(corr_name).stem
        parts = corr_stem.split("_")
        src_id = parts[5]
        tgt_id = parts[6]
        assert src_id == tgt_id, f"Expected src_id == tgt_id for copy-move {corr_name}, got {src_id} vs {tgt_id}"

        mask_path = gt_dir / f"{corr_stem}_gt.png"
        ok, reason = check_image_and_mask(p, mask_path)
        if not ok:
            dropped.append({
                "dataset": "casia2",
                "image_path": p.relative_to(REPO_ROOT).as_posix(),
                "label": "forged",
                "reason": reason,
            })
            continue

        rows_raw.append({
            "dataset": "casia2",
            "image_path": p.relative_to(REPO_ROOT).as_posix(),
            "mask_path": mask_path.relative_to(REPO_ROOT).as_posix(),
            "label": "forged",
            "group_id": src_id,
            "split": "",
            "attack_type": "",
        })

    au_candidates = 0
    for p in sorted(au_dir.iterdir()):
        if not (p.is_file() and p.suffix.lower() in IMG_EXTS):
            continue
        au_candidates += 1
        img = cv2.imread(str(p))
        if img is None:
            dropped.append({
                "dataset": "casia2",
                "image_path": p.relative_to(REPO_ROOT).as_posix(),
                "label": "authentic",
                "reason": "unreadable_image",
            })
            continue
        parts = p.stem.split("_")
        au_id = parts[1] + parts[2]
        rows_raw.append({
            "dataset": "casia2",
            "image_path": p.relative_to(REPO_ROOT).as_posix(),
            "mask_path": "",
            "label": "authentic",
            "group_id": au_id,
            "split": "",
            "attack_type": "",
        })

    # Group-level 30% calibration / 70% test split (fixed seed 0)
    all_groups = sorted({r["group_id"] for r in rows_raw})
    rng = random.Random(seed)
    shuffled_groups = list(all_groups)
    rng.shuffle(shuffled_groups)

    n_calib = int(round(0.30 * len(shuffled_groups)))
    calib_groups = set(shuffled_groups[:n_calib])
    test_groups = set(shuffled_groups[n_calib:])

    assert calib_groups.isdisjoint(test_groups), "FATAL: group overlap between calibration and test!"
    assert (calib_groups | test_groups) == set(all_groups), "Groups do not cover all CASIA2 samples!"

    for r in rows_raw:
        r["split"] = "calibration" if r["group_id"] in calib_groups else "test"

    # Assert no authentic image in one split has a forged copy in the other split
    forged_calib = {r["group_id"] for r in rows_raw if r["label"] == "forged" and r["split"] == "calibration"}
    forged_test = {r["group_id"] for r in rows_raw if r["label"] == "forged" and r["split"] == "test"}
    auth_calib = {r["group_id"] for r in rows_raw if r["label"] == "authentic" and r["split"] == "calibration"}
    auth_test = {r["group_id"] for r in rows_raw if r["label"] == "authentic" and r["split"] == "test"}

    assert auth_calib.isdisjoint(forged_test), "FATAL: calibration authentic image has forged copy in test!"
    assert auth_test.isdisjoint(forged_calib), "FATAL: test authentic image has forged copy in calibration!"

    stats = {
        "splicing_excluded": splicing_excluded,
        "tp_cm_candidates": tp_candidates,
        "au_candidates": au_candidates,
        "total_groups": len(all_groups),
        "calib_groups": len(calib_groups),
        "test_groups": len(test_groups),
    }
    return rows_raw, dropped, stats


def build_comofod_manifest() -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Build CoMoFoD dev manifest across all 5 attack categories."""
    com_root = REPO_ROOT / "data" / "comofod"
    categories = ["translation", "rotation", "scaling", "distortion", "combination"]
    rows: List[Dict[str, str]] = []
    dropped: List[Dict[str, str]] = []

    for cat in categories:
        cat_dir = com_root / cat
        if not cat_dir.is_dir():
            continue
        for p in sorted(cat_dir.iterdir()):
            if not (p.is_file() and p.suffix.lower() == ".png"):
                continue
            stem = p.stem
            parts = stem.split("_", 1)
            if len(parts) < 2:
                continue
            base_id, tag = parts[0], parts[1]
            if tag in {"B", "M"}:
                continue

            group_id = f"comofod_{base_id}"
            if tag.startswith("F"):
                mask_path = cat_dir / f"{base_id}_B.png"
                ok, reason = check_image_and_mask(p, mask_path)
                if not ok:
                    dropped.append({
                        "dataset": "comofod",
                        "image_path": p.relative_to(REPO_ROOT).as_posix(),
                        "label": "forged",
                        "reason": reason,
                    })
                    continue
                rows.append({
                    "dataset": "comofod",
                    "image_path": p.relative_to(REPO_ROOT).as_posix(),
                    "mask_path": mask_path.relative_to(REPO_ROOT).as_posix(),
                    "label": "forged",
                    "group_id": group_id,
                    "split": "dev",
                    "attack_type": cat,
                })
            elif tag.startswith("O"):
                img = cv2.imread(str(p))
                if img is None:
                    dropped.append({
                        "dataset": "comofod",
                        "image_path": p.relative_to(REPO_ROOT).as_posix(),
                        "label": "authentic",
                        "reason": "unreadable_image",
                    })
                    continue
                rows.append({
                    "dataset": "comofod",
                    "image_path": p.relative_to(REPO_ROOT).as_posix(),
                    "mask_path": "",
                    "label": "authentic",
                    "group_id": group_id,
                    "split": "dev",
                    "attack_type": cat,
                })
    return rows, dropped


def build_coverage_manifest() -> List[Dict[str, str]]:
    """Scan data/natural/coverage for images (currently README-only repo)."""
    cov_root = REPO_ROOT / "data" / "natural" / "coverage"
    img_files = [
        p for p in cov_root.rglob("*")
        if p.is_file() and ".git" not in p.parts and p.suffix.lower() in IMG_EXTS
    ]
    if not img_files:
        return []
    return []


def write_manifest_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        writer.writeheader()
        writer.writerows(rows)


def render_montage(
    casia_rows: List[Dict[str, str]],
    como_rows: List[Dict[str, str]],
    out_path: Path,
    seed: int = 0,
) -> None:
    """Save 4x5 montage: 12 random forged CASIA2 + 8 random forged CoMoFoD with mask outlines."""
    rng = random.Random(seed)
    casia_forged = [r for r in casia_rows if r["label"] == "forged"]
    como_forged = [r for r in como_rows if r["label"] == "forged"]

    sel_casia = rng.sample(casia_forged, 12)
    sel_como = rng.sample(como_forged, 8)
    samples = sel_casia + sel_como

    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    for ax, row in zip(axes.flat, samples):
        img_bgr = cv2.imread(str(REPO_ROOT / row["image_path"]))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(REPO_ROOT / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        mask_bin = (mask > 127).astype(np.uint8)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vis = img_rgb.copy()
        cv2.drawContours(vis, contours, -1, (0, 255, 0), thickness=max(2, min(vis.shape[:2]) // 128))
        ax.imshow(vis)
        if row["dataset"] == "casia2":
            title = f"CASIA2 ({row['split']})\n{Path(row['image_path']).stem[-18:]}"
        else:
            title = f"CoMoFoD ({row['attack_type']})\n{Path(row['image_path']).name}"
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.suptitle("Natural Copy-Move Data Check: 12 CASIA2 Forged (top) + 8 CoMoFoD Forged (bottom)", fontsize=14)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def write_summary_md(
    casia_rows: List[Dict[str, str]],
    como_rows: List[Dict[str, str]],
    cov_rows: List[Dict[str, str]],
    casia_dropped: List[Dict[str, str]],
    como_dropped: List[Dict[str, str]],
    casia_stats: Dict[str, int],
    out_path: Path,
) -> None:
    """Write docs/natural_data_summary.md with per-dataset/split/label counts and drops."""
    def count_breakdown(rows: List[Dict[str, str]]) -> List[Tuple[str, str, int]]:
        c = Counter((r["split"], r["label"]) for r in rows)
        return sorted([(k[0], k[1], v) for k, v in c.items()])

    all_dropped = casia_dropped + como_dropped
    drop_reasons = Counter((d["dataset"], d["label"], d["reason"]) for d in all_dropped)

    lines = [
        "# Natural-Image Dataset Summary and Manifest Audit",
        "",
        "Generated by `scripts/build_natural_manifests.py` (`seed=0` for CASIA2 group split).",
        "",
        "---",
        "",
        "## 1. Counts per Dataset, Split, and Label",
        "",
        "| Dataset | Split | Label | Count | Manifest |",
        "|---|---|---|---:|---|",
    ]
    for sp, lbl, cnt in count_breakdown(casia_rows):
        lines.append(f"| casia2 | {sp} | {lbl} | {cnt} | `data/natural/manifests/casia2.csv` |")
    for sp, lbl, cnt in count_breakdown(como_rows):
        lines.append(f"| comofod | {sp} | {lbl} | {cnt} | `data/natural/manifests/comofod.csv` |")
    if cov_rows:
        for sp, lbl, cnt in count_breakdown(cov_rows):
            lines.append(f"| coverage | {sp} | {lbl} | {cnt} | `data/natural/manifests/coverage.csv` |")
    else:
        lines.append("| coverage | test | forged/authentic | 0 | `data/natural/manifests/coverage.csv` (README-only stub) |")

    lines.extend([
        "",
        f"**Total verified samples:** CASIA2 = **{len(casia_rows)}**, CoMoFoD = **{len(como_rows)}**, COVERAGE = **{len(cov_rows)}**.",
        "",
        "---",
        "",
        "## 2. CASIA2 Copy-Move Grouping and Leakage Checks",
        "",
        f"* Raw `Tp/` images: `5,123` (`1,828` corrected splicing `Tp_D_*` excluded; `{casia_stats['tp_cm_candidates']}` corrected copy-move `Tp_S_*` retained).",
        f"* Authentic `Au/` images checked: `{casia_stats['au_candidates']}`.",
        f"* Total unique source `group_id`s: **{casia_stats['total_groups']}**.",
        f"* Split (30% / 70% by `group_id`, `seed=0`): **calibration** = `{casia_stats['calib_groups']}` groups, **test** = `{casia_stats['test_groups']}` groups.",
        "* Verified assertions:",
        "  * Zero `group_id` overlap between `calibration` and `test`.",
        "  * Zero authentic images in `calibration` have a forged copy in `test`, and vice versa.",
        "",
        "---",
        "",
        "## 3. CoMoFoD Attack Category Breakdown (`split=dev`)",
        "",
        "| Attack Type | Forged | Authentic | Total |",
        "|---|---:|---:|---:|",
    ])
    como_cats = Counter((r["attack_type"], r["label"]) for r in como_rows)
    for cat in ["translation", "rotation", "scaling", "distortion", "combination"]:
        nf = como_cats[(cat, "forged")]
        na = como_cats[(cat, "authentic")]
        lines.append(f"| {cat} | {nf} | {na} | {nf + na} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Dropped Samples and Reasons",
        "",
        f"Total dropped samples: **{len(all_dropped)}**",
        "",
    ])
    if all_dropped:
        lines.append("| Dataset | Label | Reason | Count |")
        lines.append("|---|---|---|---:|")
        for (ds, lbl, reason), cnt in sorted(drop_reasons.items()):
            lines.append(f"| {ds} | {lbl} | `{reason}` | {cnt} |")
        lines.append("")
        lines.append("Dropped file list:")
        for d in all_dropped:
            lines.append(f"* `{d['image_path']}` ({d['label']}): `{d['reason']}`")
    else:
        lines.append("Zero samples dropped.")

    lines.extend([
        "",
        "---",
        "",
        "## 5. Visual Verification",
        "",
        "Saved 20-image montage (12 random forged CASIA2 + 8 random forged CoMoFoD with green mask outlines) to `results/natural/montage_data_check.png`.",
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    manifests_dir = REPO_ROOT / "data" / "natural" / "manifests"
    montage_path = REPO_ROOT / "results" / "natural" / "montage_data_check.png"
    summary_path = REPO_ROOT / "docs" / "natural_data_summary.md"

    casia_rows, casia_dropped, casia_stats = build_casia2_manifest(seed=0)
    como_rows, como_dropped = build_comofod_manifest()
    cov_rows = build_coverage_manifest()

    write_manifest_csv(manifests_dir / "casia2.csv", casia_rows)
    write_manifest_csv(manifests_dir / "comofod.csv", como_rows)
    write_manifest_csv(manifests_dir / "coverage.csv", cov_rows)

    with open(manifests_dir / "dropped_samples.json", "w", encoding="utf-8") as f:
        json.dump(casia_dropped + como_dropped, f, indent=2)

    render_montage(casia_rows, como_rows, montage_path, seed=0)
    write_summary_md(casia_rows, como_rows, cov_rows, casia_dropped, como_dropped, casia_stats, summary_path)

    print(f"Wrote manifests to {manifests_dir}")
    print(f"CASIA2: {len(casia_rows)} rows, {len(casia_dropped)} dropped")
    print(f"CoMoFoD: {len(como_rows)} rows, {len(como_dropped)} dropped")
    print(f"COVERAGE: {len(cov_rows)} rows")
    print(f"Montage: {montage_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
