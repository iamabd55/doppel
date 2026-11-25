#!/usr/bin/env python3
"""
build_phase5_notebook.py: Build the platform-agnostic Phase 5 training notebook.

Why this script exists:
Programmatically constructs notebooks/phase5_train.ipynb conforming strictly to
the nbformat v4 specification for ONE training run on Kaggle, Colab, or local GPU.

Features of the generated notebook:
1. Single top configuration cell containing all paths (code archive, datasets,
   output directory) with clean defaults for Kaggle (/kaggle/input/...,
   /kaggle/working/...) and Colab (/content/...), with zero Drive dependency.
2. Saves a checkpoint every epoch (epoch_XX.pt, last.pt, best.pt), supports
   automatic resume from the latest checkpoint, and logs per-epoch validation
   recall at the 1% FPR budget in train_log.csv.
3. Checkpoint selection rule: picked by validation recall at a 1% FPR budget
   over pristine plus unrelated_paste negatives (not by loss).
4. Disjoint Real Dev evaluation: calibrates thresholds strictly on Half A of the
   177 held-out pristine docs and evaluates Target 1 strictly on Half B (plus the
   val-only threshold saved inside best.pt).
5. Evaluates Targets 2 and 3 on synthetic_v3/test without touching the locked
   real BioFors IDD test split.
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "phase5_train.ipynb"


def make_markdown_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_lines],
    }


def make_code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_lines],
    }


def build_notebook():
    cells = []

    # Cell 0: Header & Target Metrics Reference
    cells.append(
        make_markdown_cell([
            "# Doppel Phase 5: Platform-Agnostic CNN Training and Target Verification",
            "",
            "**Notebook:** `phase5_train.ipynb` (Kaggle, Colab, or local GPU)  ",
            "**Architecture:** ResNetCorrNet v2 with Image Head (25 epochs, AdamW, AMP)  ",
            "**Data:** `synthetic_v3` (triplet format: `pristine`, `forged`, `unrelated_paste`)  ",
            "**Checkpoint Selection Rule:** Pick `best.pt` by validation recall on forged panels at a 1.0% FPR budget over combined negatives (`pristine` + `unrelated_paste`), NOT by loss.  ",
            "",
            "### Pre-Registered Targets Reference (`docs/phase5_targets.md`)",
            "1. **Target 1 (Real Pristine Dev Half B FAR):** Clean FAR <= 2.0% on **Half B** (89 docs, 2,070 panels), strictly disjoint from **Half A** (88 docs, 1,969 panels used for calibration) and `synthetic_v3/train`. Also evaluated at the val-only threshold stored in `best.pt`.",
            "2. **Target 2 (Seam Shortcut Suppression):** On `synthetic_v3/test`, FPR on `unrelated_paste` <= 1.5x pristine FPR and <= 3.0% absolute.",
            "3. **Target 3 (Synthetic Forged Recall):** On `synthetic_v3/test`, forged recall >= 30.0% at the operating threshold.",
            "4. **Real BioFors IDD Test Split:** Strictly NOT evaluated here (one-shot exploratory evaluation reserved for the end).",
        ])
    )

    # Cell 1: Unified Configuration Cell (All Paths & Hyperparameters)
    cells.append(
        make_code_cell([
            "import sys",
            "from pathlib import Path",
            "",
            "# Choose 'auto', 'kaggle', 'colab', or 'local'",
            "PLATFORM = \"auto\"",
            "if PLATFORM == \"auto\":",
            "    if Path(\"/kaggle\").exists():",
            "        PLATFORM = \"kaggle\"",
            "    elif \"google.colab\" in sys.modules or Path(\"/content\").exists():",
            "        PLATFORM = \"colab\"",
            "    else:",
            "        PLATFORM = \"local\"",
            "",
            "# Platform-specific input and output paths (edit here if your dataset slugs differ)",
            "if PLATFORM == \"kaggle\":",
            "    CODE_SOURCE = Path(\"/kaggle/input/doppel-code/doppel_code.zip\")",
            "    SYNTH_V3_SOURCE = Path(\"/kaggle/input/doppel-synthetic-v3/synthetic_v3\")",
            "    PRISTINE_SOURCE = Path(\"/kaggle/input/doppel-pristine-heldout/pristine_heldout.zip\")",
            "    WORK_ROOT = Path(\"/kaggle/working/doppel_work\")",
            "    OUT_DIR = Path(\"/kaggle/working/runs_phase5/corr_v3\")",
            "    SYNC_DIR = None",
            "elif PLATFORM == \"colab\":",
            "    CODE_SOURCE = Path(\"/content/doppel_code.zip\")",
            "    SYNTH_V3_SOURCE = Path(\"/content/synthetic_v3.zip\")",
            "    PRISTINE_SOURCE = Path(\"/content/pristine_heldout.zip\")",
            "    WORK_ROOT = Path(\"/content/doppel_work\")",
            "    OUT_DIR = Path(\"/content/runs_phase5/corr_v3\")",
            "    SYNC_DIR = None  # Optional persistent folder if you decide to attach one",
            "else:",
            "    CODE_SOURCE = Path(\"data/doppel_code.zip\")",
            "    SYNTH_V3_SOURCE = Path(\"data/synthetic_v3\")",
            "    PRISTINE_SOURCE = Path(\"data/pristine_heldout.zip\")",
            "    WORK_ROOT = Path(\".\")",
            "    OUT_DIR = Path(\"runs_phase5/corr_v3\")",
            "    SYNC_DIR = None",
            "",
            "# Training configuration",
            "EPOCHS = 25",
            "BATCH_SIZE = 16",
            "LR = 3e-4",
            "WEIGHT_DECAY = 1e-4",
            "CROP_SIZE = 256",
            "SAVE_EVERY = 1        # Save epoch_XX.pt + last.pt every epoch",
            "RESUME_MODE = \"auto\"  # 'auto' resumes from latest checkpoint if interrupted",
            "",
            "print(f\"Platform        : {PLATFORM}\")",
            "print(f\"CODE_SOURCE     : {CODE_SOURCE}\")",
            "print(f\"SYNTH_V3_SOURCE : {SYNTH_V3_SOURCE}\")",
            "print(f\"PRISTINE_SOURCE : {PRISTINE_SOURCE}\")",
            "print(f\"WORK_ROOT       : {WORK_ROOT}\")",
            "print(f\"OUT_DIR         : {OUT_DIR}\")",
            "print(f\"SYNC_DIR        : {SYNC_DIR}\")",
        ])
    )

    # Cell 2: Environment Check and Input Path Resolution (Supports .zip or Extracted Kaggle Folders)
    cells.append(
        make_code_cell([
            "import subprocess",
            "import zipfile",
            "import torch",
            "",
            "print(f\"Python version : {sys.version.split()[0]}\")",
            "print(f\"PyTorch version: {torch.__version__}\")",
            "print(f\"CUDA Available : {torch.cuda.is_available()}\")",
            "if torch.cuda.is_available():",
            "    print(f\"Active GPU     : {torch.cuda.get_device_name(0)}\")",
            "    subprocess.run([\"nvidia-smi\"], check=False)",
            "else:",
            "    print(\"[WARNING] CUDA is not available. Enable GPU accelerator before full training.\")",
            "",
            "WORK_ROOT.mkdir(parents=True, exist_ok=True)",
            "OUT_DIR.mkdir(parents=True, exist_ok=True)",
            "",
            "# 1. Resolve or extract Doppel codebase",
            "def resolve_code_root(src: Path, work_root: Path) -> Path:",
            "    if src.is_file() and src.suffix == \".zip\":",
            "        dest = work_root / \"doppel\"",
            "        if not (dest / \"detector\").exists():",
            "            print(f\"Extracting code archive {src} -> {work_root}...\")",
            "            with zipfile.ZipFile(src, \"r\") as zf:",
            "                zf.extractall(work_root)",
            "        if (work_root / \"doppel\" / \"detector\").exists():",
            "            return work_root / \"doppel\"",
            "        if (work_root / \"detector\").exists():",
            "            return work_root",
            "    for cand in [src, src / \"doppel\", src.parent, src.parent / \"doppel\"]:",
            "        if cand.is_dir() and (cand / \"detector\").exists():",
            "            return cand",
            "    raise FileNotFoundError(f\"Could not locate Doppel code at {src}. Check CODE_SOURCE in Cell 1.\")",
            "",
            "# 2. Resolve or extract synthetic_v3 dataset",
            "def resolve_synth_dir(src: Path, work_root: Path) -> Path:",
            "    if src.is_file() and src.suffix == \".zip\":",
            "        dest = work_root / \"data\" / \"synthetic_v3\"",
            "        if not (dest / \"train\" / \"meta.jsonl\").exists():",
            "            (work_root / \"data\").mkdir(parents=True, exist_ok=True)",
            "            print(f\"Extracting synthetic_v3 archive {src} -> {work_root / 'data'}...\")",
            "            with zipfile.ZipFile(src, \"r\") as zf:",
            "                zf.extractall(work_root / \"data\")",
            "        if (dest / \"train\" / \"meta.jsonl\").exists():",
            "            return dest",
            "        if (work_root / \"data\" / \"train\" / \"meta.jsonl\").exists():",
            "            return work_root / \"data\"",
            "    for cand in [src, src / \"synthetic_v3\", src.parent, src.parent / \"synthetic_v3\"]:",
            "        if cand.is_dir() and (cand / \"train\" / \"meta.jsonl\").exists():",
            "            return cand",
            "    raise FileNotFoundError(f\"Could not locate synthetic_v3 at {src}. Check SYNTH_V3_SOURCE in Cell 1.\")",
            "",
            "# 3. Resolve pristine held-out source (.zip or extracted directory on Kaggle)",
            "def resolve_pristine_source(src: Path) -> Path:",
            "    if src.exists():",
            "        return src",
            "    for cand in [src.parent / \"pristine_heldout\", src.parent]:",
            "        if (cand / \"pristine_manifest.csv\").exists() or (cand / \"pristine_heldout\" / \"pristine_manifest.csv\").exists():",
            "            return cand",
            "    raise FileNotFoundError(f\"Could not locate pristine held-out data at {src}. Check PRISTINE_SOURCE in Cell 1.\")",
            "",
            "DOPPEL_ROOT = resolve_code_root(CODE_SOURCE, WORK_ROOT)",
            "SYNTH_DIR = resolve_synth_dir(SYNTH_V3_SOURCE, WORK_ROOT)",
            "PRISTINE_DATA_PATH = resolve_pristine_source(PRISTINE_SOURCE)",
            "",
            "if str(DOPPEL_ROOT) not in sys.path:",
            "    sys.path.insert(0, str(DOPPEL_ROOT))",
            "",
            "print(f\"Resolved DOPPEL_ROOT       : {DOPPEL_ROOT}\")",
            "print(f\"Resolved SYNTH_DIR         : {SYNTH_DIR}\")",
            "print(f\"Resolved PRISTINE_DATA_PATH: {PRISTINE_DATA_PATH}\")",
        ])
    )

    # Cell 3: Dataset Sanity Check & Real Dev Split Verification
    cells.append(
        make_code_cell([
            "import json",
            "from torch.utils.data import DataLoader",
            "from detector.colab_dataset import ColabDoppelDataset, colab_collate_fn, verify_splits_disjointness",
            "",
            "print(\"=== DATASET & DEV SPLIT SANITY CHECK ===\")",
            "train_ds = ColabDoppelDataset(",
            "    manifest_path=str(SYNTH_DIR / \"train\"),",
            "    split=\"train\",",
            "    crop_size=(CROP_SIZE, CROP_SIZE),",
            "    pos_crop_prob=0.50,",
            "    augment=True,",
            "    seed=42,",
            ")",
            "val_ds = ColabDoppelDataset(",
            "    manifest_path=str(SYNTH_DIR / \"val\"),",
            "    split=\"val\",",
            "    pad_multiple=32,",
            "    augment=False,",
            ")",
            "",
            "print(f\"Train samples: {len(train_ds)}\")",
            "print(f\"Val samples  : {len(val_ds)}\")",
            "verify_splits_disjointness(train_ds, val_ds)",
            "",
            "loader_sanity = DataLoader(val_ds, batch_size=6, shuffle=False, collate_fn=colab_collate_fn)",
            "batch = next(iter(loader_sanity))",
            "imgs, masks, labels = batch[\"image\"], batch[\"mask\"], batch[\"label\"]",
            "kinds = batch[\"meta\"][\"kind\"]",
            "print(f\"Batch shapes : image={tuple(imgs.shape)}, mask={tuple(masks.shape)}, label={tuple(labels.shape)}\")",
            "print(f\"Sample kinds : {kinds}\")",
            "",
            "for idx, k in enumerate(kinds):",
            "    m_sum = float(masks[idx].sum().item())",
            "    lbl = float(labels[idx].item())",
            "    if k in [\"pristine\", \"unrelated_paste\"]:",
            "        assert m_sum == 0.0, f\"Expected empty mask for {k}, got {m_sum}\"",
            "        assert lbl == 0.0, f\"Expected 0.0 label for {k}, got {lbl}\"",
            "    elif k == \"forged\":",
            "        assert m_sum > 0.0, f\"Expected non-empty union mask for forged, got {m_sum}\"",
            "        assert lbl == 1.0, f\"Expected 1.0 label for forged, got {lbl}\"",
            "",
            "dev_split_json = DOPPEL_ROOT / \"configs\" / \"pristine_dev_split.json\"",
            "with open(dev_split_json, \"r\", encoding=\"utf-8\") as f:",
            "    dev_split = json.load(f)",
            "half_a = set(str(d) for d in dev_split[\"half_a\"])",
            "half_b = set(str(d) for d in dev_split[\"half_b\"])",
            "assert half_a.isdisjoint(half_b), \"Overlap between Half A and Half B!\"",
            "assert len(half_a) + len(half_b) == 177",
            "print(f\"[VERIFIED] Dev split disjoint: Half A ({len(half_a)} docs calibration) vs Half B ({len(half_b)} docs Target 1).\")",
        ])
    )

    # Cell 4: Quick Smoke Test (2 Epochs on Subset, Save Every Epoch)
    cells.append(
        make_code_cell([
            "import time",
            "import pandas as pd",
            "",
            "print(\"================ Running 2-Epoch Smoke Test ================\")",
            "t_start = time.time()",
            "smoke_out = WORK_ROOT / \"smoke_run_phase5\"",
            "smoke_out.mkdir(parents=True, exist_ok=True)",
            "",
            "cmd_smoke = [",
            "    \"python3\", str(DOPPEL_ROOT / \"scripts\" / \"train_cnn.py\"),",
            "    \"--model\", \"corr\",",
            "    \"--image-head\",",
            "    \"--epochs\", \"2\",",
            "    \"--batch-size\", str(BATCH_SIZE),",
            "    \"--subset\", \"64\",",
            "    \"--crop-size\", str(CROP_SIZE),",
            "    \"--data-dir\", str(SYNTH_DIR),",
            "    \"--pristine-val-count\", \"0\",",
            "    \"--save-every\", \"1\",",
            "    \"--out-dir\", str(smoke_out),",
            "    \"--device\", \"cuda\" if torch.cuda.is_available() else \"cpu\",",
            "]",
            "subprocess.run(cmd_smoke, check=True)",
            "df_smoke = pd.read_csv(smoke_out / \"train_log.csv\")",
            "assert \"val_recall_budget\" in df_smoke.columns",
            "assert (smoke_out / \"epoch_01.pt\").exists() and (smoke_out / \"epoch_02.pt\").exists()",
            "print(f\"[SUCCESS] Smoke test finished in {time.time() - t_start:.1f}s. Per-epoch checkpoints & train_log.csv verified.\")",
        ])
    )

    # Cell 5: Full Single Training Run (Save Every Epoch + Resume Support)
    cells.append(
        make_code_cell([
            "def find_latest_checkpoint(dirs):",
            "    best_cand = None",
            "    best_ep = -1",
            "    for d in dirs:",
            "        if d is None or not Path(d).exists():",
            "            continue",
            "        d = Path(d)",
            "        cands = ([d / \"last.pt\"] if (d / \"last.pt\").exists() else []) + sorted(d.glob(\"epoch_*.pt\"))",
            "        for p in cands:",
            "            try:",
            "                ck = torch.load(p, map_location=\"cpu\", weights_only=False)",
            "            except TypeError:",
            "                ck = torch.load(p, map_location=\"cpu\")",
            "            ep = int(ck.get(\"epoch\", 0))",
            "            if ep > best_ep:",
            "                best_ep = ep",
            "                best_cand = p",
            "    return best_cand, best_ep",
            "",
            "resume_ckpt, resume_epoch = (None, 0)",
            "if RESUME_MODE == \"auto\":",
            "    resume_ckpt, resume_epoch = find_latest_checkpoint([OUT_DIR, SYNC_DIR])",
            "",
            "if resume_ckpt is not None and resume_epoch >= EPOCHS and (OUT_DIR / \"best.pt\").exists() and (OUT_DIR / \"train_log.csv\").exists():",
            "    print(f\"[INFO] Training already reached epoch {resume_epoch}/{EPOCHS} at {OUT_DIR}. Skipping re-run.\")",
            "else:",
            "    print(f\"[INFO] Starting/resuming single ResNetCorrNet v2 run for {EPOCHS} epochs (save_every={SAVE_EVERY})...\")",
            "    print(\"[INFO] Checkpoint selection rule: highest validation recall at 1% FPR budget over pristine + unrelated_paste negatives.\")",
            "    cmd_train = [",
            "        \"python3\", str(DOPPEL_ROOT / \"scripts\" / \"train_cnn.py\"),",
            "        \"--model\", \"corr\",",
            "        \"--image-head\",",
            "        \"--epochs\", str(EPOCHS),",
            "        \"--batch-size\", str(BATCH_SIZE),",
            "        \"--lr\", str(LR),",
            "        \"--weight-decay\", str(WEIGHT_DECAY),",
            "        \"--crop-size\", str(CROP_SIZE),",
            "        \"--data-dir\", str(SYNTH_DIR),",
            "        \"--pristine-val-count\", \"0\",",
            "        \"--save-every\", str(SAVE_EVERY),",
            "        \"--out-dir\", str(OUT_DIR),",
            "        \"--device\", \"cuda\" if torch.cuda.is_available() else \"cpu\",",
            "    ]",
            "    if resume_ckpt is not None:",
            "        print(f\"[INFO] Resuming from checkpoint {resume_ckpt} (epoch {resume_epoch})\")",
            "        cmd_train.extend([\"--resume\", str(resume_ckpt)])",
            "    if SYNC_DIR is not None:",
            "        Path(SYNC_DIR).mkdir(parents=True, exist_ok=True)",
            "        cmd_train.extend([\"--drive-sync-dir\", str(SYNC_DIR)])",
            "",
            "    subprocess.run(cmd_train, check=True)",
            "    print(f\"[SUCCESS] Training complete. Best checkpoint: {OUT_DIR / 'best.pt'}\")",
        ])
    )

    # Cell 6: Plot Training and Validation Progression (Including Per-Epoch Recall @ 1% FPR)
    cells.append(
        make_code_cell([
            "import matplotlib.pyplot as plt",
            "import pandas as pd",
            "",
            "log_csv = OUT_DIR / \"train_log.csv\"",
            "if log_csv.exists():",
            "    df_log = pd.read_csv(log_csv)",
            "    display_cols = [",
            "        c for c in [",
            "            \"epoch\", \"train_loss\", \"val_forged_f1\", \"val_roc_auc\",",
            "            \"val_recall_budget\", \"val_synth_clean_far\",",
            "            \"val_pristine_clean_far\", \"val_unrelated_paste_far\", \"val_meets_budget\"",
            "        ] if c in df_log.columns",
            "    ]",
            "    print(df_log[display_cols].tail(10).to_string(index=False))",
            "",
            "    fig, axes = plt.subplots(1, 3, figsize=(18, 5))",
            "    axes[0].plot(df_log[\"epoch\"], df_log[\"train_loss\"], label=\"Train Loss\", color=\"crimson\")",
            "    axes[0].set_title(\"Training Loss\")",
            "    axes[0].set_xlabel(\"Epoch\")",
            "    axes[0].grid(True, alpha=0.3)",
            "    axes[0].legend()",
            "",
            "    axes[1].plot(df_log[\"epoch\"], df_log[\"val_forged_f1\"], label=\"Val Forged Pixel F1\", color=\"forestgreen\")",
            "    axes[1].plot(df_log[\"epoch\"], df_log[\"val_roc_auc\"], label=\"Val ROC-AUC\", color=\"dodgerblue\")",
            "    axes[1].set_title(\"Validation Pixel F1 and ROC-AUC\")",
            "    axes[1].set_xlabel(\"Epoch\")",
            "    axes[1].grid(True, alpha=0.3)",
            "    axes[1].legend()",
            "",
            "    axes[2].plot(df_log[\"epoch\"], df_log[\"val_recall_budget\"] * 100.0, label=\"Val Recall @ 1% Neg FPR\", color=\"darkorange\", marker=\"o\")",
            "    axes[2].set_title(\"Val Forged Recall @ 1% Neg FPR Budget\")",
            "    axes[2].set_xlabel(\"Epoch\")",
            "    axes[2].set_ylabel(\"Recall (%)\")",
            "    axes[2].grid(True, alpha=0.3)",
            "    axes[2].legend()",
            "",
            "    plt.tight_layout()",
            "    plt.savefig(OUT_DIR / \"training_curves.png\", dpi=150)",
            "    plt.show()",
            "else:",
            "    print(f\"No log CSV found at {log_csv}\")",
        ])
    )

    # Cell 7: Threshold Calibration on synthetic_v3/val + Real Pristine Half A ONLY
    cells.append(
        make_code_cell([
            "select_script = DOPPEL_ROOT / \"scripts\" / \"select_threshold.py\"",
            "best_ckpt = OUT_DIR / \"best.pt\"",
            "val_split_path = SYNTH_DIR / \"val\"",
            "class_map_path = DOPPEL_ROOT / \"configs\" / \"v3_panel_classes.json\"",
            "dev_split_json = DOPPEL_ROOT / \"configs\" / \"pristine_dev_split.json\"",
            "cache_npz_path = OUT_DIR / \"val_pristine_halfA_preds.npz\"",
            "out_yaml = OUT_DIR / \"cnn_threshold_phase5.yaml\"",
            "",
            "print(f\"[INFO] Calibrating thresholds on synthetic_v3/val + Real Pristine Half A ONLY...\")",
            "!python3 \"$select_script\" \\",
            "    --checkpoint \"$best_ckpt\" \\",
            "    --model corr \\",
            "    --image-head \\",
            "    --val-manifest \"$val_split_path\" \\",
            "    --data-dir \"$SYNTH_DIR\" \\",
            "    --pristine-zip \"$PRISTINE_DATA_PATH\" \\",
            "    --dev-split-json \"$dev_split_json\" \\",
            "    --dev-half A \\",
            "    --class-map \"$class_map_path\" \\",
            "    --cache-npz \"$cache_npz_path\" \\",
            "    --out-yaml \"$out_yaml\" \\",
            "    --budgets 0.01 0.02 0.03",
            "",
            "if out_yaml.exists():",
            "    print(\"\\n--- Selected cnn_threshold_phase5.yaml (Calibrated on Half A) ---\")",
            "    with open(out_yaml, \"r\", encoding=\"utf-8\") as f:",
            "        print(f.read())",
        ])
    )

    # Cell 8: Pre-Registered Target Verification (Half B Target 1 + synthetic_v3/test Targets 2 & 3)
    cells.append(
        make_code_cell([
            "print(\"=== PRE-REGISTERED TARGET VERIFICATION (DISJOINT HALF B + SYNTHETIC V3 TEST) ===\")",
            "eval_p5_script = DOPPEL_ROOT / \"scripts\" / \"eval_phase5.py\"",
            "test_split_path = SYNTH_DIR / \"test\"",
            "",
            "!python3 \"$eval_p5_script\" \\",
            "    --checkpoint \"$best_ckpt\" \\",
            "    --config \"$out_yaml\" \\",
            "    --budget 0.01 \\",
            "    --test-dir \"$test_split_path\" \\",
            "    --pristine-zip \"$PRISTINE_DATA_PATH\" \\",
            "    --dev-split-json \"$dev_split_json\" \\",
            "    --class-map \"$class_map_path\" \\",
            "    --out-dir \"$OUT_DIR\"",
        ])
    )

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": [],
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2)

    print(f"[SUCCESS] Built {NOTEBOOK_PATH} with {len(cells)} cells.")


if __name__ == "__main__":
    build_notebook()
