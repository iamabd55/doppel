#!/usr/bin/env python3
"""
build_phase5_notebook.py: Build the Google Colab training notebook for Phase 5.

Why this script exists:
Programmatically constructs notebooks/phase5_train.ipynb conforming strictly to
the nbformat v4 specification. This avoids manual JSON editing errors and allows
automated unit test validation.

Features of the generated notebook:
1. Self-contained environment setup (drive mount, unzip archives).
2. Data sanity checks (verifying triplet format, empty masks for negatives).
3. ONE training run of ResNetCorrNet v2 (image head, 25 epochs, AMP, AdamW).
4. Checkpoint selection rule: picked by validation recall at 1% FPR budget over
   pristine plus unrelated_paste negatives (not by loss).
5. Automatic drive syncing of best.pt, last.pt, and train_log.csv.
6. Dev threshold selection and validation against pre-registered targets:
   - Target 1: real pristine dev FAR <= 2.0%.
   - Target 2: synthetic v3 test unrelated_paste FPR <= 1.5x pristine FPR and <= 3.0% absolute.
   - Target 3: synthetic v3 test forged recall >= 30.0%.
   - Real BioFors test split is strictly excluded (exploratory only, run at the very end).
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
            "# Doppel Phase 5: CNN Training and Pre-Registered Target Verification",
            "",
            "**Notebook:** `phase5_train.ipynb`  ",
            "**Target Architecture:** ResNetCorrNet v2 with Image Head (25 epochs, AdamW, AMP)  ",
            "**Data:** `data/synthetic_v3` (triplet format: pristine, forged, unrelated_paste)  ",
            "**Checkpoint Selection Rule:** Pick `best.pt` by validation recall on forged panels at a 1.0% False Positive Rate budget over combined negatives (pristine plus unrelated_paste), NOT by loss.  ",
            "",
            "### Pre-Registered Targets Reference (docs/phase5_targets.md)",
            "1. **Target 1 (Real Pristine Dev FAR):** Clean FAR <= 2.0% on 4,039 pristine panels from 177 held-out documents.",
            "2. **Target 2 (Seam Shortcut Suppression):** On synthetic v3 test, FPR on `unrelated_paste` <= 1.5x pristine FPR and <= 3.0% absolute.",
            "3. **Target 3 (Synthetic Forged Recall):** On synthetic v3 test, forged recall >= 30.0% at the operating threshold.",
            "4. **Real BioFors IDD Test Split:** Strictly NOT evaluated here. One-shot exploratory evaluation reserved for final phase.",
        ])
    )

    # Cell 1: Environment & GPU Verification
    cells.append(
        make_code_cell([
            "import sys",
            "import torch",
            "",
            "print(f\"Python version: {sys.version.split()[0]}\")",
            "print(f\"PyTorch version: {torch.__version__}\")",
            "print(f\"CUDA Available: {torch.cuda.is_available()}\")",
            "",
            "if torch.cuda.is_available():",
            "    gpu_name = torch.cuda.get_device_name(0)",
            "    print(f\"Active GPU: {gpu_name}\")",
            "    !nvidia-smi",
            "else:",
            "    print(\"[WARNING] CUDA is not available. Please go to Runtime -> Change runtime type -> T4 GPU.\")",
        ])
    )

    # Cell 2: Google Drive Mount & Workspace Directories
    cells.append(
        make_code_cell([
            "from google.colab import drive",
            "from pathlib import Path",
            "import os",
            "",
            "# Mount Google Drive",
            "drive.mount('/content/drive')",
            "",
            "DRIVE_DIR = Path('/content/drive/MyDrive/doppel')",
            "DRIVE_RUNS_DIR = DRIVE_DIR / 'runs_phase5'",
            "DRIVE_RUNS_DIR.mkdir(parents=True, exist_ok=True)",
            "",
            "CONTENT_DATA = Path('/content/data')",
            "CONTENT_DATA.mkdir(parents=True, exist_ok=True)",
            "",
            "DOPPEL_ROOT = Path('/content/doppel')",
            "DOPPEL_ROOT.mkdir(parents=True, exist_ok=True)",
            "sys.path.insert(0, str(DOPPEL_ROOT))",
            "",
            "print(f\"Drive root: {DRIVE_DIR} (exists: {DRIVE_DIR.exists()})\")",
            "print(f\"Phase 5 runs: {DRIVE_RUNS_DIR}\")",
        ])
    )

    # Cell 3: Unpack Archives into Fast Local Scratch Storage
    cells.append(
        make_code_cell([
            "import shutil",
            "import zipfile",
            "",
            "# 1. Unpack code package",
            "code_zip = DRIVE_DIR / 'doppel_code.zip'",
            "if not code_zip.exists():",
            "    # Fallback to local upload location if not directly in Drive folder",
            "    code_zip = Path('/content/doppel_code.zip')",
            "",
            "if code_zip.exists():",
            "    print(f\"Unpacking {code_zip} to /content ...\")",
            "    with zipfile.ZipFile(code_zip, 'r') as zf:",
            "        zf.extractall('/content')",
            "else:",
            "    print(\"[WARNING] doppel_code.zip not found. Please upload it to MyDrive/doppel/\")",
            "",
            "# 2. Unpack synthetic v3 dataset",
            "synth_zip = DRIVE_DIR / 'synthetic_v3.zip'",
            "if not synth_zip.exists():",
            "    synth_zip = Path('/content/synthetic_v3.zip')",
            "",
            "SYNTH_DIR = CONTENT_DATA / 'synthetic_v3'",
            "if (SYNTH_DIR / 'train' / 'meta.jsonl').exists():",
            "    print(f\"Found extracted synthetic_v3 at {SYNTH_DIR}\")",
            "elif synth_zip.exists():",
            "    print(f\"Unpacking {synth_zip} to {CONTENT_DATA} ...\")",
            "    with zipfile.ZipFile(synth_zip, 'r') as zf:",
            "        zf.extractall(CONTENT_DATA)",
            "    print(\"Unpack synthetic_v3 complete.\")",
            "else:",
            "    print(\"[WARNING] synthetic_v3.zip not found. Please upload it to MyDrive/doppel/\")",
            "",
            "# 3. Copy real pristine held-out dev archive to local scratch for fast evaluation",
            "pristine_zip = DRIVE_DIR / 'pristine_heldout.zip'",
            "if not pristine_zip.exists():",
            "    pristine_zip = Path('/content/pristine_heldout.zip')",
            "",
            "PRISTINE_ZIP_LOCAL = CONTENT_DATA / 'pristine_heldout.zip'",
            "if PRISTINE_ZIP_LOCAL.exists():",
            "    print(f\"Found local pristine archive at {PRISTINE_ZIP_LOCAL}\")",
            "elif pristine_zip.exists():",
            "    print(f\"Copying {pristine_zip} to {PRISTINE_ZIP_LOCAL} ...\")",
            "    shutil.copy2(pristine_zip, PRISTINE_ZIP_LOCAL)",
            "    print(\"Copy pristine_heldout.zip complete.\")",
            "else:",
            "    print(\"[WARNING] pristine_heldout.zip not found. Please upload it to MyDrive/doppel/\")",
        ])
    )

    # Cell 4: Dataset Sanity Check (v3 Triplet Structure and Disjointness)
    cells.append(
        make_code_cell([
            "from torch.utils.data import DataLoader",
            "from detector.colab_dataset import ColabDoppelDataset, colab_collate_fn, verify_splits_disjointness",
            "",
            "train_split = SYNTH_DIR / 'train'",
            "val_split = SYNTH_DIR / 'val'",
            "",
            "print(\"=== DATASET SANITY CHECK ===\")",
            "train_ds = ColabDoppelDataset(",
            "    manifest_path=str(train_split),",
            "    split='train',",
            "    crop_size=(256, 256),",
            "    pos_crop_prob=0.50,",
            "    augment=True,",
            "    seed=42,",
            ")",
            "val_ds = ColabDoppelDataset(",
            "    manifest_path=str(val_split),",
            "    split='val',",
            "    pad_multiple=32,",
            "    augment=False,",
            ")",
            "",
            "print(f\"Train samples: {len(train_ds)}\")",
            "print(f\"Val samples  : {len(val_ds)}\")",
            "verify_splits_disjointness(train_ds, val_ds)",
            "",
            "# Verify sample batch and mask properties",
            "loader_sanity = DataLoader(val_ds, batch_size=6, shuffle=False, collate_fn=colab_collate_fn)",
            "batch = next(iter(loader_sanity))",
            "",
            "imgs = batch['image']",
            "masks = batch['mask']",
            "labels = batch['label']",
            "kinds = batch['meta']['kind']",
            "",
            "print(f\"Batch image shape: {imgs.shape} (dtype: {imgs.dtype})\")",
            "print(f\"Batch mask shape : {masks.shape} (dtype: {masks.dtype})\")",
            "print(f\"Batch label shape: {labels.shape}\")",
            "print(f\"Sample kinds     : {kinds}\")",
            "",
            "# Check negative empty masks",
            "for idx in range(len(kinds)):",
            "    k = kinds[idx]",
            "    m_sum = float(masks[idx].sum().item())",
            "    lbl = float(labels[idx].item())",
            "    if k in ['pristine', 'unrelated_paste']:",
            "        assert m_sum == 0.0, f\"Expected empty mask for {k}, got {m_sum}\"",
            "        assert lbl == 0.0, f\"Expected 0.0 label for {k}, got {lbl}\"",
            "    elif k == 'forged':",
            "        assert m_sum > 0.0, f\"Expected non-empty mask for forged, got {m_sum}\"",
            "        assert lbl == 1.0, f\"Expected 1.0 label for forged, got {lbl}\"",
            "",
            "print(\"[SUCCESS] Verified: empty masks for negatives (pristine and unrelated_paste), union mask for forged.\")",
        ])
    )

    # Cell 5: Quick Smoke Test (2 Epochs on Subset)
    cells.append(
        make_code_cell([
            "import subprocess",
            "import time",
            "",
            "print(\"================ Running Smoke Test for Phase 5 CorrNet v2 ================\")",
            "t_start = time.time()",
            "smoke_out = Path('/content/smoke_run')",
            "smoke_out.mkdir(parents=True, exist_ok=True)",
            "",
            "cmd_smoke = [",
            "    \"python3\", str(DOPPEL_ROOT / \"scripts\" / \"train_cnn.py\"),",
            "    \"--model\", \"corr\",",
            "    \"--image-head\",",
            "    \"--epochs\", \"2\",",
            "    \"--batch-size\", \"16\",",
            "    \"--subset\", \"64\",",
            "    \"--crop-size\", \"256\",",
            "    \"--data-dir\", str(SYNTH_DIR),",
            "    \"--pristine-val-count\", \"0\",",
            "    \"--out-dir\", str(smoke_out),",
            "    \"--device\", \"cuda\" if torch.cuda.is_available() else \"cpu\",",
            "]",
            "res = subprocess.run(cmd_smoke, check=True)",
            "elapsed_smoke = time.time() - t_start",
            "print(f\"[SUCCESS] Smoke test completed in {elapsed_smoke:.1f}s. Pipeline ready for full run.\")",
        ])
    )

    # Cell 6: Full Single Training Run of ResNetCorrNet v2
    cells.append(
        make_code_cell([
            "import subprocess",
            "",
            "corr_v3_drive_dir = DRIVE_RUNS_DIR / \"corr_v3\"",
            "corr_v3_local_dir = Path(\"/content/runs/corr_v3\")",
            "corr_v3_local_dir.mkdir(parents=True, exist_ok=True)",
            "",
            "# Check if completed run already exists on Drive",
            "if (corr_v3_drive_dir / \"best.pt\").exists() and (corr_v3_drive_dir / \"train_log.csv\").exists():",
            "    print(f\"[INFO] Found completed CorrNet v3 run at {corr_v3_drive_dir}.\")",
            "    shutil.copytree(corr_v3_drive_dir, corr_v3_local_dir, dirs_exist_ok=True)",
            "else:",
            "    print(\"[INFO] Starting full training for ResNetCorrNet v2 on synthetic_v3 (25 epochs)...\")",
            "    print(\"[INFO] Checkpoint selection rule: highest validation recall at 1% FPR budget over negatives.\")",
            "    cmd_train = [",
            "        \"python3\", str(DOPPEL_ROOT / \"scripts\" / \"train_cnn.py\"),",
            "        \"--model\", \"corr\",",
            "        \"--image-head\",",
            "        \"--epochs\", \"25\",",
            "        \"--batch-size\", \"16\",",
            "        \"--lr\", \"3e-4\",",
            "        \"--weight-decay\", \"1e-4\",",
            "        \"--crop-size\", \"256\",",
            "        \"--data-dir\", str(SYNTH_DIR),",
            "        \"--pristine-val-count\", \"0\",",
            "        \"--out-dir\", str(corr_v3_local_dir),",
            "        \"--drive-sync-dir\", str(corr_v3_drive_dir),",
            "        \"--save-every\", \"2\",",
            "        \"--device\", \"cuda\" if torch.cuda.is_available() else \"cpu\",",
            "    ]",
            "    subprocess.run(cmd_train, check=True)",
            "    print(f\"[SUCCESS] Full training complete. Best checkpoint saved to {corr_v3_drive_dir / 'best.pt'}\")",
        ])
    )

    # Cell 7: Plot Training and Validation Progression
    cells.append(
        make_code_cell([
            "import matplotlib.pyplot as plt",
            "import pandas as pd",
            "",
            "log_csv = corr_v3_local_dir / \"train_log.csv\"",
            "if log_csv.exists():",
            "    df_log = pd.read_csv(log_csv)",
            "    fig, axes = plt.subplots(1, 3, figsize=(18, 5))",
            "",
            "    # Train loss & train F1",
            "    axes[0].plot(df_log['epoch'], df_log['train_loss'], label='Train Loss', color='crimson')",
            "    axes[0].set_title('Training Loss')",
            "    axes[0].set_xlabel('Epoch')",
            "    axes[0].grid(True, alpha=0.3)",
            "    axes[0].legend()",
            "",
            "    # Validation F1 and ROC-AUC",
            "    axes[1].plot(df_log['epoch'], df_log['val_f1'], label='Val Pixel F1', color='forestgreen')",
            "    axes[1].plot(df_log['epoch'], df_log['val_roc_auc'], label='Val ROC-AUC', color='dodgerblue')",
            "    axes[1].set_title('Validation Pixel Metrics')",
            "    axes[1].set_xlabel('Epoch')",
            "    axes[1].grid(True, alpha=0.3)",
            "    axes[1].legend()",
            "",
            "    # Validation Recall at 1% Negative FPR budget",
            "    axes[2].plot(df_log['epoch'], df_log['val_recall_budget'] * 100, label='Recall @ 1% Neg FPR', color='darkorange', marker='o')",
            "    axes[2].set_title('Val Forged Recall @ 1% Neg FPR Budget')",
            "    axes[2].set_xlabel('Epoch')",
            "    axes[2].set_ylabel('Recall (%)')",
            "    axes[2].grid(True, alpha=0.3)",
            "    axes[2].legend()",
            "",
            "    plt.tight_layout()",
            "    plt.savefig(corr_v3_local_dir / 'training_curves.png', dpi=150)",
            "    plt.show()",
            "else:",
            "    print(f\"No log CSV found at {log_csv}\")",
        ])
    )

    # Cell 8: Threshold Calibration and Config Generation
    cells.append(
        make_code_cell([
            "select_script = DOPPEL_ROOT / \"scripts\" / \"select_threshold.py\"",
            "best_ckpt = corr_v3_local_dir / \"best.pt\"",
            "val_split_path = SYNTH_DIR / \"val\"",
            "class_map_path = DOPPEL_ROOT / \"configs\" / \"v3_panel_classes.json\"",
            "cache_npz_path = corr_v3_local_dir / \"val_pristine_preds.npz\"",
            "out_yaml = DRIVE_DIR / \"cnn_threshold_phase5.yaml\"",
            "",
            "print(f\"[INFO] Selecting operating points on {best_ckpt}...\")",
            "!python3 \"$select_script\" \\",
            "    --checkpoint \"$best_ckpt\" \\",
            "    --model corr \\",
            "    --image-head \\",
            "    --val-manifest \"$val_split_path\" \\",
            "    --data-dir \"$CONTENT_DATA\" \\",
            "    --pristine-zip \"$PRISTINE_ZIP_LOCAL\" \\",
            "    --class-map \"$class_map_path\" \\",
            "    --cache-npz \"$cache_npz_path\" \\",
            "    --out-yaml \"$out_yaml\" \\",
            "    --budgets 0.01 0.02 0.03",
            "",
            "if out_yaml.exists():",
            "    print(\"\\n--- Selected configs/cnn_threshold_phase5.yaml ---\")",
            "    with open(out_yaml, 'r', encoding='utf-8') as f:",
            "        print(f.read())",
        ])
    )

    # Cell 9: Pre-Registered Target Verification on Held-Out Sets
    cells.append(
        make_code_cell([
            "print(\"=== PRE-REGISTERED TARGET VERIFICATION ===\")",
            "eval_p5_script = DOPPEL_ROOT / \"scripts\" / \"eval_phase5.py\"",
            "test_split_path = SYNTH_DIR / \"test\"",
            "",
            "!python3 \"$eval_p5_script\" \\",
            "    --checkpoint \"$best_ckpt\" \\",
            "    --config \"$out_yaml\" \\",
            "    --budget 0.01 \\",
            "    --test-dir \"$test_split_path\" \\",
            "    --pristine-zip \"$PRISTINE_ZIP_LOCAL\" \\",
            "    --class-map \"$class_map_path\" \\",
            "    --out-dir \"$corr_v3_drive_dir\"",
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
