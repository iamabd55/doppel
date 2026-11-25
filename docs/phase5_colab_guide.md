# Phase 5: Google Colab Training Quickstart and Run Guide

This guide documents the procedure for executing the Phase 5 CNN training run on Google Colab T4 GPU.

---

## 1. Package Preparation and Google Drive Upload

The following three archives must be uploaded to your Google Drive under the directory `My Drive/doppel/`:

1. `data/synthetic_v3.zip` (723 MB)
   - Contains `train/`, `val/`, and `test/` splits.
   - Formatted as triplets (`pristine`, `forged`, `unrelated_paste`) with `meta.jsonl`.
   - Ground truth masks: empty for pristine and unrelated_paste; union of source and target copies for forged.
   - Strictly document-disjoint across splits.

2. `data/pristine_heldout.zip` (174 MB)
   - Contains 4,039 real pristine panels across 177 held-out documents (`pristine_manifest.csv`).
   - Programmatically verified to have zero document overlap with `synthetic_v3/train`.
   - Serves as the Real Dev set for verifying Target 1 (clean FAR <= 2.0%).

3. `data/doppel_code.zip`
   - Contains code (`detector/`), scripts (`scripts/train_cnn.py`, `scripts/select_threshold.py`, `scripts/eval_cnn.py`, `scripts/eval_phase5.py`, `scripts/seam_test_a3.py`, `scripts/verify_real_dev_b4.py`), configuration files (including `configs/v3_panel_classes.json`), and target documentation (`docs/phase5_targets.md`).
   - Generated with `bash scripts/make_colab_code_zip.sh`.

---

## 2. Notebook Workflow (`notebooks/phase5_train.ipynb`)

Open `notebooks/phase5_train.ipynb` in Google Colab and set Runtime to **T4 GPU**.

The notebook executes ONE training run following the strict pre-registered protocol:

### Step 1: Environment & GPU Check (Cell 1)
- Verifies Python, PyTorch, and CUDA availability with `nvidia-smi`.

### Step 2: Mount Drive & Directories (Cell 2)
- Mounts Google Drive at `/content/drive`.
- Sets up sync directories: `/content/drive/MyDrive/doppel/runs_phase5/corr_v3`.

### Step 3: Fast Local Extraction (Cell 3)
- Extracts `doppel_code.zip` into `/content/doppel`.
- Extracts `synthetic_v3.zip` into `/content/data/synthetic_v3`.
- Copies `pristine_heldout.zip` into `/content/data/pristine_heldout.zip`.

### Step 4: Dataset Sanity Verification (Cell 4)
- Instantiates `ColabDoppelDataset` on train and val splits.
- Asserts strict document-level disjointness between train (284 docs) and val (34 docs).
- Confirms empty masks and label 0.0 for all `pristine` and `unrelated_paste` samples.
- Confirms non-empty masks and label 1.0 for all `forged` samples.

### Step 5: Smoke Test (Cell 5)
- Executes a 2-epoch smoke test on a 64-sample subset to ensure the pipeline runs cleanly.

### Step 6: Full Single Training Run (Cell 6)
- **Model:** ResNetCorrNet v2 (`--image-head`, stride-8 self-correlation, top-4 similarity map).
- **Epochs:** 25 epochs.
- **Optimization:** AdamW (lr=3e-4, weight_decay=1e-4), cosine annealing, PyTorch AMP.
- **Validation Evaluation:** Evaluates after each epoch on `data/synthetic_v3/val` (300 forged, 300 pristine, 300 unrelated_paste).
- **Checkpoint Selection Rule:** `best.pt` is selected by **validation recall on forged panels at a 1.0% False Positive Rate budget over the 600 combined negatives (pristine plus unrelated_paste)**, NOT by loss.
- **Drive Synchronization:** Syncs `last.pt`, `best.pt`, and `train_log.csv` to Google Drive after each epoch.

### Step 7: Progression Visualization (Cell 7)
- Plots training loss, validation pixel metrics (F1 and ROC-AUC), and validation recall at the 1% negative FPR budget.

### Step 8: Threshold Calibration (Cell 8)
- Sweeps image-head threshold `s`, pixel threshold `tau`, and `min_area` on validation and held-out pristine panels across budgets (1%, 2%, 3%).
- Writes `configs/cnn_threshold_phase5.yaml`.

### Step 9: Target Verification on Held-Out Sets (Cell 9)
- Runs `scripts/eval_phase5.py` against `best.pt` and `cnn_threshold_phase5.yaml`.
- Evaluates Target 1: Real Pristine Dev FAR <= 2.0% on the 177 held-out documents (reporting both calibrated and val-only operating points).
- Evaluates Target 2: Synthetic v3 test `unrelated_paste` FPR <= 1.5x pristine FPR and <= 3.0% absolute.
- Evaluates Target 3: Synthetic v3 test forged recall >= 30.0% at the operating threshold.
- Explicitly holds out the locked real BioFors IDD test split.

---

## 3. Strict Integrity Guarantees

1. Zero document leakage between training documents and held-out test/dev sets.
2. The locked real BioFors IDD test set (`data/biofors/splits/test_idd_manifest.csv`) is never loaded, evaluated, or touched during this run.
3. No em dashes appear in code, markdown, logs, or comments.
