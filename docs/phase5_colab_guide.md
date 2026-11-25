# Phase 5: Platform-Agnostic Training Quickstart (Kaggle and Colab)

This guide documents the procedure for executing the Phase 5 CNN training run on Kaggle GPU (T4 / P100), Google Colab GPU, or a local GPU machine.

---

## 1. Input Archives / Datasets

Upload or attach these three items (no Google Drive mount is required):

1. `data/synthetic_v3.zip` (or extracted folder `synthetic_v3/`)
   - Contains `train/`, `val/`, and `test/` splits in triplet format (`pristine`, `forged`, `unrelated_paste`) with `meta.jsonl`.
   - Ground-truth masks: empty for `pristine` and `unrelated_paste`; union of source and target copies for `forged`.
   - Strictly document-disjoint across splits.

2. `data/pristine_heldout.zip` (or extracted folder `pristine_heldout/`)
   - Contains 4,039 real pristine panels across 177 held-out documents (`pristine_manifest.csv`).
   - Split deterministically (`configs/pristine_dev_split.json`, seed=42) by document into two disjoint halves:
     * **Half A** (88 docs, 1,969 panels): used solely during threshold calibration (`select_threshold.py --dev-half A`).
     * **Half B** (89 docs, 2,070 panels): held out strictly for Target 1 evaluation (`eval_phase5.py`).

3. `data/doppel_code.zip`
   - Built with `bash scripts/make_colab_code_zip.sh`.
   - Contains `detector/`, `configs/` (including `pristine_dev_split.json` and `v3_panel_classes.json`), `scripts/` (`train_cnn.py`, `select_threshold.py`, `eval_phase5.py`, `split_pristine_dev_halves.py`), and `docs/phase5_targets.md`.

---

## 2. Platform-Agnostic Notebook Workflow (`notebooks/phase5_train.ipynb`)

Open `notebooks/phase5_train.ipynb` on Kaggle or Colab with GPU enabled.

### Cell 1: Single Top Configuration Cell
- Auto-detects `PLATFORM` (`kaggle`, `colab`, or `local`).
- Sets all paths in one place:
  * **Kaggle defaults:** `/kaggle/input/doppel-code/...`, `/kaggle/input/doppel-synthetic-v3/synthetic_v3`, `/kaggle/input/doppel-pristine-heldout/pristine_heldout.zip`, output to `/kaggle/working/runs_phase5/corr_v3`.
  * **Colab defaults:** `/content/doppel_code.zip`, `/content/synthetic_v3.zip`, `/content/pristine_heldout.zip`, output to `/content/runs_phase5/corr_v3`. No Google Drive mount is required.
- Configures `EPOCHS = 25`, `SAVE_EVERY = 1`, and `RESUME_MODE = "auto"`.

### Cell 2: Environment & Source Resolution
- Verifies CUDA GPU availability and resolves both `.zip` archives and auto-extracted Kaggle `/kaggle/input` directories cleanly.

### Cell 3: Dataset & Dev Split Verification
- Instantiates `ColabDoppelDataset`, asserts train/val document disjointness, verifies empty masks for `pristine` / `unrelated_paste` and union masks for `forged`, and verifies zero overlap between Half A and Half B in `configs/pristine_dev_split.json`.

### Cell 4: 2-Epoch Smoke Test
- Validates training, per-epoch checkpointing (`--save-every 1`), and `val_recall_budget` logging in `train_log.csv`.

### Cell 5: Full Training Run (Save Every Epoch + Auto-Resume)
- Saves `epoch_01.pt` through `epoch_25.pt`, `last.pt`, and `best.pt` every epoch.
- Automatically resumes from the highest-epoch checkpoint in `OUT_DIR` if interrupted before epoch 25.
- **Checkpoint Selection Rule:** `best.pt` is chosen by **validation recall on forged panels at a 1.0% FPR budget over the 600 combined negatives (pristine + unrelated_paste)**, NOT by loss.
- Logs per-epoch `val_recall_budget` (plus `val_synth_clean_far`, `val_pristine_clean_far`, `val_unrelated_paste_far`) in `train_log.csv`.

### Cell 6: Progression Visualization
- Displays recent epochs from `train_log.csv` and plots train loss, `val_forged_f1` / `val_roc_auc`, and `val_recall_budget * 100`.

### Cell 7: Threshold Calibration (Half A ONLY)
- Runs `select_threshold.py` on `synthetic_v3/val` and **Half A** (`--dev-split-json configs/pristine_dev_split.json --dev-half A`) to produce `cnn_threshold_phase5.yaml`.

### Cell 8: Target Verification (Half B + `synthetic_v3/test`)
- Runs `eval_phase5.py`:
  * **Target 1:** Measured strictly on disjoint **Half B** (`<= 2.0%` FAR), reported both at the Half-A-calibrated threshold and at the val-only threshold stored inside `best.pt`.
  * **Targets 2 & 3:** Measured on `synthetic_v3/test` (`unrelated_paste` FPR `<= 1.5x` pristine FPR and `<= 3.0%` absolute; `forged` recall `>= 30.0%`).
  * Real BioFors IDD test split remains completely untouched.
