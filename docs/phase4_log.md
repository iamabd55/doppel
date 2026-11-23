# Phase 4: CNN Training and Evaluation Log

## Test Set Access Counter

- **Real BioFors IDD Test Set Touches**: 0 / 1 allowed (strictly locked until Step 7 after thresholds are frozen)
- **Synthetic Test Set Touches**: 1 (evaluated in Step 3 on CPU with locked thresholds from configs/cnn_threshold.yaml; no calibration drift, no overfitting)

## Phase 4 Step Tracker

| Step | Description | Status | Date | Notes |
|---|---|---|---|---|
| Step 0 | Freeze check on SIFT and DCT | COMPLETED | 2026-11-22 | CoMoFoD held-out regression passed (74.0% detection, 0.540 mean F1, 40.0% clean FAR). |
| Step 1 | Colab Assets preparation | COMPLETED | 2026-11-22 | `data/doppel_code.zip` (31 KB) and `data/pristine_heldout.zip` (174 MB, 4,039 panels). |
| Step 2 | CNN Models (ResNetUNet, ResNetCorrNet) | COMPLETED | 2026-11-22 | Verified forward passes, variable image sizes, valid-pixel masking, fp16/autocast, gradient flow. 8/8 tests passed. |
| Step 3 | Training script implementation | COMPLETED | 2026-11-22 | `scripts/train_cnn.py` with AdamW, CosineAnnealingLR, BCE+Dice loss, native full-res validation. |
| Step 4 | Smoke test and CPU overfit verification | COMPLETED | 2026-11-22 | ResNetCorrNet overfit test on subset 64 reached 0.9857 train F1 (target > 0.9) in ~5.5 minutes on CPU. |
| Step A | Fix Dice loss clean twin masking | COMPLETED | 2026-11-22 | Isolated Dice computation to forged samples; clean samples contribute to BCE only and 0 to Dice. 5 unit tests passed. |
| Step B | Validation evaluation and ablation hooks | COMPLETED | 2026-11-22 | Threshold sweep subject to 1% FAR on 1,000 pristine panels and clean twins; `scripts/eval_cnn.py` with slice breakdowns; `--ablate-corr` hook. |
| Step C | Google Colab training notebook | COMPLETED | 2026-11-22 | `notebooks/phase4_train.ipynb` (13 cells) and `scripts/select_threshold.py`. Code archive rebuilt (`data/doppel_code.zip`, 60 KB). |
| Step C.2 | CorrNet v2 & FA Diagnostics | COMPLETED | 2026-11-23 | Partner-guaranteed crops (100% pairs complete), ResNetCorrNet v2 with `--image-head`, lowest max-FAR fallback, extended diagnostic suite, and 15-cell notebook. |
| Step D | Validation threshold selection | COMPLETED | 2026-11-23 | Executed on CPU on synthetic VAL (1,000 images) + all 4,039 pristine panels. Locked configs/cnn_threshold.yaml. |
| Step 2 (Wrap-up) | Inference interface implementation | COMPLETED | 2026-11-23 | `detector/cnn/infer.py` matching DCT/SIFT interface with pad/crop, locked YAML lookup, CPU caching. |
| Step 3 (Wrap-up) | Synthetic test benchmark | COMPLETED | 2026-11-23 | Evaluated 600 synthetic test pairs (300 forged, 300 clean twins). Clean FAR 0.00% [0.00%, 1.26%], Blot/Gel recall 73.0% [65.5%, 79.5%], Microscopy recall 25.0% [18.7%, 32.6%]. Zero calibration drift, zero overfitting. |
| Step E | One-shot final real evaluation | PENDING | - | Touch count 1: evaluate locked models on synthetic test and real BioFors IDD test after user approval. |
