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
| Part A | Standardized fusion module | COMPLETED | 2026-11-23 | `detector/fusion.py` implementing exact tier rules (high, medium, low, none) with unit tests (8 tests). |
| Part B | Phase 4 Model and Threshold Freeze | COMPLETED | 2026-11-23 | Tagged `phase4-frozen`. Logged checkpoint SHA256, threshold yaml, and fusion rules. |
| Part C | Dry run on 50 pristine train panels | PENDING | - | Sanity check on CPU before real evaluation. |
| Part D | One-shot real BioFors IDD test evaluation | PENDING | - | Final touch on real test split (91 manipulated, 2,195 clean). |

## Phase 4 Freeze Specifications (Tag: phase4-frozen)

### 1. Model Checkpoint Provenance
- **Checkpoint Path**: `checkpoints/corr_v2/last.pt`
- **Model Architecture**: ResNetCorrNet v2 with image classification head (`ResNetCorrNet(backbone='resnet18', proj_dim=128, top_k=4, image_head=True)`)
- **Trained Epochs**: 25 epochs on Google Colab (T4 GPU)
- **Checkpoint SHA256**: `0f463d51823f8394f93a2f5a5e84e4afeb7554f2e078da9db7a398e310527b60`

### 2. Multi-Detector Fusion Decision Rules (`detector/fusion.py`)
- **High Confidence**: 2 or more of (adapted DCT, frozen SIFT, CNN) flag the panel.
- **Medium Confidence**: CNN only flags the panel (Adapted DCT and Frozen SIFT are silent).
- **Low Confidence**: Exactly one of Adapted DCT or Frozen SIFT flags the panel, and CNN is silent.
- **None**: No detector flags the panel.

### 3. Locked Operating Thresholds Configuration (`configs/cnn_threshold.yaml`)

```yaml
# Phase 4 CNN locked operating thresholds
# Selected on synthetic validation split + 4,039 real pristine held-out train panels
# Selection date: 2026-11-23 16:42:41
# Strict protocol: BioFors test split and synthetic test split were NEVER accessed.
# Note: Image class (Blot/Gel vs Microscopy) is assumed known at test time.
# BioFors provides this metadata; in the production pipeline, an upstream panel
# classifier supplies the domain tag before routing to detector thresholds.

model_type: "corr"
checkpoint_file: "last.pt"
primary_budget: 0.01

# Root-level keys represent overall operating point at 1.0% budget for backward compatibility
image_threshold: 0.99
pixel_threshold: 0.699999988079071
min_area_pixels: 25

overall:
  image_threshold: 0.99
  pixel_threshold: 0.699999988079071
  min_area_pixels: 25
  metrics:
    validation_recall: 0.514
    validation_recall_ci: [0.4703, 0.5575]
    synthetic_val_clean_far: 0.01
    synthetic_val_clean_far_ci: [0.004279, 0.023193]
    real_pristine_heldout_far: 0.006932
    real_pristine_heldout_far_ci: [0.004801, 0.010001]
    budget_met: true

per_class:
  blot_gel:
    image_threshold: 0.95
    pixel_threshold: 0.30000001192092896
    min_area_pixels: 25
    metrics:
      validation_recall: 0.6865
      validation_recall_ci: [0.6268, 0.7406]
      synthetic_val_clean_far: 0.007937
      synthetic_val_clean_far_ci: [0.002179, 0.02847]
      real_pristine_heldout_far: 0.00861
      real_pristine_heldout_far_ci: [0.005924, 0.012498]
      budget_met: true
  microscopy:
    image_threshold: 0.99
    pixel_threshold: 0.30000001192092896
    min_area_pixels: 1600
    metrics:
      validation_recall: 0.2379
      validation_recall_ci: [0.1892, 0.2946]
      synthetic_val_clean_far: 0.012097
      synthetic_val_clean_far_ci: [0.004122, 0.034956]
      real_pristine_heldout_far: 0.005537
      real_pristine_heldout_far_ci: [0.002367, 0.012896]
      budget_met: false

budgets:
  budget_1_0pct:
    target_far: 0.01
    overall:
      image_threshold: 0.99
      pixel_threshold: 0.699999988079071
      min_area_pixels: 25
      validation_recall: 0.514
      validation_recall_ci: [0.4703, 0.5575]
      synthetic_val_clean_far: 0.01
      synthetic_val_clean_far_ci: [0.004279, 0.023193]
      real_pristine_heldout_far: 0.006932
      real_pristine_heldout_far_ci: [0.004801, 0.010001]
      budget_met: true
    blot_gel:
      image_threshold: 0.95
      pixel_threshold: 0.30000001192092896
      min_area_pixels: 25
      validation_recall: 0.6865
      validation_recall_ci: [0.6268, 0.7406]
      synthetic_val_clean_far: 0.007937
      synthetic_val_clean_far_ci: [0.002179, 0.02847]
      real_pristine_heldout_far: 0.00861
      real_pristine_heldout_far_ci: [0.005924, 0.012498]
      budget_met: true
    microscopy:
      image_threshold: 0.99
      pixel_threshold: 0.30000001192092896
      min_area_pixels: 1600
      validation_recall: 0.2379
      validation_recall_ci: [0.1892, 0.2946]
      synthetic_val_clean_far: 0.012097
      synthetic_val_clean_far_ci: [0.004122, 0.034956]
      real_pristine_heldout_far: 0.005537
      real_pristine_heldout_far_ci: [0.002367, 0.012896]
      budget_met: false
  budget_1_5pct:
    target_far: 0.015
    overall:
      image_threshold: 0.95
      pixel_threshold: 0.4000000059604645
      min_area_pixels: 400
      validation_recall: 0.56
      validation_recall_ci: [0.5162, 0.6029]
      synthetic_val_clean_far: 0.014
      synthetic_val_clean_far_ci: [0.006798, 0.028613]
      real_pristine_heldout_far: 0.009903
      real_pristine_heldout_far_ci: [0.007281, 0.013457]
      budget_met: true
    blot_gel:
      image_threshold: 0.8
      pixel_threshold: 0.800000011920929
      min_area_pixels: 0
      validation_recall: 0.7341
      validation_recall_ci: [0.6764, 0.7849]
      synthetic_val_clean_far: 0.011905
      synthetic_val_clean_far_ci: [0.004057, 0.03441]
      real_pristine_heldout_far: 0.014349
      real_pristine_heldout_far_ci: [0.010742, 0.019146]
      budget_met: true
    microscopy:
      image_threshold: 0.99
      pixel_threshold: 0.30000001192092896
      min_area_pixels: 1600
      validation_recall: 0.2379
      validation_recall_ci: [0.1892, 0.2946]
      synthetic_val_clean_far: 0.012097
      synthetic_val_clean_far_ci: [0.004122, 0.034956]
      real_pristine_heldout_far: 0.005537
      real_pristine_heldout_far_ci: [0.002367, 0.012896]
      budget_met: true
  budget_2_0pct:
    target_far: 0.02
    overall:
      image_threshold: 0.95
      pixel_threshold: 0.5
      min_area_pixels: 0
      validation_recall: 0.574
      validation_recall_ci: [0.5303, 0.6166]
      synthetic_val_clean_far: 0.018
      synthetic_val_clean_far_ci: [0.009498, 0.033852]
      real_pristine_heldout_far: 0.010646
      real_pristine_heldout_far_ci: [0.007914, 0.014309]
      budget_met: true
    blot_gel:
      image_threshold: 0.5
      pixel_threshold: 0.699999988079071
      min_area_pixels: 150
      validation_recall: 0.7659
      validation_recall_ci: [0.7098, 0.8139]
      synthetic_val_clean_far: 0.011905
      synthetic_val_clean_far_ci: [0.004057, 0.03441]
      real_pristine_heldout_far: 0.019452
      real_pristine_heldout_far_ci: [0.015173, 0.024906]
      budget_met: true
    microscopy:
      image_threshold: 0.99
      pixel_threshold: 0.699999988079071
      min_area_pixels: 200
      validation_recall: 0.379
      validation_recall_ci: [0.3209, 0.4408]
      synthetic_val_clean_far: 0.016129
      synthetic_val_clean_far_ci: [0.00629, 0.04073]
      real_pristine_heldout_far: 0.007752
      real_pristine_heldout_far_ci: [0.00376, 0.015914]
      budget_met: true
  budget_3_0pct:
    target_far: 0.03
    overall:
      image_threshold: 0.9
      pixel_threshold: 0.5
      min_area_pixels: 0
      validation_recall: 0.606
      validation_recall_ci: [0.5625, 0.6479]
      synthetic_val_clean_far: 0.028
      synthetic_val_clean_far_ci: [0.016751, 0.046446]
      real_pristine_heldout_far: 0.014112
      real_pristine_heldout_far_ci: [0.010909, 0.018239]
      budget_met: true
    blot_gel:
      image_threshold: 0.0
      pixel_threshold: 0.8500000238418579
      min_area_pixels: 25
      validation_recall: 0.8056
      validation_recall_ci: [0.7523, 0.8497]
      synthetic_val_clean_far: 0.019841
      synthetic_val_clean_far_ci: [0.008504, 0.045598]
      real_pristine_heldout_far: 0.029974
      real_pristine_heldout_far_ci: [0.024558, 0.036542]
      budget_met: true
    microscopy:
      image_threshold: 0.95
      pixel_threshold: 0.4000000059604645
      min_area_pixels: 400
      validation_recall: 0.4597
      validation_recall_ci: [0.3987, 0.5218]
      synthetic_val_clean_far: 0.020161
      synthetic_val_clean_far_ci: [0.008642, 0.046319]
      real_pristine_heldout_far: 0.016611
      real_pristine_heldout_far_ci: [0.010092, 0.027226]
      budget_met: true
```
