# Changelog

All notable changes to the Doppel repository are documented here.

## [2026-11-19] - Initial Living Documentation and Phase 2 Preparation

### Added
- Created `agentlogs/` directory containing living project documentation.
- Created `agentlogs/architecture.md` detailing the frozen DCT baseline pipeline and target multi-branch architecture.
- Created `agentlogs/files.md` providing an inventory and summary of every file in the repository.
- Created `agentlogs/changelog.md` to track running project changes.

### Verified
- Step 2.0 baseline reproduction: confirmed 36/40 (90%) detection with 0.817 F1 on CoMoFoD translation, and 0/40 false alarms on originals.

### Added (Phase 2, Step 2.1)
- Created `detector/sift.py` with `extract_keypoints` and `draw_keypoints` functions using OpenCV SIFT.
- Created `scripts/test_sift_keypoints.py` to extract, visualize, and analyze SIFT keypoint distribution across textured vs flat regions.

### Added (Phase 2, Step 2.2)
- Added `extract_descriptors` and `detect_and_compute` to `detector/sift.py` to produce 128-d floating point descriptors for detected keypoints.
- Created `scripts/test_sift_descriptors.py` to verify descriptor extraction, dimensionality, and normalization properties.
- Updated `DOPPEL_AGENT.md` current step to Step 2.2.

### Added (Phase 2, Step 2.3)
- Implemented `match_within_image` in `detector/sift.py` using kNN (k=3), dropping self-matches, applying Lowe's ratio test, filtering by spatial distance, and deduplicating pairs.
- Added `draw_matches` in `detector/sift.py` to render connected match lines and keypoint endpoints on single images.
- Created `scripts/test_sift_matching.py` to test within-image matching across translation, rotation, scaling, and original images.
- Updated `DOPPEL_AGENT.md` current step to Step 2.3.

### Added (Phase 2, Step 2.4)
- Implemented `verify_matches_ransac` in `detector/sift.py` combining agglomerative hierarchical clustering with full 6-parameter RANSAC affine estimation (`cv2.estimateAffine2D`).
- Created `scripts/test_sift_ransac.py` to verify inlier recovery and spurious match pruning across translation, rotation, scaling, and originals (updated rotation sample to 062_F.png, labeled untouched original test explicitly).
- Created `scripts/diagnose_rotation.py` to diagnose match counts and ground-truth alignment on rotation set 045.
- Created `scripts/inspect_rotation_category.py` to evaluate ground truth keypoint density, inlier counts, and rotation angles across multiple rotation samples (041-050).
- Updated `DOPPEL_AGENT.md` current step to Step 2.4.

### Added (Phase 2, Step 2.5)
- Implemented `create_sift_mask` and end-to-end `detect_sift` in `detector/sift.py` to convert inliers into solid binary masks using convex hulls and keypoint dilation.
- Created `scripts/test_sift_localization.py` to measure precision, recall, and F1 against ground truth mask M and produce tri-panel comparisons.
- Updated `DOPPEL_AGENT.md` current step to Step 2.5.

### Added (Phase 2, Step 2.6)
- Implemented `map_flip_coordinates` and `match_flipped` to support reflection/mirrored clone detection via a horizontal flip pass.
- Updated `detect_sift` with `check_flip=True` to run both standard within-image and flip passes seamlessly.
- Created `tests/test_flip.py` with unit tests for lossless coordinate re-mapping and mirrored forgery detection.
- Created `scripts/test_sift_flip.py` to demonstrate and visualize mirrored copy-move detection.
- Added `pytest` to `pyproject.toml` dev dependencies.
- Updated `DOPPEL_AGENT.md` current step to Step 2.6.

### Refactored (Phase 2, Code Structure)
- Modularized monolithic `detector/sift.py` into dedicated subpackage `detector/sift/`:
  - `detector/sift/features.py`: Keypoints, descriptors, and rich keypoint drawing.
  - `detector/sift/matching.py`: Within-image and horizontal flip descriptor matching.
  - `detector/sift/verification.py`: Hierarchical clustering and RANSAC affine verification.
  - `detector/sift/localization.py`: Convex hull mask creation and end-to-end pipeline runner.
  - `detector/sift/__init__.py`: Public API export preserving all existing function signatures.
- Replaced `detector/sift.py` with backward-compatible re-exports.

### Added (Phase 2, Step 2.7)
- Created `scripts/batch_test_sift.py` to benchmark SIFT across all 5 CoMoFoD categories (200 forged images) and 40 originals.
- Automated generation of `results/phase2_sift/sift_benchmark.csv` and side-by-side DCT vs SIFT comparison in `results/phase2_sift/benchmark_summary.md`.
- Updated `DOPPEL_AGENT.md` current step to Step 2.7.

### Fixed (Phase 2, Localization & Tuning)
- Fixed convex hull localization in `detector/sift/localization.py`: inlier points are now clustered spatially with convex hulls filled per local cluster, eliminating image-wide polygon bloat.
- Reduced mask dilation radius from 20 px to 12 px to tighten boundary precision.
- Tuned default ratio threshold from 0.60 to 0.70 to improve matching across rotated and scaled keypoints.
- Increased RANSAC spatial cluster distance to 100.0 px and added direct RANSAC comparison in `detector/sift/verification.py` to prevent keypoint fragmentation.

### Added (Phase 2, Calibration and Ablation Study)
- Implemented `filter_by_orientation` in `detector/sift/matching.py` with circular distance consensus filtering to reject spurious matches that lack orientation agreement.
- Added keypoint angle extraction and flip angle re-mapping `(180 - theta) % 360` to `match_flipped`.
- Enhanced `verify_matches_ransac` in `detector/sift/verification.py` with `min_inlier_ratio` filtering to prune weak coincidences.
- Updated `detect_sift` in `detector/sift/localization.py` with configurable `min_inlier_ratio`, separate `min_inliers_flip`, `filter_orientation`, and conservative localization.
- Created `scripts/ablate_sift.py` to benchmark 9 systematically varied parameter configurations on the 50/50 tuning split (100 forged images + 20 clean originals) without data leakage onto held-out test data.

### Completed (Phase 2, Step 2.8: Failure Analysis & Baseline Freeze)
- Evaluated `cfg8_best_combo` on the 100-image held-out test split (74.0% detection, 0.540 mean F1, 40.0% clean false alarm rate across 20 images).
- Authored `results/phase2_sift/failure_analysis.md` documenting keypoint sparsity on flat patches, inlier sensitivity on small clones, repetitive architecture false alarms, and the natural-to-scientific dataset gap.
- Froze `cfg8_best_combo` as the default SIFT configuration in `detector/sift/localization.py`.
- Officially concluded Phase 2; locked SIFT branch as an expert component for Phase 5 hybrid fusion.

### Measured (Phase 2, Held-Out Evaluation)
- Evaluated `cfg8_best_combo` once on the held-out split (100 forged images + 20 clean originals).
- Recorded results in `results/phase2_sift/ablation_held_out_split.csv` and `results/phase2_sift/ablation_held_out_split.md`.
- Result: 74/100 forged images detected at F1 > 0.3, mean F1 0.540; category detection was translation 14/20, rotation 14/20, scaling 15/20, distortion 16/20, combination 15/20.
- Clean-image false alarms were 8/20 (40.0%), mean pixel FAR 1.7554%, maximum false-positive area 31,129 pixels. These results show improved forged-image detection over tuning, but false alarms remain substantial.
- Updated current step to Phase 2, Step 2.8 failure analysis.

## [Phase 3: Synthetic Benchmark & Training Data Generator]

### Added (Phase 3, Step 3.0: Data Audit)
- Created `scripts/audit_biofors.py` to audit downloaded BioFors images against official annotations.
- Verified 696 train documents (19,105 Blot/Gel, 10,458 Microscopy images) and 54 IDD test documents (2,286 panels).
- Confirmed strict zero document overlap between train and test pools.
- Analyzed image resolution distributions: revealed narrow horizontal blot strips (median height 58 px, min 19 px), proving blind downscaling/resizing must be avoided.

### Added (Phase 3, Step 3.1: Config & Splits Manifest)
- Created `configs/synthetic.yaml` to centrally govern all Phase 3 generation parameters, transform distributions, and frozen SIFT baseline values.
- Created `scripts/create_splits.py` to partition the 696 train documents into 85% Train (592 documents) and 15% Validation (104 documents) by document ID.
- Generated `data/biofors/splits/train_manifest.csv`, `data/biofors/splits/val_manifest.csv`, and `data/biofors/splits/test_idd_manifest.csv`.

### Added (Phase 3, Step 3.2: Generator, Unit Tests, and Visualizer)
- Created `detector/synthetic.py`:
  - `apply_patch_augmentation`: supports H/V flips, cardinal rotations, continuous arbitrary rotations with canvas expansion, and scaling.
  - `apply_patch_degradations`: applies blur, noise, gamma adjustments, JPEG recompression, and edge feathering.
  - `SyntheticCopyMoveGenerator`: native resolution synthesis, texture-aware donor sampling with 10% hard low-texture cases, transform distribution (60% translation, 20% cardinal/flip, 15% arbitrary, 5% combo), and 1:1 clean negative generation.
  - Produces complete BioFors IDD format: `_F` (forged/negative), `_M` (target mask), `_S` (source mask), `_A` (union mask), and `_meta.json`.
- Created `tests/test_synthetic.py`: automated pytest suite verifying augmentation invariants, mask union identity (`_A == _M | _S`), clean negative empty masks, exact determinism across identical seeds, thin Western blot strip handling, and zero document leakage.
- Created `scripts/make_synthetic.py`: batch generation tool reading `configs/synthetic.yaml` to produce train, validation, and test synthetic sets with split manifests.
- Created `scripts/visualize_synthetic.py`: inspection tool generating multi-column visual grids (`figures/synthetic_samples_grid.png`) with bounding box overlays and metadata annotations.
- Fixed pytest package discovery by adding `sys.path` insertion to `tests/test_synthetic.py` and adding `[tool.pytest.ini_options] pythonpath = ["."]` in `pyproject.toml`.
- Generated full Phase 3 synthetic dataset: 2,600 samples (2,000 train, 300 validation, 300 test) with 1:1 clean negative controls and consolidated manifests.

### Added (Phase 3, Step 3.3: Scientific Benchmark Runner)
- Created `scripts/benchmark.py`:
  - Evaluates frozen DCT baseline (`Q=8, K=6, min_votes=150`) and frozen SIFT baseline (`cfg8_best_combo`).
  - Implements official BioFors IDD metrics: pooled pixel precision/recall/F1/MCC and image-level TP/FP/TN/FN/F1/MCC and clean false alarm rate.
  - Supports dual benchmarking on both synthetic scientific test set (300 images) and real BioFors IDD test set (2,286 panels).
  - Generates `synthetic_benchmark.csv`, `real_biofors_benchmark.csv`, and `benchmark_summary.md`.
- Fixed `detect_sift` in `detector/sift/localization.py` to automatically convert 3-channel inputs to grayscale, and updated unpacking in `scripts/benchmark.py`.
- Benchmark evaluation completed:
  - Real BioFors IDD (2,286 panels: 91 forged, 2,195 clean):
    - DCT: 14/91 (15.4%) forged detection, only 8/2,195 (0.36%) clean false alarms.
    - SIFT: 33/91 (36.3%) forged detection, 49/2,195 (2.23%) clean false alarms.
    - SIFT achieved 13/17 (76.5%) detection on real microscopy figures (where DCT only caught 1/17 = 5.9%).
  - Synthetic Test (300 images: 150 forged, 150 clean):
    - DCT: 20/150 (13.3%) detection, 1/150 (0.7%) false alarm. Caught 20/87 (23.0%) translations, 0/63 on non-rigid transforms.
    - SIFT: 31/150 (20.7%) detection, 6/150 (4.0%) false alarm. Successfully detected rotations, flips, and combinations where DCT failed.

### Added (Phase 3, Diagnostic Audit & Translation Investigation)
- Created `scripts/diagnose_translations.py` to investigate low translation recall:
  - Generates 100 zero-degradation clean translation samples on scientific base images.
  - Tests frozen DCT vs patch-size-adapted DCT across patch size splits (<500 px vs >=500 px).
  - Tests SIFT and computes Wilson score 95% confidence intervals.
  - Audits synthetic training set for global statistical shortcuts and zero document leakage.

### Refactored (Phase 3, Paired-Twin Negative Generation & Degradation Tiers)
- Refactored `detector/synthetic.py` and `scripts/make_synthetic.py` with Paired-Twin architecture:
  - Clean negatives are now paired twins sharing the exact same base crop, class, dimensions, doc_id, and whole-image degradation pipeline as the forged twin (no copy-paste).
  - Implemented 4 concrete degradation tiers ('zero', 'light', 'moderate', 'heavy') distributed evenly at 25% each.
  - Recorded degradation level, patch area, and pair ID in JSON metadata and manifests.
- Created `scripts/audit_dataset_shortcuts.py`:
  - Trains Logistic Regression and HistGradientBoosting on 6 global statistical features (mean, std, Laplacian variance, width, height, file size).
  - Evaluates via 5-fold cross-validation strictly grouped by document ID (doc_id) to verify near 50% chance-level performance.
- Created `scripts/tune_adapted_dct.py`:
  - Evaluates parameter grid on synthetic validation split only (`val_manifest.csv`) to tune adapted DCT without test set leakage.
- Updated `scripts/benchmark.py`:
  - Enforces ground-truth overlap (`pix_tp > 0`) for image-level detection.
  - Computes Wilson score 95% confidence intervals.
  - Reports recall stratified by degradation tier and patch area bucket.

### Added (Phase 3, Step A & B Implementation)
- Locked adapted DCT configuration in `configs/dct_adapted.yaml` (`min_votes=75, min_size=64`) chosen strictly on synthetic VAL split.
- Updated `scripts/audit_dataset_shortcuts.py` to evaluate all 6 features, 5 features without file_size, and individual features alone to verify none exceeds 55% chance threshold.
- Created `scripts/visualize_twins.py` generating a 12-pair QA grid (`results/phase3_synthetic/twin_grid.png`) covering all 4 degradation tiers and confirming identical base crops, dimensions, and tiers.
- Enhanced `scripts/benchmark.py` to evaluate Frozen DCT, Adapted DCT, and Frozen SIFT side-by-side with overlap vs any-flagged rules, Wilson 95% CIs, synthetic breakdowns (tier, patch area, transform type), and a 1,000-sample pristine train check.

### Added (Phase 3, Step C: Diagnostic Audits, Stratified Scaling, and Packaging Prep)
- Executed diagnostic audits in `scripts/pre_step_c_checks.py`:
  - Discovered SIFT tier anomaly root cause: class and degradation tier were deterministically phase-locked by modular loop counter (`pair_idx % 2` vs `pair_idx % 4`).
  - Identified patch-size realism gap: 40.6% of synthetic patches were below real BioFors IDD Q1 (1,593 px).
  - Evaluated Real BioFors IDD fusion: Adapted DCT OR SIFT achieved 48.4% recall at 3.78% clean FAR; AND fusion dropped recall to 12.1%.
- Refactored `detector/synthetic.py` and `scripts/make_synthetic.py`:
  - Stratified class x degradation tier sampling with seeded RNG: equal sample counts across all 8 cells (chi-square p = 1.0000).
  - Enforced 400 px floor for donor patch area with log-uniform area sampling up to 25% of image area.
  - Implemented multi-copy forgery generation: ~20% of forged samples receive 2 to 3 targets from one source, including adjacent-band repeats for Western blots.
  - Base images sampled without replacement, capping single document share at 1% of split (80 pairs max for train).
  - Enforced 100% exclusion of the 177 held-out pristine BioFors train documents from the generator pool (`configs/held_out_pristine_docs.json`).
- Generated scaled synthetic dataset:
  - 16,000 train samples (8,000 paired twins across 412 unique documents).
  - 1,000 validation samples (500 paired twins across 59 unique documents).
  - 600 synthetic test samples (300 paired twins across 42 unique documents).
- Re-ran shortcut classification audit:
  - Logistic Regression accuracy: 50.57% (all 6 features), 50.59% (file size dropped).
  - HistGradientBoosting accuracy: 51.78% (all 6 features), 51.67% (file size dropped).
  - Individual non-file-size features: mean 50.02%, std 50.08%, laplacian 50.50%, width 50.00%, height 50.00%. All <= 51.3% (below 55% threshold).
  - Chi-square independence test: chi2 = 0.0000, p = 1.0000 (perfect stratification).
- Re-ran synthetic test benchmark (600 samples):
  - Zero tier recall: Adapted DCT 70.7% (Blot/Gel 78.9%, Microscopy 62.2%), SIFT 18.7% (Blot/Gel 2.6%, Microscopy 35.1%).
  - Degrades gracefully across tiers: DCT recall 70.7% (zero) -> 44.0% (light) -> 33.3% (moderate) -> 5.3% (heavy).
  - Every single (class x tier) cell has N >= 37 >= 30 (zero cells with low statistical confidence).
- Created `docs/phase4_targets.md` pre-registering CNN alone recall > 35.2% at clean FAR <= 1.50%, hybrid fusion recall > 41.8% at clean FAR <= 2.50%, and pixel F1 > 0.1200.
- Implemented `detector/colab_dataset.py` supporting native variable image sizes (zero-padding / positive-aware random crops), joint geometric augmentations, and split disjointness verification.
- Authored `notebooks/colab_setup.md` guiding Colab GPU training setup and dataset sanity checks.

## [Phase 4: Learned-Feature CNN Detection Branch]

### Added (Phase 4, Step 0: Freeze Verification)
- Verified SIFT defensive conversion in `detector/sift/localization.py`.
- Ran CoMoFoD held-out regression benchmark on `cfg8_best_combo`: 74/100 (74.0%) detection, 0.540 mean F1, 8/20 (40.0%) clean original false alarms. Zero regression observed.
- Verified git object tracking and confirmed data directories and 761 MB dataset zip are properly gitignored.

### Added (Phase 4, Step 1: Colab Archive Packaging)
- Created `scripts/make_colab_code_zip.sh` archiving core detection code, tests, configs, and training scripts into `data/doppel_code.zip` (31 KB).
- Created `scripts/make_pristine_heldout_zip.py` archiving 4,039 pristine panels from 177 held-out train documents into `data/pristine_heldout.zip` (174 MB) with metadata manifest. Verified zero overlap with test documents.

### Added (Phase 4, Step 2: CNN Model Architectures)
- Created `detector/cnn/unet.py` implementing baseline ResNet-18 U-Net segmentation network with ImageNet pretraining, multi-scale skip connections, and native-resolution mask logits.
- Created `detector/cnn/corr_net.py` implementing ResNet-18 copy-move correlation network with stride-8 self-correlation module, 128-d projection, Chebyshev spatial self-masking (<= 1 cell), zero-pad valid-mask filtering, top-4 match extraction, and multi-scale skip decoder.
- Created `tests/test_cnn_models.py` with 8 unit tests validating output shapes, narrow biological aspect ratios, valid-mask boundary handling, gradient backpropagation, and AMP fp16 autocast compatibility.
- Updated `detector/colab_dataset.py` with `valid_mask` generation and deterministic crop offsets for evaluation and overfit testing.

### Added (Phase 4, Step 3: PyTorch Training Script)
- Created `scripts/train_cnn.py` implementing standalone model training:
  - Supports `--model unet` and `--model corr`.
  - Configurable loss function: Binary Cross-Entropy with pos_weight plus soft Dice loss. Clean twins included as all-zero targets.
  - AdamW optimizer with CosineAnnealingLR scheduler, AMP mixed-precision, and gradient norm clipping at 1.0.
  - Full-resolution validation at native aspect ratios (padding only, no resizing) evaluating pixel F1, MCC, ROC AUC, synthetic clean FAR, and real pristine held-out FAR.
  - Checkpoint tracking saving `best.pt`, `last.pt`, `config.json` with commit hash, and per-epoch CSV metrics in `train_log.csv`.
  - In-memory RAM dataset caching for rapid small-subset testing.

### Verified (Phase 4, Step 4: Smoke Test and Overfit Verification)
- Executed smoke test on CPU for ResNetUNet (2 epochs, subset 64): training loss dropped from 1.5772 to 1.5012 in 35.3s total.
- Executed smoke test on CPU for ResNetCorrNet (2 epochs, subset 64): training loss dropped from 1.7716 to 1.6933 in 32.7s total.
- Executed 100-epoch overfit verification on CPU for ResNetCorrNet (subset 64, no augmentation, batch size 8, lr 1e-3):
  - Initial (Epoch 1): Train Loss 1.7253, Train F1 0.1587.
  - Midpoint (Epoch 25): Train Loss 1.0323, Train F1 0.7606.
  - Target Cross (Epoch 50): Train Loss 0.7498, Train F1 0.9135 (passed > 0.9 target).
  - Final (Epoch 100): Train Loss 0.6410, Train F1 0.9857.
  - Total overfit training time: ~5.5 minutes on CPU.
- Created `docs/phase4_log.md` to track test set access count (0 touches) and phase milestones.

### Fixed (Phase 4, Step A: Dice Loss Clean Twin Masking)
- Resolved Dice loss distortion on clean negatives where all-zero target masks contributed ~1.0 loss regardless of prediction quality.
- Created `detector/cnn/loss.py` with `soft_dice_loss` and `CombinedBCEDiceLoss`:
  - Computes Dice loss strictly over samples with forged pixels (`targets_flat.sum(dim=1) > 0.5`).
  - Sets Dice loss to 0.0 with zero gradient for batches containing only clean samples.
  - Ensures clean samples contribute to BCE only and never add gradient to the Dice term.
- Updated `scripts/train_cnn.py` to use `CombinedBCEDiceLoss`.
- Created `tests/test_dice_loss.py` with 5 unit tests validating:
  - Mixed batch with perfect predictions yields near-zero Dice loss (< 1e-3).
  - Mixed batch with all-zero predictions yields high Dice loss (> 0.95).
  - Clean-only batch yields exactly 0.0 Dice loss.
  - Clean samples contribute strictly zero gradient to Dice in mixed and clean batches.
  - Combined loss backward pass runs cleanly without NaN.
- All 30 unit tests in repository passed.

### Added (Phase 4, Step B: Validation Evaluation and Ablation Hooks)
- Implemented `detector/cnn/selection.py`:
  - `sweep_thresholds`: Sweeps candidate thresholds to maximize recall subject to synthetic val clean FAR <= 1.0% AND real pristine held-out FAR <= 1.0%.
  - `is_better_checkpoint`: Implements strict priority rules where any epoch meeting the 1% FAR budget beats non-budget epochs, higher recall wins within budget, and ROC AUC serves as fallback.
- Updated `scripts/train_cnn.py`:
  - Fixed seeded subset sampling of 1,000 pristine panels (`seed=42`) for per-epoch validation.
  - Checkpoint selection now selects `best.pt` via `is_better_checkpoint`.
  - Added `--time-budget-min` to cleanly terminate long runs and save `last.pt`.
  - Full resume support (`--resume`) restoring epoch, optimizer, scaler, scheduler, and best checkpoint state.
- Enhanced `detector/cnn/corr_net.py` with `--ablate-corr` support to zero out correlation channels at decoder input for ablation analysis.
- Created `scripts/eval_cnn.py`:
  - Evaluates models on validation split with fine-grained slice breakdowns: class, degradation tier, transform type, copy count, and patch area.
  - Computes overlap recall and any-flagged recall with Wilson 95% confidence intervals, pixel F1/MCC, and clean FAR.
  - Automatically flags slices with low sample size (`N < 30`).
  - Supports `--ablate-corr` and optional CSV export (`--out-csv`).
- Created `tests/test_checkpoint_selection.py` (6 unit tests) and added ablation test in `tests/test_cnn_models.py`. All 37 tests in test suite passed.

### Added (Phase 4, Step C: Google Colab Training Notebook)
- Created `scripts/build_phase4_notebook.py` using `nbformat` to construct `notebooks/phase4_train.ipynb`.
- Notebook contains top README markdown guide and 12 executable cells matching the exact specified workflow:
  1. GPU and environment check (`nvidia-smi`, `torch.cuda.is_available()`, versions).
  2. Google Drive mount and asset directory path definition.
  3. Ephemeral `/content` data copying, unzipping, `sys.path` registration, and file verification.
  4. Dataset loader sanity check (batch shapes, forged fraction, mask coverage).
  5. 2-epoch smoke test on both models (`--subset 64`) with full-run time extrapolation.
  6. 100-epoch overfit test for both models asserting train F1 > 0.90 and consistent Dice loss behavior.
  7. Full training cell for ResNetUNet (25 epochs, batch 16, 256 crops, pos_crop_prob 0.5, AdamW, cosine, AMP, sync to Drive).
  8. Full training cell for ResNetCorrNet with identical settings.
  9. Resumption cell for clean restarts after disconnections.
  10. Comparison plotting cell rendering loss, val F1, AUC, recall at 1% FAR, and pristine FAR across epochs.
  11. Multi-model slice evaluation cell running `eval_cnn.py` across slices for UNet, CorrNet, and ablated CorrNet.
  12. Operating point threshold selection cell invoking `scripts/select_threshold.py`.
- Implemented `scripts/select_threshold.py` (Step D) to optimize `(pixel_threshold, min_area)` over synthetic validation split and 4,039 pristine panels subject to FAR <= 1.0% budgets, writing `configs/cnn_threshold.yaml`.
- Rebuilt `data/doppel_code.zip` (68 KB) with all latest code, configs, tests, and scripts.

### Fixed (Phase 4, Colab Notebook Call Audit & Local Dry-Run Verification)
- Fixed `TypeError: ColabDoppelDataset.__init__() got an unexpected keyword argument 'csv_path'` in notebook Cell 4 by aligning dataset invocation across `scripts/build_phase4_notebook.py`, `scripts/eval_cnn.py`, and `scripts/select_threshold.py` to match the real `manifest_path` and `images_dir` signature.
- Audited all notebook cells (4 to 12) for kwargs, CLI flags, paths, filenames, and manifest names:
  - Added `--drive-sync-dir` support to `scripts/train_cnn.py` copying `last.pt`, `train_log.csv`, `config.json`, and `best.pt` to Drive after EVERY epoch to prevent progress loss on disconnect.
  - Added `--pristine-zip` to validation evaluation calls in Cell 11.
  - Updated CLI parsers to accept optional `args` lists for testability.
- Added `tests/test_notebook_calls.py` with 4 AST and signature/argparse validation tests checking dataset kwargs and CLI commands for `train_cnn.py`, `eval_cnn.py`, and `select_threshold.py`.
- Created `scripts/dryrun_notebook.py` executing notebook code cells 4, 5, and shortened 6 locally in a temporary directory on CPU with `--subset 16`, verifying end-to-end execution and per-epoch Drive synchronization.
- Rebuilt `notebooks/phase4_train.ipynb` and `data/doppel_code.zip` (60 KB) with updated working tree contents. All 41 unit tests passed.

### Fixed (Phase 4, False Alarm Diagnosis, Strict Spatial Cropping, and 2D Grid Selection)
- Resolved 100% false alarm issue caused by padding boundary artifacts and single-pixel thresholding:
  - Created `detector/cnn/infer.py` with `crop_to_original`, `pad_to_multiple`, `filter_min_area`, `compute_top_k_mean`, and `predict_mask`.
  - Enforced strict cropping back to unpadded original resolution across `train_cnn.py`, `eval_cnn.py`, `select_threshold.py`, and `infer.py` before any thresholding, component filtering, or pixel confusion calculation.
  - Added unit test `test_padded_strip_negative_and_cropping` in `tests/test_cnn_models.py` verifying that high activations confined to padded borders produce an image-level negative.
- Upgraded validation selection in `detector/cnn/selection.py` and `scripts/train_cnn.py`:
  - Implemented 2D grid sweep covering tau in {0.3, 0.5, 0.7, 0.9} x min_area in {0, 25, 50, 100, 200, 400, 800} to maximize recall subject to FAR <= 1.0% on both synthetic clean twins and real pristine panels.
  - Replaced single max pixel scoring with the mean of the top-100 pixel probabilities per image for ROC AUC calculation and fallback checkpoint selection.
  - Added `--save-every N` (default 2) to `scripts/train_cnn.py` writing `epoch_XX.pt` periodic checkpoints and syncing to Drive.
- Created `scripts/diagnose_fa.py`:
  - Evaluates model checkpoints on synthetic clean twins (500) and 1,000 seeded pristine panels.
  - Reports max probability, predicted area at tau in {0.5, 0.7, 0.9}, and distance from max-prob pixel to nearest image edge.
  - Prints FAR and recall grids with Wilson 95% confidence intervals and reports false alarms disappearing when excluding pixels within 8 px of the edge.
  - Generates overlay figure of the 12 worst clean panels saved to disk.
- Added diagnostic cell to `notebooks/phase4_train.ipynb` running `diagnose_fa.py` on `last.pt` and `best.pt`.
- Extended `scripts/dryrun_notebook.py` to execute `diagnose_fa.py` on CPU. All 5 dry-run steps passed end-to-end.
- Rebuilt `data/doppel_code.zip` (72 KB). All 44 unit tests passing.

### Added and Fixed (Phase 4, CorrNet v2 with Image Head, Partner-Guaranteed Cropping, and Diagnostics Suite)
- Partner-Guaranteed Cropping in `detector/colab_dataset.py`:
  - Audited positive crop completeness over 2,000 training samples. Before fix: 483 (94.71%) both source and target present, 27 (5.29%) single-patch orphaned crops. After fix: 514 (100.00%) both present, 0 (0.00%) orphaned.
  - Implemented partner-guaranteed cropping in `_crop_pair`: expands crop bounding box up to 512 px with margin. If pair exceeds 512 px, zeroes out masks for components whose partner is excluded, and marks the crop clean if no complete pair exists.
  - Implemented `colab_collate_fn` to support dynamic batch padding for variable crop sizes.
- ResNetCorrNet v2 with Image Head in `detector/cnn/corr_net.py`:
  - Added `--image-head` architecture: global average pooling on layer4 (512-d) concatenated with top-4 correlation summary statistics (max and mean across valid coordinates = 8-d), mapped through linear classification layer.
  - Updated `train_cnn.py` to optimize image-level BCE loss (weight 1.0) alongside pixel BCE (`pos_weight=1.0`) and soft Dice loss on forged samples only.
  - Updated `infer.py` to evaluate compound decision rule: `(image_prob >= s) and (pred_mask_area >= a)`.
- Validation Selection and Checkpoint Handling in `detector/cnn/selection.py`:
  - Upgraded threshold optimization to 3D grid sweep over (image_threshold s, pixel_threshold tau, min_area a).
  - Implemented lowest max-FAR fallback: when no checkpoint reaches FAR <= 1.0% on both validation sets, selects the checkpoint minimizing `max(synth_clean_far, pristine_far)`.
  - Added robust `load_checkpoint` helper handling PyTorch 2.6+ `weights_only=False` across all CNN scripts.
- Diagnostic Suite in `scripts/diagnose_fa.py`:
  - Extended grid to tau in {0.3, 0.5, 0.7, 0.9, 0.95, 0.98} x min_area in {0, 50, 100, 200, 400, 800, 1600}.
  - Added training config and loss settings display from `config.json` and checkpoint dictionary.
  - Added false alarm breakdowns by class and degradation tier for synthetic clean twins and pristine panels.
  - Added recall breakdowns by class, tier, transform type, and single vs multi-copy at tau in {0.5, 0.9} with Wilson 95% CIs.
  - Added `--image-head` support and displays image probability `img_p` in worst-clean overlay figure.
- Notebook and Colab Assets:
  - Added Cell 8b in `notebooks/phase4_train.ipynb` for CorrNet v2 full training (25 epochs, 120 min budget) and immediate diagnostics on `last.pt`.
  - Extended `scripts/dryrun_notebook.py` to 7 steps (loader check, smoke test, overfit test, CorrNet v2 train, resume from Drive, and diagnose_fa on CPU).
  - Rebuilt `data/doppel_code.zip` (80 KB).
  - All 47 unit tests passing. Zero em dashes across repository.
