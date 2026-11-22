# File Inventory

Summary of files in the Doppel repository:

## Root Configuration and Documentation
- `configs/synthetic.yaml`: Centralized configuration for Phase 3 synthetic generation, BioFors splits, degradation parameters, and frozen SIFT baseline values.
- `configs/dct_adapted.yaml`: Locked adapted DCT parameters (min_votes=75, min_size=64) chosen strictly on synthetic VAL split to improve recall on narrow Western blot bands while bounding clean validation false alarms.
- `configs/held_out_pristine_docs.json`: JSON record of the 177 held-out pristine BioFors train documents strictly excluded from the synthetic generator.
- `docs/phase4_targets.md`: Pre-registered evaluation target metrics and methodological protocol for Phase 4 CNN training.
- `docs/phase4_log.md`: Test set touch counter and step tracker for Phase 4 CNN training and evaluation.
- `notebooks/colab_setup.md`: Guide and sanity check instructions for setting up the Google Colab training environment.
- `DOPPEL_AGENT.md`: Primary briefing and rulebook for autonomous and assisted development on Doppel. Contains rules, architecture, frozen baseline numbers, and phase roadmaps.
- `README.md`: Public project documentation, including problem statement, methodology comparison, status table, and baseline benchmarks.
- `pyproject.toml`: Python project configuration specifying dependencies (Flask, NumPy, OpenCV, scikit-image, scikit-learn, SciPy) and Python version requirements.
- `uv.lock`: Dependency lockfile managed by `uv`.
- `.gitignore`: Specifies ignored directories including `.venv`, model weights, and local data.

## Core Detection Module (`detector/`)
- `detector/__init__.py`: Package initialization.
- `detector/detect.py`: Top-level detection pipeline. Orchestrates feature extraction, lexicographic matching, shift vector voting, block painting, morphological closing, and small component removal.
- `detector/features.py`: Feature extraction using sliding-window 8x8 block decomposition, standard deviation texture filtering, 2D orthonormal DCT, zigzag coefficient ordering, and quantization.
- `detector/matching.py`: Candidate matching via lexicographic sorting (`np.lexsort`), neighbor window search, Euclidean distance thresholding, and canonical shift vector orientation.
- `detector/evaluate.py`: Evaluation utilities computing pixel-level true positives, false positives, false negatives, precision, recall, and F1 score against ground truth masks.
- `detector/synthetic.py`: Synthetic copy-move generator supporting realistic transformations, post-processing degradations, narrow Western blot strip handling, log-uniform area sampling with 400 px floor, multi-copy support, and BioFors IDD mask conventions (_F, _M, _S, _A, and metadata).
- `detector/colab_dataset.py`: PyTorch Dataset implementation for Google Colab training. Supports native variable sizes via padding or random crops, joint geometric augmentations, and document-level split disjointness assertions.

### SIFT Detection Subpackage (`detector/sift/`)
- `detector/sift/__init__.py`: Package entrypoint re-exporting all SIFT public functions for clean imports.
- `detector/sift/features.py`: Keypoint detection (`extract_keypoints`), 128-d descriptor extraction (`extract_descriptors`, `detect_and_compute`), and keypoint drawing (`draw_keypoints`).
- `detector/sift/matching.py`: Within-image kNN matching (`match_within_image`), horizontal flip pass cross-matching (`match_flipped`), orientation consistency filtering (`filter_by_orientation`), coordinate re-mapping (`map_flip_coordinates`), and match line drawing (`draw_matches`).
- `detector/sift/verification.py`: Hierarchical spatial clustering and RANSAC 6-parameter affine estimation (`verify_matches_ransac`) with `min_inlier_ratio` filtering and cluster metadata reporting.
- `detector/sift/localization.py`: Binary mask generation via per-cluster convex hull and keypoint dilation (`create_sift_mask`), and configurable end-to-end detection pipeline (`detect_sift`).

### CNN Detection Subpackage (`detector/cnn/`)
- `detector/cnn/__init__.py`: Package initialization exporting ResNetUNet and ResNetCorrNet models.
- `detector/cnn/unet.py`: Baseline ResNet-18 U-Net segmentation network with ImageNet pretraining, multi-scale skip connections, and native resolution mask logits.
- `detector/cnn/corr_net.py`: Copy-move correlation network with ResNet-18 backbone, stride-8 feature self-correlation module with 128-d projection, Chebyshev spatial self-mask, zero-pad valid-mask filtering, top-4 match extraction, and multi-scale skip decoder.

## Evaluation and Utility Scripts (`scripts/`)
- `scripts/make_colab_code_zip.sh`: Archives repository code, tests, and training modules into `data/doppel_code.zip` for Colab upload.
- `scripts/make_pristine_heldout_zip.py`: Packages 4,039 pristine panels from the 177 held-out train documents into `data/pristine_heldout.zip` for false alarm validation.
- `scripts/train_cnn.py`: Standalone CNN training script supporting ResNetUNet and ResNetCorrNet, AMP, AdamW, CosineAnnealingLR, BCE + Dice loss, variable-size padding, and native full-resolution validation.
- `scripts/audit_biofors.py`: Step 3.0 data audit script verifying BioFors document integrity, image counts, class distribution, and resolution profiles.
- `scripts/create_splits.py`: Step 3.1 split creation script partitioning BioFors train documents into disjoint Train and Validation pools and generating CSV manifests.
- `scripts/make_synthetic.py`: Step 3.2 batch generation script building synthetic datasets (train, val, test) with full metadata and CSV manifests.
- `scripts/visualize_synthetic.py`: Visual inspection utility rendering multi-column sample grids (original, forged, M, S, A masks) to figures/synthetic_samples_grid.png.
- `scripts/visualize_twins.py`: Step A.2 visual QA grid utility rendering 12 paired twins across degradation tiers to results/phase3_synthetic/twin_grid.png.
- `scripts/benchmark.py`: Step 3.3 scientific benchmark runner evaluating frozen DCT and frozen SIFT on synthetic test sets and real BioFors IDD test panels with image-level and pixel-level metrics.
- `scripts/diagnose_translations.py`: Diagnostic suite running 100 zero-degradation clean translations, patch size breakdown (<500 px vs >=500 px), and dataset shortcut/leakage audits.
- `scripts/audit_dataset_shortcuts.py`: Dataset shortcut audit training Logistic Regression and HistGradientBoosting classifiers with 5-fold grouped CV by doc_id to verify ~50% chance-level performance.
- `scripts/tune_adapted_dct.py`: Validation tuning suite evaluating adapted DCT thresholds strictly on synthetic validation data without test set leakage.
- `scripts/ablate_sift.py`: SIFT calibration and ablation study runner evaluating parameter sweeps across the 50/50 tuning split (images 1-20) or held-out test split (images 21-40).
- `scripts/batch_test.py`: Benchmark evaluation script running the DCT detector on CoMoFoD images 001-040 across categories, evaluating against full mask M and paste-only mask B.
- `scripts/batch_test_sift.py`: Step 2.7 benchmark runner evaluating SIFT across all 5 CoMoFoD categories (200 forged images) and 40 originals, recording per-image CSV and side-by-side DCT vs SIFT summary.
- `scripts/false_alarm_test.py`: False alarm evaluation testing untouched original images (001_O through 040_O) to verify zero or low false positive rates.
- `scripts/experiments.py`: Robustness experimentation script measuring F1 resilience against Gaussian noise, Gaussian blur, and JPEG compression degradation.
- `scripts/inspect_comofod.py`: Diagnostic utility comparing pixel differences between forged (F), original (O), and ground truth masks (B, M) for CoMoFoD set 001.
- `scripts/make_test.py`: Simple synthetic generation script producing a translated copy-paste test image and mask.
- `scripts/run.py`: Quick runner script evaluating the detector on a synthetic sample image.
- `scripts/try_001.py`: Parameter exploration script testing the effect of different `min_votes` thresholds on CoMoFoD set 001.
- `scripts/test_sift_keypoints.py`: Step 2.1 diagnostic script testing SIFT keypoint extraction, visualizing detections, and measuring keypoint distribution over textured vs flat regions.
- `scripts/test_sift_descriptors.py`: Step 2.2 verification script validating 128-dimensional descriptor extraction, shapes, and normalization.
- `scripts/test_sift_matching.py`: Step 2.3 verification script matching keypoints within images across CoMoFoD categories (translation, rotation, scaling, original) and drawing match lines.
- `scripts/test_sift_ransac.py`: Step 2.4 verification script testing RANSAC affine geometric verification and outlier rejection on translation, rotation, scaling, and untouched images.
- `scripts/diagnose_rotation.py`: Diagnostic utility inspecting SIFT matching on CoMoFoD rotation samples against ground truth to debug ratio sensitivity and cluster fragmentation.
- `scripts/inspect_rotation_category.py`: Batch diagnostic script analyzing keypoint density inside ground truth, inlier counts, and estimated transformation parameters (rotation angle, scale) across rotation samples (041-050).
- `scripts/test_sift_localization.py`: Step 2.5 verification script evaluating binary mask generation (convex hull + dilation) against ground truth masks with precision, recall, and F1 scores.
- `scripts/test_sift_flip.py`: Step 2.6 demonstration script comparing standard SIFT vs flip pass SIFT on mirrored synthetic copy-move forgeries.

## Automated Test Suite (`tests/`)
- `tests/test_cnn_models.py`: Unit tests verifying forward pass shapes, variable dimensions, valid-mask spatial boundary handling, gradient backpropagation, and AMP fp16 compatibility for ResNetUNet and ResNetCorrNet.
- `tests/test_flip.py`: Unit tests validating flip coordinate mapping round-trips, boundary margins, and synthetic mirrored clone detection.
- `tests/test_synthetic.py`: Unit tests validating synthetic copy-move generator invariants, mask union identities, determinism, narrow blot strips, and document split disjointness.
- `tests/test_colab_dataset.py`: Unit tests verifying ColabDoppelDataset PyTorch batch loading, shapes, dtypes, forged fraction, mask coverage, val padding, and split disjointness.

## Interactive Web Baseline Demo (`web/`)
- `web/app.py`: Flask application providing an interactive web interface and REST API endpoints (`/api/detect`, `/api/comofod/list`, `/api/comofod/<num>`).
- `web/templates/index.html`: Web dashboard HTML template.
- `web/static/app.js`: Client-side interaction logic for image upload, parameter control, and result visualization.
- `web/static/style.css`: Dashboard styling.
- `web/static/lenis.min.js`: Smooth scrolling library.
- `web/static/lenis.css`: Styles for smooth scrolling.

## Learning and Visuals
- `learning/dct_demo.py`: Educational script illustrating 1D and 2D DCT fundamentals, energy compaction, and inverse reconstruction.
- `figures/robustness.png`: Performance chart showing F1 score under noise, blur, and JPEG compression.

## Agent Documentation (`agentlogs/`)
- `agentlogs/architecture.md`: Architectural layout, data flow, and pipeline progression across phases.
- `agentlogs/files.md`: Inventory and responsibility summary of every file in the repository.
- `agentlogs/changelog.md`: Chronological log of changes, phase updates, and task milestones.

## Data and Results Directories (Untracked / Outputs)
- `data/`: Contains CoMoFoD benchmark images and local synthetic test sets.
- `results/`: Stores experiment CSV tables, diagnostic plots, and markdown summaries per phase.
  - `results/phase2_sift/sift_benchmark.csv`: Full benchmark evaluation across all 200 CoMoFoD forged images and 40 originals.
  - `results/phase2_sift/benchmark_summary.md`: Side-by-side DCT vs initial SIFT benchmark comparison.
  - `results/phase2_sift/ablation_tuning_split.md` / `.csv`: Systematic 9-configuration calibration sweep on the 50/50 tuning split (120 images).
  - `results/phase2_sift/ablation_held_out_split.md` / `.csv`: Held-out validation of frozen configuration `cfg8_best_combo` (120 unseen images).
  - `results/phase2_sift/failure_analysis.md`: Detailed failure analysis and Phase 2 sign-off document.
  - `results/phase3_synthetic/synthetic_benchmark.csv`: Evaluation metrics of frozen DCT and SIFT on the 300 synthetic test images.
  - `results/phase3_synthetic/real_biofors_benchmark.csv`: Evaluation metrics of frozen DCT and SIFT on 2,286 real BioFors IDD test panels.
  - `results/phase3_synthetic/benchmark_summary.md`: Consolidated summary and cross-method scientific findings.
