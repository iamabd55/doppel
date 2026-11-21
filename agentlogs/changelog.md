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
