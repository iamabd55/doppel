# Architecture and Data Flow

Doppel is a research prototype for detecting copy-move forgery in scientific images (such as western blots and microscopy figures).

## Current System (Phase 1: Frozen DCT Baseline)

The current pipeline implements a block-based discrete cosine transform (DCT) detector:

1. Input preprocessing:
   - Takes an image, converted to single-channel grayscale (uint8 or float32).

2. Feature extraction (`detector/features.py`):
   - Sliding window extracts all overlapping 8x8 blocks (`sliding_window_view`).
   - Texture filter: computes standard deviation across each block. Discards flat/smooth blocks where `std < min_std` (default 4.0) to eliminate false alarms from uniform backgrounds.
   - 2D DCT (`scipy.fft.dctn`) with orthonormal normalization applied to surviving blocks.
   - Zigzag scan orders coefficients from lowest to highest spatial frequencies. The first K coefficients (default K=6 or 16) are retained.
   - Quantization: coefficients are divided by Q (default Q=8 or 4) and rounded to integer fingerprints (`int16`).
   - Block coordinate tracking: each fingerprint retains its top-left (row, col) pixel position.

3. Matching (`detector/matching.py`):
   - Lexicographic sort (`np.lexsort` on reversed fingerprint columns) groups identical or near-identical fingerprints together.
   - Neighbor comparison: compares each block with its neighbors up to a search window `k` (default 1..5).
   - Distance pruning: filters out matches where Euclidean distance between block positions is less than `min_dist` (default 30 px).
   - Canonical shift vector: normalizes shift vectors `(dy, dx)` to point in a single direction (down or right).

4. Shift vector voting and cleanup (`detector/detect.py`):
   - Frequency count of shift vectors. Only candidate pairs sharing a shift vector with at least `min_votes` (default 150) are retained.
   - Mask generation: surviving blocks are painted as white (255) into a zero-initialized binary mask.
   - Morphological closing (`cv2.morphologyEx` with 9x9 kernel) bridges intra-block gaps.
   - Small object removal (`skimage.morphology.remove_small_objects`, min size 500 px) removes isolated noise patches.

5. Evaluation (`detector/evaluate.py`):
   - Compares predicted binary mask against ground truth mask.
   - Computes pixel-level true positives (TP), false positives (FP), false negatives (FN).
   - Returns precision, recall, and F1 score.

## Target Multi-Branch Pipeline (Phases 2 to 5)

```
Input Image
   │
   ├─► [Branch 1: DCT]  ──► Block DCT ──► Lexsort ──► Shift Voting ──► DCT Mask
   │
   ├─► [Branch 2: SIFT] ──► Keypoints ──► Descriptors ──► 2NN Match ──► RANSAC Affine ──► Hull & Dilate ──► SIFT Mask
   │        └─ (Includes horizontal flip pass for mirrored copies)
   │
   ├─► [Branch 3: CNN]  ──► Backbone Features ──► Self-Correlation ──► Decoder ──► CNN Mask
   │
   └─► Hybrid Fusion ──► Heuristic (Union / Intersection / Vote) or U-Net ──► Final Mask
```

### SIFT Branch Details (Phase 2)
1. Keypoints & Descriptors: `cv2.SIFT_create()` extracts distinctive scale-space extrema and 128-d gradient histograms.
2. Within-Image Matching: kNN (k=3) against self, discard self-match, Lowe's ratio test (d1 / d2 < ratio_threshold), spatial distance pruning (>= min_dist), symmetric deduplication.
3. Horizontal Flip Pass: Evaluates mirrored/reflected clones by matching original descriptors against descriptors from `cv2.flip(image, 1)` and re-mapping coordinates via `x_orig = W - 1 - x_flip`. Uses a dedicated `min_inliers_flip` threshold to prevent false alarms on symmetric architecture.
4. Orientation Consistency Filter: Rejects spurious match pairs whose angle difference `delta = (ang2 - ang1) % 360` lacks consensus support from neighboring matches.
5. Geometric Verification: Agglomerative hierarchical clustering groups matched points into distinct spatial regions; 6-parameter RANSAC affine estimation (`cv2.estimateAffine2D`) fits a transformation and validates both absolute inlier count (`min_inliers`) and cluster inlier ratio (`min_inlier_ratio`).
6. Localization: Inliers are painted onto a binary mask using per-cluster spatial convex hulls (preventing cross-image polygon bloat) combined with circular dilation and morphological closing. Conservative dilation-only mode supported via `use_convex_hull=False`.
7. Calibration & Validation: Systematic 50/50 partition splits evaluation into tuning sets (images 1-20) and held-out test sets (images 21-40) to guard against threshold overfitting.
8. Status: Frozen as `cfg8_best_combo` (ratio 0.70, inliers 6, inlier ratio 0.25, flip inliers 10, orientation filter on, cluster convex hull). Failure analysis complete in `results/phase2_sift/failure_analysis.md`. SIFT is locked as an expert branch for transformation coverage.

### Synthetic Benchmark & Training Generator (Phase 3)
1. Base Dataset: BioFors dataset (PLOS ONE open-access images, ICCV 2021). Strict zero document leakage between train pool (696 documents) and held-out IDD test pool (54 documents). Additionally, 177 pristine train documents are completely excluded from the generator.
2. Native Resolution Synthesis: Images are preserved at native resolution (no blind resizing), protecting thin Western blot bands (median height 58 px, minimum 19 px).
3. Paired-Twin Architecture: Clean controls undergo identical degradation pipelines as forged counterparts with zero pasted content. Equal counts across all 8 (class x degradation tier) cells (chi-square independence test: p = 1.0000).
4. Patch Area Floor and Scaling: Log-uniform donor area sampling with a 400 px floor up to 25% of image area. Multi-copy support introduces 2 to 3 targets in ~20% of forged samples, including adjacent-band repeats for Western blots.
5. Shortcut Audit: Grouped 5-fold cross-validation by document ID confirms Logistic Regression (50.57%) and HistGradientBoosting (51.78%) cannot separate clean from forged panels on global image statistics.
6. BioFors IDD Mask Conventions: Produces composite image `_F`, target paste mask `_M`, donor source mask `_S`, union mask `_A = _M | _S`, and full transformation metadata `_meta.json`.
7. Colab Dataset Loader: `detector/colab_dataset.py` provides PyTorch loading with zero resizing, random positive-aware crops for training (50% positive share, not always centered), full zero-padded images for validation, and split disjointness checks.
8. Packaging: `data/synthetic_dataset.zip` (761 MB) encapsulates 16,000 train samples and 1,000 validation samples with manifests and `PROVENANCE.txt`, strictly excluding test splits and real BioFors data.

## Planned Product Layer (Phase 7)

```
Next.js Frontend ──► FastAPI Backend ──► Redis Queue / Celery Worker ──► Detector Engine
                                      └── Local Disk / MinIO Storage
                                      └── PostgreSQL Database
```
