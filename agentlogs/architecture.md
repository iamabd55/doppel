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
3. Horizontal Flip Pass: Evaluates mirrored/reflected clones by matching original descriptors against descriptors from `cv2.flip(image, 1)` and re-mapping coordinates via `x_orig = W - 1 - x_flip`.
4. Geometric Verification: Agglomerative hierarchical clustering groups matched points into distinct spatial regions; 6-parameter RANSAC affine estimation (`cv2.estimateAffine2D`) fits a transformation and extracts verified inliers.
5. Localization: Inliers are painted onto a binary mask using convex hull polygon filling of inlier clusters and circular dilation around keypoint coordinates, followed by morphological closing.
6. Verification: Binary mask evaluated directly against ground truth masks using `detector/evaluate.py:scores(mask, gt)`.

## Planned Product Layer (Phase 7)

```
Next.js Frontend ──► FastAPI Backend ──► Redis Queue / Celery Worker ──► Detector Engine
                                      └── Local Disk / MinIO Storage
                                      └── PostgreSQL Database
```
