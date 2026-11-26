# Doppel Natural-Image Evaluation Protocol

**Status: FROZEN after initial commit. Any change must be a dated addendum appended below -- never an edit to the existing text.**

Written on: 2026-11-25. No detector has been run on any image at the time of writing.

---

## 0. Purpose

This document fixes, in advance, every decision that governs the evaluation of copy-move forgery detection on natural images. It is written before any detector runs. After the commit that introduces this file, the protocol is immutable. Seeing results does not entitle anyone to edit it.

---

## 1. Dataset Roles

### 1.1 CASIA v2.0 Copy-Move Subset

Source manifests: `data/natural/manifests/casia2.csv` (seed=0, group split 30/70).

| Split | Label | Count | Role |
|---|---|---|---|
| calibration | forged | 964 | Threshold and fusion calibration ONLY |
| calibration | authentic | 2247 | False-alarm budget calibration ONLY |
| test | forged | 2320 | Final held-out evaluation |
| test | authentic | 5244 | Final held-out evaluation |

Groups are defined by `group_id` (the source authentic image ID). No `group_id` appears in both calibration and test. No authentic image in calibration has a forged copy in test, and vice versa. These assertions are verified in `scripts/build_natural_manifests.py`.

The calibration split is used freely for threshold search. Test splits are evaluated in ONE final pass with `--final` and results are never used to change any threshold or rule.

### 1.2 CoMoFoD

Source manifest: `data/natural/manifests/comofod.csv` (all rows have `split=dev`).

Counts: 3,200 forged + 3,200 authentic, balanced 640/640 across 5 attack categories (translation, rotation, scaling, distortion, combination).

**Role: development benchmark.** DCT and SIFT were tuned on CoMoFoD (the frozen configs come from that tuning). Results on CoMoFoD are reported separately and are explicitly labeled "optimistic for DCT and SIFT" in every table and figure. DOA-GAN is cross-dataset here (trained on USC-ISI, never on CoMoFoD), so its CoMoFoD numbers are not inflated.

CoMoFoD is not held out. It may be inspected freely throughout development.

### 1.3 COVERAGE

Source manifest: `data/natural/manifests/coverage.csv` (header only, 0 rows).

**Role: extra small test set, if images are downloaded.** If images become available, they are evaluated in ONE final pass alongside the CASIA test split, with `--final`. No thresholds are touched after COVERAGE is seen. If images are never downloaded, this dataset is omitted from the final report with a note.

---

## 2. Detectors

### 2.1 Frozen DCT (control)

Config source: `configs/dct_adapted.yaml` -- the CoMoFoD baseline config (NOT the adapted scientific config).

**Exact frozen parameters:**
```
B=8, K=16, Q=4, min_std=4.0, window=5, min_dist=30,
min_votes=50, kernel=9, min_size=500
```

Entry point: `detector/detect.py`, function `detect(gray_uint8, ...)`. Input is a 2D uint8 grayscale array. Output is a uint8 mask in {0, 255} at native image resolution. Image-level decision: `(mask > 0).any()`.

The frozen DCT is always reported. It is the control.

**Note:** The adapted scientific DCT (Q=8, K=6, min_votes=75/150) is NOT used. That config was tuned on synthetic scientific forgeries and is not appropriate here.

### 2.2 Calibrated DCT

The calibrated variant searches only two knobs on the CASIA calibration split. Everything else is identical to the frozen config.

**Calibration grid (exhaustive):**

```
min_votes : [25, 50, 75, 100, 125, 150]
min_size  : [100, 250, 500, 750, 1000]
```

That is 30 grid points. Selection criterion: maximize image-level recall on CASIA calibration forged images subject to false alarm rate <= 2% on CASIA calibration authentic images. If multiple grid points tie on recall, pick the one with the lowest false alarm rate.

### 2.3 Frozen SIFT (control)

Config source: `cfg8_best_combo` from `scripts/benchmark.py`.

**Exact frozen parameters:**
```
ratio_threshold=0.70, min_dist=30.0, ransac_threshold=3.0,
min_inliers=6, cluster_threshold=100.0, mask_radius=12,
use_convex_hull=True, check_flip=True,
min_inlier_ratio=0.25, min_inliers_flip=10,
filter_orientation=True, orientation_tolerance=20.0
```

Entry point: `detector/sift/localization.py`, function `detect_sift(image, ...)`. Input is a grayscale or BGR uint8 image. Output is a uint8 mask in {0, 255}, plus matched inlier point pairs and transforms. Image-level decision: `(mask > 0).any()`. Continuous image score: inlier count `len(pts1)`.

The frozen SIFT is always reported.

### 2.4 Calibrated SIFT

Searches one knob only on the CASIA calibration split. All other parameters are identical to the frozen config.

**Calibration grid:**

```
min_inliers : [3, 4, 5, 6, 8, 10, 12, 15]
```

That is 8 grid points. Selection criterion: maximize image-level recall on CASIA calibration forged images subject to false alarm rate <= 2% on CASIA calibration authentic images. Ties broken by lowest false alarm rate.

`min_inlier_ratio` is NOT calibrated. Tuning both simultaneously risks overfitting 964 calibration images.

### 2.5 DOA-GAN (pretrained, no fine-tuning)

**Checkpoint:** `doa-gan_weights/three_channel.pkl` (`out_channel=3`). The three-channel checkpoint is used because it separates source (channel 1) and target (channel 0) regions, which matters for the mask-convention discussion in Section 4.

**Preprocessing (fixed):**
1. Load image as BGR with `cv2.imread`, take `[..., :3]`.
2. Resize to `(320, 320)` with `cv2.INTER_LINEAR`.
3. Convert to RGB float32, divide by 255.
4. Normalize per-channel: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.
5. Convert to tensor `[1, 3, 320, 320]`.

**Outputs:**

`model(x)` returns `(mask_logits, det_logit)`.

- **Image-level score:** `s = torch.sigmoid(det_logit).item()`. This is the model's own detection head score, in [0, 1]. Higher means more likely forged.
- **Pixel probability heatmap at 320x320:** `prob = torch.softmax(mask_logits, dim=1)[0]`. Combined forged probability is `heatmap = 1.0 - prob[2].cpu().numpy()` (1 minus background channel). This is equivalent to `prob[0] + prob[1]` and is numerically smoother.
- **Mapping back to native resolution:** resize `heatmap` from `(320, 320)` to `(H_orig, W_orig)` with `cv2.resize(..., (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)` before thresholding. Binary masks use `INTER_NEAREST` after thresholding.

**Thresholds calibrated on CASIA calibration split:**

Three knobs, searched jointly by grid on the CASIA calibration split:

```
image_score_threshold  : [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
pixel_threshold        : [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
min_forged_area_px     : [50, 200, 500, 1000, 2000]
```

That is 7 x 6 x 5 = 210 grid points. Selection criterion for each budget: maximize image-level recall on CASIA calibration forged images subject to false alarm rate <= budget on CASIA calibration authentic images. Ties broken by lowest false alarm rate. `pixel_threshold` and `min_forged_area_px` affect only the pixel-level output and do NOT affect image-level recall (which is determined by `image_score_threshold` alone). So the image-level grid collapses to 7 points; the pixel knobs are picked separately on the calibration forged images by maximizing pixel F1.

**Reporting budgets:** 1%, 2%, 5%. Primary budget: **2%**.

---

## 3. Fusion Rules

Five fusion rules, run with BOTH the frozen and calibrated classical variants, combined with DOA-GAN. Total combinations: 5 rules x 2 classical variants = 10 rows in the main table (DOA-GAN is the same in all).

| ID | Rule | Image-level logic | Pixel-level logic |
|---|---|---|---|
| F1 | DCT OR SIFT | `dct_decision OR sift_decision` | union of masks |
| F2 | DCT AND SIFT | `dct_decision AND sift_decision` | intersection of masks |
| F3 | DCT OR SIFT OR DOA-GAN | any of the three is positive | union of all three masks |
| F4 | 2-of-3 majority | at least 2 of the 3 are positive | union of the two (or three) agreeing masks |
| F5 | (DCT OR SIFT) AND DOA-GAN | `(dct OR sift) AND doagan` | intersection of (dct-union-sift) with doagan mask |

Image-level decisions are primary. Pixel-level fusion is secondary and uses the convention above.

DCT and SIFT produce binary decisions directly (from their `(mask > 0).any()` logic). DOA-GAN image-level decision: `image_score >= image_score_threshold` at the calibrated threshold for the 2% budget.

For the frozen classical variants in fusion, DCT and SIFT decisions are those of their frozen configs with no threshold tuning. For the calibrated variants, the calibrated `min_votes`/`min_size`/`min_inliers` are used.

No other fusion rules are reported.

---

## 4. Metrics

### 4.1 Image-Level Metrics

For every detector and fusion rule, compute on each split:

- **Recall (TPR):** fraction of forged images flagged as forged.
- **False alarm rate (FAR):** fraction of authentic images flagged as forged.
- **Wilson 95% confidence intervals** on both recall and FAR. Wilson intervals are used because they behave correctly near 0 and 1 (Wald intervals do not).

Formula for the Wilson CI on a proportion p with n trials:

```
center = (p + z^2/(2n)) / (1 + z^2/n)
half   = z * sqrt(p*(1-p)/n + z^2/(4n^2)) / (1 + z^2/n)
where z = 1.96 for 95% CI
```

### 4.2 Pixel-Level Metrics (Forged Images Only)

Computed only on images where the label is forged AND a ground-truth mask is available.

**Mask convention mismatch:**

- CASIA2 ground-truth marks only the pasted TARGET region (confirmed: 19/20 checked masks have 1 connected component).
- CoMoFoD ground-truth marks BOTH source and target regions.
- DOA-GAN three-channel output separates channels: channel 0 = target, channel 1 = source.

**Reporting rules:**

For CASIA2: pixel F1 and IoU are computed against the target-only ground-truth mask using the full combined heatmap `(1 - prob[2])` thresholded, because we cannot separate source from target in the DCT and SIFT outputs. This is stated in every table.

For CoMoFoD: the ground-truth is source+target combined. Predicted mask is the full combined output of each detector. F1 and IoU are computed against the combined mask. Additionally, for DOA-GAN only, we report F1 separately for target-only (channel 0 thresholded vs target portion of mask) and source-only (channel 1 vs source portion) when feasible.

**Pixel F1:**

```
F1 = 2*TP / (2*TP + FP + FN)
```

where TP, FP, FN are pixel counts at the native image resolution.

**IoU (Intersection over Union):**

```
IoU = TP / (TP + FP + FN)
```

Pixels are compared at native resolution after all masks are resized back from detector space (DOA-GAN 320x320 -> native; DCT and SIFT produce native-resolution masks directly).

Pixel metrics are averaged over all forged images in the split (macro-average over images, not global pixel pooling).

### 4.3 Overlap Recall

An image is an "overlap hit" if the predicted mask and the ground-truth mask share at least one foreground pixel (intersection of binarized masks is non-empty). Overlap recall is the fraction of forged images that are overlap hits. Reported with Wilson CI. This is a weaker criterion than pixel F1 and captures rough localization.

---

## 5. Ablation Table Structure

One table per dataset covering:
- Each single detector (frozen DCT, calibrated DCT, frozen SIFT, calibrated SIFT, DOA-GAN at 1%/2%/5% budgets).
- Each fusion rule F1 through F5, with frozen and calibrated classical variants.
- Columns: Recall (CI), FAR (CI), Overlap Recall (CI), Pixel F1, Pixel IoU, N forged, N authentic.

CoMoFoD table is additionally broken down per attack category (translation, rotation, scaling, distortion, combination). Each category row shows the same column set.

---

## 6. Test Discipline

The CASIA calibration split and CoMoFoD (dev) may be used freely at any time for threshold search, debugging, and development inspection.

The CASIA test split and COVERAGE (if available) are evaluated ONCE in a final pass. The evaluation script accepts `--final` to unlock test-split output. Without `--final`, the script computes and prints only calibration and dev metrics. Any attempt to print test metrics without `--final` raises a hard error.

No threshold, parameter, or fusion rule is changed after seeing test results. If a change is needed after the fact (e.g., a discovered bug), it must be documented as a dated addendum at the bottom of this file, never as an edit to the protocol text above.

---

## 7. Failure Analysis

The following breakdowns are pre-committed for the final report, regardless of whether they favor or hurt the results:

### 7.1 By Image Size

Images in each split are binned by their shorter side:
- Small: shorter side < 400 px
- Medium: 400 to 800 px
- Large: > 800 px

For each bin, report Recall and FAR (with Wilson CIs) for every detector and fusion rule. This reveals whether detectors fail on small images (fewer features for SIFT, coarser blocks for DCT).

### 7.2 By Manipulation Type

For CASIA2 test: the filename encodes the manipulation type in position 3 of the underscore-split stem (e.g., `NRN`, `CRN`, `NND`). Report recall broken down by this code where sample counts allow (at least 20 images per bucket).

For CoMoFoD: break down by attack category (translation, rotation, scaling, distortion, combination). This is already in the ablation table.

### 7.3 Montage of Successes and Failures

For each detector (frozen DCT, frozen SIFT, DOA-GAN), produce a montage of:
- 8 randomly sampled successes: forged images correctly detected, with predicted mask overlaid in green and ground-truth mask boundary in red.
- 8 randomly sampled failures: forged images missed (false negatives), showing the image and ground-truth mask boundary.

Montages are saved to `results/natural/montage_{detector}_successes.png` and `results/natural/montage_{detector}_failures.png`. They are generated during the final pass with `--final` and committed to `results/` only if the results directory is not gitignored (currently it is; they will be noted in the report as local artifacts).

---

## 8. Reproducibility Commitments

- All random seeds fixed: manifest split seed=0 (committed in `scripts/build_natural_manifests.py`).
- Manifests are committed as CSV files in `data/natural/manifests/`.
- All detector configs are pinned in code (not read from mutable YAML that could drift).
- The calibration grid is defined above. The search is exhaustive, not random.
- All scripts print their version (git commit hash) and the manifest path at the start of each run.

---

## 9. What Is Not Done

- No deep model training.
- No fine-tuning of DOA-GAN on any dataset.
- No new feature extractors beyond DCT, SIFT, and DOA-GAN.
- No ensemble learning or learned fusion weights.
- No use of COVERAGE for calibration (it is test-only or omitted).
- No use of CoMoFoD for threshold calibration of classical detectors (the frozen configs already came from CoMoFoD tuning; running a second round of tuning on it would be double-dipping).
- No editing of this protocol after the commit that introduces it.

---

## Addenda

*(Append dated addenda here if any post-hoc correction is required. Do not edit the text above.)*

## Addendum (2026-11-26) | DOA-GAN removed from fusion

DOA-GAN is dropped from fusion because its outputs did not adequately discriminate forged from authentic images on the CASIA2 calibration smoke set. The corrected three-channel scores favored authentic images on average. The single-channel scores showed partial ranking signal with substantial class overlap and did not establish performance at the primary 2% false-alarm budget. See [DOA-GAN calibration smoke test](natural_doagan_failure.md).

Only fusion rules (a) F1, DCT OR SIFT (mask union), and (b) F2, DCT AND SIFT (mask intersection), remain. Each rule runs with both frozen and calibrated classical variants. This addendum was recorded before any test-split evaluation.

## Addendum (2026-11-26) | Single-channel DOA-GAN restored for three-detector fusion

The project will evaluate three-detector fusion using the pretrained single-channel checkpoint. This supersedes the removal decision above and retains the smoke-test findings as evidence. DOA-GAN uses the authors' folder preprocessing and sigmoid localization output. Its image score is the fraction of 320×320 pixels with foreground probability greater than 0.5. Its binary mask uses the same pixel threshold after linearly resizing probabilities to the image size. The detection logit is excluded.

Use only CASIA2 calibration images to select score thresholds for the existing 1%, 2%, and 5% FAR budgets. Choose the threshold with highest recall, then lowest FAR, then highest threshold. A threshold above 1 is available to return zero positives when needed. Use the 2% threshold for fusion. Rules F1 through F5 are restored so both classical-only and three-detector rules can be compared. DOA-GAN contributes its mask only when its image decision is positive. The F4 mask is the union of masks from the positive voters. Report CASIA2 localization against its target-only ground truth as specified above. Record thresholds and the comparison before opening the reserved test pass.

Run the following from the repository root. The calibrated evaluation uses the settings written by `natural_aggregate.py`.

```bash
.venv/bin/python -u scripts/natural_run_doagan.py --split calibration
.venv/bin/python -u scripts/natural_run_fusion.py calibrate
.venv/bin/python -u scripts/natural_run_fusion.py evaluate
.venv/bin/python -u scripts/natural_run_fusion.py evaluate --calibrated
```

For one RGB image, `natural_run_fusion.py predict IMAGE --masks-out OUTPUT.npz` prints the detector and fusion decisions and saves the image-size masks for inspection.
