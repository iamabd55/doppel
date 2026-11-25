# Phase 5 Diagnosis: Complete Task A Audit

This document records the empirical results for Task A (A1, A2, A3) as specified in `docs/AGENT_PROMPT.md`.

---

## Task A1: Real Forged Panels Audit in BioFors TRAIN Split

We audited all 25,405 panels across the 591 train pool documents in `data/biofors/splits/train_manifest.csv` against the official BioFors ground-truth manipulation annotations:
* `data/biofors/annotations/idd_gt.json` (Intra-Document Duplication / Copy-Move)
* `data/biofors/annotations/cstd_gt.json` (Cross-Scale / Transform Duplication)
* `data/biofors/annotations/edd_gt.json` (Extra-Document Duplication)

### Findings
* **Document Overlap:** All 54 IDD documents in `idd_gt.json` reside strictly in the test split (`test_pdfs.pkl`). There is zero overlap with `train_manifest.csv`.
* **Ground-Truth Forged Panels in Train:** Exactly **0 (zero)**.
* **Panels with Usable Region Masks in Train:** Exactly **0 (zero)**.

### A1 Results Table

| Split | Modality / Class | Total Panels | Real Forged Panels | Usable Region Masks |
|---|---|---|---|---|
| **TRAIN** | Blot/Gel | 16,386 | **0** | **0** |
| **TRAIN** | Microscopy | 9,019 | **0** | **0** |
| **TRAIN Total** | Both Classes | 25,405 | **0** | **0** |

**Conclusion for Task B5:** Because the official BioFors benchmark only annotates forgeries in the test split, there are no real forged panels in the train split. All training positives must originate from synthetic generation.

---

## Task A2: Baseline Detectors & CNN Audit on Real BioFors IDD Test Set

> [!NOTE]
> All metrics below are compiled directly from the recorded one-shot test results (`results/phase4_cnn/real_biofors_benchmark.csv`). The locked test set was not rerun or re-evaluated.

Evaluated on 2,286 real BioFors panels (91 forged, 2,195 clean; 74 forged Blot/Gel, 1,225 clean Blot/Gel; 17 forged Microscopy, 970 clean Microscopy).

### Per-Class Detection and Localization Quality

| Detector | Class | Forged Panels | Overlap Recall [95% CI] | Any-Flagged Recall [95% CI] | Clean Panels | False Alarms | Clean FAR [95% CI] | Micro Pixel F1 | Micro Pixel IoU | Macro Pixel F1 | Macro Pixel IoU |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Frozen DCT** | Overall | 91 | 14/91 (15.4%) [9.4%, 24.2%] | 14/91 (15.4%) [9.4%, 24.2%] | 2,195 | 8/2,195 (0.36%) [0.18%, 0.72%] | 0.1011 | 0.0533 | 0.0973 | 0.0788 |
| **Frozen DCT** | Blot/Gel | 74 | 13/74 (17.6%) [10.6%, 27.8%] | 13/74 (17.6%) [10.6%, 27.8%] | 1,225 | 8/1,225 (0.65%) [0.33%, 1.28%] | 0.1930 | 0.1068 | 0.1124 | 0.0919 |
| **Frozen DCT** | Microscopy | 17 | 1/17 (5.9%) [1.0%, 27.0%] | 1/17 (5.9%) [1.0%, 27.0%] | 970 | 0/970 (0.00%) [0.00%, 0.39%] | 0.0111 | 0.0056 | 0.0317 | 0.0217 |
| **Adapted DCT** | Overall | 91 | 23/91 (25.3%) [17.5%, 35.1%] | 23/91 (25.3%) [17.5%, 35.1%] | 2,195 | 36/2,195 (1.64%) [1.19%, 2.26%] | 0.1311 | 0.0702 | 0.1419 | 0.1093 |
| **Adapted DCT** | Blot/Gel | 74 | 22/74 (29.7%) [20.5%, 40.9%] | 22/74 (29.7%) [20.5%, 40.9%] | 1,225 | 36/1,225 (2.94%) [2.13%, 4.04%] | 0.2485 | 0.1419 | 0.1672 | 0.1294 |
| **Adapted DCT** | Microscopy | 17 | 1/17 (5.9%) [1.0%, 27.0%] | 1/17 (5.9%) [1.0%, 27.0%] | 970 | 0/970 (0.00%) [0.00%, 0.39%] | 0.0113 | 0.0057 | 0.0319 | 0.0219 |
| **Frozen SIFT** | Overall | 91 | 32/91 (35.2%) [26.1%, 45.4%] | 33/91 (36.3%) [27.1%, 46.5%] | 2,195 | 49/2,195 (2.23%) [1.69%, 2.94%] | 0.1216 | 0.0647 | 0.1343 | 0.0929 |
| **Frozen SIFT** | Blot/Gel | 74 | 19/74 (25.7%) [17.1%, 36.7%] | 20/74 (27.0%) [18.2%, 38.1%] | 1,225 | 37/1,225 (3.02%) [2.20%, 4.14%] | 0.1083 | 0.0572 | 0.0788 | 0.0521 |
| **Frozen SIFT** | Microscopy | 17 | 13/17 (76.5%) [52.7%, 90.4%] | 13/17 (76.5%) [52.7%, 90.4%] | 970 | 12/970 (1.24%) [0.71%, 2.15%] | 0.1330 | 0.0712 | 0.3759 | 0.2705 |
| **CorrNet v2 (CNN)** | Overall | 91 | 2/91 (2.2%) [0.6%, 7.7%] | 2/91 (2.2%) [0.6%, 7.7%] | 2,195 | 27/2,195 (1.23%) [0.85%, 1.78%] | 0.0057 | 0.0028 | 0.0060 | 0.0036 |
| **CorrNet v2 (CNN)** | Blot/Gel | 74 | 2/74 (2.7%) [0.7%, 9.3%] | 2/74 (2.7%) [0.7%, 9.3%] | 1,225 | 13/1,225 (1.06%) [0.62%, 1.81%] | 0.0120 | 0.0060 | 0.0073 | 0.0044 |
| **CorrNet v2 (CNN)** | Microscopy | 17 | 0/17 (0.0%) [0.0%, 18.4%] | 0/17 (0.0%) [0.0%, 18.4%] | 970 | 14/970 (1.44%) [0.86%, 2.41%] | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Classical Detector Fusion Summary

| Fusion Combination | Real Forged Recall [95% CI] | Clean False Alarm Rate [95% CI] | Operational Role |
|---|---|---|---|
| **Adapted DCT OR Frozen SIFT** | 44/91 (48.4%) [38.4%, 58.5%] | 83/2,195 (3.78%) [3.06%, 4.66%] | Sensitive Screening (Catches ~48.4% of forgeries) |
| **Adapted DCT AND Frozen SIFT** | 11/91 (12.1%) [6.9%, 20.4%] | 2/2,195 (0.09%) [0.02%, 0.33%] | High Confidence Verification (0.09% FAR, 84.6% precision) |

---

## Task A3: Seam Test on the Current CorrNet v2 Model

We evaluated `checkpoints/corr_v2/last.pt` on the newly generated `data/synthetic_v3/test` split using the locked operational thresholds from `configs/cnn_threshold.yaml` (budget 0.01).

The test split contains 300 triplets (900 images total):
* **Pristine:** 300 untouched clean base images.
* **Forged:** 300 copy-move images (source copied from the same image with irregular masks and blending).
* **Unrelated Paste:** 300 clean images with a patch copied from a DIFFERENT document through the exact same irregular mask, transform, and blending pipeline.

### Seam Test Quantitative Results

| Class | Pristine FPR (Clean Negative) | Unrelated Paste FPR (Clean Negative) | FPR Ratio (Unrelated / Pristine) | Forged Recall (Positive) |
|---|---|---|---|---|
| **Overall (300 triplets)** | 7/300 (2.33% [1.13%, 4.74%]) | 20/300 (6.67% [4.36%, 10.07%]) | **2.86x** | 20/300 (6.67% [4.36%, 10.07%]) |
| **Blot/Gel (52 triplets)** | 4/52 (7.69% [3.03%, 18.17%]) | 10/52 (19.23% [10.80%, 31.90%]) | **2.50x** | 13/52 (25.00% [15.23%, 38.21%]) |
| **Microscopy (248 triplets)** | 3/248 (1.21% [0.41%, 3.50%]) | 10/248 (4.03% [2.20%, 7.26%]) | **3.33x** | 7/248 (2.82% [1.37%, 5.71%]) |

### Seam Shortcut Hypothesis Verdict

**CONFIRMED: The false positive rate on unrelated_paste is 2.86x higher than on pristine (up to 3.33x higher in Microscopy).**

Furthermore:
1. **Identical Trigger Rate:** The current CorrNet v2 flags `unrelated_paste` at the **exact same rate (20/300, 6.67%)** as it flags `forged` (20/300, 6.67%). The model cannot distinguish between intra-image copy-move and cross-document splicing.
2. **Recall Collapse on v3:** When rectangular paste boxes were eliminated in v3, CorrNet v2's recall collapsed from **49.3%** down to **6.67%**.
3. **Conclusion:** CorrNet v2 was primarily triggering on pasting seams and boundary artifacts rather than semantic content correlation. The v3 dataset design (supervising `unrelated_paste` with an empty mask) directly targets and eliminates this shortcut.
