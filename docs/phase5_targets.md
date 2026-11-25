# Phase 5: Pre-Registered Target Metrics and Evaluation Protocol

**Pre-registration Date:** November 2026  
**Status:** FROZEN and COMMITTED before model training or dev threshold tuning.  
**Branch:** `phase5-cnn-v3`  

---

## 1. Context and Problem Statement

In Phase 4, the CorrNet v2 model achieved 49.3% recall on the v2 synthetic test set, but collapsed to 2.2% recall (2/91) on the locked real BioFors IDD test benchmark. 

The Task A3 diagnostic seam test confirmed the working hypothesis:
* On `data/synthetic_v3/test`, CorrNet v2 triggered on `unrelated_paste` at **6.67%**, which is **2.86x higher** than on `pristine` (2.33%).
* The model flagged `unrelated_paste` at the exact same rate as `forged` (6.67%), proving that it learned to detect rectangular pasting seams and boundary artifacts rather than semantic intra-image duplication.

Phase 5 addresses this by training on `data/synthetic_v3` (irregular blob masks, feathered/Poisson blending, realistic structure, with `unrelated_paste` supervised by empty masks to penalize seam shortcuts) and evaluating against pre-registered criteria.

---

## 2. Pre-Registered Operational Targets

The following targets are formally pre-registered and locked prior to training. Model checkpoint selection and threshold calibration will be conducted on validation data, and these targets will be verified on the held-out test sets.

### Target 1: Real Held-Out Pristine False Alarm Rate
* **Dataset:** Real pristine panels from the 177 held-out train documents (`configs/held_out_pristine_docs.json`), split by document with a fixed seed (`configs/pristine_dev_split.json`) into two disjoint halves: **Half A** (88 docs, 1,969 panels, used solely for threshold calibration) and **Half B** (89 docs, 2,070 panels, reserved strictly for held-out evaluation).
* **Evaluation Set:** **Half B** of the real held-out pristine dev set (disjoint from both `synthetic_v3/train` and Half A calibration documents).
* **Metric:** False Alarm Rate (FAR) on clean real Half B panels under the frozen threshold (and also reported at the `best.pt` val-only threshold, which never saw any real panels).
* **Target:** **<= 2.0%** clean false alarm rate on **Half B**.

### Target 2: Seam Shortcut Suppression on Synthetic v3 Test
* **Dataset:** `data/synthetic_v3/test` (300 triplets: 300 pristine, 300 unrelated_paste, 300 forged).
* **Metric A (Relative Seam Parity):** False Positive Rate (FPR) on `unrelated_paste` must be **<= 1.5x** the FPR on `pristine`.
* **Metric B (Absolute Seam False Alarms):** FPR on `unrelated_paste` must be **<= 3.0%** absolute.
* **Criterion:** Both Metric A and Metric B must be met simultaneously, confirming that the new model no longer relies on paste boundary seams as a detection shortcut.

### Target 3: Synthetic v3 Forged Recall
* **Dataset:** `data/synthetic_v3/test` (300 forged positive panels).
* **Metric:** Image-level Recall on `forged` panels evaluated at the operating thresholds defined by Targets 1 and 2 (FPR <= 3.0% on unrelated_paste, FAR <= 2.0% on real pristine).
* **Target:** **>= 30.0%** image-level recall.

### Target 4: Real Forged Recall Protocol (Exploratory Only)
* **Dataset:** Locked real BioFors IDD test split (91 forged panels, 2,195 clean negative panels).
* **Rule:** **No pre-registered target number.**
* **Protocol:** Because the real BioFors test split was one-shot evaluated in Phase 4 (counter 1/1, final), any subsequent evaluation on this locked set will be run **exactly once at the very end** and strictly labeled as **"Exploratory Evaluation"**.
* **Integrity Guarantee:** The locked BioFors test split will never be used for tuning, model selection, or threshold calibration.

---

## 3. Evaluation Protocol and Methodology

1. **Document-Level Disjointness:** Splitting is performed strictly by document ID. None of the 177 held-out pristine documents are ever seen in `data/synthetic_v3/train`.
2. **Model Selection Criterion:** The checkpoint will be selected by validation recall at a 1.0% False Positive Rate budget over combined pristine plus unrelated_paste negatives (not by validation loss).
3. **Decision Rule:** Image detection requires an above-threshold predicted mask with connected component area exceeding the minimum area threshold (`min_area_pixels`).
4. **Style and Communication:** No em dashes are permitted in docs, code comments, or logs.
