# Phase 4: Target Metrics and Evaluation Protocol

These evaluation targets are formally documented and locked BEFORE commencing Phase 4 CNN training.

---

## 1. Context and Baseline Reference

In Phase 3, we evaluated two frozen classical baselines (Frozen DCT, Frozen SIFT) and one validation-adapted baseline (Adapted DCT) on the real BioFors IDD test benchmark (2,286 panels: 91 manipulated panels across Blot/Gel and Microscopy, and 2,195 clean negative panels):

- **Frozen DCT (`votes=150, min_size=500`):**
  - Image Recall (Overlap): 14/91 (15.4%) [9.4%, 24.2%]
  - Clean False Alarm Rate: 8/2,195 (0.36%) [0.18%, 0.72%]
  - Pixel F1: 0.0973

- **Adapted DCT (`votes=75, min_size=64`, tuned on synthetic VAL only):**
  - Image Recall (Overlap): 23/91 (25.3%) [17.5%, 35.1%]
  - Clean False Alarm Rate: 36/2,195 (1.64%) [1.19%, 2.26%]
  - Pixel F1: 0.1154

- **Frozen SIFT (`cfg8_best_combo`):**
  - Image Recall (Overlap): 32/91 (35.2%) [26.1%, 45.4%]
  - Clean False Alarm Rate: 49/2,195 (2.23%) [1.69%, 2.94%]
  - Pixel F1: 0.1126

- **Classical OR Fusion (Adapted DCT OR Frozen SIFT):**
  - Image Recall (Overlap): 44/91 (48.4%) [38.4%, 58.5%]
  - Clean False Alarm Rate: 83/2,195 (3.78%) [3.06%, 4.66%]

---

## 2. Phase 4 Target Metrics (Pre-Registration)

### Target A: CNN Alone on Real BioFors IDD Test Set
- **Dataset:** 91 manipulated panels (74 Blot/Gel, 17 Microscopy) and 2,195 clean panels.
- **Image Recall (Overlap):** **> 35.2%** (surpassing Frozen SIFT's individual recall on the same real benchmark).
- **Clean False Alarm Rate:** **<= 1.50%** (strictly lower than SIFT's 2.23% and Adapted DCT's 1.64%).
- **Threshold Selection Rule:** The operational decision threshold must be selected strictly using synthetic validation data (`val_manifest.csv`) combined with the 1,000 real pristine train panels from unused train documents. Neither the BioFors test split nor the synthetic test split may be used for threshold selection.

### Target B: CNN + Classical Hybrid Fusion
- **Image Recall (Overlap):** **> 41.8%** (surpassing Frozen DCT OR SIFT baseline, targeting >= 50.0%).
- **Clean False Alarm Rate:** **<= 2.50%** (substantially tighter than pure Adapted DCT OR SIFT at 3.78%).
- **Fusion Decision Protocol:** To be frozen on validation data.

### Target C: Localization Quality (Pixel Level)
- **Pixel F1 Score:** **> 0.1200** pooled across all test panels (exceeding Adapted DCT's 0.1154 and SIFT's 0.1126).
- **Pixel MCC:** **> 0.1500** against union ground-truth bounding box masks.

---

## 3. Methodological Rules and Constraints
1. **Decision Threshold Protocol:** Detection thresholds will be chosen strictly on synthetic validation data (`val_manifest.csv`) plus the 1,000 real pristine train panels from unused train documents. Thresholds are NEVER tuned or chosen on the BioFors test split or synthetic test split.
2. **Real Benchmark Evaluation Subset:** The real evaluation benchmark subset is strictly Blot/Gel + Microscopy (exactly 91 forged manipulated panels and 2,195 clean negative panels). Macroscopy and FACS categories are excluded as established in Phase 3.
3. **Zero Test Set Tuning:** No hyperparameters, loss weights, or detection thresholds will be selected on the BioFors IDD test set or the synthetic test set.
4. **Document-Level Disjointness:** Train, validation, and test splits are strictly partitioned by document ID. Zero images or donor patches from the validation or test sets are present during training.
5. **No Blind Resizing:** Scientific images (especially narrow Western blot strips with heights under 30 px) must not be arbitrarily resized via bilinear or bicubic interpolation, as resizing introduces interpolation artifacts and distorts native pixel noise.
6. **Cautious Language:** Any positive output is framed as a "suspicious region needing human review", never as proof of misconduct.
