# Doppel: CNN v3 plan (paste this to the coding agent)

## Context
Project: Doppel, a hybrid copy-move forgery detector for scientific images (western blots, microscopy). Repo: forgery-detection. Branches so far: phase3-synthetic, phase4-cnn.

Detectors: block-based DCT (frozen and adapted), SIFT (frozen), and a ResNet-18 correlation CNN (corr v2, last.pt).

Measured facts you must respect:
- Synthetic test: CNN recall 49.3% at 0/300 false alarms, adapted DCT 38.3%, SIFT 16.0%.
- Real BioFors IDD test (91 forged, 2,195 clean, blot + microscopy), ONE-SHOT, already used (counter 1/1, final): CNN recall 2.2% (2/91) at 1.23% false alarms, 0/17 on microscopy. Adapted DCT OR SIFT: 48.4% recall at 3.78% false alarms. Adapted DCT AND SIFT: 12.1% recall at 0.09% false alarms.
- Corr v1 had about 12.5% false alarms on real pristine panels.

Working diagnosis (a hypothesis, not proven): the CNN learned generator artifacts (rectangular pastes, clean seams, narrow transforms) instead of "two regions have the same content". Pretrained DOA-GAN also fails on real panels, which supports a domain-shift explanation.

## Hard rules
1. NEVER touch the real BioFors test split again. It is spent. Any real-test number reported later is labeled "exploratory".
2. Split by DOCUMENT everywhere, never by panel.
3. Pre-register targets in `docs/phase5_targets.md` and commit it BEFORE any training run.
4. ONE retrain only. No tuning loops against dev beyond threshold selection.
5. Explain before you code: for every script, give a short "why this exists and what it does" first. I have a viva and must be able to explain every line. Keep code simple and commented.
6. Do not silently change file formats, label conventions or the existing loader. If anything is ambiguous (mask convention, doc id from filenames, image layout), STOP and ask me.
7. Commit after each task on a new branch `phase5-cnn-v3`.
8. No em dashes in docs or comments.

## Task A: Diagnose (no training)
A1. Count real forged panels in the BioFors TRAIN split (not test), per class (blot/gel vs microscopy), and how many have usable region masks. Report a table. This decides whether real positives go into training.
A2. For adapted DCT and frozen SIFT on real BioFors, report per-class recall and false alarms, plus localization quality (mask IoU or F1) where masks exist. Output a markdown table in `docs/phase5_diagnosis.md`. Use only data that is NOT the locked test split, or reuse the numbers already stored from the one-shot run, do not rerun the locked test.
A3. Seam test on the CURRENT corr v2 model: use the new generator's `unrelated_paste` images (see Task B). Report image-level false positive rate on `unrelated_paste` vs `pristine`. If the rate on unrelated_paste is clearly higher, seams are a learned shortcut (hypothesis confirmed). If equal, STOP and report to me before building anything else.

## Task B: Data v3
B1. Use `make_synthetic_v3.py` (provided) as the generator. Read it fully and explain it to me in plain words before running. It makes TRIPLETS from each base image: pristine (clean), forged (copy from same image), unrelated_paste (patch from a different document through the identical pipeline, clean). Irregular masks, feather or Poisson blending, rotation/scale/flip/photometric/blur/noise on the pasted copy, whole-image JPEG with the same probability for all classes.
B2. Run it on BioFors TRAIN pristine images only (blot/gel and microscopy). Check how BioFors filenames encode the paper/document and set `--doc-key` accordingly (`prefix`, `dir`, or adapt the function). Verify with `audit_v3.py` that there is zero document leakage.
B3. Run `audit_v3.py`. Seam AUC (forged vs unrelated_paste ring energy) should be near 0.5. If it is far from 0.5, fix the generator, do not proceed.
B4. Real dev set: carve out documents from the BioFors TRAIN split (never the test split), document-level, include real forged panels if A1 shows any with masks, plus real pristine panels. This set is NEVER trained on. It is the only place thresholds and targets are judged.
B5. Real forged panels from BioFors train (only if A1 says there are enough): add them to training as positives, document-split from the dev set. Report counts. If there are fewer than about 30 usable panels, do not use them in training, use them only in dev.
B6. Adapt the existing loader to read v3 (images, `mask` = union of source and target). Training label for `unrelated_paste` and `pristine` is an EMPTY mask. `mask_paste` is for audits only and must NEVER be used as a label. Package a zip for Colab like before.
B7. Quick shortcut audit on v3 before spending GPU time: train a tiny baseline (e.g. 3 epochs, small subset) and check its false positive rate on `unrelated_paste` vs `pristine`. They should be similar.

## Task C: Targets, train, decide
C1. Write `docs/phase5_targets.md` BEFORE training, with numbers decided now. Suggested starting point (I will confirm):
  - On the real dev set, at the dev-tuned threshold: false alarm rate on real pristine panels at most 2%, and false positive rate on unrelated_paste within 1.5x of pristine.
  - Success as a branch: recall above adapted DCT OR SIFT on dev at equal or lower false alarms.
  - Success as a verifier: keeps at least 90% of the true forged panels that DCT/SIFT already catch while cutting their false alarms by at least half.
C2. Train ONE run on Colab T4 using the existing corr architecture (dual-softmax or ZeroWindow only if I explicitly approve it). Mix in hard negatives: pristine multi-lane blots and dense cell fields should make up a large share of clean samples.
C3. Evaluate on real dev against the pre-registered targets. Re-run the seam test (unrelated_paste FPR vs pristine FPR) on the new model.
C4. Report which outcome we got: full third branch, verifier on DCT+SIFT candidates, or documented negative result. Do not oversell.

## Deliverables
- `docs/phase5_diagnosis.md`, `docs/phase5_targets.md`
- generator + audit committed, dataset zip for Colab
- training notebook, dev evaluation script, seam test script
- a short viva note per component: what it does, why it exists, one failure mode

## If anything contradicts these instructions
Stop and ask me. Do not improvise.
