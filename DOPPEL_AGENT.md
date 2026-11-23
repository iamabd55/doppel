# DOPPEL_AGENT.md

Briefing for the coding agent working on **Doppel** (Scientific Image Integrity).
Tagline: "See the copy. Reveal the manipulation."
Repo: https://github.com/iamabd55/doppel (renamed from forgery-detection, GitHub redirects the old URL). The Python package name `detector/` is intentionally NOT renamed.
Owner: Abdul Rehman Ali ("ABD"), COMSATS University Islamabad, Lahore Campus. DIP course project.

Environment: Ubuntu (Linux), Python 3.11+, `uv`. ABD has NO local GPU. Everything through Phase 3 runs on CPU locally. CNN training (Phase 4 and the optional U-Net in Phase 5) happens on Google Colab, so write those scripts to run both ways: a single training script with a config file, paths and device chosen by arguments, checkpoints saved to Google Drive, and small precomputed files (synthetic dataset zip, saved weights) that are easy to upload and download. Local inference must still work on CPU (cap image size, batch size 1). The full PRD (problem, solution, architecture, tech stack, references) exists as a separate document, but this file is enough to work from. If the two ever disagree, this file wins for implementation and ABD decides for scope.

Read this whole file before doing anything. Re-read the "Rules" and "Current step" sections at the start of every session.

The agentlogs/ folder is the project's living documentation. Updating it is part of every task. A task is NOT finished until the logs are updated. Before starting any task, read these files first so you know the current state.
Files
agentlogs/architecture.md: architecture and data flow
agentlogs/files.md: summary of every file
agentlogs/changelog.md: running history of changes
When to update
After ANY change that creates, deletes, renames, or meaningfully edits a file, or changes how data moves. Do it in the same task, before telling the user you are done. Do not batch updates for later.


---

## 1. What we are building

A small research prototype that finds copy-move forgery (a region copied and pasted inside the same image) in scientific images such as western blots and microscopy photos. It is a hybrid of three detectors:

1. **DCT** block-based detector. DONE.
2. **SIFT** keypoint detector with geometric verification. DONE.
3. **Synthetic Generator** paired-twin benchmark and Colab package. DONE.
4. **CNN** learned-feature detector (pretrained backbone + self-correlation). NEXT.

Then a fusion step, an ablation study, scientific-image evaluation, a basic report, and (last) a thin product wrapper.

Doppel is inspired by Proofig AI but is NOT a replica. It is a prototype. Do not claim parity with any commercial tool. Do not quote any vendor accuracy numbers.

Research title: "A Hybrid Computer Vision Framework for Copy-Move Forgery Detection in Scientific Images".

---

### Architecture in one view (matches the Doppel diagram)

Research pipeline: input image -> preprocessing (grayscale, normalize, resize) -> three parallel branches (DCT: block DCT, features, block matching; SIFT: keypoints, descriptors, ratio test + RANSAC, flip pass; CNN: ResNet/VGG features, self-correlation map) -> hybrid fusion -> localization (optional small U-Net, dashed) -> final mask and visualization. Evaluation strip: original images + ground truth, SIFT evaluation, synthetic generation, CoMoFoD (test set), scientific figures, ablation, metrics (precision, recall, F1, false alarm rate).

Product (built last, Phase 7): Next.js frontend -> FastAPI (validation, orchestrator, report) -> queue (Celery + Redis, only when needed) -> detection engine as worker -> file storage (local disk, later MinIO) and PostgreSQL metadata. Optional and later: PDF figure extraction, panel splitting, review flags, report export.

Development phases: 1 DCT baseline, 2 SIFT, 3 synthetic generator, 4 CNN, 5 fusion and ablation, 6 scientific image evaluation, 7 product wrapper.

---

## 2. Rules (follow strictly)

### How to work with ABD
- Build **one step at a time**. Never implement several phases at once.
- Before each step's code, explain **why** the step exists, in simple words. Then the code. Then how to check it works.
- One step per message. Wait for ABD to confirm before moving on.
- Style: casual, direct, concise. Honest over polished. **No em dashes** anywhere (code comments, docs, messages). Use commas, periods or hyphens.
- If something ABD asks seems wrong or risky, say so clearly and briefly, then offer the alternative.

### Research honesty
- Never present a planned feature as achieved. Only report numbers that were actually measured by a script in this repo.
- Do not start the next phase until the current phase has written results (a CSV and a short markdown summary in `results/`).
- Do **not** silently change the DCT baseline. Its parameters are frozen (see section 4). New settings go in separate config files and are reported next to the baseline.
- Do not delete or replace `web/` (the Flask DCT demo). It is the working baseline demo. When adding SIFT, CNN or fusion, do not change the public function signatures that `web/app.py` and `scripts/` already use. Add new modules and new functions instead, so the baseline demo keeps running.
- Same benchmark, same splits, same metrics for every branch.
- Failures are results. Report them.

### Git workflow
- The commit tagged `dct-baseline` is the frozen DCT baseline. Never move or delete that tag.
- One branch per phase: `phase2-sift`, `phase3-synthetic`, `phase4-cnn`, `phase5-fusion`, `phase6-scientific`, `phase7-product`. Work happens on the phase branch, never directly on `main`.
- Small commits per step with clear messages (for example `sift: add keypoint detection and visualization`). Commit results files together with the code that produced them.
- Merge a phase into `main` only when its results exist in `results/` and ABD confirms. ABD does the merge or approves it.

### Code quality
- Python 3.11+, managed with `uv`. Type hints on public functions. Small functions. No giant scripts.
- Every experiment is reproducible: fixed random seeds, config in a YAML file, outputs saved under `results/`.
- Add `pytest` tests for matching logic, coordinate mapping (especially the flip pass), and the synthetic generator.
- Do not commit datasets or model weights. Keep them in `data/` and `models/` (both git-ignored).

---

## 3. Current step

**Phase 4 Wrap-Up Step 1 (Thresholds Selection) Complete; Ready for Step 2 (Inference):**
- **Threshold Selection (`scripts/select_threshold.py`):** Executed once on CPU using `checkpoints/corr_v2/last.pt` on all 4,039 held-out pristine panels and 1,000 synthetic validation images (500 clean twins, 500 forged).
- **Prediction Caching & Provenance:** Cached per-image outputs to `checkpoints/corr_v2/val_pristine_preds.npz` (59.5 KB), enabling instant (<4s) full grid sweeps. Hardened with strict provenance validation (`verify_cache_provenance`) across checkpoint/data sizes, mtimes, subsets, and architectures to reject stale caches. All 52 tests pass.
- **Operating Points:** Evaluated compound decision rule (s x tau x min_area) across FAR budgets (1.0%, 1.5%, 2.0%, 3.0%) for Overall, Blot/Gel, and Microscopy with 95% Wilson confidence intervals.
  - Primary 1.0% Budget:
    - Overall: Recall 51.4% [47.0-55.8], s=0.99, tau=0.70, area=25 px (Synth Clean FAR: 1.00%, Real Pristine FAR: 0.69%)
    - Blot/Gel: Recall 68.7% [62.7-74.1], s=0.95, tau=0.30, area=25 px (Synth Clean FAR: 0.79%, Real Pristine FAR: 0.86%)
    - Microscopy: Recall 23.8% [18.9-29.5], s=0.99, tau=0.30, area=1600 px (Synth Clean FAR: 1.21%, Real Pristine FAR: 0.55%, lowest max FAR fallback)
- **Configuration Locked:** Exported locked operating parameters into `configs/cnn_threshold.yaml`.
- **Test Integrity:** BioFors IDD test set and synthetic test set strictly untouched (touch count: 0).
- **Next Step:** Step 2: Inference (`detector/cnn/infer.py`) standardized predict interface matching DCT and SIFT detectors.

(Update this line when a step finishes.)

---

## 4. Existing repo (verified from GitHub)

```
doppel/   (previously forgery-detection)
  detector/      detect.py, features.py, matching.py, evaluate.py
  web/           Flask dashboard (app.py, templates/, static/)
  scripts/       batch_test.py, false_alarm_test.py, experiments.py, make_test.py, try_001.py
  figures/       robustness.png
  learning/      dct_demo.py
  pyproject.toml, uv.lock, README.md
```

Data (not in repo, local only): `data/comofod/` with translation, rotation, scaling, distortion, combination folders (sets 001 to 200, 40 per category), plus originals (`*_O.png`).

### Frozen DCT baseline parameters
- Dense 8x8 sliding window, grayscale
- Texture filter: block std >= 4.0
- DCT via `scipy.fft.dctn`, zigzag, keep first K coefficients (K between 6 and 16)
- Quantization: divide by Q=8, round
- Lexicographic sort (`numpy.lexsort`), neighbor window k = 1..5
- Min distance between matched blocks: 30 px
- Shift vector voting, min votes 150, canonical direction
- Morphological closing 9x9, remove objects smaller than 500 px

### Baseline results (CoMoFoD small, 40 sets per category)
| Category | Detection | Avg F1 |
|---|---|---|
| Translation | 36/40 (90.0%) | 0.817 |
| Rotation | 5/40 (12.5%) | 0.082 |
| Scaling | 1/40 (2.5%) | 0.021 |
| Distortion | 8/40 (20.0%) | 0.145 |
| Combination | 1/40 (2.5%) | 0.018 |

False alarms on 40 originals: 17/40 without texture filter, 0/40 with it.

### README rules
- The README was rewritten for the Doppel rename. It has a Status table, a Limitations section, and a "DCT vs Keypoint Methods" table where the SIFT column is marked "from literature, not measured yet".
- After each phase finishes, update the Status table and replace "to be tested" claims with measured numbers from `results/`. Never write a number in the README that is not in a results file.
- Keep the README honest: planned work is labeled "Planned", finished work "Done".
- No em dashes, no marketing words ("advanced", "high-performance", "robust") unless backed by a measured result.
- Keep `DOPPEL_AGENT.md` in the repo root. Keep the tag `dct-baseline` untouched.

---

## 5. Locked technical decisions

| Area | Decision |
|---|---|
| SIFT | OpenCV `cv2.SIFT_create()`, 128-d descriptors |
| Matching | Within-image matching, kNN with the self-match removed, Lowe-style ratio test, minimum pixel distance between the two keypoints |
| Verification | RANSAC affine transform with `cv2.estimateAffine2D` (full 6-parameter affine, so mirrored copies can fit). Do NOT use `estimateAffinePartial2D`, it cannot represent a reflection. Require a minimum number of inliers |
| Flip | One extra pass: match original descriptors against descriptors of the horizontally flipped image, map coordinates back with `x = W - 1 - x'` |
| CNN | Pretrained ResNet or VGG backbone, features from a mid-level layer, self-correlation map, small decoder |
| Fusion | Start simple (union, intersection, weighted vote). Optional small U-Net with input channels [image, DCT mask, SIFT mask, CNN map] |
| Texture filter | Keep the original (std >= 4.0) as baseline. Test other thresholds as separate configs. Expect to retune for blots and microscopy |
| Data | CoMoFoD for established benchmark. Synthetic set for training and controlled tests. Scientific images for domain tests |

Notes:
- A vertical flip equals a horizontal flip plus a 180 degree rotation, and SIFT already handles rotation, so one horizontal flip pass is enough.
- SIFT starting values below (ratio threshold, RANSAC threshold, min inliers) are **starting points to tune**, not verified results.

---

## 6. Step-by-step plan

Each step: explain why, code, test, record results, then wait for ABD.

### Phase 1: DCT baseline (DONE)
Remaining small items: README fixes from section 4; make sure `results/` folder convention exists (`results/<phase>/<experiment>.csv` plus a `.md` summary).

### Phase 2: SIFT branch (DONE)

Evaluated on CoMoFoD with 50/50 tuning and held-out splits. Frozen at `cfg8_best_combo` (74.0% detection, 0.540 mean F1 on 100 held-out transformed images; 40.0% false alarm rate on 20 clean images). Failure analysis complete in `results/phase2_sift/failure_analysis.md`. SIFT is frozen as an expert geometric branch.

### Phase 3: Synthetic benchmark generator (DONE)

Built Paired-Twin architecture and Colab training package:
- 16,000 train samples (8,000 pairs), 1,000 val samples (500 pairs), 600 test samples (300 pairs) strictly partitioned by document ID.
- Chi-square independence verified across class x tier cells (chi2=0.0000, p=1.0000).
- Shortcut classifier audit passed (LogReg 50.57%, HistGB 51.78%, grouped CV by doc_id; all individual non-file-size features <= 51.3%).
- Benchmark evaluated on synthetic test split (600 samples) and real BioFors IDD test split (2,286 panels: 91 forged, 2,195 clean).
- Colab package generated in `data/synthetic_dataset.zip` (761 MB) excluding test split and BioFors real data, with full `PROVENANCE.txt`.
- Colab PyTorch dataset loader verified with unit tests (`tests/test_colab_dataset.py`).
- Phase 4 evaluation targets pre-registered in `docs/phase4_targets.md`.

### Phase 4: CNN branch (NEXT)

- **4.1 Backbone features.** Pretrained ResNet or VGG (frozen). Extract a mid-level feature map (for example stride 16).
- **4.2 Self-correlation.** L2-normalize features, compute the correlation of every location with every other, mask out self and near neighbors, keep top-k matches per location. Visualize.
- **4.3 Decoder.** Small convolutional decoder predicts the forgery mask. Loss: BCE + Dice. Train on the synthetic train split, validate on validation, keep the test split untouched.
- **4.4 Evaluation.** Score on synthetic test and on real BioFors IDD test benchmark. Compare with DCT and SIFT baselines. Save to `results/phase4_cnn/`.
- Pre-registered targets (docs/phase4_targets.md): CNN alone recall > 35.2% at clean FAR <= 1.50%; Hybrid fusion recall > 41.8% at clean FAR <= 2.50%; Pixel F1 > 0.1200.
- Threshold selection protocol: Strictly chosen on synthetic val plus 1,000 pristine train panels, never on BioFors test or synthetic test.
- Training executes on Google Colab GPU via `data/synthetic_dataset.zip` and `detector/colab_dataset.py`. Local CPU inference must still be supported.
- Optional: unfreeze part of the backbone only if frozen results are clearly limited.
- Training can happen on Colab or Kaggle if there is no local GPU. Keep the training script runnable on both.

### Phase 5: Fusion and ablation

- **5.1 Simple fusion.** Union, intersection, weighted vote of the three masks. Report each.
- **5.2 Learned fusion (optional).** Small U-Net with channels [image, DCT, SIFT, CNN]. Train on synthetic train split.
- **5.3 Ablation table.** Rows: DCT, SIFT, CNN, DCT+SIFT, DCT+CNN, SIFT+CNN, all three (and U-Net if built). Columns: translation, rotation, scaling, flip, combination, JPEG, noise, false alarm rate. Save to `results/phase5_fusion/`.
- Be honest if fusion does not help. That is still a finding.

### Phase 6: Scientific images and report

- Collect open-access figures (for example from the PubMed Central open access subset or bioRxiv), respecting licenses. Insert synthetic forgeries so masks exist. Evaluate. Real flagged cases are only a qualitative demo, never part of scores.
- Retune the texture filter for blots and microscopy as a separate config. Report both baseline and tuned settings.
- Basic report output: original, overlay, mask, matched pairs, confidence, which detector fired. PDF export.

### Phase 7: Product wrapper (only after Phases 1 to 5 have results)

Build in this order and add each tool only when needed:

1. **FastAPI** API: upload, validation, job creation, result fetch. Keep the old Flask app as the baseline demo.
2. **Next.js + TypeScript** dashboard: upload, overlay viewer, confirm/dismiss flags.
3. **PostgreSQL**: users, jobs, results, review decisions.
4. **File storage**: local disk first, S3-compatible (MinIO) later.
5. **Celery + Redis**: only when detection jobs are slow enough to block requests (CNN branch, PDFs with many figures). Redis is the broker. The worker runs the detection engine.
6. **Docker + Docker Compose**: once there are three or more services (API, worker, Redis, database). One command to start everything.
7. Nice to have: **PyMuPDF** for extracting images from PDFs, simple OpenCV panel splitting, review dashboard polish.

Not used on purpose: Kubernetes, microservices, cloud autoscaling.

Target layout (grow into this, do not create all of it upfront):
```
detector/        dct, sift, cnn, fusion, evaluate, synthetic
scripts/         experiment runners
configs/         yaml settings per experiment
results/         csv + md summaries per phase
api/             FastAPI (Phase 7)
worker/          Celery tasks (Phase 7)
web/             existing Flask demo of the DCT baseline. DO NOT delete, rewrite or overwrite it.
frontend/        new Next.js dashboard (Phase 7), separate folder
docker-compose.yml  (Phase 7)
```

---

## 7. Evaluation protocol (all phases)

- **Pixel-level:** precision, recall, F1 against ground-truth masks.
- **Image-level:** detection rate (image flagged or not).
- **False alarms:** rate on original (untouched) images.
- **Attacks:** rotation, scaling, flip, JPEG quality levels, noise levels, blur.
- **Splits:** CoMoFoD is test only. Synthetic data has train/validation/test split by source image. Seeds are fixed and recorded.
- **Reporting:** one CSV per experiment plus a short markdown summary. Include the config used.
- Compare against the DCT baseline table in section 4 every time.

---

## 8. Success criteria

1. SIFT clearly improves rotation, scaling and combination results over the DCT baseline.
2. Mirrored copies are detected on the synthetic set.
3. The CNN branch and the fusion are evaluated with a full ablation.
4. False alarms and failure cases are reported.
5. Nothing is claimed that was not measured.

---

## 9. Reference papers (verified)

1. Bik, Casadevall, Fang (2016). The prevalence of inappropriate image duplication in biomedical research publications. mBio 7(3), e00809-16.
2. Fridrich, Soukal, Lukas (2003). Detection of copy-move forgery in digital images. DFRWS.
3. Lowe (2004). Distinctive image features from scale-invariant keypoints. IJCV 60(2), 91-110.
4. Fischler, Bolles (1981). Random sample consensus. Communications of the ACM 24(6), 381-395.
5. Amerini et al. (2011). A SIFT-based forensic method for copy-move attack detection and transformation recovery. IEEE TIFS 6(3), 1099-1110.
6. Christlein et al. (2012). An evaluation of popular copy-move forgery detection approaches. IEEE TIFS 7(6), 1841-1854.
7. Tralic et al. (2013). CoMoFoD: new database for copy-move forgery detection. ELMAR-2013, 49-54.
8. Wu, Abd-Almageed, Natarajan (2018). BusterNet. ECCV, LNCS 11210, 170-186.
9. Islam et al. (2020). DOA-GAN. CVPR, 4676-4685.
10. He et al. (2016). Deep residual learning for image recognition. CVPR, 770-778.
11. Ronneberger, Fischer, Brox (2015). U-Net. MICCAI, LNCS 9351, 234-241.

Use these as the basis for method choices. If a method is taken from a paper, note the paper in the code docstring.

---

## 10. Session checklist for the agent

At the start of a session:
1. Read this file. Check section 3 for the current step.
2. Look at `results/` to see what is already measured.
3. Run `git status` and the existing tests.

At the end of a step:
1. Tests pass. Results saved under `results/`.
2. Update section 3 (Current step) in this file.
3. Give ABD a short summary: what changed, what was measured, what is next.
