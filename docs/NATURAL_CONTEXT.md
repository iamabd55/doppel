# Doppel natural-image context (read this first in every task)

Project: Doppel, a hybrid copy-move forgery detection and localization system for natural images (photo verification for news and social media). Course topic approved by the teacher: Copy-Move Forgery Detection and Localization using Block-Based DCT Features, extended to a hybrid of DCT, SIFT and a pretrained deep model with fusion.

History: the project started on scientific images (BioFors). That work is kept on tag scientific-phase5 and becomes a short domain-shift appendix. Key finding there: a CNN trained on synthetic scientific forgeries learned paste seams, not duplication (it fired on unrelated pastes 2.86x more than on pristine panels, 6.67% vs 2.33%).

Assets: DCT code and frozen config, SIFT code and frozen config (tuned on CoMoFoD), CoMoFoD in data/comofod, pretrained DOA-GAN weights already in the repo (path in docs/natural_inventory.md), an old Flask dashboard, data/natural/casia2, data/natural/casia2_masks, data/natural/coverage.

Datasets: the authors' CASIA-CMFD and USC-ISI files are no longer available. We build our own CASIA v2.0 copy-move subset (Tp copy-move images with ground-truth masks, Au authentic images as negatives). CoMoFoD is the DEVELOPMENT benchmark because DCT and SIFT were tuned on it. COVERAGE is an optional small extra test. DOA-GAN was trained on USC-ISI, so it is cross-dataset on all of these.

Rules:
- NO deep model training. Pretrained DOA-GAN inference only.
- Never tune anything on a test split. Calibration uses the CASIA calibration split only.
- The protocol (docs/natural_protocol.md) is written BEFORE any evaluation and is not edited afterwards.
- Test splits are evaluated in ONE final pass (see Prompt 6). Aggregation scripts refuse to print test-split metrics unless run with --final.
- Metrics: image-level recall and false alarm rate with Wilson 95% confidence intervals, pixel-level F1 and IoU on forged images only.
- Commit code and docs only. Never commit data, weights or results caches.
- Explain every script in plain words before writing it. The student must be able to explain every line in a viva.
- No em dashes anywhere. Ask when unsure.
