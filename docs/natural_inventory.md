# Natural-Image Inventory (Doppel)

Generated from read-only inspection of the repository, datasets, and weights. No evaluation, training, or code modifications were performed.

---

## 1. DOA-GAN (`doa-gan_weights/` and `doagan_clean/`)

* **Weights location:**
  * `doa-gan_weights/three_channel.pkl` (`107,962,887` bytes, ~103 MB)
  * `doa-gan_weights/single_channel.pkl` (`107,960,800` bytes, ~103 MB)
* **Codebase and loading:**
  * Cloned into `doagan_clean/` from `https://github.com/asrafulashiq/doagan_clean`. (No prior audit script existed in the current branch.)
  * `doagan_clean/run_on_folder.py` instantiates `models.DOA(out_channel=args.out_channel)` (`3` or `1`), loads `torch.load(ckpt_path, map_location=device)["model_state"]` with `strict=True`, and sets `model.eval()`.
* **Input size and preprocessing:**
  * Fixed spatial size `(320, 320)` (`args.size = "320x320"` in `config.py`).
  * RGB image (`[..., :3]`) resized to `(320, 320)` with bilinear/linear interpolation, scaled to `[0.0, 1.0]`, and ImageNet-normalized (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) into tensor shape `[1, 3, 320, 320]`.
* **Outputs:** `model(x)` returns `(mask_logits, det_logit)`.
  * **Three-channel checkpoint (`out_channel=3`):** `torch.softmax(mask_logits, dim=-3)` gives shape `[1, 3, 320, 320]`. From `doagan_clean/dataset.py` (`np.dstack([tgt_mask, src_mask, background])`), channel `0` is **target** (pasted copy), channel `1` is **source**, and channel `2` is **background**. Combined forged probability heatmap is `prob[0] + prob[1]` (or `1.0 - prob[2]`).
  * **Single-channel checkpoint (`out_channel=1`):** `torch.sigmoid(mask_logits)` gives shape `[1, 1, 320, 320]`, representing combined source + target (`np.maximum(src_mask, tgt_mask)`).
  * **Image-level score:** `torch.sigmoid(det_logit)` scalar in `[0, 1]`. (In our 5-image check on CoMoFoD forged images, `det_score` was `1.0000` on all 5.)
* **Mapping masks back to original image size:**
  * DOA-GAN predictions are always `(320, 320)`. To compare against native-resolution ground-truth masks or fuse with DCT/SIFT, resize the probability map back to `(W_orig, H_orig)` via `cv2.resize(..., (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)` before thresholding (or `INTER_NEAREST` on binary masks).

---

## 2. CoMoFoD (`data/comofod/`)

* **Layout:** 5 attack subdirectories (`translation`, `rotation`, `scaling`, `distortion`, `combination`) plus `readme.txt` and `psnr.txt`.
* **Counts:**
  * **40** base IDs per category (`001..040` translation, `041..080` rotation, `081..120` scaling, `121..160` distortion, `161..200` combination) = **200** base sets total.
  * Each base ID has **52** PNG files (`2,080` files per category; `10,400` PNGs total):
    * **25** authentic images (`XXX_O.png` + 24 post-processed `O_*` variants) -> **5,000** authentic images total.
    * **25** forged images (`XXX_F.png` + 24 post-processed `F_*` variants) -> **5,000** forged images total.
    * **2** mask files (`XXX_B.png` binary mask, `XXX_M.png` colored/alpha mask) -> **200** of each.
* **Mask format:** `XXX_B.png` is `512x512x4` RGBA `uint8` with values `[0, 237, 255]` (binarize with `> 10` or `> 127`); marks **both** source and target regions.
* **File-name attack encoding:**
  * Subdirectory gives geometric attack (`translation`, `rotation`, `scaling`, `distortion`, `combination`).
  * Suffix gives post-processing: `BC1..3` (brightness/contrast), `CA1..3` (color adjustment), `CR1..3` (contrast/gamma), `IB1..3` (image blurring), `JC1..9` (JPEG compression), `NA1..3` (noise adding).

---

## 3. CASIA v2.0 & COVERAGE (`data/natural/`)

### CASIA v2.0 (`data/natural/casia2/` and `data/natural/casia2_masks/`)
* **Layout and raw counts (`data/natural/casia2/CASIA2/`):**
  * `Au/`: **7,491** authentic images (`7,437 .jpg`, `54 .bmp`, plus `Thumbs.db`). Naming: `Au_<category>_<id>.<ext>`.
  * `Tp/`: **5,123** tampered images (`3,059 .tif`, `2,064 .jpg`, plus `Thumbs.db` and `1 .txt`).
  * `CASIA 2 Groundtruth/`: **5,123** PNG masks named `<corrected_stem>_gt.png`.
* **Ground-truth repo (`data/natural/casia2_masks/`):**
  * Contains `CASIA2.0_Groundtruth.zip` (`5,123` PNGs), `Notes/fileNamesCorrection.xlsx`, `Notes/au_list.txt`, `Notes/tp_list.txt`, `Notes/groundtruth_list.txt`, and `Samples/`.
* **Copy-move vs splicing in `Tp` filenames:**
  * `Tp_S_...` = Same-image tampering (**copy-move**); `Tp_D_...` = Different-image tampering (**splicing**).
  * Raw `Tp/` filenames have `3,274` `Tp_S_*` and `1,849` `Tp_D_*`.
  * `Notes/fileNamesCorrection.xlsx` lists **141** filename corrections (`39` `S -> D`, `60` `D -> S`, plus ID/typo fixes). After applying all 141 corrections, there are **3,295** copy-move (`Tp_S_*`) images and **1,828** spliced (`Tp_D_*`) images.
  * **Mask mapping:** Without corrections, `3,216/3,274` raw `Tp_S_*` stems match `_gt.png`. After applying the `141` renames from `fileNamesCorrection.xlsx`, **3,295 / 3,295 (100%)** copy-move images match `<corrected_stem>_gt.png`.
* **Mask semantics (checked 20 `Tp_S_*` masks):**
  * Binary grayscale PNGs (`[0, 255]`, matching image resolution, e.g. `256x384`).
  * **19/20** checked copy-move masks had **1 connected component** (1/20 had 5 components), confirming that CASIA2 ground-truth masks mark **only the pasted (target) region**, not both source and target.

### COVERAGE (`data/natural/coverage/`)
* Currently contains only `.git/` and `README.md` (links to external OneDrive/Baidu downloads). **0** images and **0** masks are present on disk.

---

## 4. Frozen DCT and SIFT Configs

* **DCT (`detector/detect.py`, `configs/dct_adapted.yaml`, `scripts/benchmark.py`):**
  * **CoMoFoD baseline (`dct-baseline` tag):** `detect(gray, B=8, K=16, Q=4, min_std=4.0, window=5, min_dist=30, min_votes=50, kernel=9, min_size=500)` in `detector/detect.py`.
  * **Frozen scientific baseline / adapted (`configs/dct_adapted.yaml`):** tuned on scientific synthetic val (`Q=8, K=6, min_votes=150, min_size=500` frozen; `min_votes=75, min_size=64` adapted).
  * **How to run on one image:** `mask = detect(gray_uint8, ...)` where `gray_uint8` is 2D grayscale `uint8`.
  * **Outputs:** `mask` (`uint8` `(H, W)` in `{0, 255}`). Image decision is `bool((mask > 0).any())`; no continuous score is returned directly (area of components or vote counts can be derived).
* **SIFT (`detector/sift/localization.py`, `scripts/benchmark.py`):**
  * **Frozen `cfg8_best_combo`:** tuned on CoMoFoD. Parameters in `scripts/benchmark.py`: `ratio_threshold=0.70, min_dist=30.0, ransac_threshold=3.0, min_inliers=6, cluster_threshold=100.0, mask_radius=12, use_convex_hull=True, check_flip=True, min_inlier_ratio=0.25, min_inliers_flip=10, filter_orientation=True, orientation_tolerance=20.0`.
  * **How to run on one image:** `mask, pts1, pts2, transforms = detect_sift(gray_or_bgr, ...)` in `detector/sift/localization.py`.
  * **Outputs:** `mask` (`uint8` `(H, W)` in `{0, 255}`), `pts1`, `pts2` (matched inlier coordinates), `transforms`. Image decision is `bool((mask > 0).any())`; image score is inlier count `len(pts1)`.

---

## 5. Dependencies & DOA-GAN CPU Timing

* **Environment (`.venv`):** Python 3.11, PyTorch, torchvision, OpenCV, scikit-image, scipy, numpy, pandas, matplotlib are installed.
* **Dependency / offline caveats:**
  * `doagan_clean/utils.py` imports `imgaug` (not installed in `.venv`); preprocessing can be done directly with `cv2` + `numpy` without importing `utils.py`.
  * `doagan_clean/models.py` (`Extractor_VGG19`) calls `torchvision.models.vgg19(pretrained=True)` before `load_state_dict(..., strict=True)` overwrites all weights; offline environments without cached `vgg19-dcbb9e9d.pth` must patch `vgg19(weights=None)` before loading `ckpt["model_state"]`.
  * `pandas.read_excel` requires `openpyxl` (not installed in `.venv`); `fileNamesCorrection.xlsx` can be parsed directly with Python stdlib `zipfile` + `xml.etree.ElementTree`.
* **5-image CPU timing test (`data/comofod/translation/001_F.png..005_F.png` at `320x320`):**
  * `three_channel.pkl` (`out_channel=3`): `0.736s, 0.502s, 0.502s, 0.498s, 0.505s` -> **mean 0.549s/image** (~0.50s steady state).
  * `single_channel.pkl` (`out_channel=1`): **0.531s** on `005_F.png`.
  * **Conclusion:** DOA-GAN runs cleanly on CPU at ~0.5s per image.
