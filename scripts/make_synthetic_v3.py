#!/usr/bin/env python3
"""
make_synthetic_v3.py: Doppel synthetic copy-move generator, version 3.

Why v3 exists
-------------
v1/v2 forgeries were rectangles with clean seams. The CNN got 49% recall on
synthetic test but 2.2% on real BioFors, so it most likely learned generator
artifacts (box shape, seam edges) instead of "these two regions are the same
content".

Design rule: every base image yields a TRIPLET, and all three go through the
same post-processing so no cue except real duplication separates them.

  pristine         : the untouched base image                     (label: clean)
  forged           : a region copied from the SAME image, pasted   (label: forged)
                     elsewhere in the same image
  unrelated_paste  : a region taken from a DIFFERENT document,     (label: clean)
                     pasted with the identical pipeline
                     (same mask shapes, transforms, blending, seams)

If a model fires on unrelated_paste it has learned seams, not duplication.

Other changes vs v2
-------------------
* irregular blob masks (not rectangles), including elongated band-like shapes
* two blend modes: feathered alpha and Poisson (cv2.seamlessClone)
* patches must contain structure (std and gradient filters), so blank background is never the copied content
* wide transform range on the pasted patch: rotation, scale, flip,
  brightness/contrast, blur, noise
* whole-image JPEG recompression applied to ALL classes with the same
  probability (so JPEG is not a cue)
* split by DOCUMENT (hash of doc id), never by panel, so no leakage

Outputs (per split)
-------------------
  <out>/<split>/images/<id>.png
  <out>/<split>/mask_target/<id>.png   255 where pasted copy sits (forged only, else zeros)
  <out>/<split>/mask_source/<id>.png   255 where the original region is (forged only, else zeros)
  <out>/<split>/mask/<id>.png          union of source and target (what a localizer predicts)
  <out>/<split>/mask_paste/<id>.png    where ANY pasted patch sits (forged and unrelated_paste); audits only, never a training label
  <out>/<split>/meta.jsonl             one json per image: id, group, base, doc, kind, params
  <out>/manifest.json                  settings, seed, counts

Usage
-----
  python make_synthetic_v3.py --src /path/to/pristine_images --out data_v3 \
      --n-train 3000 --n-val 300 --n-test 300 --seed 0 --doc-key prefix

--n-* is the number of TRIPLETS (so images = 3x that). Documents are grouped
with --doc-key: "dir" = parent folder name, "prefix" = filename text before
the first underscore, "stem" = every file is its own document (only use this if
you are sure panels of one paper are not in the same folder tree).
CHECK that the doc grouping matches how your BioFors files are named.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def doc_id(path: Path, mode: str) -> str:
    if mode == "dir":
        return path.parent.name
    if mode == "prefix":
        return path.stem.split("_")[0]
    return path.stem


def split_of(doc: str, fr_val: float, fr_test: float, seed: int) -> str:
    h = int(hashlib.sha1(f"{seed}:{doc}".encode()).hexdigest(), 16) % 10000 / 10000
    if h < fr_test:
        return "test"
    if h < fr_test + fr_val:
        return "val"
    return "train"


def load_rgb(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def blob_mask(bw, bh, rng):
    """Irregular blob inside a (bh, bw) box: radial polygon with smoothed jitter."""
    n = rng.randint(8, 16)
    ang = np.sort(np.array([rng.uniform(0, 2 * np.pi) for _ in range(n)]))
    rad = np.array([rng.uniform(0.55, 1.0) for _ in range(n)])
    # smooth radii so shapes are organic, not spiky
    rad = (rad + np.roll(rad, 1) + np.roll(rad, -1)) / 3
    cx, cy = bw / 2, bh / 2
    pts = np.stack([cx + rad * np.cos(ang) * bw / 2, cy + rad * np.sin(ang) * bh / 2], 1)
    m = np.zeros((bh, bw), np.uint8)
    cv2.fillPoly(m, [pts.astype(np.int32)], 255)
    if rng.random() < 0.5:  # roughen edges
        k = rng.choice([3, 5])
        m = cv2.GaussianBlur(m, (k, k), 0)
        m = (m > 127).astype(np.uint8) * 255
    return m


def sample_patch(src_img, rng, min_area=150, min_std=8.0, grad_pct=75):
    """Cut an irregular patch (image crop + mask) out of src_img."""
    H, W = src_img.shape[:2]
    # structure map: smoothed gradient magnitude. A patch must be more structured than
    # the grad_pct-th percentile of the image, so blank membrane / dark background is
    # never the copied content, while faint noise and JPEG blocks do not count as structure.
    g0 = src_img.mean(axis=2) if src_img.ndim == 3 else src_img
    g0 = cv2.GaussianBlur(g0.astype(np.float32), (0, 0), 1.0)
    G = np.hypot(cv2.Sobel(g0, cv2.CV_32F, 1, 0), cv2.Sobel(g0, cv2.CV_32F, 0, 1))
    g_thr = np.percentile(G, grad_pct)
    for _ in range(60):
        bw = int(rng.uniform(0.08, 0.35) * W)
        bh = int(rng.uniform(0.15, 0.60) * H)
        if rng.random() < 0.35:  # band-like (elongated) shapes for blots
            if W >= H:
                bw, bh = int(rng.uniform(0.10, 0.30) * W), int(rng.uniform(0.15, 0.45) * H)
            else:
                bw, bh = int(rng.uniform(0.15, 0.45) * W), int(rng.uniform(0.10, 0.30) * H)
        bw, bh = max(bw, 16), max(bh, 16)
        bw, bh = min(bw, W - 2), min(bh, H - 2)
        if bw < 16 or bh < 16:
            continue
        m = blob_mask(bw, bh, rng)
        if (m > 0).sum() < min_area:
            continue
        x0 = rng.randint(0, W - bw)
        y0 = rng.randint(0, H - bh)
        crop = src_img[y0:y0 + bh, x0:x0 + bw]
        # skip blank background: copying empty membrane is not a meaningful forgery
        gray = crop.mean(axis=2) if crop.ndim == 3 else crop
        if gray[m > 0].std() < min_std:
            continue
        if G[y0:y0 + bh, x0:x0 + bw][m > 0].mean() < g_thr:
            continue
        return crop.copy(), m, (x0, y0, bw, bh)
    return None


def transform_patch(patch, mask, rng, mode_rich=True):
    """Rotation, scale, flip, photometric jitter, blur, noise. Returns patch, mask, params."""
    params = {}
    h, w = patch.shape[:2]
    # geometry
    if rng.random() < 0.5:  # identity-ish geometry half the time (plain copy-move is common)
        ang, sc = 0.0, 1.0
    else:
        ang = rng.choice([rng.uniform(-15, 15), rng.uniform(-180, 180)])
        sc = rng.uniform(0.8, 1.25)
    flip = rng.random() < 0.2
    params.update(angle=round(ang, 2), scale=round(sc, 3), flip=bool(flip))
    if flip:
        patch, mask = patch[:, ::-1], mask[:, ::-1]
    # rotate/scale with padding so nothing is cut off
    diag = int(np.ceil(np.hypot(h, w) * sc)) + 4
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    M[0, 2] += (diag - w) / 2
    M[1, 2] += (diag - h) / 2
    patch = cv2.warpAffine(np.ascontiguousarray(patch), M, (diag, diag), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(np.ascontiguousarray(mask), M, (diag, diag), flags=cv2.INTER_LINEAR)
    mask = (mask > 127).astype(np.uint8) * 255
    ys, xs = np.where(mask > 0)
    if len(ys) < 50:
        return None
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    patch, mask = patch[y1:y2, x1:x2], mask[y1:y2, x1:x2]
    # photometric jitter (applied to the pasted copy only)
    p = patch.astype(np.float32)
    alpha = rng.uniform(0.85, 1.15) if rng.random() < 0.6 else 1.0
    beta = rng.uniform(-18, 18) if rng.random() < 0.6 else 0.0
    p = p * alpha + beta
    params.update(contrast=round(alpha, 3), brightness=round(beta, 2))
    patch = np.clip(p, 0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        s = rng.uniform(0.4, 1.6)
        patch = cv2.GaussianBlur(patch, (0, 0), s)
        params["blur"] = round(s, 2)
    if rng.random() < 0.3:
        sd = rng.uniform(1, 5)
        patch = np.clip(patch.astype(np.float32) + np.random.normal(0, sd, patch.shape), 0, 255).astype(np.uint8)
        params["noise"] = round(sd, 2)
    return patch, mask, params


def paste(dst, patch, mask, forbid_boxes, rng, blend, match=0.0):
    """Paste patch at a random location not overlapping forbid_boxes. Returns out_img, target_mask, box."""
    H, W = dst.shape[:2]
    ph, pw = mask.shape
    if ph >= H - 2 or pw >= W - 2:
        return None
    for _ in range(40):
        x = rng.randint(1, W - pw - 1)
        y = rng.randint(1, H - ph - 1)
        ok = True
        for (fx, fy, fw, fh) in forbid_boxes:
            if not (x + pw <= fx or fx + fw <= x or y + ph <= fy or fy + fh <= y):
                ok = False
                break
        if ok:
            break
    else:
        return None
    if match > 0:
        # Foreign patches come from a different paper (different brightness and contrast),
        # so their seams are harsher than same-image copies. Shift each channel of the
        # patch toward the local statistics of the place where it lands.
        roi = dst[y:y + ph, x:x + pw].astype(np.float32)
        pf = patch.astype(np.float32)
        mm = mask > 0
        for c in range(pf.shape[2]):
            mu_p, sd_p = pf[..., c][mm].mean(), pf[..., c][mm].std()
            mu_d, sd_d = roi[..., c].mean(), roi[..., c].std()
            k = float(np.clip(sd_d / (sd_p + 1e-3), 0.5, 2.0))
            pf[..., c] = (1 - match) * pf[..., c] + match * ((pf[..., c] - mu_p) * k + mu_d)
        patch = np.clip(pf, 0, 255).astype(np.uint8)
    out = dst.copy()
    tmask = np.zeros((H, W), np.uint8)
    tmask[y:y + ph, x:x + pw] = mask
    used = blend
    if blend == "poisson":
        try:
            m3 = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
            p3 = cv2.copyMakeBorder(patch, 1, 1, 1, 1, cv2.BORDER_REFLECT)
            center = (x + pw // 2, y + ph // 2)
            # seamlessClone needs the (padded) patch fully inside dst
            if x - 1 >= 0 and y - 1 >= 0 and x + pw + 1 <= W and y + ph + 1 <= H:
                out = cv2.seamlessClone(p3, dst, m3, center, cv2.NORMAL_CLONE)
            else:
                used = "feather"
        except cv2.error:
            used = "feather"
    if used != "poisson":
        sg = rng.uniform(0.5, 3.0)
        a = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sg)
        a = np.minimum(a, mask.astype(np.float32) / 255.0)  # keep the paste inside the mask
        a = a[..., None]
        roi = out[y:y + ph, x:x + pw].astype(np.float32)
        out[y:y + ph, x:x + pw] = np.clip(a * patch.astype(np.float32) + (1 - a) * roi, 0, 255).astype(np.uint8)
        used = f"feather{sg:.2f}"
    return out, tmask, (x, y, pw, ph), used


def maybe_jpeg(img, rng, p):
    """Whole-image recompression, same probability for every class."""
    if rng.random() >= p:
        return img, None
    q = rng.randint(60, 95)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), q


# ----------------------------------------------------------------------------
# triplet builder
# ----------------------------------------------------------------------------
def build_triplet(base, foreign, rng, jpeg_p, blend_mode, match_strength=1.0):
    """Returns dict of three (img, src_mask, tgt_mask, meta) or None if any step fails."""
    H, W = base.shape[:2]
    blend = blend_mode if blend_mode != "mixed" else rng.choice(["feather", "poisson"])

    # forged: patch from SAME image
    cut = sample_patch(base, rng)
    if cut is None:
        return None
    patch, pmask, (sx, sy, sw, sh) = cut
    tp = transform_patch(patch, pmask, rng)
    if tp is None:
        return None
    patch_t, mask_t, tparams = tp
    res = paste(base, patch_t, mask_t, [(sx, sy, sw, sh)], rng, blend)
    if res is None:
        return None
    forged, tmask, tbox, used = res
    smask = np.zeros((H, W), np.uint8)
    smask[sy:sy + sh, sx:sx + sw] = pmask

    # unrelated paste: patch from a DIFFERENT document, identical pipeline params
    fh, fw = foreign.shape[:2]
    # resize foreign so its scale is comparable to base
    fs = min(1.0, max(H / fh, 0.5)) if fh > H else 1.0
    f2 = cv2.resize(foreign, (max(int(fw * fs), 20), max(int(fh * fs), 20))) if fs != 1.0 else foreign
    cut2 = sample_patch(f2, rng)
    if cut2 is None:
        return None
    p2, m2, _ = cut2
    tp2 = transform_patch(p2, m2, rng)
    if tp2 is None:
        return None
    p2t, m2t, tparams2 = tp2
    res2 = paste(base, p2t, m2t, [], rng, blend, match=match_strength)
    if res2 is None:
        return None
    unrelated, pmask2, _, used2 = res2

    # identical post-processing probability across the three
    out = {}
    Z = np.zeros((H, W), np.uint8)
    for kind, img, sm, tm, pm, extra in (
        ("pristine", base, Z, Z, Z, {}),
        ("forged", forged, smask, tmask, tmask, {"transform": tparams, "blend": used}),
        # unrelated_paste is a CLEAN image: its training masks stay empty. pm (paste mask)
        # records where the foreign patch sits, for audits only.
        ("unrelated_paste", unrelated, Z, Z, pmask2, {"transform": tparams2, "blend": used2}),
    ):
        img_j, q = maybe_jpeg(img, rng, jpeg_p)
        extra = dict(extra, jpeg=q)
        out[kind] = (img_j, sm, tm, pm, extra)
    return out


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder of PRISTINE base images (recursive)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=3000, help="triplets")
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--val-frac", type=float, default=0.10, help="fraction of DOCUMENTS to val")
    ap.add_argument("--test-frac", type=float, default=0.10, help="fraction of DOCUMENTS to test")
    ap.add_argument("--doc-key", choices=["dir", "prefix", "stem"], default="prefix")
    ap.add_argument("--blend", choices=["feather", "poisson", "mixed"], default="mixed")
    ap.add_argument("--match-strength", type=float, default=1.0,
                    help="0..1: how strongly unrelated-paste patches are matched to local brightness/contrast")
    ap.add_argument("--jpeg-prob", type=float, default=0.5)
    ap.add_argument("--min-side", type=int, default=96, help="skip base images with a shorter side below this")
    ap.add_argument("--max-side", type=int, default=1024, help="downscale bases whose long side exceeds this")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    src = Path(args.src)
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT)
    if not files:
        raise SystemExit(f"no images found under {src}")

    groups = {"train": {}, "val": {}, "test": {}}
    for p in files:
        d = doc_id(p, args.doc_key)
        groups[split_of(d, args.val_frac, args.test_frac, args.seed)].setdefault(d, []).append(p)
    print({k: f"{len(v)} docs / {sum(len(x) for x in v.values())} panels" for k, v in groups.items()})
    for k, v in groups.items():
        if len(v) < 2:
            print(f"WARNING: split '{k}' has fewer than 2 documents; unrelated-paste needs a foreign document")

    out_root = Path(args.out)
    targets = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    counts = {}
    cache = {}

    def get(path):
        if path not in cache:
            img = load_rgb(path)
            if img is not None:
                h, w = img.shape[:2]
                if min(h, w) < args.min_side:
                    img = None
                elif max(h, w) > args.max_side:
                    s = args.max_side / max(h, w)
                    img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            if len(cache) > 300:
                cache.clear()
            cache[path] = img
        return cache[path]

    for split, n in targets.items():
        docs = groups[split]
        if not docs or n == 0:
            continue
        for sub in ("images", "mask", "mask_source", "mask_target", "mask_paste"):
            (out_root / split / sub).mkdir(parents=True, exist_ok=True)
        meta_f = open(out_root / split / "meta.jsonl", "w")
        doc_list = sorted(docs)
        all_panels = [(d, p) for d in doc_list for p in docs[d]]
        made, tries = 0, 0
        while made < n and tries < n * 30:
            tries += 1
            d, p = rng.choice(all_panels)
            base = get(p)
            if base is None:
                continue
            # foreign panel from a different document in the SAME split
            others = [(dd, pp) for dd, pp in all_panels if dd != d]
            if not others:
                continue
            fd, fp = rng.choice(others)
            foreign = get(fp)
            if foreign is None:
                continue
            tri = build_triplet(base, foreign, rng, args.jpeg_prob, args.blend, args.match_strength)
            if tri is None:
                continue
            gid = f"{split}_{made:06d}"
            for kind, (img, sm, tm, pm, extra) in tri.items():
                iid = f"{gid}_{kind}"
                cv2.imwrite(str(out_root / split / "images" / f"{iid}.png"), img)
                cv2.imwrite(str(out_root / split / "mask_source" / f"{iid}.png"), sm)
                cv2.imwrite(str(out_root / split / "mask_target" / f"{iid}.png"), tm)
                cv2.imwrite(str(out_root / split / "mask_paste" / f"{iid}.png"), pm)
                cv2.imwrite(str(out_root / split / "mask" / f"{iid}.png"), np.maximum(sm, tm))
                meta_f.write(json.dumps({
                    "id": iid, "group": gid, "base": str(p.relative_to(src)), "doc": d,
                    "kind": kind, "label": 1 if kind == "forged" else 0,
                    "foreign_doc": fd if kind == "unrelated_paste" else None, **extra,
                }) + "\n")
            made += 1
            if made % 200 == 0:
                print(f"  {split}: {made}/{n}")
        meta_f.close()
        counts[split] = {"triplets": made, "images": made * 3, "docs": len(docs)}
        print(split, counts[split])

    (out_root / "manifest.json").write_text(json.dumps({
        "version": "v3", "seed": args.seed, "args": vars(args), "counts": counts}, indent=2))
    print("done ->", out_root)


if __name__ == "__main__":
    main()