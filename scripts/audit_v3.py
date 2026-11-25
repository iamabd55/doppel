#!/usr/bin/env python3
"""
audit_v3.py: sanity checks on a make_synthetic_v3.py output folder.

1. Document leakage: no document id appears in more than one split.
2. Balance: counts per kind and per split.
3. Seam parity: edge energy around the paste boundary should look the same for
   forged and unrelated_paste. We report the AUC of that energy for telling them
   apart. About 0.5 is good; far from 0.5 means seams leak the label.

Usage: python audit_v3.py data_v3
"""
import json, sys
from pathlib import Path
import cv2
import numpy as np


def auc(a, b):
    a, b = np.asarray(a), np.asarray(b)
    allv = np.concatenate([a, b])
    ranks = allv.argsort().argsort() + 1
    ra = ranks[: len(a)].sum()
    return (ra - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


def main(root):
    root = Path(root)
    docs, kinds = {}, {}
    for split in ("train", "val", "test"):
        mf = root / split / "meta.jsonl"
        if not mf.exists():
            continue
        rows = [json.loads(l) for l in open(mf)]
        docs[split] = {r["doc"] for r in rows}
        kinds[split] = {}
        for r in rows:
            kinds[split][r["kind"]] = kinds[split].get(r["kind"], 0) + 1
    print("== balance ==")
    for s, k in kinds.items():
        print(s, k)
    print("== leakage ==")
    sp = list(docs)
    bad = False
    for i in range(len(sp)):
        for j in range(i + 1, len(sp)):
            inter = docs[sp[i]] & docs[sp[j]]
            print(f"{sp[i]} & {sp[j]}: {len(inter)} shared documents")
            bad |= bool(inter)
    print("LEAKAGE" if bad else "no document leakage")

    # seam parity: forged vs unrelated_paste, both measured on the ring around their own paste mask
    print("== seam check (forged vs unrelated_paste, ring around paste) ==")
    split = "train" if "train" in docs else sp[0]
    rows = [json.loads(l) for l in open(root / split / "meta.jsonl")]
    k = np.ones((5, 5), np.uint8)
    g = lambda im: np.hypot(cv2.Sobel(im, cv2.CV_32F, 1, 0), cv2.Sobel(im, cv2.CV_32F, 0, 1))
    vals = {"forged": [], "unrelated_paste": []}
    areas = {"forged": [], "unrelated_paste": []}
    for kind in vals:
        for r in [x for x in rows if x["kind"] == kind][:300]:
            pm = cv2.imread(str(root / split / "mask_paste" / f"{r['id']}.png"), 0)
            im = cv2.imread(str(root / split / "images" / f"{r['id']}.png"), 0)
            if pm is None or im is None or pm.max() == 0:
                continue
            ring = (cv2.dilate(pm, k) > 0) & (cv2.erode(pm, k) == 0)
            vals[kind].append(g(im)[ring].mean())
            areas[kind].append((pm > 0).mean())
    if vals["forged"] and vals["unrelated_paste"]:
        f, u = vals["forged"], vals["unrelated_paste"]
        print(f"ring edge energy: forged {np.mean(f):.2f}, unrelated {np.mean(u):.2f}; AUC {auc(f, u):.3f} "
              "(about 0.5 means seams do not separate the classes)")
        print(f"paste area fraction: forged {np.mean(areas['forged']):.3f}, unrelated {np.mean(areas['unrelated_paste']):.3f}")
    print("Reminder: the decisive seam test is the trained model on unrelated_paste images. "
          "Its false positive rate there should match its rate on pristine.")


if __name__ == "__main__":
    main(sys.argv[1])
