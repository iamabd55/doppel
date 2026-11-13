"""
Batch test: run detector on CoMoFoD forged images 001–040.
Score against mask M (binarized) which covers BOTH source and paste regions.
Also score against mask B (paste only) for comparison.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from detector.detect import detect
from detector.evaluate import scores

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "comofod")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

CATEGORIES = ["translation", "rotation", "scaling", "distortion", "combination"]

def find_file(filename):
    for cat in CATEGORIES:
        p = os.path.join(DATA, cat, filename)
        if os.path.exists(p):
            return p
    p = os.path.join(DATA, filename)
    return p if os.path.exists(p) else None

START, END = 1, 40          # image range to test

print(f"{'img':>5}  {'P(M)':>6} {'R(M)':>6} {'F1(M)':>6}  "
      f"{'P(B)':>6} {'R(B)':>6} {'F1(B)':>6}  {'flagged':>7}  {'time':>5}")
print("-" * 75)

all_scores_m = []
all_scores_b = []
total_time = 0

for i in range(START, END + 1):
    tag = f"{i:03d}"

    # load image
    forged_path = find_file(f"{tag}_F.png")
    if not forged_path:
        print(f"{tag:>5}  SKIP (image not found)")
        continue
    gray = cv2.imread(forged_path, 0)
    if gray is None:
        print(f"{tag:>5}  SKIP (image not found)")
        continue

    # load ground truth masks
    m_path = find_file(f"{tag}_M.png")
    M = cv2.imread(m_path) if m_path else None
    gt_m = cv2.cvtColor(M, cv2.COLOR_BGR2GRAY) if M is not None else None
    if gt_m is not None:
        gt_m = np.where(gt_m > 10, 255, 0).astype(np.uint8)

    # B mask: binary, paste region only
    b_path = find_file(f"{tag}_B.png")
    B = cv2.imread(b_path, 0) if b_path else None
    gt_b = np.where(B > 127, 255, 0).astype(np.uint8) if B is not None else None

    # run detector
    t0 = time.time()
    mask = detect(gray, Q=8, K=6, min_votes=150)
    dt = time.time() - t0
    total_time += dt

    flagged = (mask > 0).sum()

    # score against M (fair: source + paste)
    if gt_m is not None:
        pm, rm, f1m = scores(mask, gt_m)
    else:
        pm, rm, f1m = 0, 0, 0

    # score against B (paste only)
    if gt_b is not None:
        pb, rb, f1b = scores(mask, gt_b)
    else:
        pb, rb, f1b = 0, 0, 0

    all_scores_m.append((pm, rm, f1m))
    all_scores_b.append((pb, rb, f1b))

    print(f"{tag:>5}  {pm:6.3f} {rm:6.3f} {f1m:6.3f}  "
          f"{pb:6.3f} {rb:6.3f} {f1b:6.3f}  {flagged:7d}  {dt:5.1f}s")

# summary
print("-" * 75)
arr_m = np.array(all_scores_m)
arr_b = np.array(all_scores_b)

# mean scores
mean_m = arr_m.mean(axis=0)
mean_b = arr_b.mean(axis=0)
print(f"{'AVG':>5}  {mean_m[0]:6.3f} {mean_m[1]:6.3f} {mean_m[2]:6.3f}  "
      f"{mean_b[0]:6.3f} {mean_b[1]:6.3f} {mean_b[2]:6.3f}  "
      f"{'':>7}  {total_time:5.1f}s")

# how many had F1(M) > 0.5, > 0.3, == 0
detected = (arr_m[:, 2] > 0.3).sum()
missed   = (arr_m[:, 2] == 0).sum()
print(f"\nDetected (F1>0.3): {detected}/{len(arr_m)}")
print(f"Completely missed: {missed}/{len(arr_m)}")
print(f"Total time: {total_time:.1f}s  ({total_time/len(arr_m):.1f}s per image)")

