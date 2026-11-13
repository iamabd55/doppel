"""
False alarm test: run detector on ORIGINAL (unforged) CoMoFoD images.
Nothing was pasted, so the detector should flag nothing.
Any flagged pixels are false alarms.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from detector.detect import detect

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "comofod")
CATEGORIES = ["translation", "rotation", "scaling", "distortion", "combination"]

def find_file(filename):
    for cat in CATEGORIES:
        p = os.path.join(DATA, cat, filename)
        if os.path.exists(p):
            return p
    p = os.path.join(DATA, filename)
    return p if os.path.exists(p) else None

START, END = 1, 40

print(f"{'img':>5}  {'flagged':>8}  {'% of image':>10}  {'time':>5}")
print("-" * 40)

false_alarms = 0
total_pixels = 0
total_time = 0
alarm_count = 0

for i in range(START, END + 1):
    tag = f"{i:03d}"

    orig_path = find_file(f"{tag}_O.png")
    if not orig_path:
        print(f"{tag:>5}  SKIP")
        continue
    gray = cv2.imread(orig_path, 0)
    if gray is None:
        print(f"{tag:>5}  SKIP")
        continue

    t0 = time.time()
    mask = detect(gray, Q=8, K=6, min_votes=150)
    dt = time.time() - t0
    total_time += dt

    flagged = (mask > 0).sum()
    pct = 100.0 * flagged / gray.size

    false_alarms += flagged
    total_pixels += gray.size
    if flagged > 0:
        alarm_count += 1

    print(f"{tag:>5}  {flagged:8d}  {pct:9.3f}%  {dt:5.1f}s")

print("-" * 40)
print(f"Images with false alarms: {alarm_count}/{END - START + 1}")
print(f"Total false alarm pixels: {false_alarms}")
print(f"Average false alarm rate: {100.0 * false_alarms / total_pixels:.4f}%")
print(f"Total time: {total_time:.1f}s")

