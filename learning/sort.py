import cv2
import numpy as np
from detector.features import extract_features
from detector.matching import find_matches

gray = cv2.cvtColor(cv2.imread("data/forged.jpg"), cv2.COLOR_BGR2GRAY)
feats, pos = extract_features(gray,Q=4)
a, b, shift = find_matches(feats, pos)

# print("matching pairs found:", len(shift))

# print("\nfirst 5 pairs (block A -> block B, shift):")
# for i in range(5):
#     print(a[i], "->", b[i], "  shift", shift[i])

# # peeking at the answer key, only to check our work:
# true_hits = (shift == [250, 300]).all(axis=1).sum()
# print("\npairs with the true shift (250, 300):", true_hits)


vectors, counts = np.unique(shift, axis=0, return_counts=True)
order = np.argsort(counts)[::-1]          # biggest vote count first

print("different shift vectors seen:", len(vectors))
print("\ntop 10 shift vectors:")
for i in order[:10]:
    print(vectors[i], " votes:", counts[i])