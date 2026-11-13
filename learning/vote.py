import cv2
import numpy as np
from detector.features import extract_features
from detector.matching import find_matches
from skimage.morphology import remove_small_objects

B = 8
MIN_VOTES = 50

img = cv2.imread("data/forged.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
feats, pos = extract_features(gray, Q=4)
a, b, shift = find_matches(feats, pos)

# voting: which shift does each pair belong to, and how many votes did it get
vectors, inverse, counts = np.unique(shift, axis=0, return_inverse=True, return_counts=True)
inverse = inverse.reshape(-1)

keep = counts[inverse] >= MIN_VOTES          # True for pairs on a popular shift
a, b = a[keep], b[keep]
print("shift vectors kept:", (counts >= MIN_VOTES).sum())
print("pairs kept:", len(a), "of", len(shift))

# paint every surviving block (both sides of each pair) white
mask = np.zeros(gray.shape, np.uint8)
for r, c in np.concatenate([a, b]):
    mask[r:r+B, c:c+B] = 255

cv2.imwrite("results/raw_mask.jpg", mask)

# compare with the answer key
gt = cv2.imread("data/mask.jpg", 0)
cv2.imwrite("results/compare.jpg", np.hstack([mask, gt]))   # left: ours, right: truth

print("\nwhite pixels in our mask:   ", (mask > 0).sum())
print("white pixels in answer key: ", (gt > 0).sum())
print("pixels we got right:        ", ((mask > 0) & (gt > 0)).sum())



def scores(m, gt):
    tp = ((m > 0) & (gt > 0)).sum()     # flagged and truly forged
    fp = ((m > 0) & (gt == 0)).sum()    # flagged but not forged
    fn = ((m == 0) & (gt > 0)).sum()    # forged but missed
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return p, r, 2 * p * r / (p + r)


kernel = np.ones((9, 9), np.uint8)
closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
clean = remove_small_objects(closed > 0, min_size=500)
clean = (clean * 255).astype(np.uint8)

cv2.imwrite("results/clean_mask.jpg", clean)
cv2.imwrite("results/compare_clean.jpg", np.hstack([mask, clean, gt]))   # raw | clean | truth

print("\nbefore cleanup  precision %.3f  recall %.3f  F1 %.3f" % scores(mask, gt))
print("after cleanup   precision %.3f  recall %.3f  F1 %.3f" % scores(clean, gt))