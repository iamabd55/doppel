import cv2
import numpy as np
from detector.detect import detect

D = "data/comofod/translation/"
gray = cv2.imread(D + "001_F.png", 0)
B = cv2.imread(D + "001_B.png", 0) > 127

for mv in [150, 50, 20]:
    mask = detect(gray, Q=8, K=6, min_votes=mv) > 0
    print("min_votes %3d: flagged %6d px | inside B %5d of %d | outside B %6d"
          % (mv, mask.sum(), (mask & B).sum(), B.sum(), (mask & ~B).sum()))
    out = np.hstack([(mask * 255).astype(np.uint8), (B * 255).astype(np.uint8)])
    cv2.imwrite("results/try_001_votes%d.png" % mv, out)