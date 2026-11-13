import time
import cv2
import numpy as np
from detector.detect import detect
from detector.evaluate import scores

gray = cv2.imread("data/synthetic/forged.jpg", 0)
gt = cv2.imread("data/synthetic/mask.jpg", 0)
gt = np.where(gt > 127, 255, 0).astype(np.uint8)

t = time.time()
mask = detect(gray)
print("time: %.1f s" % (time.time() - t))

print("precision %.3f  recall %.3f  F1 %.3f" % scores(mask, gt))
cv2.imwrite("results/run_mask.png", np.hstack([mask, gt]))