import cv2
import numpy as np
from scipy.fft import dct,dctn,idctn

np.set_printoptions(precision=1, suppress=True, linewidth=150)

print("PART 1: one row of 8 pixels\n")
rows = {
    "flat       ": np.full(8, 100.0),
    "smooth ramp": np.arange(0, 80, 10, dtype=float),
    "busy       ": np.array([0, 100, 0, 100, 0, 100, 0, 100], dtype=float),
}
for name, row in rows.items():
    print(name, row)
    print("  coefficients:", dct(row, norm="ortho"), "\n")
    
print("PART 2: a real 8x8 block\n")
gray = cv2.cvtColor(cv2.imread("data/forged.jpg"), cv2.COLOR_BGR2GRAY)
block = gray[50:58, 50:58].astype(np.float32)

coef = dctn(block, norm="ortho")
print("pixels:\n", block)
print("\nDCT coefficients (top-left = smooth, bottom-right = busy):\n", coef)

# Proof 1: nothing is lost
back = idctn(coef, norm="ortho")
print("\nrebuilt from all 64 == original:", np.allclose(back, block, atol=1e-3))

# Proof 2: a short fingerprint is enough
small = np.zeros_like(coef)
print("all zero matrix:", small)
small[:4, :4] = coef[:4, :4]          # keep only 16 of 64
approx = idctn(small, norm="ortho")
print("approximate matrix:", approx)
print("biggest pixel error using only 16 of 64:", np.abs(approx - block).max().round(1), "(out of 255)")