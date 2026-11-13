import cv2
import numpy as np

D = "data/comofod/translation/"
F = cv2.imread(D + "001_F.png")
O = cv2.imread(D + "001_O.png")
B = cv2.imread(D + "001_B.png")
M = cv2.imread(D + "001_M.png")

for name, img in [("F", F), ("O", O), ("B", B), ("M", M)]:
    print(name, "shape:", None if img is None else img.shape)

# where does the forged image differ from the original?
diff = (np.abs(F.astype(int) - O.astype(int)).sum(axis=2) > 0)
print("\npixels that differ (F vs O):", int(diff.sum()))

for name, m in [("B", B), ("M", M)]:
    white = m.max(axis=2) > 127
    print("\nmask", name)
    print("  distinct colors:", np.unique(m.reshape(-1, 3), axis=0)[:6].tolist())
    print("  non-black pixels:", int(white.sum()))
    print("  of those, inside the difference area:", int((white & diff).sum()))

# one picture to look at: F, O, diff on top; B, M below
g = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
d = (diff * 255).astype(np.uint8)
blank = np.zeros_like(d)
top = np.hstack([g(F), g(O), d])
bottom = np.hstack([g(B), g(M), blank])
cv2.imwrite("results/inspect_001.png", np.vstack([top, bottom]))
print("\nsaved results/inspect_001.png")