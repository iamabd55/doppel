import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import dctn


def zigzag_indices(B):
    """(row, col) pairs in zigzag order: smooth patterns first."""
    cells = [(i, j) for i in range(B) for j in range(B)]
    return sorted(cells, key=lambda p: (p[0] + p[1],
                                        p[0] if (p[0] + p[1]) % 2 else -p[0]))


def extract_features(gray, B=8, K=16, Q=16, min_std=4.0):
    g = gray.astype(np.float32)

    blocks = sliding_window_view(g, (B, B))              # every BxB block
    H, W = blocks.shape[:2]

    # Filter out flat/smooth blocks (uniform sky, blank walls) that cause false matches
    if min_std > 0:
        stds = np.std(blocks, axis=(-2, -1)).reshape(-1)
        valid = stds >= min_std
    else:
        valid = None

    coefs = dctn(blocks, axes=(-2, -1), norm="ortho")    # DCT of all blocks

    zz = zigzag_indices(B)[:K]                           # first K in zigzag order
    rows = np.array([p[0] for p in zz])
    cols = np.array([p[1] for p in zz])
    feats = coefs[:, :, rows, cols]                      # keep only those K

    feats = np.round(feats / Q).astype(np.int16).reshape(-1, K)   # quantize

    ys, xs = np.divmod(np.arange(H * W), W)
    pos = np.stack([ys, xs], axis=1)                     # (row, col) per block

    if valid is not None:
        feats = feats[valid]
        pos = pos[valid]

    return feats, pos