import numpy as np


def find_matches(feats, pos, window=5, min_dist=30):
    # 1. sort fingerprint rows like a dictionary.
    #    np.lexsort treats the LAST key as most important, so we reverse
    #    the columns to make column 0 the most important.
    order = np.lexsort(feats.T[::-1])
    f = feats[order]          # fingerprints in sorted order
    p = pos[order]            # positions rearranged the same way

    all_a, all_b = [], []

    # 2. compare each row with the row k steps below it, for k = 1..window
    for k in range(1, window + 1):
        same = (f[:-k] == f[k:]).all(axis=1)   # True where the two rows are identical
        all_a.append(p[:-k][same])
        all_b.append(p[k:][same])

    a = np.concatenate(all_a)
    b = np.concatenate(all_b)
    shift = b - a

    # 3. throw away pairs that are too close together
    dist = np.sqrt((shift ** 2).sum(axis=1))
    keep = dist >= min_dist
    a, b, shift = a[keep], b[keep], shift[keep]

    # 4. make the shift always point "down" (or "right" if level)
    flip = (shift[:, 0] < 0) | ((shift[:, 0] == 0) & (shift[:, 1] < 0))
    a2 = np.where(flip[:, None], b, a)
    b2 = np.where(flip[:, None], a, b)
    return a2, b2, b2 - a2