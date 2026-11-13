import numpy as np


def scores(mask, gt):
    tp = ((mask > 0) & (gt > 0)).sum()     # flagged and truly forged
    fp = ((mask > 0) & (gt == 0)).sum()    # flagged but not forged
    fn = ((mask == 0) & (gt > 0)).sum()    # forged but missed
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1