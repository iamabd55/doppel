import cv2
import numpy as np
from detector.features import extract_features
from detector.matching import find_matches

gray = cv2.imread("data/forged.jpg", 0)


def add_noise(img, sigma):
    rng = np.random.default_rng(0)
    noisy = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_jpeg(img, quality):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, 0)


def margin(img, Q, K):
    feats, pos = extract_features(img, K=K, Q=Q)
    a, b, shift = find_matches(feats, pos)
    if len(shift) == 0:
        return 0, 0
    vec, cnt = np.unique(shift, axis=0, return_counts=True)
    near = (np.abs(vec - [250, 300]) <= 3).all(axis=1)   # true shift and its neighbors
    true_votes = int(cnt[(vec == [250, 300]).all(axis=1)].sum())
    junk_max = int(cnt[~near].max()) if (~near).any() else 0
    return true_votes, junk_max


conditions = {
    "clean": gray,
    "noise 2": add_noise(gray, 2),
    "noise 5": add_noise(gray, 5),
    "jpeg 70": add_jpeg(gray, 70),
    "jpeg 50": add_jpeg(gray, 50),
}
configs = [(4, 6), (8, 6), (12, 6), (8, 10)]    # (Q, K)

for cname, img in conditions.items():
    print()
    for Q, K in configs:
        t, j = margin(img, Q, K)
        print("%-8s Q=%2d K=%2d  true=%5d  junk-max=%5d" % (cname, Q, K, t, j))