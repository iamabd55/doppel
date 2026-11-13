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


def votes(img, Q, K):
    feats, pos = extract_features(img, K=K, Q=Q)
    a, b, shift = find_matches(feats, pos)
    if len(shift) == 0:
        return 0, [], 0
    vec, cnt = np.unique(shift, axis=0, return_counts=True)
    order = np.argsort(cnt)[::-1]
    top = [(tuple(int(x) for x in vec[i]), int(cnt[i])) for i in order[:3]]
    true_votes = int(cnt[(vec == [250, 300]).all(axis=1)].sum())
    return len(shift), top, true_votes


conditions = {
    "clean": gray,
    "noise 2": add_noise(gray, 2),
    "noise 5": add_noise(gray, 5),
    "jpeg 70": add_jpeg(gray, 70),
}
configs = [(4, 16), (4, 6), (8, 6)]    # (Q, K)

for cname, img in conditions.items():
    print()
    for Q, K in configs:
        n, top, tv = votes(img, Q, K)
        print("%-8s Q=%d K=%2d  pairs=%6d  true-shift votes=%5d  top3=%s" % (cname, Q, K, n, tv, top))