import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")                  # draw to a file, no window needed
import matplotlib.pyplot as plt
from detector.detect import detect
from detector.evaluate import scores

gray = cv2.imread("data/synthetic/forged.jpg", 0)
gt = cv2.imread("data/synthetic/mask.jpg", 0)
gt = np.where(gt > 127, 255, 0).astype(np.uint8)


def add_noise(img, sigma):
    rng = np.random.default_rng(0)                       # fixed seed
    noisy = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_blur(img, sigma):
    if sigma == 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma)


def add_jpeg(img, quality):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, 0)


tests = {
    "Gaussian noise (sigma)": (add_noise, [0, 2, 5, 10, 20]),
    "Gaussian blur (sigma)": (add_blur, [0, 0.5, 1, 1.5, 2]),
    "JPEG quality": (add_jpeg, [95, 85, 70, 50, 30]),
}
# settings = {"Q=4": 4, "Q=16": 16}
settings = {"Q=4, K=16, votes 50": (4, 16, 50), "Q=8, K=6, votes 150": (8, 6, 150)}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (title, (damage, levels)) in zip(axes, tests.items()):
    print("\n" + title)
    for name, (Q,K,mv) in settings.items():
        f1s = []
        for lv in levels:
            mask = detect(damage(gray, lv), Q=Q,K=K,min_votes=mv)
            p, r, f1 = scores(mask, gt)
            f1s.append(f1)
            print("  %-5s level %-5s precision %.3f  recall %.3f  F1 %.3f" % (name, lv, p, r, f1))
        ax.plot(range(len(levels)), f1s, marker="o", label=name)

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([str(l) for l in levels])
    ax.set_title(title)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig("results/robustness.png", dpi=150)
print("\nsaved results/robustness.png")