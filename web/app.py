"""Flask backend for ForgeryDetect web application."""
import sys, os, time, base64

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify

from detector.detect import detect
from detector.features import extract_features
from detector.matching import find_matches

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "comofod")
CATEGORIES = ["translation", "rotation", "scaling", "distortion", "combination"]


def img_to_b64(img):
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("utf-8")


def find_comofod_path(tag, suffix="_F.png"):
    for cat in CATEGORIES:
        p = os.path.join(DATA_DIR, cat, f"{tag}{suffix}")
        if os.path.exists(p):
            return p
    # fallback to root data dir
    p = os.path.join(DATA_DIR, f"{tag}{suffix}")
    if os.path.exists(p):
        return p
    return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/detect", methods=["POST"])
def api_detect():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Parameters with sensible defaults
    B = int(request.form.get("B", 8))
    K = int(request.form.get("K", 6))
    Q = int(request.form.get("Q", 8))
    min_std = float(request.form.get("min_std", 4.0))
    min_votes = int(request.form.get("min_votes", 150))
    min_dist = int(request.form.get("min_dist", 30))
    window = int(request.form.get("window", 5))
    kernel = int(request.form.get("kernel", 9))
    min_size = int(request.form.get("min_size", 500))

    try:
        raw = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Cannot decode image"}), 400

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # ---------- detection ----------
        t0 = time.time()
        mask = detect(gray, B=B, K=K, Q=Q, min_std=min_std, window=window,
                      min_dist=min_dist, min_votes=min_votes,
                      kernel=kernel, min_size=min_size)
        elapsed = time.time() - t0

        # ---------- overlay ----------
        overlay = img.copy()
        alpha = np.zeros_like(img)
        alpha[mask > 0] = (0, 0, 200)
        overlay = cv2.addWeighted(overlay, 0.65, alpha, 0.35, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

        # ---------- top shift vectors ----------
        feats, pos = extract_features(gray, B=B, K=K, Q=Q, min_std=min_std)
        a_pts, b_pts, shifts = find_matches(feats, pos,
                                            window=window, min_dist=min_dist)
        top_shifts = []
        if len(shifts) > 0:
            vecs, _, cnts = np.unique(shifts, axis=0,
                                      return_inverse=True, return_counts=True)
            for idx in np.argsort(-cnts)[:5]:
                if cnts[idx] >= min_votes:
                    top_shifts.append({
                        "dy": int(vecs[idx][0]),
                        "dx": int(vecs[idx][1]),
                        "votes": int(cnts[idx]),
                    })

        flagged = int((mask > 0).sum())

        return jsonify({
            "original": img_to_b64(img),
            "mask": img_to_b64(mask),
            "overlay": img_to_b64(overlay),
            "stats": {
                "width": w, "height": h,
                "flagged": flagged,
                "total": int(gray.size),
                "pct": round(100 * flagged / gray.size, 2),
                "time": round(elapsed, 3),
                "regions": len(contours),
            },
            "shifts": top_shifts,
            "params": {"B": B, "K": K, "Q": Q, "min_std": min_std, "min_votes": min_votes,
                       "min_dist": min_dist, "window": window,
                       "kernel": kernel, "min_size": min_size},
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/comofod/list")
def comofod_list():
    """Return available CoMoFoD images grouped by category."""
    if not os.path.isdir(DATA_DIR):
        return jsonify({"categories": {}, "images": []})

    cat_map = {}
    all_nums = set()
    for cat in CATEGORIES:
        cat_dir = os.path.join(DATA_DIR, cat)
        if os.path.isdir(cat_dir):
            nums = sorted({int(f[:3]) for f in os.listdir(cat_dir)
                           if f.endswith("_F.png") and f[:3].isdigit()})
            cat_map[cat] = nums
            all_nums.update(nums)
        else:
            cat_map[cat] = []

    return jsonify({
        "categories": cat_map,
        "images": sorted(all_nums)
    })


@app.route("/api/comofod/<int:num>")
def comofod_load(num):
    """Load a CoMoFoD forged image as base64."""
    tag = f"{num:03d}"
    path = find_comofod_path(tag, "_F.png")
    if not path or not os.path.exists(path):
        return jsonify({"error": f"Image {tag} not found"}), 404
    img = cv2.imread(path)
    return jsonify({"image": img_to_b64(img), "filename": f"{tag}_F.png"})


if __name__ == "__main__":
    print("\n  🔍 ForgeryDetect running at http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)

