"""Visual inspection utility for generated synthetic samples.

Loads a sample of generated synthetic pairs and renders a multi-column comparison:
- Column 1: Base original image
- Column 2: Composite forged image (with source green / target red bounding boxes)
- Column 3: Target mask (_M)
- Column 4: Source mask (_S)
- Column 5: Union mask (_A, BioFors convention)

Saves output to figures/synthetic_samples_grid.png.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize synthetic copy-move samples.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/synthetic/train_manifest.csv",
        help="Path to synthetic manifest CSV",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of samples to visualize (default: 8)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/synthetic_samples_grid.png",
        help="Output image path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = REPO_ROOT / args.manifest
    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        print(f"ERROR: Manifest {manifest_path} does not exist. Run scripts/make_synthetic.py first.")
        sys.exit(1)

    df = pd.read_csv(manifest_path)

    # Select representative mix of forged and clean samples
    forged_df = df[df["is_forged"]].sample(min(args.num_samples - 2, len(df[df["is_forged"]])), random_state=42)
    clean_df = df[~df["is_forged"]].sample(min(2, len(df[~df["is_forged"]])), random_state=42)
    sample_df = pd.concat([forged_df, clean_df]).reset_index(drop=True)

    num_rows = len(sample_df)
    fig, axes = plt.subplots(num_rows, 5, figsize=(16, 2.8 * num_rows))
    if num_rows == 1:
        axes = np.expand_dims(axes, 0)

    col_titles = [
        "1. Base Image",
        "2. Forged / Negative (_F)",
        "3. Target Mask (_M)",
        "4. Source Mask (_S)",
        "5. Union Mask (_A)",
    ]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=12, fontweight="bold", pad=8)

    biofors_images_dir = REPO_ROOT / "data" / "biofors" / "images"

    for row_idx, (_, row) in enumerate(sample_df.iterrows()):
        base_path = biofors_images_dir / row["source_image"]
        f_path = REPO_ROOT / "data" / row["path_F"]
        m_path = REPO_ROOT / "data" / row["path_M"]
        s_path = REPO_ROOT / "data" / row["path_S"]
        a_path = REPO_ROOT / "data" / row["path_A"]
        meta_path = REPO_ROOT / "data" / row["path_meta"]

        # Read images
        base_bgr = cv2.imread(str(base_path))
        f_bgr = cv2.imread(str(f_path))
        m_gray = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
        s_gray = cv2.imread(str(s_path), cv2.IMREAD_GRAYSCALE)
        a_gray = cv2.imread(str(a_path), cv2.IMREAD_GRAYSCALE)

        with open(meta_path, "r", encoding="utf-8") as jf:
            meta = json.load(jf)

        base_rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB) if base_bgr is not None else np.zeros((50, 50, 3), dtype=np.uint8)
        f_rgb = cv2.cvtColor(f_bgr, cv2.COLOR_BGR2RGB) if f_bgr is not None else np.zeros((50, 50, 3), dtype=np.uint8)

        # Draw bounding box overlays on a copy of forged image
        annotated_f = f_rgb.copy()
        if meta.get("is_forged"):
            sbox = meta.get("source_bbox")
            tbox = meta.get("target_bbox")
            if sbox:
                sx1, sy1, sx2, sy2 = sbox
                cv2.rectangle(annotated_f, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)  # Green source
            if tbox:
                tx1, ty1, tx2, ty2 = tbox
                cv2.rectangle(annotated_f, (tx1, ty1), (tx2, ty2), (255, 0, 0), 2)  # Red target

        # Display images
        axes[row_idx, 0].imshow(base_rgb)
        axes[row_idx, 1].imshow(annotated_f)
        axes[row_idx, 2].imshow(m_gray if m_gray is not None else np.zeros((10, 10)), cmap="gray")
        axes[row_idx, 3].imshow(s_gray if s_gray is not None else np.zeros((10, 10)), cmap="gray")
        axes[row_idx, 4].imshow(a_gray if a_gray is not None else np.zeros((10, 10)), cmap="gray")

        # Row labeling
        c_name = row["class"]
        is_f = row["is_forged"]
        tf_name = row["transform_type"]
        deg_info = []
        deg_dict = meta.get("degradations", {})
        if deg_dict.get("blur"):
            deg_info.append("blur")
        if deg_dict.get("noise"):
            deg_info.append("noise")
        if deg_dict.get("jpeg_quality"):
            deg_info.append(f"jpg:{deg_dict['jpeg_quality']}")
        deg_str = ",".join(deg_info) if deg_info else "none"

        status_str = f"FORGED ({tf_name}, deg:[{deg_str}])" if is_f else "CLEAN NEGATIVE"
        axes[row_idx, 0].set_ylabel(
            f"[{c_name}]\n{row['sample_id']}\n{status_str}",
            fontsize=8,
            rotation=0,
            labelpad=50,
            va="center",
        )

        for col_idx in range(5):
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Inspection grid successfully saved to: {output_path}")


if __name__ == "__main__":
    main()
