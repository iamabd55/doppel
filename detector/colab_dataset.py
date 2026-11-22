"""PyTorch Dataset for Phase 4 CNN Training on Google Colab.

Features:
1. Native Resolution Handling: Does NOT resize images (avoids distorting banding or pixel noise).
   Supports zero-padding to a multiple of 32 (or fixed size) and smart random crops
   guaranteeing forged pixel coverage for positive samples.
2. Synchronous Joint Geometric Augmentations: Random horizontal/vertical flips and cardinal
   rotations applied jointly to image tensor and ground-truth mask tensor.
3. Strict Document Split Disjointness: Built-in validation assertions verifying zero document
   overlap between train and validation splits.
4. Paired-Twin Format: Returns (image, mask_A, label, meta) where label is 1 for forged and 0 for clean.
"""
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    # Graceful fallback for local CPU execution environments without PyTorch installed
    class Dataset:
        pass


class ColabDoppelDataset(Dataset):
    """PyTorch Dataset for synthetic scientific copy-move detection on Colab."""

    def __init__(
        self,
        manifest_path: str,
        images_dir: Optional[str] = None,
        split: str = "train",
        crop_size: Optional[Tuple[int, int]] = (256, 256),
        pos_crop_prob: float = 0.50,
        pad_multiple: int = 32,
        augment: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            manifest_path: Path to train_manifest.csv or val_manifest.csv.
            images_dir: Root directory of synthetic data (defaults to manifest parent).
            split: 'train' or 'val'.
            crop_size: (H, W) for random cropping during training (val always returns full image with padding).
            pos_crop_prob: Share of positive training crops guaranteed to contain forged pixels (default 50%).
            pad_multiple: Pad image and mask dimensions to a multiple of this value (e.g. 32 for U-Net/ResNet).
            augment: Whether to apply joint random geometric augmentations (flips and cardinal rotations).
            seed: Random seed for augmentations.
        """
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.df = pd.read_csv(self.manifest_path)
        self.images_dir = Path(images_dir) if images_dir else self.manifest_path.parent
        self.split = split
        self.crop_size = crop_size
        self.pos_crop_prob = pos_crop_prob
        self.pad_multiple = pad_multiple
        self.augment = augment and (split == "train")
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @property
    def doc_ids(self) -> Set[str]:
        """Return the unique document IDs present in this dataset split."""
        return set(self.df["doc_id"].astype(str).unique())

    def __len__(self) -> int:
        return len(self.df)

    def _pad_to_multiple(
        self, img: np.ndarray, mask: np.ndarray, valid_mask: np.ndarray, multiple: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Zero-pad image, mask and valid_mask to next multiple of multiple (e.g. 32). Never resizes."""
        h, w = img.shape[:2]
        target_h = int(np.ceil(h / multiple) * multiple)
        target_w = int(np.ceil(w / multiple) * multiple)

        pad_h = target_h - h
        pad_w = target_w - w

        if pad_h == 0 and pad_w == 0:
            return img, mask, valid_mask

        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        if img.ndim == 3:
            padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        padded_mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        padded_vm = cv2.copyMakeBorder(valid_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        return padded_img, padded_mask, padded_vm

    def _crop_pair(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        valid_mask: np.ndarray,
        crop_h: int,
        crop_w: int,
        is_forged: bool,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Crop or pad image, mask, and valid_mask to fixed (crop_h, crop_w) for training.

        Configurable share of crops (pos_crop_prob, default 50%) contain forged pixels
        by placing a randomly chosen forged pixel somewhere inside the crop window,
        without forcing the crop to be centered on the forgery.
        """
        if rng is None:
            rng = self.rng
        h, w = img.shape[:2]

        # 1. Zero-pad if image is smaller than target crop
        if h < crop_h or w < crop_w:
            pad_h = max(0, crop_h - h)
            pad_w = max(0, crop_w - w)
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left

            if img.ndim == 3:
                img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            else:
                img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            valid_mask = cv2.copyMakeBorder(valid_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            h, w = img.shape[:2]

        # 2. Extract crop
        if h == crop_h and w == crop_w:
            return img, mask, valid_mask

        max_y = h - crop_h
        max_x = w - crop_w

        if is_forged and np.sum(mask) > 0 and (rng.random() < self.pos_crop_prob):
            # Sample a crop that contains forged pixels, but NOT always centered
            ys, xs = np.where(mask > 0)
            if len(ys) > 0:
                rand_idx = int(rng.integers(0, len(ys)))
                yf = ys[rand_idx]
                xf = xs[rand_idx]

                min_y = max(0, yf - crop_h + 1)
                max_y_cand = min(yf, max_y)
                start_y = int(rng.integers(min_y, max_y_cand + 1)) if min_y <= max_y_cand else int(rng.integers(0, max_y + 1))

                min_x = max(0, xf - crop_w + 1)
                max_x_cand = min(xf, max_x)
                start_x = int(rng.integers(min_x, max_x_cand + 1)) if min_x <= max_x_cand else int(rng.integers(0, max_x + 1))
            else:
                start_y = int(rng.integers(0, max_y + 1))
                start_x = int(rng.integers(0, max_x + 1))
        else:
            # Completely uniform random crop
            start_y = int(rng.integers(0, max_y + 1))
            start_x = int(rng.integers(0, max_x + 1))

        end_y = start_y + crop_h
        end_x = start_x + crop_w

        return (
            img[start_y:end_y, start_x:end_x],
            mask[start_y:end_y, start_x:end_x],
            valid_mask[start_y:end_y, start_x:end_x],
        )

    def _joint_augment(
        self, img: np.ndarray, mask: np.ndarray, valid_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply random flips and rotations jointly to image, mask, and valid_mask."""
        # Horizontal flip
        if self.rng.random() < 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)
            valid_mask = np.fliplr(valid_mask)

        # Vertical flip
        if self.rng.random() < 0.5:
            img = np.flipud(img)
            mask = np.flipud(mask)
            valid_mask = np.flipud(valid_mask)

        # Random cardinal rotation (0, 90, 180, 270)
        k = int(self.rng.integers(0, 4))
        if k > 0:
            img = np.rot90(img, k)
            mask = np.rot90(mask, k)
            valid_mask = np.rot90(valid_mask, k)

        return (
            np.ascontiguousarray(img),
            np.ascontiguousarray(mask),
            np.ascontiguousarray(valid_mask),
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        f_rel = Path(row["path_F"])
        a_rel = Path(row["path_A"])

        # Flexible path resolution for local, Colab, or extracted zip layouts
        candidates_f = [
            self.images_dir / f_rel,
            self.images_dir.parent / f_rel,
            self.manifest_path.parent / f_rel,
            self.manifest_path.parent / self.split / f_rel.name,
            self.manifest_path.parent / f_rel.name,
        ]
        f_path = next((p for p in candidates_f if p.exists()), None)
        if f_path is None:
            raise FileNotFoundError(f"Could not find image {f_rel} in any candidate location: {[str(p) for p in candidates_f]}")

        candidates_a = [
            self.images_dir / a_rel,
            self.images_dir.parent / a_rel,
            self.manifest_path.parent / a_rel,
            self.manifest_path.parent / self.split / a_rel.name,
            self.manifest_path.parent / a_rel.name,
        ]
        a_path = next((p for p in candidates_a if p.exists()), None)

        img = cv2.imread(str(f_path))
        if img is None:
            raise RuntimeError(f"Could not read image: {f_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        is_forged = bool(row["is_forged"])
        if is_forged and a_path and a_path.exists():
            mask = cv2.imread(str(a_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
        else:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # Valid mask: initially all 1s (unpadded)
        valid_mask = np.ones(img.shape[:2], dtype=np.uint8)

        # 1. Cropping or padding:
        # Train uses random crops (with 50% positive coverage); Val always returns full image with padding only
        if self.split == "train" and self.crop_size is not None:
            crop_rng = self.rng if self.augment else np.random.default_rng(self.seed + idx)
            img, mask, valid_mask = self._crop_pair(
                img, mask, valid_mask, self.crop_size[0], self.crop_size[1], is_forged, rng=crop_rng
            )
        elif self.pad_multiple > 1:
            img, mask, valid_mask = self._pad_to_multiple(
                img, mask, valid_mask, self.pad_multiple
            )

        # 2. Joint geometric augmentation
        if self.augment:
            img, mask, valid_mask = self._joint_augment(img, mask, valid_mask)

        # 3. Convert to float tensors
        # Image: [C, H, W] normalized to [0, 1]
        img_tensor = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        # Mask: [1, H, W] float in {0.0, 1.0}
        mask_tensor = (mask > 127).astype(np.float32)[None, :, :]
        # Valid mask: [1, H, W] float in {0.0, 1.0}
        valid_mask_tensor = (valid_mask > 0).astype(np.float32)[None, :, :]
        label_val = 1.0 if is_forged else 0.0

        meta = {
            "sample_id": row["sample_id"],
            "pair_id": row["pair_id"],
            "doc_id": str(row["doc_id"]),
            "class": row["class"],
            "is_forged": is_forged,
            "degradation_level": row.get("degradation_level", "zero"),
            "transform_type": row.get("transform_type", "none"),
        }

        try:
            return {
                "image": torch.from_numpy(img_tensor),
                "mask": torch.from_numpy(mask_tensor),
                "valid_mask": torch.from_numpy(valid_mask_tensor),
                "label": torch.tensor(label_val, dtype=torch.float32),
                "meta": meta,
            }
        except NameError:
            # Fallback if torch is not installed
            return {
                "image": img_tensor,
                "mask": mask_tensor,
                "valid_mask": valid_mask_tensor,
                "label": label_val,
                "meta": meta,
            }


def verify_splits_disjointness(train_dataset: ColabDoppelDataset, val_dataset: ColabDoppelDataset):
    """Assert zero document leakage between train and validation datasets."""
    overlap = train_dataset.doc_ids.intersection(val_dataset.doc_ids)
    if len(overlap) > 0:
        raise AssertionError(f"FATAL: Document leakage detected between train and val splits ({len(overlap)} overlapping docs: {list(overlap)[:5]}...)")
    print(f"[VERIFIED] Train ({len(train_dataset.doc_ids)} docs) and Val ({len(val_dataset.doc_ids)} docs) are strictly disjoint.")
