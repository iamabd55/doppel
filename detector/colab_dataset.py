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
    import torch.nn.functional as F
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
        """Zero-pad image, mask and valid_mask to next multiple of multiple (e.g. 32). Never resizes."""
        h, w = img.shape[:2]
        target_h = int(np.ceil(h / multiple) * multiple)
        target_w = int(np.ceil(w / multiple) * multiple)

        pad_h = target_h - h
        pad_w = target_w - w

        if pad_h == 0 and pad_w == 0:
            return img, mask, valid_mask, (0, 0, 0, 0)

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
        return padded_img, padded_mask, padded_vm, (top, bottom, left, right)

    def _crop_pair(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        valid_mask: np.ndarray,
        crop_h: int,
        crop_w: int,
        is_forged: bool,
        mask_s: Optional[np.ndarray] = None,
        mask_m: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Extract crop guaranteeing both source and target partner presence for positive samples.

        If a forged pair does not fit in (crop_h, crop_w), enlarges crop to union bounding
        box plus margin (capped at 512, multiple of 32). If still incomplete, masks out
        unpartnered regions and treats crop as clean.
        """
        if rng is None:
            rng = self.rng
        h, w = img.shape[:2]

        attempt_pos = is_forged and (mask is not None) and (np.sum(mask) > 0) and (rng.random() < self.pos_crop_prob)

        if attempt_pos and (mask_s is not None) and (mask_m is not None):
            ys_s, xs_s = np.where(mask_s > 0)
            ys_m, xs_m = np.where(mask_m > 0)

            if len(ys_s) > 0 and len(ys_m) > 0:
                # Find connected components of target mask M to select a target partner
                num_m, labels_m, stats_m, _ = cv2.connectedComponentsWithStats(
                    (mask_m > 0).astype(np.uint8), connectivity=8
                )
                cand_targets = list(range(1, num_m)) if num_m > 1 else []
                if cand_targets:
                    chosen_t = int(rng.choice(cand_targets))
                    t_ys, t_xs = np.where(labels_m == chosen_t)
                else:
                    t_ys, t_xs = ys_m, xs_m

                # Union bounding box of source S and chosen target
                u_y1 = min(int(ys_s.min()), int(t_ys.min()))
                u_y2 = max(int(ys_s.max()), int(t_ys.max()))
                u_x1 = min(int(xs_s.min()), int(t_xs.min()))
                u_x2 = max(int(xs_s.max()), int(t_xs.max()))

                req_h = u_y2 - u_y1 + 1
                req_w = u_x2 - u_x1 + 1

                margin = 32
                target_h = max(crop_h, int(np.ceil((req_h + 2 * margin) / 32) * 32))
                target_w = max(crop_w, int(np.ceil((req_w + 2 * margin) / 32) * 32))
                target_h = min(512, target_h)
                target_w = min(512, target_w)

                # Pad image and masks if smaller than target crop
                pad_h = max(0, target_h - h)
                pad_w = max(0, target_w - w)
                top = pad_h // 2
                bottom = pad_h - top
                left = pad_w // 2
                right = pad_w - left

                if pad_h > 0 or pad_w > 0:
                    if img.ndim == 3:
                        c_img_src = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    else:
                        c_img_src = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
                    c_mask_src = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
                    c_vm_src = cv2.copyMakeBorder(valid_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
                    c_s_src = cv2.copyMakeBorder(mask_s, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
                    c_m_src = cv2.copyMakeBorder(mask_m, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
                    adj_u_y1 = u_y1 + top
                    adj_u_y2 = u_y2 + top
                    adj_u_x1 = u_x1 + left
                    adj_u_x2 = u_x2 + left
                else:
                    c_img_src = img
                    c_mask_src = mask
                    c_vm_src = valid_mask
                    c_s_src = mask_s
                    c_m_src = mask_m
                    adj_u_y1, adj_u_y2 = u_y1, u_y2
                    adj_u_x1, adj_u_x2 = u_x1, u_x2

                cur_h, cur_w = c_img_src.shape[:2]
                max_y = cur_h - target_h
                max_x = cur_w - target_w

                min_sy = max(0, adj_u_y2 - target_h)
                max_sy = min(adj_u_y1, max_y)
                sy = int(rng.integers(min_sy, max_sy + 1)) if min_sy <= max_sy else int(rng.integers(0, max_y + 1))

                min_sx = max(0, adj_u_x2 - target_w)
                max_sx = min(adj_u_x1, max_x)
                sx = int(rng.integers(min_sx, max_sx + 1)) if min_sx <= max_sx else int(rng.integers(0, max_x + 1))

                crop_img = c_img_src[sy : sy + target_h, sx : sx + target_w]
                crop_mask = c_mask_src[sy : sy + target_h, sx : sx + target_w].copy()
                crop_vm = c_vm_src[sy : sy + target_h, sx : sx + target_w]
                crop_s = c_s_src[sy : sy + target_h, sx : sx + target_w]
                crop_m = c_m_src[sy : sy + target_h, sx : sx + target_w]

                has_s = np.sum(crop_s > 0) > 0
                has_m = np.sum(crop_m > 0) > 0

                if has_s and has_m:
                    return crop_img, crop_mask, crop_vm, True
                else:
                    # Partner missing: zero out unpartnered pixels and treat as clean
                    crop_mask[:] = 0
                    return crop_img, crop_mask, crop_vm, False

        # Fallback / Uniform Random Crop
        target_h, target_w = crop_h, crop_w
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        if pad_h > 0 or pad_w > 0:
            if img.ndim == 3:
                c_img_src = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            else:
                c_img_src = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            c_mask_src = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            c_vm_src = cv2.copyMakeBorder(valid_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            c_s_src = cv2.copyMakeBorder(mask_s, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0) if mask_s is not None else None
            c_m_src = cv2.copyMakeBorder(mask_m, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0) if mask_m is not None else None
        else:
            c_img_src = img
            c_mask_src = mask
            c_vm_src = valid_mask
            c_s_src = mask_s
            c_m_src = mask_m

        cur_h, cur_w = c_img_src.shape[:2]
        max_y = cur_h - target_h
        max_x = cur_w - target_w

        sy = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
        sx = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0

        crop_img = c_img_src[sy : sy + target_h, sx : sx + target_w]
        crop_mask = c_mask_src[sy : sy + target_h, sx : sx + target_w].copy()
        crop_vm = c_vm_src[sy : sy + target_h, sx : sx + target_w]

        if is_forged and (c_s_src is not None) and (c_m_src is not None):
            crop_s = c_s_src[sy : sy + target_h, sx : sx + target_w]
            crop_m = c_m_src[sy : sy + target_h, sx : sx + target_w]
            has_s = np.sum(crop_s > 0) > 0
            has_m = np.sum(crop_m > 0) > 0
            if has_s and has_m:
                return crop_img, crop_mask, crop_vm, True
            else:
                crop_mask[:] = 0
                return crop_img, crop_mask, crop_vm, False
        elif is_forged and np.sum(crop_mask) > 0:
            return crop_img, crop_mask, crop_vm, True
        else:
            crop_mask[:] = 0
            return crop_img, crop_mask, crop_vm, False

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
        mask_s = None
        mask_m = None

        if is_forged and a_path and a_path.exists():
            mask = cv2.imread(str(a_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros(img.shape[:2], dtype=np.uint8)

            if self.split == "train":
                s_rel = Path(row["path_S"]) if "path_S" in row and pd.notna(row["path_S"]) else None
                m_rel = Path(row["path_M"]) if "path_M" in row and pd.notna(row["path_M"]) else None

                if s_rel is not None:
                    candidates_s = [
                        self.images_dir / s_rel,
                        self.images_dir.parent / s_rel,
                        self.manifest_path.parent / s_rel,
                        self.manifest_path.parent / self.split / s_rel.name,
                        self.manifest_path.parent / s_rel.name,
                    ]
                    s_path = next((p for p in candidates_s if p.exists()), None)
                    if s_path:
                        mask_s = cv2.imread(str(s_path), cv2.IMREAD_GRAYSCALE)

                if m_rel is not None:
                    candidates_m = [
                        self.images_dir / m_rel,
                        self.images_dir.parent / m_rel,
                        self.manifest_path.parent / m_rel,
                        self.manifest_path.parent / self.split / m_rel.name,
                        self.manifest_path.parent / m_rel.name,
                    ]
                    m_path = next((p for p in candidates_m if p.exists()), None)
                    if m_path:
                        mask_m = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # Valid mask: initially all 1s (unpadded)
        orig_h, orig_w = img.shape[:2]
        valid_mask = np.ones((orig_h, orig_w), dtype=np.uint8)
        pad_box = (0, 0, 0, 0)

        # 1. Cropping or padding:
        # Train uses random crops (with guaranteed source/target partner pairs for positives);
        # Val always returns full image with padding only
        if self.split == "train" and self.crop_size is not None:
            crop_rng = self.rng if self.augment else np.random.default_rng(self.seed + idx)
            img, mask, valid_mask, is_forged = self._crop_pair(
                img,
                mask,
                valid_mask,
                self.crop_size[0],
                self.crop_size[1],
                is_forged,
                mask_s=mask_s,
                mask_m=mask_m,
                rng=crop_rng,
            )
        elif self.pad_multiple > 1:
            img, mask, valid_mask, pad_box = self._pad_to_multiple(
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
            "num_targets": int(row.get("num_targets", 1)),
            "is_multi_copy": bool(row.get("is_multi_copy", False)),
            "orig_h": orig_h,
            "orig_w": orig_w,
            "pad_top": pad_box[0],
            "pad_bottom": pad_box[1],
            "pad_left": pad_box[2],
            "pad_right": pad_box[3],
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


def colab_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate batch with variable image/mask spatial sizes by zero-padding to batch maximum multiple of 32."""
    if not batch:
        return {}

    max_h = max(item["image"].shape[-2] for item in batch)
    max_w = max(item["image"].shape[-1] for item in batch)

    target_h = int(np.ceil(max_h / 32) * 32)
    target_w = int(np.ceil(max_w / 32) * 32)

    padded_images = []
    padded_masks = []
    padded_valid_masks = []
    labels = []
    metas = []

    for item in batch:
        img = item["image"]
        mask = item["mask"]
        vm = item["valid_mask"]
        lbl = item["label"]
        meta = item["meta"]

        h, w = img.shape[-2:]
        pad_h = target_h - h
        pad_w = target_w - w

        if pad_h > 0 or pad_w > 0:
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left

            img_p = F.pad(img, (left, right, top, bottom), value=0.0)
            mask_p = F.pad(mask, (left, right, top, bottom), value=0.0)
            vm_p = F.pad(vm, (left, right, top, bottom), value=0.0)
        else:
            img_p = img
            mask_p = mask
            vm_p = vm

        padded_images.append(img_p)
        padded_masks.append(mask_p)
        padded_valid_masks.append(vm_p)
        labels.append(lbl)
        metas.append(meta)

    return {
        "image": torch.stack(padded_images, dim=0),
        "mask": torch.stack(padded_masks, dim=0),
        "valid_mask": torch.stack(padded_valid_masks, dim=0),
        "label": torch.stack(labels, dim=0) if isinstance(labels[0], torch.Tensor) else torch.tensor(labels, dtype=torch.float32),
        "meta": {k: [m[k] for m in metas] for k in metas[0].keys()},
    }
