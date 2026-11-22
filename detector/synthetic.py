"""Synthetic copy-move forgery generator for scientific images (BioFors).

Generates realistic copy-move pairs at native image resolution:
- Paired Twin negatives: clean negative samples undergo the exact same degradation
  pipeline as their forged twins (no shortcut features for CNNs).
- Degradation severity tiers: 'zero', 'light', 'moderate', 'heavy'.
- Donor patch selection with texture filtering and lane-aware strip sampling.
- Transform distribution: 60% translation, 20% cardinal/flip, 15% arbitrary, 5% combo.
- Masks: _F (composite/clean), _M (target), _S (source), _A (union, BioFors convention).
"""
import json
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def apply_patch_augmentation(
    patch: np.ndarray,
    transform_type: str,
    rotation_deg: float = 0.0,
    scale_factor: float = 1.0,
    flip_type: str = "none",
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply geometric transformation to a patch and its binary alpha mask."""
    h, w = patch.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255

    # 1. Flip
    if flip_type == "H-Flip":
        patch = cv2.flip(patch, 1)
        mask = cv2.flip(mask, 1)
    elif flip_type == "V-Flip":
        patch = cv2.flip(patch, 0)
        mask = cv2.flip(mask, 0)

    # 2. Scale
    if abs(scale_factor - 1.0) > 0.01:
        new_w = max(4, int(round(w * scale_factor)))
        new_h = max(4, int(round(h * scale_factor)))
        patch = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # 3. Rotation
    if abs(rotation_deg % 360.0) > 0.1:
        rot_norm = round(rotation_deg) % 360
        if rot_norm == 90:
            patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
            mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
        elif rot_norm == 180:
            patch = cv2.rotate(patch, cv2.ROTATE_180)
            mask = cv2.rotate(mask, cv2.ROTATE_180)
        elif rot_norm == 270:
            patch = cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
            mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            curr_h, curr_w = patch.shape[:2]
            center = (curr_w / 2.0, curr_h / 2.0)
            M = cv2.getRotationMatrix2D(center, -rotation_deg, 1.0)
            cos_a = abs(M[0, 0])
            sin_a = abs(M[0, 1])
            bound_w = int(curr_h * sin_a + curr_w * cos_a)
            bound_h = int(curr_h * cos_a + curr_w * sin_a)
            M[0, 2] += (bound_w / 2.0) - center[0]
            M[1, 2] += (bound_h / 2.0) - center[1]

            patch = cv2.warpAffine(patch, M, (bound_w, bound_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, M, (bound_w, bound_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return patch, mask


def sample_degradation_profile(level: str, rng: np.random.Generator) -> Dict[str, Any]:
    """Sample concrete degradation parameters for a chosen severity level."""
    profile: Dict[str, Any] = {
        "level": level,
        "blur_ksize": None,
        "blur_sigma": 0.0,
        "noise_sigma": 0.0,
        "gamma": 1.0,
        "jpeg_quality": None,
    }
    if level == "zero":
        return profile

    if level == "light":
        if rng.random() < 0.35:
            profile["blur_ksize"] = 3
            profile["blur_sigma"] = float(round(rng.uniform(0.4, 0.7), 2))
        if rng.random() < 0.35:
            profile["noise_sigma"] = float(round(rng.uniform(1.5, 3.5), 2))
        if rng.random() < 0.25:
            profile["gamma"] = float(round(rng.uniform(0.96, 1.04), 3))
        if rng.random() < 0.40:
            profile["jpeg_quality"] = int(rng.integers(88, 96))

    elif level == "moderate":
        if rng.random() < 0.50:
            profile["blur_ksize"] = int(rng.choice([3, 5]))
            profile["blur_sigma"] = float(round(rng.uniform(0.6, 1.0), 2))
        if rng.random() < 0.50:
            profile["noise_sigma"] = float(round(rng.uniform(3.0, 5.5), 2))
        if rng.random() < 0.40:
            profile["gamma"] = float(round(rng.uniform(0.92, 1.08), 3))
        if rng.random() < 0.60:
            profile["jpeg_quality"] = int(rng.integers(70, 86))

    elif level == "heavy":
        if rng.random() < 0.70:
            profile["blur_ksize"] = int(rng.choice([3, 5]))
            profile["blur_sigma"] = float(round(rng.uniform(0.9, 1.4), 2))
        if rng.random() < 0.70:
            profile["noise_sigma"] = float(round(rng.uniform(5.0, 8.0), 2))
        if rng.random() < 0.50:
            profile["gamma"] = float(round(rng.uniform(0.85, 1.15), 3))
        if rng.random() < 0.80:
            profile["jpeg_quality"] = int(rng.integers(50, 69))

    return profile


def apply_degradation_pipeline(img: np.ndarray, profile: Dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    """Apply the unified whole-image degradation pipeline to an image."""
    if profile.get("level") == "zero":
        return img.copy()

    out = img.astype(np.float32)

    # 1. Blur
    if profile.get("blur_ksize") is not None and profile.get("blur_sigma", 0) > 0:
        k = profile["blur_ksize"]
        s = profile["blur_sigma"]
        out = cv2.GaussianBlur(out, (k, k), s)

    # 2. Additive Gaussian noise
    if profile.get("noise_sigma", 0) > 0:
        noise = rng.normal(0.0, profile["noise_sigma"], out.shape)
        out = np.clip(out + noise, 0.0, 255.0)

    # 3. Gamma adjustment
    gamma = profile.get("gamma", 1.0)
    if abs(gamma - 1.0) > 0.005:
        inv_gamma = 1.0 / max(0.01, gamma)
        out = 255.0 * ((out / 255.0) ** inv_gamma)
        out = np.clip(out, 0.0, 255.0)

    out = np.round(out).astype(np.uint8)

    # 4. JPEG recompression
    q = profile.get("jpeg_quality")
    if q is not None:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(q)]
        success, enc = cv2.imencode(".jpg", out, encode_param)
        if success:
            dec = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
            if dec is not None:
                if img.ndim == 3 and dec.ndim == 2:
                    dec = cv2.cvtColor(dec, cv2.COLOR_GRAY2BGR)
                elif img.ndim == 2 and dec.ndim == 3:
                    dec = cv2.cvtColor(dec, cv2.COLOR_BGR2GRAY)
                out = dec

    return out


class SyntheticCopyMoveGenerator:
    """Configurable synthetic copy-move generator for scientific images."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.gen_cfg = config.get("generation", {})
        self.tf_cfg = config.get("transforms", {})
        self.deg_cfg = config.get("degradations", {})

        self.min_texture_std = self.gen_cfg.get("min_texture_std", 4.0)
        self.low_texture_share = self.gen_cfg.get("low_texture_share", 0.10)
        self.patch_min_area = float(self.gen_cfg.get("patch_min_area", 400.0))
        self.patch_max_area_ratio = float(self.gen_cfg.get("patch_max_area_ratio", 0.25))
        self.multi_copy_prob = float(self.gen_cfg.get("multi_copy_prob", 0.20))
        self.feather_prob = self.deg_cfg.get("edge_feathering_prob", 0.50)

    def select_donor_coordinates(
        self,
        img: np.ndarray,
        rng: np.random.Generator,
        is_blot: bool,
    ) -> Tuple[Optional[Tuple[int, int, int, int]], bool]:
        """Select a valid donor patch bounding box [x1, y1, x2, y2].

        Area is sampled log-uniformly from a floor of 400 px up to 25% of image area,
        without tuning quartiles to real test boxes.
        """
        h, w = img.shape[:2]
        img_area = float(h * w)

        allow_low_texture = rng.random() < self.low_texture_share
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        # Target patch area: log-uniform from floor 400 px up to 25% of image area
        min_floor = self.patch_min_area
        max_possible_area = 0.45 * w * (0.90 * h if is_blot else 0.45 * h)
        min_area = min(min_floor, max(64.0, 0.85 * max_possible_area))
        max_area = max(min_area + 10.0, min(self.patch_max_area_ratio * img_area, max_possible_area))
        if max_area <= min_area:
            max_area = min_area + 20.0

        for _ in range(50):
            target_area = float(np.exp(rng.uniform(np.log(min_area), np.log(max_area))))
            aspect_ratio = rng.uniform(0.3, 3.0)
            patch_w = int(np.sqrt(target_area * aspect_ratio))
            patch_h = int(np.sqrt(target_area / aspect_ratio))

            max_h_allowed = int(0.90 * h) if is_blot else int(0.45 * h)
            max_w_allowed = int(0.45 * w)

            patch_w = max(8, min(patch_w, max_w_allowed))
            patch_h = max(8, min(patch_h, max_h_allowed))

            # Ensure floor is satisfied after integer truncation if canvas geometry permits
            if patch_w * patch_h < min_area:
                if patch_w < max_w_allowed:
                    patch_w = min(max_w_allowed, int(np.ceil(min_area / max(1, patch_h))))
                if patch_w * patch_h < min_area and patch_h < max_h_allowed:
                    patch_h = min(max_h_allowed, int(np.ceil(min_area / max(1, patch_w))))

            if patch_h >= h or patch_w >= w:
                continue

            x1 = int(rng.integers(0, w - patch_w))
            y1 = int(rng.integers(0, h - patch_h))
            x2 = x1 + patch_w
            y2 = y1 + patch_h

            patch_region = gray[y1:y2, x1:x2]
            std_val = float(np.std(patch_region))

            if std_val >= self.min_texture_std or allow_low_texture:
                is_low_tex = bool(std_val < self.min_texture_std)
                return (x1, y1, x2, y2), is_low_tex

        # Fallback to center region
        pw = max(8, min(int(0.25 * w), w - 4))
        ph = max(8, min(int(0.25 * h), h - 4))
        x1 = max(0, (w - pw) // 4)
        y1 = max(0, (h - ph) // 4)
        return (x1, y1, x1 + pw, y1 + ph), True

    def sample_transformation(
        self,
        rng: np.random.Generator,
    ) -> Tuple[str, float, float, str]:
        """Sample transformation parameters according to the defined transform mix."""
        p = rng.random()
        p_trans = self.tf_cfg.get("translation_prob", 0.60)
        p_card = self.tf_cfg.get("cardinal_rot_flip_prob", 0.20)
        p_arb = self.tf_cfg.get("arbitrary_rot_scale_prob", 0.15)

        rot_deg = 0.0
        scale = 1.0
        flip = "none"

        if p < p_trans:
            tf_type = "translation"
        elif p < p_trans + p_card:
            tf_type = "cardinal_flip"
            cardinal_choice = rng.choice(["H-Flip", "V-Flip", "90", "180", "270"])
            if cardinal_choice in ["H-Flip", "V-Flip"]:
                flip = str(cardinal_choice)
            else:
                rot_deg = float(cardinal_choice)
        elif p < p_trans + p_card + p_arb:
            tf_type = "arbitrary"
            if rng.random() < 0.5:
                rot_deg = float(rng.uniform(self.tf_cfg.get("rotation_range", [1.0, 180.0])[0], self.tf_cfg.get("rotation_range", [1.0, 180.0])[1]))
            if rng.random() < 0.5:
                scale = float(rng.uniform(self.tf_cfg.get("scale_range", [0.5, 2.0])[0], self.tf_cfg.get("scale_range", [0.5, 2.0])[1]))
        else:
            tf_type = "combination"
            flip = str(rng.choice(["H-Flip", "V-Flip", "none"]))
            rot_deg = float(rng.uniform(1.0, 180.0))
            scale = float(rng.uniform(0.7, 1.4))

        return tf_type, round(rot_deg, 2), round(scale, 3), flip

    def generate(
        self,
        base_img: np.ndarray,
        sample_meta: Dict[str, Any],
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Generate a single sample with unified degradation handling."""
        is_forged = sample_meta.get("is_forged", True)
        h, w = base_img.shape[:2]

        # Determine degradation level and profile
        deg_level = sample_meta.get("degradation_level")
        if not deg_level:
            deg_level = str(rng.choice(["zero", "light", "moderate", "heavy"]))

        deg_profile = sample_meta.get("deg_profile")
        if not deg_profile:
            deg_profile = sample_degradation_profile(deg_level, rng)

        mask_M = np.zeros((h, w), dtype=np.uint8)
        mask_S = np.zeros((h, w), dtype=np.uint8)
        mask_A = np.zeros((h, w), dtype=np.uint8)

        meta = dict(sample_meta)
        meta["image_shape"] = [int(h), int(w)]
        meta["degradation_level"] = deg_level
        meta["degradations"] = deg_profile

        # 1. Clean Negative Sample (Paired Twin)
        if not is_forged:
            meta["is_forged"] = False
            meta["transform_type"] = "none"
            meta["rotation_deg"] = 0.0
            meta["scale_factor"] = 1.0
            meta["flip_type"] = "none"
            meta["source_bbox"] = None
            meta["target_bbox"] = None
            meta["patch_area"] = 0
            meta["low_texture"] = False

            # Apply the exact same whole-image degradation pipeline
            clean_degraded = apply_degradation_pipeline(base_img, deg_profile, rng)
            return clean_degraded, mask_M, mask_S, mask_A, meta

        # 2. Forged Copy-Move Sample
        is_blot = (meta.get("class") == "Blot/Gel")
        donor_coords, is_low_tex = self.select_donor_coordinates(base_img, rng, is_blot)
        if donor_coords is None:
            # Fallback to pristine if donor selection fails
            meta["is_forged"] = False
            clean_degraded = apply_degradation_pipeline(base_img, deg_profile, rng)
            return clean_degraded, mask_M, mask_S, mask_A, meta

        sx1, sy1, sx2, sy2 = donor_coords
        donor_patch = base_img[sy1:sy2, sx1:sx2].copy()

        # Determine if multi-copy (about 20% get 2 to 3 targets from one source)
        is_multi_copy = (rng.random() < self.multi_copy_prob)
        num_targets = int(rng.choice([2, 3], p=[0.80, 0.20])) if is_multi_copy else 1

        composited = base_img.copy()
        target_bboxes = []
        occupied_boxes = [[sx1, sy1, sx2, sy2]]
        primary_tf_type = None
        primary_rot_deg = 0.0
        primary_scale = 1.0
        primary_flip = "none"

        for tgt_idx in range(num_targets):
            is_adjacent_repeat = (is_blot and tgt_idx > 0 and rng.random() < 0.70)

            if is_adjacent_repeat:
                # Adjacent-band repeat: horizontal lane shift with subtle vertical variation
                tf_type = "translation"
                rot_deg = 0.0
                scale = 1.0
                flip = "none"
            else:
                tf_type, rot_deg, scale, flip = self.sample_transformation(rng)

            if tgt_idx == 0:
                primary_tf_type = tf_type
                primary_rot_deg = rot_deg
                primary_scale = scale
                primary_flip = flip

            trans_patch, patch_alpha = apply_patch_augmentation(donor_patch, tf_type, rot_deg, scale, flip)
            th, tw = trans_patch.shape[:2]

            # Fit target patch inside canvas
            if th >= h or tw >= w:
                scale = min((h - 4) / max(1, th), (w - 4) / max(1, tw)) * scale
                trans_patch, patch_alpha = apply_patch_augmentation(donor_patch, tf_type, rot_deg, scale, flip)
                th, tw = trans_patch.shape[:2]

            th = min(th, h - 2)
            tw = min(tw, w - 2)
            trans_patch = trans_patch[:th, :tw]
            patch_alpha = patch_alpha[:th, :tw]

            best_tx, best_ty = None, None

            if is_adjacent_repeat:
                # Try placing in adjacent lane
                delta_x = tgt_idx * int(1.25 * (sx2 - sx1) + rng.integers(12, 35))
                lane_jitter = int(rng.integers(-3, 4))
                cand_ty = max(0, min(sy1 + lane_jitter, h - th))
                if sx1 + delta_x + tw < w:
                    best_tx, best_ty = sx1 + delta_x, cand_ty
                elif sx1 - delta_x >= 0:
                    best_tx, best_ty = sx1 - delta_x, cand_ty

            if best_tx is None:
                # Sample non-overlapping placement
                for _ in range(35):
                    tx = int(rng.integers(0, max(1, w - tw)))
                    ty = int(rng.integers(0, max(1, h - th)))
                    overlaps = False
                    for ob in occupied_boxes:
                        ox1, oy1, ox2, oy2 = ob
                        inter_x1 = max(ox1, tx)
                        inter_y1 = max(oy1, ty)
                        inter_x2 = min(ox2, tx + tw)
                        inter_y2 = min(oy2, ty + th)
                        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                        ref_area = (ox2 - ox1) * (oy2 - oy1)
                        if inter_area / max(1, ref_area) > 0.30:
                            overlaps = True
                            break
                    if not overlaps:
                        best_tx, best_ty = tx, ty
                        break

            if best_tx is None:
                if tgt_idx == 0:
                    best_tx = int((sx1 + w // 2) % max(1, w - tw))
                    best_ty = int(sy1) if (sy1 + th) < h else 0
                else:
                    # Skip secondary target if cannot find valid placement
                    continue

            tx1, ty1 = best_tx, best_ty
            tx2, ty2 = tx1 + tw, ty1 + th

            # Edge feathering
            if rng.random() < self.feather_prob and th > 4 and tw > 4:
                alpha_f = patch_alpha.astype(np.float32) / 255.0
                alpha_f = cv2.GaussianBlur(alpha_f, (3, 3), 0.8)
            else:
                alpha_f = (patch_alpha > 128).astype(np.float32)

            bg_region = composited[ty1:ty2, tx1:tx2].astype(np.float32)
            fg_region = trans_patch.astype(np.float32)

            if composited.ndim == 3 and alpha_f.ndim == 2:
                alpha_3d = alpha_f[:, :, None]
                blended = alpha_3d * fg_region + (1.0 - alpha_3d) * bg_region
            else:
                blended = alpha_f * fg_region + (1.0 - alpha_f) * bg_region

            composited[ty1:ty2, tx1:tx2] = np.clip(blended, 0.0, 255.0).astype(np.uint8)
            mask_M[ty1:ty2, tx1:tx2] = np.maximum(mask_M[ty1:ty2, tx1:tx2], (patch_alpha > 0).astype(np.uint8) * 255)
            target_bboxes.append([int(tx1), int(ty1), int(tx2), int(ty2)])
            occupied_boxes.append([tx1, ty1, tx2, ty2])

        if len(target_bboxes) == 0:
            meta["is_forged"] = False
            clean_degraded = apply_degradation_pipeline(base_img, deg_profile, rng)
            return clean_degraded, mask_M, mask_S, mask_A, meta

        # Apply degradation pipeline to the final composited image
        forged_degraded = apply_degradation_pipeline(composited, deg_profile, rng)

        mask_S[sy1:sy2, sx1:sx2] = 255
        mask_A = cv2.bitwise_or(mask_M, mask_S)

        patch_area = int((sx2 - sx1) * (sy2 - sy1))

        meta["is_forged"] = True
        meta["transform_type"] = primary_tf_type
        meta["rotation_deg"] = primary_rot_deg
        meta["scale_factor"] = primary_scale
        meta["flip_type"] = primary_flip
        meta["source_bbox"] = [int(sx1), int(sy1), int(sx2), int(sy2)]
        meta["target_bbox"] = target_bboxes[0]
        meta["target_bboxes"] = target_bboxes
        meta["num_targets"] = len(target_bboxes)
        meta["is_multi_copy"] = bool(len(target_bboxes) > 1)
        meta["patch_area"] = patch_area
        meta["low_texture"] = is_low_tex

        return forged_degraded, mask_M, mask_S, mask_A, meta

    def generate_twin_pair(
        self,
        base_img: np.ndarray,
        pair_id_prefix: str,
        class_name: str,
        doc_id: str,
        rel_path: str,
        rng: np.random.Generator,
        degradation_level: Optional[str] = None,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]]:
        """Generate a paired twin: 1 forged sample and 1 matching clean control with identical degradations."""
        if not degradation_level:
            degradation_level = str(rng.choice(["zero", "light", "moderate", "heavy"]))

        deg_profile = sample_degradation_profile(degradation_level, rng)

        # Forged twin
        forged_meta_in = {
            "sample_id": f"{pair_id_prefix}_forged",
            "doc_id": doc_id,
            "source_image": rel_path,
            "class": class_name,
            "is_forged": True,
            "degradation_level": degradation_level,
            "deg_profile": deg_profile,
        }
        forged_result = self.generate(base_img, forged_meta_in, rng)

        # Clean twin (exact same base image, exact same degradation profile, zero paste)
        clean_meta_in = {
            "sample_id": f"{pair_id_prefix}_clean",
            "doc_id": doc_id,
            "source_image": rel_path,
            "class": class_name,
            "is_forged": False,
            "degradation_level": degradation_level,
            "deg_profile": deg_profile,
        }
        clean_result = self.generate(base_img, clean_meta_in, rng)

        return forged_result, clean_result
