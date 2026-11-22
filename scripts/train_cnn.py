#!/usr/bin/env python3
"""Training script for Phase 4 CNN copy-move detection on synthetic data.

Models supported:
  - unet: Plain U-Net baseline with ResNet-18 backbone.
  - corr: ResNetCorrNet with stride-8 self-correlation and top-4 similarity map.

Features:
  - Loss: BCE with configurable pos_weight + soft Dice on union mask _A.
  - Optimizer: AdamW, cosine annealing lr schedule, gradient clipping, AMP support.
  - Checkpointing: last.pt every epoch, best.pt by validation metric, resume support.
  - Validation: Full native resolution images (padding only, no cropping).
  - Metrics: Pixel F1 and MCC, image-level ROC AUC, recall at threshold 0.5 (min area 50 px),
             synthetic val clean false alarm rate, and real pristine held-out false alarm rate.
  - Output tracking: config.json with git commit hash, and train_log.csv.
"""
import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from detector.cnn import ResNetCorrNet, ResNetUNet
from detector.colab_dataset import ColabDoppelDataset


def get_git_commit() -> str:
    """Retrieve current git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def set_seed(seed: int):
    """Set random seeds for determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CombinedLoss(nn.Module):
    """BCE with logits and soft Dice loss on binary masks."""

    def __init__(self, pos_weight: float = 2.0, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Weighted BCE
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )

        # Soft Dice
        probs = torch.sigmoid(logits)
        b = logits.shape[0]
        probs_flat = probs.view(b, -1)
        targets_flat = targets.view(b, -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        cardinality = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        # Smoothing numerator and denominator
        dice = 1.0 - (2.0 * intersection + 1.0) / (cardinality + 1.0)
        dice = dice.mean()

        total = self.bce_weight * bce + self.dice_weight * dice
        return total, bce, dice


def filter_min_area(binary_mask: np.ndarray, min_area: int = 50) -> np.ndarray:
    """Remove connected components smaller than min_area."""
    if np.sum(binary_mask) == 0:
        return binary_mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    filtered = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 1
    return filtered


def load_pristine_panels(pristine_zip_path: str, max_count: int = 200) -> List[Tuple[np.ndarray, str]]:
    """Load sample panels from pristine_heldout.zip for false alarm validation."""
    if max_count <= 0 or not os.path.exists(pristine_zip_path):
        return []

    panels = []
    with zipfile.ZipFile(pristine_zip_path, "r") as zf:
        # Read manifest
        manifest_names = [n for n in zf.namelist() if n.endswith("pristine_manifest.csv")]
        if not manifest_names:
            return []
        manifest_bytes = zf.read(manifest_names[0]).decode("utf-8")
        reader = csv.DictReader(manifest_bytes.splitlines())
        rows = list(reader)

        # Subsample if necessary
        if max_count > 0 and len(rows) > max_count:
            # Deterministic strided sampling
            stride = max(1, len(rows) // max_count)
            rows = rows[::stride][:max_count]

        for r in rows:
            arc_name = f"pristine_heldout/{r['path']}"
            if arc_name in zf.namelist():
                img_bytes = zf.read(arc_name)
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    panels.append((img, r["class"]))
    return panels


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    pristine_panels: List[Tuple[np.ndarray, str]],
    device: torch.device,
    fixed_thresh: float = 0.5,
    min_area: int = 50,
) -> Dict[str, float]:
    """Run full validation evaluation on uncropped validation panels."""
    model.eval()

    all_scores: List[float] = []
    all_labels: List[int] = []

    # Forged panel metric aggregators
    forged_tp = 0
    forged_fp = 0
    forged_fn = 0
    forged_tn = 0
    forged_detected = 0
    forged_total = 0

    # Synthetic clean twin metrics
    clean_flagged = 0
    clean_total = 0

    with torch.no_grad():
        for batch in val_loader:
            image = batch["image"].to(device)
            mask_gt = batch["mask"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            is_forged = bool(batch["label"].item() > 0.5)

            logits = model(image, valid_mask=valid_mask)
            probs = torch.sigmoid(logits)

            # Strip padding
            vm_bool = (valid_mask > 0.5)[0, 0].cpu().numpy()
            probs_np = probs[0, 0].cpu().numpy()
            mask_gt_np = (mask_gt[0, 0].cpu().numpy() > 0.5).astype(np.uint8)

            valid_probs = probs_np[vm_bool]
            max_prob = float(valid_probs.max()) if len(valid_probs) > 0 else 0.0

            all_scores.append(max_prob)
            all_labels.append(1 if is_forged else 0)

            # Binary prediction with min area filtering on valid region
            pred_binary = np.zeros_like(probs_np, dtype=np.uint8)
            pred_binary[(probs_np >= fixed_thresh) & vm_bool] = 1
            pred_filtered = filter_min_area(pred_binary, min_area=min_area)

            if is_forged:
                forged_total += 1
                # Overlap rule for detection: at least 1 overlapping pixel with GT
                overlap_pixels = int(np.sum((pred_filtered > 0) & (mask_gt_np > 0)))
                if overlap_pixels > 0:
                    forged_detected += 1

                # Pixel confusion
                p_valid = pred_filtered[vm_bool]
                g_valid = mask_gt_np[vm_bool]

                tp = int(np.sum((p_valid == 1) & (g_valid == 1)))
                fp = int(np.sum((p_valid == 1) & (g_valid == 0)))
                fn = int(np.sum((p_valid == 0) & (g_valid == 1)))
                tn = int(np.sum((p_valid == 0) & (g_valid == 0)))

                forged_tp += tp
                forged_fp += fp
                forged_fn += fn
                forged_tn += tn
            else:
                clean_total += 1
                if np.sum(pred_filtered) > 0:
                    clean_flagged += 1

    # Compute pixel metrics on forged validation panels
    denom_f1 = 2 * forged_tp + forged_fp + forged_fn
    val_f1 = (2 * forged_tp / denom_f1) if denom_f1 > 0 else 0.0

    mcc_num = float(forged_tp * forged_tn - forged_fp * forged_fn)
    mcc_den = np.sqrt(
        float(forged_tp + forged_fp)
        * float(forged_tp + forged_fn)
        * float(forged_tn + forged_fp)
        * float(forged_tn + forged_fn)
    )
    val_mcc = (mcc_num / mcc_den) if mcc_den > 0 else 0.0

    val_recall = (forged_detected / forged_total) if forged_total > 0 else 0.0
    val_clean_far = (clean_flagged / clean_total) if clean_total > 0 else 0.0

    # ROC AUC
    try:
        from sklearn.metrics import roc_auc_score
        val_auc = float(roc_auc_score(all_labels, all_scores))
    except Exception:
        val_auc = 0.50

    # Real pristine panel false alarm rate
    pristine_flagged = 0
    pristine_total = len(pristine_panels)

    if pristine_total > 0:
        with torch.no_grad():
            for p_img, _ in pristine_panels:
                h, w = p_img.shape[:2]
                pad_multiple = 32
                target_h = int(np.ceil(h / pad_multiple) * pad_multiple)
                target_w = int(np.ceil(w / pad_multiple) * pad_multiple)

                top = (target_h - h) // 2
                bottom = (target_h - h) - top
                left = (target_w - w) // 2
                right = (target_w - w) - left

                padded_img = cv2.copyMakeBorder(p_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                vm = np.zeros((target_h, target_w), dtype=np.uint8)
                vm[top : top + h, left : left + w] = 1

                t_img = torch.from_numpy(padded_img.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0).to(device)
                t_vm = torch.from_numpy(vm.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

                logits = model(t_img, valid_mask=t_vm)
                probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

                pred_bin = np.zeros_like(probs, dtype=np.uint8)
                pred_bin[(probs >= fixed_thresh) & (vm > 0)] = 1
                filtered_pred = filter_min_area(pred_bin, min_area=min_area)

                if np.sum(filtered_pred) > 0:
                    pristine_flagged += 1

        pristine_far = pristine_flagged / pristine_total
    else:
        pristine_far = 0.0

    return {
        "val_forged_f1": val_f1,
        "val_forged_mcc": val_mcc,
        "val_roc_auc": val_auc,
        "val_recall_at_50": val_recall,
        "val_synth_clean_far": val_clean_far,
        "val_pristine_far": pristine_far,
        "forged_count": forged_total,
        "clean_count": clean_total,
        "pristine_count": pristine_total,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train Phase 4 CNN copy-move detector.")
    parser.add_argument("--model", type=str, choices=["unet", "corr"], required=True, help="Model architecture.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--pos-weight", type=float, default=2.0, help="Positive class weight in BCE.")
    parser.add_argument("--subset", type=int, default=None, help="Train/val on a small subset of samples.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training.")
    parser.add_argument("--out-dir", type=str, default="checkpoints/run1", help="Output directory for checkpoints and logs.")
    parser.add_argument("--data-dir", type=str, default="data/synthetic", help="Path to synthetic dataset directory.")
    parser.add_argument("--pristine-zip", type=str, default="data/pristine_heldout.zip", help="Path to pristine held-out zip.")
    parser.add_argument("--pristine-val-count", type=int, default=200, help="Max pristine panels to evaluate per epoch.")
    parser.add_argument("--crop-size", type=int, default=256, help="Square crop size for training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--val-interval", type=int, default=1, help="Run validation every N epochs.")
    parser.add_argument("--no-aug", action="store_true", help="Disable training augmentations (for overfit test).")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"[INFO] Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Record run config and git commit hash
    git_hash = get_git_commit()
    config_dict = vars(args).copy()
    config_dict["git_commit"] = git_hash
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # Instantiate datasets
    train_manifest = Path(args.data_dir) / "train_manifest.csv"
    val_manifest = Path(args.data_dir) / "val_manifest.csv"

    if not train_manifest.exists():
        raise FileNotFoundError(f"Missing train manifest: {train_manifest}")
    if not val_manifest.exists():
        raise FileNotFoundError(f"Missing val manifest: {val_manifest}")

    train_dataset = ColabDoppelDataset(
        manifest_path=str(train_manifest),
        images_dir=args.data_dir,
        split="train",
        crop_size=(args.crop_size, args.crop_size),
        pos_crop_prob=0.50,
        pad_multiple=32,
        augment=not args.no_aug,
        seed=args.seed,
    )

    val_dataset = ColabDoppelDataset(
        manifest_path=str(val_manifest),
        images_dir=args.data_dir,
        split="val",
        crop_size=None,  # Full image with padding only
        pad_multiple=32,
        augment=False,
    )

    if args.subset is not None and args.subset > 0:
        subset_n = min(args.subset, len(train_dataset))
        train_dataset = Subset(train_dataset, list(range(subset_n)))
        val_subset_n = min(args.subset // 2 if args.subset > 4 else args.subset, len(val_dataset))
        val_dataset = Subset(val_dataset, list(range(val_subset_n)))
        print(f"[INFO] Running on subset: {len(train_dataset)} train, {len(val_dataset)} val samples")
    else:
        print(f"[INFO] Running on full dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")

    # If augmentations are disabled (e.g. overfit test), pre-cache samples in RAM to eliminate disk I/O
    if args.no_aug:
        print("[INFO] Pre-caching dataset in RAM for zero I/O training loop...")
        cached_train = [train_dataset[i] for i in range(len(train_dataset))]
        cached_val = [val_dataset[i] for i in range(len(val_dataset))]

        class _InMemory(torch.utils.data.Dataset):
            def __init__(self, data_list):
                self.data = data_list
            def __len__(self):
                return len(self.data)
            def __getitem__(self, i):
                return self.data[i]

        train_dataset = _InMemory(cached_train)
        val_dataset = _InMemory(cached_val)
        print("[INFO] Pre-caching complete.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Val loader uses batch_size=1 to accommodate variable native sizes
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Load pristine panels for false alarm checking
    pristine_panels = load_pristine_panels(args.pristine_zip, max_count=args.pristine_val_count)
    print(f"[INFO] Loaded {len(pristine_panels)} pristine held-out panels for false alarm checking")

    # Instantiate Model
    if args.model == "unet":
        model = ResNetUNet(pretrained=True)
    elif args.model == "corr":
        model = ResNetCorrNet(pretrained=True)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model.to(device)

    # Loss, Optimizer, Scheduler, Scaler
    criterion = CombinedLoss(pos_weight=args.pos_weight)
    criterion.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 1
    best_val_metric = -1.0
    last_val_metrics = {
        "val_forged_f1": 0.0,
        "val_forged_mcc": 0.0,
        "val_roc_auc": 0.5,
        "val_recall_at_50": 0.0,
        "val_synth_clean_far": 0.0,
        "val_pristine_far": 0.0,
    }

    # Resume support
    if args.resume and os.path.isfile(args.resume):
        print(f"[INFO] Resuming checkpoint from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt and use_amp:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_metric = ckpt.get("best_val_metric", -1.0)
        print(f"[INFO] Resumed from epoch {ckpt['epoch']}, best metric was {best_val_metric:.4f}")

    # CSV log setup
    log_csv_path = out_dir / "train_log.csv"
    csv_fields = [
        "epoch",
        "train_loss",
        "train_bce",
        "train_dice",
        "train_f1",
        "val_forged_f1",
        "val_forged_mcc",
        "val_roc_auc",
        "val_recall_at_50",
        "val_synth_clean_far",
        "val_pristine_far",
        "lr",
        "time_sec",
    ]

    if not log_csv_path.exists() or args.resume is None:
        with open(log_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(csv_fields)

    print(f"\n======================================================================")
    print(f"STARTING TRAINING: model={args.model}, epochs={args.epochs}, lr={args.lr}")
    print(f"Output directory: {out_dir}")
    print(f"======================================================================\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()

        total_loss = 0.0
        total_bce = 0.0
        total_dice = 0.0
        train_tp = 0
        train_fp = 0
        train_fn = 0
        num_batches = 0

        for batch in train_loader:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            valid_mask = batch["valid_mask"].to(device)

            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(image, valid_mask=valid_mask)
                loss, bce, dice = criterion(logits, mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_bce += bce.item()
            total_dice += dice.item()
            num_batches += 1

            # Monitor train pixel F1
            with torch.no_grad():
                preds = (torch.sigmoid(logits) >= 0.5).float()
                train_tp += int(((preds == 1.0) & (mask == 1.0)).sum().item())
                train_fp += int(((preds == 1.0) & (mask == 0.0)).sum().item())
                train_fn += int(((preds == 0.0) & (mask == 1.0)).sum().item())

        scheduler.step()
        elapsed = time.time() - t0

        avg_loss = total_loss / max(1, num_batches)
        avg_bce = total_bce / max(1, num_batches)
        avg_dice = total_dice / max(1, num_batches)
        train_denom = 2 * train_tp + train_fp + train_fn
        train_f1 = (2 * train_tp / train_denom) if train_denom > 0 else 0.0

        current_lr = optimizer.param_groups[0]["lr"]

        # Run Validation if due
        if (epoch % args.val_interval == 0) or (epoch == args.epochs):
            val_metrics = evaluate(
                model,
                val_loader,
                pristine_panels,
                device,
                fixed_thresh=0.5,
                min_area=50,
            )
            last_val_metrics = val_metrics
        else:
            val_metrics = last_val_metrics

        val_f1 = val_metrics["val_forged_f1"]
        val_mcc = val_metrics["val_forged_mcc"]
        val_auc = val_metrics["val_roc_auc"]
        val_rec = val_metrics["val_recall_at_50"]
        synth_far = val_metrics["val_synth_clean_far"]
        pristine_far = val_metrics["val_pristine_far"]

        print(
            f"Epoch {epoch:3d}/{args.epochs:3d} | "
            f"Train Loss: {avg_loss:.4f} (F1: {train_f1:.4f}) | "
            f"Val F1: {val_f1:.4f} | MCC: {val_mcc:.4f} | AUC: {val_auc:.4f} | "
            f"Rec@50: {val_rec*100:.1f}% | "
            f"SynthFAR: {synth_far*100:.2f}% | PristineFAR: {pristine_far*100:.2f}% | "
            f"({elapsed:.1f}s)"
        )

        # Log row to CSV
        with open(log_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{avg_loss:.6f}",
                f"{avg_bce:.6f}",
                f"{avg_dice:.6f}",
                f"{train_f1:.4f}",
                f"{val_f1:.4f}",
                f"{val_mcc:.4f}",
                f"{val_auc:.4f}",
                f"{val_rec:.4f}",
                f"{synth_far:.6f}",
                f"{pristine_far:.6f}",
                f"{current_lr:.6e}",
                f"{elapsed:.2f}",
            ])

        # Checkpoint saving
        ckpt_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if use_amp else None,
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_metric": best_val_metric,
            "val_metrics": val_metrics,
            "config": config_dict,
        }

        # Always save last.pt for resume
        torch.save(ckpt_state, out_dir / "last.pt")

        # Select best checkpoint by val metric: composite of Val F1 and AUC
        val_target_metric = val_f1 + 0.5 * val_auc
        if val_target_metric > best_val_metric:
            best_val_metric = val_target_metric
            ckpt_state["best_val_metric"] = best_val_metric
            torch.save(ckpt_state, out_dir / "best.pt")
            print(f"  --> Saved new best checkpoint to {out_dir / 'best.pt'} (score: {best_val_metric:.4f})")

    print(f"\n[INFO] Training complete. Artifacts saved in {out_dir}")


if __name__ == "__main__":
    main()
