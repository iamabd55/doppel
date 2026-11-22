# Colab Environment Setup and Training Quickstart (Phase 4)

Guide for uploading and training the CNN branch on Google Colab GPU.

---

## 1. Package Preparation and Upload

The synthetic training dataset `data/synthetic_dataset.zip` contains:
- `train/`: 16,000 samples (8,000 forged images, 8,000 clean paired controls, binary ground-truth masks `_A`, `_M`, `_S`, and `_meta.json`).
- `val/`: 1,000 samples (500 forged images, 500 clean paired controls, binary ground-truth masks, and `_meta.json`).
- `train_manifest.csv` and `val_manifest.csv`.
- Excludes test split and all real BioFors data (held-out locally).

### Upload Steps
1. Upload `synthetic_dataset.zip` to your Google Drive under `My Drive/doppel/`.
2. In Colab, mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```
3. Unzip the dataset into Colab local storage (fast NVMe):
```bash
!mkdir -p /content/data/synthetic
!unzip -q /content/drive/MyDrive/doppel/synthetic_dataset.zip -d /content/data/synthetic
```

---

## 2. Environment Verification

Check GPU availability:
```python
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
```

---

## 3. Dataset Loader Sanity Check

```python
import sys
sys.path.append('/content/doppel')  # or path to cloned repo

from detector.colab_dataset import ColabDoppelDataset, verify_splits_disjointness
from torch.utils.data import DataLoader

train_ds = ColabDoppelDataset(
    manifest_path="/content/data/synthetic/train_manifest.csv",
    split="train",
    crop_size=(256, 256),
    augment=True,
)

val_ds = ColabDoppelDataset(
    manifest_path="/content/data/synthetic/val_manifest.csv",
    split="val",
    crop_size=(256, 256),
    augment=False,
)

# Assert strict document disjointness
verify_splits_disjointness(train_ds, val_ds)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
batch = next(iter(train_loader))

print("Batch image shape:", batch["image"].shape)  # [16, 3, 256, 256]
print("Batch mask shape :", batch["mask"].shape)   # [16, 1, 256, 256]
print("Batch label shape:", batch["label"].shape)  # [16]
print("Forged in batch  :", int(batch["label"].sum().item()), "/ 16")
```

---

## 4. Phase 4 Training Targets Reference
See `docs/phase4_targets.md` for pre-registered performance targets before model selection:
- CNN alone on BioFors IDD: Recall > 35.2% at Clean False Alarm Rate <= 1.50%.
- CNN + Classical Fusion: Recall > 41.8% at Clean False Alarm Rate <= 2.50%.
- Pixel F1: > 0.1200 pooled across manipulated panels.
