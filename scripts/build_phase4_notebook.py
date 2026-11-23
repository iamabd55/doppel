"""Generate notebooks/phase4_train.ipynb using nbformat."""

import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

cells = []

# Cell 0: README Block
readme_text = """# Doppel Phase 4: CNN Training and Operating Point Selection (Google Colab T4)

This notebook runs the training, ablation, and threshold selection for the learned-feature branch of Doppel.

## 1. Setup Instructions (What to Upload to Google Drive)
Before running this notebook, place the following three zip files into a folder on your Google Drive:
- Target Drive directory: `My Drive/doppel/`
- Required archive files:
  1. `doppel_code.zip` (code archive containing detector, configs, tests, and training scripts)
  2. `synthetic_dataset.zip` (761 MB: 16,000 train samples + 1,000 val samples)
  3. `pristine_heldout.zip` (174 MB: 4,039 pristine panels from 177 held-out train documents)

## 2. Expected GPU Runtime
- Smoke test (2 epochs, 64 samples): ~5 to 10 seconds.
- Overfit test (100 epochs, 64 samples): ~1 to 2 minutes on T4 GPU.
- Full training per model (25 epochs, 16,000 samples, batch size 16): ~25 to 35 minutes on T4 GPU with AMP.
- Total active GPU time: ~1 hour for both models.

## 3. Disconnect and Resume Policy
If your Colab session disconnects during training:
- Reconnect and run Cells 1 through 3 to mount Drive and unpack assets into `/content`.
- Use the dedicated Resume Cell (Cell 9) to resume seamlessly from `last.pt` on Google Drive.

## 4. What to Copy and Paste Back
After the notebook finishes, copy and paste the following back:
1. Overfit test outcomes (Cell 6) confirming train F1 > 0.90 and Dice loss behavior.
2. The comparison evaluation table from Cell 11 comparing ResNetUNet, ResNetCorrNet, and ResNetCorrNet with ablated correlation.
3. The selected operating point printed in Cell 12 (`configs/cnn_threshold.yaml`).
"""
cells.append(nbf.v4.new_markdown_cell(readme_text))

# Cell 1: Check GPU and Versions
cell1_code = """import sys
import torch

print(f"Python version: {sys.version.split()[0]}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Active GPU: {gpu_name}")
    !nvidia-smi
else:
    print("[WARNING] CUDA is not available. Please go to Runtime -> Change runtime type -> T4 GPU.")
"""
cells.append(nbf.v4.new_code_cell(cell1_code))

# Cell 2: Mount Drive
cell2_code = """from google.colab import drive
import os
from pathlib import Path

# Mount Google Drive
drive.mount('/content/drive')

# Specify Drive directory containing the 3 uploaded zips
DRIVE_DIR = Path('/content/drive/MyDrive/doppel')
DRIVE_RUNS_DIR = DRIVE_DIR / 'runs'
DRIVE_RUNS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Drive directory: {DRIVE_DIR}")
print(f"Drive runs directory: {DRIVE_RUNS_DIR}")

# Verify zip files exist on Drive
required_zips = ['synthetic_dataset.zip', 'pristine_heldout.zip', 'doppel_code.zip']
missing = [z for z in required_zips if not (DRIVE_DIR / z).exists()]
if missing:
    print(f"[ERROR] Missing files on Google Drive: {missing}")
    print(f"Please upload them to {DRIVE_DIR} before continuing.")
else:
    print("[SUCCESS] All required asset archives located on Google Drive.")
"""
cells.append(nbf.v4.new_code_cell(cell2_code))

# Cell 3: Copy to /content and Unzip
cell3_code = """import shutil
import subprocess

CONTENT_DATA = Path('/content/data')
CONTENT_DATA.mkdir(parents=True, exist_ok=True)
SYNTH_DIR = CONTENT_DATA / 'synthetic'
SYNTH_DIR.mkdir(parents=True, exist_ok=True)

print("[INFO] Copying archive files from Google Drive to local /content...")
for zname in ['synthetic_dataset.zip', 'pristine_heldout.zip', 'doppel_code.zip']:
    src = DRIVE_DIR / zname
    dst = CONTENT_DATA / zname if zname != 'doppel_code.zip' else Path('/content') / zname
    if not dst.exists():
        shutil.copy2(src, dst)
        print(f"  Copied {zname}")

# Unzip code archive to /content
print("[INFO] Unpacking doppel_code.zip...")
!unzip -q -o /content/doppel_code.zip -d /content/

# Add doppel package to sys.path
DOPPEL_ROOT = Path('/content/doppel')
if str(DOPPEL_ROOT) not in sys.path:
    sys.path.insert(0, str(DOPPEL_ROOT))
    print(f"[INFO] Added {DOPPEL_ROOT} to sys.path.")

# Unzip synthetic dataset
print("[INFO] Unpacking synthetic_dataset.zip into local ephemeral disk...")
!unzip -q -o /content/data/synthetic_dataset.zip -d /content/data/synthetic/

# Verify file counts
train_files = list((SYNTH_DIR / 'train').glob('*_F.png'))
val_files = list((SYNTH_DIR / 'val').glob('*_F.png'))
print(f"[INFO] Verified synthetic dataset: {len(train_files)} train images, {len(val_files)} val images.")

# Install any missing dependencies
!pip install -q scikit-learn scikit-image opencv-python pyyaml
print("[SUCCESS] Environment and datasets initialized.")
"""
cells.append(nbf.v4.new_code_cell(cell3_code))

# Cell 4: Loader Sanity Check
cell4_code = """from torch.utils.data import DataLoader
from detector.colab_dataset import ColabDoppelDataset, colab_collate_fn

train_manifest = SYNTH_DIR / 'train_manifest.csv'
dataset_sanity = ColabDoppelDataset(
    manifest_path=str(train_manifest),
    images_dir=str(SYNTH_DIR),
    split='train',
    crop_size=(256, 256),
    pos_crop_prob=0.5,
    augment=True,
    seed=42,
)

loader_sanity = DataLoader(dataset_sanity, batch_size=8, shuffle=True, collate_fn=colab_collate_fn)
batch = next(iter(loader_sanity))

imgs = batch["image"]
masks = batch["mask"]
vms = batch["valid_mask"]
labels = batch["label"]

forged_frac = float(labels.mean().item())
mask_coverage = float((masks > 0).float().mean().item())

print(f"Batch image shape: {imgs.shape} (dtype: {imgs.dtype})")
print(f"Batch mask shape:  {masks.shape} (dtype: {masks.dtype})")
print(f"Batch valid_mask:  {vms.shape} (dtype: {vms.dtype})")
print(f"Forged sample fraction in batch: {forged_frac*100:.1f}%")
print(f"Mask pixel coverage: {mask_coverage*100:.2f}%")

assert imgs.shape[0] == 8 and imgs.shape[1] == 3 and imgs.shape[2] % 32 == 0 and imgs.shape[3] % 32 == 0
assert masks.shape[0] == 8 and masks.shape[1] == 1 and masks.shape[2] == imgs.shape[2] and masks.shape[3] == imgs.shape[3]
assert vms.shape == masks.shape
print("[PASS] Dataset loader sanity checks passed.")
"""
cells.append(nbf.v4.new_code_cell(cell4_code))

# Cell 5: Smoke Test
cell5_code = """import time

for mtype in ["unet", "corr"]:
    print(f"\\n================ Running Smoke Test for {mtype.upper()} ================")
    t_start = time.time()
    cmd = [
        "python3", str(DOPPEL_ROOT / "scripts" / "train_cnn.py"),
        "--model", mtype,
        "--epochs", "2",
        "--batch-size", "16",
        "--subset", "64",
        "--crop-size", "256",
        "--data-dir", str(SYNTH_DIR),
        "--pristine-zip", str(CONTENT_DATA / "pristine_heldout.zip"),
        "--pristine-val-count", "100",
        "--out-dir", f"/content/smoke_{mtype}",
        "--device", "cuda" if torch.cuda.is_available() else "cpu",
    ]
    subprocess.run(cmd, check=True)
    t_elapsed = time.time() - t_start
    sec_per_epoch = t_elapsed / 2.0
    est_full_min = (sec_per_epoch * 250 * 25) / 60.0
    print(f"[SMOKE {mtype.upper()}] Time for 2 epochs: {t_elapsed:.1f}s ({sec_per_epoch:.2f}s/epoch)")
    print(f"[EXTRAPOLATION] Estimated full 25-epoch run: ~{est_full_min:.1f} minutes on active device.")
"""
cells.append(nbf.v4.new_code_cell(cell5_code))

# Cell 6: Overfit Test
cell6_code = """import pandas as pd
import matplotlib.pyplot as plt

overfit_results = {}

for mtype in ["corr", "unet"]:
    print(f"\\n================ Running 100-Epoch Overfit Test for {mtype.upper()} ================")
    out_dir = Path(f"/content/overfit_{mtype}")
    cmd = [
        "python3", str(DOPPEL_ROOT / "scripts" / "train_cnn.py"),
        "--model", mtype,
        "--epochs", "100",
        "--batch-size", "8",
        "--subset", "64",
        "--crop-size", "128",
        "--lr", "1e-3",
        "--no-aug",
        "--val-interval", "20",
        "--data-dir", str(SYNTH_DIR),
        "--pristine-zip", str(CONTENT_DATA / "pristine_heldout.zip"),
        "--pristine-val-count", "0",
        "--out-dir", str(out_dir),
        "--device", "cuda" if torch.cuda.is_available() else "cpu",
    ]
    subprocess.run(cmd, check=True)

    df_log = pd.read_csv(out_dir / "train_log.csv")
    final_f1 = float(df_log["train_f1"].iloc[-1])
    final_dice = float(df_log["train_dice"].iloc[-1])
    overfit_results[mtype] = (final_f1, final_dice, df_log)

    print(f"[{mtype.upper()}] Final Train F1: {final_f1:.4f} | Final Dice Loss: {final_dice:.4f}")
    assert final_f1 > 0.90, f"Overfit test failed for {mtype}: Train F1 ({final_f1:.4f}) <= 0.90"
    diff = abs(final_dice - (1.0 - final_f1))
    print(f"[{mtype.upper()}] |Dice - (1 - F1)| = {diff:.4f} (target <= 0.15)")
    assert diff <= 0.15, f"Inconsistent Dice loss for {mtype}: diff {diff:.4f} > 0.15"
    print(f"[PASS] {mtype.upper()} cleanly memorized the training batch.")

# Plot overfit loss curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
for mtype in ["corr", "unet"]:
    plt.plot(overfit_results[mtype][2]["epoch"], overfit_results[mtype][2]["train_loss"], label=f"{mtype} Loss")
plt.title("Overfit Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
for mtype in ["corr", "unet"]:
    plt.plot(overfit_results[mtype][2]["epoch"], overfit_results[mtype][2]["train_f1"], label=f"{mtype} F1")
plt.title("Overfit Train F1")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""
cells.append(nbf.v4.new_code_cell(cell6_code))

# Cell 7: Train ResNetUNet
cell7_code = """unet_drive_dir = DRIVE_RUNS_DIR / "unet_run"
unet_local_dir = Path("/content/runs/unet_run")
unet_local_dir.mkdir(parents=True, exist_ok=True)

# Skip if finished run already exists on Drive
if (unet_drive_dir / "best.pt").exists() and (unet_drive_dir / "train_log.csv").exists():
    print(f"[INFO] Found completed UNet run at {unet_drive_dir}. Skipping training.")
    shutil.copytree(unet_drive_dir, unet_local_dir, dirs_exist_ok=True)
else:
    print("[INFO] Starting full training for ResNetUNet (25 epochs)...")
    cmd_unet = [
        "python3", str(DOPPEL_ROOT / "scripts" / "train_cnn.py"),
        "--model", "unet",
        "--epochs", "25",
        "--batch-size", "16",
        "--lr", "3e-4",
        "--crop-size", "256",
        "--data-dir", str(SYNTH_DIR),
        "--pristine-zip", str(CONTENT_DATA / "pristine_heldout.zip"),
        "--pristine-val-count", "1000",
        "--save-every", "2",
        "--out-dir", str(unet_local_dir),
        "--drive-sync-dir", str(unet_drive_dir),
        "--device", "cuda" if torch.cuda.is_available() else "cpu",
    ]
    subprocess.run(cmd_unet, check=True)
    print(f"[INFO] Syncing UNet artifacts to {unet_drive_dir}...")
    shutil.copytree(unet_local_dir, unet_drive_dir, dirs_exist_ok=True)
    print("[SUCCESS] UNet training complete and saved to Google Drive.")
"""
cells.append(nbf.v4.new_code_cell(cell7_code))

# Cell 8: Train ResNetCorrNet
cell8_code = """corr_drive_dir = DRIVE_RUNS_DIR / "corr_run"
corr_local_dir = Path("/content/runs/corr_run")
corr_local_dir.mkdir(parents=True, exist_ok=True)

# Skip if finished run already exists on Drive
if (corr_drive_dir / "best.pt").exists() and (corr_drive_dir / "train_log.csv").exists():
    print(f"[INFO] Found completed CorrNet run at {corr_drive_dir}. Skipping training.")
    shutil.copytree(corr_drive_dir, corr_local_dir, dirs_exist_ok=True)
else:
    print("[INFO] Starting full training for ResNetCorrNet (25 epochs)...")
    cmd_corr = [
        "python3", str(DOPPEL_ROOT / "scripts" / "train_cnn.py"),
        "--model", "corr",
        "--epochs", "25",
        "--batch-size", "16",
        "--lr", "3e-4",
        "--crop-size", "256",
        "--data-dir", str(SYNTH_DIR),
        "--pristine-zip", str(CONTENT_DATA / "pristine_heldout.zip"),
        "--pristine-val-count", "1000",
        "--save-every", "2",
        "--out-dir", str(corr_local_dir),
        "--drive-sync-dir", str(corr_drive_dir),
        "--device", "cuda" if torch.cuda.is_available() else "cpu",
    ]
    subprocess.run(cmd_corr, check=True)
    print(f"[INFO] Syncing CorrNet artifacts to {corr_drive_dir}...")
    shutil.copytree(corr_local_dir, corr_drive_dir, dirs_exist_ok=True)
    print("[SUCCESS] CorrNet training complete and saved to Google Drive.")
"""
cells.append(nbf.v4.new_code_cell(cell8_code))

# Cell 8b: Train ResNetCorrNet v2 (with Image Head)
cell8b_code = """corr_v2_drive_dir = DRIVE_RUNS_DIR / "corr_v2"
corr_v2_local_dir = Path("/content/runs/corr_v2")
corr_v2_local_dir.mkdir(parents=True, exist_ok=True)

# Skip if finished run already exists on Drive
if (corr_v2_drive_dir / "best.pt").exists() and (corr_v2_drive_dir / "train_log.csv").exists():
    print(f"[INFO] Found completed CorrNet v2 run at {corr_v2_drive_dir}. Skipping training.")
    shutil.copytree(corr_v2_drive_dir, corr_v2_local_dir, dirs_exist_ok=True)
else:
    print("[INFO] Starting full training for ResNetCorrNet v2 with Image Head (25 epochs, 120 min budget)...")
    cmd_corr_v2 = [
        "python3", str(DOPPEL_ROOT / "scripts" / "train_cnn.py"),
        "--model", "corr",
        "--image-head",
        "--epochs", "25",
        "--time-budget-min", "120",
        "--batch-size", "16",
        "--lr", "3e-4",
        "--pos-weight", "1.0",
        "--crop-size", "256",
        "--data-dir", str(SYNTH_DIR),
        "--pristine-zip", str(CONTENT_DATA / "pristine_heldout.zip"),
        "--pristine-val-count", "1000",
        "--save-every", "2",
        "--out-dir", str(corr_v2_local_dir),
        "--drive-sync-dir", str(corr_v2_drive_dir),
        "--device", "cuda" if torch.cuda.is_available() else "cpu",
    ]
    subprocess.run(cmd_corr_v2, check=True)
    print(f"[INFO] Syncing CorrNet v2 artifacts to {corr_v2_drive_dir}...")
    shutil.copytree(corr_v2_local_dir, corr_v2_drive_dir, dirs_exist_ok=True)
    print("[SUCCESS] CorrNet v2 training complete and saved to Google Drive.")

# Run diagnostics on last.pt immediately
diag_script = DOPPEL_ROOT / "scripts" / "diagnose_fa.py"
val_csv = SYNTH_DIR / "val_manifest.csv"
pristine_zip = CONTENT_DATA / "pristine_heldout.zip"
diag_v2_out = corr_v2_drive_dir / "diagnostics"
diag_v2_out.mkdir(parents=True, exist_ok=True)

ckpt_v2_last = corr_v2_drive_dir / "last.pt"
if not ckpt_v2_last.exists():
    ckpt_v2_last = corr_v2_local_dir / "last.pt"

if ckpt_v2_last.exists():
    print("\\n================ Running False Alarm Diagnostics on CorrNet v2 last.pt ================")
    !python3 "$diag_script" \\
        --checkpoint "$ckpt_v2_last" \\
        --model corr \\
        --image-head \\
        --val-manifest "$val_csv" \\
        --data-dir "$CONTENT_DATA" \\
        --pristine-zip "$pristine_zip" \\
        --pristine-subset 1000 \\
        --out-dir "$diag_v2_out"

    worst_fig = diag_v2_out / "worst_clean_panels.png"
    if worst_fig.exists():
        from IPython.display import Image, display
        print("\\nOverlay figure of 12 worst clean panels (CorrNet v2):")
        display(Image(filename=str(worst_fig)))
"""
cells.append(nbf.v4.new_code_cell(cell8b_code))

# Cell 9: Resume Cell
cell9_code = """# Instructions for Resuming after Disconnection:
# If session disconnected mid-run:
# 1. Run Cells 1-3 to remount Drive and setup /content data.
# 2. Set RESUME_MODEL to 'corr_v2', 'corr', or 'unet'.
# 3. Execute this cell to resume training from the latest checkpoint on Drive.

RESUME_MODEL = "corr_v2"  # 'corr_v2', 'corr', or 'unet'
run_dir_name = "corr_v2" if RESUME_MODEL == "corr_v2" else f"{RESUME_MODEL}_run"
resume_ckpt = DRIVE_RUNS_DIR / run_dir_name / "last.pt"

if resume_ckpt.exists():
    print(f"[INFO] Resuming {RESUME_MODEL} from {resume_ckpt}...")
    local_resume_dir = Path(f"/content/runs/{run_dir_name}")
    local_resume_dir.mkdir(parents=True, exist_ok=True)

    cmd_resume = [
        "python3", str(DOPPEL_ROOT / "scripts" / "train_cnn.py"),
        "--model", "corr",
        "--image-head",
        "--pos-weight", "1.0",
        "--epochs", "25",
        "--time-budget-min", "120",
        "--batch-size", "16",
        "--lr", "3e-4",
        "--crop-size", "256",
        "--resume", str(resume_ckpt),
        "--data-dir", str(SYNTH_DIR),
        "--pristine-zip", str(CONTENT_DATA / "pristine_heldout.zip"),
        "--pristine-val-count", "1000",
        "--save-every", "2",
        "--out-dir", str(local_resume_dir),
        "--drive-sync-dir", str(DRIVE_RUNS_DIR / run_dir_name),
        "--device", "cuda" if torch.cuda.is_available() else "cpu",
    ]
    subprocess.run(cmd_resume, check=True)
    shutil.copytree(local_resume_dir, DRIVE_RUNS_DIR / run_dir_name, dirs_exist_ok=True)
    print("[SUCCESS] Resumed training finished successfully.")
else:
    print(f"[INFO] No existing checkpoint found at {resume_ckpt} to resume.")
"""
cells.append(nbf.v4.new_code_cell(cell9_code))

# Cell 10: Diagnose False Alarms on CorrNet Checkpoints
cell10_diag_code = """diag_script = DOPPEL_ROOT / "scripts" / "diagnose_fa.py"
val_csv = SYNTH_DIR / "val_manifest.csv"
pristine_zip = CONTENT_DATA / "pristine_heldout.zip"
diag_out_dir = DRIVE_RUNS_DIR / "corr_v2" / "diagnostics"
if not (DRIVE_RUNS_DIR / "corr_v2").exists():
    diag_out_dir = DRIVE_RUNS_DIR / "corr_run" / "diagnostics"
diag_out_dir.mkdir(parents=True, exist_ok=True)

for ckpt_name in ["last.pt", "best.pt"]:
    ckpt_path = DRIVE_RUNS_DIR / "corr_v2" / ckpt_name
    if not ckpt_path.exists():
        ckpt_path = DRIVE_RUNS_DIR / "corr_run" / ckpt_name
    if not ckpt_path.exists():
        ckpt_path = Path(f"/content/runs/corr_v2/{ckpt_name}")
    if not ckpt_path.exists():
        ckpt_path = Path(f"/content/runs/corr_run/{ckpt_name}")

    if ckpt_path.exists():
        print(f"\\n================ Running False Alarm Diagnostics on {ckpt_name} ================")
        !python3 "$diag_script" \\
            --checkpoint "$ckpt_path" \\
            --model corr \\
            --val-manifest "$val_csv" \\
            --data-dir "$CONTENT_DATA" \\
            --pristine-zip "$pristine_zip" \\
            --pristine-subset 1000 \\
            --out-dir "$diag_out_dir"
    else:
        print(f"[INFO] Checkpoint {ckpt_path} not found. Skipping {ckpt_name}.")

# Display overlay figure of 12 worst clean panels
worst_fig = diag_out_dir / "worst_clean_panels.png"
if worst_fig.exists():
    from IPython.display import Image, display
    print("\\nOverlay figure of 12 worst clean panels:")
    display(Image(filename=str(worst_fig)))
"""
cells.append(nbf.v4.new_code_cell(cell10_diag_code))

# Cell 10b: Rescore Checkpoints in CorrNet v2 Run Directory
cell10b_rescore_code = """rescore_script = DOPPEL_ROOT / "scripts" / "rescore_checkpoints.py"
val_csv = SYNTH_DIR / "val_manifest.csv"
pristine_zip = CONTENT_DATA / "pristine_heldout.zip"
rescore_run_dir = DRIVE_RUNS_DIR / "corr_v2"
if not rescore_run_dir.exists():
    rescore_run_dir = Path("/content/runs/corr_v2")

if rescore_run_dir.exists():
    print(f"\\n================ Rescoring All Checkpoints in {rescore_run_dir.name} ================")
    rescore_out_csv = rescore_run_dir / "rescore_summary.csv"
    !python3 "$rescore_script" \\
        --run-dir "$rescore_run_dir" \\
        --val-manifest "$val_csv" \\
        --data-dir "$CONTENT_DATA" \\
        --pristine-zip "$pristine_zip" \\
        --pristine-subset 1000 \\
        --budgets 0.01 0.02 0.03 \\
        --out-csv "$rescore_out_csv"
else:
    print(f"[INFO] Run directory {rescore_run_dir} not found. Skipping rescore.")
"""
cells.append(nbf.v4.new_code_cell(cell10b_rescore_code))

# Cell 11: Plot Cell
cell11_plot_code = """import pandas as pd
import matplotlib.pyplot as plt

unet_csv = DRIVE_RUNS_DIR / "unet_run" / "train_log.csv"
corr_csv = DRIVE_RUNS_DIR / "corr_run" / "train_log.csv"

if unet_csv.exists() and corr_csv.exists():
    df_u = pd.read_csv(unet_csv)
    df_c = pd.read_csv(corr_csv)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Train Loss
    axes[0, 0].plot(df_u["epoch"], df_u["train_loss"], label="UNet", color="blue")
    axes[0, 0].plot(df_c["epoch"], df_c["train_loss"], label="CorrNet", color="crimson")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    # 2. Val F1
    axes[0, 1].plot(df_u["epoch"], df_u["val_forged_f1"], label="UNet", color="blue")
    axes[0, 1].plot(df_c["epoch"], df_c["val_forged_f1"], label="CorrNet", color="crimson")
    axes[0, 1].set_title("Validation Pixel F1")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # 3. ROC AUC
    axes[0, 2].plot(df_u["epoch"], df_u["val_roc_auc"], label="UNet", color="blue")
    axes[0, 2].plot(df_c["epoch"], df_c["val_roc_auc"], label="CorrNet", color="crimson")
    axes[0, 2].set_title("Validation ROC AUC")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].grid(True)
    axes[0, 2].legend()

    # 4. Recall at Budget
    axes[1, 0].plot(df_u["epoch"], df_u["val_recall_budget"], label="UNet", color="blue")
    axes[1, 0].plot(df_c["epoch"], df_c["val_recall_budget"], label="CorrNet", color="crimson")
    axes[1, 0].set_title("Validation Recall at Budget (<=1% FAR)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    # 5. Synthetic Clean FAR
    axes[1, 1].plot(df_u["epoch"], df_u["val_synth_clean_far"] * 100, label="UNet", color="blue")
    axes[1, 1].plot(df_c["epoch"], df_c["val_synth_clean_far"] * 100, label="CorrNet", color="crimson")
    axes[1, 1].axhline(1.0, color="gray", linestyle="--", label="1.0% Budget")
    axes[1, 1].set_title("Synthetic Val Clean FAR (%)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    # 6. Pristine Held-Out FAR
    axes[1, 2].plot(df_u["epoch"], df_u["val_pristine_far"] * 100, label="UNet", color="blue")
    axes[1, 2].plot(df_c["epoch"], df_c["val_pristine_far"] * 100, label="CorrNet", color="crimson")
    axes[1, 2].axhline(1.0, color="gray", linestyle="--", label="1.0% Budget")
    axes[1, 2].set_title("Pristine Held-Out FAR (%)")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].grid(True)
    axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig(DRIVE_RUNS_DIR / "training_curves_comparison.png", dpi=150)
    plt.show()
else:
    print("[WARNING] Log CSVs not found for both models. Ensure training has run.")
"""
cells.append(nbf.v4.new_code_cell(cell11_plot_code))

# Cell 11: Final Evaluation Cell
cell11_code = """eval_script = DOPPEL_ROOT / "scripts" / "eval_cnn.py"
val_csv = SYNTH_DIR / "val_manifest.csv"

unet_ckpt = DRIVE_RUNS_DIR / "unet_run" / "best.pt"
corr_ckpt = DRIVE_RUNS_DIR / "corr_v2" / "best.pt"
if not corr_ckpt.exists():
    corr_ckpt = DRIVE_RUNS_DIR / "corr_run" / "best.pt"

unet_eval_csv = DRIVE_RUNS_DIR / "eval_unet.csv"
corr_eval_csv = DRIVE_RUNS_DIR / "eval_corr.csv"
corr_abl_csv = DRIVE_RUNS_DIR / "eval_corr_ablated.csv"

print("================ Evaluating ResNetUNet ================")
!python3 "$eval_script" --checkpoint "$unet_ckpt" --model unet --val-manifest "$val_csv" --data-dir "$CONTENT_DATA" --pristine-zip "/content/data/pristine_heldout.zip" --out-csv "$unet_eval_csv"

print("\\n================ Evaluating ResNetCorrNet ================")
!python3 "$eval_script" --checkpoint "$corr_ckpt" --model corr --val-manifest "$val_csv" --data-dir "$CONTENT_DATA" --pristine-zip "/content/data/pristine_heldout.zip" --out-csv "$corr_eval_csv"

print("\\n================ Evaluating ResNetCorrNet (--ablate-corr) ================")
!python3 "$eval_script" --checkpoint "$corr_ckpt" --model corr --ablate-corr --val-manifest "$val_csv" --data-dir "$CONTENT_DATA" --pristine-zip "/content/data/pristine_heldout.zip" --out-csv "$corr_abl_csv"

# Build unified comparison table
df_u = pd.read_csv(unet_eval_csv)
df_c = pd.read_csv(corr_eval_csv)
df_a = pd.read_csv(corr_abl_csv)

print("\\n" + "=" * 110)
print(f"{'Slice':<22} | {'N':<6} | {'UNet Rec (Overlap)':<20} | {'CorrNet Rec (Overlap)':<22} | {'Ablated Corr Rec':<18} | {'Corr Delta':<10}")
print("-" * 110)

for i in range(len(df_u)):
    s_name = f"{df_u['category'].iloc[i]}: {df_u['slice'].iloc[i]}"
    n_val = df_u['n_forged'].iloc[i]
    n_str = f"{n_val}{'*' if n_val < 30 else ''}"
    r_u = df_u['recall_overlap'].iloc[i] * 100
    r_c = df_c['recall_overlap'].iloc[i] * 100
    r_a = df_a['recall_overlap'].iloc[i] * 100
    delta = r_c - r_a
    print(f"{s_name:<22} | {n_str:<6} | {r_u:>6.1f}%              | {r_c:>6.1f}%                | {r_a:>6.1f}%           | {delta:>+6.1f}%")

print("-" * 110)
print("* N < 30 indicates small sample size.")
print("=" * 110)
"""
cells.append(nbf.v4.new_code_cell(cell11_code))

# Cell 12: Threshold Selection Cell
cell12_code = """select_script = DOPPEL_ROOT / "scripts" / "select_threshold.py"
chosen_ckpt = DRIVE_RUNS_DIR / "corr_v2" / "best.pt"
if not chosen_ckpt.exists():
    chosen_ckpt = DRIVE_RUNS_DIR / "corr_run" / "best.pt"
out_yaml = DRIVE_DIR / "cnn_threshold.yaml"

print(f"[INFO] Running threshold selection on {chosen_ckpt}...")
!python3 "$select_script" \\
    --checkpoint "$chosen_ckpt" \\
    --model corr \\
    --val-manifest "$val_csv" \\
    --data-dir "$CONTENT_DATA" \\
    --pristine-zip "/content/data/pristine_heldout.zip" \\
    --out-yaml "$out_yaml" \\
    --budgets 0.01 0.02 0.03

print(f"\\n[SUCCESS] Threshold selection complete. Output saved to {out_yaml}")
if out_yaml.exists():
    print("\\n--- configs/cnn_threshold.yaml Contents ---")
    with open(out_yaml, 'r', encoding='utf-8') as f:
        print(f.read())
"""
cells.append(nbf.v4.new_code_cell(cell12_code))

nb["cells"] = cells

notebook_path = Path("notebooks/phase4_train.ipynb")
notebook_path.parent.mkdir(parents=True, exist_ok=True)
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"[SUCCESS] Wrote {notebook_path} ({len(cells)} cells).")
