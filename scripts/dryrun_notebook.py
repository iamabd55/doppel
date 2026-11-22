#!/usr/bin/env python3
"""Local dry-run script executing notebooks/phase4_train.ipynb cells 4, 5, and shortened 6.

Validates the full Colab workflow locally on CPU with:
  - Google Drive replaced by a temporary folder with the three zips
  - Small subsets (--subset 16)
  - CPU compute only (no GPU, no nvidia-smi)
  - Verification of dataset loading, smoke testing, and per-epoch Drive synchronization

Strict rule: Never run full training locally.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from detector.colab_dataset import ColabDoppelDataset


def main():
    print("======================================================================")
    print("STARTING LOCAL DRY-RUN OF NOTEBOOK CELLS (CPU, SUBSET 16)")
    print("======================================================================")

    # 1. Verify source zips exist
    zips = {
        "code": REPO_ROOT / "data" / "doppel_code.zip",
        "synth": REPO_ROOT / "data" / "synthetic_dataset.zip",
        "pristine": REPO_ROOT / "data" / "pristine_heldout.zip",
    }

    for name, p in zips.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing required zip asset: {p}")
        print(f"[INFO] Found {name} zip: {p} ({p.stat().st_size / (1024*1024):.1f} MB)")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        fake_drive = tmp_path / "drive" / "MyDrive" / "doppel"
        fake_drive.mkdir(parents=True, exist_ok=True)
        drive_runs_dir = fake_drive / "runs"
        drive_runs_dir.mkdir(parents=True, exist_ok=True)

        content_dir = tmp_path / "content"
        content_dir.mkdir(parents=True, exist_ok=True)
        content_data = content_dir / "data"
        content_data.mkdir(parents=True, exist_ok=True)
        synth_dir = content_data / "synthetic"
        synth_dir.mkdir(parents=True, exist_ok=True)

        # 2. Simulate Cell 2 and Cell 3: copy zips to fake Drive, then copy to content and unpack
        print("\n[STEP 1/4] Simulating Drive copy and content unpacking (Cells 2 & 3)...")
        shutil.copy2(zips["synth"], fake_drive / "synthetic_dataset.zip")
        shutil.copy2(zips["pristine"], fake_drive / "pristine_heldout.zip")
        shutil.copy2(zips["code"], fake_drive / "doppel_code.zip")

        # Copy from fake Drive to content
        shutil.copy2(fake_drive / "synthetic_dataset.zip", content_data / "synthetic_dataset.zip")
        shutil.copy2(fake_drive / "pristine_heldout.zip", content_data / "pristine_heldout.zip")
        shutil.copy2(fake_drive / "doppel_code.zip", content_dir / "doppel_code.zip")

        # Unpack doppel_code.zip to content
        with zipfile.ZipFile(content_dir / "doppel_code.zip", "r") as zf:
            zf.extractall(content_dir)

        # Unpack synthetic_dataset.zip to synth_dir
        with zipfile.ZipFile(content_data / "synthetic_dataset.zip", "r") as zf:
            zf.extractall(synth_dir)

        doppel_root = content_dir / "doppel"
        if str(doppel_root) not in sys.path:
            sys.path.insert(0, str(doppel_root))

        print(f"[INFO] Unpacked files into {content_dir}")
        train_manifest = synth_dir / "train_manifest.csv"
        val_manifest = synth_dir / "val_manifest.csv"
        assert train_manifest.exists(), "Missing train_manifest.csv after extraction"
        assert val_manifest.exists(), "Missing val_manifest.csv after extraction"

        # 3. Simulate Cell 4: Loader sanity check
        print("\n[STEP 2/4] Executing Cell 4: Dataset loader sanity check...")
        dataset_sanity = ColabDoppelDataset(
            manifest_path=str(train_manifest),
            images_dir=str(synth_dir),
            split="train",
            crop_size=(256, 256),
            pos_crop_prob=0.5,
            augment=True,
            seed=42,
        )

        loader_sanity = DataLoader(dataset_sanity, batch_size=8, shuffle=True)
        batch = next(iter(loader_sanity))

        imgs = batch["image"]
        masks = batch["mask"]
        vms = batch["valid_mask"]
        labels = batch["label"]

        print(f"  Batch image shape: {imgs.shape} (dtype: {imgs.dtype})")
        print(f"  Batch mask shape:  {masks.shape} (dtype: {masks.dtype})")
        print(f"  Batch valid_mask:  {vms.shape} (dtype: {vms.dtype})")
        print(f"  Batch labels:      {labels.tolist()}")

        assert imgs.shape == (8, 3, 256, 256), f"Unexpected image shape: {imgs.shape}"
        assert masks.shape == (8, 1, 256, 256), f"Unexpected mask shape: {masks.shape}"
        assert vms.shape == (8, 1, 256, 256), f"Unexpected valid_mask shape: {vms.shape}"
        print("[PASS] Cell 4 loader sanity check passed.")

        # 4. Simulate Cell 5: Smoke test (both models, 2 epochs, subset 16)
        print("\n[STEP 3/4] Executing Cell 5: Smoke test (unet and corr, subset 16, 2 epochs)...")
        for mtype in ["unet", "corr"]:
            smoke_out = content_dir / f"smoke_{mtype}"
            smoke_drive_sync = drive_runs_dir / f"smoke_{mtype}"
            cmd = [
                sys.executable,
                str(doppel_root / "scripts" / "train_cnn.py"),
                "--model", mtype,
                "--epochs", "2",
                "--batch-size", "8",
                "--subset", "16",
                "--crop-size", "128",
                "--data-dir", str(synth_dir),
                "--pristine-zip", str(content_data / "pristine_heldout.zip"),
                "--pristine-val-count", "10",
                "--out-dir", str(smoke_out),
                "--drive-sync-dir", str(smoke_drive_sync),
                "--device", "cpu",
            ]
            t0 = time.time()
            res = subprocess.run(cmd, capture_output=True, text=True)
            elapsed = time.time() - t0
            if res.returncode != 0:
                print(f"[FAIL] Smoke test failed for {mtype}:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
                sys.exit(1)

            assert (smoke_out / "last.pt").exists(), f"Missing local last.pt for {mtype}"
            assert (smoke_out / "train_log.csv").exists(), f"Missing local train_log.csv for {mtype}"
            # Verify per-epoch Drive synchronization
            assert (smoke_drive_sync / "last.pt").exists(), f"Missing synced last.pt on fake Drive for {mtype}"
            assert (smoke_drive_sync / "train_log.csv").exists(), f"Missing synced train_log.csv on fake Drive for {mtype}"

            print(f"[PASS] Smoke test for {mtype} passed in {elapsed:.1f}s. Drive sync verified.")

        # 5. Simulate Cell 6: Shortened overfit test (corr model, subset 16, 1 epoch)
        print("\n[STEP 4/4] Executing Cell 6: Shortened overfit test invocation (corr, 1 epoch)...")
        overfit_out = content_dir / "overfit_corr"
        overfit_drive_sync = drive_runs_dir / "overfit_corr"
        cmd_overfit = [
            sys.executable,
            str(doppel_root / "scripts" / "train_cnn.py"),
            "--model", "corr",
            "--epochs", "1",
            "--batch-size", "8",
            "--subset", "16",
            "--crop-size", "128",
            "--lr", "1e-3",
            "--no-aug",
            "--val-interval", "1",
            "--data-dir", str(synth_dir),
            "--pristine-zip", str(content_data / "pristine_heldout.zip"),
            "--pristine-val-count", "0",
            "--out-dir", str(overfit_out),
            "--drive-sync-dir", str(overfit_drive_sync),
            "--device", "cpu",
        ]
        t0 = time.time()
        res_of = subprocess.run(cmd_overfit, capture_output=True, text=True)
        elapsed_of = time.time() - t0
        if res_of.returncode != 0:
            print(f"[FAIL] Overfit test call failed:\nSTDOUT:\n{res_of.stdout}\nSTDERR:\n{res_of.stderr}")
            sys.exit(1)

        assert (overfit_out / "last.pt").exists(), "Missing overfit last.pt"
        assert (overfit_drive_sync / "last.pt").exists(), "Missing synced overfit last.pt on fake Drive"
        assert (overfit_drive_sync / "train_log.csv").exists(), "Missing synced overfit train_log.csv on fake Drive"
        print(f"[PASS] Overfit test call passed in {elapsed_of:.1f}s. Drive sync verified.")

    print("\n======================================================================")
    print("ALL DRY-RUN TESTS PASSED SUCCESSFULLY END-TO-END")
    print("Notebook calls, ColabDoppelDataset kwargs, CLI arguments, and Drive syncing are verified.")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
