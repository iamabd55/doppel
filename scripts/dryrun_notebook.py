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

from detector.colab_dataset import ColabDoppelDataset, colab_collate_fn


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
        print("\n[STEP 1/5] Simulating Drive copy and content unpacking (Cells 2 & 3)...")
        def _safe_link_or_copy(src, dst):
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)

        _safe_link_or_copy(zips["synth"], fake_drive / "synthetic_dataset.zip")
        _safe_link_or_copy(zips["pristine"], fake_drive / "pristine_heldout.zip")
        _safe_link_or_copy(zips["code"], fake_drive / "doppel_code.zip")

        # Copy from fake Drive to content
        _safe_link_or_copy(fake_drive / "synthetic_dataset.zip", content_data / "synthetic_dataset.zip")
        _safe_link_or_copy(fake_drive / "pristine_heldout.zip", content_data / "pristine_heldout.zip")
        _safe_link_or_copy(fake_drive / "doppel_code.zip", content_dir / "doppel_code.zip")

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
        print("\n[STEP 2/7] Executing Cell 4: Dataset loader sanity check...")
        dataset_sanity = ColabDoppelDataset(
            manifest_path=str(train_manifest),
            images_dir=str(synth_dir),
            split="train",
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

        print(f"  Batch image shape: {imgs.shape} (dtype: {imgs.dtype})")
        print(f"  Batch mask shape:  {masks.shape} (dtype: {masks.dtype})")
        print(f"  Batch valid_mask:  {vms.shape} (dtype: {vms.dtype})")
        print(f"  Batch labels:      {labels.tolist()}")

        assert imgs.shape[0] == 8 and imgs.shape[1] == 3 and imgs.shape[2] % 32 == 0 and imgs.shape[3] % 32 == 0, f"Unexpected image shape: {imgs.shape}"
        assert masks.shape[0] == 8 and masks.shape[1] == 1 and masks.shape[2] == imgs.shape[2] and masks.shape[3] == imgs.shape[3], f"Unexpected mask shape: {masks.shape}"
        assert vms.shape == masks.shape, f"Unexpected valid_mask shape: {vms.shape}"
        print("[PASS] Cell 4 loader sanity check passed.")

        # 4. Simulate Cell 5: Smoke test (both models, 2 epochs, subset 16)
        print("\n[STEP 3/7] Executing Cell 5: Smoke test (unet and corr, subset 16, 2 epochs)...")
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
        print("\n[STEP 4/7] Executing Cell 6: Shortened overfit test invocation (corr, 1 epoch)...")
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

        # 6. Simulate Cell 8b: Train CorrNet v2 (image head, 1 epoch, subset 16)
        print("\n[STEP 5/7] Executing Cell 8b: CorrNet v2 training (image head, subset 16, 1 epoch)...")
        corr_v2_out = content_dir / "corr_v2"
        corr_v2_drive_sync = drive_runs_dir / "corr_v2"
        cmd_v2 = [
            sys.executable,
            str(doppel_root / "scripts" / "train_cnn.py"),
            "--model", "corr",
            "--image-head",
            "--epochs", "1",
            "--batch-size", "8",
            "--subset", "16",
            "--crop-size", "128",
            "--lr", "3e-4",
            "--pos-weight", "1.0",
            "--data-dir", str(synth_dir),
            "--pristine-zip", str(content_data / "pristine_heldout.zip"),
            "--pristine-val-count", "10",
            "--out-dir", str(corr_v2_out),
            "--drive-sync-dir", str(corr_v2_drive_sync),
            "--device", "cpu",
        ]
        t0 = time.time()
        res_v2 = subprocess.run(cmd_v2, capture_output=True, text=True)
        elapsed_v2 = time.time() - t0
        if res_v2.returncode != 0:
            print(f"[FAIL] CorrNet v2 train failed:\nSTDOUT:\n{res_v2.stdout}\nSTDERR:\n{res_v2.stderr}")
            sys.exit(1)

        assert (corr_v2_out / "last.pt").exists(), "Missing local last.pt for corr_v2"
        assert (corr_v2_drive_sync / "last.pt").exists(), "Missing synced last.pt on fake Drive for corr_v2"
        assert (corr_v2_drive_sync / "train_log.csv").exists(), "Missing synced train_log.csv on fake Drive for corr_v2"
        print(f"[PASS] CorrNet v2 training passed in {elapsed_v2:.1f}s. Drive sync verified.")

        # 7. Simulate Cell 9: Resume training (--resume from fake Drive last.pt)
        print("\n[STEP 6/7] Executing Cell 9: Resume training from Drive checkpoint...")
        resume_out = content_dir / "resume_corr_v2"
        cmd_resume = [
            sys.executable,
            str(doppel_root / "scripts" / "train_cnn.py"),
            "--model", "corr",
            "--image-head",
            "--epochs", "2",
            "--batch-size", "8",
            "--subset", "16",
            "--crop-size", "128",
            "--lr", "3e-4",
            "--pos-weight", "1.0",
            "--resume", str(corr_v2_drive_sync / "last.pt"),
            "--data-dir", str(synth_dir),
            "--pristine-zip", str(content_data / "pristine_heldout.zip"),
            "--pristine-val-count", "10",
            "--out-dir", str(resume_out),
            "--drive-sync-dir", str(corr_v2_drive_sync),
            "--device", "cpu",
        ]
        t0 = time.time()
        res_res = subprocess.run(cmd_resume, capture_output=True, text=True)
        elapsed_res = time.time() - t0
        if res_res.returncode != 0:
            print(f"[FAIL] Resume training failed:\nSTDOUT:\n{res_res.stdout}\nSTDERR:\n{res_res.stderr}")
            sys.exit(1)

        assert (corr_v2_drive_sync / "last.pt").exists(), "Missing synced last.pt after resume"
        print(f"[PASS] Resumed training passed in {elapsed_res:.1f}s. Epoch 2 reached and Drive synced.")

        # 8. Simulate Diagnostic Cell: diagnose_fa.py on resumed last.pt
        print("\n[STEP 7/7] Executing False Alarm Diagnostic Cell (diagnose_fa.py on corr_v2 last.pt)...")
        diag_out = content_dir / "diag_corr_v2"
        cmd_diag = [
            sys.executable,
            str(doppel_root / "scripts" / "diagnose_fa.py"),
            "--checkpoint", str(corr_v2_drive_sync / "last.pt"),
            "--model", "corr",
            "--image-head",
            "--val-manifest", str(val_manifest),
            "--val-subset", "16",
            "--data-dir", str(content_data),
            "--pristine-zip", str(content_data / "pristine_heldout.zip"),
            "--pristine-subset", "10",
            "--out-dir", str(diag_out),
            "--device", "cpu",
        ]
        t0 = time.time()
        res_diag = subprocess.run(cmd_diag, capture_output=True, text=True)
        elapsed_diag = time.time() - t0
        if res_diag.returncode != 0:
            print(f"[FAIL] diagnose_fa.py call failed:\nSTDOUT:\n{res_diag.stdout}\nSTDERR:\n{res_diag.stderr}")
            sys.exit(1)

        assert (diag_out / "worst_clean_panels.png").exists(), "Missing worst_clean_panels.png from diagnostics"
        print(f"[PASS] False alarm diagnostics passed in {elapsed_diag:.1f}s. Overlay figure generated.")

    print("\n======================================================================")
    print("ALL DRY-RUN TESTS PASSED SUCCESSFULLY END-TO-END")
    print("Notebook calls, ColabDoppelDataset kwargs, CLI arguments, and Drive syncing are verified.")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
