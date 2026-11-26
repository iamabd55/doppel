"""Audit the two upstream DOA-GAN checkpoints on 50 CASIA2 calibration images."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import skimage
import skimage.io
import torch
import torchvision
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = ROOT / "doagan_clean"
WEIGHTS = ROOT / "doa-gan_weights"
RESULTS = ROOT / "results" / "natural"
SIZE = (320, 320)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TO_TENSOR = torchvision.transforms.ToTensor()


def load_model(model_class, checkpoint, channels):
    # Strict loading supplies all VGG weights from the checkpoint.
    vgg19 = torchvision.models.vgg19
    with patch.object(torchvision.models, "vgg19", lambda **_: vgg19(weights=None)):
        model = model_class(out_channel=channels)
    state = torch.load(WEIGHTS / checkpoint, map_location="cpu", weights_only=True)["model_state"]
    expected = model.state_dict()
    missing = sorted(expected.keys() - state.keys())
    unexpected = sorted(state.keys() - expected.keys())
    print(f"1. {checkpoint}: missing={missing}, unexpected={unexpected}, "
          f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}")
    model.load_state_dict(state, strict=True)
    return model.eval(), state


def audit_eval(model, state, checkpoint):
    batch_norms = [
        (name, module) for name, module in model.named_modules()
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))
    ]
    stats_loaded = bool(batch_norms) and all(
        torch.equal(getattr(module, stat), state[f"{name}.{stat}"])
        for name, module in batch_norms for stat in ("running_mean", "running_var")
    )
    assert not model.training and not torch.is_grad_enabled() and stats_loaded
    print(f"2. {checkpoint}: eval=True, no_grad=True, "
          f"checkpoint running mean/variance verified for {len(batch_norms)} BatchNorm modules")


def preprocess(path):
    # run_on_folder.py resizes the decoded RGB array before converting to float.
    rgb = cv2.resize(skimage.io.imread(str(path))[..., :3], SIZE, interpolation=cv2.INTER_LINEAR)
    # CustomTransform receives an already resized image.
    image = skimage.img_as_float32(skimage.img_as_float(rgb))
    tensor = TO_TENSOR((image - MEAN) / STD).float().unsqueeze(0)
    return rgb, tensor


def report_distributions(results):
    labels = results["label"].eq("forged")
    for score in ("peak", "mean", "area_gt_0.5", "det_logit"):
        values = results[score]
        groups = []
        for label in ("forged", "authentic"):
            subset = values[results["label"].eq(label)]
            groups.append(f"{label} mean={subset.mean():.6f}, median={subset.median():.6f}, "
                          f"range=[{subset.min():.6f}, {subset.max():.6f}]")
        print(f"   {score}: {'; '.join(groups)}; AUROC={roc_auc_score(labels, values):.4f}")


def smoke(model, checkpoint, channels, samples):
    results = []
    tiles = []
    for row in samples.itertuples(index=False):
        rgb, tensor = preprocess(ROOT / row.image_path)
        mask_logits, det_logit = model(tensor)
        if channels == 3:
            probabilities = mask_logits.softmax(dim=1)[0]
            # The upstream test uses channel 0 for target and channel 1 for source.
            heatmap = (probabilities[0] + probabilities[1]).clamp(0, 1)
        else:
            heatmap = mask_logits.sigmoid()[0, 0]
        heatmap = heatmap.numpy()
        peak = float(heatmap.max())
        results.append({
            "label": row.label,
            "peak": peak,
            "mean": float(heatmap.mean()),
            "area_gt_0.5": float((heatmap > 0.5).mean()),
            "det_logit": det_logit.item(),
        })

        original = cv2.cvtColor(skimage.img_as_ubyte(rgb), cv2.COLOR_RGB2BGR)
        colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        tile = np.concatenate((original, colored), axis=1)
        caption = f"{row.label}  peak={peak:.3f}"
        cv2.putText(tile, caption, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(tile, caption, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(tile)

    montage = np.concatenate([np.concatenate(tiles[i:i + 5], axis=1)
                              for i in range(0, len(tiles), 5)], axis=0)
    montage_path = RESULTS / f"montage_bugcheck_{channels}ch.png"
    if not cv2.imwrite(str(montage_path), montage):
        raise OSError(f"Could not write {montage_path}")
    print(f"6. {checkpoint}: 25 forged and 25 authentic; montage={montage_path}")
    report_distributions(pd.DataFrame(results))


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(ROOT / "data/natural/manifests/casia2.csv")
    calibration = manifest[manifest["split"].eq("calibration")]
    forged = calibration[calibration["label"].eq("forged")].head(25)
    authentic = calibration[calibration["label"].eq("authentic")].head(25)
    if len(forged) != 25 or len(authentic) != 25:
        raise ValueError("Expected at least 25 calibration images per class")
    samples = pd.concat([forged, authentic], ignore_index=True)

    spec = importlib.util.spec_from_file_location("upstream_doagan", UPSTREAM / "models.py")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    three_channel, three_state = load_model(upstream.DOA, "three_channel.pkl", 3)
    audit_eval(three_channel, three_state, "three_channel.pkl")

    example = ROOT / forged.iloc[0]["image_path"]
    _, tensor = preprocess(example)
    print(f"3. {example.name}: upstream RGB, linear resize {SIZE} before float conversion, "
          f"mean={np.array2string(MEAN)}, "
          f"std={np.array2string(STD)}, tensor mean={tensor.mean():.6f}, std={tensor.std():.6f}")

    logits = []
    for label, subset in (("forged", forged), ("authentic", authentic)):
        values = []
        for row in subset.head(5).itertuples(index=False):
            _, tensor = preprocess(ROOT / row.image_path)
            _, logit = three_channel(tensor)
            values.append(f"{logit.item():+.4f}")
        logits.append(f"{label} [{', '.join(values)}]")
    print(f"4. three_channel.pkl raw det_logit: {'; '.join(logits)}")
    print("5. Upstream run_on_folder.py discards det_logit and exports only softmax or "
          "sigmoid masks; it defines no image-level decision. Peak, mean and area "
          "above 0.5 summarize those masks for this audit.")

    smoke(three_channel, "three_channel.pkl", 3, samples)
    single_channel, single_state = load_model(upstream.DOA, "single_channel.pkl", 1)
    audit_eval(single_channel, single_state, "single_channel.pkl")
    smoke(single_channel, "single_channel.pkl", 1, samples)


if __name__ == "__main__":
    main()
