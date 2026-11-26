"""Compare five real HTTP uploads with direct locked-pipeline inference."""

import argparse
import base64
from io import BytesIO
from pathlib import Path
import sys
from time import perf_counter

import httpx
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.main import DETECTORS, RULES, Settings, decode_image, load_pipeline


def fixtures():
    manifests = ROOT / "data/natural/manifests"
    casia = pd.read_csv(manifests / "casia2.csv")
    calibration = casia[casia["split"].eq("calibration") & casia["image_path"].str.lower().str.endswith(".jpg")]
    forged = calibration[calibration["label"].eq("forged")].head(2)
    authentic = calibration[calibration["label"].eq("authentic")].head(2)
    comofod = pd.read_csv(manifests / "comofod.csv")
    dev = comofod[comofod["split"].eq("dev") & comofod["label"].eq("forged")
                  & comofod["image_path"].str.lower().str.endswith(".png")].head(1)
    selected = pd.concat([forged, authentic, dev])
    assert len(selected) == selected["image_path"].nunique() == 5
    return selected.itertuples(index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    settings = Settings.from_env()
    pipeline = load_pipeline()
    masks = [*DETECTORS, *RULES]
    lines = [
        "# FastAPI smoke test",
        "",
        "Five distinct local dataset images were uploaded over HTTP to Uvicorn with the real locked DOA-GAN checkpoint.",
        "The test compared all eight decisions and decoded masks with direct pipeline outputs for the same RGB pixels.",
        "Timings include upload, analysis, and the JSON response with all eight PNG masks. Labels describe the dataset.",
        "",
        "| Image | Split | Label | HTTP | Flagged detectors | Positive rules | Elapsed ms |",
        "|---|---|---|---:|---|---|---:|",
    ]
    with httpx.Client(base_url=args.url, timeout=120, trust_env=False) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/readyz").json() == {"status": "ready"}
        version = client.get("/api/v1/capabilities").json()["pipeline_version"]
        observed_f1 = set()
        for row in fixtures():
            path = ROOT / row.image_path
            data = path.read_bytes()
            rgb = decode_image(data, settings)
            direct = pipeline.predict(rgb)
            started = perf_counter()
            response = client.post("/api/v1/analyze", params=[("mask", name) for name in masks],
                                   files={"image": (path.name, data)})
            elapsed = 1000 * (perf_counter() - started)
            response.raise_for_status()
            body = response.json()
            assert body["pipeline_version"] == version
            assert body["request_id"] == response.headers["x-request-id"]
            assert (body["height"], body["width"]) == rgb.shape[:2]
            assert set(body["detectors"]) == set(DETECTORS) and set(body["fusion"]) == set(RULES)
            assert body["detectors"]["doagan"]["score"] == direct.doagan_score
            assert body["detectors"]["doagan"]["cutoff"] == pipeline.threshold
            assert set(body["masks"]) == set(masks)
            for name, expected in (direct.detectors | direct.fusion).items():
                decision = body["detectors" if name in DETECTORS else "fusion"][name]
                assert decision["flagged"] == expected.flagged, (path.name, name, "decision")
                returned = body["masks"][name]
                assert returned["media_type"] == "image/png"
                with Image.open(BytesIO(base64.b64decode(returned["data_base64"], validate=True))) as mask:
                    assert (mask.format, mask.mode, mask.size) == ("PNG", "L", (body["width"], body["height"]))
                    assert np.array_equal(np.asarray(mask), expected.mask.astype(np.uint8) * 255), (path.name, name, "mask")
            observed_f1.add(body["fusion"]["F1"]["flagged"])
            flagged = ", ".join(name for name in DETECTORS if direct.detectors[name].flagged) or "None"
            positive = ", ".join(name for name in RULES if direct.fusion[name].flagged) or "None"
            line = (f"| `{path.name}` | {row.dataset}/{row.split} | {row.label} | {response.status_code} | "
                    f"{flagged} | {positive} | {elapsed:.0f} |")
            print(line, flush=True)
            lines.append(line)
        assert observed_f1 == {True, False}, "Smoke images must exercise both F1 decisions"
    if args.report:
        args.report.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
