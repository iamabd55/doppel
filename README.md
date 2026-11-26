# Doppel

Doppel detects and localizes copy-move forgery in natural images. The pipeline runs DCT, SIFT, and single-channel DOA-GAN and exposes their decisions and five fusion rules through a FastAPI backend for Next.js.

## Results and workflow

On 7,564 held-out CASIA2 images, calibrated DCT OR SIFT achieved 57.03% recall at a 1.98% false-alarm rate. Adding DOA-GAN to that OR rule raised recall to 58.79% and the false-alarm rate to 3.66%. See the [final evaluation](docs/natural_results.md) and [locked protocol](docs/natural_protocol.md) for the full comparison.

- `api/` serves uploads and returns detector decisions, fusion decisions, and PNG localization masks.
- `detector/` contains the natural-image pipeline, with DCT and SIFT in `detector/dct/` and `detector/sift/`.
- `configs/` contains the calibrated settings and the locked CoMoFoD subset.
- `scripts/` contains the natural-image evaluation tools and the API smoke test.
- `docs/` records the evaluation protocol, results, and API contract.

The earlier scientific-image experiments and the original `scripts/benchmark.py` referenced by the frozen protocol are available at Git tag `scientific-phase5`. The retired Flask demo is available in pre-cleanup Git history.

## FastAPI backend for Next.js

The natural-image API runs locally without a database. It uses the locked DCT, SIFT, and DOA-GAN pipeline and returns all five fusion decisions. F1 (calibrated DCT OR SIFT) is the default display rule.

From the repository root, install the API dependencies and the compatible CPU model runtime. The tested CPU pair is PyTorch 2.15.0 and torchvision 0.29.2. Use the PyTorch CPU wheel index for that pair.

```bash
uv sync --locked --group api --inexact
uv pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.15.0' 'torchvision==0.29.2'
```

Provision `doa-gan_weights/single_channel.pkl` from the DOA-GAN authors' checkpoint release and clone the pinned source if it is missing.

```bash
git clone https://github.com/asrafulashiq/doagan_clean.git doagan_clean
git -C doagan_clean checkout 081cc88aa1b67c250290311e78a7c4f101fc28d7
.venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

The server verifies the checkpoint, model source, and evaluation configs during startup. A failed load leaves `/readyz` unavailable and prints the cause in the server log. Visit `http://127.0.0.1:8000/docs` for the typed API contract.

```bash
curl -fsS http://127.0.0.1:8000/readyz
curl -fsS http://127.0.0.1:8000/api/v1/capabilities
curl -fsS -F 'image=@photo.jpg' 'http://127.0.0.1:8000/api/v1/analyze?mask=F1&mask=doagan'
```

For Next.js, send a `FormData` field named `image` and let the browser set the multipart header. Decode `masks[rule].data_base64` as a PNG and render it over the oriented image using the dimensions in the response. Use `mask=none` for decisions alone. JPEG, PNG, and single-frame WebP are supported. Images with transparency are composited on white. The default limits are 10 MiB per file, two million pixels, and 4,096 pixels on the longest side. Configure `DOPPEL_CORS_ORIGINS` as a comma-separated allowlist and the image limits with `DOPPEL_MAX_FILE_BYTES`, `DOPPEL_MAX_PIXELS`, and `DOPPEL_MAX_SIDE`. The default allowlist contains the localhost Next.js origins on port 3000. Analysis responses are not cached and the backend keeps no upload history.

Run `.venv/bin/python -m pytest -q tests/test_api.py` for the contract tests. With the model server running and the local datasets available, run `.venv/bin/python scripts/smoke_fastapi.py` for the five-image integration check.
