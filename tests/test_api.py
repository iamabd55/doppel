import asyncio
import base64
from contextlib import asynccontextmanager
from io import BytesIO
from threading import Event

import anyio
from httpx import ASGITransport, AsyncClient
import pytest
import numpy as np
from PIL import Image

from api.main import DETECTORS, RULES, Settings, create_app
from detector.natural import Detection, NaturalResult


@pytest.fixture
def anyio_backend():
    return "asyncio"


@asynccontextmanager
async def api_client(app):
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            yield client


class StubPipeline:
    threshold = 0.065703125

    def predict(self, rgb):
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        mask[0, 0] = True
        detection = Detection(True, mask)
        return NaturalResult(dict.fromkeys(DETECTORS, detection), dict.fromkeys(RULES, detection), 0.25)


def image_bytes(format="PNG", size=(7, 5), *, orientation=None):
    out = BytesIO()
    exif = Image.Exif()
    if orientation is not None:
        exif[274] = orientation
    Image.new("RGB", size, "white").save(out, format=format, exif=exif)
    return out.getvalue()


async def upload(client, data=None, query=""):
    return await client.post("/api/v1/analyze" + query, files={"image": ("input", data if data is not None else image_bytes())})


@pytest.mark.anyio
async def test_contract():
    async with api_client(create_app(pipeline_factory=StubPipeline)) as client:
        assert (await client.get("/healthz")).json() == {"status": "ok"}
        assert (await client.get("/readyz")).json() == {"status": "ready"}
        assert set((await client.get("/api/v1/capabilities")).json()["fusion_rules"]) == set(RULES)
        assert "/api/v1/analyze" in (await client.get("/openapi.json")).json()["paths"]
        response = await upload(client)
        assert response.status_code == 200
        body = response.json()
        assert body["request_id"] == response.headers["x-request-id"]
        assert response.headers["cache-control"] == "no-store"
        assert set(body["detectors"]) == set(DETECTORS)
        assert set(body["fusion"]) == set(RULES)
        assert set(body["masks"]) == {"F1"}
        assert body["detectors"]["doagan"]["cutoff"] == StubPipeline.threshold
        with Image.open(BytesIO(base64.b64decode(body["masks"]["F1"]["data_base64"]))) as mask:
            assert (mask.format, mask.mode, mask.size) == ("PNG", "L", (7, 5))
            assert set(np.unique(np.asarray(mask))) == {0, 255}
        assert (await upload(client, query="?mask=none")).json()["masks"] == {}
        assert set((await upload(client, query="?mask=F3&mask=F3&mask=sift")).json()["masks"]) == {"F3", "sift"}


@pytest.mark.anyio
async def test_validation():
    async with api_client(create_app(Settings(max_pixels=35, max_file_bytes=512), StubPipeline)) as client:
        for response, status, code in (
            (await client.post("/api/v1/analyze"), 422, "invalid_request"),
            (await upload(client, b"corrupt"), 422, "invalid_image"),
            (await upload(client, image_bytes("GIF")), 415, "unsupported_format"),
            (await upload(client, image_bytes(size=(8, 5))), 413, "image_too_large"),
            (await upload(client, b"x" * 513), 413, "upload_too_large"),
            (await upload(client, b"x" * 70_000), 413, "upload_too_large"),
            (await upload(client, query="?mask=bogus"), 422, "invalid_request"),
            (await upload(client, query="?mask=none&mask=F1"), 422, "invalid_mask"),
        ):
            assert response.status_code == status
            assert response.json()["error"]["code"] == code
            assert response.json()["request_id"] == response.headers["x-request-id"]


@pytest.mark.anyio
async def test_image_formats_and_orientation():
    async with api_client(create_app(pipeline_factory=StubPipeline)) as client:
        response = await upload(client, image_bytes("WEBP"))
        assert response.status_code == 200
        body = (await upload(client, image_bytes("JPEG", orientation=6))).json()
        assert (body["width"], body["height"]) == (5, 7)
        with Image.open(BytesIO(base64.b64decode(body["masks"]["F1"]["data_base64"]))) as mask:
            assert mask.size == (5, 7)


@pytest.mark.anyio
async def test_unready_and_cors():
    def unavailable():
        raise RuntimeError("missing checkpoint")

    async with api_client(create_app(pipeline_factory=unavailable)) as client:
        assert (await client.get("/healthz")).status_code == 200
        assert (await client.get("/readyz")).status_code == 503
        assert (await upload(client)).json()["error"]["code"] == "not_ready"
        response = await client.options("/api/v1/analyze", headers={
            "Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST",
        })
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


@pytest.mark.anyio
async def test_busy_while_health_remains_available():
    entered, release = Event(), Event()

    class BlockingPipeline(StubPipeline):
        def predict(self, rgb):
            entered.set()
            assert release.wait(5)
            return super().predict(rgb)

    async with api_client(create_app(pipeline_factory=BlockingPipeline)) as client:
        first = asyncio.create_task(upload(client))
        try:
            assert await anyio.to_thread.run_sync(entered.wait, 5)
            busy = await upload(client)
            assert busy.status_code == 503 and busy.headers["retry-after"] == "1"
            assert busy.json()["error"]["code"] == "busy"
            assert (await client.get("/healthz")).status_code == 200
        finally:
            release.set()
            result = await first
        assert result.status_code == 200
