"""HTTP interface for Doppel's locked natural-image pipeline."""

import asyncio
import base64
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hashlib
from io import BytesIO
import logging
import os
from time import perf_counter
from typing import Annotated, Literal
from uuid import uuid4

import anyio
import numpy as np
from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel
from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException

from detector.natural import CLASSICAL_CONFIG, DOAGAN_CONFIG, ROOT, NaturalPipeline

logger = logging.getLogger("uvicorn.error")
FORMATS = ("JPEG", "PNG", "WEBP")
DETECTORS = ("dct", "sift", "doagan")
RULES = {
    "F1": "DCT OR SIFT",
    "F2": "DCT AND SIFT",
    "F3": "DCT OR SIFT OR DOA-GAN",
    "F4": "At least two of the three detectors",
    "F5": "(DCT OR SIFT) AND DOA-GAN",
}
DEFAULT_RULE = "F1"
MaskId = Literal["dct", "sift", "doagan", "F1", "F2", "F3", "F4", "F5", "none"]
LOCKED_FILES = {
    CLASSICAL_CONFIG: "e338ee56b65037e349b41f3a00935806653ce533651bf5b78e883ab7a3f7c51e",
    DOAGAN_CONFIG: "563b25ef232d4ee5dd5c7d5a5774a57f894671af337bd5658792a0a2047b5e99",
    ROOT / "doa-gan_weights/single_channel.pkl": "ffa14ea9a47c9779d9a013a150521247b3006ed5dddee900d18b4bc420f0edb5",
    ROOT / "doagan_clean/models.py": "4eb5d5033fe6bb07ec02d6941ad384b7e78387d4ea90cae3168d9c1503938a6c",
}
PIPELINE_VERSION = "natural-v1-" + hashlib.sha256(
    "".join(LOCKED_FILES.values()).encode("ascii")
).hexdigest()[:12]


@dataclass(frozen=True)
class Settings:
    max_file_bytes: int = 10 * 1024 * 1024
    max_pixels: int = 2_000_000
    max_side: int = 4096
    origins: tuple[str, ...] = ("http://localhost:3000", "http://127.0.0.1:3000")

    @classmethod
    def from_env(cls):
        defaults = cls()
        origins = os.getenv("DOPPEL_CORS_ORIGINS", ",".join(defaults.origins))
        settings = cls(
            max_file_bytes=int(os.getenv("DOPPEL_MAX_FILE_BYTES", str(defaults.max_file_bytes))),
            max_pixels=int(os.getenv("DOPPEL_MAX_PIXELS", str(defaults.max_pixels))),
            max_side=int(os.getenv("DOPPEL_MAX_SIDE", str(defaults.max_side))),
            origins=tuple(origin.strip() for origin in origins.split(",") if origin.strip()),
        )
        if min(settings.max_file_bytes, settings.max_pixels, settings.max_side) < 1:
            raise ValueError("Image limits must be positive")
        if "*" in settings.origins:
            raise ValueError("Configure specific CORS origins")
        return settings


class Decision(BaseModel):
    flagged: bool


class DoaganDecision(Decision):
    score: float
    cutoff: float


class DetectorDecisions(BaseModel):
    dct: Decision
    sift: Decision
    doagan: DoaganDecision


class Mask(BaseModel):
    media_type: Literal["image/png"] = "image/png"
    data_base64: str


class Analysis(BaseModel):
    request_id: str
    width: int
    height: int
    pipeline_version: str
    default_rule: Literal["F1"] = DEFAULT_RULE
    detectors: DetectorDecisions
    fusion: dict[str, Decision]
    masks: dict[str, Mask]


class Capabilities(BaseModel):
    pipeline_version: str
    formats: tuple[str, ...]
    max_file_bytes: int
    max_pixels: int
    max_side: int
    detectors: tuple[str, ...]
    fusion_rules: dict[str, str]
    default_rule: Literal["F1"] = DEFAULT_RULE
    mask_options: list[str]


class ErrorDetails(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    request_id: str
    error: ErrorDetails


def error_response(request_id, status, code, message, headers=None):
    return JSONResponse(
        {"request_id": request_id, "error": {"code": code, "message": message}},
        status_code=status, headers=headers,
    )


class RequestBoundary:
    """Apply a byte limit before parsing and attach a correlation header."""

    def __init__(self, app, *, max_body_bytes):
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        analysis = scope["method"] == "POST" and scope["path"].rstrip("/") == "/api/v1/analyze"

        async def send_headers(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = request_id
                if analysis:
                    headers["Cache-Control"] = "no-store"
            await send(message)

        if analysis:
            body = bytearray()
            while True:
                chunk = await receive()
                if chunk["type"] == "http.disconnect":
                    return
                data = chunk.get("body", b"")
                if len(body) + len(data) > self.max_body_bytes:
                    response = error_response(request_id, 413, "upload_too_large", "Request exceeds the upload limit.")
                    await response(scope, receive, send_headers)
                    return
                body.extend(data)
                if not chunk.get("more_body", False):
                    break

            async def replay_body():
                if body is not None:
                    return take_body()
                return await receive()

            def take_body():
                nonlocal body
                buffered = bytes(body)
                body = None
                return {"type": "http.request", "body": buffered, "more_body": False}

            await self.app(scope, replay_body, send_headers)
            return
        await self.app(scope, receive, send_headers)


class AnalysisError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status = status
        self.code = code


def decode_image(data, settings):
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format not in FORMATS:
                raise AnalysisError(415, "unsupported_format", "Upload a JPEG, PNG, or WebP image.")
            width, height = image.size
            if max(width, height) > settings.max_side or width * height > settings.max_pixels:
                raise AnalysisError(413, "image_too_large", "Image dimensions exceed the decoded size limit.")
            if getattr(image, "n_frames", 1) != 1:
                raise AnalysisError(415, "unsupported_format", "Upload a single-frame image.")
            oriented = ImageOps.exif_transpose(image)
            if "A" in oriented.getbands() or "transparency" in oriented.info:
                rgba = oriented.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                oriented = Image.alpha_composite(background, rgba)
            return np.asarray(oriented.convert("RGB"))
    except Image.DecompressionBombError as exc:
        raise AnalysisError(413, "image_too_large", "Image dimensions exceed the decoded size limit.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise AnalysisError(422, "invalid_image", "The uploaded image could not be decoded.") from exc


def analyze_image(data, pipeline, masks, settings, request_id):
    rgb = decode_image(data, settings)
    height, width = rgb.shape[:2]
    result = pipeline.predict(rgb)
    outputs = result.detectors | result.fusion
    images = {}
    for name in masks:
        encoded = BytesIO()
        Image.fromarray(outputs[name].mask.astype(np.uint8) * 255).save(encoded, format="PNG")
        images[name] = Mask(data_base64=base64.b64encode(encoded.getvalue()).decode("ascii"))
    return Analysis(
        request_id=request_id, width=width, height=height, pipeline_version=PIPELINE_VERSION,
        detectors=DetectorDecisions(
            dct=Decision(flagged=result.detectors["dct"].flagged),
            sift=Decision(flagged=result.detectors["sift"].flagged),
            doagan=DoaganDecision(flagged=result.detectors["doagan"].flagged,
                                  score=result.doagan_score, cutoff=pipeline.threshold),
        ),
        fusion={name: Decision(flagged=output.flagged) for name, output in result.fusion.items()},
        masks=images,
    )


def load_pipeline():
    import cv2
    import torch

    for path, expected in LOCKED_FILES.items():
        with path.open("rb") as artifact:
            if hashlib.file_digest(artifact, "sha256").hexdigest() != expected:
                raise ValueError(f"Locked file differs from the evaluation revision: {path}")
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    return NaturalPipeline()


def create_app(settings=None, pipeline_factory=load_pipeline):
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        app.state.pipeline = None
        app.state.inference = asyncio.Lock()
        try:
            app.state.pipeline = await anyio.to_thread.run_sync(pipeline_factory)
        except Exception:
            logger.exception("Could not load Doppel pipeline")
        yield
        app.state.pipeline = None

    app = FastAPI(title="Doppel API", version="1.0", lifespan=lifespan)
    app.add_middleware(RequestBoundary, max_body_bytes=settings.max_file_bytes + 64 * 1024)
    app.add_middleware(
        CORSMiddleware, allow_origins=list(settings.origins), allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"], expose_headers=["X-Request-ID", "Retry-After"],
    )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        return error_response(request.state.request_id, 422, "invalid_request", "Check the image field and mask options.")

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        message = "Request could not be processed." if exc.status_code != 404 else "Endpoint not found."
        return error_response(request.state.request_id, exc.status_code, "http_error", message, exc.headers)

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    @app.get("/readyz", responses={503: {"model": ErrorResponse}})
    async def readiness(request: Request):
        if getattr(app.state, "pipeline", None) is None:
            return error_response(request.state.request_id, 503, "not_ready", "The analysis pipeline is unavailable.")
        return {"status": "ready"}

    @app.get("/api/v1/capabilities", response_model=Capabilities)
    async def capabilities():
        return Capabilities(
            pipeline_version=PIPELINE_VERSION, formats=FORMATS, max_file_bytes=settings.max_file_bytes,
            max_pixels=settings.max_pixels, max_side=settings.max_side, detectors=DETECTORS,
            fusion_rules=RULES, mask_options=[*DETECTORS, *RULES, "none"],
        )

    @app.post("/api/v1/analyze", response_model=Analysis, responses={
        status: {"model": ErrorResponse} for status in (413, 415, 422, 500, 503)
    })
    async def analyze(
        request: Request, image: Annotated[UploadFile, File()],
        mask: Annotated[list[MaskId] | None, Query()] = None,
    ):
        request_id = request.state.request_id
        requested = list(dict.fromkeys(mask if mask is not None else [DEFAULT_RULE]))
        if "none" in requested and len(requested) != 1:
            return error_response(request_id, 422, "invalid_mask", "Use mask=none by itself.")
        pipeline = getattr(app.state, "pipeline", None)
        if pipeline is None:
            return error_response(request_id, 503, "not_ready", "The analysis pipeline is unavailable.")
        lock = app.state.inference
        if lock.locked():
            return error_response(request_id, 503, "busy", "The analysis pipeline is busy.", {"Retry-After": "1"})
        started = perf_counter()
        status = 500
        dimensions = "-"
        async with lock:
            try:
                data = await image.read(settings.max_file_bytes + 1)
                if len(data) > settings.max_file_bytes:
                    raise AnalysisError(413, "upload_too_large", "Image exceeds the file size limit.")
                result = await anyio.to_thread.run_sync(
                    analyze_image, data, pipeline, [] if requested == ["none"] else requested,
                    settings, request_id,
                )
                status = 200
                dimensions = f"{result.width}x{result.height}"
                return result
            except AnalysisError as exc:
                status = exc.status
                return error_response(request_id, status, exc.code, str(exc))
            except Exception:
                logger.exception("Analysis failed request_id=%s", request_id)
                return error_response(request_id, 500, "analysis_failed", "Image analysis failed.")
            finally:
                await image.close()
                logger.info("analysis request_id=%s status=%s dimensions=%s pipeline=%s elapsed_ms=%.1f",
                            request_id, status, dimensions, PIPELINE_VERSION, (perf_counter() - started) * 1000)

    return app


app = create_app()
