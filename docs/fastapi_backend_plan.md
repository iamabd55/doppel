# Doppel FastAPI backend plan

**Status** Implemented. See the [API smoke-test report](fastapi_smoke_results.md) for live verification.

## Scope and database decision

Build a Python JSON API for analyzing one natural image at a time. The later Next.js frontend can upload a file, choose a fusion rule to display, and render a localization overlay. The original Flask demo was later retired during repository cleanup and remains available in Git history.

**No database for v1.** Analysis starts and finishes within one request. The response carries the decisions and requested masks, and the service does not retain uploads or results after sending it. The request ID correlates logs and errors. It is not a result lookup key. A browser that needs to display the result again can keep it in application state. Revisit storage if the product adds analysis history, accounts, sharing, or durable background jobs. A future implementation can store metadata in a relational database and image blobs in object storage if those features require persistence.

## API contract

Use `/api/v1` for frontend-facing routes. Return typed JSON and publish the schema through FastAPI's built-in OpenAPI endpoint.

| Method and path | Purpose | Success |
|---|---|---|
| `GET /healthz` | Check whether the server responds | `200` with `status: "ok"` |
| `GET /readyz` | Check whether the pipeline is loaded and ready | `200` with `status: "ready"` |
| `GET /api/v1/capabilities` | Discover formats, input limits, detector names, fusion rules, mask options, and the active model version | `200` with the capabilities object |
| `POST /api/v1/analyze` | Upload one image as multipart field `image` and run DCT, SIFT, DOA-GAN, and F1–F5 | `200` with decisions and requested masks |
| `GET /openapi.json`, `GET /docs` | Machine-readable contract and interactive API reference | Built-in FastAPI responses |

### Analysis request

- Upload one JPEG, PNG, or WebP image. Support grayscale and RGB and normalize them to a single RGB representation. Reject animated and multipage inputs. Convert alpha images to RGB with one documented conversion policy. Apply EXIF orientation before inference.
- Accept repeated optional `mask` query parameters from `dct`, `sift`, `doagan`, and `F1` through `F5`. Default to `F1`. Allow `mask=none` by itself for decisions only. Deduplicate mask requests.
- Start with configurable limits of 10 MiB per file, 2 million decoded pixels, and 4,096 pixels on the longest side. Allow a bounded amount of multipart overhead. Check decoded dimensions before loading pixel data. Keep images at their accepted decoded size for classical inference. Measure latency and peak resource use before changing limits.

### Analysis response

| Field | Meaning |
|---|---|
| `request_id` | Opaque request identifier, also returned in `X-Request-ID` |
| `width`, `height` | Dimensions in the oriented image used for inference and masks |
| `pipeline_version` | Identifier for the implementation and locked detector configuration |
| `default_rule` | `F1`, calibrated DCT OR SIFT |
| `detectors` | Decisions for `dct`, `sift`, and `doagan`, each with a boolean `flagged`. DOA-GAN also returns its area score and cutoff. |
| `fusion` | Decisions for every rule from `F1` to `F5`, each with a boolean `flagged` |
| `masks` | Object keyed by the requested IDs. Each entry contains `media_type: "image/png"` and a base64-encoded grayscale binary PNG at the returned dimensions. Pixel values are 0 and 255. |

Return an all-zero mask when requested and the result has no localized pixels. Preserve the pipeline's distinction between an image-level flag and mask overlap. For example, F2 can flag an image when DCT and SIFT both vote positive even if their masks do not intersect. The frontend can decode PNGs to browser blobs and draw them against the oriented upload. The service should use `Cache-Control: no-store` on analysis responses.

### Error contract

Use a stable JSON envelope with `request_id`, `error.code`, and `error.message`. Return 413 for byte or dimension limits, 415 for unsupported image formats, 422 for missing fields, invalid options or unreadable images, and 503 for unready or saturated inference. A busy response should include `Retry-After`. Return a generic 500 to clients on unexpected failures and log the details server-side without image contents or upload filenames.

## Detector and runtime decisions

- Reuse `detector.natural.NaturalPipeline` and the committed classical and DOA-GAN configs. It already returns the three detector outputs and all five fusion results. Keep the default display rule separate from the inference rules. Calibrated F1 achieved 57.03% recall at a 1.98% observed false-alarm rate on held-out CASIA2. Adding DOA-GAN in F3 reached 58.79% recall at 3.66% false alarms. F1 is the default for the lower observed false-alarm rate. Describe detector flags as findings for review and the DOA-GAN area score as a model score. [Final evaluation](natural_results.md)
- Load one `NaturalPipeline` per server process using the FastAPI lifespan. Start with one Uvicorn worker and one in-flight inference slot on CPU. Run decoding, inference, and encoding off the event loop under that bound. Return a busy response when the slot is occupied. Release it only when the worker actually finishes, including after a client disconnect. Health routes should remain responsive during inference.
- Make fresh setup reproducible with an API dependency group and updated lockfile. Include FastAPI, Uvicorn, `python-multipart`, and an image decoder that supports header inspection. Document compatible PyTorch and torchvision installation and setup for `doagan_clean/` and `doa-gan_weights/single_channel.pkl`. Pin the upstream source to the revision and checkpoint hash already recorded in the protocol. Verify both before serving analysis. Provision these locally so requests never fetch models over the network. [Frozen protocol](natural_protocol.md)
- Enforce the request-body limit at the ASGI boundary before multipart parsing. Validate the decoded format and pixel budget rather than trusting the extension or MIME header. Close upload handles and discard temporary data on success and error. Accept uploaded bytes only. Log request ID, status, dimensions, pipeline version, and elapsed time.
- Bind to localhost for the first integration. Configure an explicit CORS origin allowlist for direct browser calls in development. A later Next.js server proxy can forward requests over a private connection. Public hosting would require a separate access-control and rate-limit decision. [FastAPI file uploads](https://fastapi.tiangolo.com/tutorial/request-files/) · [FastAPI CORS](https://fastapi.tiangolo.com/tutorial/cors/) · [FastAPI body limits](https://fastapi.tiangolo.com/advanced/request-body-limits/)

## Implementation and acceptance

1. Add an `api/` package with a FastAPI app factory, typed schemas, settings, image handling, and the routes above. Document setup, local launch, a sample upload request, and the contract for Next.js in the README.
2. Add API tests for health, readiness, capabilities, OpenAPI, upload validation, stable errors, byte and dimension limits, invalid mask IDs, busy handling, and mask encoding. Use a stub only in contract and failure-path tests to keep them deterministic. [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/) · [Testing lifespan](https://fastapi.tiangolo.com/advanced/testing-events/)
3. Start Uvicorn locally with the real checkpoint and use an HTTP client to upload at least **five distinct images**. Use two forged and two authentic JPEGs from CASIA2 calibration and one CoMoFoD dev PNG. Include a positive and negative fixture already confirmed by the locked pipeline if available. Choose fixtures from manifests and record their identifiers for reproducibility. Exercise WebP and orientation in separate format checks.
4. For those real HTTP requests, verify 200 responses, all three detector keys, all five fusion keys, and agreement with direct locked-pipeline predictions on the same decoded pixels. Decode returned masks and verify PNG format, binary values, native dimensions, and alignment after orientation. Record fixture, status, observed flags, rule results, and latency in a smoke-test report. Ground-truth labels provide context for the test. An API integration test succeeds when the service faithfully exposes the pipeline, even if the detector misses a forgery.
5. Verify that a fresh local setup can start, reach readiness, and serve a valid upload without a database or an external service. Leave Next.js implementation for the frontend phase.
