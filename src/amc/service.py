"""HTTP entry point for batches of already preprocessed, per-signal IQ.

This service deliberately does not crop detector boxes, frequency-shift, filter,
or resample IQ. Run one worker: ``uvicorn amc.service:app --workers 1``.
"""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Callable

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import numpy as np
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from .prepared_inference import PreparedSignal, SignalMetadata


# Reuse uvicorn's configured INFO handler for normal server runs without
# changing root logging or a future detector application's logger configuration.
LOGGER = logging.getLogger("uvicorn.error.amc")
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "models" / "amc20_v1.pt"
METADATA_MAX_BYTES = 16 * 1024
PART_NAME = re.compile(r"^(iq|metadata)_(0|[1-9][0-9]*)$")


def _positive_int(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if parsed < 1 or (not isinstance(value, str) and parsed != value):
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


def _score_threshold(value):
    if isinstance(value, bool):
        raise ValueError("score_threshold must be a finite number in [0,1].")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("score_threshold must be a finite number in [0,1].") from exc
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise ValueError("score_threshold must be a finite number in [0,1].")
    return parsed


class RequestSizeLimitMiddleware:
    """Bound the actual incoming body, including requests without Content-Length.

    Buffer to a capped spool before multipart parsing. This avoids leaving
    multipart temporary files open if a stream exceeds its limit mid-upload.
    """

    def __init__(self, app, max_bytes):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if (scope["type"] != "http" or scope.get("method") != "POST"
                or scope.get("path") != "/v1/amc/infer"):
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                declared_length = int(declared)
                if declared_length < 0:
                    raise ValueError
            except ValueError:
                await JSONResponse({"detail": "Invalid Content-Length."}, status_code=400)(scope, receive, send)
                return
            if declared_length > self.max_bytes:
                await JSONResponse({"detail": "Request body exceeds the byte limit."}, status_code=413)(scope, receive, send)
                return
        with tempfile.SpooledTemporaryFile(max_size=1024 * 1024, mode="w+b") as spool:
            total = 0
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                total += len(chunk)
                if total > self.max_bytes:
                    await JSONResponse({"detail": "Request body exceeds the byte limit."}, status_code=413)(scope, receive, send)
                    return
                await run_in_threadpool(spool.write, chunk)
                if not message.get("more_body", False):
                    break
            await run_in_threadpool(spool.seek, 0)
            remaining = total

            async def replay():
                nonlocal remaining
                if remaining == 0:
                    return {"type": "http.request", "body": b"", "more_body": False}
                chunk = await run_in_threadpool(spool.read, min(64 * 1024, remaining))
                remaining -= len(chunk)
                return {"type": "http.request", "body": chunk, "more_body": remaining > 0}

            await self.app(scope, replay, send)


class AMCService:
    """One resident classifier; model initialization and execution are serialized."""

    def __init__(self, checkpoint, device, window_batch_size, classifier_factory=None,
                 score_threshold=.5):
        self.checkpoint = Path(checkpoint)
        self.device = device
        self.window_batch_size = window_batch_size
        self.classifier_factory = classifier_factory
        self.score_threshold = score_threshold
        self._classifier = None
        self._lock = threading.Lock()

    def _load_locked(self):
        if self._classifier is None:
            if self.classifier_factory is None:
                from .prepared_inference import PreparedIQBatchClassifier
                factory = PreparedIQBatchClassifier
            else:
                factory = self.classifier_factory
            classifier = factory(
                self.checkpoint, device=self.device,
                window_batch_size=self.window_batch_size, max_windows=5,
                score_threshold=self.score_threshold,
            )
            classifier.warmup()
            self._classifier = classifier
        return self._classifier

    def ready(self):
        with self._lock:
            classifier = self._load_locked()
            return {"status": "ready", "device": str(classifier.device), "classes": list(classifier.classes),
                    "unknown_threshold": self.score_threshold}

    def predict(self, frame_id, signals):
        with self._lock:
            classifier = self._load_locked()
            response, diagnostics = classifier.predict_frame(frame_id, signals)
            # Keep per-signal winning score/reason/timing in normal logs. Full
            # Class score vectors are available at DEBUG without growing every INFO
            # request record proportionally to classes * signals.
            summary = dict(diagnostics)
            if "signals" in summary:
                summary["signals"] = [{k: v for k, v in item.items() if k != "class_scores"}
                                      for item in summary["signals"]]
            LOGGER.info("AMC request completed frame_id=%r signals=%d diagnostics=%s", frame_id, len(signals), summary)
            LOGGER.debug("AMC complete score diagnostics=%s", diagnostics)
            return response


class _ClosingMultipartParser(MultiPartParser):
    """Also clean up temporary files on malformed/truncated multipart streams."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.completed = False

    def on_end(self):
        self.completed = True
        super().on_end()

    async def parse(self):
        try:
            form = await super().parse()
            if not self.completed:
                raise MultiPartException("Incomplete multipart body.")
            return form
        except BaseException:
            # Starlette closes these for MultiPartException only; malformed
            # parser syntax and cancellation must close partially written files too.
            for file in self._files_to_close_on_error:
                file.close()
            raise


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}.")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON number: {value}.")


def _decode_iq(raw, metadata):
    expected = metadata.sample_count * 2 * np.dtype("<f4").itemsize
    if len(raw) != expected:
        raise ValueError(f"{metadata.signal_id}: IQ byte count does not match sample_count (expected {expected}).")
    iq = np.frombuffer(raw, dtype="<f4").reshape(-1, 2)
    if not np.isfinite(iq).all():
        raise ValueError(f"{metadata.signal_id}: IQ must contain only finite float32 values.")
    return PreparedSignal(metadata=metadata, iq=iq)


async def _parse_signals(request, max_signals):
    if request.headers.get("content-encoding", "identity").lower() != "identity":
        raise HTTPException(400, "Content-Encoding is not supported.")
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type != "multipart/form-data":
        raise HTTPException(400, "Use multipart/form-data with frame_id and paired iq_N/metadata_N parts.")
    try:
        form = await _ClosingMultipartParser(
            request.headers, request.stream(),
            max_files=2 * max_signals + 2,
            max_fields=2 * max_signals + 4,
            max_part_size=METADATA_MAX_BYTES,
        ).parse()
    except MultiPartException as exc:
        detail = str(exc)
        oversized = "Too many" in detail or "exceeded maximum size" in detail
        raise HTTPException(413 if oversized else 400, detail) from exc
    except Exception as exc:
        LOGGER.warning("Malformed multipart request: %s", exc)
        raise HTTPException(400, "Malformed multipart body.") from exc
    try:
        parts = {}
        for key, value in form.multi_items():
            if key in parts:
                raise HTTPException(400, f"Duplicate multipart field: {key}.")
            parts[key] = value
        frame_id = parts.pop("frame_id", None)
        if not isinstance(frame_id, str) or not frame_id.strip() or len(frame_id) > 128:
            raise HTTPException(400, "frame_id must be a nonempty string of at most 128 characters.")
        iq_parts, metadata_parts = {}, {}
        for key, value in parts.items():
            match = PART_NAME.fullmatch(key)
            if match is None:
                raise HTTPException(400, f"Unexpected multipart field: {key}.")
            if len(match.group(2)) > 8:
                raise HTTPException(400, "Signal part index is too large.")
            kind, index = match.group(1), int(match.group(2))
            (iq_parts if kind == "iq" else metadata_parts)[index] = value
        if max(len(iq_parts), len(metadata_parts)) > max_signals:
            raise HTTPException(413, f"At most {max_signals} signals are allowed per request.")
        if set(iq_parts) != set(metadata_parts):
            raise HTTPException(400, "Each iq_N part must have exactly one metadata_N part.")
        if set(iq_parts) != set(range(len(iq_parts))):
            raise HTTPException(400, "Signal part indices must be contiguous and start at 0.")
        parsed_metadata, identifiers = [], set()
        for index in range(len(iq_parts)):
            text = metadata_parts[index]
            if not isinstance(text, str):
                raise HTTPException(400, f"metadata_{index} must be a JSON text field, not a file.")
            if len(text.encode("utf-8")) > METADATA_MAX_BYTES:
                raise HTTPException(413, f"metadata_{index} exceeds the metadata byte limit.")
            try:
                metadata = SignalMetadata.from_dict(json.loads(
                    text, parse_constant=_invalid_constant, object_pairs_hook=_json_object,
                ))
            except (ValueError, TypeError, KeyError, RecursionError) as exc:
                raise HTTPException(400, f"metadata_{index}: {exc}") from exc
            if metadata.signal_id in identifiers:
                raise HTTPException(400, f"Duplicate signal_id: {metadata.signal_id}.")
            identifiers.add(metadata.signal_id)
            parsed_metadata.append(metadata)
            if not isinstance(iq_parts[index], UploadFile):
                raise HTTPException(400, f"iq_{index} must be a binary file part.")
        signals = []
        for index, metadata in enumerate(parsed_metadata):
            raw = await iq_parts[index].read()
            try:
                signals.append(await run_in_threadpool(_decode_iq, raw, metadata))
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        return frame_id, signals
    finally:
        await form.close()


def create_amc_router(service, max_signals):
    """Router can later be mounted into the detector application unchanged."""
    router = APIRouter(prefix="/v1/amc", tags=["modulation classification"])

    @router.get("/ready")
    async def ready():
        try:
            return await run_in_threadpool(service.ready)
        except Exception:
            LOGGER.exception("AMC classifier initialization or warmup failed.")
            raise HTTPException(503, "AMC classifier is not ready.")

    @router.post(
        "/infer",
        summary="Classify a batch of already preprocessed single-signal IQ records",
        description=("Multipart frame_id text plus paired iq_0 binary / metadata_0 JSON text, "
                     "iq_1 / metadata_1, etc. Indices start at zero and are contiguous. "
                     "IQ is little-endian float32 in I0,Q0,I1,Q1 order. No detector JSON is required. "
                     "Returns unknown when the maximum averaged window score is below the configured "
                     "threshold (default 0.5); insufficient-length or zero-power IQ returns null. "
                     "Scores are not included in the public response."),
        openapi_extra={"requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
            "type": "object", "required": ["frame_id"], "properties": {
                "frame_id": {"type": "string", "example": "frame_0001"},
                "iq_0": {"type": "string", "format": "binary"},
                "metadata_0": {"type": "string", "example": json.dumps({
                    "signal_id": "signal_0", "dtype": "float32", "sample_rate_hz": 100000,
                    "estimated_bandwidth_hz": 25000, "sample_count": 12000,
                })},
            }, "additionalProperties": True,
        }}}}},
    )
    async def infer(request: Request):
        frame_id, signals = await _parse_signals(request, max_signals)
        if not signals:
            return {"frame_id": frame_id, "results": []}
        try:
            return await run_in_threadpool(service.predict, frame_id, signals)
        except Exception:
            LOGGER.exception("AMC inference failed frame_id=%r signals=%d", frame_id, len(signals))
            raise HTTPException(503, "AMC inference failed; no classification results were returned.")

    return router


def install_amc(application, *, checkpoint=None, device=None, window_batch_size=None,
                max_signals=None, max_request_bytes=None, score_threshold=None,
                classifier_factory: Callable | None = None):
    """Add AMC to an existing FastAPI app without changing its detector routes.

    Configuration and body limits are shared by standalone and integrated runs.
    The size middleware applies only to POST /v1/amc/infer.
    """
    checkpoint = checkpoint if checkpoint is not None else os.environ.get("AMC_CHECKPOINT", str(DEFAULT_CHECKPOINT))
    device = device if device is not None else os.environ.get("AMC_DEVICE", "auto")
    score_threshold = _score_threshold(
        score_threshold if score_threshold is not None else os.environ.get("AMC_SCORE_THRESHOLD", "0.5"),
    )
    window_batch_size = _positive_int(
        window_batch_size if window_batch_size is not None else os.environ.get("AMC_WINDOW_BATCH_SIZE", "64"),
        "window_batch_size",
    )
    max_signals = _positive_int(
        max_signals if max_signals is not None else os.environ.get("AMC_MAX_SIGNALS", "64"), "max_signals",
    )
    max_request_bytes = _positive_int(
        max_request_bytes if max_request_bytes is not None else os.environ.get("AMC_MAX_REQUEST_BYTES", str(256 * 1024 * 1024)),
        "max_request_bytes",
    )
    service = AMCService(checkpoint, device, window_batch_size, classifier_factory, score_threshold)
    application.state.amc_service = service
    application.add_middleware(RequestSizeLimitMiddleware, max_bytes=max_request_bytes)
    application.include_router(create_amc_router(service, max_signals))
    return service


def create_app(**amc_options):
    """Build without loading a checkpoint; first ready/infer call loads once."""
    application = FastAPI(title="AMC single-signal batch classification", version="0.1.1")

    @application.get("/health")
    async def health():
        return {"status": "ok"}

    install_amc(application, **amc_options)
    return application


app = create_app()
