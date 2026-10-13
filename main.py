"""Skynetic AI inference service.

Wraps the Keras CNN trained in cnn.ipynb behind a versioned HTTP endpoint.
Only the platform API talks to this service (shared-secret header); it never
stores images or results — persistence, authorization and auditing live in
the platform API.

Run:  uvicorn main:app --host 0.0.0.0 --port 5001
"""
from __future__ import annotations

import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Callable, Protocol

import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("skynetic.ai")

# --- model metadata -------------------------------------------------------
MODEL_NAME = "skynetic-cnn"
MODEL_VERSION = os.environ.get("MODEL_VERSION", "model2")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "model2.h5"))
FRAMEWORK = "tensorflow-keras-2.17"

# Class order is the alphabetical folder order used by image_dataset_from_directory.
CLASS_LABELS = ["low_risk", "high_risk", "clear_skin"]
CLASS_NAMES_RO = ["Grad de risc scăzut", "Grad de risc crescut", "Piele curată"]

IMG_SIZE = (180, 180)
MAX_BYTES = 10 * 1024 * 1024
TOKEN_HEADER = "X-AI-Token"


class Predictor(Protocol):
    def predict(self, batch: np.ndarray) -> np.ndarray: ...


class KerasPredictor:
    """Loads model2.h5 once; thread-safe for inference."""

    def __init__(self, path: str = MODEL_PATH):
        from tensorflow.keras.models import load_model  # imported lazily: slow

        log.info("loading model %s", path)
        self.model = load_model(path, compile=False)

    def predict(self, batch: np.ndarray) -> np.ndarray:
        return self.model.predict(batch, verbose=0)


def preprocess(image_bytes: bytes) -> np.ndarray:
    """Exactly what training did: RGB, bilinear resize to 180×180, raw 0–255
    values (the model's first layer is Rescaling(1/255)). No enhancement."""
    image = Image.open(io.BytesIO(image_bytes))
    image.load()  # force decoding so truncated files fail here
    image = image.convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    return np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def create_app(predictor_factory: Callable[[], Predictor] | None = None, token: str | None = None) -> FastAPI:
    """Build the app. Tests pass a fake predictor and an explicit token."""
    factory = predictor_factory or KerasPredictor
    required_token = os.environ.get("AI_SERVICE_TOKEN", "") if token is None else token

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = None
        try:
            app.state.predictor = factory()
            log.info("model ready: %s/%s", MODEL_NAME, MODEL_VERSION)
        except Exception:  # keep serving /health so the failure is observable
            log.exception("model failed to load")
        yield

    app = FastAPI(title="Skynetic AI inference", version="1.0.0", lifespan=lifespan)

    def require_token(x_ai_token: str | None = Header(default=None, alias=TOKEN_HEADER)) -> None:
        if required_token and x_ai_token != required_token:
            raise HTTPException(status_code=401, detail="invalid service token")

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException):
        codes = {401: "unauthorized", 413: "payload_too_large", 422: "validation_failed", 503: "model_unavailable"}
        return _error(exc.status_code, codes.get(exc.status_code, "error"), str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        return _error(422, "validation_failed", "multipart field 'image' is required")

    @app.get("/health")
    def health(request: Request):
        loaded = request.app.state.predictor is not None
        return {
            "status": "ok" if loaded else "degraded",
            "model_loaded": loaded,
            "model": {"name": MODEL_NAME, "version": MODEL_VERSION, "framework": FRAMEWORK,
                      "input_size": list(IMG_SIZE), "class_labels": CLASS_LABELS},
        }

    @app.post("/v1/predict", dependencies=[Depends(require_token)])
    async def predict(request: Request, image: UploadFile = File(...)):
        predictor: Predictor | None = request.app.state.predictor
        if predictor is None:
            raise HTTPException(status_code=503, detail="model is not loaded")

        data = await image.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(status_code=413, detail="image exceeds 10 MB")
        if not data:
            return _error(400, "invalid_image", "empty upload")
        try:
            batch = preprocess(data)
        except (UnidentifiedImageError, OSError, ValueError):
            return _error(400, "invalid_image", "the file is not a decodable image")

        t0 = time.perf_counter()
        try:
            preds = predictor.predict(batch)
        except Exception:
            log.exception("inference failed")
            return _error(500, "inference_failed", "the model could not process the image")
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        scores = [float(s) for s in np.asarray(preds[0]).ravel().tolist()]
        if len(scores) != len(CLASS_LABELS):
            return _error(500, "inference_failed", "unexpected model output shape")
        class_id = int(np.argmax(scores))
        return {
            "model": {"name": MODEL_NAME, "version": MODEL_VERSION, "framework": FRAMEWORK},
            "predicted_class": class_id,
            "predicted_label": CLASS_LABELS[class_id],
            "predicted_name_ro": CLASS_NAMES_RO[class_id],
            "class_labels": CLASS_LABELS,
            # Raw softmax outputs. Not calibrated probabilities.
            "scores": scores,
            "inference_ms": elapsed_ms,
            "input_size": list(IMG_SIZE),
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
