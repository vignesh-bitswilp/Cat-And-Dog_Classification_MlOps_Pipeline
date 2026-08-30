"""
app/main.py
-----------
FastAPI inference service for the Cats vs Dogs classifier.

Endpoints:
  GET  /health    -> liveness/readiness probe (M2)
  POST /predict   -> accepts an image file, returns class label + probabilities (M2)
  GET  /metrics   -> Prometheus-format metrics: request count, latency (M5)

Also implements basic request/response logging (excluding raw image bytes /
any sensitive payload) and in-app request counters, per Assignment M5.
"""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from src.data_preprocessing import preprocess_image
from src.model import load_model, predict_from_array

# ---------------------------------------------------------------------------
# Logging setup (M5: request/response logging, excluding sensitive data)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("inference-service")

# ---------------------------------------------------------------------------
# Prometheus metrics (M5: request count and latency)
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "inference_requests_total", "Total number of inference requests", ["endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "inference_request_latency_seconds", "Request latency in seconds", ["endpoint"]
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.pt"

app = FastAPI(title="Cats vs Dogs Inference Service", version="1.0.0")

_model = None  # lazy-loaded singleton


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Model artifact not found at {MODEL_PATH}. Run `python -m src.train` first."
            )
        _model = load_model(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    return _model


@app.on_event("startup")
def _startup():
    # Attempt eager load so failures surface immediately, but don't crash
    # the whole service if the model isn't baked into the image yet.
    try:
        get_model()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Model not loaded at startup: %s", exc)


@app.get("/health")
def health():
    """Liveness/readiness probe used by Kubernetes and CD smoke tests."""
    start = time.time()
    model_ready = MODEL_PATH.exists()
    status = "ok" if model_ready else "degraded"
    REQUEST_COUNT.labels(endpoint="/health", status=status).inc()
    REQUEST_LATENCY.labels(endpoint="/health").observe(time.time() - start)
    logger.info("GET /health -> status=%s model_ready=%s", status, model_ready)
    return {"status": status, "model_ready": model_ready}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file and returns the predicted class label
    and per-class probabilities.
    """
    start = time.time()
    endpoint = "/predict"
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed: np.ndarray = preprocess_image(image)

        model = get_model()
        result = predict_from_array(model, processed)

        REQUEST_COUNT.labels(endpoint=endpoint, status="success").inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.time() - start)

        # Log metadata only -- never log raw image bytes or full payloads.
        logger.info(
            "POST /predict -> filename=%s content_type=%s label=%s latency_ms=%.1f",
            file.filename,
            file.content_type,
            result["label"],
            (time.time() - start) * 1000,
        )
        return result

    except Exception as exc:  # noqa: BLE001
        REQUEST_COUNT.labels(endpoint=endpoint, status="error").inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.time() - start)
        logger.error("POST /predict failed: %s", exc)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc


@app.get("/metrics")
def metrics():
    """Prometheus scrape endpoint (M5: basic monitoring)."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
