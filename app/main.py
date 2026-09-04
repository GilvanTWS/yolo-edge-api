import base64
import io
import json
import os
import time
import uuid

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from PIL import Image

from app import metrics
from app.model import get_model
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    Detection,
    PredictRequest,
    PredictResponse,
)

app = FastAPI(title="YOLO Edge API")
MODEL_NAME = os.environ.get("MODEL_NAME", "yolov8n.pt")

try:
    model = get_model(MODEL_NAME)
    _MODEL_LOADED = True
except Exception:
    model = None
    _MODEL_LOADED = False

metrics.MODEL_LOADED.set(1 if _MODEL_LOADED else 0)


@app.middleware("http")
async def instrument_requests(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        status = 500
        raise
    finally:
        metrics.REQUEST_SECONDS.labels(request.method, request.url.path).observe(
            time.perf_counter() - start
        )
    metrics.REQUESTS_TOTAL.labels(request.method, request.url.path, status).inc()
    return response


def log_event(event: str, level: str = "INFO", **kwargs):
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "level": level,
        "event": event,
        **kwargs,
    }
    print(json.dumps(record, ensure_ascii=False), flush=True)


def _decode_image(b64_str: str) -> np.ndarray:
    try:
        image_data = base64.b64decode(b64_str)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return np.array(image)
    except Exception as e:
        raise ValueError("String base64 inválida") from e


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": _MODEL_LOADED,
        "model_name": MODEL_NAME,
    }


@app.get("/metrics")
def metrics_endpoint():
    body, content_type = metrics.render_prometheus_metrics()
    return Response(content=body, media_type=content_type)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]

    log_event("predict_start",
              request_id=request_id,
              model=MODEL_NAME,
              confidence=req.confidence)

    try:
        if not req.image_base64:
            raise ValueError("missing_input")
        img_array = _decode_image(req.image_base64)
    except ValueError as e:
        log_event("predict_error", level="WARN", request_id=request_id, reason=str(e))
        raise HTTPException(status_code=400, detail="Erro ao decodificar a imagem")

    try:
        h, w, _ = img_array.shape
        results = model(img_array, conf=req.confidence, verbose=False)

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                detections.append(Detection(
                    label=model.names[int(box.cls)],
                    confidence=float(box.conf),
                    bbox=[x1, y1, x2, y2]
                ))

        inference_ms = (time.time() - start_time) * 1000.0
        metrics.INFERENCE_SECONDS.observe(inference_ms / 1000.0)

        log_event("predict_complete",
                  request_id=request_id,
                  model=MODEL_NAME,
                  detections=len(detections),
                  inference_ms=inference_ms,
                  image_size=f"{w}x{h}")

        return PredictResponse(
            detections=detections,
            inference_ms=inference_ms,
            model_used=MODEL_NAME,
            image_width=w,
            image_height=h
        )
    except Exception as e:
        log_event("predict_error", level="ERROR", request_id=request_id, reason=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest):
    start_time = time.time()
    results = [predict(PredictRequest(image_base64=b64, confidence=req.confidence)) for b64 in req.images_base64]
    return BatchPredictResponse(
        results=results,
        total_inference_ms=(time.time() - start_time) * 1000.0
    )
