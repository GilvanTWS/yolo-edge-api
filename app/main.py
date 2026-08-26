import base64
import io
import os
import time
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from ultralytics import YOLO

# 1. Configuração da API e Modelo
app = FastAPI(title="YOLO Edge API")
MODEL_NAME = os.environ.get("MODEL_NAME", "yolov8n.pt")
model = YOLO(MODEL_NAME)

# 2. Schemas de Validação (Pydantic)
class PredictRequest(BaseModel):
    image_base64: str
    confidence: float = 0.25

class BatchPredictRequest(BaseModel):
    images_base64: List[str]
    confidence: float = 0.25

class Detection(BaseModel):
    label: str
    confidence: float
    bbox: List[float]

class PredictResponse(BaseModel):
    detections: List[Detection]
    inference_ms: float
    model_used: str
    image_width: int
    image_height: int

class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total_inference_ms: float

# 3. Função de Decodificação Exigida pelos Testes
def _decode_image(b64_str: str) -> np.ndarray:
    try:
        image_data = base64.b64decode(b64_str)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return np.array(image)
    except Exception as e:
        raise ValueError("String base64 inválida") from e

# 4. Endpoints
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME
    }

@app.get("/metrics")
def metrics():
    return {"metrics": "Endpoint de métricas ativado."}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start_time = time.time()
    
    try:
        img_array = _decode_image(req.image_base64)
    except ValueError:
        raise HTTPException(status_code=400, detail="Erro ao decodificar a imagem")

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
            
    return PredictResponse(
        detections=detections,
        inference_ms=(time.time() - start_time) * 1000.0,
        model_used=MODEL_NAME,
        image_width=w,
        image_height=h
    )

@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest):
    start_time = time.time()
    results = [predict(PredictRequest(image_base64=b64, confidence=req.confidence)) for b64 in req.images_base64]
    return BatchPredictResponse(
        results=results,
        total_inference_ms=(time.time() - start_time) * 1000.0
    )
