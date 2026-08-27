"""app/schemas.py
Modelos Pydantic (contratos de I/O) da YOLO Inference API.
"""
from typing import List

from pydantic import BaseModel


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
