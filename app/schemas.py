from pydantic import BaseModel


class PredictRequest(BaseModel):
    image_base64: str
    confidence: float = 0.25


class BatchPredictRequest(BaseModel):
    images_base64: list[str]
    confidence: float = 0.25


class Detection(BaseModel):
    label: str
    confidence: float
    bbox: list[float]


class PredictResponse(BaseModel):
    detections: list[Detection]
    inference_ms: float
    model_used: str
    image_width: int
    image_height: int


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    total_inference_ms: float
