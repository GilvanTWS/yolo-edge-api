import base64
import io
import json
import os
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

os.environ.setdefault("MODEL_NAME", "yolov8n.pt")
from app.main import app, _decode_image

client = TestClient(app)
ASSETS = Path(__file__).parent / "assets"


class TestSmoke:
    def test_health_status_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_payload_structure(self):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_name" in data

    def test_metrics_endpoint_accessible(self):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")

    def test_metrics_exposes_prometheus_metrics(self):
        body = client.get("/metrics").text
        assert "yolo_api_requests_total" in body
        assert "yolo_api_inference_seconds" in body
        assert "yolo_model_loaded" in body

    def test_metrics_counts_requests(self):
        client.get("/health")
        body = client.get("/metrics").text
        assert 'yolo_api_requests_total{endpoint="/health"' in body
        assert "yolo_api_request_seconds_count" in body


class TestDecodeImage:
    def _make_b64_image(self, width=32, height=32, fmt="JPEG"):
        img = Image.new("RGB", (width, height), color=(128, 64, 192))
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode()

    def test_returns_numpy_array(self):
        result = _decode_image(self._make_b64_image())
        assert isinstance(result, np.ndarray)

    def test_correct_shape(self):
        result = _decode_image(self._make_b64_image(64, 48))
        assert result.shape == (48, 64, 3)

    def test_png_format(self):
        result = _decode_image(self._make_b64_image(fmt="PNG"))
        assert len(result.shape) == 3

    def test_invalid_base64_raises(self):
        with pytest.raises(Exception):
            _decode_image("dado_invalido_nao_e_base64")


class TestPredictEndpoint:
    @pytest.fixture
    def zid_b64(self):
        img_path = ASSETS / "zidane.jpg"
        return base64.b64encode(img_path.read_bytes()).decode()

    def test_predict_returns_200(self, zid_b64):
        resp = client.post("/predict", json={
            "image_base64": zid_b64,
            "confidence": 0.3,
        })
        assert resp.status_code == 200

    def test_predict_detects_at_least_one_object(self, zid_b64):
        data = client.post("/predict", json={
            "image_base64": zid_b64,
            "confidence": 0.3,
        }).json()
        assert len(data["detections"]) >= 1

    def test_predict_response_schema(self, zid_b64):
        data = client.post("/predict", json={
            "image_base64": zid_b64,
            "confidence": 0.3,
        }).json()
        assert "detections" in data
        assert "inference_ms" in data
        assert "model_used" in data
        assert "image_width" in data
        assert "image_height" in data
        assert data["inference_ms"] > 0

    def test_predict_detection_fields(self, zid_b64):
        data = client.post("/predict", json={
            "image_base64": zid_b64,
            "confidence": 0.3,
        }).json()
        for det in data["detections"]:
            assert isinstance(det["label"], str)
            assert 0.0 <= det["confidence"] <= 1.0
            assert len(det["bbox"]) == 4

    def test_predict_missing_input_returns_422(self):
        resp = client.post("/predict", json={
            "confidence": 0.3
        })
        assert resp.status_code == 422


class TestBatchEndpoint:
    @pytest.fixture
    def two_images_b64(self):
        img_path = ASSETS / "zidane.jpg"
        b64 = base64.b64encode(img_path.read_bytes()).decode()
        return [b64, b64]

    def test_batch_returns_correct_count(self, two_images_b64):
        data = client.post("/predict/batch", json={
            "images_base64": two_images_b64,
            "confidence": 0.3,
        }).json()
        assert len(data["results"]) == 2

    def test_batch_total_ms_is_positive(self, two_images_b64):
        data = client.post("/predict/batch", json={
            "images_base64": two_images_b64,
            "confidence": 0.3,
        }).json()
        assert data["total_inference_ms"] > 0
