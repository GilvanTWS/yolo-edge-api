# YOLO Edge API

Object detection inference API using YOLO (Ultralytics), optimized to run on edge devices like the Raspberry Pi (ARM64 architecture).

> **Status: work in progress** — the project is not finished yet. Endpoints, routes, and behaviors may change as the course evolves.

## About

Serves a pretrained YOLO model (`yolov8n.pt` by default) over HTTP, receiving images in base64 and returning detections (labels, confidence, and bounding boxes).

- Inference stack: Ultralytics YOLO + PyTorch (CPU-only, no CUDA)
- API: FastAPI + Uvicorn
- Target: edge / ARM64 (Raspberry Pi), with multi-architecture Docker builds

## Features

- Single image inference (`/predict`)
- Batch inference (`/predict/batch`)
- Health endpoint (`/health`) and metrics endpoint (`/metrics`)
- Structured JSON event logs
- In-memory model caching
- CI/CD pipeline with lint, tests, ARM64 build, and mAP quality gate
- Automated edge deploy and rollback via `scripts/deploy.sh`

## Project structure

```
.
├── app/                # FastAPI app (main.py, model.py, schemas.py)
├── client/             # Command-line client to exercise the API
├── scripts/            # validate_model.py (quality gate) and deploy.sh
├── tests/              # Smoke, unit, and integration tests
├── models/             # Model weights (managed by DVC)
├── Dockerfile.api      # ARM64 API image
├── Dockerfile.client   # Client image
├── docker-compose.yml  # Local orchestration
└── .github/workflows/  # CI/CD pipeline
```

## Requirements

- Python 3.11+
- Docker + Docker Compose (to run in a container)
- A YOLO weights file in `models/` (e.g. `yolov8n.pt`)

## Running locally (without Docker)

```bash
# install dependencies
pip install -r app/requirements.txt

# make sure the model exists
mkdir -p models
wget -q -O models/yolov8n.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt

# start the API
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Running with Docker

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`. Interactive docs (Swagger) at `http://localhost:8000/docs`.

## Endpoints

| Method | Route            | Description                              |
|--------|------------------|------------------------------------------|
| GET    | `/health`        | API status and whether the model loaded  |
| GET    | `/metrics`       | Metrics endpoint                         |
| POST   | `/predict`       | Inference on a single image (base64)     |
| POST   | `/predict/batch` | Inference on multiple images (batch)     |

### Usage example with the client

```bash
pip install requests
python -m client.client --url http://localhost:8000 --image path/to/image.jpg --confidence 0.3
```

### `/predict` request example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "<image base64>", "confidence": 0.3}'
```

Response schema: `detections[]`, `inference_ms`, `model_used`, `image_width`, `image_height`.

## Tests

```bash
pip install pytest ruff
pytest tests/ -v --tb=short
ruff check app/
```

## Roadmap (next steps)

- [ ] Finalize the metrics endpoint
- [ ] Document integration with the real Raspberry Pi
- [ ] Add support for other YOLO models
- [ ] Implement a more robust watchdog/rollback

## License

MIT — see the [LICENSE](LICENSE) file.
