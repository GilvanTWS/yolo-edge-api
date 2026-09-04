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
- Health endpoint (`/health`) and Prometheus metrics endpoint (`/metrics`)
- Structured JSON event logs
- Observability integration: YOLO inference + hardware metrics scraped by Grafana Alloy → Grafana Cloud (Aula 7)
- In-memory model caching
- CI/CD pipeline with lint, tests, ARM64 build, and mAP quality gate
- Automated edge deploy and rollback via `scripts/deploy.sh`

## Project structure

```
.
├── app/                # FastAPI app (main.py, model.py, schemas.py, metrics.py)
├── client/             # Command-line client to exercise the API
├── scripts/            # validate_model.py (quality gate) and deploy.sh
├── yolo_lab/           # Aula 7: standalone YOLO monitoring script (prometheus-client)
├── deploy/             # Aula 7: Grafana Alloy config template (observability)
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

## Observability (Edge AI monitoring)

Stack: **Raspberry Pi → Node Exporter + YOLO inference → Grafana Alloy → Prometheus (Grafana Cloud) → Dashboards**

### What is instrumented in this repo

| Target | Endpoint | Metrics |
|--------|----------|---------|
| API (FastAPI) | `GET /metrics` (port 8000) | `yolo_api_requests_total`, `yolo_api_request_seconds`, `yolo_api_inference_seconds`, `yolo_model_loaded` + process metrics |
| Stream server (Flask) | `GET /metrics` (port 5000) | `yolo_inference_time_seconds`, `yolo_stream_fps`, `yolo_stream_detections`, `yolo_stream_frames_total` |
| YOLO lab script | `GET /metrics` (port 8000) | `yolo_inference_time_seconds` (single gauge) |

### Manual steps (Raspberry Pi + Grafana Cloud)

These require interactive/cloud access and cannot be automated from this repo:

1. **Grafana Cloud** — create an account, a *Stack*, and a Prometheus *Access Policy* token with `metrics:write`. Save the Prometheus *Details* (User/ID) and the *remote write* URL.
2. **On the Raspberry Pi**, install and start Node Exporter:
   ```bash
   sudo apt install prometheus-node-exporter -y
   curl http://localhost:9100/metrics
   ```
3. **Install Grafana Alloy** (official Grafana apt repo, see Aula 7) and copy the ready template:
   ```bash
   sudo cp deploy/observability/config.alloy /etc/alloy/config.alloy
   sudo nano /etc/alloy/config.alloy   # fill in URL, username, token
   sudo systemctl restart alloy
   systemctl status alloy
   ```
4. **Run the YOLO monitoring lab** (headless-safe, single gauge):
   ```bash
   cd yolo_lab
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   mkdir -p videos            # put the course video (cars) at videos/transito.mp4
   python yolo_monitor.py --imgsz 640
   curl http://localhost:8000/metrics
   ```
   Change `--imgsz` between `640` and `320` to compare inference time, CPU, memory, and temperature.

   > Tip: if the FastAPI service already occupies port `8000` on the Pi, run with
   > `python yolo_monitor.py --port 8001` and point the Alloy `yolo_metrics` scrape
   > target to `localhost:8001` instead.

5. **Dashboards & alerts** (Grafana Cloud UI) — queries from Aula 7:
   - CPU: `node_load1`
   - Memory (%): `(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100`
   - Temperature (°C): `max(node_hwmon_temp_celsius{chip="thermal_thermal_zone0"})`
   - Inference: `yolo_inference_time_seconds`
   - Alert: `Uso de memória RAM > 80% → email` (contact point `Administrador`)

### Validation points (Aula 7)

- Node Exporter exposing metrics on `:9100`; Alloy `active (running)`.
- `up` returns `1` in *Explore*; `yolo_inference_time_seconds` present in Grafana Cloud.
- Changing `IMG_SIZE` produces visible variation in the inference time panel.
- Memory alert rule active (evaluates every ~1 min).

## Tests

```bash
pip install pytest ruff
pytest tests/ -v --tb=short
ruff check app/
```

## Roadmap (next steps)

- [x] Finalize the metrics endpoint (Prometheus format via `prometheus-client`)
- [ ] Complete the Aula 7 hands-on on the Raspberry Pi (dashboards + alerts)
- [ ] Add support for other YOLO models

## License

MIT — see the [LICENSE](LICENSE) file.
