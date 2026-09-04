from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

REQUESTS_TOTAL = Counter(
    "yolo_api_requests_total",
    "Total de requisições recebidas pela API",
    ["method", "endpoint", "status"],
)

REQUEST_SECONDS = Histogram(
    "yolo_api_request_seconds",
    "Latência total das requisições em segundos",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

INFERENCE_SECONDS = Histogram(
    "yolo_api_inference_seconds",
    "Tempo de inferência do modelo YOLO em segundos",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

MODEL_LOADED = Gauge(
    "yolo_model_loaded",
    "Indica se o modelo YOLO foi carregado (1 = sim, 0 = não)",
)


def render_prometheus_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST