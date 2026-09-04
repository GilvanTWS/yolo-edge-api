from prometheus_client import Counter, Gauge

INFERENCE_TIME = Gauge(
    "yolo_inference_time_seconds",
    "Tempo de inferência do YOLO em segundos",
)

STREAM_FPS = Gauge(
    "yolo_stream_fps",
    "Frames processados por segundo pelo servidor de stream",
)

DETECTIONS = Gauge(
    "yolo_stream_detections",
    "Detecções presentes no último frame processado",
)

FRAMES_TOTAL = Counter(
    "yolo_stream_frames_total",
    "Total de frames processados pelo servidor de stream",
)