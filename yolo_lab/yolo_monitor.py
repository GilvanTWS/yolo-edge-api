#!/usr/bin/env python3
"""Monitor da inferência YOLO na Raspberry Pi (Aula 7).

Executa o YOLO sobre um vídeo em loop (simulando um stream contínuo) e
expõe a métrica ``yolo_inference_time_seconds`` no formato Prometheus,
na porta 8000 por padrão, para ser coletada pelo Grafana Alloy.

Comportamento headless por padrão: não abre janela gráfica, então
funciona via SSH. Use ``--display`` para ver as detecções em um
monitor com sessão gráfica.
"""

import argparse
import time

import cv2
from prometheus_client import Gauge, start_http_server
from ultralytics import YOLO

INFERENCE_TIME = Gauge(
    "yolo_inference_time_seconds",
    "Tempo de inferência do YOLO em segundos",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=str, default="videos/transito.mp4")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Resolução de inferência (testar 640 e 320)")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--classes", type=int, nargs="+", default=[2],
                        help="Classes COCO a detectar (padrão: 2 = carro)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--display", action="store_true",
                        help="Abre janela com as detecções (requer X11)")
    return parser.parse_args()


def main():
    args = parse_args()

    start_http_server(args.port)
    print(f"[INFO] Servidor de métricas iniciado na porta {args.port}/metrics")

    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"[ERRO] Não foi possível abrir o vídeo: {args.video}")
    print(f"[INFO] Rodando YOLO com imgsz={args.imgsz}, conf={args.conf}, "
          f"classes={args.classes} sobre: {args.video}")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        start = time.time()
        results = model(frame, imgsz=args.imgsz, conf=args.conf,
                        classes=args.classes)
        end = time.time()

        inference_time = end - start
        INFERENCE_TIME.set(inference_time)
        print(f"\r[INFO] inferência = {inference_time * 1000:.1f} ms "
              f"| detecções = {len(results[0].boxes)}", end="", flush=True)

        if args.display:
            annotated = results[0].plot()
            cv2.imshow("YOLO Stream", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()