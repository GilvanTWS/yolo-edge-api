import argparse
import base64
import sys
from pathlib import Path

import requests


def load_config():
    parser = argparse.ArgumentParser(description="Cliente da YOLO Inference API")
    parser.add_argument("--url", default="http://localhost:8000",
                        help="URL base da API")
    parser.add_argument("--image", required=True, help="Caminho da imagem")
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="Limiar de confiança")
    return parser.parse_args()


def encode_image(image_path: str) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode()


def main():
    args = load_config()
    b64 = encode_image(args.image)
    resp = requests.post(
        f"{args.url}/predict",
        json={"image_base64": b64, "confidence": args.confidence},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    print(f"Modelo: {data['model_used']} | Latência: {data['inference_ms']:.1f} ms")
    print(f"Tamanho: {data['image_width']}x{data['image_height']}")
    print(f"Detecções ({len(data['detections'])}):")
    for det in data["detections"]:
        print(f"  - {det['label']}: {det['confidence']:.3f} {det['bbox']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        sys.exit(1)
