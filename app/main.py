#!/usr/bin/env python3
"""
app.py — Inferencia YOLO para Visao Computacional Embarcada
Modos:
--mode classify  → Endpoint 1: lista classes detectadas (stdout)
--mode detect    → Endpoint 2: desenha bboxes e salva imagem anotada
Saidas:
/app/output/results.txt  → classes e confiancas
/app/output/annotated.jpg → imagem com bounding boxes (modo detect)
"""
import argparse
import os
import sys
from pathlib import Path
import cv2
from ultralytics import YOLO

OUTPUT_DIR = Path("/app/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_model() -> YOLO:
    """Carrega o modelo a partir da variavel de ambiente MODEL_NAME."""
    model_path = os.environ.get("MODEL_NAME", "yolov8n.pt")
    print(f"[INFO] Carregando modelo: {model_path}")
    return YOLO(model_path)

# ───────────────────────────────────────────────────────
# ENDPOINT 1 — Classificacao
# ───────────────────────────────────────────────────────
def endpoint_classify(model: YOLO, image_path: str) -> None:
    print(f"[INFO] Modo: classificacao | Imagem: {image_path}")
    results = model(image_path, verbose=False)
    txt_path = OUTPUT_DIR / "results.txt"
    with open(txt_path, "w") as f:
        f.write(f"Imagem: {image_path}\n")
        f.write(f"Modelo: {os.environ.get('MODEL_NAME', 'yolov8n.pt')}\n\n")
        f.write("Classes detectadas:\n")
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                confidence = float(box.conf)
                line = f"  - {class_name}: {confidence:.2%}"
                print(line)
                f.write(line + "\n")
    print(f"[INFO] Resultados salvos em: {txt_path}")

# ───────────────────────────────────────────────────────
# ENDPOINT 2 — Deteccao com Bounding Boxes
# ───────────────────────────────────────────────────────
def endpoint_detect(model: YOLO, image_path: str) -> None:
    print(f"[INFO] Modo: deteccao | Imagem: {image_path}")
    results = model(image_path, verbose=False)
    # Carrega a imagem original via OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERRO] Nao foi possivel abrir: {image_path}", file=sys.stderr)
        sys.exit(1)

    txt_path = OUTPUT_DIR / "results.txt"
    annotated_path = OUTPUT_DIR / "annotated.jpg"

    with open(txt_path, "w") as f:
        f.write(f"Imagem: {image_path}\n")
        f.write(f"Modelo: {os.environ.get('MODEL_NAME', 'yolov8n.pt')}\n\n")
        f.write("Deteccoes:\n")
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                confidence = float(box.conf)
                
                # O PONTO DA CORREÇÃO: Usando  antes do .tolist()
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Registrar no .txt
                line = f"  - {class_name}: {confidence:.2%} @ [{x1},{y1},{x2},{y2}]"
                print(line)
                f.write(line + "\n")

                # Desenhar bounding box na imagem
                color = (0, 255, 0)   # verde (BGR)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {confidence:.0%}"
                cv2.putText(img, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Salvar imagem anotada
    cv2.imwrite(str(annotated_path), img)
    print(f"[INFO] Imagem anotada salva em: {annotated_path}")
    print(f"[INFO] Resultados salvos em:   {txt_path}")

# ───────────────────────────────────────────────────────
# PONTO DE ENTRADA
# ───────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Inferencia YOLO")
    parser.add_argument("--mode", choices=["classify", "detect"],
                        default="detect", help="Modo de operacao")
    parser.add_argument("--input", required=True,
                        help="Caminho para a imagem de entrada")
    args = parser.parse_args()

    model = load_model()
    if args.mode == "classify":
        endpoint_classify(model, args.input)
    else:
        endpoint_detect(model, args.input)

if __name__ == "__main__":
    main()
