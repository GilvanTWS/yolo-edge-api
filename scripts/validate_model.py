"""
scripts/validate_model.py
Portão de Qualidade do Modelo (Model Quality Gate).
Verifica se a precisão do YOLOv8 atende ao limite mínimo antes do deploy.
"""
import sys
from pathlib import Path
from ultralytics import YOLO

def validate():
    print("Iniciando validação de qualidade estatística do modelo...")
    model_path = Path("models/yolov8n.pt")
    
    if not model_path.exists():
        print(f"Erro: Modelo não encontrado em {model_path}")
        sys.exit(1)
        
    try:
        # Carrega o modelo de forma local
        model = YOLO(model_path)
        
        # Executa validação rápida usando o dataset reduzido coco8
        print("Rodando validação no dataset coco8...")
        metrics = model.val(data="coco8.yaml", imgsz=640, plots=False)
        
        # Extrai o mAP@0.5 (Mean Average Precision)
        map50 = metrics.results_dict.get("metrics/mAP50(B)", 0.0)
        print(f"Métrica mAP@0.5 obtida: {map50:.4f}")
        
        LIMIAR_MINIMO = 0.50
        if map50 < LIMIAR_MINIMO:
            print(f"❌ REPROVADO: Precisão ({map50:.4f}) abaixo do limiar mínimo de {LIMIAR_MINIMO}!")
            sys.exit(1)
            
        print("✅ APROVADO: O modelo passou com sucesso pelo Portão de Qualidade!")
        sys.exit(0)
        
    except Exception as e:
        print(f"Erro crítico durante o portão de qualidade: {e}")
        sys.exit(1)

if __name__ == "__main__":
    validate()
