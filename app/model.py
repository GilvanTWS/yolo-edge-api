"""app/model.py
Carregamento e cache de modelos YOLO em memória.
Resolve os pesos sempre em MODELS_DIR = Path("/app/models") — caminho absoluto
usado dentro do container Docker (mapeado pelo docker-compose.yml).
"""
import os
from functools import lru_cache
from pathlib import Path

from ultralytics import YOLO

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/app/models"))


class ModelStore:
    """Cache de modelos YOLO por nome de arquivo."""

    def __init__(self) -> None:
        self._models: dict[str, YOLO] = {}

    def load(self, model_name: str) -> YOLO:
        """Carrega (e cachead) o modelo pelo nome do arquivo de pesos."""
        if model_name not in self._models:
            model_path = MODELS_DIR / model_name
            if not model_path.exists():
                raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
            self._models[model_name] = YOLO(str(model_path))
        return self._models[model_name]


@lru_cache(maxsize=4)
def get_model(model_name: str) -> YOLO:
    """Retorna o modelo YOLO para o nome informado, com cache em memória."""
    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    return YOLO(str(model_path))
