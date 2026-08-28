# YOLO Edge API

API de inferência de detecção de objetos com YOLO (Ultralytics), otimizada para rodar em dispositivos de borda (edge) como Raspberry Pi (arquitetura ARM64).

> **Status: em desenvolvimento** — o projeto ainda não está finalizado. Endpoints, rotas e comportamentos podem mudar enquanto a disciplina evolui.

## Sobre o projeto

Serve um modelo YOLO pré-treinado (`yolov8n.pt` por padrão) via HTTP, recebendo imagens em base64 e devolvendo as detecções (rótulos, confiança e caixas delimitadoras).

- Stack de inferência: Ultralytics YOLO + PyTorch (CPU-only, sem CUDA)
- API: FastAPI + Uvicorn
- Alvo: edge / ARM64 (Raspberry Pi), com build Docker multi-arquitetura

## Funcionalidades

- Inferência de imagem única (`/predict`)
- Inferência em lote (`/predict/batch`)
- Endpoint de saúde (`/health`) e de métricas (`/metrics`)
- Logs de eventos estruturados em JSON
- Cache de modelos em memória
- Piplina de CI/CD com lint, testes, build ARM64 e quality gate de mAP
- Deploy e rollback automatizado no edge via `scripts/deploy.sh`

## Estrutura do projeto

```
.
├── app/                # API FastAPI (main.py, model.py, schemas.py)
├── client/             # Cliente de linha de comando p/ testar a API
├── scripts/            # validate_model.py (quality gate) e deploy.sh
├── tests/              # Smoke, unit e integration tests
├── models/             # Pesos dos modelos (gerenciados por DVC)
├── Dockerfile.api      # Imagem da API em ARM64
├── Dockerfile.client   # Imagem do cliente
├── docker-compose.yml  # Orquestração local
└── .github/workflows/  # Pipeline de CI/CD
```

## Requisitos

- Python 3.11+
- Docker + Docker Compose (para execução via container)
- Um arquivo de pesos YOLO em `models/` (ex.: `yolov8n.pt`)

## Como executar local (sem Docker)

```bash
# instala as dependências
pip install -r app/requirements.txt

# garante que o modelo existe
mkdir -p models
wget -q -O models/yolov8n.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt

# sobe a API
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Como executar com Docker

```bash
docker-compose up --build
```

A API ficará disponível em `http://localhost:8000`. Documentação interativa (Swagger) em `http://localhost:8000/docs`.

## Endpoints

| Método | Rota             | Descrição                                   |
|--------|------------------|---------------------------------------------|
| GET    | `/health`        | Status da API e se o modelo foi carregado   |
| GET    | `/metrics`       | Endpoint de métricas                        |
| POST   | `/predict`       | Inferência em uma única imagem (base64)     |
| POST   | `/predict/batch` | Inferência em várias imagens (lote)         |

### Exemplo de uso com o cliente

```bash
pip install requests
python -m client.client --url http://localhost:8000 --image caminho/imagem.jpg --confidence 0.3
```

### Exemplo de requisição `/predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "<base64 da imagem>", "confidence": 0.3}'
```

Resposta (schema): `detections[]`, `inference_ms`, `model_used`, `image_width`, `image_height`.

## Testes

```bash
pip install pytest ruff
pytest tests/ -v --tb=short
ruff check app/
```

## Roadmap (próximos passos)

- [ ] Finalizar endpoint de métricas
- [ ] Documentar integração com o Raspberry Pi real
- [ ] Adicionar suporte a outros modelos YOLO
- [ ] Implementar watchdog/rollback refinado

## Licença

MIT — veja o arquivo [LICENSE](LICENSE).
