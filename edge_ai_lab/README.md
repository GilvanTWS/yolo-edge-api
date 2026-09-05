# Aula 8 — Otimização de Modelos para Edge AI

Hands-on proposto na Aula 8: otimizar um modelo de classificação de imagens
(MobileNetV2) e avaliar o impacto das técnicas de otimização em dispositivo
embarcado (Raspberry Pi).

## Pipeline

O fluxo completo, rodado dentro de um contêiner Docker:

1. **Baseline** — inferência com MobileNetV2 original (TensorFlow, FP32)
2. **Conversão TFLite FP32** — `modelo/mobilenet_fp32.tflite`
3. **Quantização pós-treinamento (INT8)** — `modelo/mobilenet_int8.tflite`
4. **Inferência com modelos otimizados** (TFLite FP32 e INT8)
5. **Análise comparativa** de tempo de inferência das três versões

> A atividade usa **imagens baixadas por URL** (objetos do cotidiano: celular,
> xícara, livro etc.). Não usa câmera.

## Estrutura

```
edge_ai_lab/
├─ Dockerfile
├─ scripts/
│  ├─ inferencia_url.py        # Baseline (MobileNetV2 + TensorFlow)
│  ├─ conversao_tflite.py      # Converte para TFLite FP32
│  ├─ quantizacao_tflite.py    # Quantização pós-treinamento (INT8)
│  └─ inferencia_tflite_url.py # Inferência com modelo TFLite
└─ modelo/                     # Modelos convertidos (não versionado)
```

## Passo a passo (na Raspberry Pi)

### 1. Criar e construir o ambiente Docker

```bash
cd edge_ai_lab
docker build -t edgeai_lab .
```

### 2. Inicializar o contêiner

```bash
docker run -it -v $(pwd):/app edgeai_lab
```

A partir daqui todos os comandos rodam **dentro** do contêiner.

### 3. Inferência com o modelo TensorFlow (baseline)

```bash
python scripts/inferencia_url.py
# Digite a URL da imagem: <URL de uma imagem com celular, xícara, livro...>
```

Anote o tempo e as classes preditas.

### 4. Conversão para TFLite FP32

```bash
python scripts/conversao_tflite.py
# Modelo FP32 salvo em modelo/mobilenet_fp32.tflite
```

### 5. Quantização pós-treinamento (INT8)

```bash
python scripts/quantizacao_tflite.py
# Modelo INT8 salvo em modelo/mobilenet_int8.tflite
```

### 6. Inferência com os modelos otimizados

Use a **mesma URL** do passo 3 para comparar de forma justa:

```bash
python scripts/inferencia_tflite_url.py modelo/mobilenet_fp32.tflite
python scripts/inferencia_tflite_url.py modelo/mobilenet_int8.tflite
```

### 7. Análise comparativa

Preencha a tabela com os tempos medidos:

| Modelo                  | Tempo de Inferência (s) | Variação (%) | Observações |
|-------------------------|-------------------------|--------------|-------------|
| TensorFlow (Base)       |                         |              |             |
| TFLite FP32             |                         |              |             |
| TFLite INT8             |                         |              |             |

Perguntas para refletir:

- Qual versão apresentou menor tempo de inferência?
- A quantização proporcionou ganhos significativos?
- Houve impacto perceptível na qualidade da classificação?
- Qual configuração é mais adequada para aplicações embarcadas?

## Observações

- **Python/aula original:** Python 3.10, TensorFlow 2.13 (compatível com ARM64 da
  Raspberry Pi). A imagem `python:3.10-slim-bullseye` garante suporte à
  arquitetura ARM.
- **Scripts da atividade (referência):**
  https://drive.google.com/drive/folders/1zt4IbbhgkOusIP2P8y8bXYmqkAIJxXL5?usp=sharing
- **Runtimes alternativos** (citados na aula): ONNX Runtime e ExecuTorch. Esta
  atividade usa TensorFlow Lite como estudo de caso.
- Sem o arquivo `.tflite` ainda, é possível testar localmente em máquina x86
  (basta `pip install tensorflow`); o Dockerfile é o caminho recomendado para a Pi.