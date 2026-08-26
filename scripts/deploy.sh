#!/bin/bash
# scripts/deploy.sh
# Script de deploy contínuo in-place com monitoramento de saúde e rollback.
set -e

APP_DIR="/home/gilvantws/yolo-edge-api"
cd "$APP_DIR"

echo "=== INICIANDO IMPLANTAÇÃO NO EDGE ==="

# 1. Armazena o estado atual do commit Git para rollback em caso de falha de saúde
PREVIOUS_COMMIT=$(git rev-parse HEAD)
echo "Versão atual de segurança (Rollback Point): $PREVIOUS_COMMIT"

# 2. Puxa as novas atualizações de código e ponteiros do DVC
echo "Sincronizando repositório com o GitHub..."
git pull origin main

# 3. Puxa o modelo real correspondente do DVC storage
echo "Puxando modelo do DVC..."
dvc pull

# 4. Reconstrói e sobe os containers atualizados
echo "Subindo os containers Docker..."
docker compose up -d --build api

# 5. Cão de Guarda (Watchdog) de Saúde por 60 segundos (12 tentativas a cada 5 segundos)
echo "Iniciando Watchdog de Saúde..."
HEALTH_URL="http://localhost:8000/health"
DEPLOY_SUCCESS=false

for i in {1..12}; do
    echo "Tentativa $i/12 de checar saúde da API..."
    if curl -s -f "$HEALTH_URL" > /dev/null; then
        echo "✅ Sucesso! O novo container respondeu perfeitamente ao Health Check."
        DEPLOY_SUCCESS=true
        break
    fi
    echo "Serviço indisponível. Aguardando 5 segundos..."
    sleep 5
done

# 6. Se o teste de saúde falhar, faz o rollback automático para o commit anterior estável
if [ "$DEPLOY_SUCCESS" = false ]; then
    echo "❌ CRÍTICO: O novo container falhou no teste de saúde! Iniciando rollback automático..."
    
    # Reverte o Git para o commit anterior
    git checkout "$PREVIOUS_COMMIT"
    
    # Restaura o modelo do DVC para o correspondente daquele commit anterior
    dvc checkout
    
    # Sobe o container novamente com a versão restaurada estável
    docker compose up -d --build api
    
    echo "🔄 Rollback executado com sucesso! Sistema restaurado para a versão anterior estável."
    exit 1
fi

echo "=== DEPLOY CONCLUÍDO COM SUCESSO ABSOLUTO! 🚀 ==="
