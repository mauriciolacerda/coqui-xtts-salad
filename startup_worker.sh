#!/bin/bash
set -e

echo "🚀 Inicializando F5-TTS Worker..."

# Instala FFmpeg se não estiver presente
if ! command -v ffmpeg &> /dev/null; then
    echo "📦 Instalando FFmpeg..."
    apt-get update -qq && apt-get install -y -qq ffmpeg libsndfile1 > /dev/null 2>&1
    echo "✅ FFmpeg instalado"
else
    echo "✅ FFmpeg já instalado"
fi

# Variáveis de configuração
REPO_URL="${GIT_REPO_URL:-https://github.com/mauricioalacerda/tts-worker-f5-code.git}"
GIT_BRANCH="${GIT_BRANCH:-main}"
CODE_DIR="/app/code"

cd /app

# Se código já existe, faz pull; senão, clona
if [ -d "$CODE_DIR/.git" ]; then
    echo "📦 Código encontrado, atualizando..."
    cd "$CODE_DIR"
    git fetch origin
    git reset --hard origin/$GIT_BRANCH
    echo "✅ Código atualizado para branch: $GIT_BRANCH"
else
    echo "📥 Clonando código do repositório..."
    git clone --branch $GIT_BRANCH $REPO_URL $CODE_DIR
    echo "✅ Código clonado da branch: $GIT_BRANCH"
fi

# Exibe último commit
cd "$CODE_DIR"
echo "📌 Commit atual:"
git log -1 --oneline

# Copia credenciais GCS para o diretório de código
if [ -f "/app/credentials/gcs-key.json" ]; then
    mkdir -p "$CODE_DIR/credentials"
    cp /app/credentials/gcs-key.json "$CODE_DIR/credentials/"
    echo "✅ Credenciais GCS copiadas"
fi

# Executa worker
echo "🎯 Iniciando worker..."
cd "$CODE_DIR"
exec python3 worker.py
