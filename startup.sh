#!/bin/bash
set -e

echo "🚀 Iniciando aplicação..."
echo "🔍 GPU disponível: $(python3 -c 'import torch; print("✅ CUDA" if torch.cuda.is_available() else "⚠️ CPU")')"

# Inicia servidor direto (Torch já instalado no Dockerfile)
exec uvicorn app:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 120
