#!/bin/bash
set -e

echo "🔍 Detectando ambiente..."

# Instala Torch adequado ao ambiente
python3 install_torch.py

echo "🚀 Iniciando servidor..."
uvicorn app:app --host 0.0.0.0 --port 8000
