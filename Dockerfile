FROM python:3.10-slim

# Instalar dependências básicas
RUN apt-get update && apt-get install -y \
    libsndfile1 ffmpeg git wget curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ✅ Detectar GPU e instalar Torch ANTES de copiar o resto
COPY install_torch.py .
RUN python3 install_torch.py

COPY requirements.txt .

# Instalar dependências do app (sem torch no requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# ✅ PRÉ-CARREGAR modelo XTTS para evitar download no runtime
ENV COQUI_TOS_AGREED=1
RUN python3 -c "from TTS.api import TTS; print('📦 Baixando modelo XTTS...'); TTS('tts_models/multilingual/multi-dataset/xtts_v2'); print('✅ Modelo cacheado!')"

# Copiar arquivos
COPY startup.sh ./
RUN chmod +x startup.sh

COPY . .

EXPOSE 8000

CMD ["./startup.sh"]