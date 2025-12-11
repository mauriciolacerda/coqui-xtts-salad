FROM python:3.10-slim

# Instalar dependências básicas
RUN apt-get update && apt-get install -y \
    libsndfile1 ffmpeg git wget curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Instalar dependências do app (sem torch no requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Copiar script de instalação do Torch
COPY install_torch.py .

# Instalação automática do Torch adequado ao ambiente
RUN python3 install_torch.py

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]