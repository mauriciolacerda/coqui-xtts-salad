FROM python:3.10-slim

# Instalar dependências básicas
RUN apt-get update && apt-get install -y \
    libsndfile1 ffmpeg git wget curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Instalar dependências do app (sem torch no requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Copiar arquivos
COPY install_torch.py startup.sh ./
RUN chmod +x startup.sh

COPY . .

EXPOSE 8000

# ✅ Executa script que instala Torch e inicia servidor
CMD ["./startup.sh"]