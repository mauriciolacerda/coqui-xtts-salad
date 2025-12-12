#!/usr/bin/env python3
import subprocess
import sys

def has_nvidia_gpu():
    """Detecta se há GPU NVIDIA disponível"""
    try:
        result = subprocess.run(
            ["nvidia-smi"], 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL,
            timeout=5
        )
        return result.returncode == 0
    except:
        return False

if has_nvidia_gpu():
    print("✅ GPU NVIDIA detectada! Instalando Torch com CUDA 12.1...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "torch==2.1.2", "torchaudio==2.1.2",
        "--index-url", "https://download.pytorch.org/whl/cu121",
        "--verbose"
    ])
else:
    print("⚠️ GPU não detectada. Instalando Torch CPU...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch==2.1.2", "torchaudio==2.1.2",
        "--index-url", "https://download.pytorch.org/whl/cpu",
        "--verbose"
    ])

print("🎉 Torch instalado com sucesso!")

# Verificação
import torch
print(f"🔍 Torch {torch.__version__} - CUDA: {torch.cuda.is_available()}")
