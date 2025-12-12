# XTTS Voice Cloning API

API de clonagem de voz usando Coqui XTTS v2 com suporte universal para GPU/CPU.

## 🚀 Features

- ✅ Detecção automática de GPU (CUDA) ou CPU
- ✅ Mesma imagem Docker funciona em qualquer ambiente
- ✅ Clonagem de voz com áudio de referência
- ✅ Suporte multilíngue
- ✅ Autenticação via API Key (opcional)
- ✅ Health checks para Kubernetes/Salad

## 🔒 Autenticação

### Desenvolvimento Local (sem autenticação)
Por padrão, a API funciona **sem autenticação** localmente.

### Produção (Salad/Cloud)
Configure a variável de ambiente `API_KEY`:

```bash
export API_KEY="sua-chave-super-secreta-123xyz"
```

No Salad Portal:
1. Vá em **Container Groups** → **Edit**
2. Adicione **Environment Variable**:
   - Key: `API_KEY`
   - Value: `sua-chave-secreta`

## 📡 Endpoints

### Health Checks
- `GET /started` - Container iniciado
- `GET /live` - Aplicação viva
- `GET /ready` - Modelo carregado e pronto

### TTS Endpoints (requer API Key em produção)

#### TTS Simples
```bash
POST /tts
Headers:
  X-API-Key: sua-chave-secreta
Body (JSON):
{
  "text": "Olá, mundo!",
  "language": "pt"
}
```

#### TTS com Clonagem de Voz
```bash
POST /tts-with-ref
Headers:
  X-API-Key: sua-chave-secreta
Body (form-data):
  text: "Texto para sintetizar"
  language: "pt"
  speaker_wav: [arquivo WAV de referência]
```

## 🧪 Testes

### PowerShell (com autenticação)
```powershell
$headers = @{
    "X-API-Key" = "sua-chave-secreta"
}

$form = @{
    text = "Teste de voz"
    language = "pt"
    speaker_wav = Get-Item -Path "david.wav"
}

Invoke-WebRequest -Method Post `
    -Uri "http://localhost:8000/tts-with-ref" `
    -Form $form `
    -Headers $headers `
    -OutFile "output.wav"
```

### cURL (com autenticação)
```bash
curl -X POST http://localhost:8000/tts-with-ref \
  -H "X-API-Key: sua-chave-secreta" \
  -F "text=Olá mundo" \
  -F "language=pt" \
  -F "speaker_wav=@david.wav;type=audio/wav" \
  -o output.wav
```

## 🐳 Docker

### Build Local
```bash
docker build -t xtts-universal .
```

### Run Local (sem GPU)
```bash
docker run -d -p 8000:8000 xtts-universal
```

### Run Local (com GPU)
```bash
docker run -d --gpus all -p 8000:8000 xtts-universal
```

### Run com API Key
```bash
docker run -d -p 8000:8000 \
  -e API_KEY="sua-chave-secreta" \
  xtts-universal
```

## ⚡ Performance

| Ambiente | Tempo médio |
|----------|-------------|
| CPU local | ~22s |
| GPU 1050 Ti | ~5-7s |
| GPU Cloud (RTX 3060+) | ~3-5s |

## 📦 Deploy no Salad

1. A imagem já está publicada em: `ghcr.io/mauriciolacerda/xtts-salad:latest`
2. Configure as Environment Variables:
   - `API_KEY=sua-chave-forte`
3. Configure Health Checks:
   - Startup: `/started` (10s delay)
   - Liveness: `/live` (30s delay)
   - Readiness: `/ready` (180s delay)
4. Recursos recomendados:
   - vCPU: 4
   - RAM: 16GB
   - GPU: RTX 3060 ou superior
   - Storage: 20GB

## 🔧 Variáveis de Ambiente

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `API_KEY` | *(vazio)* | Chave de autenticação (opcional em dev) |
| `COQUI_TOS_AGREED` | `1` | Aceite dos termos Coqui |

## 📝 License

Este projeto usa Coqui TTS (CPML License).
Verifique: https://coqui.ai/cpml.txt
