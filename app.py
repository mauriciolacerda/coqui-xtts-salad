from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse, JSONResponse
from TTS.api import TTS
import torch
import io
import soundfile as sf

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Carrega o modelo XTTS v2 uma vez na inicialização
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

@app.get("/started")
async def started():
    return {"status": "started"}

@app.get("/live")
async def live():
    return {"status": "live"}

@app.get("/ready")
async def ready():
    return {"status": "ready"}

@app.post("/tts")
async def tts_endpoint(
    text: str = Body(..., embed=True),
    language: str = Body("pt", embed=True),
    speaker: str = Body("Ana Florence", embed=True),  # exemplo de speaker
):
    # Gera waveform com XTTS v2
    wav = tts.tts(
        text=text,
        speaker=speaker,
        language=language,
    )

    buffer = io.BytesIO()
    sf.write(buffer, wav, 24000, format="WAV")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": 'inline; filename="audio.wav"'},
    )
