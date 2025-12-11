import os
import io
import soundfile as sf
import torch

from fastapi import FastAPI, Body, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from TTS.api import TTS
from tempfile import NamedTemporaryFile

os.environ["COQUI_TOS_AGREED"] = "1"

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Carrega o modelo XTTS v2 uma vez na inicialização
try:
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
except Exception as e:
    raise RuntimeError(f"Failed to load TTS model: {e}")

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
async def tts_simple(
    text: str = Body(...),
    language: str = Body("pt"),
):
    try:
        wav = tts.tts(text=text, language=language)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    buffer = io.BytesIO()
    sf.write(buffer, wav, 24000, format="WAV")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="audio/wav")

@app.post("/tts-with-ref")
async def tts_with_reference(
    text: str = Form(...),
    language: str = Form("pt"),
    speaker_wav: UploadFile = File(...),
):
    # Verifica tipo de arquivo (ideal .wav)
    if speaker_wav.content_type not in ("audio/wav", "audio/x-wav", "audio/wave"):
        raise HTTPException(status_code=400, detail="Envie um arquivo WAV em speaker_wav.")

    # Salva o arquivo de referência em um temp file
    try:
        with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await speaker_wav.read())
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao salvar arquivo de referência: {e}")

    try:
        # Chama o XTTS passando o caminho do arquivo de referência
        wav = tts.tts(
            text=text,
            language=language,
            speaker_wav=tmp_path,  # aqui é a mágica
        )
    except Exception as e:
        # Loga o erro e retorna 500
        print("Erro ao gerar TTS:", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Limpa o arquivo temporário
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    # Converte para WAV em memória e retorna
    buffer = io.BytesIO()
    sf.write(buffer, wav, 24000, format="WAV")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": 'inline; filename="tts_xtts_ref.wav"'},
    )
