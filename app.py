import os
import io
import soundfile as sf
import torch
import asyncio

from fastapi import FastAPI, Body, HTTPException, UploadFile, File, Form, Header
from fastapi.responses import StreamingResponse, JSONResponse
from TTS.api import TTS
from tempfile import NamedTemporaryFile

os.environ["COQUI_TOS_AGREED"] = "1"

app = FastAPI()

# ✅ API Key de autenticação (via variável de ambiente)
API_KEY = os.getenv("API_KEY", "")

def verify_api_key(x_api_key: str = Header(None)):
    """Valida o API Key no header X-API-Key"""
    if not API_KEY:
        # Se não houver API_KEY configurada, permite acesso (dev local)
        return True
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(
            status_code=401, 
            detail="API Key inválida ou ausente. Use header: X-API-Key"
        )
    return True

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = None  # ✅ Inicializa como None

@app.on_event("startup")
async def startup_event():
    """Carrega o modelo de forma assíncrona com timeout"""
    global tts
    try:
        print(f"🚀 Carregando modelo XTTS no device: {device}")
        print(f"🔍 CUDA disponível: {torch.cuda.is_available()}")
        
        # Timeout de 5 minutos para carregamento
        async def load_model():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                lambda: TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
            )
        
        tts = await asyncio.wait_for(load_model(), timeout=300)
        print("✅ Modelo carregado com sucesso!")
        
    except asyncio.TimeoutError:
        print("❌ Timeout ao carregar modelo (>5min)")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        # Não raise aqui - deixa a aplicação subir

@app.get("/started")
async def started():
    """Health check de inicialização"""
    return JSONResponse({"status": "started"}, status_code=200)

@app.get("/live")
async def live():
    """Health check de liveness"""
    return JSONResponse({"status": "live"}, status_code=200)

@app.get("/ready")
async def ready():
    """Health check de readiness - só retorna 200 quando modelo estiver carregado"""
    if tts is None:
        return JSONResponse({"status": "loading", "message": "Model still loading"}, status_code=503)
    return JSONResponse({"status": "ready"}, status_code=200)

@app.post("/tts")
async def tts_simple(
    text: str = Body(...),
    language: str = Body("pt"),
    x_api_key: str = Header(None),
):
    verify_api_key(x_api_key)  # ✅ Valida API Key
    
    if tts is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
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
    x_api_key: str = Header(None),
):
    verify_api_key(x_api_key)  # ✅ Valida API Key
    
    if tts is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    if speaker_wav.content_type not in ("audio/wav", "audio/x-wav", "audio/wave"):
        raise HTTPException(status_code=400, detail="Envie um arquivo WAV em speaker_wav.")

    try:
        with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await speaker_wav.read())
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao salvar arquivo de referência: {e}")

    try:
        wav = tts.tts(
            text=text,
            language=language,
            speaker_wav=tmp_path,
        )
    except Exception as e:
        print("Erro ao gerar TTS:", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    buffer = io.BytesIO()
    sf.write(buffer, wav, 24000, format="WAV")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": 'inline; filename="tts_xtts_ref.wav"'},
    )
