"""
Worker GPU para processar jobs de TTS da fila Redis.
Processa chunks com checkpoint por chunk no GCS, permite retomada de jobs
interrompidos, e usa processamento paralelo limitado.
Suporta múltiplos modelos TTS (XTTS, F5-TTS) via factory pattern.
"""
import asyncio
import hashlib
import json
import logging
import os
import pickle
import signal
import time
import uuid
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import redis.asyncio as aioredis
import soundfile as sf
import torch
from google.cloud import storage

import config
from audio_processor import process_and_export, cleanup_temp_files
from tts_models import get_tts_model, BaseTTSModel

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Aceitar termos de serviço do Coqui
os.environ["COQUI_TOS_AGREED"] = "1"

# ============================================================================
# WORKER ID E SHUTDOWN
# ============================================================================

# ID único do worker (gerado no startup)
WORKER_ID = str(uuid.uuid4())

# Flag para shutdown graceful
shutdown_flag = False


def trim_silence(audio: np.ndarray, threshold: float = 0.01, sample_rate: int = 24000) -> np.ndarray:
    """
    Remove silêncios das bordas (início e fim) do áudio.
    
    Args:
        audio: Array numpy com dados do áudio
        threshold: Limiar de amplitude para considerar como silêncio (0.01 = 1% do máximo)
        sample_rate: Taxa de amostragem do áudio
    
    Returns:
        Array numpy com silêncios removidos das bordas
    """
    # Encontra índices onde amplitude está acima do threshold
    non_silent = np.abs(audio) > threshold
    
    if not np.any(non_silent):
        # Áudio completamente silencioso, retorna pequeno array
        return audio[:int(sample_rate * 0.1)]  # 100ms
    
    # Encontra primeiro e último sample não-silencioso
    first_sound = np.argmax(non_silent)
    last_sound = len(non_silent) - np.argmax(non_silent[::-1]) - 1
    
    # Adiciona pequena margem (50ms) para evitar cortes abruptos
    margin = int(sample_rate * 0.05)
    first_sound = max(0, first_sound - margin)
    last_sound = min(len(audio), last_sound + margin)
    
    return audio[first_sound:last_sound]


def signal_handler(signum, frame):
    """Handler para sinais de shutdown."""
    global shutdown_flag
    logger.warning(f"⚠️ Sinal {signum} recebido. Iniciando shutdown graceful...")
    shutdown_flag = True


# Registra handlers para SIGTERM e SIGINT
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


# ============================================================================
# CLIENTE REDIS
# ============================================================================

redis_client: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    """
    Retorna o cliente Redis (singleton) com retry robusto.
    Implementa exponential backoff para lidar com falhas temporárias de DNS.
    """
    global redis_client
    
    if redis_client is None:
        max_retries = 5
        base_delay = 1.0  # segundos
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🔄 Conectando ao Redis (tentativa {attempt + 1}/{max_retries})...")
                
                redis_client = await aioredis.from_url(
                    config.get_redis_url(),
                    encoding="utf-8",
                    decode_responses=False,  # Para suportar bytes (áudio)
                    socket_connect_timeout=10,
                    socket_keepalive=True,
                    health_check_interval=30,
                    retry_on_timeout=True,
                    max_connections=50
                )
                
                # Testa conexão
                await redis_client.ping()
                logger.info("✅ Redis conectado com sucesso")
                break
                
            except (aioredis.ConnectionError, OSError) as e:
                logger.warning(f"⚠️ Falha na conexão Redis (tentativa {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    # Exponential backoff
                    delay = base_delay * (2 ** attempt)
                    logger.info(f"⏳ Aguardando {delay:.1f}s antes de retry...")
                    await asyncio.sleep(delay)
                else:
                    logger.error("❌ Falha ao conectar ao Redis após todas as tentativas")
                    raise
    
    return redis_client


# ============================================================================
# CLIENTE GOOGLE CLOUD STORAGE
# ============================================================================

gcs_client: Optional[storage.Client] = None


def get_gcs_client() -> storage.Client:
    """Retorna o cliente GCS (singleton) com connection pool aumentado."""
    global gcs_client
    if gcs_client is None:
        try:
            # Configuração de connection pool para alto throughput
            # Aumenta de 10 (padrão) para 50 conexões simultâneas
            import google.auth.transport.requests
            from google.auth.transport.requests import AuthorizedSession
            import requests.adapters
            
            # Tenta usar a variável de ambiente primeiro
            gcs_credentials_json = os.getenv("GCS_CREDENTIALS_JSON")
            
            if gcs_credentials_json:
                from google.oauth2 import service_account
                
                logger.info("🔑 Usando credenciais GCS da variável de ambiente")
                credentials_dict = json.loads(gcs_credentials_json)
                credentials = service_account.Credentials.from_service_account_info(credentials_dict)
                
                # Cria client com connection pool aumentado
                gcs_client = storage.Client(
                    credentials=credentials,
                    project=credentials_dict.get('project_id')
                )
            else:
                # Fallback para arquivo JSON (desenvolvimento local)
                logger.info("🔑 Tentando usar arquivo de credenciais GCS")
                if config.GCS_CREDENTIALS_PATH and os.path.exists(config.GCS_CREDENTIALS_PATH):
                    gcs_client = storage.Client.from_service_account_json(
                        config.GCS_CREDENTIALS_PATH
                    )
                else:
                    logger.warning("⚠️ Nenhuma credencial GCS encontrada")
                    gcs_client = storage.Client()
            
            # Configura connection pool após inicialização
            # Aumenta max pool size de 10 para 50
            if hasattr(gcs_client, '_http'):
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=50,
                    pool_maxsize=50,
                    max_retries=3,
                    pool_block=False
                )
                gcs_client._http.mount("https://", adapter)
                gcs_client._http.mount("http://", adapter)
                logger.info("✅ GCS Client inicializado (pool: 50 conexões)")
            else:
                logger.info("✅ GCS Client inicializado com sucesso")
                
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar GCS Client: {str(e)}")
            raise
    
    return gcs_client


def upload_chunk_to_gcs_sync(job_id: str, chunk_index: int, audio_path: str) -> str:
    """
    Faz upload síncrono de um chunk para GCS (executado em thread).
    
    Args:
        job_id: ID do job
        chunk_index: Índice do chunk
        audio_path: Caminho local do arquivo de áudio
    
    Returns:
        URI do chunk no GCS
    """
    gcs = get_gcs_client()
    bucket = gcs.bucket(config.GCS_BUCKET)
    blob_path = f"{config.GCS_TEMP_PREFIX}{job_id}/chunk_{chunk_index:04d}.wav"
    blob = bucket.blob(blob_path)
    
    blob.upload_from_filename(audio_path)
    
    gcs_uri = f"gs://{config.GCS_BUCKET}/{blob_path}"
    logger.debug(f"✅ Chunk {chunk_index} uploaded: {gcs_uri}")
    return gcs_uri


async def upload_chunk_to_gcs(job_id: str, chunk_index: int, audio_path: str) -> str:
    """
    Faz upload assíncrono de um chunk para GCS (não bloqueia processamento).
    
    Args:
        job_id: ID do job
        chunk_index: Índice do chunk
        audio_path: Caminho local do arquivo de áudio
    
    Returns:
        URI do chunk no GCS (gs://bucket/temp/job_id/chunk_XXXX.wav)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        upload_chunk_to_gcs_sync,
        job_id,
        chunk_index,
        audio_path
    )


async def download_chunk_from_gcs(job_id: str, chunk_index: int, output_path: str) -> bool:
    """
    Baixa um chunk do GCS temporário.
    
    Args:
        job_id: ID do job
        chunk_index: Índice do chunk
        output_path: Caminho local para salvar o arquivo
    
    Returns:
        True se download bem-sucedido, False se chunk não existe
    """
    try:
        gcs = get_gcs_client()
        bucket = gcs.bucket(config.GCS_BUCKET)
        blob_path = f"{config.GCS_TEMP_PREFIX}{job_id}/chunk_{chunk_index:04d}.wav"
        blob = bucket.blob(blob_path)
        
        if not blob.exists():
            return False
        
        blob.download_to_filename(output_path)
        logger.debug(f"✅ Chunk {chunk_index} downloaded from GCS")
        return True
    except Exception as e:
        logger.error(f"❌ Erro ao baixar chunk {chunk_index}: {e}")
        return False


async def list_completed_chunks(job_id: str) -> List[int]:
    """
    Lista todos os chunks já completados no GCS para um job.
    
    Args:
        job_id: ID do job
    
    Returns:
        Lista de índices de chunks completados
    """
    try:
        gcs = get_gcs_client()
        bucket = gcs.bucket(config.GCS_BUCKET)
        prefix = f"{config.GCS_TEMP_PREFIX}{job_id}/"
        
        blobs = bucket.list_blobs(prefix=prefix)
        chunk_indices = []
        
        for blob in blobs:
            # Extrai índice do nome: chunk_0000.wav -> 0
            filename = blob.name.split("/")[-1]
            if filename.startswith("chunk_") and filename.endswith(".wav"):
                index_str = filename[6:10]  # chunk_XXXX.wav
                chunk_indices.append(int(index_str))
        
        return sorted(chunk_indices)
    except Exception as e:
        logger.error(f"❌ Erro ao listar chunks: {e}")
        return []


async def delete_job_temp_chunks(job_id: str) -> int:
    """
    Deleta todos os chunks temporários de um job do GCS.
    
    Args:
        job_id: ID do job
    
    Returns:
        Número de chunks deletados
    """
    try:
        gcs = get_gcs_client()
        bucket = gcs.bucket(config.GCS_BUCKET)
        prefix = f"{config.GCS_TEMP_PREFIX}{job_id}/"
        
        blobs = list(bucket.list_blobs(prefix=prefix))
        
        if blobs:
            bucket.delete_blobs(blobs)
            logger.info(f"🗑️ Deletados {len(blobs)} chunks temporários do job {job_id}")
        
        return len(blobs)
    except Exception as e:
        logger.error(f"❌ Erro ao deletar chunks temporários: {e}")
        return 0


# ============================================================================
# TRACKING DE CHUNKS NO REDIS
# ============================================================================

async def update_chunk_status(
    redis: aioredis.Redis,
    job_id: str,
    chunk_index: int,
    status: str,
    retry_count: int = 0,
    error: Optional[str] = None
):
    """
    Atualiza o status de um chunk específico no Redis.
    
    Args:
        redis: Cliente Redis
        job_id: ID do job
        chunk_index: Índice do chunk
        status: Status do chunk (pending/processing/completed/failed)
        retry_count: Número de tentativas já realizadas
        error: Mensagem de erro (se houver)
    """
    chunks_status_key = f"chunks_status:{job_id}"
    chunk_field = f"chunk_{chunk_index:04d}"
    
    chunk_data = {
        "status": status,
        "timestamp": time.time(),
        "retry_count": retry_count
    }
    
    if error:
        chunk_data["error"] = error
    
    await redis.hset(chunks_status_key, chunk_field, json.dumps(chunk_data))


async def get_chunks_status(redis: aioredis.Redis, job_id: str) -> Dict[int, Dict]:
    """
    Recupera o status de todos os chunks de um job.
    
    Args:
        redis: Cliente Redis
        job_id: ID do job
    
    Returns:
        Dicionário mapeando índice do chunk -> dados de status
        Ex: {0: {"status": "completed", "timestamp": 123456, "retry_count": 0}, ...}
    """
    chunks_status_key = f"chunks_status:{job_id}"
    chunks_raw = await redis.hgetall(chunks_status_key)
    
    chunks_status = {}
    for field_bytes, value_bytes in chunks_raw.items():
        field = field_bytes.decode()
        # Extrai índice: chunk_0000 -> 0
        if field.startswith("chunk_"):
            index = int(field[6:])
            chunks_status[index] = json.loads(value_bytes.decode())
    
    return chunks_status


async def initialize_chunks_status(
    redis: aioredis.Redis,
    job_id: str,
    total_chunks: int
):
    """
    Inicializa o status de todos os chunks como 'pending'.
    
    Args:
        redis: Cliente Redis
        job_id: ID do job
        total_chunks: Número total de chunks
    """
    chunks_status_key = f"chunks_status:{job_id}"
    
    # Usa pipeline para eficiência
    pipe = redis.pipeline()
    
    for chunk_index in range(total_chunks):
        chunk_field = f"chunk_{chunk_index:04d}"
        chunk_data = {
            "status": "pending",
            "timestamp": time.time(),
            "retry_count": 0
        }
        pipe.hset(chunks_status_key, chunk_field, json.dumps(chunk_data))
    
    # Define TTL para o hash inteiro
    pipe.expire(chunks_status_key, config.JOB_TTL)
    
    await pipe.execute()
    logger.info(f"✅ Inicializados {total_chunks} chunks como 'pending'")


# ============================================================================
# MODELO TTS (usando Factory Pattern)
# ============================================================================

# Instancia modelo TTS baseado na configuração (XTTS ou F5)
logger.info(f"🔧 Inicializando modelo TTS: {config.TTS_MODEL_TYPE}")
tts_model: BaseTTSModel = get_tts_model(config.TTS_MODEL_TYPE)


# ============================================================================
# COMPATIBILIDADE: Classe TTSModel legado (DEPRECATED)
# Mantida apenas para referência, código foi movido para tts_models/
# ============================================================================


# ============================================================================
# PROCESSAMENTO DE JOBS
# ============================================================================

async def get_or_compute_speaker_embeddings(
    audio_hash: str,
    speaker_wav_path: str,
    redis: aioredis.Redis
) -> Any:
    """
    Busca embeddings do cache Redis ou computa e cacheia.
    Suporta múltiplos modelos TTS (formato de embedding é model-specific).
    
    Args:
        audio_hash: Hash SHA256 do áudio
        speaker_wav_path: Caminho do áudio de referência (para computar se não estiver em cache)
        redis: Cliente Redis
    
    Returns:
        Voice embedding (formato depende do modelo TTS)
    """
    # Key inclui tipo de modelo para evitar conflito entre XTTS e F5
    embedding_key = f"{config.VOICE_EMBEDDING_PREFIX}{config.TTS_MODEL_TYPE}:{audio_hash}"
    
    # Tenta buscar do cache
    cached_data = await redis.get(embedding_key)
    
    if cached_data is not None:
        try:
            # Deserializa embeddings do pickle
            voice_embedding = pickle.loads(cached_data)
            
            # Para XTTS, embeddings são tupla de tensors - move para GPU se disponível
            if isinstance(voice_embedding, tuple) and len(voice_embedding) == 2:
                if torch.cuda.is_available():
                    gpt_cond_latent, speaker_embedding = voice_embedding
                    voice_embedding = (
                        gpt_cond_latent.to("cuda"),
                        speaker_embedding.to("cuda")
                    )
            
            logger.info(f"✅ Embeddings recuperados do cache ({config.TTS_MODEL_TYPE}): {audio_hash[:8]}")
            return voice_embedding
        except Exception as e:
            logger.warning(f"⚠️ Erro ao deserializar embeddings do cache: {e}. Recomputando...")
    
    # Computa embeddings
    logger.info(f"🧮 Computando embeddings para {audio_hash[:8]} ({config.TTS_MODEL_TYPE})...")
    start_time = time.time()
    
    voice_embedding = tts_model.get_voice_embedding(speaker_wav_path)
    
    compute_time = time.time() - start_time
    logger.info(f"✅ Embeddings computados em {compute_time:.2f}s")
    
    # Prepara para cache (move tensors para CPU se necessário)
    embeddings_to_cache = voice_embedding
    if isinstance(voice_embedding, tuple) and len(voice_embedding) == 2:
        # XTTS embeddings
        gpt_cond_latent, speaker_embedding = voice_embedding
        embeddings_to_cache = (gpt_cond_latent.cpu(), speaker_embedding.cpu())
    
    # Serializa e salva no cache
    try:
        embeddings_bytes = pickle.dumps(embeddings_to_cache)
        await redis.set(embedding_key, embeddings_bytes, ex=config.CACHE_TTL)
        
        size_kb = len(embeddings_bytes) / 1024
        logger.info(f"💾 Embeddings salvos no cache: {audio_hash[:8]} ({size_kb:.1f} KB)")
    except Exception as e:
        logger.error(f"❌ Erro ao cachear embeddings: {e}")
    
    return voice_embedding


async def update_job_status(
    redis: aioredis.Redis,
    job_id: str,
    status: str,
    **kwargs
):
    """
    Atualiza o status de um job no Redis.
    
    Args:
        redis: Cliente Redis
        job_id: ID do job
        status: Novo status
        **kwargs: Campos adicionais para atualizar
    """
    job_key = f"{config.JOB_KEY_PREFIX}{job_id}"
    
    updates = {"status": status}
    updates.update(kwargs)
    
    # Converte valores para string
    updates = {k: str(v) for k, v in updates.items()}
    
    await redis.hset(job_key, mapping=updates)
    logger.debug(f"Job {job_id} atualizado: {status}")


async def process_chunk_with_retry(
    chunk: Dict,
    voice_embedding: Any,
    language: str,
    output_path: str,
    max_retries: int = config.MAX_RETRIES_PER_CHUNK,
    temperature: float = 0.75,
    repetition_penalty: float = 2.0,
    length_penalty: float = 1.0
) -> bool:
    """
    Processa um chunk de texto com retry em caso de falha.
    Usa voice embeddings pré-computados (formato depende do modelo TTS).
    
    Args:
        chunk: Dicionário com informações do chunk
        voice_embedding: Voice embedding pré-computado (formato model-specific)
        language: Código do idioma
        output_path: Caminho para salvar o WAV gerado
        max_retries: Número máximo de tentativas
        temperature: Aleatoriedade (0.1-1.0, default 0.75)
        repetition_penalty: Anti-repetição (1.0-10.0, default 2.0)
        length_penalty: Duração (0.5-2.0, default 1.0)
    
    Returns:
        True se sucesso, False se falhou após todas as tentativas
    """
    chunk_index = chunk['index']
    chunk_text = chunk['text']
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Processando chunk {chunk_index} (tentativa {attempt}/{max_retries})")
            
            start_time = time.time()
            
            # Sintetiza usando modelo TTS (abstração suporta XTTS e F5)
            wav = tts_model.synthesize(
                text=chunk_text,
                voice_embedding=voice_embedding,
                language=language,
                # Parâmetros XTTS específicos (ignorados por outros modelos)
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty
            )
            
            # Converte para numpy array se necessário
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()
            elif isinstance(wav, list):
                wav = np.array(wav, dtype=np.float32)
            
            processing_time = time.time() - start_time
            
            # Salva WAV temporário (usa sample rate do modelo)
            sample_rate = tts_model.get_sample_rate()
            sf.write(output_path, wav, sample_rate, format="WAV")
            
            # Calcula estatísticas de performance
            audio_duration = len(wav) / sample_rate
            rtf = processing_time / audio_duration if audio_duration > 0 else 0
            
            logger.info(
                f"✅ Chunk {chunk_index} processado "
                f"({processing_time:.1f}s para {audio_duration:.1f}s de áudio, RTF: {rtf:.2f}x)"
            )
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro no chunk {chunk_index} (tentativa {attempt}): {e}")
            
            if attempt < max_retries:
                # Aguarda antes de tentar novamente (backoff exponencial)
                wait_time = 2 ** attempt
                logger.info(f"⏳ Aguardando {wait_time}s antes de tentar novamente...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"❌ Chunk {chunk_index} falhou após {max_retries} tentativas")
                return False
    
    return False


async def init_chunk_tracking(redis: aioredis.Redis, job_id: str, total_chunks: int):
    """
    Inicializa estrutura de tracking de chunks para processamento cooperativo.
    Cada chunk tem status próprio para reserva atômica entre workers.
    
    Args:
        redis: Cliente Redis
        job_id: ID do job
        total_chunks: Número total de chunks
    """
    logger.info(f"🔀 Inicializando tracking cooperativo para {total_chunks} chunks")
    
    pipe = redis.pipeline()
    for i in range(total_chunks):
        chunk_key = f"chunk_status:{job_id}:{i}"
        pipe.hset(chunk_key, mapping={
            "status": "pending",
            "reserved_by": "",
            "attempt": "0",
            "started_at": "0"
        })
        pipe.expire(chunk_key, config.JOB_TTL)
    
    await pipe.execute()
    logger.info(f"✅ Tracking de chunks inicializado")


async def reserve_next_chunk(redis: aioredis.Redis, job_id: str, total_chunks: int, worker_id: str) -> Optional[int]:
    """
    Reserva atomicamente o próximo chunk disponível para processamento.
    Usa HSETNX para garantir que apenas um worker processe cada chunk.
    
    Args:
        redis: Cliente Redis
        job_id: ID do job
        total_chunks: Número total de chunks
        worker_id: ID único do worker
    
    Returns:
        Índice do chunk reservado ou None se todos estão processados/reservados
    """
    for i in range(total_chunks):
        chunk_key = f"chunk_status:{job_id}:{i}"
        
        # Busca status atual do chunk
        chunk_data = await redis.hgetall(chunk_key)
        
        if not chunk_data:
            continue  # Chunk não existe
        
        # Converte bytes para string se necessário
        if isinstance(chunk_data.get(b"status"), bytes):
            status = chunk_data[b"status"].decode()
            reserved_by = chunk_data.get(b"reserved_by", b"").decode()
        else:
            status = chunk_data.get("status", "")
            reserved_by = chunk_data.get("reserved_by", "")
        
        # Só tenta reservar se: pending E não reservado
        if status == "pending" and not reserved_by:
            # Tenta reservar atomicamente com SET NX
            reserved = await redis.set(
                f"chunk_lock:{job_id}:{i}",
                worker_id,
                nx=True,
                ex=300  # Lock expira em 5 minutos
            )
            
            if reserved:
                # Atualiza status
                await redis.hset(chunk_key, mapping={
                    "status": "processing",
                    "reserved_by": worker_id,
                    "started_at": str(time.time())
                })
                logger.info(f"🔒 Worker {worker_id[:8]} reservou chunk {i}")
                return i
    
    return None


async def process_job_cooperative(job_id: str):
    """
    Processa job em modo cooperativo - múltiplos workers compartilham chunks.
    Cada worker reserva chunks atomicamente e processa de forma independente.
    
    Args:
        job_id: ID do job a processar
    """
    redis = await get_redis()
    heartbeat_task_obj = None
    chunks_processed = 0
    
    try:
        logger.info(f"🔀 Iniciando processamento COOPERATIVO do job {job_id}")
        
        # Busca dados do job
        job_key = f"{config.JOB_KEY_PREFIX}{job_id}"
        job_data_raw = await redis.hgetall(job_key)
        job_data = {k.decode(): v.decode() for k, v in job_data_raw.items()}
        
        language = job_data.get("language", "pt")
        audio_hash = job_data.get("audio_hash")
        
        # Extrai parâmetros de síntese
        temperature = float(job_data.get("temperature", "0.75"))
        repetition_penalty = float(job_data.get("repetition_penalty", "2.0"))
        length_penalty = float(job_data.get("length_penalty", "1.0"))
        
        # Busca chunks
        chunks_key = f"chunks:{job_id}"
        chunks_data = await redis.get(chunks_key)
        if chunks_data is None:
            raise RuntimeError(f"Chunks não encontrados para job {job_id}")
        
        chunks = json.loads(chunks_data.decode())
        total_chunks = len(chunks)
        
        logger.info(f"Job cooperativo {job_id}: {total_chunks} chunks disponíveis")
        
        # Marca job como processing (garante que status seja "processing")
        # NOTA: Primeiro worker da fila já marcou, mas garantimos aqui também
        current_status = job_data.get("status", "pending")
        if current_status != "processing":
            logger.info(f"⚠️ Job {job_id} estava com status '{current_status}', marcando como 'processing'")
            await redis.hset(job_key, "status", "processing")
        
        await redis.hset(job_key, "worker_id", WORKER_ID)
        await redis.hset(job_key, "reserved_at", str(time.time()))
        await redis.hset(job_key, "last_heartbeat", str(time.time()))
        
        # Inicializa tracking se necessário
        chunk_key_test = f"chunk_status:{job_id}:0"
        if not await redis.exists(chunk_key_test):
            await init_chunk_tracking(redis, job_id, total_chunks)
        
        # Inicia heartbeat
        heartbeat_task_obj = asyncio.create_task(heartbeat_task(redis, job_id))
        
        # Busca áudio de referência (com fallback para GCS)
        audio_ref_key = f"audio_ref:{audio_hash}"
        audio_data = await redis.get(audio_ref_key)
        
        if audio_data is None:
            # Fallback: tenta buscar do GCS usando voice_id
            voice_id = job_data.get("voice_id", "")
            
            if voice_id:
                logger.warning(f"⚠️ Áudio {audio_hash[:8]} expirou no Redis. Buscando do GCS: {voice_id}")
                try:
                    from google.cloud import storage
                    from google.oauth2 import service_account
                    
                    # Inicializa cliente GCS
                    gcs_credentials_json = os.getenv("GCS_CREDENTIALS_JSON")
                    if gcs_credentials_json:
                        credentials_dict = json.loads(gcs_credentials_json)
                        credentials = service_account.Credentials.from_service_account_info(credentials_dict)
                        gcs_client = storage.Client(credentials=credentials, project=config.GCS_PROJECT_ID)
                    else:
                        gcs_client = storage.Client()
                    
                    # Download do GCS
                    bucket = gcs_client.bucket(config.GCS_BUCKET)
                    blob_path = f"{config.GCS_VOICES_PREFIX}{voice_id}.wav"
                    blob = bucket.blob(blob_path)
                    
                    if blob.exists():
                        audio_data = blob.download_as_bytes()
                        logger.info(f"✅ Áudio recuperado do GCS: {voice_id} ({len(audio_data)} bytes)")
                        
                        # Re-salva no Redis para próximas requisições
                        await redis.set(audio_ref_key, audio_data, ex=config.JOB_TTL)
                        logger.info(f"♻️ Áudio re-salvo no Redis com TTL {config.JOB_TTL}s")
                    else:
                        raise RuntimeError(f"Áudio não encontrado no GCS: {blob_path}")
                        
                except Exception as e:
                    logger.error(f"❌ Falha ao recuperar áudio do GCS: {e}")
                    raise RuntimeError(f"Áudio de referência não encontrado no Redis e falha ao buscar do GCS: {audio_hash}")
            else:
                raise RuntimeError(f"Áudio de referência não encontrado no Redis (hash: {audio_hash}) e voice_id não disponível para fallback")
        
        # Salva áudio de referência temporariamente
        os.makedirs(config.TEMP_DIR, exist_ok=True)
        speaker_wav_path = os.path.join(config.TEMP_DIR, f"{job_id}_ref_{WORKER_ID[:8]}.wav")
        
        with open(speaker_wav_path, "wb") as f:
            f.write(audio_data)
        
        logger.info(f"Áudio de referência salvo: {speaker_wav_path}")
        
        # Computa embeddings uma vez (formato depende do modelo TTS)
        logger.info(f"🧮 Computando embeddings do speaker ({config.TTS_MODEL_TYPE})...")
        voice_embedding = await get_or_compute_speaker_embeddings(
            audio_hash=audio_hash,
            speaker_wav_path=speaker_wav_path,
            redis=redis
        )
        logger.info(f"✅ Embeddings prontos")
        
        # Loop: reserva e processa chunks até não haver mais disponíveis
        while True:
            # Reserva próximo chunk disponível
            chunk_idx = await reserve_next_chunk(redis, job_id, total_chunks, WORKER_ID)
            
            if chunk_idx is None:
                logger.info(f"✅ Sem mais chunks disponíveis para worker {WORKER_ID[:8]}")
                break
            
            # Processa chunk
            chunk = chunks[chunk_idx]
            output_path = os.path.join(config.TEMP_DIR, f"{job_id}_chunk_{chunk_idx:04d}.wav")
            
            chunk_key = f"chunk_status:{job_id}:{chunk_idx}"
            attempt = int(await redis.hget(chunk_key, "attempt") or 0)
            
            success = await process_chunk_with_retry(
                chunk=chunk,
                voice_embedding=voice_embedding,
                language=language,
                output_path=output_path,
                max_retries=config.MAX_RETRIES_PER_CHUNK - attempt,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty
            )
            
            if success:
                # Upload para GCS
                await upload_chunk_to_gcs(job_id, chunk_idx, output_path)
                
                # Marca como completed e remove lock
                await redis.hset(chunk_key, "status", "completed")
                await redis.delete(f"chunk_lock:{job_id}:{chunk_idx}")
                
                # Incrementa contador global
                completed_count = await redis.hincrby(job_key, "processed_chunks", 1)
                chunks_processed += 1
                
                # Atualiza progresso
                progress = (completed_count / total_chunks) * 100
                await redis.hset(job_key, "progress", f"{progress:.1f}")
                await redis.hset(job_key, "progress_visual", f"{progress:.1f}")
                
                logger.info(f"✅ Chunk {chunk_idx} completado ({completed_count}/{total_chunks} = {progress:.1f}%)")
                
                # Cleanup temp file
                if os.path.exists(output_path):
                    os.remove(output_path)
            else:
                # Falhou, incrementa tentativas
                new_attempt = await redis.hincrby(chunk_key, "attempt", 1)
                
                if new_attempt >= config.MAX_RETRIES_PER_CHUNK:
                    await redis.hset(chunk_key, "status", "failed")
                    await redis.delete(f"chunk_lock:{job_id}:{chunk_idx}")
                    logger.error(f"❌ Chunk {chunk_idx} falhou definitivamente após {new_attempt} tentativas")
                else:
                    # Ainda tem retries, volta para pending e libera lock
                    await redis.hset(chunk_key, "status", "pending")
                    await redis.hset(chunk_key, "reserved_by", "")
                    await redis.delete(f"chunk_lock:{job_id}:{chunk_idx}")
                    logger.warning(f"⚠️ Chunk {chunk_idx} falhou (tentativa {new_attempt}/{config.MAX_RETRIES_PER_CHUNK}), voltando para fila")
        
        # Cleanup
        if os.path.exists(speaker_wav_path):
            os.remove(speaker_wav_path)
        
        logger.info(f"✅ Worker {WORKER_ID[:8]} processou {chunks_processed} chunks do job {job_id}")
        
        # Verifica se é o último worker (todos chunks completed/failed)
        completed_count = int(await redis.hget(job_key, "processed_chunks") or 0)
        
        if completed_count >= total_chunks:
            logger.info(f"🎯 Último worker finalizando job {job_id}")
            await finalize_job(job_id, chunks, redis)
        
    except Exception as e:
        logger.error(f"❌ Erro no processamento cooperativo do job {job_id}: {e}", exc_info=True)
        
        # Marca chunks deste worker como pending para retry
        for i in range(total_chunks):
            chunk_key = f"chunk_status:{job_id}:{i}"
            reserved_by = await redis.hget(chunk_key, "reserved_by")
            
            if reserved_by == WORKER_ID.encode() or reserved_by == WORKER_ID:
                await redis.hset(chunk_key, mapping={
                    "status": "pending",
                    "reserved_by": ""
                })
                logger.info(f"♻️ Chunk {i} liberado para retry por outro worker")
        
        raise
    
    finally:
        if heartbeat_task_obj:
            heartbeat_task_obj.cancel()
            try:
                await heartbeat_task_obj
            except asyncio.CancelledError:
                pass


async def finalize_job(job_id: str, chunks: List[Dict], redis: aioredis.Redis):
    """
    Finaliza job concatenando chunks e fazendo upload para GCS.
    Chamado pelo último worker a completar chunks.
    
    Args:
        job_id: ID do job
        chunks: Lista de chunks
        redis: Cliente Redis
    """
    job_key = f"{config.JOB_KEY_PREFIX}{job_id}"
    
    try:
        logger.info(f"🎬 Finalizando job {job_id}...")
        
        # Verifica chunks falhados
        failed_chunks = []
        for i in range(len(chunks)):
            chunk_key = f"chunk_status:{job_id}:{i}"
            status = await redis.hget(chunk_key, "status")
            
            if status != b"completed" and status != "completed":
                failed_chunks.append(i)
        
        if failed_chunks:
            error_msg = f"Job incompleto: {len(failed_chunks)} chunks falharam: {failed_chunks[:10]}"
            logger.error(f"❌ {error_msg}")
            await redis.hset(job_key, mapping={
                "status": "failed",
                "error": error_msg,
                "completed_at": str(time.time())
            })
            return
        
        # Baixa todos os chunks do GCS
        logger.info(f"📥 Baixando {len(chunks)} chunks do GCS...")
        temp_files = []
        
        for chunk in chunks:
            chunk_index = chunk["index"]
            output_path = os.path.join(config.TEMP_DIR, f"{job_id}_chunk_{chunk_index:04d}.wav")
            
            downloaded = await download_chunk_from_gcs(job_id, chunk_index, output_path)
            
            if not downloaded:
                raise RuntimeError(f"Falha ao baixar chunk {chunk_index} do GCS")
            
            temp_files.append(output_path)
        
        # Concatena áudios com remoção de silêncios e pausas naturais
        logger.info(f"🔗 Concatenando {len(temp_files)} arquivos (removendo silêncios das bordas)...")
        combined_audio = []
        
        # Pausa natural entre chunks (150ms de silêncio)
        pause_duration = int(config.SAMPLE_RATE * 0.15)  # 150ms
        natural_pause = np.zeros(pause_duration, dtype=np.float32)
        
        for i, temp_file in enumerate(temp_files):
            audio_data, _ = sf.read(temp_file)
            
            # Remove silêncios das bordas
            audio_data = trim_silence(audio_data, threshold=0.01, sample_rate=config.SAMPLE_RATE)
            
            combined_audio.append(audio_data)
            
            # Adiciona pausa natural entre chunks (exceto no último)
            if i < len(temp_files) - 1:
                combined_audio.append(natural_pause)
        
        # Junta todos os arrays
        final_audio = np.concatenate(combined_audio)
        
        # Salva áudio final
        final_output_path = os.path.join(config.TEMP_DIR, f"{job_id}_final.wav")
        sf.write(final_output_path, final_audio, config.SAMPLE_RATE, format="WAV")
        
        logger.info(f"✅ Áudio final gerado: {final_output_path}")
        
        # Upload para GCS
        gcs_client = get_gcs_client()
        bucket = gcs_client.bucket(config.GCS_BUCKET)
        blob_path = f"{config.GCS_OUTPUT_PREFIX}{job_id}.wav"
        blob = bucket.blob(blob_path)
        
        blob.upload_from_filename(final_output_path)
        gcs_url = f"gs://{config.GCS_BUCKET}/{blob_path}"
        
        logger.info(f"✅ Áudio enviado para GCS: {gcs_url}")
        
        # Atualiza job como completed
        await redis.hset(job_key, mapping={
            "status": "completed",
            "audio_url": gcs_url,
            "completed_at": str(time.time())
        })
        
        logger.info(f"🎉 Job {job_id} finalizado com sucesso!")
        
        # Cleanup arquivos temporários
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        if os.path.exists(final_output_path):
            os.remove(final_output_path)
        
        logger.info(f"🧹 Arquivos temporários removidos")
        
    except Exception as e:
        logger.error(f"❌ Erro na finalização do job {job_id}: {e}", exc_info=True)
        await redis.hset(job_key, mapping={
            "status": "failed",
            "error": f"Erro na finalização: {str(e)}",
            "completed_at": str(time.time())
        })
        raise


async def process_job(job_id: str):
    """
    Processa um job completo de TTS com checkpoint por chunk e processamento paralelo.
    Roteia entre modo cooperativo (jobs grandes) e sequencial (jobs pequenos).
    
    Args:
        job_id: ID do job a processar
    """
    redis = await get_redis()
    
    # Busca dados do job para verificar se é spliteable
    job_key = f"{config.JOB_KEY_PREFIX}{job_id}"
    job_data_raw = await redis.hgetall(job_key)
    job_data = {k.decode(): v.decode() for k, v in job_data_raw.items()}
    
    is_spliteable = job_data.get("is_spliteable", "false") == "true"
    
    # Roteamento baseado no tipo de job
    if is_spliteable:
        logger.info(f"🔀 Job {job_id} é SPLITEABLE - modo cooperativo ativado")
        await process_job_cooperative(job_id)
    else:
        logger.info(f"📝 Job {job_id} é SEQUENCIAL - processamento tradicional")
        await process_job_sequential(job_id)


async def process_job_sequential(job_id: str):
    """
    Processa um job completo de TTS com checkpoint por chunk e processamento paralelo.
    Modo sequencial tradicional (job processado por um único worker).
    
    Args:
        job_id: ID do job a processar
    """
    redis = await get_redis()
    heartbeat_task = None  # Inicializa para evitar AttributeError no finally
    
    try:
        logger.info(f"🎯 Iniciando processamento do job {job_id}")
        
        # Busca dados do job
        job_key = f"{config.JOB_KEY_PREFIX}{job_id}"
        job_data_raw = await redis.hgetall(job_key)
        
        # Converte bytes para string
        job_data = {k.decode(): v.decode() for k, v in job_data_raw.items()}
        
        language = job_data.get("language", "pt")
        audio_hash = job_data.get("audio_hash")
        
        # Extrai parâmetros de síntese
        temperature = float(job_data.get("temperature", "0.75"))
        repetition_penalty = float(job_data.get("repetition_penalty", "2.0"))
        length_penalty = float(job_data.get("length_penalty", "1.0"))
        
        # Busca chunks
        chunks_key = f"chunks:{job_id}"
        chunks_data = await redis.get(chunks_key)
        if chunks_data is None:
            raise RuntimeError(f"Chunks não encontrados para job {job_id}")
        
        chunks = json.loads(chunks_data.decode())
        total_chunks = len(chunks)
        
        logger.info(f"Job {job_id}: {total_chunks} chunks para processar")
        
        # Inicializa ou recupera status dos chunks
        chunks_status = await get_chunks_status(redis, job_id)
        
        if not chunks_status:
            # Primeira vez processando este job
            await initialize_chunks_status(redis, job_id, total_chunks)
            chunks_status = {i: {"status": "pending", "retry_count": 0} for i in range(total_chunks)}
            
            # Inicializa contador de chunks em processamento
            await redis.hset(job_key, "processing_chunks", "0")
            
            logger.info("📝 Status de chunks inicializado")
        else:
            # Retomando job interrompido
            completed = len([c for c in chunks_status.values() if c["status"] == "completed"])
            pending = len([c for c in chunks_status.values() if c["status"] == "pending"])
            failed = len([c for c in chunks_status.values() if c["status"] == "failed"])
            processing = len([c for c in chunks_status.values() if c["status"] == "processing"])
            
            # Sincroniza contadores com chunks_status
            await redis.hset(job_key, "processed_chunks", str(completed))
            await redis.hset(job_key, "processing_chunks", "0")  # Reset ao retomar
            
            progress = (completed / total_chunks) * 100 if total_chunks > 0 else 0
            await redis.hset(job_key, "progress", f"{progress:.1f}")
            await redis.hset(job_key, "progress_visual", f"{progress:.1f}")
            
            logger.info(
                f"♻️ Retomando job: {completed} completados, {pending} pendentes, "
                f"{failed} falhados, {processing} interrompidos ({progress:.1f}%)"
            )
        
        # Busca áudio de referência (com fallback para GCS)
        audio_ref_key = f"audio_ref:{audio_hash}"
        audio_data = await redis.get(audio_ref_key)
        
        if audio_data is None:
            # Fallback: tenta buscar do GCS usando voice_id
            voice_id = job_data.get("voice_id", "")
            
            if voice_id:
                logger.warning(f"⚠️ Áudio {audio_hash[:8]} expirou no Redis. Buscando do GCS: {voice_id}")
                try:
                    from google.cloud import storage
                    from google.oauth2 import service_account
                    
                    # Inicializa cliente GCS
                    gcs_credentials_json = os.getenv("GCS_CREDENTIALS_JSON")
                    if gcs_credentials_json:
                        credentials_dict = json.loads(gcs_credentials_json)
                        credentials = service_account.Credentials.from_service_account_info(credentials_dict)
                        gcs_client = storage.Client(credentials=credentials, project=config.GCS_PROJECT_ID)
                    else:
                        gcs_client = storage.Client()
                    
                    # Download do GCS
                    bucket = gcs_client.bucket(config.GCS_BUCKET)
                    blob_path = f"{config.GCS_VOICES_PREFIX}{voice_id}.wav"
                    blob = bucket.blob(blob_path)
                    
                    if blob.exists():
                        audio_data = blob.download_as_bytes()
                        logger.info(f"✅ Áudio recuperado do GCS: {voice_id} ({len(audio_data)} bytes)")
                        
                        # Re-salva no Redis para próximas requisições
                        await redis.set(audio_ref_key, audio_data, ex=config.JOB_TTL)
                        logger.info(f"♻️ Áudio re-salvo no Redis com TTL {config.JOB_TTL}s")
                    else:
                        raise RuntimeError(f"Áudio não encontrado no GCS: {blob_path}")
                        
                except Exception as e:
                    logger.error(f"❌ Falha ao recuperar áudio do GCS: {e}")
                    raise RuntimeError(f"Áudio de referência não encontrado no Redis e falha ao buscar do GCS: {audio_hash}")
            else:
                raise RuntimeError(f"Áudio de referência não encontrado no Redis (hash: {audio_hash}) e voice_id não disponível para fallback")
        
        # Salva áudio de referência temporariamente
        os.makedirs(config.TEMP_DIR, exist_ok=True)
        speaker_wav_path = os.path.join(config.TEMP_DIR, f"{job_id}_ref.wav")
        
        with open(speaker_wav_path, "wb") as f:
            f.write(audio_data)
        
        logger.info(f"Áudio de referência salvo: {speaker_wav_path}")
        
        # ========================================
        # FASE 1: COMPUTA EMBEDDINGS UMA VEZ (cacheia e reusa)
        # ========================================
        
        logger.info(f"🧮 Preparando voice embedding ({config.TTS_MODEL_TYPE})...")
        
        # F5-TTS precisa do caminho do áudio diretamente, não embeddings pré-computados
        if config.TTS_MODEL_TYPE == "f5":
            voice_embedding = {
                "audio_path": speaker_wav_path,
                "ref_text": job_data.get("ref_text", "Olá, este é um teste de voz.")
            }
            logger.info(f"✅ F5-TTS: Usando áudio de referência: {speaker_wav_path}")
        else:
            # XTTS usa embeddings pré-computados
            voice_embedding = await get_or_compute_speaker_embeddings(
                audio_hash=audio_hash,
                speaker_wav_path=speaker_wav_path,
                redis=redis
            )
            logger.info(f"✅ Embeddings prontos para reuso em {total_chunks} chunks")
        
        # ========================================
        # PROCESSAMENTO PARALELO DE CHUNKS
        # ========================================
        
        # Semaphore para limitar concorrência
        sem = asyncio.Semaphore(config.MAX_PARALLEL_CHUNKS)
        
        # Função para processar um chunk individual
        async def process_single_chunk(chunk: Dict, chunk_status: Dict) -> bool:
            """Processa um chunk com retry e checkpoint."""
            chunk_index = chunk["index"]
            status = chunk_status.get("status", "pending")
            retry_count = chunk_status.get("retry_count", 0)
            was_already_completed = False  # Flag para evitar double-counting
            
            # Pula chunks já completados
            if status == "completed":
                output_path = os.path.join(config.TEMP_DIR, f"{job_id}_chunk_{chunk_index:04d}.wav")
                
                # Tenta baixar do GCS
                downloaded = await download_chunk_from_gcs(job_id, chunk_index, output_path)
                
                if downloaded:
                    logger.debug(f"✅ Chunk {chunk_index} já processado (recuperado do GCS)")
                    return True
                else:
                    # Chunk marcado como completed mas não está no GCS, reprocessar
                    logger.warning(
                        f"⚠️ Chunk {chunk_index} marcado como completed mas ausente no GCS. "
                        f"Reprocessando SEM incrementar contador (já foi contado antes)..."
                    )
                    status = "pending"
                    retry_count = 0
                    was_already_completed = True  # NÃO incrementar contador novamente!
            
            # Pula chunks que excederam limite de retries
            if status == "failed" and retry_count >= config.MAX_RETRIES_PER_CHUNK:
                logger.error(f"❌ Chunk {chunk_index} falhou definitivamente após {retry_count} tentativas")
                return False
            
            # Processa chunks pending ou failed (com retries disponíveis)
            async with sem:
                output_path = os.path.join(config.TEMP_DIR, f"{job_id}_chunk_{chunk_index:04d}.wav")
                
                # Marca como processing E incrementa contador
                await update_chunk_status(redis, job_id, chunk_index, "processing", retry_count)
                
                # Incrementa contador de chunks em processamento (para progress mais granular)
                await redis.hincrby(job_key, "processing_chunks", 1)
                
                # Processa usando embeddings pré-computados (formato model-specific)
                success = await process_chunk_with_retry(
                    chunk=chunk,
                    voice_embedding=voice_embedding,
                    language=language,
                    output_path=output_path,
                    max_retries=config.MAX_RETRIES_PER_CHUNK - retry_count,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty
                )
                
                if success:
                    # Inicia upload para GCS em background (não bloqueia)
                    upload_task = asyncio.create_task(
                        upload_chunk_to_gcs(job_id, chunk_index, output_path)
                    )
                    
                    # Marca como completed IMEDIATAMENTE (chunk já processado)
                    await update_chunk_status(redis, job_id, chunk_index, "completed", retry_count)
                    
                    # Atualiza progresso no job COM PROTEÇÃO contra over-counting
                    try:
                        # CORREÇÃO: Só incrementa se chunk NÃO estava completed antes
                        if not was_already_completed:
                            # Usa pipeline para atualizar atomicamente múltiplos contadores
                            pipe = redis.pipeline()
                            pipe.hincrby(job_key, "processed_chunks", 1)
                            pipe.hincrby(job_key, "processing_chunks", -1)  # Decrementa processing
                            results = await pipe.execute()
                            
                            completed_count = results[0]
                            processing_count = max(0, results[1])  # Garante não-negativo
                            
                            # PROTEÇÃO: Limita ao máximo de total_chunks
                            if completed_count > total_chunks:
                                logger.error(
                                    f"⚠️ OVER-COUNT detectado! processed_chunks={completed_count} > total_chunks={total_chunks}. "
                                    f"Corrigindo para {total_chunks}..."
                                )
                                await redis.hset(job_key, "processed_chunks", str(total_chunks))
                                completed_count = total_chunks
                        else:
                            # Chunk já estava completed, apenas decrementa processing_chunks
                            processing_count = await redis.hincrby(job_key, "processing_chunks", -1)
                            processing_count = max(0, processing_count)
                            completed_count = int(await redis.hget(job_key, "processed_chunks") or total_chunks)
                            
                            logger.info(
                                f"✅ Chunk {chunk_index} reprocessado (não incrementado - já estava completed antes)"
                            )
                        
                        # Progresso baseado apenas em chunks completados
                        progress = (completed_count / total_chunks) * 100
                        
                        # Progresso "visual" incluindo chunks em processamento (mais suave)
                        progress_with_processing = ((completed_count + processing_count) / total_chunks) * 100
                        
                        await redis.hset(job_key, "progress", f"{progress:.1f}")
                        await redis.hset(job_key, "progress_visual", f"{progress_with_processing:.1f}")
                        
                        if not was_already_completed:
                            logger.info(
                                f"✅ Chunk {chunk_index} completado "
                                f"({completed_count}/{total_chunks} completed, {processing_count} processing - {progress:.1f}%)"
                            )
                    except Exception as progress_err:
                        logger.error(f"❌ Erro ao atualizar progresso: {progress_err}")
                    
                    # Aguarda upload terminar (mas não bloqueia outros chunks)
                    await upload_task
                    
                    return True
                else:
                    # Marca como failed e incrementa retry_count
                    new_retry_count = retry_count + 1
                    await update_chunk_status(
                        redis, job_id, chunk_index, "failed", new_retry_count,
                        error=f"Falhou após {new_retry_count} tentativas"
                    )
                    logger.error(f"❌ Chunk {chunk_index} falhou (tentativa {new_retry_count}/{config.MAX_RETRIES_PER_CHUNK})")
                    return False
        
        # Processa todos os chunks em paralelo
        logger.info(f"🚀 Iniciando processamento paralelo (máx {config.MAX_PARALLEL_CHUNKS} simultâneos)")
        
        tasks = []
        for chunk in chunks:
            chunk_index = chunk["index"]
            chunk_status = chunks_status.get(chunk_index, {"status": "pending", "retry_count": 0})
            task = process_single_chunk(chunk, chunk_status)
            tasks.append(task)
        
        # Aguarda todos os chunks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verifica resultados
        failed_chunks = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Exceção ao processar chunk {idx}: {result}")
                failed_chunks.append(idx)
            elif not result:
                failed_chunks.append(idx)
        
        if failed_chunks:
            error_msg = f"Falha ao processar {len(failed_chunks)} chunks: {failed_chunks}"
            
            # Verifica se ainda há retries disponíveis
            retry_count = int(job_data.get("retry_count", 0))
            
            if retry_count >= 3:
                # Sem mais retries - marca como failed DEFINITIVAMENTE
                logger.error(f"❌ Job {job_id} falhou definitivamente após {retry_count} tentativas globais")
                await update_job_status(
                    redis,
                    job_id,
                    "failed",
                    error=error_msg,
                    completed_at=time.time()
                )
            else:
                # Ainda há retries - NÃO marca como failed, recoloca na fila
                new_retry_count = retry_count + 1
                logger.warning(
                    f"⚠️  Job {job_id} falhou mas será retentado "
                    f"(tentativa {new_retry_count}/3): {error_msg}"
                )
                
                # Recoloca na fila para retry
                await redis.rpush(config.REDIS_QUEUE_NAME, job_id)
                
                # Atualiza status para 'pending' e incrementa retry_count
                await redis.hset(job_key, mapping={
                    "status": "pending",
                    "retry_count": str(new_retry_count),
                    "last_error": error_msg
                })
                await redis.hdel(job_key, "worker_id", "reserved_at")
            
            # Remove job de processing:{WORKER_ID}
            await redis.lrem(f"processing:{WORKER_ID}", 1, job_id)
            logger.info(f"🗑️ Job {job_id} removido de processing:{WORKER_ID}")
            
            # Cancela heartbeat task
            if heartbeat_task and not heartbeat_task.done():
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
                logger.debug("💔 Heartbeat cancelado")
            
            cleanup_temp_files([speaker_wav_path])
            logger.error(f"❌ Job {job_id} falhou - NÃO será reprocessado: {error_msg}")
            return
        
        logger.info(f"✅ Todos os {total_chunks} chunks processados com sucesso")
        
        # ========================================
        # POST-PROCESSAMENTO E CONCATENAÇÃO
        # ========================================
        
        logger.info("🎵 Iniciando post-processamento e concatenação...")
        
        # Baixa todos os chunks do GCS em ordem
        chunk_files = []
        for chunk_index in range(total_chunks):
            output_path = os.path.join(config.TEMP_DIR, f"{job_id}_chunk_{chunk_index:04d}.wav")
            
            # Verifica se já existe localmente
            if not os.path.exists(output_path):
                downloaded = await download_chunk_from_gcs(job_id, chunk_index, output_path)
                if not downloaded:
                    raise RuntimeError(f"Chunk {chunk_index} não encontrado no GCS durante concatenação")
            
            chunk_files.append(output_path)
        
        # Concatena e exporta
        final_output_path = os.path.join(config.TEMP_DIR, f"{job_id}_final.mp3")
        
        process_and_export(
            chunk_files=chunk_files,
            output_path=final_output_path,
            format="mp3",
            bitrate=config.MP3_BITRATE,
            normalize=True,
            target_dBFS=-20.0,
            crossfade_ms=50
        )
        
        logger.info("✅ Post-processamento concluído")
        
        # Upload para GCS
        logger.info("☁️  Fazendo upload para GCS...")
        
        gcs = get_gcs_client()
        bucket = gcs.bucket(config.GCS_BUCKET)
        blob_path = f"{config.GCS_AUDIO_PREFIX}{job_id}.mp3"
        blob = bucket.blob(blob_path)
        
        blob.upload_from_filename(final_output_path)
        
        gcs_uri = f"gs://{config.GCS_BUCKET}/{blob_path}"
        logger.info(f"✅ Upload concluído: {gcs_uri}")
        
        # Deleta chunks temporários do GCS
        deleted_count = await delete_job_temp_chunks(job_id)
        logger.info(f"🗑️ {deleted_count} chunks temporários removidos do GCS")
        
        # Atualiza job como completed
        await update_job_status(
            redis,
            job_id,
            "completed",
            audio_url=gcs_uri,
            completed_at=time.time(),
            processed_chunks=total_chunks
        )
        
        # Limpa arquivos temporários
        logger.info("🧹 Limpando arquivos temporários...")
        cleanup_temp_files([speaker_wav_path, final_output_path] + chunk_files)
        
        logger.info(f"🎉 Job {job_id} concluído com sucesso!")
        
    except Exception as e:
        logger.error(f"❌ Erro fatal ao processar job {job_id}: {e}", exc_info=True)
        
        try:
            await update_job_status(
                redis,
                job_id,
                "failed",
                error=str(e),
                completed_at=time.time()
            )
        except Exception as update_error:
            logger.error(f"❌ Erro ao atualizar status de falha: {update_error}")


# ============================================================================
# HEARTBEAT E RECOVERY
# ============================================================================

async def heartbeat_task(redis: aioredis.Redis, job_id: str):
    """
    Task assíncrona que atualiza o heartbeat do job periodicamente.
    
    Args:
        redis: Cliente Redis
        job_id: ID do job sendo processado
    """
    global shutdown_flag
    
    job_key = f"{config.JOB_KEY_PREFIX}{job_id}"
    
    while not shutdown_flag:
        try:
            await redis.hset(job_key, "last_heartbeat", str(time.time()))
            logger.debug(f"💓 Heartbeat atualizado para job {job_id}")
            await asyncio.sleep(config.HEARTBEAT_INTERVAL)
        except Exception as e:
            logger.error(f"❌ Erro ao atualizar heartbeat: {e}")
            break


async def recover_orphaned_jobs():
    """
    Detecta e recupera jobs órfãos (sem heartbeat recente).
    Roda em loop infinito em background.
    """
    global shutdown_flag
    redis = await get_redis()
    
    logger.info("🔄 Serviço de recovery iniciado")
    
    while not shutdown_flag:
        try:
            await asyncio.sleep(config.RECOVERY_CHECK_INTERVAL)
            
            # Busca todos os jobs em "processing"
            cursor = 0
            orphaned_count = 0
            
            while True:
                cursor, keys = await redis.scan(
                    cursor,
                    match=f"{config.JOB_KEY_PREFIX}*",
                    count=100
                )
                
                for key_bytes in keys:
                    key = key_bytes.decode()
                    job_data_raw = await redis.hgetall(key)
                    
                    if not job_data_raw:
                        continue
                    
                    job_data = {k.decode(): v.decode() for k, v in job_data_raw.items()}
                    
                    if job_data.get("status") != "processing":
                        continue
                    
                    # Verifica heartbeat
                    last_heartbeat = float(job_data.get("last_heartbeat", 0))
                    time_since_heartbeat = time.time() - last_heartbeat
                    
                    if time_since_heartbeat > config.JOB_HEARTBEAT_TIMEOUT:
                        # Job órfão detectado
                        job_id = job_data.get("job_id")
                        worker_id = job_data.get("worker_id", "unknown")
                        
                        # Verifica retry count
                        retry_count = int(job_data.get("retry_count", 0))
                        
                        if retry_count >= 3:
                            # Excedeu limite de retries - marca como failed DEFINITIVAMENTE
                            logger.error(
                                f"❌ Job {job_id} excedeu limite de 3 retries globais "
                                f"(worker: {worker_id}). Marcando como failed definitivamente."
                            )
                            
                            await redis.hset(key, mapping={
                                "status": "failed",
                                "error": "Job excedeu limite de 3 tentativas após timeouts",
                                "completed_at": str(time.time())
                            })
                            
                            # Remove de processing
                            if worker_id and worker_id != "unknown":
                                processing_key = f"processing:{worker_id}"
                                await redis.lrem(processing_key, 0, job_id)
                            
                            orphaned_count += 1
                            continue
                        
                        logger.warning(
                            f"⚠️  Job órfão detectado: {job_id} "
                            f"(worker: {worker_id}, sem heartbeat há {time_since_heartbeat:.0f}s, "
                            f"retry {retry_count}/3)"
                        )
                        
                        # Remove da lista processing do worker
                        if worker_id and worker_id != "unknown":
                            processing_key = f"processing:{worker_id}"
                            await redis.lrem(processing_key, 0, job_id)
                        
                        # Incrementa retry_count
                        new_retry_count = retry_count + 1
                        
                        # Recoloca na fila principal
                        await redis.rpush(config.REDIS_QUEUE_NAME, job_id)
                        
                        # Reseta campos de reserva mas mantém chunks_status e incrementa retry
                        await redis.hdel(key, "worker_id", "reserved_at")
                        await redis.hset(key, mapping={
                            "status": "pending",
                            "last_heartbeat": "0",
                            "retry_count": str(new_retry_count)
                        })
                        
                        orphaned_count += 1
                        logger.info(f"✅ Job {job_id} recolocado na fila (tentativa {new_retry_count}/3)")
                
                if cursor == 0:
                    break
            
            if orphaned_count > 0:
                logger.info(f"🔄 Recovery: {orphaned_count} jobs órfãos recuperados")
                
        except Exception as e:
            logger.error(f"❌ Erro no recovery: {e}", exc_info=True)
    
    logger.info("🔄 Serviço de recovery finalizado")


async def cleanup_old_jobs():
    """
    Remove jobs antigos (completed/failed) do Redis.
    Roda em loop infinito em background.
    """
    global shutdown_flag
    redis = await get_redis()
    
    logger.info("🧹 Serviço de cleanup iniciado")
    
    while not shutdown_flag:
        try:
            await asyncio.sleep(config.CLEANUP_INTERVAL)
            
            cursor = 0
            cleaned_count = 0
            current_time = time.time()
            
            while True:
                cursor, keys = await redis.scan(
                    cursor,
                    match=f"{config.JOB_KEY_PREFIX}*",
                    count=100
                )
                
                for key_bytes in keys:
                    key = key_bytes.decode()
                    job_data_raw = await redis.hgetall(key)
                    
                    if not job_data_raw:
                        continue
                    
                    job_data = {k.decode(): v.decode() for k, v in job_data_raw.items()}
                    status = job_data.get("status")
                    
                    if status not in ["completed", "failed"]:
                        continue
                    
                    # Verifica idade do job
                    created_at = float(job_data.get("created_at", 0))
                    age = current_time - created_at
                    
                    if age > config.JOB_TTL:
                        job_id = job_data.get("job_id")
                        
                        # Deleta job metadata
                        await redis.delete(key)
                        
                        # Deleta chunks_status
                        chunks_status_key = f"chunks_status:{job_id}"
                        await redis.delete(chunks_status_key)
                        
                        # Deleta chunks_key
                        chunks_key = f"chunks:{job_id}"
                        await redis.delete(chunks_key)
                        
                        cleaned_count += 1
                        logger.debug(f"🗑️ Job antigo removido: {job_id} (status: {status}, idade: {age/3600:.1f}h)")
                
                if cursor == 0:
                    break
            
            if cleaned_count > 0:
                logger.info(f"🧹 Cleanup: {cleaned_count} jobs antigos removidos")
                
        except Exception as e:
            logger.error(f"❌ Erro no cleanup: {e}", exc_info=True)
    
    logger.info("🧹 Serviço de cleanup finalizado")


async def cleanup_reserved_chunks():
    """
    Libera todos os chunks reservados por este worker antes de desligar.
    Chamado durante shutdown para evitar chunks órfãos.
    """
    redis = await get_redis()
    
    logger.info(f"♻️ Liberando chunks reservados pelo worker {WORKER_ID[:8]}...")
    
    try:
        cursor = 0
        released_count = 0
        
        while True:
            cursor, keys = await redis.scan(
                cursor,
                match="chunk_status:*",
                count=100
            )
            
            for key_bytes in keys:
                key = key_bytes.decode()
                reserved_by = await redis.hget(key, "reserved_by")
                
                # Libera chunks reservados por este worker
                if reserved_by and (reserved_by.decode() == WORKER_ID or reserved_by == WORKER_ID):
                    await redis.hset(key, mapping={
                        "status": "pending",
                        "reserved_by": ""
                    })
                    released_count += 1
                    logger.debug(f"♻️ Chunk liberado: {key}")
            
            if cursor == 0:
                break
        
        if released_count > 0:
            logger.info(f"✅ {released_count} chunks liberados antes de shutdown")
        else:
            logger.info("✅ Nenhum chunk reservado para liberar")
            
    except Exception as e:
        logger.error(f"❌ Erro ao liberar chunks reservados: {e}", exc_info=True)


async def recover_orphaned_chunks():
    """
    Detecta e recupera chunks órfãos (reservados por workers inativos).
    Roda em loop infinito em background junto com recover_orphaned_jobs.
    """
    global shutdown_flag
    redis = await get_redis()
    
    logger.info("🔄 Serviço de recovery de chunks órfãos iniciado")
    
    while not shutdown_flag:
        try:
            await asyncio.sleep(config.RECOVERY_CHECK_INTERVAL)
            
            # Lista workers ativos (com heartbeat recente)
            active_workers = set()
            cursor = 0
            
            while True:
                cursor, keys = await redis.scan(
                    cursor,
                    match="processing:*",
                    count=100
                )
                
                for key_bytes in keys:
                    key = key_bytes.decode()
                    worker_id = key.replace("processing:", "")
                    
                    # Verifica jobs deste worker
                    jobs = await redis.lrange(key, 0, -1)
                    
                    if jobs:
                        # Worker tem jobs, verifica se algum tem heartbeat recente
                        for job_id_bytes in jobs:
                            job_id = job_id_bytes.decode()
                            job_key = f"{config.JOB_KEY_PREFIX}{job_id}"
                            last_heartbeat = await redis.hget(job_key, "last_heartbeat")
                            
                            if last_heartbeat:
                                age = time.time() - float(last_heartbeat)
                                if age < config.JOB_HEARTBEAT_TIMEOUT:
                                    active_workers.add(worker_id)
                                    break
                
                if cursor == 0:
                    break
            
            # Recupera chunks reservados por workers inativos
            cursor = 0
            recovered = 0
            
            while True:
                cursor, keys = await redis.scan(
                    cursor,
                    match="chunk_status:*",
                    count=100
                )
                
                for key_bytes in keys:
                    key = key_bytes.decode()
                    reserved_by = await redis.hget(key, "reserved_by")
                    status = await redis.hget(key, "status")
                    started_at = await redis.hget(key, "started_at")
                    
                    # Verifica se chunk está órfão
                    if reserved_by:
                        reserved_by_str = reserved_by.decode() if isinstance(reserved_by, bytes) else reserved_by
                        status_str = status.decode() if isinstance(status, bytes) else status
                        
                        # Chunk reservado por worker inativo?
                        is_orphaned = (
                            reserved_by_str and
                            reserved_by_str not in active_workers and
                            status_str == "processing"
                        )
                        
                        # OU chunk em processing há muito tempo (>5min)?
                        if started_at and not is_orphaned:
                            started_at_float = float(started_at.decode() if isinstance(started_at, bytes) else started_at)
                            age = time.time() - started_at_float
                            if age > 300:  # 5 minutos
                                is_orphaned = True
                        
                        if is_orphaned:
                            await redis.hset(key, mapping={
                                "status": "pending",
                                "reserved_by": ""
                            })
                            recovered += 1
                            logger.info(f"♻️ Chunk órfão recuperado: {key} (worker: {reserved_by_str[:8]})")
                
                if cursor == 0:
                    break
            
            if recovered > 0:
                logger.info(f"✅ {recovered} chunks órfãos recuperados")
                
        except Exception as e:
            logger.error(f"❌ Erro no recovery de chunks: {e}", exc_info=True)
    
    logger.info("🔄 Serviço de recovery de chunks órfãos finalizado")


# ============================================================================
# LOOP PRINCIPAL DO WORKER
# ============================================================================

async def worker_loop():
    """Loop principal que consome jobs da fila Redis e processa múltiplos jobs em paralelo."""
    global shutdown_flag
    
    logger.info(f"🚀 Worker iniciado (ID: {WORKER_ID})")
    
    # ========================================
    # DIAGNÓSTICO DE GPU
    # ========================================
    import subprocess
    
    logger.info("=" * 60)
    logger.info("🔍 DIAGNÓSTICO DE GPU")
    logger.info("=" * 60)
    
    # Verifica PyTorch
    logger.info(f"📦 PyTorch Version: {torch.__version__}")
    logger.info(f"🔧 CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"🎮 CUDA Version: {torch.version.cuda}")
        logger.info(f"🎮 GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.warning("⚠️  CUDA não disponível!")
        
        # Verifica nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"✅ nvidia-smi encontrado: {result.stdout.strip()}")
                logger.error("❌ PROBLEMA: nvidia-smi funciona MAS PyTorch não detecta CUDA!")
                logger.error("   Possível causa: PyTorch instalado sem suporte CUDA")
            else:
                logger.error(f"❌ nvidia-smi falhou (code {result.returncode})")
        except FileNotFoundError:
            logger.error("❌ nvidia-smi não encontrado - drivers NVIDIA ausentes")
        except Exception as e:
            logger.error(f"❌ Erro ao executar nvidia-smi: {e}")
    
    logger.info("=" * 60)
    
    logger.info(f"📊 Configurações:")
    logger.info(f"  - Worker ID: {WORKER_ID}")
    logger.info(f"  - Modelo TTS: {config.TTS_MODEL_TYPE}")
    logger.info(f"  - Redis: {config.REDIS_HOST}:{config.REDIS_PORT}")
    queue_name = getattr(config, f"REDIS_QUEUE_NAME_{config.TTS_MODEL_TYPE.upper()}", config.REDIS_QUEUE_NAME)
    logger.info(f"  - Fila: {queue_name}")
    logger.info(f"  - GCS Bucket: {config.GCS_BUCKET}")
    logger.info(f"  - Temp Dir: {config.TEMP_DIR}")
    logger.info(f"  - Chunks Paralelos: {config.MAX_PARALLEL_CHUNKS}")
    logger.info(f"  - Jobs Simultâneos: {config.MAX_CONCURRENT_JOBS}")
    
    # Carrega modelo TTS (XTTS ou F5)
    logger.info(f"🚀 Carregando modelo TTS: {config.TTS_MODEL_TYPE}")
    tts_model.load()
    
    # Conecta ao Redis
    redis = await get_redis()
    await redis.ping()
    logger.info("✅ Conectado ao Redis")
    
    # Inicia serviços em background
    recovery_task = asyncio.create_task(recover_orphaned_jobs())
    chunk_recovery_task = asyncio.create_task(recover_orphaned_chunks())
    cleanup_task = asyncio.create_task(cleanup_old_jobs())
    logger.info("✅ Serviços de recovery (jobs + chunks) e cleanup iniciados")
    
    # Nome da lista de processing deste worker
    processing_key = f"processing:{WORKER_ID}"
    
    # Semáforo para limitar jobs concorrentes
    job_semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_JOBS)
    
    # Set de tasks ativas
    active_tasks = set()
    
    async def process_job_wrapper(job_id: str):
        """Wrapper que processa job e gerencia heartbeat."""
        heartbeat = None
        try:
            async with job_semaphore:
                logger.info(f"📥 Iniciando job {job_id} ({len(active_tasks)}/{config.MAX_CONCURRENT_JOBS} ativos)")
                
                # Marca job como reservado por este worker
                job_key = f"{config.JOB_KEY_PREFIX}{job_id}"
                
                # GARANTE que status seja "processing" de forma robusta
                await redis.hset(job_key, mapping={
                    "worker_id": WORKER_ID,
                    "reserved_at": str(time.time()),
                    "status": "processing",
                    "last_heartbeat": str(time.time())
                })
                
                # Verifica se foi atualizado corretamente (debug)
                verify_status = await redis.hget(job_key, "status")
                if verify_status != b"processing" and verify_status != "processing":
                    logger.error(f"❌ ERRO: Status não foi atualizado! Esperado 'processing', obtido: {verify_status}")
                else:
                    logger.info(f"✅ Job {job_id} marcado como 'processing' com sucesso")
                
                # Inicia heartbeat em background
                heartbeat = asyncio.create_task(heartbeat_task(redis, job_id))
                
                try:
                    # Processa o job
                    await process_job(job_id)
                finally:
                    # Cancela heartbeat
                    if heartbeat and not heartbeat.done():
                        heartbeat.cancel()
                        try:
                            await heartbeat
                        except asyncio.CancelledError:
                            pass
                    
                    # Remove da lista processing
                    await redis.lrem(processing_key, 0, job_id)
                    logger.info(f"✅ Job {job_id} concluído e removido de processing")
        
        except Exception as e:
            logger.error(f"❌ Erro ao processar job {job_id}: {e}", exc_info=True)
        finally:
            # Remove task do set
            active_tasks.discard(asyncio.current_task())
    
    # Loop infinito consumindo jobs
    logger.info(f"👂 Aguardando jobs na fila (capacidade: {config.MAX_CONCURRENT_JOBS} simultâneos)...")
    
    try:
        while not shutdown_flag:
            try:
                # Verifica se há capacidade para mais jobs
                if len(active_tasks) >= config.MAX_CONCURRENT_JOBS:
                    # Aguarda pelo menos 1 job terminar
                    done, active_tasks_set = await asyncio.wait(
                        active_tasks,
                        timeout=1,
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    active_tasks = active_tasks_set
                    continue
                
                # BRPOPLPUSH: move job da fila TTS para processing:{WORKER_ID}
                # Atomicamente reserva o job para este worker
                # Fila baseada no tipo de modelo (XTTS ou F5)
                queue_name = getattr(config, f"REDIS_QUEUE_NAME_{config.TTS_MODEL_TYPE.upper()}", config.REDIS_QUEUE_NAME)
                
                job_id_bytes = await redis.brpoplpush(
                    queue_name,
                    processing_key,
                    timeout=1
                )
                
                if job_id_bytes is None:
                    # Timeout, continua esperando
                    # Aproveita para limpar tasks concluídas
                    active_tasks = {t for t in active_tasks if not t.done()}
                    continue
                
                job_id = job_id_bytes.decode()
                logger.info(f"📨 Job reservado da fila: {job_id}")
                
                # Cria task para processar job em paralelo
                task = asyncio.create_task(process_job_wrapper(job_id))
                active_tasks.add(task)
                
            except asyncio.CancelledError:
                logger.warning("⚠️  Worker cancelado")
                break
            except Exception as e:
                logger.error(f"❌ Erro no loop do worker: {e}", exc_info=True)
                await asyncio.sleep(5)
    
    finally:
        # Shutdown graceful
        logger.info("🛑 Iniciando shutdown do worker...")
        
        # Libera chunks reservados por este worker (ANTES de aguardar jobs)
        await cleanup_reserved_chunks()
        
        # Aguarda jobs ativos finalizarem (com timeout)
        if active_tasks:
            logger.info(f"⏳ Aguardando {len(active_tasks)} jobs ativos finalizarem...")
            done, pending = await asyncio.wait(active_tasks, timeout=30)
            
            # Cancela jobs que não terminaram
            if pending:
                logger.warning(f"⚠️  Cancelando {len(pending)} jobs pendentes...")
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
        
        # Cancela tasks de background
        recovery_task.cancel()
        chunk_recovery_task.cancel()
        cleanup_task.cancel()
        
        try:
            await asyncio.gather(recovery_task, chunk_recovery_task, cleanup_task, return_exceptions=True)
        except:
            pass
        
        # Recoloca jobs em processamento de volta na fila
        processing_jobs = await redis.lrange(processing_key, 0, -1)
        if processing_jobs:
            logger.info(f"🔄 Recolocando {len(processing_jobs)} jobs na fila...")
            for job_id_bytes in processing_jobs:
                job_id = job_id_bytes.decode()
                await redis.rpush(config.REDIS_QUEUE_NAME, job_id)
                
                # Reseta status
                job_key = f"{config.JOB_KEY_PREFIX}{job_id}"
                await redis.hdel(job_key, "worker_id", "reserved_at")
                await redis.hset(job_key, "status", "pending")
            
            # Limpa lista processing
            await redis.delete(processing_key)
        
        logger.info("👋 Worker finalizado")


# ============================================================================
# ENTRYPOINT
# ============================================================================

if __name__ == "__main__":
    logger.info(f"🚀 Iniciando Worker GPU (ID: {WORKER_ID})...")
    
    # Inicia worker loop
    try:
        asyncio.run(worker_loop())
    except KeyboardInterrupt:
        logger.info("🛑 Worker finalizado")
