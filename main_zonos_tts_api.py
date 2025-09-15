# //////////////////////////////////////////////////////////////////////////////////////
# ///// Setup Logger ///////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////

from pathlib import Path

from logger import set_logger_name

set_logger_name(logger_name=Path(__file__).stem)

# //////////////////////////////////////////////////////////////////////////////////////
# ///// Imports ////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////

import os
import json
import uuid
import time
import torch
import traceback
import torchaudio
import multiprocessing
from io import BytesIO
from typing import Dict, Optional, List, Union

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, conlist, confloat

from zonos.model import Zonos
from zonos.conditioning import make_cond_dict, supported_language_codes
from logger import logger

# //////////////////////////////////////////////////////////////////////////////////////
# ///// Initialization /////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////

multiprocessing.set_start_method('spawn', force=True)

app = FastAPI(title="Zonos API", description="OpenAI-compatible TTS API for Zonos")

MODELS = {"transformer": None, "hybrid": None}
VOICE_STORAGE_DIR = os.environ.get("VOICE_STORAGE_DIR", "data/voice_storage")
VOICE_METADATA_FILE = os.path.join(VOICE_STORAGE_DIR, "voice_metadata.json")
VOICE_CACHE: Dict[str, torch.Tensor] = {}

os.makedirs(VOICE_STORAGE_DIR, exist_ok=True)

# //////////////////////////////////////////////////////////////////////////////////////
# ///// Helpers ////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////

def load_models():
    try:
        logger.info("Loading Zonos models into VRAM...")
        device = "cuda"
        MODELS["transformer"] = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device).eval().requires_grad_(False)
        MODELS["hybrid"] = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device).eval().requires_grad_(False)
        logger.info("Models loaded successfully.")
    except Exception:
        logger.error("Failed to load models:\n%s", traceback.format_exc())

def load_voice_metadata():
    try:
        if os.path.exists(VOICE_METADATA_FILE):
            with open(VOICE_METADATA_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        logger.error("Error loading voice metadata:\n%s", traceback.format_exc())
    return {}

def save_voice_metadata(metadata):
    try:
        with open(VOICE_METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception:
        logger.error("Error saving voice metadata:\n%s", traceback.format_exc())

def load_voice_embeddings():
    metadata = load_voice_metadata()
    for voice_id, info in metadata.items():
        tensor_path = os.path.join(VOICE_STORAGE_DIR, f"{voice_id}.pt")
        try:
            if os.path.exists(tensor_path):
                VOICE_CACHE[voice_id] = torch.load(tensor_path, map_location="cuda")
                logger.info(f"Loaded voice: {info.get('name', voice_id)}")
        except Exception:
            logger.error("Error loading voice embedding %s:\n%s", voice_id, traceback.format_exc())

def save_voice_embedding(voice_id, embedding):
    try:
        torch.save(embedding, os.path.join(VOICE_STORAGE_DIR, f"{voice_id}.pt"))
        return True
    except Exception:
        logger.error("Error saving voice embedding %s:\n%s", voice_id, traceback.format_exc())
        return False

def get_voice_embedding(voice_identifier):
    metadata = load_voice_metadata()
    if voice_identifier in VOICE_CACHE:
        return VOICE_CACHE[voice_identifier]

    for voice_id, info in metadata.items():
        if info.get("name") == voice_identifier:
            tensor_path = os.path.join(VOICE_STORAGE_DIR, f"{voice_id}.pt")
            try:
                if os.path.exists(tensor_path):
                    embedding = torch.load(tensor_path, map_location="cuda")
                    VOICE_CACHE[voice_id] = embedding
                    return embedding
            except Exception:
                logger.error("Error loading voice %s:\n%s", voice_id, traceback.format_exc())
    return None

# //////////////////////////////////////////////////////////////////////////////////////
# ///// Request Models /////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////

class SpeechRequest(BaseModel):
    model: str = Field("Zyphra/Zonos-v0.1-transformer")
    input: str = Field(..., max_length=500)
    voice: Optional[str] = None
    speed: float = Field(1.0, ge=0.5, le=2.0)
    language: str = Field("en-us")
    emotion: Optional[Dict[str, float]] = None
    response_format: str = Field("mp3")
    prefix_audio: Optional[str] = None
    top_k: Optional[int] = Field(None, ge=1)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_p: Optional[float] = Field(0.15, ge=0.0, le=1.0)
    pitch_std: Optional[float] = Field(None, ge=0.0, le=400.0)
    speaking_rate: Optional[float] = Field(None, ge=0.0, le=40.0)
    fmax: Optional[float] = Field(None, ge=0.0, le=24000.0)
    vqscore_8: Optional[conlist(confloat(ge=0.5, le=0.8), min_length=8, max_length=8)] = None
    dnsmos_ovrl: Optional[float] = Field(None, ge=1.0, le=5.0)
    speaker_noised: Optional[bool] = None

class VoiceResponse(BaseModel):
    voice_id: str
    name: Optional[str]
    created: int

class VoiceListResponse(BaseModel):
    voices: List[Dict[str, Union[str, int, None]]]

# //////////////////////////////////////////////////////////////////////////////////////
# ///// Endpoints //////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////

@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    try:
        model = MODELS["transformer" if "transformer" in request.model else "hybrid"]
        speaking_rate = 15.0 * request.speed

        emotion_tensor = None
        if request.emotion:
            emotion_tensor = torch.tensor([
                request.emotion.get("happiness", 1.0),
                request.emotion.get("sadness", 0.05),
                request.emotion.get("disgust", 0.05),
                request.emotion.get("fear", 0.05),
                request.emotion.get("surprise", 0.05),
                request.emotion.get("anger", 0.05),
                request.emotion.get("other", 0.1),
                request.emotion.get("neutral", 0.2),
            ], device="cuda").unsqueeze(0)

        speaker_embedding = get_voice_embedding(request.voice) if request.voice else None
        if request.voice and speaker_embedding is None:
            raise HTTPException(status_code=404, detail=f"Voice '{request.voice}' not found.")

        cond_kwargs = {
            "text": request.input,
            "language": request.language,
            "speaker": speaker_embedding,
            "emotion": emotion_tensor,
            "speaking_rate": request.speaking_rate if request.speaking_rate is not None else speaking_rate,
            "device": "cuda",
            "unconditional_keys": [] if request.emotion else ["emotion"],
        }
        if request.pitch_std is not None:
            cond_kwargs["pitch_std"] = request.pitch_std
        if request.fmax is not None:
            cond_kwargs["fmax"] = request.fmax
        if request.vqscore_8 is not None:
            cond_kwargs["vqscore_8"] = request.vqscore_8
        if request.dnsmos_ovrl is not None:
            cond_kwargs["dnsmos_ovrl"] = request.dnsmos_ovrl
        if request.speaker_noised is not None:
            cond_kwargs["speaker_noised"] = request.speaker_noised

        cond_dict = make_cond_dict(**cond_kwargs)

        conditioning = model.prepare_conditioning(cond_dict)

        sampling_params = {
            k: v for k, v in {
                "top_k": request.top_k,
                "top_p": request.top_p,
                "min_p": request.min_p,
            }.items() if v is not None
        } or {"min_p": 0.15}

        codes = model.generate(
            prefix_conditioning=conditioning,
            max_new_tokens=86 * 30,
            cfg_scale=2.0,
            batch_size=1,
            sampling_params=sampling_params,
        )

        wav_out = model.autoencoder.decode(codes).cpu().detach()
        sr_out = model.autoencoder.sampling_rate

        if wav_out.dim() > 2:
            wav_out = wav_out.squeeze()
        if wav_out.dim() == 1:
            wav_out = wav_out.unsqueeze(0)

        buffer = BytesIO()
        torchaudio.save(buffer, wav_out, sr_out, format=request.response_format)
        buffer.seek(0)

        return StreamingResponse(buffer, media_type=f"audio/{request.response_format}")

    except Exception:
        logger.exception("Error during speech synthesis")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to generate speech",
                "traceback": traceback.format_exc(),
            },
        )

@app.post("/v1/audio/voice")
async def create_voice(file: UploadFile = File(...), name: Optional[str] = Form(None)):
    try:
        wav, sr = torchaudio.load(BytesIO(await file.read()))
        speaker_embedding = MODELS["transformer"].make_speaker_embedding(wav, sr)

        timestamp = int(time.time())
        voice_id = f"voice_{timestamp}_{uuid.uuid4().hex[:8]}"
        VOICE_CACHE[voice_id] = speaker_embedding.to("cuda")

        if not save_voice_embedding(voice_id, speaker_embedding.to("cuda")):
            raise HTTPException(status_code=500, detail="Failed to save voice embedding")

        metadata = load_voice_metadata()
        metadata[voice_id] = {"created": timestamp, "name": name}
        save_voice_metadata(metadata)

        return VoiceResponse(voice_id=voice_id, name=name, created=timestamp)

    except Exception:
        logger.exception("Error during voice creation")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to process uploaded voice",
                "traceback": traceback.format_exc(),
            },
        )

@app.get("/v1/audio/voices")
async def list_voices():
    metadata = load_voice_metadata()
    return VoiceListResponse(voices=[
        {"voice_id": vid, "name": info.get("name"), "created": info.get("created")}
        for vid, info in metadata.items()
    ])

@app.get("/v1/audio/models")
async def list_models():
    return {
        "models": [
            {
                "id": "Zyphra/Zonos-v0.1-transformer",
                "created": 1234567890,
                "object": "model",
                "owned_by": "zyphra"
            },
            {
                "id": "Zyphra/Zonos-v0.1-hybrid",
                "created": 1234567890,
                "object": "model",
                "owned_by": "zyphra"
            }
        ]
    }

@app.on_event("startup")
async def startup_event():
    load_models()
    load_voice_embeddings()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
