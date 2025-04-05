# //////////////////////////////////////////////////////////////////////////////////////
# ///// Setup Logger ///////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////

from pathlib import Path

from logger import set_logger_name

set_logger_name(logger_name=Path(__file__).stem)

# //////////////////////////////////////////////////////////////////////////////////////
# ///// Start Application //////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////

import traceback
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import torch
import torchaudio
import os
import json
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Union
from io import BytesIO
import numpy as np
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict, supported_language_codes
from fastapi.responses import StreamingResponse
import time
import uuid

app = FastAPI(title="Zonos API", description="OpenAI-compatible TTS API for Zonos")

# Model Management
MODELS = {
    "transformer": None,
    "hybrid": None
}

# Voice storage settings
VOICE_STORAGE_DIR = os.environ.get("VOICE_STORAGE_DIR", "data/voice_storage")
VOICE_METADATA_FILE = os.path.join(VOICE_STORAGE_DIR, "voice_metadata.json")
VOICE_CACHE: Dict[str, torch.Tensor] = {}

# Ensure voice storage directory exists
os.makedirs(VOICE_STORAGE_DIR, exist_ok=True)

def load_models():
    """Load both models at startup and keep them in VRAM"""
    device = "cuda"
    MODELS["transformer"] = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
    MODELS["transformer"].requires_grad_(False).eval()
    MODELS["hybrid"] = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)
    MODELS["hybrid"].requires_grad_(False).eval()

def load_voice_metadata():
    """Load voice metadata from disk"""
    if os.path.exists(VOICE_METADATA_FILE):
        try:
            with open(VOICE_METADATA_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading voice metadata: {e}")
    return {}

def save_voice_metadata(metadata):
    """Save voice metadata to disk"""
    try:
        with open(VOICE_METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"Error saving voice metadata: {e}")

def load_voice_embeddings():
    """Load all voice embeddings from disk into cache"""
    metadata = load_voice_metadata()
    for voice_id, voice_info in metadata.items():
        tensor_path = os.path.join(VOICE_STORAGE_DIR, f"{voice_id}.pt")
        if os.path.exists(tensor_path):
            try:
                VOICE_CACHE[voice_id] = torch.load(tensor_path, map_location="cuda")
                print(f"Loaded voice: {voice_info.get('name', voice_id)}")
            except Exception as e:
                print(f"Error loading voice embedding {voice_id}: {e}")

def save_voice_embedding(voice_id, embedding):
    """Save a voice embedding to disk"""
    tensor_path = os.path.join(VOICE_STORAGE_DIR, f"{voice_id}.pt")
    try:
        torch.save(embedding, tensor_path)
        return True
    except Exception as e:
        print(f"Error saving voice embedding {voice_id}: {e}")
        return False

def get_voice_embedding(voice_identifier):
    """Get voice embedding by ID or name"""
    # If it's a direct ID match, return it
    if voice_identifier in VOICE_CACHE:
        return VOICE_CACHE[voice_identifier]

    # Check if it's a name
    metadata = load_voice_metadata()
    for voice_id, info in metadata.items():
        if info.get("name") == voice_identifier:
            # If we have it in cache, return it
            if voice_id in VOICE_CACHE:
                return VOICE_CACHE[voice_id]

            # Otherwise try to load from disk
            tensor_path = os.path.join(VOICE_STORAGE_DIR, f"{voice_id}.pt")
            if os.path.exists(tensor_path):
                try:
                    embedding = torch.load(tensor_path, map_location="cuda")
                    VOICE_CACHE[voice_id] = embedding
                    return embedding
                except Exception as e:
                    print(f"Error loading voice {voice_id}: {e}")

    return None

# API Models
class SpeechRequest(BaseModel):
    model: str = Field("Zyphra/Zonos-v0.1-transformer", description="Model to use")
    input: str = Field(..., max_length=500, description="Text to synthesize")
    voice: Optional[str] = Field(None, description="Voice ID or name to use")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speaking speed multiplier")
    language: str = Field("en-us", description="Language code")
    emotion: Optional[Dict[str, float]] = None
    response_format: str = Field("mp3", description="Audio format (mp3 or wav)")
    prefix_audio: Optional[str] = Field(None, description="Voice ID or name to use as audio prefix")

    # New sampling parameters
    top_k: Optional[int] = Field(None, ge=1, description="Top-K sampling: Limits selection to K most likely tokens")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Top-P (nucleus) sampling: Dynamically limits token selection")
    min_p: Optional[float] = Field(0.15, ge=0.0, le=1.0, description="Min-P sampling: Excludes tokens below probability threshold")

class VoiceResponse(BaseModel):
    voice_id: str
    name: Optional[str]
    created: int  # Unix timestamp

class VoiceListResponse(BaseModel):
    voices: List[Dict[str, Union[str, int, None]]]

# API Endpoints
@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    try:
        model = MODELS["transformer" if "transformer" in request.model else "hybrid"]

        # Convert speed to speaking_rate (15.0 is default)
        speaking_rate = 15.0 * request.speed

        # Prepare emotion tensor if provided
        emotion_tensor = None
        if request.emotion:
            emotion_values = [
                request.emotion.get("happiness", 1.0),
                request.emotion.get("sadness", 0.05),
                request.emotion.get("disgust", 0.05),
                request.emotion.get("fear", 0.05),
                request.emotion.get("surprise", 0.05),
                request.emotion.get("anger", 0.05),
                request.emotion.get("other", 0.1),
                request.emotion.get("neutral", 0.2)
            ]
            emotion_tensor = torch.tensor(emotion_values, device="cuda").unsqueeze(0)

        # Get voice embedding from cache or by name
        speaker_embedding = None
        if request.voice:
            speaker_embedding = get_voice_embedding(request.voice)
            if speaker_embedding is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Voice '{request.voice}' not found. Please check voice ID or name."
                )

        # Default conditioning parameters
        cond_dict = make_cond_dict(
            text=request.input,
            language=request.language,
            speaker=speaker_embedding,
            emotion=emotion_tensor,
            speaking_rate=speaking_rate,
            device="cuda",
            unconditional_keys=[] if request.emotion else ["emotion"]
        )

        conditioning = model.prepare_conditioning(cond_dict)

        # Build sampling parameters dictionary
        sampling_params = {}

        # Add non-None parameters to the dictionary
        if request.min_p is not None:
            sampling_params['min_p'] = request.min_p
        if request.top_k is not None:
            sampling_params['top_k'] = request.top_k
        if request.top_p is not None:
            sampling_params['top_p'] = request.top_p

        # Use default min_p if no sampling parameters were provided
        if not sampling_params:
            sampling_params = dict(min_p=0.15)

        # Generate audio
        codes = model.generate(
            prefix_conditioning=conditioning,
            max_new_tokens=86 * 30,
            cfg_scale=2.0,
            batch_size=1,
            sampling_params=sampling_params
        )

        wav_out = model.autoencoder.decode(codes).cpu().detach()
        sr_out = model.autoencoder.sampling_rate

        # Ensure proper shape
        if wav_out.dim() > 2:
            wav_out = wav_out.squeeze()
        if wav_out.dim() == 1:
            wav_out = wav_out.unsqueeze(0)

        # Convert to requested format
        buffer = BytesIO()
        torchaudio.save(buffer, wav_out, sr_out, format=request.response_format)
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type=f"audio/{request.response_format}"
        )

    except Exception as e:
        # Fixed the logger issue
        print(f"Error in /v1/audio/speech: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/audio/voice")
async def create_voice(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None)
):
    try:
        # Read the audio file
        content = await file.read()
        audio_data = BytesIO(content)

        # Load and process audio
        wav, sr = torchaudio.load(audio_data)

        # Generate embedding using transformer model (handles GPU automatically)
        speaker_embedding = MODELS["transformer"].make_speaker_embedding(wav, sr)

        # Generate unique voice ID
        timestamp = int(time.time())
        voice_id = f"voice_{timestamp}_{str(uuid.uuid4())[:8]}"

        # Store embedding in memory cache
        VOICE_CACHE[voice_id] = speaker_embedding.to("cuda")

        # Save to persistent storage
        success = save_voice_embedding(voice_id, speaker_embedding.to("cuda"))
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to save voice embedding to disk"
            )

        # Update metadata
        metadata = load_voice_metadata()
        metadata[voice_id] = {
            "created": timestamp,
            "name": name,
        }
        save_voice_metadata(metadata)

        return VoiceResponse(
            voice_id=voice_id,
            name=name,
            created=timestamp
        )

    except Exception as e:
        error_msg = str(e)
        if "cuda" in error_msg.lower():
            error_msg = "GPU error while processing voice. Please try again."
        elif "load" in error_msg.lower():
            error_msg = "Failed to load audio file. Please ensure it's a valid audio format."

        raise HTTPException(
            status_code=500,
            detail=error_msg
        )

@app.get("/v1/audio/voices")
async def list_voices():
    """List all saved voices"""
    metadata = load_voice_metadata()
    voices = []

    for voice_id, info in metadata.items():
        voices.append({
            "voice_id": voice_id,
            "name": info.get("name"),
            "created": info.get("created")
        })

    return VoiceListResponse(voices=voices)

@app.get("/v1/audio/models")
async def list_models():
    """List available models and their status"""
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

# Load models at startup
@app.on_event("startup")
async def startup_event():
    load_models()
    load_voice_embeddings()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
