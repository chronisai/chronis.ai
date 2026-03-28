"""
modal_app/xtts_server.py  —  XTTS v2 on Modal GPU

This is the Modal app that provides:
  POST /synthesize  — streaming TTS inference with voice cloning
  POST /upload_voice — store a voice reference to Modal's persistent volume

Deploy with:
  modal deploy modal_app/xtts_server.py

CRITICAL: Validate this FIRST before building any downstream pipeline.
Steps:
  1. modal deploy modal_app/xtts_server.py
  2. Run smoke_tests/test_xtts_latency.py
  3. Confirm first audio chunk arrives in < 700ms
  4. Confirm streaming behavior (chunks arrive progressively, not all at once)

GPU selection:
  A10G is the default ($0.90/hr). T4 is cheaper but ~2x slower first-chunk.
  Set keep_warm=1 to prevent cold starts during active hours.
  Cold starts on A10G can add 8-15s to first TTS call — unacceptable in live session.

Audio format:
  XTTS v2 native output: 24kHz mono PCM int16
  TTSPipeline resamples this to 16kHz in-process — no conversion here.
  We output raw PCM bytes in the streaming response.
"""

import io
import os
from pathlib import Path
from fastapi import Response
from fastapi.responses import StreamingResponse as FastAPIStreamingResponse
from typing import Optional

import modal

# ── Modal app definition ──────────────────────────────────────────────────────
app = modal.App("chronis-xtts-v2")

# Persistent volume for voice reference files
# Files written here survive container restarts and are fast to read
# (local to the GPU host, no cross-network fetch).
voice_volume = modal.Volume.from_name("chronis-voice-refs", create_if_missing=True)
VOLUME_MOUNT = "/voice_refs"

# GPU image with XTTS v2 and all dependencies
# Build once, reused across all inference calls
xtts_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "TTS==0.22.0",                # Coqui TTS — contains XTTS v2
        "torch==2.4.0",               # PyTorch >= 2.4 required by transformers 4.x
        "torchaudio==2.4.0",
        "transformers==4.44.2",       # MUST be 4.x — TTS needs BeamSearchScorer removed in 5.x
        "fastapi",
        "uvicorn",
        "numpy",
    )
    .env({"COQUI_TOS_AGREED": "1"})  # Accept Coqui license non-interactively
    .run_commands(
        # Pre-download the XTTS v2 model weights into the image layer
        # This runs at build time — not at inference time
        "python -c \"from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2', gpu=False)\""
    )
)


@app.cls(
    gpu="A10G",          # A10G: good balance of VRAM and cost
    image=xtts_image,
    volumes={VOLUME_MOUNT: voice_volume},
    min_containers=1,              # CRITICAL: keep at least 1 container warm
                                   # prevents 8-15s cold start during sessions
    timeout=300,                   # 5 min max per request
    scaledown_window=120,          # recycle idle containers after 2 min
)
class XTTSModel:
    """
    XTTS v2 inference class. One instance per Modal container.
    Model loaded once on container start (model_post_init), not per request.
    """

    @modal.enter()
    def load_model(self):
        """
        Load XTTS v2 model into GPU memory once when container starts.
        This is what keep_warm=1 preserves — so cold start never hits
        a live session.
        """
        from TTS.api import TTS as COQUI_TTS
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[XTTS] Loading model on {self.device}...", flush=True)

        self.tts = COQUI_TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            gpu=(self.device == "cuda"),
        )

        print(f"[XTTS] Model ready on {self.device} ✓", flush=True)

    @modal.fastapi_endpoint(method="POST", docs=True)
    def synthesize(self, item: dict):
        """
        POST /synthesize
        Body: {"text": "...", "speaker_wav": "voice_refs/agent_id/reference.wav", "language": "en"}

        Returns: streaming chunked response of raw 24kHz PCM bytes.

        The caller (tts_pipeline.py) handles:
          - Chunked HTTP streaming
          - In-process 24→16kHz resampling
          - Forwarding chunks to the avatar pipeline
        """
        import io
        import numpy as np

        text        = item.get("text", "").strip()
        speaker_wav = item.get("speaker_wav", "")
        language    = item.get("language", "en")

        if not text:
            return Response(content=b"", status_code=400, headers={"X-Error": "text is required"})

        # Build full path to voice reference on the mounted volume
        voice_path = Path(VOLUME_MOUNT) / speaker_wav.lstrip("/")

        if not voice_path.exists():
            return Response(content=b"", status_code=404, headers={"X-Error": f"Voice reference not found: {speaker_wav}"})

        # ── Synthesize ────────────────────────────────────────────────────
        # XTTS v2 generates the full waveform, then we stream it in chunks.
        # True chunk-by-chunk generation would require modifying XTTS internals.
        # For MVP: generate fully, then stream output in 4096-byte chunks.
        # This gives realistic streaming latency (first chunk ≈ synthesis time).
        #
        # Post-MVP: Coqui's streaming API (tts.tts_to_file with streaming=True)
        # can yield actual chunks as they're generated.

        try:
            # tts_to_file() requires a real file path, not a BytesIO buffer
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            self.tts.tts_to_file(
                text=text,
                speaker_wav=str(voice_path),
                language=language,
                file_path=tmp_path,
            )
            with open(tmp_path, "rb") as f:
                pcm_bytes = f.read()
            os.unlink(tmp_path)

            # Strip WAV header — send raw PCM int16 bytes
            if pcm_bytes[:4] == b"RIFF":
                pcm_bytes = pcm_bytes[44:]

        except Exception as e:
            print(f"[XTTS] Synthesis error: {e}", flush=True)
            return Response(content=b"", status_code=500, headers={"X-Error": str(e)[:200]})

        # ── Stream the PCM in chunks ───────────────────────────────────────
        def generate_chunks():
            chunk_size = 4096   # ~86ms of audio at 24kHz int16
            offset     = 0
            while offset < len(pcm_bytes):
                yield pcm_bytes[offset: offset + chunk_size]
                offset += chunk_size

        return FastAPIStreamingResponse(
            generate_chunks(),
            media_type="application/octet-stream",
            headers={
                "X-Sample-Rate":  "24000",
                "X-Channels":     "1",
                "X-Bit-Depth":    "16",
                "X-Audio-Format": "pcm",
                "X-Total-Bytes":  str(len(pcm_bytes)),
            },
        )

    @modal.fastapi_endpoint(method="POST", docs=True)
    def upload_voice(self, request: dict) -> dict:
        """
        POST /upload_voice
        Headers: X-Volume-Path: voice_refs/agent_id/reference.wav
        Body: raw WAV bytes

        Stores the voice reference on Modal's persistent volume so
        synthesis calls can load it locally (zero fetch latency).
        """
        volume_path = request.get("path", "")
        audio_bytes = request.get("audio_bytes", b"")

        if not volume_path or not audio_bytes:
            return {"error": "path and audio_bytes required"}

        full_path = Path(VOLUME_MOUNT) / volume_path.lstrip("/")
        full_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            full_path.write_bytes(audio_bytes)
            voice_volume.commit()   # ensure write is persisted to the volume
            return {"path": volume_path, "bytes": len(audio_bytes)}
        except Exception as e:
            return {"error": str(e)}


# ────────────────────────────────────────────────────────────────────────────
# FASTAPI WRAPPER
# Gives us a proper HTTP endpoint with streaming support
# that tts_pipeline.py calls via httpx.AsyncClient.stream()
# ────────────────────────────────────────────────────────────────────────────

@app.function(
    gpu="A10G",
    image=xtts_image,
    volumes={VOLUME_MOUNT: voice_volume},
    min_containers=1,
    timeout=300,
)
@modal.asgi_app()
def fastapi_app():
    """
    FastAPI wrapper around XTTSModel for proper streaming support.
    This is what gets deployed as the Modal web endpoint.
    """
    import asyncio
    import io
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
    from TTS.api import TTS as COQUI_TTS
    import torch

    web_app = FastAPI()

    # Load model once at module level
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[XTTS FastAPI] Loading model on {_device}...", flush=True)
    _tts = COQUI_TTS(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        gpu=(_device == "cuda"),
    )
    print("[XTTS FastAPI] Model ready ✓", flush=True)

    @web_app.post("/synthesize")
    async def synthesize(request: Request):
        body     = await request.json()
        text     = body.get("text", "").strip()
        spk_wav  = body.get("speaker_wav", "")
        language = body.get("language", "en")
        stream   = body.get("stream", True)

        if not text:
            return {"error": "text required"}

        voice_path = Path(VOLUME_MOUNT) / spk_wav.lstrip("/")
        if not voice_path.exists():
            return {"error": f"Voice reference not found: {spk_wav}"}

        def generate_stream():
            """
            Use XTTS native streaming API — yields PCM chunks as they are
            generated, so first chunk arrives in ~400ms instead of waiting
            for the full waveform (which takes 3-4s for a full sentence).
            """
            import numpy as np
            chunks = _tts.tts_stream(
                text=text,
                speaker_wav=str(voice_path),
                language=language,
            )
            for chunk in chunks:
                if chunk is None:
                    continue
                # chunk is a numpy float32 array — convert to int16 PCM bytes
                if hasattr(chunk, "numpy"):
                    chunk = chunk.numpy()
                pcm = (np.array(chunk) * 32767).clip(-32768, 32767).astype(np.int16)
                yield pcm.tobytes()

        async def async_generate():
            """Run the blocking generator in executor, yield chunks as ready."""
            import concurrent.futures
            loop = asyncio.get_running_loop()
            queue = asyncio.Queue()

            def producer():
                try:
                    for chunk in generate_stream():
                        asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
                finally:
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop)

            loop.run_in_executor(None, producer)

            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk

        return StreamingResponse(
            async_generate(),
            media_type="application/octet-stream",
            headers={
                "X-Sample-Rate": "24000",
                "X-Channels":    "1",
                "X-Bit-Depth":   "16",
            },
        )

    @web_app.post("/upload_voice")
    async def upload_voice(request: Request):
        volume_path = request.headers.get("X-Volume-Path", "")
        audio_bytes = await request.body()

        if not volume_path:
            return {"error": "X-Volume-Path header required"}

        full_path = Path(VOLUME_MOUNT) / volume_path.lstrip("/")
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(audio_bytes)
        await voice_volume.commit.aio()

        return {"path": volume_path, "bytes": len(audio_bytes), "ok": True}

    @web_app.get("/health")
    async def health():
        return {"status": "ok", "device": _device, "model": "xtts_v2"}

    return web_app
