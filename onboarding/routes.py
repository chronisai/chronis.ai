"""
onboarding/routes.py  —  Onboarding API Routes

Routes:
  POST /api/v2/onboard/validate          — validate video before touching any API
  POST /api/v2/onboard/start             — kick off full onboarding (async)
  GET  /api/v2/onboard/status/{agent_id} — poll onboarding progress
  GET  /api/v2/onboard/stream/{agent_id} — SSE stream of progress events
  GET  /api/v2/onboard/agents/{user_id}  — list user's agents

Input: VIDEO FILE ONLY
  The frontend uploads one video. We extract:
    - Best face frame → photo for Simli
    - Audio track    → voice reference for XTTS

  This replaces the old photo + audio two-file flow.
"""

import asyncio
import io
import json
import os
import tempfile
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from onboarding.onboarding import OnboardingPipeline
from services.supabase_client import get_supabase
from validators.photo_validator import get_photo_validator
from utils.ffmpeg_utils import extract_best_frames, cleanup_frames, convert_audio_to_wav_16k

router = APIRouter(prefix="/api/v2/onboard", tags=["onboarding"])

_onboarding_progress: dict[str, list] = {}


# ── Video processing helper ───────────────────────────────────────────────────

def _extract_photo_and_audio_from_video(
    video_path: str,
) -> tuple[bytes | None, str | None, str | None]:
    """
    Given a video file path, extract:
      - The best face frame as JPEG bytes
      - The audio track as a 16kHz mono WAV temp file path

    Returns (photo_bytes, audio_path, error_message).
    photo_bytes is None on failure.
    audio_path must be cleaned up by caller.
    """
    # ── Extract frames ────────────────────────────────────────────────────
    frame_paths, err = extract_best_frames(video_path, n_frames=15)
    if err or not frame_paths:
        return None, None, f"Could not extract frames from video: {err or 'no frames found'}"

    # ── Pick best frame with face detection ───────────────────────────────
    validator   = get_photo_validator()
    best_bytes  = None
    best_score  = -1.0

    for frame_path in frame_paths:
        try:
            with open(frame_path, "rb") as f:
                frame_bytes = f.read()
            result = validator.validate(frame_bytes)
            if result["valid"]:
                score = float(result.get("sharpness", 0))
                if score > best_score:
                    best_score = score
                    best_bytes = frame_bytes
        except Exception:
            continue

    cleanup_frames(frame_paths)

    if best_bytes is None:
        return None, None, (
            "No clear face found in the video. "
            "Please upload a video where the person's face is visible and facing the camera."
        )

    # ── Extract audio track ───────────────────────────────────────────────
    fd, audio_path = tempfile.mkstemp(suffix="_voice.wav")
    os.close(fd)

    converted, err = convert_audio_to_wav_16k(video_path, audio_path)
    if err or not converted:
        try:
            os.unlink(audio_path)
        except Exception:
            pass
        return None, None, f"Could not extract audio from video: {err or 'unknown error'}"

    return best_bytes, audio_path, None


# ── POST /api/v2/onboard/validate ────────────────────────────────────────────

@router.post("/validate")
async def validate_inputs(video: UploadFile = File(...)):
    """
    Validate a video before burning any API credits.
    Checks that a clear face frame can be extracted and audio is present.
    Fast: runs in ~2-5 seconds.
    """
    video_bytes = await video.read()
    if len(video_bytes) > 500 * 1024 * 1024:   # 500MB limit
        raise HTTPException(400, "Video too large. Maximum 500MB.")

    # Write to temp file
    suffix = "." + (video.filename or "video.mp4").rsplit(".", 1)[-1].lower()
    fd, video_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, video_bytes)
        os.close(fd)

        photo_bytes, audio_path, err = _extract_photo_and_audio_from_video(video_path)
        if err:
            return {"valid": False, "failed_step": "video", "reason": err}

        # Clean up extracted audio
        try:
            os.unlink(audio_path)
        except Exception:
            pass

        # Validate the extracted photo one more time for stats
        validator = get_photo_validator()
        photo_result = validator.validate(photo_bytes)

        return {
            "valid": True,
            "photo": {
                "resolution": photo_result.get("resolution"),
                "sharpness":  photo_result.get("sharpness"),
            },
            "audio": {"extracted": True},
        }

    finally:
        try:
            os.unlink(video_path)
        except Exception:
            pass


# ── POST /api/v2/onboard/start ───────────────────────────────────────────────

@router.post("/start")
async def start_onboarding(
    user_id:     str        = Form(...),
    agent_name:  str        = Form(...),
    personality: str        = Form(...),
    video:       UploadFile = File(...),
):
    """
    Start full onboarding from a video file.
    Returns immediately with agent_id + status='creating'.
    Poll GET /status/{agent_id} every 5 seconds.
    """
    db = get_supabase()

    # ── Idempotency ────────────────────────────────────────────────────────
    existing = await db.select(
        "agents",
        f"user_id=eq.{user_id}&name=eq.{agent_name}&status=eq.ready&select=id,simli_agent_id"
    )
    if existing:
        return {
            "agent_id": existing[0]["id"],
            "status":   "ready",
            "message":  "Agent already exists — returning existing ID",
        }

    # ── Save video to temp file ───────────────────────────────────────────
    video_bytes = await video.read()
    suffix      = "." + (video.filename or "video.mp4").rsplit(".", 1)[-1].lower()
    fd, video_path = tempfile.mkstemp(suffix=suffix)
    os.write(fd, video_bytes)
    os.close(fd)

    # ── Create agent record ───────────────────────────────────────────────
    agent_row = await db.insert("agents", {
        "user_id":     user_id,
        "name":        agent_name,
        "personality": personality,
        "status":      "creating",
    })
    if not agent_row:
        try:
            os.unlink(video_path)
        except Exception:
            pass
        raise HTTPException(500, "Failed to create agent record")

    agent_id = agent_row["id"]
    _onboarding_progress[agent_id] = []

    def on_progress(step: str, message: str):
        _onboarding_progress.setdefault(agent_id, []).append(
            {"step": step, "message": message}
        )
        # Print every step so Render logs show exactly what's happening
        print(f"[Onboard:{agent_id[:8]}] [{step.upper()}] {message}", flush=True)

    # ── Run pipeline in background ────────────────────────────────────────
    async def run_pipeline():
        audio_path  = None
        photo_bytes = None
        try:
            on_progress("extract", "Extracting face and voice from video...")

            # Run blocking extraction in executor so we don't block the event loop
            loop = asyncio.get_running_loop()
            photo_bytes, audio_path, err = await loop.run_in_executor(
                None, _extract_photo_and_audio_from_video, video_path
            )

            if err:
                on_progress("failed", err)
                await db.update("agents", "id", agent_id, {"status": "failed"})
                return

            on_progress("extract", "Face and voice extracted ✓")

            photo_validator = get_photo_validator()
            pipeline        = OnboardingPipeline(db=db, photo_validator=photo_validator)
            try:
                result = await pipeline.run(
                    user_id=user_id,
                    agent_name=agent_name,
                    personality=personality,
                    photo_bytes=photo_bytes,
                    audio_path=audio_path,
                    agent_id=agent_id,
                    on_progress=on_progress,
                )
                if result.get("error"):
                    on_progress("failed", result["error"])
                    await db.update("agents", "id", agent_id, {"status": "failed"})
                else:
                    on_progress("complete", "Replica created ✓")
            finally:
                await pipeline.close()

        except Exception as e:
            on_progress("failed", str(e))
            await db.update("agents", "id", agent_id, {"status": "failed"})
        finally:
            for p in [video_path, audio_path]:
                if p:
                    try:
                        os.unlink(p)
                    except Exception:
                        pass

    asyncio.create_task(run_pipeline())

    return {
        "agent_id": agent_id,
        "status":   "creating",
        "message":  "Onboarding started. Poll /status/{agent_id} for progress.",
    }


# ── GET /api/v2/onboard/status/{agent_id} ────────────────────────────────────

@router.get("/status/{agent_id}")
async def get_onboarding_status(agent_id: str):
    db    = get_supabase()
    agent = await db.get_agent(agent_id)

    if not agent:
        raise HTTPException(404, f"Agent {agent_id} not found")

    status   = agent.get("status", "creating")
    progress = _onboarding_progress.get(agent_id, [])

    if status == "ready":
        voices = agent.get("voices") or []
        return {
            "status":         "ready",
            "agent_id":       agent_id,
            "simli_agent_id": agent.get("simli_agent_id"),
            "voice_ready":    bool(voices and voices[0].get("status") == "ready"),
            "progress":       progress,
        }
    elif status == "failed":
        failure_msg = next(
            (p["message"] for p in reversed(progress) if p["step"] == "failed"),
            "Onboarding failed — check logs",
        )
        return {
            "status":   "failed",
            "agent_id": agent_id,
            "error":    failure_msg,
            "progress": progress,
        }
    else:
        last_step = progress[-1] if progress else {"step": "starting", "message": "Initializing..."}
        return {
            "status":    "creating",
            "agent_id":  agent_id,
            "last_step": last_step,
            "progress":  progress,
        }


# ── GET /api/v2/onboard/stream/{agent_id} ────────────────────────────────────

@router.get("/stream/{agent_id}")
async def stream_onboarding_progress(agent_id: str):
    db = get_supabase()

    async def event_generator() -> AsyncGenerator[str, None]:
        seen_count = 0
        timeout    = 240

        for _ in range(timeout):
            await asyncio.sleep(1.0)

            agent = await db.get_agent(agent_id)
            if not agent:
                yield f"data: {json.dumps({'status': 'error', 'message': 'Agent not found'})}\n\n"
                return

            progress = _onboarding_progress.get(agent_id, [])
            for entry in progress[seen_count:]:
                yield f"data: {json.dumps(entry)}\n\n"
            seen_count = len(progress)

            status = agent.get("status", "creating")
            if status == "ready":
                voices = agent.get("voices") or []
                yield f"data: {json.dumps({'step': 'complete', 'status': 'ready', 'agent_id': agent_id, 'voice_ready': bool(voices and voices[0].get('status') == 'ready')})}\n\n"
                return
            elif status == "failed":
                failure_msg = next(
                    (p["message"] for p in reversed(progress) if p["step"] == "failed"),
                    "Onboarding failed",
                )
                yield f"data: {json.dumps({'step': 'failed', 'status': 'failed', 'message': failure_msg})}\n\n"
                return

        yield f"data: {json.dumps({'step': 'timeout', 'status': 'failed', 'message': 'Onboarding timed out'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ── GET /api/v2/onboard/agents/{user_id} ─────────────────────────────────────

@router.get("/agents/{user_id}")
async def list_agents(user_id: str):
    db   = get_supabase()
    rows = await db.select(
        "agents",
        f"user_id=eq.{user_id}&status=eq.ready&select=id,name,simli_agent_id,created_at,voices(status,duration_seconds)"
    )
    return {"agents": rows, "count": len(rows)}