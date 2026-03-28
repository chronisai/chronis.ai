"""
onboarding/routes.py  —  Onboarding API Routes

FastAPI router that exposes the OnboardingPipeline over HTTP.

Routes:
  POST /api/v2/onboard/validate    — validate photo + audio before touching any API
  POST /api/v2/onboard/start       — kick off full onboarding (async, returns agent_id)
  GET  /api/v2/onboard/status/{agent_id}  — poll onboarding progress
  GET  /api/v2/onboard/stream/{agent_id} — SSE stream of progress events (optional)

Design:
  The Simli face creation takes 45-90 seconds. We handle this with:
    a) Immediate response with agent_id + status='creating'
    b) Frontend polls GET /status every 5 seconds
    c) OR frontend connects to SSE stream for real-time progress

  Idempotency:
    POST /start with the same user_id + agent_name returns the existing
    agent_id if one already exists in 'ready' state — no duplicate creation.
"""

import asyncio
import io
import json
import os
import tempfile
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from onboarding.onboarding import OnboardingPipeline
from services.supabase_client import get_supabase
from validators.photo_validator import get_photo_validator

router = APIRouter(prefix="/api/v2/onboard", tags=["onboarding"])

# Track active onboarding jobs: agent_id → list of progress messages
# Used by SSE stream endpoint
_onboarding_progress: dict[str, list] = {}


# ────────────────────────────────────────────────────────────────────────────
# POST /api/v2/onboard/validate
# ────────────────────────────────────────────────────────────────────────────

@router.post("/validate")
async def validate_inputs(
    photo: UploadFile = File(...),
    audio: UploadFile = File(...),
):
    """
    Validate photo and audio before burning any API credits.

    Run this on form submission — before Stripe, before Simli, before Modal.
    Returns detailed validation results with actionable rejection messages.

    Fast: photo validation ~50ms, audio validation ~2s.
    """
    # ── Photo validation ────────────────────────────────────────────────────
    photo_bytes = await photo.read()
    if len(photo_bytes) > 20 * 1024 * 1024:   # 20MB limit
        raise HTTPException(400, "Photo too large. Maximum 20MB.")

    photo_validator = get_photo_validator()
    photo_result    = photo_validator.validate(photo_bytes)

    if not photo_result["valid"]:
        return {
            "valid":       False,
            "failed_step": "photo",
            "reason":      photo_result["reason"],
        }

    # ── Audio validation ────────────────────────────────────────────────────
    audio_bytes = await audio.read()
    if len(audio_bytes) > 100 * 1024 * 1024:  # 100MB limit
        raise HTTPException(400, "Audio file too large. Maximum 100MB.")

    # Write to temp file for librosa
    suffix = "." + (audio.filename or "audio.wav").rsplit(".", 1)[-1].lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    audio_result = {"valid": False, "reason": "Validation did not run"}
    try:
        from validators.audio_validator import validate_audio
        audio_result = validate_audio(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        # Clean up converted file if validation passed
        if audio_result.get("valid") and audio_result.get("converted_path"):
            try:
                os.unlink(audio_result["converted_path"])
            except Exception:
                pass

    if not audio_result["valid"]:
        return {
            "valid":       False,
            "failed_step": "audio",
            "reason":      audio_result["reason"],
        }

    return {
        "valid": True,
        "photo": {
            "resolution": photo_result.get("resolution"),
            "sharpness":  photo_result.get("sharpness"),
            "yaw_deg":    photo_result.get("yaw_deg"),
            "pitch_deg":  photo_result.get("pitch_deg"),
        },
        "audio": {
            "duration_s":   audio_result.get("duration_s"),
            "snr_db":       audio_result.get("snr_db"),
            "speech_ratio": audio_result.get("speech_ratio"),
        },
    }


# ────────────────────────────────────────────────────────────────────────────
# POST /api/v2/onboard/start
# ────────────────────────────────────────────────────────────────────────────

@router.post("/start")
async def start_onboarding(
    user_id:     str        = Form(...),
    agent_name:  str        = Form(...),
    personality: str        = Form(...),
    photo:       UploadFile = File(...),
    audio:       UploadFile = File(...),
):
    """
    Start the full onboarding pipeline.

    Returns immediately with:
      {"agent_id": "...", "status": "creating"}

    Then poll GET /status/{agent_id} every 5 seconds.
    OR connect to GET /stream/{agent_id} for SSE progress.

    Idempotent: if an agent with the same name already exists for this user,
    returns the existing agent_id without re-creating.
    """
    db = get_supabase()

    # ── Idempotency check ──────────────────────────────────────────────────
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

    # ── Read uploaded files ────────────────────────────────────────────────
    photo_bytes = await photo.read()
    audio_bytes = await audio.read()

    # Write audio to temp file (audio_validator needs a path)
    audio_suffix = "." + (audio.filename or "audio.wav").rsplit(".", 1)[-1].lower()
    audio_tmp    = tempfile.NamedTemporaryFile(suffix=audio_suffix, delete=False)
    audio_tmp.write(audio_bytes)
    audio_tmp.close()
    audio_path = audio_tmp.name

    # ── Create agent record immediately (status=creating) ─────────────────
    # This gives us an agent_id to return right away
    agent_row = await db.insert("agents", {
        "user_id":     user_id,
        "name":        agent_name,
        "personality": personality,
        "status":      "creating",
    })
    if not agent_row:
        raise HTTPException(500, "Failed to create agent record")

    agent_id = agent_row["id"]

    # ── Initialize progress tracking ───────────────────────────────────────
    _onboarding_progress[agent_id] = []

    def on_progress(step: str, message: str):
        entry = {"step": step, "message": message}
        _onboarding_progress.setdefault(agent_id, []).append(entry)

    # ── Run pipeline in background ─────────────────────────────────────────
    async def run_pipeline():
        photo_validator = get_photo_validator()
        pipeline        = OnboardingPipeline(db=db, photo_validator=photo_validator)
        try:
            result = await pipeline.run(
                user_id=user_id,
                agent_name=agent_name,
                personality=personality,
                photo_bytes=photo_bytes,
                audio_path=audio_path,
                agent_id=agent_id,   # pin pipeline to the row we just created
                on_progress=on_progress,
            )
            if result.get("error"):
                on_progress("failed", result["error"])
                await db.update("agents", "id", agent_id, {"status": "failed"})
            else:
                on_progress("complete", "Onboarding complete ✓")
        except Exception as e:
            on_progress("failed", str(e))
            await db.update("agents", "id", agent_id, {"status": "failed"})
        finally:
            try:
                os.unlink(audio_path)
            except Exception:
                pass
            await pipeline.close()

    asyncio.create_task(run_pipeline())

    return {
        "agent_id": agent_id,
        "status":   "creating",
        "message":  "Onboarding started. Poll /status/{agent_id} for progress.",
    }


# ────────────────────────────────────────────────────────────────────────────
# GET /api/v2/onboard/status/{agent_id}
# ────────────────────────────────────────────────────────────────────────────

@router.get("/status/{agent_id}")
async def get_onboarding_status(agent_id: str):
    """
    Poll onboarding progress. Call every 5 seconds from the frontend.

    Returns:
      {"status": "creating", "progress": [...steps...]}
      {"status": "ready",    "simli_agent_id": "..."}
      {"status": "failed",   "error": "..."}
    """
    db    = get_supabase()
    agent = await db.get_agent(agent_id)

    if not agent:
        raise HTTPException(404, f"Agent {agent_id} not found")

    status   = agent.get("status", "creating")
    progress = _onboarding_progress.get(agent_id, [])

    if status == "ready":
        voices = agent.get("voices") or []
        return {
            "status":          "ready",
            "agent_id":        agent_id,
            "simli_agent_id":  agent.get("simli_agent_id"),
            "voice_ready":     bool(voices and voices[0].get("status") == "ready"),
            "progress":        progress,
        }

    elif status == "failed":
        # Find the failure message from progress log
        failure_msg = next(
            (p["message"] for p in reversed(progress) if p["step"] == "failed"),
            "Onboarding failed — check logs for details",
        )
        return {
            "status":   "failed",
            "agent_id": agent_id,
            "error":    failure_msg,
            "progress": progress,
        }

    else:
        # Still creating
        last_step = progress[-1] if progress else {"step": "starting", "message": "Initializing..."}
        return {
            "status":     "creating",
            "agent_id":   agent_id,
            "last_step":  last_step,
            "progress":   progress,
        }


# ────────────────────────────────────────────────────────────────────────────
# GET /api/v2/onboard/stream/{agent_id}
# ────────────────────────────────────────────────────────────────────────────

@router.get("/stream/{agent_id}")
async def stream_onboarding_progress(agent_id: str):
    """
    Server-Sent Events stream for real-time onboarding progress.

    Alternative to polling — connect once and receive progress events as they happen.
    The frontend progress bar updates in real-time without polling overhead.

    Usage (JavaScript):
        const es = new EventSource(`/api/v2/onboard/stream/${agentId}`);
        es.onmessage = (e) => {
            const {step, message, status} = JSON.parse(e.data);
            updateProgressBar(step, message);
            if (status === 'ready' || status === 'failed') es.close();
        };
    """
    db = get_supabase()

    async def event_generator() -> AsyncGenerator[str, None]:
        seen_count = 0
        timeout    = 200   # 200 × 1s = ~3.3 minutes max (Simli can take 90s)

        for _ in range(timeout):
            await asyncio.sleep(1.0)

            # Check DB for final status
            agent = await db.get_agent(agent_id)
            if not agent:
                yield f"data: {json.dumps({'status': 'error', 'message': 'Agent not found'})}\n\n"
                return

            # Send any new progress entries
            progress = _onboarding_progress.get(agent_id, [])
            for entry in progress[seen_count:]:
                yield f"data: {json.dumps(entry)}\n\n"
                seen_count = len(progress)

            status = agent.get("status", "creating")

            # Terminal states — send final event and close stream
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

        # Timed out — should not happen in normal operation
        yield f"data: {json.dumps({'step': 'timeout', 'status': 'failed', 'message': 'Onboarding timed out'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",   # disable nginx buffering
            "Access-Control-Allow-Origin": "*",
        },
    )


# ────────────────────────────────────────────────────────────────────────────
# GET /api/v2/agents (list user's agents)
# ────────────────────────────────────────────────────────────────────────────

@router.get("/agents/{user_id}")
async def list_agents(user_id: str):
    """List all ready agents for a user."""
    db   = get_supabase()
    rows = await db.select(
        "agents",
        f"user_id=eq.{user_id}&status=eq.ready&select=id,name,simli_agent_id,created_at,voices(status,duration_seconds)"
    )
    return {"agents": rows, "count": len(rows)}
