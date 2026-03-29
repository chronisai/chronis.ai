"""
main_v2.py  —  Chronis V2  —  FastAPI Entry Point

Migrates all existing Flask routes to FastAPI.
Adds the full V2 live session pipeline:
  POST /api/v2/session/start    — create Daily room, open Simli WS, arm pipelines
  POST /api/v2/session/end      — graceful teardown
  POST /api/v2/session/heartbeat — watchdog reset
  WS   /ws/session/{session_id} — audio stream in, state events out

The WebSocket carries:
  Client → Server:  binary audio chunks (16kHz mono PCM from Daily.co)
  Client → Server:  JSON {"type":"heartbeat"}
  Server → Client:  JSON state updates, transcript partials, error messages

Session isolation:
  Each live session has its own: SessionController, EventBus, AudioPipeline,
  LLMPipeline, TTSPipeline, AvatarPipeline, DeepgramClient, SimliClient,
  MemoryService. NOTHING is shared between sessions. No globals except
  the Supabase client (which is thread/async-safe with pooling).
"""

import asyncio
import base64
import json
import os
import re
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import httpx
import requests
from services.razorpay_client import create_order, verify_payment, RAZORPAY_KEY_ID as _RZP_KEY_CONFIGURED
from fastapi import (
    BackgroundTasks, FastAPI, File, Form, HTTPException,
    Request, UploadFile, WebSocket, WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from gradio_client import Client, handle_file

# ── V2 pipeline imports ───────────────────────────────────────────────────────
from session.controller import SessionController, State
from session.event_bus import EventBus
from pipelines.audio_pipeline import AudioPipeline
from pipelines.llm_pipeline import LLMPipeline
from pipelines.tts_pipeline import TTSPipeline
from pipelines.avatar_pipeline import AvatarPipeline
from services.supabase_client import SupabaseClient, init_supabase, get_supabase
from services.simli_client import SimliClient
from services.deepgram_client import DeepgramClient
from services.memory_service import MemoryService
from services.daily_client import DailyClient, get_daily_client
from utils.sentence_chunker import SentenceChunker
from onboarding.routes import router as onboarding_router

print("🚀 Chronis V2 booting...", flush=True)

app = FastAPI(title="Chronis V2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register onboarding routes (/api/v2/onboard/*)
app.include_router(onboarding_router)

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY       = os.environ.get("GROQ_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
RESEND_API_KEY     = os.environ.get("RESEND_API_KEY", "")
NOTIFY_EMAIL       = os.environ.get("NOTIFY_EMAIL", "")
ADMIN_SECRET       = os.environ.get("ADMIN_SECRET", "")
SUPABASE_URL       = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY       = os.environ.get("SUPABASE_SERVICE_KEY", "")
# Razorpay keys are read inside services/razorpay_client.py
# _RZP_KEY_CONFIGURED is used locally only for the "not configured" guard
# RAZORPAY_KEY_SECRET is validated inside verify_payment() — not needed here
SITE_URL           = os.environ.get("SITE_URL", "https://chronis.in")
XTTS_SPACE_URL     = os.environ.get("XTTS_SPACE_URL", "")
XTTS_SECRET        = os.environ.get("XTTS_SECRET", "")
SIMLI_API_KEY      = os.environ.get("SIMLI_API_KEY", "")
DEEPGRAM_API_KEY   = os.environ.get("DEEPGRAM_API_KEY", "")
DAILY_API_KEY      = os.environ.get("DAILY_API_KEY", "")
MODAL_XTTS_URL     = os.environ.get("MODAL_XTTS_URL", "")

FREE_MSG_LIMIT = 9999  # Payments disabled — free access for all
WATCHDOG_TIMEOUT_S = 90.0    # kill session after 90s of no audio + no heartbeat

UPLOAD_FOLDER  = tempfile.gettempdir()

# ── Live session registry ────────────────────────────────────────────────────
# Maps session_id → dict of live pipeline objects
# Only populated while a session is actively running
_live_sessions: Dict[str, dict] = {}

# ── Old XTTS HuggingFace client (for demo /api/speak) ────────────────────────
xtts_client     = None
xtts_client_url = None

def get_xtts_client():
    global xtts_client, xtts_client_url
    if not XTTS_SPACE_URL:
        raise RuntimeError("XTTS_SPACE_URL missing")
    if xtts_client is None or xtts_client_url != XTTS_SPACE_URL:
        print("Initializing XTTS client...", flush=True)
        xtts_client = Client(XTTS_SPACE_URL, httpx_kwargs={"timeout": 900})
        xtts_client_url = XTTS_SPACE_URL
    return xtts_client

# ── Startup / Shutdown ────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Initialize Supabase, start log worker, recover stale jobs."""
    db = init_supabase()
    # Start the fire-and-forget session_events log worker
    asyncio.create_task(_log_worker(db))
    # Recover any stale onboarding jobs from previous crash (fix #6)
    count = await db.recover_stale_onboarding_jobs()
    if count:
        print(f"[Startup] Recovered {count} stale onboarding jobs", flush=True)
    print("✅ Chronis V2 started", flush=True)

@app.on_event("shutdown")
async def shutdown():
    """Kill all active sessions gracefully."""
    print("[Shutdown] Terminating all live sessions...", flush=True)
    for session_id, session in list(_live_sessions.items()):
        ctrl: SessionController = session.get("ctrl")
        if ctrl:
            await ctrl.kill()
    await get_supabase().close()

# ── Global log queue (filled by SessionControllers, drained here) ──────────────

# Single shared log queue — all session controllers push here
# The _log_worker drains this and writes to session_events table
_global_log_q: asyncio.Queue = asyncio.Queue(maxsize=5000)

async def _log_worker(db: SupabaseClient) -> None:
    """
    Background worker that drains the log queue and writes to Supabase.
    Fire-and-forget — pipelines never wait for this.
    Batches writes when possible to reduce DB round-trips.
    """
    print("[LogWorker] Started ✓", flush=True)
    batch = []

    while True:
        try:
            # Collect up to 10 entries or wait up to 500ms
            try:
                entry = await asyncio.wait_for(_global_log_q.get(), timeout=0.5)
                batch.append(entry)
                # Drain any additional entries that are already ready
                while not _global_log_q.empty() and len(batch) < 10:
                    try:
                        batch.append(_global_log_q.get_nowait())
                    except asyncio.QueueEmpty:
                        break
            except asyncio.TimeoutError:
                pass  # no entries — flush batch anyway

            if batch:
                for entry in batch:
                    try:
                        await db.log_event(
                            session_id=entry["session_id"],
                            event_type=entry["event_type"],
                            pipeline=entry.get("pipeline", ""),
                            payload=entry.get("payload"),
                        )
                    except Exception as e:
                        print(f"[LogWorker] Write error: {e}", flush=True)
                batch.clear()

        except Exception as e:
            print(f"[LogWorker] Fatal error: {e}", flush=True)
            await asyncio.sleep(1)

# ────────────────────────────────────────────────────────────────────────────
# DAILY.CO HELPERS
# ────────────────────────────────────────────────────────────────────────────

async def create_daily_room(session_id: str) -> str:
    """
    Create a Daily.co room with EC + NS + AGC enabled.
    Room expires after 2 hours.
    Returns the room URL.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(
            "https://api.daily.co/v1/rooms",
            headers={
                "Authorization": f"Bearer {DAILY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "name": f"chronis-{session_id[:12]}",
                "properties": {
                    "exp": int(time.time()) + 7200,   # 2 hour expiry
                    "enable_noise_cancellation_ui": True,
                    "start_audio_off": False,
                    "start_video_off": True,
                    # Enable Daily's built-in audio processing
                    "audio": {
                        "echo_cancellation":      True,
                        "noise_suppression":      True,
                        "auto_gain_control":      True,
                    },
                },
            },
        )
        if not r.is_success:
            raise RuntimeError(f"Daily room creation failed: {r.status_code} {r.text[:200]}")
        return r.json()["url"]

async def delete_daily_room(room_url: str) -> None:
    """Delete a Daily.co room on session teardown."""
    room_name = room_url.split("/")[-1]
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            await client.delete(
                f"https://api.daily.co/v1/rooms/{room_name}",
                headers={"Authorization": f"Bearer {DAILY_API_KEY}"},
            )
        except Exception as e:
            print(f"[Daily] Room delete error: {e}", flush=True)

# ────────────────────────────────────────────────────────────────────────────
# V2 LIVE SESSION ROUTES
# ────────────────────────────────────────────────────────────────────────────

@app.post("/api/v2/session/start")
async def v2_session_start(request: Request):
    """
    Start a live V2 session.

    1. Verify agent exists and is ready
    2. Create Daily.co room
    3. Open Simli WebSocket (stays open entire session)
    4. Open Deepgram WebSocket (stays open entire session)
    5. Arm all pipelines
    6. Start watchdog
    7. Return room_url to frontend
    """
    db = get_supabase()
    body = await request.json()
    agent_id = body.get("agent_id", "").strip()

    if not agent_id:
        raise HTTPException(400, "agent_id required")

    # ── Verify agent ───────────────────────────────────────────────────────
    agent = await db.get_agent(agent_id)
    if not agent:
        raise HTTPException(404, "Agent not found")
    if agent.get("status") != "ready":
        raise HTTPException(400, f"Agent not ready (status={agent.get('status')})")

    voices = agent.get("voices") or []
    if not voices or voices[0].get("status") != "ready":
        raise HTTPException(400, "Voice reference not ready")

    voice_ref = voices[0].get("modal_voice_ref", "")
    simli_face_id = agent.get("simli_agent_id", "")

    if not simli_face_id:
        raise HTTPException(400, "Simli face model not created for this agent")

    # ── Create session record ─────────────────────────────────────────────
    session_id = str(uuid.uuid4())

    # ── Create Daily room via DailyClient ────────────────────────────────
    daily = get_daily_client()
    try:
        daily_room_url = await daily.create_room(session_id)
    except Exception as e:
        raise HTTPException(500, f"Daily room creation failed: {e}")

    # ── Write session to DB ────────────────────────────────────────────────
    await db.create_session({
        "id":             session_id,
        "user_id":        body.get("user_id"),
        "agent_id":       agent_id,
        "daily_room_url": daily_room_url,
        "state":          "idle",
        "turn_count":     0,
    })

    # ── Build per-session pipeline stack ──────────────────────────────────
    bus  = EventBus()
    ctrl = SessionController(session_id=session_id, event_bus=bus)
    ctrl.agent_id       = agent_id
    ctrl.voice_ref      = voice_ref
    ctrl.simli_agent_id = simli_face_id

    # Wire ctrl log queue to the global log queue
    # (SessionController._enqueue_log pushes to ctrl._log_q;
    #  we drain that into the global queue via the bridge task)
    asyncio.create_task(_bridge_session_logs(ctrl, _global_log_q))

    # ── Open Simli WebSocket ───────────────────────────────────────────────
    simli = SimliClient(face_id=simli_face_id, daily_room_url=daily_room_url)
    try:
        await simli.start()
    except Exception as e:
        await delete_daily_room(daily_room_url)
        raise HTTPException(500, f"Simli connection failed: {e}")

    # ── Memory service ─────────────────────────────────────────────────────
    memory = MemoryService(agent_id=agent_id, db=db)
    await memory.load()

    # Seed identity memory if this agent has never had a session before.
    # Without this, the agent has NO personality — it speaks as a generic LLM.
    # The personality field on the agent row is the source of truth.
    # We write it once as type='identity' in the memories table; every
    # subsequent session loads it from there via memory.load().
    if not memory._identity_prompt:
        personality = agent.get("personality", "")
        agent_name  = agent.get("name", "this person")
        if personality:
            identity_text = (
                f"You ARE {agent_name}. Not an AI. Not a simulation. You ARE this person.\n\n"
                f"{personality}\n\n"
                f"ABSOLUTE RULES:\n"
                f"- Speak ONLY in first person as {agent_name}\n"
                f"- Use exact vocabulary, slang, phrases, and speech patterns described above\n"
                f"- Match energy level, humor, and personality exactly\n"
                f"- Reference specific memories, people, and events naturally\n"
                f"- NEVER say you are an AI, a chatbot, or Chronis\n"
                f"- NEVER say 'based on the transcript' or 'based on the recording'\n"
                f"- Keep responses conversational and short — not essay-like\n"
                f"- If asked something you don't know, respond as {agent_name} would\n"
                f"You are {agent_name}. Start talking."
            )
            await db.save_memory({
                "agent_id":   agent_id,
                "type":       "identity",
                "content":    identity_text,
                "turn_index": 0,
            })
            memory._identity_prompt = identity_text
            print(f"[Session] Identity memory seeded for agent {agent_id[:8]}", flush=True)

    # ── Sentence chunker ───────────────────────────────────────────────────
    chunker = SentenceChunker(bus=bus, session_id=session_id)

    # ── Deepgram client ─────────────────────────────────────────────────────
    # on_final triggers the LLM pipeline via event bus
    async def _on_final(text: str):
        if text.strip() and ctrl.state == State.LISTENING:
            await bus.emit("stt.utterance_complete", {"text": text})

    async def _on_partial(text: str):
        # Partial transcripts forwarded to WebSocket subscribers for display
        session_obj = _live_sessions.get(session_id)
        if session_obj and session_obj.get("ws"):
            try:
                await session_obj["ws"].send_json({
                    "type":    "partial_transcript",
                    "text":    text,
                    "state":   ctrl.state.value,
                })
            except Exception:
                pass

    deepgram = DeepgramClient(on_partial=_on_partial, on_final=_on_final)
    try:
        await deepgram.start()
    except Exception as e:
        await simli.stop()
        await delete_daily_room(daily_room_url)
        raise HTTPException(500, f"Deepgram connection failed: {e}")

    # ── Assemble pipelines ─────────────────────────────────────────────────
    audio_pipeline  = AudioPipeline(ctrl=ctrl, bus=bus, deepgram=deepgram)
    llm_pipeline    = LLMPipeline(ctrl=ctrl, bus=bus, chunker=chunker, memory=memory)
    tts_pipeline    = TTSPipeline(ctrl=ctrl, bus=bus, voice_ref_path=voice_ref)
    avatar_pipeline = AvatarPipeline(ctrl=ctrl, bus=bus, simli=simli)

    # ── Start background pipeline tasks ───────────────────────────────────
    llm_pipeline.start()
    tts_pipeline.start()
    avatar_pipeline.start()

    # ── Transition to LISTENING ────────────────────────────────────────────
    await ctrl.transition(State.LISTENING)

    # ── Start watchdog ─────────────────────────────────────────────────────
    watchdog_task = asyncio.create_task(
        _watchdog(ctrl, session_id, daily_room_url, db)
    )

    # ── Register in live sessions ──────────────────────────────────────────
    _live_sessions[session_id] = {
        "ctrl":           ctrl,
        "bus":            bus,
        "audio_pipeline": audio_pipeline,
        "llm_pipeline":   llm_pipeline,
        "tts_pipeline":   tts_pipeline,
        "avatar_pipeline": avatar_pipeline,
        "deepgram":       deepgram,
        "simli":          simli,
        "memory":         memory,
        "daily_room_url": daily_room_url,
        "watchdog_task":  watchdog_task,
        "ws":             None,   # WebSocket reference, filled in /ws handler
    }

    print(f"[Session] Started {session_id[:8]} — room: {daily_room_url}", flush=True)

    return {
        "session_id":     session_id,
        "daily_room_url": daily_room_url,
        "state":          ctrl.state.value,
    }


@app.post("/api/v2/session/end")
async def v2_session_end(request: Request):
    """Graceful session teardown from the frontend."""
    body       = await request.json()
    session_id = body.get("session_id", "")

    if session_id not in _live_sessions:
        return {"ok": True, "note": "session not active"}

    await _cleanup_session(session_id)
    return {"ok": True}


@app.post("/api/v2/session/heartbeat")
async def v2_heartbeat(request: Request):
    """
    Frontend sends this every 30 seconds to reset the watchdog.
    If the tab closes, the heartbeat stops — watchdog fires after 90s.
    """
    body       = await request.json()
    session_id = body.get("session_id", "")

    session = _live_sessions.get(session_id)
    if session:
        session["ctrl"].touch()
    return {"ok": True}


# ── WebSocket — audio stream in, state updates out ─────────────────────────────

@app.websocket("/ws/session/{session_id}")
async def ws_session(websocket: WebSocket, session_id: str):
    """
    Main real-time channel for a live session.

    Binary messages:  16kHz mono PCM audio from Daily.co (via browser)
    JSON messages:    {"type": "heartbeat"}

    Server sends:
      {"type": "state", "state": "listening"|"thinking"|"speaking"|...}
      {"type": "partial_transcript", "text": "..."}
      {"type": "error", "message": "..."}
      {"type": "session_end"}
    """
    await websocket.accept()

    session = _live_sessions.get(session_id)
    if not session:
        await websocket.send_json({"type": "error", "message": "Session not found or not started"})
        await websocket.close()
        return

    ctrl: SessionController        = session["ctrl"]
    audio_pipeline: AudioPipeline  = session["audio_pipeline"]

    # Register this WebSocket so other parts of the pipeline can send state updates
    session["ws"] = websocket

    # Subscribe to state changes and relay them to the WebSocket
    state_relay_task = asyncio.create_task(
        _state_relay(websocket, ctrl, session_id)
    )

    print(f"[WS] Client connected to session {session_id[:8]}", flush=True)

    try:
        while not ctrl.dead.is_set():
            try:
                # Receive with timeout to allow checking ctrl.dead periodically
                message = await asyncio.wait_for(websocket.receive(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                break

            if message["type"] == "websocket.disconnect":
                break

            # ── Binary audio chunk ─────────────────────────────────────────
            if "bytes" in message and message["bytes"]:
                await audio_pipeline.push(message["bytes"])

            # ── JSON control message ───────────────────────────────────────
            elif "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                    msg_type = data.get("type", "")

                    if msg_type == "heartbeat":
                        ctrl.touch()

                    elif msg_type == "end_session":
                        break

                except (json.JSONDecodeError, KeyError):
                    pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS] Error in session {session_id[:8]}: {e}", flush=True)
    finally:
        state_relay_task.cancel()
        session["ws"] = None
        print(f"[WS] Client disconnected from session {session_id[:8]}", flush=True)
        # Don't clean up the session — watchdog handles that if needed
        # Frontend may reconnect (e.g. temporary network drop)


async def _state_relay(
    websocket: WebSocket,
    ctrl: SessionController,
    session_id: str,
) -> None:
    """
    Subscribe to state changes from the EventBus and forward them
    to the WebSocket client.
    Also sends periodic state pings so the frontend stays in sync.
    """
    bus = ctrl.bus

    # We relay state by polling ctrl.state rather than subscribing to events
    # because transitions happen too fast to reliably capture via queue
    last_state = None

    try:
        while not ctrl.dead.is_set():
            await asyncio.sleep(0.1)   # check every 100ms

            current = ctrl.state.value
            if current != last_state:
                try:
                    await websocket.send_json({
                        "type":  "state",
                        "state": current,
                    })
                    last_state = current
                except Exception:
                    break

    except asyncio.CancelledError:
        pass

    # Notify frontend that session ended
    try:
        await websocket.send_json({"type": "session_end"})
    except Exception:
        pass


# ────────────────────────────────────────────────────────────────────────────
# SESSION CLEANUP AND WATCHDOG
# ────────────────────────────────────────────────────────────────────────────

async def _cleanup_session(session_id: str) -> None:
    """
    Full session teardown in the correct order:
    1. Kill controller (sets dead event, emits session.end)
    2. Stop pipelines (avatar.stop() → simli.stop())
    3. Stop Deepgram
    4. Delete Daily room
    5. Write ended_at to Supabase
    6. Remove from live sessions registry
    """
    session = _live_sessions.pop(session_id, None)
    if not session:
        return

    ctrl: SessionController = session["ctrl"]
    db = get_supabase()

    print(f"[Session] Cleaning up {session_id[:8]}...", flush=True)

    # Step 1: Kill controller — all pipelines will exit their loops
    await ctrl.kill()

    # Small delay to let pipeline loops process the kill signal
    await asyncio.sleep(0.2)

    # Step 2: Stop pipelines explicitly (belt and suspenders)
    try:
        await session["avatar_pipeline"].stop()
    except Exception:
        pass
    try:
        await session["tts_pipeline"].stop()
    except Exception:
        pass
    try:
        await session["llm_pipeline"].stop()
    except Exception:
        pass

    # Step 3: Stop Deepgram WebSocket
    try:
        await session["deepgram"].stop()
    except Exception:
        pass

    # Step 4: Close memory service HTTP client
    try:
        await session["memory"].close()
    except Exception:
        pass

    # Step 5: Delete Daily room
    try:
        await get_daily_client().delete_room(session["daily_room_url"])
    except Exception:
        pass

    # Step 6: Cancel watchdog task
    wt = session.get("watchdog_task")
    if wt and not wt.done():
        wt.cancel()

    # Step 7: Write ended_at to Supabase
    try:
        await db.end_session(session_id)
    except Exception:
        pass

    # Step 8: Unsubscribe all bus queues
    ctrl.bus.unsubscribe_all()

    duration = time.monotonic() - ctrl.session_start
    print(f"[Session] {session_id[:8]} ended. Duration: {duration:.0f}s", flush=True)


async def _watchdog(
    ctrl: SessionController,
    session_id: str,
    daily_room_url: str,
    db: SupabaseClient,
) -> None:
    """
    Fires session cleanup if WATCHDOG_TIMEOUT_S passes with no audio
    activity AND no heartbeat.

    Two independent signals prevent premature timeout:
      - Audio frames update ctrl.last_activity
      - Heartbeat pings update ctrl.last_activity
    Both must be absent for WATCHDOG_TIMEOUT_S seconds to trigger.
    """
    try:
        while not ctrl.dead.is_set():
            await asyncio.sleep(10.0)   # check every 10s
            if ctrl.dead.is_set():
                break
            idle = ctrl.idle_seconds()
            if idle > WATCHDOG_TIMEOUT_S:
                print(f"[Watchdog] Session {session_id[:8]} idle {idle:.0f}s — killing",
                      flush=True)
                await _cleanup_session(session_id)
                break
    except asyncio.CancelledError:
        pass


async def _bridge_session_logs(
    ctrl: SessionController,
    global_q: asyncio.Queue,
) -> None:
    """
    Drain the session controller's log queue and push entries to the
    global log queue that the log worker reads.
    Runs as a background task for the lifetime of the session.
    """
    try:
        while not ctrl.dead.is_set():
            try:
                entry = await asyncio.wait_for(ctrl.log_queue.get(), timeout=1.0)
                try:
                    global_q.put_nowait(entry)
                except asyncio.QueueFull:
                    pass  # global queue full — drop the log entry
            except asyncio.TimeoutError:
                continue
    except asyncio.CancelledError:
        pass
    # Drain any remaining entries after session ends
    while not ctrl.log_queue.empty():
        try:
            entry = ctrl.log_queue.get_nowait()
            global_q.put_nowait(entry)
        except (asyncio.QueueEmpty, asyncio.QueueFull):
            break


# ────────────────────────────────────────────────────────────────────────────
# SUPABASE HELPERS (migrated from app.py)
# ────────────────────────────────────────────────────────────────────────────

def _sb_headers(extra=None):
    h = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    if extra:
        h.update(extra)
    return h

def sb_select(table, query=""):
    try:
        r = requests.get(f"{SUPABASE_URL}/rest/v1/{table}?{query}",
                         headers=_sb_headers(), timeout=10)
        return r.json() if r.ok else []
    except Exception as e:
        print(f"[SB select error] {e}", flush=True)
        return []

def sb_insert(table, data):
    try:
        r = requests.post(f"{SUPABASE_URL}/rest/v1/{table}",
                          headers=_sb_headers(), json=data, timeout=10)
        rows = r.json()
        return rows[0] if r.ok and rows else None
    except Exception as e:
        print(f"[SB insert error] {e}", flush=True)
        return None

def sb_update(table, match_col, match_val, data):
    try:
        r = requests.patch(
            f"{SUPABASE_URL}/rest/v1/{table}?{match_col}=eq.{match_val}",
            headers=_sb_headers(), json=data, timeout=10,
        )
        return r.ok
    except Exception as e:
        print(f"[SB update error] {e}", flush=True)
        return False

def sb_count(table, query=""):
    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/{table}?select=id&{query}",
            headers=_sb_headers({"Prefer": "count=exact"}), timeout=10,
        )
        return int(r.headers.get("Content-Range", "0/0").split("/")[-1])
    except Exception as e:
        print(f"[SB count error] {e}", flush=True)
        return 0

WAITLIST_BASELINE = 93

def get_waitlist_count():
    return max(sb_count("waitlist") + WAITLIST_BASELINE, WAITLIST_BASELINE)

def email_exists(email):
    return len(sb_select("waitlist", f"email=eq.{requests.utils.quote(email)}&select=id")) > 0

def get_session_v1(session_id):
    rows = sb_select("sessions", f"session_id=eq.{session_id}&select=*")
    return rows[0] if rows else None

def update_session_messages(session_id, messages):
    sb_update("sessions", "session_id", session_id, {"messages": messages})

def unlock_session_v1(session_id, unlock_type):
    sb_update("sessions", "session_id", session_id,
              {"unlocked": True, "unlock_type": unlock_type})

# ────────────────────────────────────────────────────────────────────────────
# GROQ HELPERS (migrated from app.py)
# ────────────────────────────────────────────────────────────────────────────

def transcribe_groq(file_path, filename):
    if not GROQ_API_KEY:
        return None, "GROQ_API_KEY not set"
    mime_map = {
        "mp4": "video/mp4", "mov": "video/quicktime", "avi": "video/x-msvideo",
        "webm": "video/webm", "mkv": "video/x-matroska",
        "mp3": "audio/mpeg", "wav": "audio/wav", "m4a": "audio/mp4",
        "ogg": "audio/ogg", "flac": "audio/flac", "aac": "audio/aac",
    }
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    mime = mime_map.get(ext, "application/octet-stream")
    try:
        with open(file_path, "rb") as f:
            resp = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": (filename, f, mime)},
                data={"model": "whisper-large-v3", "response_format": "verbose_json",
                      "language": "en"},
                timeout=180,
            )
        data = resp.json()
        if resp.ok:
            return data.get("text", ""), None
        return None, data.get("error", {}).get("message", "Groq transcription error")
    except Exception as e:
        return None, str(e)

def chat_as_persona_groq(history, person_name, persona):
    if not GROQ_API_KEY:
        return None, "GROQ_API_KEY not set"
    system_prompt = f"""You ARE {person_name}. Not an AI. Not a simulation.
{persona}
RULES: Speak ONLY in first person. Never say you're an AI or simulation.
Keep responses conversational and short. Use their actual speech patterns."""
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-20:]:
        messages.append({
            "role": "user" if msg["role"] == "user" else "assistant",
            "content": msg["content"],
        })
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "messages": messages,
                  "temperature": 0.9, "max_tokens": 512},
            timeout=30,
        )
        data = resp.json()
        if resp.ok:
            return data["choices"][0]["message"]["content"], None
        return None, data.get("error", {}).get("message", "Groq chat error")
    except Exception as e:
        return None, str(e)

def build_profile_from_transcript(transcript, person_name):
    if not GROQ_API_KEY:
        return None, "GROQ_API_KEY not set"
    prompt = f"""You are analyzing a transcript to build a memory profile of {person_name}.
TRANSCRIPT: \"\"\"{transcript}\"\"\"
Extract: voice/speech patterns, personality, memories, memorable details.
Write as if briefing someone who needs to perfectly impersonate {person_name}. Be specific."""
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.8, "max_tokens": 2048},
            timeout=120,
        )
        data = resp.json()
        if resp.ok:
            return data["choices"][0]["message"]["content"], None
        return None, data.get("error", {}).get("message", "Groq LLM error")
    except Exception as e:
        return None, str(e)

def build_persona(person_name, profile):
    return f"""You ARE {person_name}. Not an AI. Not a simulation.
{profile}
RULES: Speak ONLY in first person. Never say you're an AI or Chronis.
Keep responses conversational, natural, and short."""

def _clean_for_xtts(text):
    for bad, good in {"\u2018": "'", "\u2019": "'", "\u201c": '"',
                      "\u201d": '"', "\u2013": "-", "\u2014": "-",
                      "\u2026": "..."}.items():
        text = text.replace(bad, good)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 200:
        text = text[:200]
    return text if len(text) > 3 else "I am here with you."

# ────────────────────────────────────────────────────────────────────────────
# STATIC FILES
# ────────────────────────────────────────────────────────────────────────────

try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    pass   # static dir may not exist in test environments

@app.get("/")
async def index():
    return FileResponse("static/index.html")

@app.get("/demo")
async def demo():
    return FileResponse("static/demo.html")

@app.get("/thankyou")
async def thankyou():
    return FileResponse("static/thankyou.html")

@app.get("/admin")
async def admin():
    return FileResponse("static/admin.html")

@app.get("/live")
async def live():
    return FileResponse("static/live.html")

@app.get("/health")
async def health():
    return {"status": "ok", "ts": datetime.utcnow().isoformat(),
            "live_sessions": len(_live_sessions)}

# ────────────────────────────────────────────────────────────────────────────
# WAITLIST ROUTES (migrated from app.py)
# ────────────────────────────────────────────────────────────────────────────

@app.get("/api/count")
async def api_count():
    return {"count": get_waitlist_count()}

@app.post("/api/join")
async def api_join(request: Request):
    data    = await request.json()
    name    = (data.get("name") or "").strip()
    email   = (data.get("email") or "").strip().lower()
    country = (data.get("country") or "").strip()

    if not name or not email or not country:
        raise HTTPException(400, "All fields are required.")
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(400, "Please enter a valid email address.")
    if email_exists(email):
        raise HTTPException(400, "This email is already on the waitlist.")

    position = get_waitlist_count() + 1
    sb_insert("waitlist", {"name": name, "email": email,
                            "country": country, "position": position})
    try:
        _send_welcome_email(name, email, position)
    except Exception as e:
        print(f"Email send error: {e}", flush=True)
    return {"success": True, "count": position}

# ────────────────────────────────────────────────────────────────────────────
# RAZORPAY ROUTES (migrated from Stripe)
# ────────────────────────────────────────────────────────────────────────────

@app.post("/api/create-checkout")
async def create_checkout(request: Request):
    if not _RZP_KEY_CONFIGURED:
        raise HTTPException(500, "Payments not configured.")
    data          = await request.json()
    session_id    = data.get("session_id", "")
    checkout_type = data.get("type", "chat")
    csession_param = session_id if checkout_type == "chat" else ""

    try:
        order = create_order(checkout_type, csession_param)
        return order  # frontend uses order_id + key_id to open Razorpay widget
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/verify-checkout")
async def verify_checkout(request: Request):
    if not _RZP_KEY_CONFIGURED:
        raise HTTPException(500, "Payments not configured.")
    data                 = await request.json()
    razorpay_order_id    = data.get("razorpay_order_id", "")
    razorpay_payment_id  = data.get("razorpay_payment_id", "")
    razorpay_signature   = data.get("razorpay_signature", "")
    chronis_session      = data.get("chronis_session", "")
    checkout_type        = data.get("type", "chat")

    if not all([razorpay_order_id, razorpay_payment_id, razorpay_signature]):
        raise HTTPException(400, "Missing Razorpay payment fields.")

    try:
        if not verify_payment(razorpay_order_id, razorpay_payment_id, razorpay_signature):
            raise HTTPException(400, "Payment verification failed — invalid signature.")

        if checkout_type == "video":
            video_token = uuid.uuid4().hex
            sb_insert("payment_tokens", {
                "token": video_token, "razorpay_order_id": razorpay_order_id,
                "razorpay_payment_id": razorpay_payment_id,
                "chronis_session_id": None, "type": "video", "used": False,
            })
            return {"success": True, "type": "video", "video_token": video_token}
        else:
            if not chronis_session:
                raise HTTPException(400, "Missing session ID for chat unlock.")
            unlock_session_v1(chronis_session, checkout_type)
            return {"success": True, "type": checkout_type}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Verification failed: {e}")

# ────────────────────────────────────────────────────────────────────────────
# SESSION + CHAT ROUTES (migrated from app.py — v1 demo)
# ────────────────────────────────────────────────────────────────────────────

@app.get("/api/session/{session_id}")
async def restore_session(session_id: str):
    session = get_session_v1(session_id)
    if not session:
        raise HTTPException(404, "Session not found or expired.")
    raw_ts  = session.get("created_at", "")
    created = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
    if datetime.now(timezone.utc) - created > timedelta(hours=6):
        raise HTTPException(400, "Session expired.")
    messages = session.get("messages") or []
    return {
        "person_name": session["person_name"], "profile": session["profile"],
        "messages": messages, "unlocked": session.get("unlocked", False),
        "unlock_type": session.get("unlock_type"), "has_voice": bool(session.get("voice_id")),
    }

@app.post("/api/chat")
async def api_chat(request: Request):
    data       = await request.json()
    session_id = data.get("session_id", "")
    user_msg   = (data.get("message") or "").strip()

    if not session_id or not user_msg:
        raise HTTPException(400, "Missing session_id or message.")
    if len(user_msg) > 500:
        raise HTTPException(400, "Message too long (max 500 chars).")

    is_admin = (data.get("admin_secret", "") == ADMIN_SECRET and bool(ADMIN_SECRET))
    session  = get_session_v1(session_id)
    if not session:
        raise HTTPException(404, "Session not found. Please upload again.")

    raw_ts  = session.get("created_at", "")
    created = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
    if not is_admin and datetime.now(timezone.utc) - created > timedelta(hours=6):
        raise HTTPException(400, "Session expired.")

    history    = session.get("messages") or []
    unlocked   = session.get("unlocked", False) or is_admin
    exchanges  = len([m for m in history if m["role"] == "user"])

    # Payments disabled — no limit check
    # if not unlocked and exchanges >= FREE_MSG_LIMIT: (disabled)

    history.append({"role": "user", "content": user_msg,
                    "ts": datetime.utcnow().isoformat()})
    reply, error = chat_as_persona_groq(history, session["person_name"],
                                        session["persona"])
    if error:
        raise HTTPException(500, f"AI error: {error}")

    history.append({"role": "assistant", "content": reply,
                    "ts": datetime.utcnow().isoformat()})
    update_session_messages(session_id, history)

    exchanges_after = exchanges + 1
    return {
        "reply": reply, "person_name": session["person_name"],
        "has_voice": bool(session.get("voice_id")),
        "messages_used": exchanges_after, "free_limit": FREE_MSG_LIMIT,
        "limit_reached": (not unlocked) and (exchanges_after >= FREE_MSG_LIMIT),
        "unlocked": unlocked,
    }

@app.post("/api/analyze-text")
async def api_analyze_text(request: Request):
    """Analyze written memory text — migrated from app.py"""
    data        = await request.json()
    person_name = (data.get("person_name") or "this person").strip()
    memory_text = (data.get("memory_text") or "").strip()
    if not memory_text:
        raise HTTPException(400, "No memory text provided.")
    if len(memory_text) < 30:
        raise HTTPException(400, "Please write at least a few sentences.")
    if len(memory_text) > 10000:
        raise HTTPException(400, "Text too long. Max 10,000 characters.")
    if not GROQ_API_KEY:
        raise HTTPException(500, "GROQ_API_KEY not set")
    prompt = (
        f"Build a memory profile of {person_name} from this written description:\n\n"
        f"{memory_text}\n\n"
        f"Extract: voice/speech patterns, personality, memories, how they would talk.\n"
        f"Write as if briefing someone to perfectly impersonate {person_name}. Be specific."
    )
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.85, "max_tokens": 2048},
            timeout=60,
        )
        resp_data = resp.json()
        if not resp.ok:
            raise HTTPException(500, resp_data.get("error", {}).get("message", "Groq error"))
        profile = resp_data["choices"][0]["message"]["content"]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
    session_id = str(uuid.uuid4())
    persona    = build_persona(person_name, profile)
    sb_insert("sessions", {
        "session_id": session_id, "person_name": person_name,
        "profile": profile, "persona": persona,
        "filename": "text_memory", "voice_id": None,
        "messages": [], "unlocked": False, "unlock_type": None,
    })
    return {"session_id": session_id, "person_name": person_name,
            "profile": profile, "has_voice": False}


@app.post("/api/analyze")
async def api_analyze(
    request: Request,
    file: UploadFile = File(...),
    person_name: str = Form("this person"),
    admin_secret: str = Form(""),
    unlock_token: str = Form(""),
):
    """File analysis — migrated from Flask app.py"""
    ALLOWED = {"mp4","mov","avi","webm","mkv","mp3","wav","m4a","ogg","flac","aac"}
    VIDEO_EXT = {"mp4","mov","avi","webm","mkv"}

    fn  = file.filename or ""
    ext = fn.rsplit(".", 1)[-1].lower() if "." in fn else ""
    if ext not in ALLOWED:
        raise HTTPException(400, "Unsupported file type.")

    is_admin   = (admin_secret == ADMIN_SECRET and bool(ADMIN_SECRET))
    is_video   = ext in VIDEO_EXT

    # Payments disabled — video analysis free for all
    # if is_video and not is_admin: (payment gate disabled)

    session_id = str(uuid.uuid4())
    safe_name  = f"{session_id}.{ext}"
    file_path  = os.path.join(UPLOAD_FOLDER, safe_name)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    print(f"📁 Analyzing {fn} as \"{person_name}\" ({session_id[:8]})", flush=True)

    try:
        if is_video:
            profile, error = _analyze_video_gemini(file_path, safe_name, person_name)
            voice_id = None
        else:
            transcript, error = transcribe_groq(file_path, safe_name)
            if error:
                raise HTTPException(500, f"Transcription failed: {error}")
            profile, error = build_profile_from_transcript(transcript, person_name)
            voice_id = None
            if not error and XTTS_SPACE_URL:
                voice_id, v_err = _store_voice_reference(file_path, session_id)
                if v_err:
                    print(f"⚠️  Voice ref storage failed: {v_err}", flush=True)
        if error:
            raise HTTPException(500, f"Analysis failed: {error}")
    finally:
        try:
            os.remove(file_path)
        except Exception:
            pass

    persona = build_persona(person_name, profile)
    sb_insert("sessions", {
        "session_id": session_id, "person_name": person_name,
        "profile": profile, "persona": persona,
        "filename": fn, "voice_id": voice_id,
        "messages": [], "unlocked": False, "unlock_type": None,
    })

    return {"session_id": session_id, "person_name": person_name,
            "profile": profile, "has_voice": voice_id is not None}

@app.post("/api/speak")
async def api_speak(request: Request):
    data       = await request.json()
    session_id = data.get("session_id", "")
    text       = (data.get("text") or "").strip()
    if not session_id or not text:
        raise HTTPException(400, "Missing session_id or text.")
    session = get_session_v1(session_id)
    if not session:
        raise HTTPException(404, "Session not found.")
    voice_id = session.get("voice_id")
    if not voice_id:
        raise HTTPException(400, "No cloned voice for this session.")
    audio_b64, error = _synthesize_xtts(text, voice_id)
    if error:
        raise HTTPException(500, error)
    return {"audio": audio_b64, "format": "wav"}

# ── Admin ─────────────────────────────────────────────────────────────────────

@app.get("/api/sessions")
async def api_sessions(secret: str = ""):
    if not ADMIN_SECRET or secret != ADMIN_SECRET:
        raise HTTPException(401, "Unauthorized")
    rows = sb_select("sessions",
        "select=session_id,person_name,created_at,filename,voice_id,messages,unlocked,unlock_type&order=created_at.desc&limit=200")
    summary = []
    for s in rows:
        msgs = s.get("messages") or []
        summary.append({
            "session_id": s["session_id"][:8], "person_name": s.get("person_name"),
            "created_at": s.get("created_at"), "messages": len(msgs),
            "filename": s.get("filename"), "has_voice": bool(s.get("voice_id")),
            "unlocked": s.get("unlocked", False), "unlock_type": s.get("unlock_type"),
        })
    live = [{"session_id": sid[:8], "state": sess["ctrl"].state.value}
            for sid, sess in _live_sessions.items()]
    return {"sessions": summary, "total": len(summary), "live_sessions": live}

@app.get("/api/waitlist")
async def api_waitlist(secret: str = ""):
    if not ADMIN_SECRET or secret != ADMIN_SECRET:
        raise HTTPException(401, "Unauthorized")
    rows = sb_select("waitlist", "select=*&order=created_at.asc")
    return {"entries": rows, "total": len(rows)}

# ────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS (kept from app.py)
# ────────────────────────────────────────────────────────────────────────────

def _analyze_video_gemini(file_path, filename, person_name):
    if not GEMINI_API_KEY:
        return None, "GEMINI_API_KEY not set"
    mime_map = {"mp4": "video/mp4", "mov": "video/quicktime", "avi": "video/x-msvideo",
                "webm": "video/webm", "mkv": "video/x-matroska"}
    ext  = filename.rsplit(".", 1)[-1].lower()
    mime = mime_map.get(ext, "video/mp4")
    with open(file_path, "rb") as f:
        b64_data = base64.b64encode(f.read()).decode()
    prompt = f"Analyze this video to build a memory profile of {person_name}. Extract voice, personality, conversations, and memorable details. Write as if briefing someone to perfectly impersonate them."
    url     = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"inline_data": {"mime_type": mime, "data": b64_data}},
                                        {"text": prompt}]}],
               "generationConfig": {"temperature": 0.85, "maxOutputTokens": 2048}}
    try:
        resp = requests.post(url, json=payload, timeout=180)
        data = resp.json()
        if resp.ok:
            return data["candidates"][0]["content"]["parts"][0]["text"], None
        return None, data.get("error", {}).get("message", "Gemini error")
    except Exception as e:
        return None, str(e)

def _store_voice_reference(audio_path, session_id):
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None, "Supabase not configured"
    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()[:12_000_000]
        filename = f"{session_id}_ref.mp3"
        r = requests.post(
            f"{SUPABASE_URL}/storage/v1/object/voice-refs/{filename}",
            headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
                     "Content-Type": "audio/mpeg", "x-upsert": "true"},
            data=audio_bytes, timeout=30,
        )
        return (filename, None) if r.ok else (None, f"Storage error: {r.status_code}")
    except Exception as e:
        return None, str(e)

def _synthesize_xtts(text, voice_ref_filename):
    if not XTTS_SPACE_URL:
        return None, "XTTS not configured"
    tmp_audio_path = None
    try:
        r = requests.get(
            f"{SUPABASE_URL}/storage/v1/object/voice-refs/{voice_ref_filename}",
            headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"},
            timeout=30,
        )
        if not r.ok:
            return None, f"Voice fetch failed: {r.status_code}"
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(r.content)
            tmp_audio_path = f.name
        cleaned = _clean_for_xtts(text)
        client  = get_xtts_client()
        result  = client.predict(cleaned, handle_file(tmp_audio_path),
                                 XTTS_SECRET, api_name="/predict")
        if isinstance(result, (list, tuple)) and len(result) >= 2:
            audio_b64, status = result[0], result[1]
            if status == "ok" and audio_b64:
                return audio_b64, None
            return None, f"XTTS error: {status}"
        return None, f"Unexpected XTTS response: {result}"
    except Exception as e:
        global xtts_client
        xtts_client = None
        return None, str(e)
    finally:
        if tmp_audio_path and os.path.exists(tmp_audio_path):
            try:
                os.unlink(tmp_audio_path)
            except Exception:
                pass

def _send_welcome_email(name, email, position):
    if not RESEND_API_KEY:
        return
    first = name.split()[0]
    try:
        requests.post("https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {RESEND_API_KEY}",
                     "Content-Type": "application/json"},
            json={"from": "Chronis <hello@chronis.in>", "to": email,
                  "subject": f"You're on the list, {first} — Chronis",
                  "html": f"<p>Welcome to Chronis, {first}! You're #{position} on the waitlist.</p>"},
            timeout=10)
    except Exception as e:
        print(f"Welcome email error: {e}", flush=True)


# ────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"FastAPI app starting on port {port}", flush=True)
    uvicorn.run(
        "main_v2:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        ws_ping_interval=20,
        ws_ping_timeout=10,
    )