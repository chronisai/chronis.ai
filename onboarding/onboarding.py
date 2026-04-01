"""
onboarding/onboarding.py  —  OnboardingPipeline

Idempotent, step-by-step onboarding for creating a Chronis replica.

Runs ONCE per agent. All the slow work (Simli agent creation: 45-90s)
happens here so the live session has zero onboarding overhead.

Seven steps:
  1. Validate photo locally (MediaPipe — milliseconds)
  2. Validate audio locally (librosa — seconds)
  3. Check idempotency — skip steps already completed
  4. Upload photo to Supabase Storage
  5. Create Simli face agent (async background — 45-90s with polling)
  6. Upload voice reference to Modal persistent volume
  7. Mark agent ready

Idempotency:
  Before creating anything, we check the DB:
  - agents.simli_agent_id IS NOT NULL AND status = 'ready' → skip Simli
  - voices.modal_voice_ref IS NOT NULL AND status = 'ready' → skip voice
  The status enum (creating | ready | failed) prevents race conditions.
  A row stuck in 'creating' for > 10 minutes is recovered by startup job.

Stale job recovery (fix #6):
  SupabaseClient.recover_stale_onboarding_jobs() handles rows stuck in
  'creating' from Railway restarts or crashed background tasks.
"""
import base64
import asyncio
import os
import uuid
from typing import Dict, Optional, Callable

import httpx

from services.supabase_client import SupabaseClient
from validators.photo_validator import PhotoValidator
from validators.audio_validator import validate_audio

SIMLI_API_KEY  = os.environ.get("SIMLI_API_KEY", "")
SIMLI_BASE_URL = "https://api.simli.ai"

MODAL_XTTS_URL = os.environ.get("MODAL_XTTS_URL", "")
SUPABASE_URL   = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY   = os.environ.get("SUPABASE_SERVICE_KEY", "")

# How long to wait between Simli status polls
SIMLI_POLL_INTERVAL_S = 5.0
# Maximum time to wait for Simli agent creation before marking failed
SIMLI_MAX_WAIT_S      = 180.0


class OnboardingPipeline:
    """
    Manages the full onboarding flow for one agent.

    Usage:
        pipeline = OnboardingPipeline(db, photo_validator)
        result = await pipeline.run(
            user_id="...",
            agent_name="Mom",
            personality="Warm, funny, loves cooking...",
            photo_bytes=b"...",
            audio_path="/tmp/voice.wav",
            on_progress=lambda step, msg: ...,
        )
    """

    def __init__(self, db: SupabaseClient, photo_validator: PhotoValidator):
        self.db              = db
        self.photo_validator = photo_validator
        self._http           = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        await self._http.aclose()

    # ────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ────────────────────────────────────────────────────────────────────────

    async def run(
        self,
        user_id: str,
        agent_name: str,
        personality: str,
        photo_bytes: bytes,
        audio_path: str,   # path to uploaded audio file (any format)
        agent_id: str = None,  # if provided, operate on this exact row (no internal creation)
        on_progress: Callable[[str, str], None] = None,
    ) -> Dict:
        """
        Run the full onboarding pipeline.

        agent_id: when the caller (route) has already created the DB row, pass it
        here. The pipeline will operate on that exact row and skip internal creation.

        on_progress(step_name, message) is called on each step completion.
        Returns {"agent_id": str, "status": "ready"} or {"error": str}
        """

        def progress(step: str, msg: str):
            print(f"[Onboarding] {step}: {msg}", flush=True)
            if on_progress:
                on_progress(step, msg)

        # ── Step 1: Validate photo ─────────────────────────────────────────
        progress("photo_validate", "Checking photo quality...")
        photo_result = self.photo_validator.validate(photo_bytes)
        if not photo_result["valid"]:
            return {"error": photo_result["reason"], "step": "photo_validate"}
        progress("photo_validate", f"Photo OK — {photo_result['resolution']}, "
                                   f"sharpness={photo_result['sharpness']}")

        # ── Step 2: Validate audio ─────────────────────────────────────────
        progress("audio_validate", "Checking voice recording quality...")
        audio_result = validate_audio(audio_path)
        if not audio_result["valid"]:
            return {"error": audio_result["reason"], "step": "audio_validate"}
        progress("audio_validate",
                 f"Audio OK — {audio_result['duration_s']}s, "
                 f"SNR={audio_result['snr_db']}dB, "
                 f"speech={audio_result['speech_ratio']:.0%}")

        converted_audio_path = audio_result["converted_path"]

        # ── Step 3: Idempotency check ─────────────────────────────────────
        # If the route provided an agent_id, look up that exact row.
        # Otherwise fall back to user+name (newest first) to avoid picking
        # up a stale creating/failed row on retry.
        skip_simli     = False
        skip_voice     = False
        simli_agent_id = None

        if agent_id:
            existing = await self.db.select(
                "agents",
                f"id=eq.{agent_id}&select=*,voices(*)"
            )
        else:
            existing = await self.db.select(
                "agents",
                f"user_id=eq.{user_id}&name=eq.{agent_name}"
                f"&order=created_at.desc&select=*,voices(*)"
            )

        if existing:
            row      = existing[0]
            agent_id = agent_id or row["id"]

            # Skip Simli if face already submitted (simli_agent_id present)
            if row.get("simli_agent_id"):
                skip_simli     = True
                simli_agent_id = row["simli_agent_id"]
                progress("idempotency", "Simli agent already exists — skipping creation")

            voices       = row.get("voices") or []
            ready_voices = [v for v in voices if v.get("status") == "ready"]
            if ready_voices:
                skip_voice = True
                progress("idempotency", "Voice reference already uploaded — skipping")

        # ── Step 4: Create agent DB record (only if no agent_id at all) ──
        # When the route passes agent_id, this step is always skipped.
        if agent_id is None:
            progress("agent_create", "Creating agent record...")
            agent_row = await self.db.insert("agents", {
                "user_id":     user_id,
                "name":        agent_name,
                "personality": personality,
                "status":      "creating",
            })
            if not agent_row:
                return {"error": "Failed to create agent record", "step": "agent_create"}
            agent_id = agent_row["id"]

        # ── Step 5: Upload photo to Supabase Storage ───────────────────────
        if not skip_simli:
            progress("photo_upload", "Uploading photo...")
            photo_url, upload_err = await self._upload_photo(photo_bytes, agent_id)
            if upload_err:
                await self.db.update("agents", "id", agent_id, {"status": "failed"})
                return {"error": f"Photo upload failed: {upload_err}", "step": "photo_upload"}

            await self.db.update("agents", "id", agent_id, {"photo_url": photo_url})
            progress("photo_upload", f"Photo uploaded: {photo_url[:60]}...")

        # ── Step 6: Create Simli face agent (async poll) ───────────────────
        if not skip_simli:
            progress("simli_create", "Creating face model (45-90s, please wait)...")
            simli_agent_id, simli_err = await self._create_simli_agent(
                agent_id=agent_id,
                photo_bytes=photo_bytes,
                on_progress=on_progress,
            )
            if simli_err:
                await self.db.update("agents", "id", agent_id, {"status": "failed"})
                return {"error": f"Simli creation failed: {simli_err}", "step": "simli_create"}

            # Store simli_agent_id but keep status='creating' until voice is also ready
            await self.db.update("agents", "id", agent_id, {
                "simli_agent_id": simli_agent_id,
            })
            progress("simli_create", f"Simli face model ready: {simli_agent_id[:12]}")

        # ── Step 7: Upload voice reference to Modal volume ─────────────────
        if not skip_voice:
            progress("voice_upload", "Uploading voice reference to Modal...")
            voice_ref, voice_err = await self._upload_voice_to_modal(
                converted_audio_path, agent_id
            )
            if voice_err:
                await self.db.insert("voices", {
                    "agent_id": agent_id,
                    "status":   "failed",
                })
                return {"error": f"Voice upload failed: {voice_err}", "step": "voice_upload"}

            # Delete any stale voice rows before inserting fresh one.
            # Prevents get_agent() from reading a failed/old row on retry.
            await self.db.delete("voices", f"agent_id=eq.{agent_id}")

            # Store voice reference in DB
            await self.db.insert("voices", {
                "agent_id":         agent_id,
                "modal_voice_ref":  voice_ref,
                "duration_seconds": audio_result["duration_s"],
                "snr_db":           audio_result["snr_db"],
                "speech_ratio":     audio_result["speech_ratio"],
                "status":           "ready",
            })
            progress("voice_upload", f"Voice reference stored: {voice_ref[:60]}")

        # ── Cleanup temp converted audio ───────────────────────────────────
        try:
            if converted_audio_path and os.path.exists(converted_audio_path):
                os.unlink(converted_audio_path)
        except Exception:
            pass

        # ── Step 8: Mark agent ready ───────────────────────────────────────
        # Only set here, after BOTH Simli face and voice are confirmed ready.
        # The route sets status="creating" on the row; we own the transition to ready.
        ok = await self.db.update("agents", "id", agent_id, {"status": "ready"})
        if not ok:
            return {"error": "Failed to mark agent ready", "step": "mark_ready"}

        progress("complete", f"Onboarding complete for agent {agent_id[:8]} ✓")

        return {
            "agent_id":       agent_id,
            "simli_agent_id": simli_agent_id,
            "status":         "ready",
        }

    # ────────────────────────────────────────────────────────────────────────
    # Photo upload
    # ────────────────────────────────────────────────────────────────────────

    async def _upload_photo(
        self, photo_bytes: bytes, agent_id: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Upload photo to Supabase Storage. Returns (storage_path, error)."""
        filename = f"agents/{agent_id}/photo.jpg"
        try:
            r = await self._http.post(
                f"{SUPABASE_URL}/storage/v1/object/agent-photos/{filename}",
                headers={
                    "apikey":        SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type":  "image/jpeg",
                    "x-upsert":      "true",
                },
                content=photo_bytes,
            )
            if r.is_success:
                return filename, None
            return None, f"Storage {r.status_code}: {r.text[:200]}"
        except Exception as e:
            return None, str(e)

    # ────────────────────────────────────────────────────────────────────────
    # Simli agent creation (async with polling)
    # ────────────────────────────────────────────────────────────────────────

    async def _create_simli_agent(
        self,
        agent_id: str,
        photo_bytes: bytes,
        on_progress: Callable = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Submit photo to Simli for face model creation.
        Polls for completion every SIMLI_POLL_INTERVAL_S seconds.
        Returns (face_id, error).
        """
        # ── Submit creation request ────────────────────────────────────────
        try:
            r = await self._http.post(
                f"{SIMLI_BASE_URL}/generateFaceID",
                headers={
                    "x-simli-api-key": SIMLI_API_KEY,
                    "Content-Type":    "application/json",
                },
                json={"image": base64.b64encode(photo_bytes).decode("utf-8")},
                timeout=30.0,
            )
            if not r.is_success:
                return None, f"Simli API error {r.status_code}: {r.text[:300]}"

            data   = r.json()
            job_id = data.get("jobId") or data.get("requestId") or data.get("request_id")
            if not job_id:
                return None, f"No job ID in Simli response: {data}"

        except Exception as e:
            return None, f"Simli submission error: {e}"

        # ── Poll for completion ────────────────────────────────────────────
        elapsed  = 0.0
        poll_num = 0

        while elapsed < SIMLI_MAX_WAIT_S:
            await asyncio.sleep(SIMLI_POLL_INTERVAL_S)
            elapsed  += SIMLI_POLL_INTERVAL_S
            poll_num += 1

            try:
                status_r = await self._http.post(
                    f"{SIMLI_BASE_URL}/getRequestStatus",
                    headers={
                        "x-simli-api-key": SIMLI_API_KEY,
                        "Content-Type":    "application/json",
                    },
                    json={"requestId": job_id},
                    timeout=10.0,
                )
                status_data   = status_r.json()
                status        = status_data.get("status", "")
                simli_face_id = (
                    status_data.get("faceId")
                    or status_data.get("face_id")
                    or status_data.get("faceID")
                )

                if on_progress:
                    on_progress(
                        "simli_poll",
                        f"Simli status: {status} ({elapsed:.0f}s elapsed)",
                    )

                if status == "completed" and simli_face_id:
                    return simli_face_id, None

                elif status == "failed":
                    error_msg = status_data.get("error", "Unknown Simli error")
                    return None, f"Simli face creation failed: {error_msg}"

                # Still processing — continue polling

            except Exception as e:
                # Poll error — log and continue (transient errors are normal)
                print(f"[Onboarding] Simli poll error (attempt {poll_num}): {e}",
                      flush=True)

        return None, f"Simli face creation timed out after {SIMLI_MAX_WAIT_S:.0f}s"

    # ────────────────────────────────────────────────────────────────────────
    # Voice upload to Modal persistent volume
    # ────────────────────────────────────────────────────────────────────────

    async def _upload_voice_to_modal(
        self, audio_path: str, agent_id: str
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Upload the converted audio file to Modal's persistent volume.

        The volume path returned here is passed to every TTS inference call.
        By co-locating the reference audio with the XTTS model, we get
        zero fetch latency during live sessions — the file is already local.

        Returns (modal_volume_path, error).
        """
        if not MODAL_XTTS_URL:
            return None, "MODAL_XTTS_URL not configured"

        try:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            volume_path = f"voice_refs/{agent_id}/reference.wav"

            # POST to the Modal XTTS server's upload endpoint
            upload_url = MODAL_XTTS_URL.replace("/synthesize", "/upload_voice")

            r = await self._http.post(
                upload_url,
                content=audio_bytes,
                headers={
                    "Content-Type":  "audio/wav",
                    "X-Volume-Path": volume_path,
                },
                timeout=60.0,
            )

            if r.is_success:
                data        = r.json()
                stored_path = data.get("path", volume_path)
                return stored_path, None
            else:
                return None, f"Modal upload error {r.status_code}: {r.text[:200]}"

        except Exception as e:
            return None, str(e)