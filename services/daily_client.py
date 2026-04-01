"""
services/daily_client.py  —  DailyClient

Manages Daily.co rooms for live sessions.

Daily.co provides:
  - WebRTC infrastructure
  - Echo cancellation (EC)
  - Noise suppression (NS)
  - Auto gain control (AGC)
  - 1,000 participant-minutes/month free

CRITICAL cost note:
  Free tier = exactly 50 users × 20 min average.
  Keep dev testing sessions under 5 minutes.
  Track usage via sessions table. Alert at 700 minutes.
  Upgrade to $12/mo BEFORE user testing begins.
"""

import os
import time
from typing import Optional

import httpx

DAILY_API_KEY  = os.environ.get("DAILY_API_KEY", "")
DAILY_BASE_URL = "https://api.daily.co/v1"

# Session duration: 2 hours max. Avatar sessions rarely exceed 30 min.
ROOM_EXPIRY_SECONDS = 7200


class DailyClient:
    """
    Thin async wrapper around Daily.co REST API.
    Used for room lifecycle management only — not for WebRTC itself
    (that's handled by the browser-side Daily.co JS SDK).
    """

    def __init__(self):
        self._http = httpx.AsyncClient(
            base_url=DAILY_BASE_URL,
            headers={
                "Authorization": f"Bearer {DAILY_API_KEY}",
                "Content-Type":  "application/json",
            },
            timeout=15.0,
        )

    async def close(self):
        await self._http.aclose()

    # ────────────────────────────────────────────────────────────────────────
    # Room management
    # ────────────────────────────────────────────────────────────────────────

    async def create_room(self, session_id: str) -> str:
        """
        Create a Daily.co room with EC + NS + AGC enabled.
        Returns the room URL.

        Room naming: chronis-{first 12 chars of session_id}
        Expiry: 2 hours from now (rooms are ephemeral — delete after session)

        EC/NS/AGC are CRITICAL. Never disable them. They're what separates
        a usable voice pipeline from an echo-chamber disaster.

        NOTE: audio_constraints is a browser-side WebRTC property — NOT a
        valid Daily.co room creation property (causes 400 invalid-request-error).
        Daily applies EC/NS/AGC automatically server-side; no override needed.
        """
        room_name = f"chronis-{session_id[:12]}"

        r = await self._http.post(
            "/rooms",
            json={
                "name": room_name,
                "properties": {
                    "exp": int(time.time()) + ROOM_EXPIRY_SECONDS,

                    # Audio processing — Daily applies EC/NS/AGC automatically.
                    "enable_noise_cancellation_ui": True,
                    "start_audio_off": False,
                    "start_video_off": True,   # We don't need browser camera

                    # Prevent random people from joining — token required
                    "enable_people_ui": False,
                    "max_participants": 2,     # user + Simli avatar stream
                },
            },
        )

        if not r.is_success:
            raise RuntimeError(
                f"Daily room creation failed: {r.status_code} {r.text[:300]}"
            )

        data = r.json()
        url  = data.get("url")
        if not url:
            raise RuntimeError(f"No URL in Daily response: {data}")

        print(f"[Daily] Room created: {url}", flush=True)
        return url

    async def delete_room(self, room_url: str) -> bool:
        """
        Delete a Daily.co room. Call this during session cleanup.
        Without deletion, rooms accumulate and you burn participant-minutes
        on zombie rooms that nobody is in.
        """
        room_name = room_url.rstrip("/").split("/")[-1]
        try:
            r = await self._http.delete(f"/rooms/{room_name}")
            if r.is_success:
                print(f"[Daily] Room deleted: {room_name}", flush=True)
                return True
            else:
                print(f"[Daily] Room delete failed: {r.status_code}", flush=True)
                return False
        except Exception as e:
            print(f"[Daily] Room delete error: {e}", flush=True)
            return False

    async def get_room(self, room_url: str) -> Optional[dict]:
        """Get room details (useful for verifying a room still exists)."""
        room_name = room_url.rstrip("/").split("/")[-1]
        try:
            r = await self._http.get(f"/rooms/{room_name}")
            return r.json() if r.is_success else None
        except Exception:
            return None

    # ────────────────────────────────────────────────────────────────────────
    # Usage tracking
    # ────────────────────────────────────────────────────────────────────────

    async def get_usage(self) -> dict:
        """
        Fetch current month participant-minutes usage.
        Free tier: 1,000 min/month.
        Alert threshold: 700 min (70%) — warn before hitting the hard limit.

        Returns: {"total_minutes": float, "pct_of_free_tier": float}
        """
        try:
            r = await self._http.get("/usage")
            if not r.is_success:
                return {"error": f"{r.status_code}"}
            data           = r.json()
            total_minutes  = data.get("participant_minutes", 0)
            pct            = (total_minutes / 1000) * 100
            over_limit     = total_minutes >= 700

            if over_limit:
                print(
                    f"[Daily] ⚠️  Usage at {total_minutes:.0f}/1000 min ({pct:.0f}%). "
                    f"Upgrade to $12/mo plan BEFORE user testing!",
                    flush=True,
                )

            return {
                "total_minutes":     round(total_minutes, 1),
                "free_tier_limit":   1000,
                "pct_of_free_tier":  round(pct, 1),
                "alert":             over_limit,
            }
        except Exception as e:
            return {"error": str(e)}

    async def create_meeting_token(self, room_url: str,
                                   is_owner: bool = False) -> str:
        """
        Create a meeting token for a specific room.
        Tokens limit who can join — important for preventing uninvited guests.

        is_owner=True: can manage the room (kick participants, etc.)
        is_owner=False: regular participant — what users get
        """
        room_name = room_url.rstrip("/").split("/")[-1]
        try:
            r = await self._http.post(
                "/meeting-tokens",
                json={
                    "properties": {
                        "room_name": room_name,
                        "is_owner":  is_owner,
                        "exp":       int(time.time()) + ROOM_EXPIRY_SECONDS,
                        "start_audio_off": False,
                        "start_video_off": True,
                    }
                },
            )
            if r.is_success:
                return r.json().get("token", "")
            return ""
        except Exception:
            return ""


# ── Module-level singleton ────────────────────────────────────────────────────
_daily_client: Optional[DailyClient] = None

def get_daily_client() -> DailyClient:
    global _daily_client
    if _daily_client is None:
        _daily_client = DailyClient()
    return _daily_client