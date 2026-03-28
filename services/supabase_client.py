"""
services/supabase_client.py  —  SupabaseClient

All DB operations go through here. Pipelines NEVER call Supabase directly.

Async throughout — uses httpx.AsyncClient with connection pooling.
Connection pool is shared across all sessions via the singleton pattern
at the bottom of this file.
"""

import asyncio
import os
import time
from typing import Any, Dict, List, Optional

import httpx


class SupabaseClient:

    def __init__(self, url: str, key: str):
        self.url     = url.rstrip("/")
        self.key     = key
        # Shared async HTTP client with connection pooling
        # timeout=15 for DB ops; storage ops may need more
        self._client = httpx.AsyncClient(
            base_url=f"{self.url}/rest/v1",
            headers={
                "apikey":        key,
                "Authorization": f"Bearer {key}",
                "Content-Type":  "application/json",
                "Prefer":        "return=representation",
            },
            timeout=15.0,
        )
        self._storage_client = httpx.AsyncClient(
            base_url=f"{self.url}/storage/v1",
            headers={
                "apikey":        key,
                "Authorization": f"Bearer {key}",
            },
            timeout=30.0,
        )

    async def close(self):
        await self._client.aclose()
        await self._storage_client.aclose()

    # ────────────────────────────────────────────────────────────────────────
    # Generic CRUD
    # ────────────────────────────────────────────────────────────────────────

    async def select(self, table: str, query: str = "") -> List[Dict]:
        try:
            r = await self._client.get(f"/{table}?{query}")
            return r.json() if r.is_success else []
        except Exception as e:
            print(f"[SB select error] {table}: {e}", flush=True)
            return []

    async def insert(self, table: str, data: Dict) -> Optional[Dict]:
        try:
            r = await self._client.post(f"/{table}", json=data)
            rows = r.json()
            return rows[0] if r.is_success and isinstance(rows, list) and rows else None
        except Exception as e:
            print(f"[SB insert error] {table}: {e}", flush=True)
            return None

    async def update(self, table: str, match_col: str, match_val: str,
                     data: Dict) -> bool:
        try:
            r = await self._client.patch(
                f"/{table}?{match_col}=eq.{match_val}", json=data
            )
            return r.is_success
        except Exception as e:
            print(f"[SB update error] {table}: {e}", flush=True)
            return False

    async def delete(self, table: str, query: str) -> bool:
        try:
            r = await self._client.delete(f"/{table}?{query}")
            return r.is_success
        except Exception as e:
            print(f"[SB delete error] {table}: {e}", flush=True)
            return False

    # ────────────────────────────────────────────────────────────────────────
    # Domain-specific helpers
    # ────────────────────────────────────────────────────────────────────────

    async def get_agent(self, agent_id: str) -> Optional[Dict]:
        """Fetch a single agent row with its voice reference joined.
        Voices are sorted so ready rows come first — defensive against
        stale rows from failed/retried onboarding runs.
        """
        rows = await self.select(
            "agents",
            f"id=eq.{agent_id}&select=*,voices(*)"
        )
        if not rows:
            return None
        agent = rows[0]
        voices = agent.get("voices") or []
        # Sort: ready first, then by any other status — safe even if list is empty
        voices.sort(key=lambda v: (0 if v.get("status") == "ready" else 1))
        agent["voices"] = voices
        return agent

    async def get_session(self, session_id: str) -> Optional[Dict]:
        rows = await self.select("sessions", f"id=eq.{session_id}&select=*")
        return rows[0] if rows else None

    async def create_session(self, data: Dict) -> Optional[Dict]:
        return await self.insert("sessions", data)

    async def end_session(self, session_id: str) -> bool:
        from datetime import datetime, timezone
        return await self.update(
            "sessions", "id", session_id,
            {"ended_at": datetime.now(timezone.utc).isoformat(), "state": "ended"}
        )

    async def update_session_state(self, session_id: str, state: str) -> bool:
        return await self.update("sessions", "id", session_id, {"state": state})

    async def increment_turn(self, session_id: str, turn_count: int) -> bool:
        return await self.update(
            "sessions", "id", session_id, {"turn_count": turn_count}
        )

    async def get_memories(self, agent_id: str) -> List[Dict]:
        """Fetch all memory rows for this agent, ordered for context building."""
        return await self.select(
            "memories",
            f"agent_id=eq.{agent_id}&order=turn_index.asc"
        )

    async def save_memory(self, data: Dict) -> Optional[Dict]:
        return await self.insert("memories", data)

    async def delete_memories(self, agent_id: str, type_: str,
                               turn_index_lt: int) -> bool:
        return await self.delete(
            "memories",
            f"agent_id=eq.{agent_id}&type=eq.{type_}&turn_index=lt.{turn_index_lt}"
        )

    # ────────────────────────────────────────────────────────────────────────
    # Session events (called by log worker — not inline in pipelines)
    # ────────────────────────────────────────────────────────────────────────

    async def log_event(self, session_id: str, event_type: str,
                        pipeline: str = "", payload: dict = None) -> None:
        """
        Write one row to session_events table.
        This is called by the background log worker, never inline.
        """
        await self.insert("session_events", {
            "session_id": session_id,
            "event_type": event_type,
            "pipeline":   pipeline,
            "payload":    payload or {},
        })

    # ────────────────────────────────────────────────────────────────────────
    # Stale-job recovery (fix #6 from code review)
    # ────────────────────────────────────────────────────────────────────────

    async def recover_stale_onboarding_jobs(self) -> int:
        """
        Mark any 'creating' agent/voice rows older than 10 minutes as 'failed'.
        Call this on startup and periodically (e.g. every 5 minutes).
        Returns count of rows recovered.
        """
        cutoff = "now() - interval '10 minutes'"
        count = 0

        # Stale agents stuck in 'creating'
        r_agents = await self._client.patch(
            f"/agents?status=eq.creating&created_at=lt.{cutoff}",
            json={"status": "failed"},
            headers={"Prefer": "return=representation"},
        )
        if r_agents.is_success:
            recovered = r_agents.json()
            count += len(recovered)
            if recovered:
                print(f"[SB recovery] Marked {len(recovered)} stale agents as failed",
                      flush=True)

        # Stale voices stuck in 'processing'
        r_voices = await self._client.patch(
            f"/voices?status=eq.processing&created_at=lt.{cutoff}",
            json={"status": "failed"},
            headers={"Prefer": "return=representation"},
        )
        if r_voices.is_success:
            recovered = r_voices.json()
            count += len(recovered)
            if recovered:
                print(f"[SB recovery] Marked {len(recovered)} stale voices as failed",
                      flush=True)

        return count


# ── Module-level singleton ───────────────────────────────────────────────────
# Instantiated once in main_v2.py and passed to all pipelines.
# Defined here so services can import it if needed.

_instance: Optional[SupabaseClient] = None

def init_supabase() -> SupabaseClient:
    global _instance
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_KEY"]
    _instance = SupabaseClient(url, key)
    return _instance

def get_supabase() -> SupabaseClient:
    if _instance is None:
        raise RuntimeError("Call init_supabase() first")
    return _instance
