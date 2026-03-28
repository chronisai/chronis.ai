"""
services/memory_service.py  —  MemoryService

Manages the LLM context window across an entire session.

Without this, the avatar gets "identity drift" after ~10 minutes:
the context window fills up, personality gets pushed out, and the
avatar starts answering like a generic chatbot.

Context structure (in this exact order every LLM call):
  [IDENTITY]          — Who this replica is. ALWAYS first. NEVER removed.
  [PINNED MEMORIES]   — Key facts, relationships, events. NEVER removed.
  [SUMMARY]           — Auto-generated after every 20 turns. Replaces raw turns.
  [LAST 6 EXCHANGES]  — Recent dialogue for conversational flow.
  [CURRENT INPUT]     — What the user just said.

Summarization trigger: every 20 turns, send the last 20 raw exchanges to
Groq, store the summary, delete the raw entries. This keeps the context
window stable for arbitrarily long conversations.
"""

import asyncio
import os
from typing import Dict, List, Optional, TYPE_CHECKING

from services.groq_client import get_groq_client

if TYPE_CHECKING:
    from services.supabase_client import SupabaseClient

# How many recent raw exchanges to keep in context at all times
RECENT_EXCHANGES_WINDOW = 6

# How many turns before we trigger summarization
SUMMARIZE_EVERY_N_TURNS = 20


class MemoryService:

    def __init__(self, agent_id: str, db: "SupabaseClient"):
        self.agent_id   = agent_id
        self.db         = db

        # In-memory turn buffer — flushed to DB periodically
        # Structure: [{"role": "user"|"assistant", "content": "...", "turn": int}]
        self._recent_turns: List[Dict] = []
        self._turn_index  = 0

        # Cached identity and pinned memories (loaded once at session start)
        self._identity_prompt: str = ""
        self._pinned_memories: List[str] = []
        self._summary: str = ""

    # ────────────────────────────────────────────────────────────────────────
    # Initialization
    # ────────────────────────────────────────────────────────────────────────

    async def load(self) -> None:
        """
        Load all existing memories for this agent from Supabase.
        Call once at session start before the first LLM call.
        """
        rows = await self.db.get_memories(self.agent_id)

        for row in rows:
            mem_type = row.get("type", "")
            content  = row.get("content", "")
            idx      = row.get("turn_index", 0)

            if mem_type == "identity":
                self._identity_prompt = content
            elif mem_type == "pinned":
                self._pinned_memories.append(content)
            elif mem_type == "summary":
                self._summary = content
            elif mem_type == "conversation":
                # Restore recent turns (only keep the last window)
                role = "user" if idx % 2 == 0 else "assistant"
                self._recent_turns.append({
                    "role":    role,
                    "content": content,
                    "turn":    idx,
                })

        # Keep only the most recent N exchanges
        self._recent_turns = self._recent_turns[-(RECENT_EXCHANGES_WINDOW * 2):]

        # Set turn index from the highest stored turn
        if rows:
            self._turn_index = max(r.get("turn_index", 0) for r in rows) + 1

        print(f"[Memory] Loaded for agent {self.agent_id[:8]} "
              f"— identity: {bool(self._identity_prompt)}, "
              f"pinned: {len(self._pinned_memories)}, "
              f"recent: {len(self._recent_turns)}", flush=True)

    # ────────────────────────────────────────────────────────────────────────
    # Context building
    # ────────────────────────────────────────────────────────────────────────

    def build_messages(self, current_user_input: str) -> List[Dict]:
        """
        Build the full message list for the Groq API call.
        Returns [{"role": "system"|"user"|"assistant", "content": "..."}]
        """
        # ── System prompt: identity + pinned memories ──────────────────────
        system_parts = []

        if self._identity_prompt:
            system_parts.append(f"[IDENTITY]\n{self._identity_prompt}")

        if self._pinned_memories:
            pinned_text = "\n".join(f"- {m}" for m in self._pinned_memories)
            system_parts.append(f"[PINNED MEMORIES]\n{pinned_text}")

        if self._summary:
            system_parts.append(f"[CONVERSATION SUMMARY]\n{self._summary}")

        system_prompt = "\n\n".join(system_parts) if system_parts else (
            "You are a helpful, natural conversational AI."
        )

        messages: List[Dict] = [{"role": "system", "content": system_prompt}]

        # ── Recent turns ───────────────────────────────────────────────────
        for turn in self._recent_turns[-(RECENT_EXCHANGES_WINDOW * 2):]:
            messages.append({
                "role":    turn["role"],
                "content": turn["content"],
            })

        # ── Current input ──────────────────────────────────────────────────
        messages.append({"role": "user", "content": current_user_input})

        return messages

    # ────────────────────────────────────────────────────────────────────────
    # Turn recording
    # ────────────────────────────────────────────────────────────────────────

    async def record_turn(self, user_text: str, assistant_text: str,
                          turn_count: int) -> None:
        """
        Record a completed exchange to memory.
        Persists to Supabase and manages the sliding window.

        Also triggers summarization every SUMMARIZE_EVERY_N_TURNS turns.
        Summarization runs as a background task — does NOT block the current response.
        """
        # Add to in-memory buffer
        self._recent_turns.append({
            "role": "user", "content": user_text, "turn": self._turn_index
        })
        self._recent_turns.append({
            "role": "assistant", "content": assistant_text, "turn": self._turn_index + 1
        })
        self._turn_index += 2

        # Persist both turns to Supabase
        await self.db.save_memory({
            "agent_id":   self.agent_id,
            "type":       "conversation",
            "content":    user_text,
            "turn_index": self._turn_index - 2,
        })
        await self.db.save_memory({
            "agent_id":   self.agent_id,
            "type":       "conversation",
            "content":    assistant_text,
            "turn_index": self._turn_index - 1,
        })

        # Trim in-memory window to keep only the most recent exchanges
        self._recent_turns = self._recent_turns[-(RECENT_EXCHANGES_WINDOW * 2):]

        # Trigger summarization in background (does NOT block)
        if turn_count > 0 and turn_count % SUMMARIZE_EVERY_N_TURNS == 0:
            asyncio.create_task(self._summarize(turn_count))

    # ────────────────────────────────────────────────────────────────────────
    # Summarization
    # ────────────────────────────────────────────────────────────────────────

    async def _summarize(self, turn_count: int) -> None:
        """
        Generate a summary of the last N conversation turns and store it.
        Replaces the raw conversation entries to keep context window small.

        Runs async in background — does not block the live response.
        """
        print(f"[Memory] Summarizing at turn {turn_count}...", flush=True)

        # Fetch the last N raw conversation entries from Supabase
        all_convs = await self.db.select(
            "memories",
            f"agent_id=eq.{self.agent_id}&type=eq.conversation&order=turn_index.desc&limit={SUMMARIZE_EVERY_N_TURNS * 2}"
        )

        if not all_convs:
            return

        # Build a readable transcript for summarization
        transcript_lines = []
        for row in sorted(all_convs, key=lambda r: r.get("turn_index", 0)):
            # Alternate user/assistant based on turn parity
            role = "User" if row.get("turn_index", 0) % 2 == 0 else "Assistant"
            transcript_lines.append(f"{role}: {row['content']}")

        transcript = "\n".join(transcript_lines)

        # Call Groq to summarize — use the shared GroqClient for retry/backoff
        try:
            summary = await get_groq_client().complete(
                messages=[{
                    "role":    "user",
                    "content": (
                        f"Summarize this conversation as bullet points. "
                        f"Preserve key facts, emotions, topics discussed, "
                        f"and anything important for continuity. Be concise.\n\n"
                        f"{transcript}"
                    ),
                }],
                temperature=0.3,
                max_tokens=500,
            )
        except Exception as e:
            print(f"[Memory] Summarization error: {e}", flush=True)
            return

        # Store the new summary
        await self.db.save_memory({
            "agent_id":   self.agent_id,
            "type":       "summary",
            "content":    summary,
            "turn_index": turn_count,
        })

        # Delete the raw conversation entries we just summarized
        # Use BOTH min AND max bounds to avoid deleting turns that arrived
        # after summarization started (background task race condition fix)
        min_turn = min(r.get("turn_index", 0) for r in all_convs)
        max_turn = max(r.get("turn_index", 0) for r in all_convs)
        await self.db.delete(
            "memories",
            f"agent_id=eq.{self.agent_id}&type=eq.conversation"
            f"&turn_index=gte.{min_turn}&turn_index=lte.{max_turn}"
        )

        # Update in-memory summary
        self._summary = summary
        print(f"[Memory] Summary stored ✓ ({len(summary)} chars)", flush=True)

    async def close(self) -> None:
        pass  # No owned resources — GroqClient singleton is closed by the app
