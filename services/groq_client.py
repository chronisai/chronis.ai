"""
services/groq_client.py  —  GroqClient

Proper service wrapper for all Groq API calls.

Centralises:
  - Streaming chat completions (LLM pipeline)
  - Audio transcription via Whisper large-v3 (demo analyze route)
  - Summarization (memory service)
  - Exponential backoff retry on rate limits (MINOR risk from blueprint)

LLM pipeline calls this via async streaming.
Memory service calls this for summarization.
Demo routes call this synchronously for profile building.

Rate limit strategy:
  Groq free tier has per-minute token limits.
  Under burst load (multiple concurrent sessions), limits can be hit.
  Exponential backoff with jitter: 1s, 2s, 4s, up to 3 retries.
  On persistent failure, raises so the caller can degrade gracefully.
"""

import asyncio
import json
import os
import random
import time
from typing import AsyncGenerator, Dict, List, Optional

import httpx
import requests  # sync version for demo routes

GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Primary model for persona chat — best quality
GROQ_CHAT_MODEL = "llama-3.3-70b-versatile"
# Fast model for summarization — quality less critical, speed matters
GROQ_FAST_MODEL = "llama-3.1-8b-instant"
# Whisper model for transcription
GROQ_STT_MODEL  = "whisper-large-v3"

# Retry config
MAX_RETRIES    = 3
RETRY_BASE_S   = 1.0   # first retry after 1s
RETRY_MAX_S    = 8.0   # cap at 8s


class GroqClient:
    """
    Async Groq client with retry logic and streaming support.
    One instance shared across all sessions via the singleton below.
    """

    def __init__(self):
        self._http = httpx.AsyncClient(
            base_url=GROQ_BASE_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type":  "application/json",
            },
            timeout=60.0,
        )

    async def close(self):
        await self._http.aclose()

    # ────────────────────────────────────────────────────────────────────────
    # Streaming chat completion (used by LLMPipeline)
    # ────────────────────────────────────────────────────────────────────────

    async def stream_chat(
        self,
        messages: List[Dict],
        model: str = GROQ_CHAT_MODEL,
        temperature: float = 0.85,
        max_tokens: int = 512,
    ) -> AsyncGenerator[str, None]:
        """
        Yield tokens from a streaming chat completion.

        Usage in LLMPipeline:
            async for token in groq.stream_chat(messages):
                if ctrl.cancel_generation:
                    break
                await chunker.feed(token)

        Handles rate limit retries transparently.
        On persistent rate limit: raises RuntimeError.
        """
        payload = {
            "model":       model,
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  max_tokens,
            "stream":      True,
        }

        for attempt in range(MAX_RETRIES + 1):
            try:
                async with self._http.stream("POST", "/chat/completions",
                                             json=payload) as resp:
                    if resp.status_code == 429:
                        # Rate limited — extract retry-after if present
                        retry_after = float(
                            resp.headers.get("retry-after", RETRY_BASE_S * (2 ** attempt))
                        )
                        jitter = random.uniform(0, 0.5)
                        wait   = min(retry_after + jitter, RETRY_MAX_S)
                        print(f"[Groq] Rate limited — waiting {wait:.1f}s "
                              f"(attempt {attempt+1}/{MAX_RETRIES+1})", flush=True)
                        await asyncio.sleep(wait)
                        continue

                    if not resp.is_success:
                        body = await resp.aread()
                        raise RuntimeError(f"Groq error {resp.status_code}: {body[:300]}")

                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        line = line[6:]
                        if line == "[DONE]":
                            return
                        try:
                            data  = json.loads(line)
                            delta = (data.get("choices", [{}])[0]
                                         .get("delta", {})
                                         .get("content") or "")
                            if delta:
                                yield delta
                        except (json.JSONDecodeError, IndexError, KeyError):
                            continue
                    return   # stream finished cleanly

            except httpx.RemoteProtocolError as e:
                # Connection reset mid-stream — common under load
                print(f"[Groq] Stream interrupted: {e}", flush=True)
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_BASE_S * (2 ** attempt))
                    continue
                raise

        raise RuntimeError(f"Groq rate limit persisted after {MAX_RETRIES} retries")

    # ────────────────────────────────────────────────────────────────────────
    # Non-streaming completion (used by MemoryService for summarization)
    # ────────────────────────────────────────────────────────────────────────

    async def complete(
        self,
        messages: List[Dict],
        model: str = GROQ_FAST_MODEL,
        temperature: float = 0.3,
        max_tokens: int = 500,
    ) -> str:
        """
        Single-shot completion. Returns the response text.
        Used for memory summarization — doesn't need streaming.
        """
        payload = {
            "model":       model,
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  max_tokens,
            "stream":      False,
        }

        for attempt in range(MAX_RETRIES + 1):
            try:
                r = await self._http.post("/chat/completions", json=payload)

                if r.status_code == 429:
                    wait = min(RETRY_BASE_S * (2 ** attempt) + random.uniform(0, 0.5),
                               RETRY_MAX_S)
                    await asyncio.sleep(wait)
                    continue

                if not r.is_success:
                    raise RuntimeError(f"Groq error {r.status_code}: {r.text[:300]}")

                data = r.json()
                return data["choices"][0]["message"]["content"]

            except httpx.TimeoutException:
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_BASE_S * (2 ** attempt))
                    continue
                raise

        raise RuntimeError(f"Groq failed after {MAX_RETRIES} retries")

    # ────────────────────────────────────────────────────────────────────────
    # Sync wrappers (for demo routes that aren't async)
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def transcribe_sync(file_path: str, filename: str) -> tuple[Optional[str], Optional[str]]:
        """
        Transcribe audio file using Whisper large-v3.
        Synchronous — used by /api/analyze demo route.
        Returns (transcript_text, error_message).
        """
        if not GROQ_API_KEY:
            return None, "GROQ_API_KEY not set"

        mime_map = {
            "mp3": "audio/mpeg", "wav": "audio/wav", "m4a": "audio/mp4",
            "ogg": "audio/ogg",  "flac": "audio/flac", "aac": "audio/aac",
            "mp4": "video/mp4",  "mov": "video/quicktime", "webm": "video/webm",
        }
        ext  = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        mime = mime_map.get(ext, "application/octet-stream")

        for attempt in range(MAX_RETRIES + 1):
            try:
                with open(file_path, "rb") as f:
                    resp = requests.post(
                        f"{GROQ_BASE_URL}/audio/transcriptions",
                        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                        files={"file": (filename, f, mime)},
                        data={"model": GROQ_STT_MODEL,
                              "response_format": "verbose_json",
                              "language": "en"},
                        timeout=180,
                    )

                if resp.status_code == 429:
                    wait = min(RETRY_BASE_S * (2 ** attempt) + random.uniform(0, 0.5),
                               RETRY_MAX_S)
                    time.sleep(wait)
                    continue

                data = resp.json()
                if resp.ok:
                    return data.get("text", ""), None
                return None, data.get("error", {}).get("message", "Groq STT error")

            except Exception as e:
                return None, str(e)

        return None, "Groq STT rate limit — try again"

    @staticmethod
    def complete_sync(
        messages: List[Dict],
        model: str = GROQ_CHAT_MODEL,
        temperature: float = 0.85,
        max_tokens: int = 512,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Synchronous single-shot completion.
        Used by demo chat route.
        Returns (response_text, error_message).
        """
        if not GROQ_API_KEY:
            return None, "GROQ_API_KEY not set"

        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    f"{GROQ_BASE_URL}/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                             "Content-Type": "application/json"},
                    json={"model": model, "messages": messages,
                          "temperature": temperature, "max_tokens": max_tokens},
                    timeout=30,
                )

                if resp.status_code == 429:
                    wait = min(RETRY_BASE_S * (2 ** attempt) + random.uniform(0, 0.5),
                               RETRY_MAX_S)
                    time.sleep(wait)
                    continue

                data = resp.json()
                if resp.ok:
                    return data["choices"][0]["message"]["content"], None
                return None, data.get("error", {}).get("message", "Groq error")

            except Exception as e:
                return None, str(e)

        return None, "Groq rate limit — try again"


# ── Module-level singleton ────────────────────────────────────────────────────
_groq_client: Optional[GroqClient] = None

def get_groq_client() -> GroqClient:
    global _groq_client
    if _groq_client is None:
        _groq_client = GroqClient()
    return _groq_client
