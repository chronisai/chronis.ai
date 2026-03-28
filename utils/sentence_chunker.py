"""
utils/sentence_chunker.py  —  SentenceChunker

Converts a streaming token feed into complete sentences for TTS.

Three rules in priority order:
  1. Punctuation boundary  — [.!?] followed by space + capital, no abbreviation match
  2. Minimum word gate     — ≥ 8 words before emitting (XTTS degrades on short fragments)
  3. Fallback flush        — 1.2s without a sentence boundary → flush anyway

Critical fix (#4 from code review):
  The fallback MUST be a real background asyncio task that loops independently.
  Checking last_token_time only on token arrival does NOT work — if token
  flow pauses without punctuation, nothing forces the flush until the next
  token arrives. The background task runs every 50ms and forces a flush when
  the buffer has been sitting idle for ≥ 1.2s.
"""

import asyncio
import re
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from session.event_bus import EventBus

# Flush timeout in seconds — 1.2s feels more responsive than 1.5s
FLUSH_TIMEOUT_SECS = 1.2

# Minimum word count before emitting to TTS.
# XTTS v2 quality degrades significantly on fragments under 8 words.
MIN_WORDS = 8

# Abbreviations that end with a period but do NOT end a sentence.
# If the buffer ends with one of these, we skip the punctuation boundary check.
ABBREVIATIONS = {
    "Dr", "Mr", "Mrs", "Ms", "Prof", "Sr", "Jr", "Lt", "Col", "Gen",
    "U.S", "U.K", "U.N", "e.g", "i.e", "vs", "etc", "approx", "est",
    "Ave", "Blvd", "St", "No",
}

# Pre-compiled: sentence-ending punctuation followed by whitespace + uppercase
_SENTENCE_END_RE = re.compile(r'[.!?]["\']?\s+[A-Z]')


def _word_count(text: str) -> int:
    return len(text.split())


def _is_abbreviation_end(text: str) -> bool:
    """
    Return True if the text ends with a known abbreviation + period.
    Used to prevent splitting on "Dr. Smith" or "U.S. Army".
    """
    stripped = text.rstrip()
    for abbr in ABBREVIATIONS:
        if stripped.endswith(abbr + "."):
            return True
    return False


def _find_sentence_boundary(text: str) -> int:
    """
    Return the index just AFTER the first valid sentence-ending punctuation.
    Returns -1 if no valid boundary found.

    Valid boundary: [.!?] followed by whitespace + capital letter,
    AND the text up to that point does not end with a known abbreviation.
    """
    for m in _SENTENCE_END_RE.finditer(text):
        # m.start() is the index of the punctuation character
        candidate = text[: m.start() + 1]   # everything up to and including the punct
        if not _is_abbreviation_end(candidate):
            # +1 to include the punctuation itself
            return m.start() + 1
    return -1


class SentenceChunker:
    """
    Feed tokens in → get complete sentences out via event bus.

    Lifecycle:
        chunker = SentenceChunker(bus, ctrl)
        chunker.start()        # starts background flush timer
        await chunker.feed(token)
        await chunker.flush()  # call on LLM stream end
        chunker.stop()         # stops background timer
    """

    def __init__(self, bus: "EventBus", session_id: str):
        self.bus        = bus
        self.session_id = session_id

        self._buffer         = ""         # accumulates tokens
        self._last_feed_time = 0.0        # time of most recent token feed
        self._timer_task: asyncio.Task | None = None
        self._active         = False      # False = don't flush (between turns)
        self._lock           = asyncio.Lock()

    # ────────────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Arm the chunker for a new LLM turn.
        Starts the background flush timer task.
        """
        self._active         = True
        self._last_feed_time = time.monotonic()
        if self._timer_task is None or self._timer_task.done():
            self._timer_task = asyncio.create_task(self._flush_watchdog())

    def stop(self) -> None:
        """Disarm. Cancels the background timer. Call between turns."""
        self._active = False
        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()
        self._timer_task = None

    async def feed(self, token: str) -> None:
        """
        Receive one token from the LLM stream.
        Checks for sentence boundary and emits if found.
        """
        if not self._active:
            return

        async with self._lock:
            self._buffer         += token
            self._last_feed_time  = time.monotonic()
            await self._try_emit()

    async def flush(self) -> None:
        """
        Force-emit whatever is left in the buffer.
        Called when the LLM stream ends — emit the final fragment
        even if it doesn't meet the punctuation or word-count rules.
        """
        async with self._lock:
            text = self._buffer.strip()
            if text:
                await self._emit_sentence(text)
            self._buffer = ""
        self.stop()

    async def reset(self) -> None:
        """Discard buffer without emitting. Called on interrupt."""
        async with self._lock:
            self._buffer = ""
        self.stop()

    # ────────────────────────────────────────────────────────────────────────
    # Internal
    # ────────────────────────────────────────────────────────────────────────

    async def _try_emit(self) -> None:
        """
        Check if the buffer contains a complete sentence.
        Called after every token feed (inside _lock).
        """
        idx = _find_sentence_boundary(self._buffer)
        if idx == -1:
            return   # no boundary found yet

        sentence = self._buffer[:idx].strip()
        remainder = self._buffer[idx:].lstrip()

        if _word_count(sentence) < MIN_WORDS:
            # Sentence too short for XTTS — accumulate into next sentence
            # Don't emit, don't trim the buffer
            return

        await self._emit_sentence(sentence)
        self._buffer = remainder

        # Recursively check the remainder for more sentences
        if remainder:
            await self._try_emit()

    async def _flush_watchdog(self) -> None:
        """
        Background timer task — the key fix from code review.

        Runs every 50ms. If the buffer has content and hasn't been fed
        a new token in ≥ FLUSH_TIMEOUT_SECS, force-flush the buffer.

        This catches the case where the LLM generates a long run-on
        response with no punctuation — without this task, the avatar
        would just wait silently until the next token arrives.
        """
        try:
            while self._active:
                await asyncio.sleep(0.05)   # check every 50ms

                if not self._active:
                    break

                async with self._lock:
                    # Nothing in buffer — nothing to do
                    if not self._buffer.strip():
                        continue

                    age = time.monotonic() - self._last_feed_time
                    if age >= FLUSH_TIMEOUT_SECS:
                        # Buffer has been sitting idle for 1.2s — force flush
                        text = self._buffer.strip()
                        if text:
                            await self._emit_sentence(text)
                        self._buffer = ""

        except asyncio.CancelledError:
            pass   # normal — stop() cancels this task between turns

    async def _emit_sentence(self, text: str) -> None:
        """Emit a sentence to the event bus for TTS pickup."""
        await self.bus.emit("llm.sentence_ready", {
            "text":       text,
            "session_id": self.session_id,
        })
