"""
session/event_bus.py  —  EventBus

Per-session asyncio.Queue-based pub/sub.

Rules:
  - ONE EventBus instance per SessionController. NEVER a global.
    Global buses with concurrent sessions produce cross-talk that is
    literally impossible to reproduce consistently.

  - Queue-based (not dict callbacks). Under 50 VAD frames/sec,
    a synchronous dict becomes a latency bottleneck.

  - VAD-sourced events use BOUNDED queues with a drop policy.
    If the consumer is too slow (e.g. Deepgram backpressure), we drop
    old frames rather than buffering unboundedly and exploding memory.

Events emitted in this system:
  user.speech_start       — VAD detected voice onset
  user.speech_end         — VAD detected silence > 500ms
  stt.utterance_complete  — Final transcript ready from Deepgram
  llm.sentence_ready      — Sentence chunker has a complete sentence
  tts.chunk_ready         — Converted audio chunk ready for avatar
  tts.buffer_empty        — All TTS sentences have been synthesized
  avatar.render_complete  — Simli has drained its buffer
  session.interrupt       — Kill signal to all pipelines (from interrupt())
  session.end             — Graceful teardown signal
  pipeline.error          — Error from any pipeline, forwarded to WS client
"""

import asyncio
from collections import defaultdict
from typing import Dict, List


# Maximum queue depth for high-frequency VAD events.
_VAD_QUEUE_MAXSIZE    = 100

# All other events — generous but bounded to catch bugs
_DEFAULT_QUEUE_MAXSIZE = 50

# High-frequency events that use drop-policy instead of blocking.
_HIGH_FREQ_EVENTS = {"user.speech_start", "user.speech_end", "tts.chunk_ready"}


class EventBus:

    def __init__(self):
        # event_type → list of asyncio.Queues (one per subscriber)
        self._queues: Dict[str, List[asyncio.Queue]] = defaultdict(list)

    def subscribe(self, event_type: str) -> asyncio.Queue:
        """
        Register interest in an event type.
        Returns a Queue the caller reads from.

        Usage:
            q = bus.subscribe("stt.utterance_complete")
            payload = await q.get()
        """
        maxsize = (
            _VAD_QUEUE_MAXSIZE
            if event_type in _HIGH_FREQ_EVENTS
            else _DEFAULT_QUEUE_MAXSIZE
        )
        q = asyncio.Queue(maxsize=maxsize)
        self._queues[event_type].append(q)
        return q

    def unsubscribe(self, event_type: str, queue: asyncio.Queue) -> None:
        """
        Remove a specific queue from the subscribers list for event_type.
        Call this in finally blocks to prevent dangling queue references
        that would block emit() after the subscriber has exited.

        Usage:
            q = bus.subscribe("pipeline.error")
            try:
                ...
            finally:
                bus.unsubscribe("pipeline.error", q)
        """
        queues = self._queues.get(event_type)
        if queues:
            try:
                queues.remove(queue)
            except ValueError:
                pass  # already removed or never subscribed — safe to ignore

    async def emit(self, event_type: str, payload: dict = None) -> None:
        """
        Broadcast event_type to all subscribers.

        For VAD events: if a subscriber's queue is full, drop the event
        for that subscriber (don't block the audio pipeline).

        For all other events: block briefly until the queue accepts the entry.
        These are low-frequency and should never be dropped.
        """
        payload = payload or {}
        for q in self._queues.get(event_type, []):
            if event_type in _HIGH_FREQ_EVENTS:
                # Drop policy for high-frequency events
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    try:
                        q.get_nowait()       # discard oldest
                        q.put_nowait(payload)
                    except (asyncio.QueueEmpty, asyncio.QueueFull):
                        pass
            else:
                # Low-frequency events — wait up to 100ms before giving up
                try:
                    await asyncio.wait_for(q.put(payload), timeout=0.1)
                except asyncio.TimeoutError:
                    pass

    def unsubscribe_all(self) -> None:
        """
        Clear all subscriptions. Called during session teardown to release
        all queue references and allow GC.
        """
        self._queues.clear()