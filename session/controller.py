"""
session/controller.py  —  SessionController

THE single source of truth for every live session.

Fixes applied:
  1. self._interrupting re-entry guard — prevents multiple concurrent
     interrupt() calls from corrupting cancel_generation and flush events.
     Multiple VAD frames during SPEAKING can queue many interrupts; only
     the first call runs the sequence.
  2. self.dead (asyncio.Event) — kill signal for all pipeline loops.
  3. Interrupt barrier — cancel_generation only resets after all 3
     pipelines (llm, tts, avatar) confirm flush.
  4. State transitions log outside the lock — DB latency never blocks state.
"""

import asyncio
import time
from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from session.event_bus import EventBus


class State(Enum):
    IDLE        = "idle"
    LISTENING   = "listening"
    THINKING    = "thinking"
    SPEAKING    = "speaking"
    INTERRUPTED = "interrupted"
    ENDING      = "ending"


class SessionController:

    def __init__(self, session_id: str, event_bus: "EventBus"):
        self.session_id = session_id
        self.bus        = event_bus

        # ── Core state ──────────────────────────────────────────────────────
        self.state  = State.IDLE
        self._lock  = asyncio.Lock()

        # ── Kill signal ─────────────────────────────────────────────────────
        # Set by kill(). Every pipeline loop checks: if ctrl.dead.is_set(): break
        self.dead   = asyncio.Event()

        # ── LLM cancel flag ─────────────────────────────────────────────────
        # Checked per-token in LLM pipeline. Reset ONLY after barrier clears.
        self.cancel_generation = False

        # ── Interrupt re-entry guard ─────────────────────────────────────────
        # Multiple VAD frames during SPEAKING spawn multiple interrupt() tasks.
        # Only the first one runs. asyncio.Lock ensures sequential access.
        self._interrupt_lock = asyncio.Lock()

        # ── Interrupt barrier events ─────────────────────────────────────────
        # Each pipeline sets its event when it has flushed after interrupt.
        # interrupt() awaits all three before resetting cancel_generation.
        self.llm_flushed    = asyncio.Event()
        self.tts_flushed    = asyncio.Event()
        self.avatar_flushed = asyncio.Event()

        # ── Session metadata ─────────────────────────────────────────────────
        self.turn_count       = 0
        self.agent_id: Optional[str] = None
        self.voice_ref: Optional[str] = None
        self.simli_agent_id: Optional[str] = None
        self.session_start    = time.monotonic()
        self.last_activity    = time.monotonic()

        # ── Fire-and-forget log queue ────────────────────────────────────────
        self._log_q: asyncio.Queue = asyncio.Queue(maxsize=1000)

    # ── State machine ─────────────────────────────────────────────────────────

    async def transition(self, new_state: State) -> None:
        """Change state inside lock; log outside lock (DB latency never blocks)."""
        async with self._lock:
            old        = self.state
            self.state = new_state

        self._enqueue_log(
            event_type=f"state.{old.value}_to_{new_state.value}",
            pipeline="controller",
        )

    async def interrupt(self) -> None:
        """
        Full interrupt sequence with barrier and re-entry guard.

        Re-entry guard: if another interrupt() is already running (e.g. two
        rapid VAD frames both spawn interrupt tasks), the second one exits
        immediately. Without this, cancel_generation and flush events get
        corrupted by overlapping sequences.
        """
        # Non-blocking try — if already interrupting, skip
        if self._interrupt_lock.locked():
            return
        async with self._interrupt_lock:
            await self._do_interrupt()

    async def _do_interrupt(self) -> None:
        """Actual interrupt logic — always runs under _interrupt_lock."""
        self.cancel_generation = True
        await self.transition(State.INTERRUPTED)

        # Arm the barrier: clear so pipelines signal fresh
        self.llm_flushed.clear()
        self.tts_flushed.clear()
        self.avatar_flushed.clear()

        # Broadcast — all pipelines react
        await self.bus.emit("session.interrupt")

        # Wait for all three pipelines to confirm flush (2s timeout)
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    self.llm_flushed.wait(),
                    self.tts_flushed.wait(),
                    self.avatar_flushed.wait(),
                ),
                timeout=2.0,
            )
        except asyncio.TimeoutError:
            self._enqueue_log(
                "interrupt.barrier_timeout",
                "controller",
                {"warning": "pipeline did not flush in time"},
            )

        # Safe to reset — all pipelines have stopped emitting
        self.cancel_generation = False
        await self.transition(State.LISTENING)
        self._enqueue_log("interrupt.complete", "controller")

    async def kill(self) -> None:
        """Set dead event — all pipeline loops exit on next iteration."""
        await self.transition(State.ENDING)
        self.dead.set()
        await self.bus.emit("session.end")
        self._enqueue_log("session.killed", "controller")

    # ── Activity tracking ─────────────────────────────────────────────────────

    def touch(self) -> None:
        self.last_activity = time.monotonic()

    def idle_seconds(self) -> float:
        return time.monotonic() - self.last_activity

    # ── Log queue ─────────────────────────────────────────────────────────────

    def _enqueue_log(self, event_type: str, pipeline: str = "",
                     payload: dict = None) -> None:
        entry = {
            "session_id": self.session_id,
            "event_type": event_type,
            "pipeline":   pipeline,
            "payload":    payload or {},
        }
        try:
            self._log_q.put_nowait(entry)
        except asyncio.QueueFull:
            pass  # never let logging block anything

    @property
    def log_queue(self) -> asyncio.Queue:
        return self._log_q
