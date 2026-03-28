"""
pipelines/avatar_pipeline.py  —  AvatarPipeline

Receives 16kHz PCM chunks from TTS → 120ms buffer → Simli.

Fixes applied:
  1. Removed duplicate bus subscriptions from __init__ — only subscribe in _run().
     Dead queues were filling up and causing 100ms stalls on every chunk emit.
  2. tts.buffer_empty now received correctly (TTS now actually emits it).
  3. sendImmediate() used exactly once after interrupt, then normal buffering resumes.
"""

import asyncio
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from session.controller import SessionController
    from session.event_bus import EventBus
    from services.simli_client import SimliClient

BUFFER_MS    = 120
BUFFER_BYTES = int(BUFFER_MS * 16000 * 2 / 1000)   # 3840 bytes


class AvatarPipeline:

    def __init__(self, ctrl, bus, simli):
        self.ctrl   = ctrl
        self.bus    = bus
        self.simli  = simli

        self._audio_buf = b""
        self._task: Optional[asyncio.Task] = None

        # DO NOT subscribe in __init__ — only inside _run().
        # Dead queues from __init__ subscriptions never get consumed,
        # fill up, and block every emit() call for 100ms each.

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.simli.stop()

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def _run(self) -> None:
        # Subscribe HERE only — not in __init__
        chunk_q     = self.bus.subscribe("tts.chunk_ready")
        interrupt_q = self.bus.subscribe("session.interrupt")
        empty_q     = self.bus.subscribe("tts.buffer_empty")
        end_q       = self.bus.subscribe("session.end")

        while not self.ctrl.dead.is_set():
            try:
                c_f = asyncio.ensure_future(chunk_q.get())
                i_f = asyncio.ensure_future(interrupt_q.get())
                e_f = asyncio.ensure_future(empty_q.get())
                x_f = asyncio.ensure_future(end_q.get())

                done, pending = await asyncio.wait(
                    [c_f, i_f, e_f, x_f], return_when=asyncio.FIRST_COMPLETED
                )
                for f in pending:
                    f.cancel()

                if self.ctrl.dead.is_set():
                    break

                # ── Session end ────────────────────────────────────────────
                if x_f in done:
                    await self._flush_buffer()
                    break

                # ── Interrupt ─────────────────────────────────────────────
                if i_f in done:
                    await self._handle_interrupt()
                    continue

                # ── TTS buffer empty — turn complete ───────────────────────
                if e_f in done:
                    await self._handle_tts_complete()
                    continue

                # ── Audio chunk ────────────────────────────────────────────
                if c_f in done:
                    payload = c_f.result()
                    await self._handle_chunk(
                        audio=payload.get("audio", b""),
                        use_immediate=payload.get("first_chunk_after_interrupt", False),
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Avatar] Loop error: {e}", flush=True)
                await asyncio.sleep(0.05)

        self.ctrl.avatar_flushed.set()

    # ── Audio handling ────────────────────────────────────────────────────────

    async def _handle_chunk(self, audio: bytes, use_immediate: bool) -> None:
        if not audio:
            return

        if use_immediate:
            # First chunk after interrupt bypasses Simli's queue
            self._audio_buf = b""
            await self.simli.send_immediate(audio)
            print("[Avatar] sendImmediate() ✓ (post-interrupt first chunk)", flush=True)
            return

        self._audio_buf += audio
        while len(self._audio_buf) >= BUFFER_BYTES:
            await self.simli.send(self._audio_buf[:BUFFER_BYTES])
            self._audio_buf = self._audio_buf[BUFFER_BYTES:]

    async def _flush_buffer(self) -> None:
        if self._audio_buf:
            await self.simli.send(self._audio_buf)
            self._audio_buf = b""

    async def _handle_interrupt(self) -> None:
        self._audio_buf = b""
        await self.simli.clear_buffer()
        self.ctrl.avatar_flushed.set()
        print("[Avatar] Interrupt handled, clearBuffer() ✓", flush=True)

    async def _handle_tts_complete(self) -> None:
        await self._flush_buffer()
        await self.bus.emit("avatar.render_complete")

        from session.controller import State
        if self.ctrl.state == State.SPEAKING:
            await self.ctrl.transition(State.LISTENING)

        self.ctrl.avatar_flushed.set()
        self.ctrl._enqueue_log("avatar.render_complete", "avatar")
        print("[Avatar] Render complete — back to LISTENING ✓", flush=True)
