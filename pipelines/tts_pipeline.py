"""
pipelines/tts_pipeline.py  —  TTSPipeline

Converts sentences from the chunker into 16kHz PCM audio for the avatar.

Fixes applied:
  1. Removed duplicate bus subscriptions from __init__ — only subscribe in _run().
     Dead queues in __init__ were never consumed, eventually filling up and
     making every emit() block for 100ms. Speech stutter in a live demo.
  2. tts.buffer_empty is now actually emitted — listens for llm.turn_done,
     drains remaining sentences, then emits buffer_empty so AvatarPipeline
     can transition back to LISTENING.
  3. In-process 24→16kHz resampling (no per-chunk FFmpeg subprocess).
  4. first_chunk_after_interrupt uses sendImmediate() exactly once per interrupt.
"""

import asyncio
import os
from typing import Optional, TYPE_CHECKING

import httpx
import numpy as np

try:
    from scipy.signal import resample_poly
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from session.controller import State

if TYPE_CHECKING:
    from session.controller import SessionController
    from session.event_bus import EventBus

MODAL_XTTS_URL     = os.environ.get("MODAL_XTTS_URL", "")
XTTS_SAMPLE_RATE   = 24000
TARGET_SAMPLE_RATE = 16000


def _resample_24k_to_16k(pcm_bytes: bytes) -> bytes:
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if _HAS_SCIPY:
        resampled = resample_poly(samples, up=2, down=3)
    else:
        ratio       = TARGET_SAMPLE_RATE / XTTS_SAMPLE_RATE
        new_length  = int(len(samples) * ratio)
        old_indices = np.linspace(0, len(samples) - 1, new_length)
        resampled   = np.interp(old_indices, np.arange(len(samples)), samples)
    return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()


class TTSPipeline:

    def __init__(self, ctrl, bus, voice_ref_path: str):
        self.ctrl      = ctrl
        self.bus       = bus
        self.voice_ref = voice_ref_path

        # first_chunk_after_interrupt: set True on interrupt, consumed once by
        # avatar pipeline via the chunk payload flag, then reset to False
        self._first_chunk_after_interrupt = False

        self._task: Optional[asyncio.Task] = None
        self._http = httpx.AsyncClient(timeout=30.0)

        # DO NOT subscribe in __init__ — only subscribe inside _run().
        # Subscriptions in __init__ create queues that are never consumed,
        # which causes emit() to block after 50 events. Speech stutter.

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._http.aclose()

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def _run(self) -> None:
        # Subscribe HERE, inside _run, not in __init__
        sentence_q   = self.bus.subscribe("llm.sentence_ready")
        interrupt_q  = self.bus.subscribe("session.interrupt")
        turn_done_q  = self.bus.subscribe("llm.turn_done")
        end_q        = self.bus.subscribe("session.end")

        while not self.ctrl.dead.is_set():
            try:
                s_f = asyncio.ensure_future(sentence_q.get())
                i_f = asyncio.ensure_future(interrupt_q.get())
                d_f = asyncio.ensure_future(turn_done_q.get())
                e_f = asyncio.ensure_future(end_q.get())

                done, pending = await asyncio.wait(
                    [s_f, i_f, d_f, e_f], return_when=asyncio.FIRST_COMPLETED
                )
                for f in pending:
                    f.cancel()

                if self.ctrl.dead.is_set():
                    break

                # ── Session end ────────────────────────────────────────────
                if e_f in done:
                    self.ctrl.tts_flushed.set()
                    break

                # ── Interrupt ─────────────────────────────────────────────
                if i_f in done:
                    # Drain any queued sentences without synthesizing
                    while not sentence_q.empty():
                        try:
                            sentence_q.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    self._first_chunk_after_interrupt = True
                    self.ctrl.tts_flushed.set()
                    print("[TTS] Flushed ✓ (interrupt)", flush=True)
                    continue

                # ── LLM turn done — drain remaining sentences then signal ──
                if d_f in done:
                    turn_data = d_f.result()
                    cancelled = turn_data.get("cancelled", False)

                    if not cancelled:
                        # Process any sentences already queued
                        while not sentence_q.empty():
                            try:
                                payload  = sentence_q.get_nowait()
                                sentence = payload.get("text", "").strip()
                                if sentence and not self.ctrl.cancel_generation:
                                    await self._synthesize(sentence)
                            except asyncio.QueueEmpty:
                                break

                    # All sentences for this turn are done — signal avatar
                    await self.bus.emit("tts.buffer_empty")
                    print("[TTS] buffer_empty emitted ✓", flush=True)
                    continue

                # ── New sentence ───────────────────────────────────────────
                if s_f in done:
                    sentence = s_f.result().get("text", "").strip()
                    if sentence and not self.ctrl.cancel_generation:
                        # Race synthesis against the interrupt queue.
                        # XTTS blocks for the full waveform before yielding chunks,
                        # so without this, an interrupt during synthesis cannot be
                        # processed until _synthesize() returns (could be 2-3s).
                        synth_task = asyncio.create_task(self._synthesize(sentence))
                        intr_watch = asyncio.ensure_future(interrupt_q.get())

                        syn_done, _ = await asyncio.wait(
                            [synth_task, intr_watch],
                            return_when=asyncio.FIRST_COMPLETED,
                        )

                        if intr_watch in syn_done:
                            # Interrupt fired while synthesis was running
                            synth_task.cancel()
                            try:
                                await synth_task
                            except asyncio.CancelledError:
                                pass
                            # Drain any queued sentences for this turn
                            while not sentence_q.empty():
                                try:
                                    sentence_q.get_nowait()
                                except asyncio.QueueEmpty:
                                    break
                            self._first_chunk_after_interrupt = True
                            self.ctrl.tts_flushed.set()
                            print("[TTS] Flushed ✓ (interrupt during synthesis)", flush=True)
                        else:
                            # Synthesis completed — cancel the interrupt watcher.
                            # Cancelling a pending queue.get() does NOT consume the
                            # item, so any interrupt stays in the queue for next loop.
                            intr_watch.cancel()
                            try:
                                await intr_watch
                            except (asyncio.CancelledError, Exception):
                                pass

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[TTS] Loop error: {e}", flush=True)
                await asyncio.sleep(0.1)

        self.ctrl.tts_flushed.set()

    # ── Synthesis ─────────────────────────────────────────────────────────────

    async def _synthesize(self, text: str) -> None:
        if not MODAL_XTTS_URL:
            print("[TTS] MODAL_XTTS_URL not configured — skipping", flush=True)
            return

        print(f"[TTS] Synthesizing: {text[:50]}...", flush=True)

        if self.ctrl.state == State.THINKING:
            await self.ctrl.transition(State.SPEAKING)

        is_first_chunk = True

        try:
            async with self._http.stream(
                "POST",
                MODAL_XTTS_URL,
                json={"text": text, "speaker_wav": self.voice_ref,
                      "language": "en", "stream": True},
                headers={"Content-Type": "application/json"},
            ) as response:
                if not response.is_success:
                    body = await response.aread()
                    print(f"[TTS] Modal error {response.status_code}: {body[:200]}",
                          flush=True)
                    return

                chunk_acc = b""
                async for chunk in response.aiter_bytes(chunk_size=4096):
                    if self.ctrl.dead.is_set() or self.ctrl.cancel_generation:
                        break

                    chunk_acc += chunk
                    usable = len(chunk_acc) - (len(chunk_acc) % 2)
                    if usable < 2:
                        continue

                    pcm_24k   = chunk_acc[:usable]
                    chunk_acc = chunk_acc[usable:]
                    pcm_16k   = _resample_24k_to_16k(pcm_24k)

                    use_immediate = is_first_chunk and self._first_chunk_after_interrupt
                    await self.bus.emit("tts.chunk_ready", {
                        "audio":                       pcm_16k,
                        "first_chunk_after_interrupt": use_immediate,
                    })

                    if use_immediate:
                        self._first_chunk_after_interrupt = False
                    is_first_chunk = False

        except Exception as e:
            if not (self.ctrl.cancel_generation or self.ctrl.dead.is_set()):
                print(f"[TTS] Synthesis error: {e}", flush=True)

        self.ctrl._enqueue_log("tts.sentence_complete", "tts", {"text": text[:50]})
