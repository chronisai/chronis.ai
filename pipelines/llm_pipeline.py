"""
pipelines/llm_pipeline.py  —  LLMPipeline

Streams tokens from Groq → sentence_chunker → TTS pipeline.

Fixes applied:
  1. Uses GroqClient.stream_chat() — no self._http (was crashing on first utterance)
  2. Idle interrupt watcher — sets llm_flushed immediately when interrupt fires
     while LLM is not generating (most common case: avatar still speaking
     after LLM already finished — previously caused 2s barrier timeout every time)
  3. Emits llm.turn_done so TTS knows when to emit tts.buffer_empty
"""

import asyncio
from typing import TYPE_CHECKING

from session.controller import State
from services.groq_client import get_groq_client

if TYPE_CHECKING:
    from session.controller import SessionController
    from session.event_bus import EventBus
    from utils.sentence_chunker import SentenceChunker
    from services.memory_service import MemoryService


class LLMPipeline:

    def __init__(self, ctrl, bus, chunker, memory):
        self.ctrl    = ctrl
        self.bus     = bus
        self.chunker = chunker
        self.memory  = memory

        # Shared singleton — DO NOT aclose(), it's used across sessions
        self._groq = get_groq_client()

        self._utterance_q   = bus.subscribe("stt.utterance_complete")
        self._end_q         = bus.subscribe("session.end")

        # True while _handle_utterance is streaming tokens
        # The idle interrupt watcher reads this to decide whether to set
        # llm_flushed immediately or let the stream exit do it
        self._is_generating = False

        self._task: asyncio.Task | None = None
        self._idle_interrupt_task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())
        self._idle_interrupt_task = asyncio.create_task(self._idle_interrupt_watcher())

    async def stop(self) -> None:
        for t in [self._task, self._idle_interrupt_task]:
            if t and not t.done():
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def _run(self) -> None:
        while not self.ctrl.dead.is_set():
            try:
                utt_f = asyncio.ensure_future(self._utterance_q.get())
                end_f = asyncio.ensure_future(self._end_q.get())

                done, pending = await asyncio.wait(
                    [utt_f, end_f], return_when=asyncio.FIRST_COMPLETED
                )
                for f in pending:
                    f.cancel()

                if self.ctrl.dead.is_set() or end_f in done:
                    break

                text = utt_f.result().get("text", "").strip()
                if text:
                    await self._handle_utterance(text)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[LLM] Loop error: {e}", flush=True)
                await asyncio.sleep(0.1)

    # ── Idle interrupt watcher ─────────────────────────────────────────────────

    async def _idle_interrupt_watcher(self) -> None:
        """
        Dedicated listener for session.interrupt events.

        If LLM is idle (not generating) when interrupt fires:
          → set llm_flushed immediately → barrier clears in ~0ms

        If LLM is generating:
          → _handle_utterance will set llm_flushed when the stream exits
          → this watcher's set() call is harmless (asyncio.Event is idempotent)

        Without this, interrupting the avatar after LLM finishes always
        hit the 2s barrier timeout — the most common real-world interrupt case.
        """
        interrupt_q = self.bus.subscribe("session.interrupt")
        end_q       = self.bus.subscribe("session.end")

        while not self.ctrl.dead.is_set():
            try:
                i_f = asyncio.ensure_future(interrupt_q.get())
                e_f = asyncio.ensure_future(end_q.get())

                done, pending = await asyncio.wait(
                    [i_f, e_f], return_when=asyncio.FIRST_COMPLETED
                )
                for f in pending:
                    f.cancel()

                if self.ctrl.dead.is_set() or e_f in done:
                    break

                if not self._is_generating:
                    self.ctrl.llm_flushed.set()
                    print("[LLM] Idle interrupt → llm_flushed set immediately ✓",
                          flush=True)
                # If generating: stream loop sees cancel_generation=True,
                # exits, then sets llm_flushed itself.

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[LLM] Idle watcher error: {e}", flush=True)
                await asyncio.sleep(0.05)

    # ── Utterance handling ────────────────────────────────────────────────────

    async def _handle_utterance(self, text: str) -> None:
        print(f"[LLM] Utterance: {text[:60]}...", flush=True)

        self._is_generating = True
        await self.ctrl.transition(State.THINKING)
        self.chunker.start()

        messages      = self.memory.build_messages(text)
        full_response = ""

        try:
            async for token in self._groq.stream_chat(
                messages=messages,
                temperature=0.85,
                max_tokens=512,
            ):
                if self.ctrl.dead.is_set():
                    break
                if self.ctrl.cancel_generation:
                    print("[LLM] Cancelled mid-stream ✓", flush=True)
                    break

                full_response += token
                await self.chunker.feed(token)

        except Exception as e:
            if not (self.ctrl.cancel_generation or self.ctrl.dead.is_set()):
                print(f"[LLM] Stream error: {e}", flush=True)

        # Flush remaining chunker buffer or discard on interrupt
        if not self.ctrl.cancel_generation and not self.ctrl.dead.is_set():
            await self.chunker.flush()
            if full_response.strip():
                self.ctrl.turn_count += 1
                asyncio.create_task(
                    self.memory.record_turn(text, full_response, self.ctrl.turn_count)
                )
        else:
            await self.chunker.reset()

        self._is_generating = False

        # Emit turn_done BEFORE setting llm_flushed so TTS sees it first
        await self.bus.emit("llm.turn_done", {"cancelled": self.ctrl.cancel_generation})

        # Signal barrier
        self.ctrl.llm_flushed.set()

        self.ctrl._enqueue_log("llm.turn_complete", "llm", {
            "response_chars": len(full_response),
            "cancelled":      self.ctrl.cancel_generation,
        })
