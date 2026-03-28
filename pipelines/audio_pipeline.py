"""
pipelines/audio_pipeline.py  —  AudioPipeline

Receives raw 16kHz mono PCM from the browser WebSocket.

Responsibilities:
  1. Split incoming chunks into 20ms frames for webrtcvad
  2. Run VAD on every frame
  3. Route audio to Deepgram when state = LISTENING
  4. Trigger interrupt when VAD fires and state = SPEAKING or THINKING
  5. Send KeepAlive to Deepgram during silence

Audio format assumption (fix #5):
  We LOG the actual incoming frame size, sample rate indicator, and byte
  count on the first 5 frames. If there's a format mismatch with what
  Daily.co sends, it shows up immediately in the logs — not 2 hours
  into the debug session.

VAD frame size:
  webrtcvad requires EXACTLY 10ms, 20ms, or 30ms frames at 16kHz.
  At 16kHz, 20ms = 320 samples = 640 bytes.
  We split incoming chunks into 640-byte frames and discard the remainder.
"""

import asyncio
import time
from typing import TYPE_CHECKING

import webrtcvad

from session.controller import State

if TYPE_CHECKING:
    from session.controller import SessionController
    from session.event_bus import EventBus
    from services.deepgram_client import DeepgramClient

# ── VAD configuration ─────────────────────────────────────────────────────────
VAD_SAMPLE_RATE  = 16000
VAD_FRAME_MS     = 20           # 20ms frames
VAD_FRAME_BYTES  = (VAD_SAMPLE_RATE * VAD_FRAME_MS // 1000) * 2  # 640 bytes (16-bit)
VAD_AGGRESSIVENESS = 2          # 0-3. 2 = balanced. 3 = very aggressive.

# Silence detection: this many consecutive silent frames = speech end signal
# 500ms / 20ms per frame = 25 frames
SILENCE_FRAMES_THRESHOLD = 25

# Keep-alive interval: send Deepgram KeepAlive every N seconds of silence
DEEPGRAM_KEEPALIVE_INTERVAL_S = 5.0


class AudioPipeline:
    """
    Manages all audio routing for one session.

    Usage:
        pipeline = AudioPipeline(ctrl, bus, deepgram)
        await pipeline.push(raw_bytes)  # called from WebSocket handler on each chunk
        await pipeline.close()          # called during session teardown
    """

    def __init__(
        self,
        ctrl: "SessionController",
        bus: "EventBus",
        deepgram: "DeepgramClient",
    ):
        self.ctrl     = ctrl
        self.bus      = bus
        self.deepgram = deepgram

        # VAD instance (one per session — not reused across sessions)
        self._vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

        # Rolling buffer for frame assembly
        # Incoming WebSocket chunks may not align to VAD frame boundaries
        self._frame_buf: bytes = b""

        # Silence / speech tracking
        self._silent_frames    = 0
        self._in_speech        = False
        self._last_keepalive   = time.monotonic()

        # Frame logging for audio format verification (fix #5)
        # Log first 5 frames in detail — surfacing format mismatches early
        self._frames_logged    = 0
        self._FORMAT_LOG_LIMIT = 5

    # ────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ────────────────────────────────────────────────────────────────────────

    async def push(self, raw_bytes: bytes) -> None:
        """
        Called by the WebSocket handler for every chunk from the browser.

        raw_bytes is 16kHz mono PCM (from Daily.co with EC/NS/AGC applied).
        We reassemble into 640-byte VAD frames and process each.
        """
        if self.ctrl.dead.is_set():
            return

        # ── Log format for the first few frames (fix #5) ──────────────────
        if self._frames_logged < self._FORMAT_LOG_LIMIT:
            self._log_format(raw_bytes)
            self._frames_logged += 1

        # Update activity timestamp (watchdog relies on this)
        self.ctrl.touch()

        # ── Append to frame buffer ─────────────────────────────────────────
        self._frame_buf += raw_bytes

        # ── Process complete 640-byte frames ──────────────────────────────
        while len(self._frame_buf) >= VAD_FRAME_BYTES:
            if self.ctrl.dead.is_set():
                return

            frame           = self._frame_buf[:VAD_FRAME_BYTES]
            self._frame_buf = self._frame_buf[VAD_FRAME_BYTES:]

            await self._process_frame(frame)

    # ────────────────────────────────────────────────────────────────────────
    # Per-frame processing
    # ────────────────────────────────────────────────────────────────────────

    async def _process_frame(self, frame: bytes) -> None:
        """
        Run VAD on one 20ms frame and route it appropriately.

        Two routing paths:
          A. State = LISTENING → forward to Deepgram for STT
          B. State = SPEAKING or THINKING → VAD activity triggers interrupt
        """
        try:
            is_speech = self._vad.is_speech(frame, VAD_SAMPLE_RATE)
        except Exception:
            # webrtcvad can throw on malformed frames — don't crash the pipeline
            return

        current_state = self.ctrl.state

        # ── Path A: We're listening — feed Deepgram ────────────────────────
        if current_state == State.LISTENING:
            if is_speech:
                self._silent_frames = 0
                self._in_speech     = True
                # Send the audio frame to Deepgram for transcription
                await self.deepgram.send_audio(frame)
            else:
                self._silent_frames += 1

                # Send silence frames to Deepgram too — it uses them for
                # endpointing (detecting end of utterance)
                if self._in_speech:
                    await self.deepgram.send_audio(frame)

                # Check if we've hit silence threshold
                if (self._in_speech and
                        self._silent_frames >= SILENCE_FRAMES_THRESHOLD):
                    # 500ms of silence after speech — signal end of utterance
                    # Deepgram's endpointing will also fire, but belt-and-suspenders
                    self._in_speech     = False
                    self._silent_frames = 0

                # Keep Deepgram connection alive during long silences
                now = time.monotonic()
                if now - self._last_keepalive > DEEPGRAM_KEEPALIVE_INTERVAL_S:
                    await self.deepgram.send_keep_alive()
                    self._last_keepalive = now

        # ── Path B: We're speaking/thinking — check for interrupt ─────────
        elif current_state in (State.SPEAKING, State.THINKING):
            if is_speech:
                # User is talking while avatar is speaking — INTERRUPT
                await self.bus.emit("user.speech_start", {
                    "while_state": current_state.value
                })
                # Trigger full interrupt sequence via controller
                asyncio.create_task(self.ctrl.interrupt())

        # IDLE / INTERRUPTED / ENDING states: drop all audio silently

    # ────────────────────────────────────────────────────────────────────────
    # Format logging (fix #5)
    # ────────────────────────────────────────────────────────────────────────

    def _log_format(self, data: bytes) -> None:
        """
        Log actual incoming frame details for format verification.
        Run this during smoke testing to confirm Daily.co is sending
        exactly 16kHz mono 16-bit PCM.

        Check the logs early: if chunk_bytes is consistently not a
        multiple of 640, the alignment is wrong. If it's 3840 bytes,
        that's 60ms of audio at 16kHz — fine. If it's odd bytes, suspect
        a format mismatch (e.g. 8kHz, stereo, or float32).
        """
        n         = len(data)
        frames_in = n // VAD_FRAME_BYTES
        remainder = n % VAD_FRAME_BYTES
        print(
            f"[AudioPipeline] frame#{self._frames_logged+1}: "
            f"bytes={n}, vad_frames={frames_in}, remainder={remainder}. "
            f"Expected: multiples of {VAD_FRAME_BYTES} bytes (16kHz 16-bit mono 20ms)",
            flush=True,
        )

    # ────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ────────────────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Drain remaining buffer and reset state."""
        self._frame_buf = b""
        self._in_speech = False
