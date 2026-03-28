"""
services/deepgram_client.py  —  DeepgramClient

Streaming WebSocket STT using Deepgram Nova-2.

Key design decisions:
  - WebSocket stays open for the ENTIRE session (never per-turn)
  - Partial transcripts are used for real-time display to the user
  - Final transcripts trigger the LLM pipeline
  - End-of-utterance uses hybrid detection:
      silence_ms > 500  OR  transcript ends with [.?!]
  - endpointing param tuned manually (default is too aggressive)
"""

import asyncio
import json
import os
from typing import Callable, Optional

import websockets
from websockets.exceptions import ConnectionClosed


DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")

# Deepgram WebSocket endpoint
# model=nova-2 — fastest available, best accuracy
# endpointing=500 — wait 500ms of silence before finalizing (tuned from default)
# interim_results=true — stream partial results
# utterance_end_ms=1000 — additional silence threshold for utterance end
DEEPGRAM_WS_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?model=nova-2"
    "&encoding=linear16"
    "&sample_rate=16000"
    "&channels=1"
    "&endpointing=500"
    "&interim_results=true"
    "&utterance_end_ms=1000"
    "&vad_events=true"
    "&smart_format=true"
)


class DeepgramClient:
    """
    Streaming STT. Opens once, receives audio continuously.

    Usage:
        client = DeepgramClient(
            on_partial=lambda text: ...,      # called on partial transcripts
            on_final=lambda text: ...,        # called on final utterance
        )
        await client.start()
        await client.send_audio(pcm_bytes)   # call from audio pipeline
        await client.stop()
    """

    def __init__(
        self,
        on_partial: Callable[[str], None] = None,
        on_final: Callable[[str], None] = None,
    ):
        self.on_partial = on_partial
        self.on_final   = on_final

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected        = False
        self._recv_task: Optional[asyncio.Task] = None
        self._current_partial  = ""
        self._send_lock        = asyncio.Lock()

    # ────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ────────────────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Open the Deepgram WebSocket and start receiving transcription results."""
        print("[Deepgram] Connecting...", flush=True)
        self._ws = await websockets.connect(
            DEEPGRAM_WS_URL,
            extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
            ping_interval=10,
            ping_timeout=5,
            close_timeout=5,
        )
        self._connected = True
        self._recv_task = asyncio.create_task(self._receive_loop())
        print("[Deepgram] Connected ✓", flush=True)

    async def stop(self) -> None:
        """Send CloseStream message and close the WebSocket."""
        self._connected = False

        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass

        if self._ws:
            try:
                # Deepgram's graceful close: send CloseStream message
                await self._ws.send(json.dumps({"type": "CloseStream"}))
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        print("[Deepgram] Disconnected ✓", flush=True)

    # ────────────────────────────────────────────────────────────────────────
    # Audio input
    # ────────────────────────────────────────────────────────────────────────

    async def send_audio(self, audio_bytes: bytes) -> None:
        """
        Send raw 16kHz mono PCM bytes to Deepgram.
        Called by audio_pipeline on every frame that passes VAD.
        """
        if not self._connected or not self._ws:
            return
        async with self._send_lock:
            try:
                await self._ws.send(audio_bytes)
            except ConnectionClosed:
                self._connected = False

    async def send_keep_alive(self) -> None:
        """
        Send a KeepAlive message during silence.
        Uses _send_lock so it doesn't race with concurrent send_audio() calls.
        """
        if not self._connected or not self._ws:
            return
        async with self._send_lock:
            try:
                await self._ws.send(json.dumps({"type": "KeepAlive"}))
            except Exception:
                pass

    # ────────────────────────────────────────────────────────────────────────
    # Receive loop
    # ────────────────────────────────────────────────────────────────────────

    async def _receive_loop(self) -> None:
        """
        Process all messages from Deepgram.

        Message types we care about:
          Results:
            - is_final=false → partial transcript (display only)
            - is_final=true + speech_final=true → complete utterance
          UtteranceEnd: Deepgram's built-in silence detection
          SpeechStarted: Deepgram detected voice onset

        End-of-utterance hybrid logic:
          Emit on_final when:
            - speech_final=true AND transcript is non-empty
          This already incorporates Deepgram's 500ms endpointing.
          We additionally check for terminal punctuation in the app-level
          hybrid detector (audio_pipeline).
        """
        try:
            async for message in self._ws:
                if isinstance(message, bytes):
                    continue   # audio echo — Deepgram doesn't send this but guard anyway

                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type", "")

                if msg_type == "Results":
                    await self._handle_results(data)

                elif msg_type == "UtteranceEnd":
                    # Deepgram detected end of utterance via silence
                    # If we have a partial accumulated, finalize it
                    if self._current_partial.strip() and self.on_final:
                        text = self._current_partial.strip()
                        self._current_partial = ""
                        if asyncio.iscoroutinefunction(self.on_final):
                            await self.on_final(text)
                        else:
                            self.on_final(text)

                elif msg_type == "SpeechStarted":
                    pass   # audio_pipeline handles VAD for this

                elif msg_type == "Error":
                    print(f"[Deepgram] Error: {data}", flush=True)

        except ConnectionClosed:
            self._connected = False
            print("[Deepgram] Connection closed", flush=True)
        except asyncio.CancelledError:
            pass

    async def _handle_results(self, data: dict) -> None:
        """Process a Deepgram Results message."""
        alternatives = (
            data.get("channel", {})
                .get("alternatives", [])
        )
        if not alternatives:
            return

        transcript = alternatives[0].get("transcript", "").strip()
        is_final   = data.get("is_final", False)
        speech_final = data.get("speech_final", False)

        if not transcript:
            return

        if not is_final:
            # Partial result — update display
            self._current_partial = transcript
            if self.on_partial:
                if asyncio.iscoroutinefunction(self.on_partial):
                    await self.on_partial(transcript)
                else:
                    self.on_partial(transcript)
            return

        # Final result
        self._current_partial = ""

        if speech_final and self.on_final:
            # Complete utterance — fire the pipeline
            if asyncio.iscoroutinefunction(self.on_final):
                await self.on_final(transcript)
            else:
                self.on_final(transcript)
