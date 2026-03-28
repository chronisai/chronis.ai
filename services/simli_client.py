"""
services/simli_client.py  —  SimliClient

Wrapper around Simli's WebSocket API for avatar rendering.

Key facts confirmed from live API test (per blueprint):
  - clearBuffer()     EXISTS  — dumps all queued audio, avatar stops immediately
  - sendImmediate()   EXISTS  — pushes audio to front of queue, bypasses buffer
  - close()           DOES NOT EXIST — use stop() for teardown
  - WebSocket stays open for the ENTIRE session (never per-turn)

Opening per-turn adds 300ms+ connection overhead and causes random avatar resets.
The WebSocket is opened once in session_start and torn down in cleanup.
"""

import asyncio
import base64
import json
import os
from typing import Optional

import websockets
from websockets.exceptions import ConnectionClosed


# ── Simli API endpoints ──────────────────────────────────────────────────────
SIMLI_API_KEY   = os.environ.get("SIMLI_API_KEY", "")
SIMLI_BASE_URL  = "https://api.simli.ai"
SIMLI_WS_URL    = "wss://api.simli.ai/startAudioToVideoSession"


class SimliClient:
    """
    Manages one persistent WebSocket connection to Simli for a single session.

    Usage:
        client = SimliClient(face_id="abc123", session_token="tok_xxx")
        await client.start()            # open WS, handshake
        await client.send(audio_bytes)  # send 16kHz PCM chunks
        await client.clear_buffer()     # on interrupt
        await client.send_immediate(first_chunk)  # after interrupt
        await client.stop()             # graceful teardown
    """

    def __init__(self, face_id: str, daily_room_url: str):
        self.face_id        = face_id
        self.daily_room_url = daily_room_url

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected      = False
        self._send_lock      = asyncio.Lock()   # serialize sends
        self._recv_task: Optional[asyncio.Task] = None

    # ────────────────────────────────────────────────────────────────────────
    # Connection lifecycle
    # ────────────────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Open WebSocket to Simli and send the initial session config.
        Simli will start rendering the avatar and stream video into the Daily room.
        Raises on connection failure — caller should propagate to session startup.
        """
        print(f"[Simli] Connecting... face_id={self.face_id[:8]}", flush=True)

        # ── Step 1: Get a short-lived session token from Simli REST API ──────
        session_token = await self._get_session_token()

        # ── Step 2: Open the WebSocket ────────────────────────────────────────
        extra_headers = {"Authorization": f"Bearer {SIMLI_API_KEY}"}

        self._ws = await websockets.connect(
            SIMLI_WS_URL,
            extra_headers=extra_headers,
            ping_interval=20,       # keep-alive
            ping_timeout=10,
            close_timeout=5,
            max_size=10 * 1024 * 1024,  # 10MB max message
        )

        # ── Step 3: Send the session init config ─────────────────────────────
        # Simli needs to know which face, the audio format, and where to send video
        init_msg = {
            "session_token":  session_token,
            "face_id":        self.face_id,
            "audio_format":   {
                "type":         "pcm",
                "sample_rate":  16000,
                "channels":     1,
                "bit_depth":    16,
            },
            "output": {
                "daily_room_url": self.daily_room_url,
            },
        }
        await self._ws.send(json.dumps(init_msg))

        # ── Step 4: Wait for ready confirmation ───────────────────────────────
        response = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
        data = json.loads(response) if isinstance(response, str) else {}
        if data.get("status") != "ready":
            raise RuntimeError(f"Simli session init failed: {data}")

        self._connected = True

        # ── Step 5: Start background receive loop for Simli status messages ──
        self._recv_task = asyncio.create_task(self._receive_loop())

        print(f"[Simli] Connected and ready ✓", flush=True)

    async def stop(self) -> None:
        """
        Graceful teardown. Sends a close signal, then closes the WebSocket.
        NOTE: The Simli API uses stop(), NOT close(). close() does not exist.
        """
        if not self._connected:
            return

        self._connected = False

        # Cancel the receive loop
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass

        # Send graceful stop signal
        if self._ws:
            try:
                await self._ws.send(json.dumps({"type": "stop"}))
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        print(f"[Simli] Disconnected ✓", flush=True)

    # ────────────────────────────────────────────────────────────────────────
    # Audio sending
    # ────────────────────────────────────────────────────────────────────────

    async def send(self, audio_bytes: bytes) -> None:
        """
        Send a 16kHz mono PCM chunk to Simli's render queue.
        Simli buffers these and renders lip-sync in order.
        """
        if not self._connected or not self._ws:
            return
        async with self._send_lock:
            try:
                # Simli expects raw binary audio
                await self._ws.send(audio_bytes)
            except ConnectionClosed:
                self._connected = False
                print("[Simli] Connection lost during send", flush=True)

    async def clear_buffer(self) -> None:
        """
        Dump all queued audio. Avatar stops immediately.
        Called on interrupt — confirmed working from live test.
        """
        if not self._connected or not self._ws:
            return
        async with self._send_lock:
            try:
                await self._ws.send(json.dumps({"type": "clearBuffer"}))
                print("[Simli] clearBuffer() sent ✓", flush=True)
            except ConnectionClosed:
                self._connected = False

    async def send_immediate(self, audio_bytes: bytes) -> None:
        """
        Push audio to the FRONT of the render queue, bypassing the buffer.
        Called once after an interrupt to send the first chunk of the new response.
        After this call, subsequent chunks use normal send() through the 120ms buffer.
        Confirmed working from live test.
        """
        if not self._connected or not self._ws:
            return
        async with self._send_lock:
            try:
                # Send the control message first
                await self._ws.send(json.dumps({
                    "type":  "sendImmediate",
                    "audio": base64.b64encode(audio_bytes).decode(),
                }))
                print("[Simli] sendImmediate() sent ✓", flush=True)
            except ConnectionClosed:
                self._connected = False

    # ────────────────────────────────────────────────────────────────────────
    # REST helper — get session token
    # ────────────────────────────────────────────────────────────────────────

    async def _get_session_token(self) -> str:
        """
        Call Simli's REST API to get a short-lived session token.
        Token is used in the WebSocket handshake.
        """
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"{SIMLI_BASE_URL}/getSessionToken",
                headers={
                    "Authorization": f"Bearer {SIMLI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"faceId": self.face_id},
            )
            if not r.is_success:
                raise RuntimeError(f"Simli token error: {r.status_code} {r.text[:200]}")
            data = r.json()
            token = data.get("session_token") or data.get("sessionToken")
            if not token:
                raise RuntimeError(f"No session token in Simli response: {data}")
            return token

    # ────────────────────────────────────────────────────────────────────────
    # Background receive loop
    # ────────────────────────────────────────────────────────────────────────

    async def _receive_loop(self) -> None:
        """
        Consume any messages Simli sends back (status updates, errors).
        Most of these are informational — we log them but don't act on them
        for MVP. The avatar drains audio and renders directly into Daily.
        """
        try:
            async for message in self._ws:
                if isinstance(message, str):
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type", "")
                        if msg_type == "error":
                            print(f"[Simli] Error from server: {data}", flush=True)
                        elif msg_type == "buffer_drained":
                            # Avatar has finished speaking
                            print("[Simli] Buffer drained", flush=True)
                    except json.JSONDecodeError:
                        pass
        except ConnectionClosed:
            self._connected = False
        except asyncio.CancelledError:
            pass
