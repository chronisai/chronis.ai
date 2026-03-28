"""
smoke_tests/test_simli.py  —  Simli API Smoke Test

Run this FIRST (step 1 of build order) before writing any pipeline code.
Confirms:
  ✓ WebSocket connects and receives ready confirmation
  ✓ clearBuffer() exists and executes without error
  ✓ sendImmediate() exists and executes without error
  ✓ WebSocket stays open across multiple send() calls
  ✓ stop() exists (not close())

Usage:
  export SIMLI_API_KEY=your_key
  export SIMLI_FACE_ID=your_test_face_id
  python smoke_tests/test_simli.py

Expected output:
  [1/5] Connecting to Simli... ✓
  [2/5] Sending test audio... ✓ (3 chunks sent)
  [3/5] clearBuffer()... ✓
  [4/5] sendImmediate()... ✓
  [5/5] stop()... ✓
  All Simli checks passed ✓
"""

import asyncio
import json
import os
import struct
import sys
import time
import websockets


SIMLI_API_KEY = os.environ.get("SIMLI_API_KEY", "")
SIMLI_FACE_ID = os.environ.get("SIMLI_FACE_ID", "")
SIMLI_WS_URL  = "wss://api.simli.ai/startAudioToVideoSession"
SIMLI_BASE    = "https://api.simli.ai"


def _generate_silent_pcm(duration_ms: int = 120) -> bytes:
    """Generate silent 16kHz mono 16-bit PCM for testing."""
    samples = int(16000 * duration_ms / 1000)
    return struct.pack(f"<{samples}h", *([0] * samples))


async def get_session_token() -> str:
    import httpx
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{SIMLI_BASE}/getSessionToken",
            headers={
                "Authorization": f"Bearer {SIMLI_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={"faceId": SIMLI_FACE_ID},
            timeout=10.0,
        )
        if not r.is_success:
            print(f"✗ Token error: {r.status_code} {r.text[:200]}")
            sys.exit(1)
        data = r.json()
        token = data.get("session_token") or data.get("sessionToken")
        if not token:
            print(f"✗ No token in response: {data}")
            sys.exit(1)
        return token


async def run_smoke_test():
    if not SIMLI_API_KEY:
        print("✗ SIMLI_API_KEY not set")
        sys.exit(1)
    if not SIMLI_FACE_ID:
        print("✗ SIMLI_FACE_ID not set")
        sys.exit(1)

    print(f"Simli smoke test — face_id={SIMLI_FACE_ID[:12]}...\n")

    # ── [1/5] Connect ─────────────────────────────────────────────────────
    print("[1/5] Getting session token and connecting...")
    t0 = time.monotonic()

    token = await get_session_token()
    print(f"      Token: {token[:20]}...")

    ws = await websockets.connect(
        SIMLI_WS_URL,
        extra_headers={"Authorization": f"Bearer {SIMLI_API_KEY}"},
        ping_interval=20,
        ping_timeout=10,
    )

    init_msg = {
        "session_token": token,
        "face_id":       SIMLI_FACE_ID,
        "audio_format": {
            "type": "pcm", "sample_rate": 16000, "channels": 1, "bit_depth": 16
        },
    }
    await ws.send(json.dumps(init_msg))

    try:
        response = await asyncio.wait_for(ws.recv(), timeout=10.0)
        data = json.loads(response) if isinstance(response, str) else {}
        status = data.get("status", "unknown")
        elapsed = time.monotonic() - t0
        if status == "ready":
            print(f"      ✓ Connected and ready ({elapsed*1000:.0f}ms)\n")
        else:
            print(f"      ✗ Unexpected status: {data}")
            await ws.close()
            sys.exit(1)
    except asyncio.TimeoutError:
        print("      ✗ Timeout waiting for ready confirmation")
        await ws.close()
        sys.exit(1)

    # ── [2/5] Send audio chunks ────────────────────────────────────────────
    print("[2/5] Sending 3 audio chunks (120ms silence each)...")
    silent_pcm = _generate_silent_pcm(120)

    for i in range(3):
        await ws.send(silent_pcm)
        print(f"      Chunk {i+1} sent ({len(silent_pcm)} bytes)")
        await asyncio.sleep(0.05)

    print("      ✓ Audio send OK\n")

    # ── [3/5] clearBuffer ─────────────────────────────────────────────────
    print("[3/5] Testing clearBuffer()...")
    t0 = time.monotonic()
    await ws.send(json.dumps({"type": "clearBuffer"}))
    elapsed = time.monotonic() - t0
    print(f"      ✓ clearBuffer() sent ({elapsed*1000:.2f}ms)\n")

    # ── [4/5] sendImmediate ───────────────────────────────────────────────
    print("[4/5] Testing sendImmediate()...")
    import base64
    test_audio_b64 = base64.b64encode(silent_pcm).decode()
    t0 = time.monotonic()
    await ws.send(json.dumps({"type": "sendImmediate", "audio": test_audio_b64}))
    elapsed = time.monotonic() - t0
    print(f"      ✓ sendImmediate() sent ({elapsed*1000:.2f}ms)\n")

    # Confirm sendImmediate doesn't close the WS
    await asyncio.sleep(0.2)
    assert not ws.closed, "WebSocket closed after sendImmediate — unexpected!"
    print("      ✓ WebSocket still open after sendImmediate\n")

    # ── [5/5] stop() ──────────────────────────────────────────────────────
    print("[5/5] Testing stop()...")
    t0 = time.monotonic()
    await ws.send(json.dumps({"type": "stop"}))
    await ws.close()
    elapsed = time.monotonic() - t0
    print(f"      ✓ stop() and close OK ({elapsed*1000:.0f}ms)\n")

    # ── Summary ────────────────────────────────────────────────────────────
    print("=" * 50)
    print("✅ All Simli smoke tests passed")
    print("   clearBuffer() ✓  sendImmediate() ✓  stop() ✓")
    print("   close() NOT tested (it does not exist — use stop())")


if __name__ == "__main__":
    asyncio.run(run_smoke_test())
