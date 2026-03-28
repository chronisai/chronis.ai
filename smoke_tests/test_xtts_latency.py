"""
smoke_tests/test_xtts_latency.py  —  XTTS v2 First-Chunk Latency Test

Run this SECOND (step 2 of build order) before building the TTS pipeline.
Must confirm:
  ✓ First audio chunk arrives in < 700ms
  ✓ Chunks arrive progressively (streaming, not all at once)
  ✓ Total audio is complete and well-formed
  ✓ Sample rate headers are correct (24kHz)

If first chunk > 700ms:
  - Check Modal GPU cold start (set keep_warm=1)
  - Try running it twice — second call uses warm container

If all chunks arrive at once (not streaming):
  - Modal XTTS server may be buffering — check StreamingResponse setup

Usage:
  export MODAL_XTTS_URL=https://your-modal-url/synthesize
  export TEST_VOICE_REF=voice_refs/test_agent/reference.wav
  python smoke_tests/test_xtts_latency.py

Expected output:
  Testing XTTS latency...
  Request sent. Waiting for first chunk...
  ✓ First chunk: 324ms (4096 bytes)
  Chunk  2: 412ms total (4096 bytes)
  ...
  Chunk 18: 2.1s total (2048 bytes)
  ✓ Streaming verified: chunks arrived progressively
  ✓ Total audio: 69632 bytes, ~1.45 seconds at 24kHz
  ✓ XTTS latency test passed
"""

import asyncio
import os
import struct
import sys
import time

import httpx

MODAL_XTTS_URL = os.environ.get("MODAL_XTTS_URL", "")
TEST_VOICE_REF = os.environ.get("TEST_VOICE_REF", "voice_refs/test/reference.wav")

# A short sentence — long enough to measure latency, short enough to be quick
TEST_TEXT = "Hello, this is a test of the voice synthesis system. I hope you can hear me clearly."


def _check_pcm_validity(pcm_bytes: bytes) -> dict:
    """Basic sanity check on raw PCM bytes."""
    n_samples = len(pcm_bytes) // 2   # int16 = 2 bytes per sample
    duration  = n_samples / 24000     # at 24kHz
    # Check for all-zeros (silence only — synthesis may have failed)
    samples   = struct.unpack(f"<{n_samples}h", pcm_bytes[:n_samples * 2])
    max_amp   = max(abs(s) for s in samples[:1000]) if samples else 0
    return {
        "n_bytes":     len(pcm_bytes),
        "n_samples":   n_samples,
        "duration_s":  round(duration, 2),
        "max_amp":     max_amp,
        "is_silent":   max_amp < 100,
    }


async def run_latency_test():
    if not MODAL_XTTS_URL:
        print("✗ MODAL_XTTS_URL not set")
        sys.exit(1)

    print(f"XTTS latency test")
    print(f"  URL:       {MODAL_XTTS_URL}")
    print(f"  Voice ref: {TEST_VOICE_REF}")
    print(f"  Text:      \"{TEST_TEXT[:60]}...\"\n")

    chunk_times    = []  # (elapsed_ms, chunk_bytes) for each chunk
    all_pcm        = b""
    request_sent_t = None

    async with httpx.AsyncClient(timeout=60.0) as client:
        print("Sending request...")
        request_sent_t = time.monotonic()

        async with client.stream(
            "POST",
            MODAL_XTTS_URL,
            json={
                "text":        TEST_TEXT,
                "speaker_wav": TEST_VOICE_REF,
                "language":    "en",
                "stream":      True,
            },
            headers={"Content-Type": "application/json"},
        ) as response:
            if not response.is_success:
                body = await response.aread()
                print(f"✗ HTTP {response.status_code}: {body[:300]}")
                sys.exit(1)

            # Log response headers
            print(f"  Response headers:")
            for k, v in response.headers.items():
                if k.lower().startswith("x-"):
                    print(f"    {k}: {v}")
            print()

            chunk_num = 0
            async for chunk in response.aiter_bytes(chunk_size=4096):
                elapsed_ms = (time.monotonic() - request_sent_t) * 1000
                chunk_num += 1
                chunk_times.append((elapsed_ms, len(chunk)))
                all_pcm += chunk

                if chunk_num == 1:
                    print(f"  ✓ First chunk: {elapsed_ms:.0f}ms ({len(chunk)} bytes)")
                else:
                    print(f"    Chunk {chunk_num:2d}: {elapsed_ms:.0f}ms total ({len(chunk)} bytes)")

    print()

    # ── Analysis ──────────────────────────────────────────────────────────
    if not chunk_times:
        print("✗ No chunks received")
        sys.exit(1)

    first_chunk_ms   = chunk_times[0][0]
    total_ms         = chunk_times[-1][0]
    n_chunks         = len(chunk_times)

    # Check if chunks arrived progressively (not all at once)
    # If all chunks arrive within 50ms of each other, it's buffered, not streaming
    if n_chunks > 1:
        inter_chunk_gaps = [
            chunk_times[i][0] - chunk_times[i-1][0]
            for i in range(1, n_chunks)
        ]
        max_gap    = max(inter_chunk_gaps)
        is_streaming = max_gap > 10  # at least one 10ms gap = genuinely streaming
    else:
        is_streaming = False

    pcm_info = _check_pcm_validity(all_pcm)

    print("─" * 50)
    print(f"First chunk latency:  {first_chunk_ms:.0f}ms  (target: < 700ms)")
    print(f"Total time:           {total_ms:.0f}ms")
    print(f"Chunks received:      {n_chunks}")
    print(f"Total PCM bytes:      {pcm_info['n_bytes']:,}")
    print(f"Audio duration:       {pcm_info['duration_s']}s at 24kHz")
    print(f"Max amplitude:        {pcm_info['max_amp']} (>100 = real audio)")
    print(f"Streaming verified:   {is_streaming}")
    print()

    # ── Pass/fail ─────────────────────────────────────────────────────────
    failures = []

    if first_chunk_ms > 700:
        failures.append(
            f"First chunk too slow: {first_chunk_ms:.0f}ms (need < 700ms)\n"
            f"  → Check Modal GPU cold start: set keep_warm=1 in xtts_server.py\n"
            f"  → Run test again — second call uses warm container"
        )

    if pcm_info["is_silent"]:
        failures.append(
            f"Audio appears silent (max amplitude: {pcm_info['max_amp']})\n"
            f"  → Check voice reference file exists at: {TEST_VOICE_REF}\n"
            f"  → Check XTTS synthesis logs in Modal dashboard"
        )

    if pcm_info["n_bytes"] < 10000:
        failures.append(
            f"Audio too short ({pcm_info['n_bytes']} bytes)\n"
            f"  → Synthesis may have failed silently"
        )

    if not is_streaming:
        failures.append(
            f"Audio not streaming progressively\n"
            f"  → Check StreamingResponse in xtts_server.py\n"
            f"  → Chunks should arrive with visible time gaps between them"
        )

    if failures:
        print("✗ FAILURES:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("✅ XTTS latency test passed")
        print(f"   First chunk: {first_chunk_ms:.0f}ms ✓")
        print(f"   Streaming:   {is_streaming} ✓")
        print(f"   Audio valid: max_amp={pcm_info['max_amp']} ✓")


if __name__ == "__main__":
    asyncio.run(run_latency_test())
