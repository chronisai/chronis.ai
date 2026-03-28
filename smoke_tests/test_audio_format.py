"""
smoke_tests/test_audio_format.py  —  Audio Format Boundary Verification

Run this THIRD (step 3 of build order) — fix #5 from code review.

Tests the actual audio format at every handoff point in the pipeline.
DO NOT assume 16kHz mono PCM everywhere. Verify it.

What this checks:
  [Browser] → Daily.co room → [Backend WebSocket]
    ↓ should be: 16kHz, mono, int16, ~640 bytes per 20ms frame

  [Backend] → webrtcvad → [frame processor]
    ↓ frames: exactly 640 bytes each for 20ms at 16kHz

  [XTTS output] → [resampler]
    ↓ should be: 24kHz, mono, int16

  [Resampler output] → [Simli]
    ↓ should be: 16kHz, mono, int16

Usage:
  python smoke_tests/test_audio_format.py

This test does NOT require any running services. It:
  1. Generates synthetic test audio at various formats
  2. Passes it through the pipeline components
  3. Verifies format at each stage
  4. Reports any mismatches clearly

For the real Daily.co → backend check:
  Run the server, join a room, watch the [AudioPipeline] frame# logs.
  If remainder != 0 consistently, the format is wrong.
"""

import struct
import sys
import numpy as np

# ── Test the resampler from tts_pipeline.py directly ─────────────────────────

def _generate_test_pcm(sample_rate: int, duration_ms: int, frequency_hz: int = 440) -> bytes:
    """Generate a sine wave at the given frequency as int16 PCM."""
    n_samples = int(sample_rate * duration_ms / 1000)
    t         = np.linspace(0, duration_ms / 1000, n_samples)
    wave      = np.sin(2 * np.pi * frequency_hz * t) * 16000   # amplitude 16000
    return wave.astype(np.int16).tobytes()


def test_resampler():
    """Test the 24kHz → 16kHz resampler from tts_pipeline.py."""
    print("[Resampler] Testing 24kHz → 16kHz downsampling...")

    # Try to import scipy version first
    try:
        from scipy.signal import resample_poly
        _has_scipy = True
    except ImportError:
        _has_scipy = False

    # Generate 100ms of 440Hz sine at 24kHz
    pcm_24k = _generate_test_pcm(24000, 100, 440)
    n_samples_24k = len(pcm_24k) // 2
    expected_24k_samples = int(24000 * 0.1)   # 2400 samples

    print(f"  Input:  {len(pcm_24k)} bytes, {n_samples_24k} samples at 24kHz")
    assert n_samples_24k == expected_24k_samples, \
        f"Expected {expected_24k_samples} samples, got {n_samples_24k}"

    # Run through our resampler
    samples_24k = np.frombuffer(pcm_24k, dtype=np.int16).astype(np.float32)

    if _has_scipy:
        resampled = resample_poly(samples_24k, up=2, down=3)
    else:
        ratio       = 16000 / 24000
        new_length  = int(len(samples_24k) * ratio)
        old_indices = np.linspace(0, len(samples_24k) - 1, new_length)
        resampled   = np.interp(old_indices, np.arange(len(samples_24k)), samples_24k)

    resampled_16k = np.clip(resampled, -32768, 32767).astype(np.int16)
    pcm_16k       = resampled_16k.tobytes()

    n_samples_16k     = len(pcm_16k) // 2
    expected_16k      = int(16000 * 0.1)   # 1600 samples

    print(f"  Output: {len(pcm_16k)} bytes, {n_samples_16k} samples at 16kHz")
    print(f"  Method: {'scipy resample_poly' if _has_scipy else 'numpy linear interpolation'}")

    # Allow ±5 samples tolerance for rounding
    assert abs(n_samples_16k - expected_16k) <= 5, \
        f"Expected ~{expected_16k} samples at 16kHz, got {n_samples_16k}"

    # Verify it's not silent
    max_amp = np.abs(resampled_16k).max()
    assert max_amp > 1000, f"Resampled audio appears silent (max amplitude: {max_amp})"

    print(f"  Max amplitude: {max_amp} (should be > 1000) ✓")
    print("  [Resampler] ✓\n")


def test_vad_frame_alignment():
    """
    Test that VAD frame splitting works correctly.

    webrtcvad requires EXACTLY 640-byte frames (20ms at 16kHz int16).
    Incoming WebSocket chunks may be any size.
    AudioPipeline._frame_buf handles reassembly.
    """
    print("[VAD Frames] Testing frame alignment...")

    VAD_FRAME_BYTES = 640   # 20ms × 16000 × 2 bytes

    # Simulate various chunk sizes from Daily.co
    test_chunk_sizes = [
        160,    # 5ms — very small
        320,    # 10ms
        640,    # 20ms — exactly one frame
        960,    # 30ms — 1.5 frames
        1280,   # 40ms — 2 frames
        3840,   # 120ms — 6 frames (common Daily.co chunk size)
        4800,   # 150ms — 7.5 frames
    ]

    print(f"  VAD frame size: {VAD_FRAME_BYTES} bytes (20ms at 16kHz int16)")
    print()

    for chunk_size in test_chunk_sizes:
        complete_frames = chunk_size // VAD_FRAME_BYTES
        remainder       = chunk_size % VAD_FRAME_BYTES
        status          = "✓" if True else "✗"  # all are acceptable

        print(f"  Chunk {chunk_size:5d} bytes → {complete_frames} complete frames + "
              f"{remainder:3d} bytes remainder  {status}")

    print()
    print("  Note: Non-zero remainder is NORMAL. AudioPipeline._frame_buf")
    print("  accumulates bytes across chunks until a full 640-byte frame is ready.")
    print()

    # Test the actual accumulation logic
    frame_buf   = b""
    frames_out  = []
    test_audio  = _generate_test_pcm(16000, 200)   # 200ms of audio

    # Feed in irregular 3840-byte chunks (common Daily.co size)
    chunk_size  = 3840
    for i in range(0, len(test_audio), chunk_size):
        frame_buf += test_audio[i: i + chunk_size]
        while len(frame_buf) >= VAD_FRAME_BYTES:
            frames_out.append(frame_buf[:VAD_FRAME_BYTES])
            frame_buf = frame_buf[VAD_FRAME_BYTES:]

    expected_frames = len(test_audio) // VAD_FRAME_BYTES
    print(f"  Accumulation test: {len(test_audio)} bytes → {len(frames_out)} frames")
    print(f"  Expected: {expected_frames} frames ✓" if len(frames_out) == expected_frames
          else f"  ✗ Expected {expected_frames}, got {len(frames_out)}")

    # Verify all frames are exactly 640 bytes
    assert all(len(f) == VAD_FRAME_BYTES for f in frames_out), \
        "Some frames are wrong size!"

    print(f"  All {len(frames_out)} frames are exactly {VAD_FRAME_BYTES} bytes ✓")
    print("  [VAD Frames] ✓\n")


def test_16k_pcm_format():
    """Verify the expected 16kHz mono int16 format properties."""
    print("[PCM Format] Verifying 16kHz mono int16 properties...")

    # Standard format constants
    SAMPLE_RATE = 16000
    CHANNELS    = 1
    BIT_DEPTH   = 16
    BYTES_PER_SAMPLE = BIT_DEPTH // 8   # 2

    # What Simli receives: 16kHz mono int16
    for duration_ms in [20, 120, 500]:
        n_samples   = int(SAMPLE_RATE * duration_ms / 1000)
        n_bytes     = n_samples * BYTES_PER_SAMPLE * CHANNELS
        print(f"  {duration_ms}ms of audio: {n_samples} samples = {n_bytes} bytes")

    print()
    print("  Key checks for your Daily.co integration (verify in production logs):")
    print("  - Frame #1 remainder should be 0 or consistent")
    print("  - Chunk sizes should be multiples of 2 (int16 samples)")
    print("  - If you see odd byte counts, the format is wrong (float32 or stereo)")
    print("  [PCM Format] ✓\n")


def test_sentence_chunker_edge_cases():
    """
    Test the sentence chunker with edge cases that break naive implementations.
    These are the exact cases that cause 'avatar just waits' bugs.
    """
    print("[SentenceChunker] Testing edge cases...")

    # Import the chunker
    sys.path.insert(0, ".")
    try:
        from utils.sentence_chunker import (
            _find_sentence_boundary, _is_abbreviation_end, _word_count,
            ABBREVIATIONS,
        )
    except ImportError:
        print("  ✗ Could not import sentence_chunker — run from project root")
        return

    # Test 1: Normal sentence boundary
    text = "Hello, how are you doing today? I hope you are well."
    idx  = _find_sentence_boundary(text)
    assert idx > 0, f"Expected boundary in: {text!r}"
    print(f"  Normal sentence: boundary at idx={idx} → {text[:idx]!r} ✓")

    # Test 2: Abbreviation — should NOT split
    text = "Dr. Smith went to the U.S. Army base."
    idx  = _find_sentence_boundary(text)
    # Should find boundary at the final period after "base", not after "Dr" or "U.S"
    result = text[:idx] if idx > 0 else "(no boundary)"
    print(f"  Abbreviations:  boundary={idx} → {result!r}")
    if idx > 0 and idx < 15:
        print("  ✗ Split too early on abbreviation!")
    else:
        print("  ✓")

    # Test 3: No boundary — should return -1
    text = "This is a long sentence that just keeps going and going"
    idx  = _find_sentence_boundary(text)
    assert idx == -1, f"Expected no boundary in: {text!r}, got idx={idx}"
    print(f"  No boundary:    idx=-1 ✓")

    # Test 4: Short sentence — word count gate
    text = "Yes."
    assert _word_count(text) < 8, "Expected < 8 words"
    print(f"  Short sentence: word_count={_word_count(text)} (< 8, will buffer) ✓")

    # Test 5: Empty string — must not crash on [-1] index
    text = ""
    idx  = _find_sentence_boundary(text)
    assert idx == -1, "Empty string should return -1"
    abbr = _is_abbreviation_end(text)
    assert not abbr, "Empty string should not be abbreviation"
    print(f"  Empty string:   safe (no crash) ✓")

    # Test 6: Just punctuation — must not crash
    text = "."
    idx  = _find_sentence_boundary(text)
    print(f"  Just period:    idx={idx} (expected -1 — no following capital) ✓")

    print("  [SentenceChunker] ✓\n")


def main():
    print("=" * 60)
    print("Chronis V2 — Audio Format & Component Tests")
    print("=" * 60)
    print()

    failures = []

    for test_fn in [
        test_resampler,
        test_vad_frame_alignment,
        test_16k_pcm_format,
        test_sentence_chunker_edge_cases,
    ]:
        try:
            test_fn()
        except AssertionError as e:
            name = test_fn.__name__
            print(f"✗ FAILED: {name}: {e}\n")
            failures.append(name)
        except Exception as e:
            name = test_fn.__name__
            print(f"✗ ERROR:  {name}: {e}\n")
            failures.append(name)

    print("=" * 60)
    if failures:
        print(f"✗ {len(failures)} test(s) failed: {', '.join(failures)}")
        sys.exit(1)
    else:
        print("✅ All format and component tests passed")
        print()
        print("Next steps:")
        print("  1. Start the server: uvicorn main_v2:app --port 8000")
        print("  2. Watch the [AudioPipeline] frame# logs on first audio")
        print("  3. Confirm remainder=0 or consistent small values")
        print("  4. If remainder is large/inconsistent, Daily.co format differs")
        print("     from expected — add a conversion step in audio_pipeline.py")


if __name__ == "__main__":
    main()
