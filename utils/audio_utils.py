"""
utils/audio_utils.py  —  Audio Processing Utilities

All audio format operations that run in the live pipeline.

Key rule: NO subprocess spawning in the live session path.
FFmpeg is only called ONCE — during onboarding in audio_validator.py.
Everything here is in-process using numpy/scipy.

Contents:
  resample_24k_to_16k()  — core resampler used by TTSPipeline
  pcm_to_float32()       — PCM int16 → float32 for analysis
  float32_to_pcm()       — float32 → PCM int16 for sending
  validate_pcm_frame()   — sanity-check a frame before sending to webrtcvad
  chunk_to_frames()      — split arbitrary bytes into VAD-aligned frames
  mix_to_mono()          — stereo → mono if Daily sends unexpected stereo
"""

import numpy as np
from typing import Iterator, List, Tuple

# Audio format constants — these are the single source of truth
# for the entire pipeline. Import from here, never hardcode.
XTTS_SAMPLE_RATE   = 24_000   # XTTS v2 native output
TARGET_SAMPLE_RATE = 16_000   # Simli + Deepgram + webrtcvad expected input
VAD_FRAME_MS       = 20       # webrtcvad frame duration
VAD_FRAME_SAMPLES  = TARGET_SAMPLE_RATE * VAD_FRAME_MS // 1000   # 320 samples
VAD_FRAME_BYTES    = VAD_FRAME_SAMPLES * 2                        # 640 bytes (int16)

try:
    from scipy.signal import resample_poly as _resample_poly
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ────────────────────────────────────────────────────────────────────────────
# Resampler — the only critical path function here
# ────────────────────────────────────────────────────────────────────────────

def resample_24k_to_16k(pcm_bytes: bytes) -> bytes:
    """
    Downsample 24kHz mono int16 PCM to 16kHz mono int16 PCM.

    Called by TTSPipeline on every audio chunk from XTTS v2.
    Must be fast: runs on every 4096-byte chunk (~85ms of audio).

    scipy.signal.resample_poly is preferred:
      - Anti-aliasing filter built in
      - Accurate 2/3 ratio (24000 * 2/3 = 16000)
      - ~0.5ms per chunk on modern hardware

    Falls back to numpy linear interpolation if scipy unavailable:
      - No anti-aliasing (acceptable for speech)
      - ~0.3ms per chunk
      - Installed by default everywhere
    """
    if len(pcm_bytes) < 2:
        return b""

    # Ensure even number of bytes (int16 = 2 bytes)
    if len(pcm_bytes) % 2 != 0:
        pcm_bytes = pcm_bytes[:-1]

    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)

    if _HAS_SCIPY:
        # up=2, down=3 → 24000 * (2/3) = 16000 exactly
        resampled = _resample_poly(samples, up=2, down=3)
    else:
        ratio       = TARGET_SAMPLE_RATE / XTTS_SAMPLE_RATE   # 2/3
        new_len     = int(len(samples) * ratio)
        old_indices = np.linspace(0, len(samples) - 1, new_len)
        resampled   = np.interp(old_indices, np.arange(len(samples)), samples)

    return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()


# ────────────────────────────────────────────────────────────────────────────
# Format conversion helpers
# ────────────────────────────────────────────────────────────────────────────

def pcm_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """
    Convert int16 PCM bytes to float32 array in range [-1.0, 1.0].
    Used by the audio format smoke test and any future DSP operations.
    """
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    return samples / 32768.0


def float32_to_pcm(float32_array: np.ndarray) -> bytes:
    """
    Convert float32 [-1.0, 1.0] audio to int16 PCM bytes.
    Used by the frontend JS conversion and any future gain adjustment.
    """
    clipped = np.clip(float32_array, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


def mix_to_mono(pcm_bytes: bytes, channels: int = 2) -> bytes:
    """
    Mix multi-channel int16 PCM to mono by averaging channels.

    Called defensively in AudioPipeline if Daily.co ever sends stereo
    (it should always send mono with our room config, but belt-and-suspenders).
    """
    if channels == 1:
        return pcm_bytes
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    # Reshape into (frames, channels), average across channels
    if len(samples) % channels != 0:
        samples = samples[:-(len(samples) % channels)]
    stereo = samples.reshape(-1, channels).astype(np.float32)
    mono   = stereo.mean(axis=1)
    return np.clip(mono, -32768, 32767).astype(np.int16).tobytes()


# ────────────────────────────────────────────────────────────────────────────
# VAD frame utilities
# ────────────────────────────────────────────────────────────────────────────

def chunk_to_vad_frames(pcm_bytes: bytes,
                        remainder: bytes = b"") -> Tuple[List[bytes], bytes]:
    """
    Split arbitrary-length PCM bytes into complete 640-byte VAD frames.

    Returns:
        (list_of_complete_frames, leftover_bytes)

    The leftover bytes should be prepended to the next chunk.
    Used by AudioPipeline._frame_buf logic.

    Example:
        frames, leftover = chunk_to_vad_frames(chunk, leftover)
        for frame in frames:
            is_speech = vad.is_speech(frame, 16000)
    """
    combined = remainder + pcm_bytes
    frames   = []

    while len(combined) >= VAD_FRAME_BYTES:
        frames.append(combined[:VAD_FRAME_BYTES])
        combined = combined[VAD_FRAME_BYTES:]

    return frames, combined  # combined is now the leftover


def validate_pcm_frame(frame: bytes) -> bool:
    """
    Quick sanity check before passing a frame to webrtcvad.

    webrtcvad raises on:
      - Wrong frame size (not exactly 640 bytes for 20ms at 16kHz)
      - All-zero frames sometimes cause issues on some versions

    Returns True if frame is safe to pass to vad.is_speech().
    """
    if len(frame) != VAD_FRAME_BYTES:
        return False
    # Check not all zeros (degenerate frame)
    return any(b != 0 for b in frame[:10])


def estimate_rms_db(pcm_bytes: bytes) -> float:
    """
    Estimate RMS level in dBFS for a PCM frame.
    Useful for logging audio levels during smoke testing.

    Returns dBFS value (0 = full scale, -60 = very quiet, -inf = silence).
    """
    if not pcm_bytes:
        return -float("inf")
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    rms     = np.sqrt(np.mean(samples ** 2))
    if rms < 1:
        return -60.0
    return float(20 * np.log10(rms / 32768.0))
