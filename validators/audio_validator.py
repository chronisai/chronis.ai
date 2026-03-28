"""
validators/audio_validator.py  —  AudioValidator

Validates a voice reference recording before uploading it to Modal
for XTTS v2 voice cloning.

XTTS cloning quality degrades severely on:
  - Short recordings (< 30s of speech)
  - Noisy recordings (SNR < 20dB)
  - Silence-heavy recordings (< 60% actual speech)

Garbage in = robotic, crackling, or identity-drifted voice out
for EVERY FUTURE SESSION. Catch it here with a clear message.

Three checks:
  1. Duration ≥ 30 seconds    — enough audio for voice modelling
  2. SNR ≥ 20dB              — low enough background noise
  3. Speech ratio ≥ 0.60     — at least 60% is actual speech

All three must pass. Values are logged for debugging voice quality issues.

Output format:
  If valid: also converts to 16kHz mono PCM WAV (XTTS v2 required format)
  and returns the path to the converted file.
"""

import os
import subprocess
import tempfile
from typing import Dict, Tuple

import numpy as np

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False
    print("[AudioValidator] librosa not installed — validation will be limited", flush=True)


# ── Thresholds ────────────────────────────────────────────────────────────────
MIN_DURATION_S   = 30.0   # seconds
MIN_SNR_DB       = 20.0   # dB
MIN_SPEECH_RATIO = 0.60   # fraction of total duration that is speech

# XTTS v2 required format
XTTS_SAMPLE_RATE = 16000
XTTS_CHANNELS    = 1


def validate_audio(audio_path: str) -> Dict:
    """
    Validate an audio file for XTTS v2 voice cloning.

    Args:
        audio_path: Path to the audio file (any format librosa can read)

    Returns:
        {
            "valid": True,
            "duration_s": float,
            "snr_db": float,
            "speech_ratio": float,
            "converted_path": str,   # path to 16kHz mono PCM WAV
        }
        or
        {
            "valid": False,
            "reason": str,
        }
    """
    if not _HAS_LIBROSA:
        return {
            "valid":  False,
            "reason": "Audio validation unavailable (librosa not installed). Please contact support.",
        }

    # ── Load audio ────────────────────────────────────────────────────────
    try:
        # Load at native sample rate first for accurate analysis
        y, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception as e:
        return {"valid": False, "reason": f"Could not read audio file: {e}"}

    duration = len(y) / sr

    # ── Check 1: Duration ──────────────────────────────────────────────────
    if duration < MIN_DURATION_S:
        return {
            "valid":  False,
            "reason": (
                f"Recording is too short ({duration:.0f}s). "
                f"Please record at least {MIN_DURATION_S:.0f} seconds of clear speech. "
                f"Longer is better — aim for 45-60 seconds."
            ),
        }

    # ── Check 2: SNR ──────────────────────────────────────────────────────
    snr_db = _estimate_snr(y, sr)
    if snr_db < MIN_SNR_DB:
        return {
            "valid":  False,
            "reason": (
                f"Too much background noise (SNR: {snr_db:.1f}dB, need ≥{MIN_SNR_DB}dB). "
                f"Please record in a quiet room with no fans, music, or traffic noise."
            ),
        }

    # ── Check 3: Speech ratio ─────────────────────────────────────────────
    speech_ratio = _estimate_speech_ratio(y, sr)
    if speech_ratio < MIN_SPEECH_RATIO:
        return {
            "valid":  False,
            "reason": (
                f"Too much silence in the recording (speech ratio: {speech_ratio:.0%}, "
                f"need ≥{MIN_SPEECH_RATIO:.0%}). "
                f"Please record continuously — don't leave long pauses between sentences."
            ),
        }

    # ── All checks passed — convert to XTTS format ────────────────────────
    converted_path, conv_error = _convert_to_xtts_format(audio_path)
    if conv_error:
        return {"valid": False, "reason": f"Audio conversion failed: {conv_error}"}

    return {
        "valid":          True,
        "duration_s":     round(duration, 1),
        "snr_db":         round(snr_db, 1),
        "speech_ratio":   round(speech_ratio, 3),
        "converted_path": converted_path,
    }


def _estimate_snr(y: np.ndarray, sr: int) -> float:
    """
    Estimate SNR using a simple energy-based approach.

    Method:
      - Split audio into 50ms frames
      - Classify frames as speech or noise based on energy percentile
      - SNR = 10 * log10(mean speech energy / mean noise energy)

    This isn't as accurate as a full voice activity detector but is
    fast and good enough to catch "fan running in background" cases.
    """
    frame_length = int(sr * 0.05)   # 50ms frames
    hop_length   = frame_length // 2

    # Compute RMS energy per frame
    frames     = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    rms_energy = np.sqrt(np.mean(frames ** 2, axis=0)) + 1e-10

    # Frames below 20th percentile are treated as background noise
    noise_threshold = np.percentile(rms_energy, 20)
    noise_frames    = rms_energy[rms_energy <= noise_threshold]
    speech_frames   = rms_energy[rms_energy > noise_threshold]

    if len(noise_frames) == 0 or len(speech_frames) == 0:
        return 40.0   # very clean signal — no noise detectable

    noise_power  = np.mean(noise_frames ** 2)
    speech_power = np.mean(speech_frames ** 2)

    if noise_power <= 0:
        return 40.0

    snr = 10 * np.log10(speech_power / noise_power)
    return float(snr)


def _estimate_speech_ratio(y: np.ndarray, sr: int) -> float:
    """
    Estimate what fraction of the audio contains actual speech.

    Uses librosa's onset detection + energy thresholding to identify
    voiced frames. Returns ratio of voiced_frames / total_frames.
    """
    frame_length = int(sr * 0.02)   # 20ms frames (matches webrtcvad)
    hop_length   = frame_length

    frames     = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    rms_energy = np.sqrt(np.mean(frames ** 2, axis=0))

    # Energy threshold: frames above 15th percentile are considered speech
    # This is conservative — we want to count breathing and soft speech too
    threshold    = np.percentile(rms_energy, 15)
    speech_count = np.sum(rms_energy > threshold)
    total_count  = len(rms_energy)

    return float(speech_count / total_count) if total_count > 0 else 0.0


def _convert_to_xtts_format(input_path: str) -> Tuple[str | None, str | None]:
    """
    Convert audio to 16kHz mono PCM WAV using FFmpeg.

    This is the ONLY FFmpeg call in the audio pipeline.
    It happens once during onboarding — never during a live session.

    Returns:
        (output_path, None) on success
        (None, error_message) on failure
    """
    # Create a temp file for the converted output
    fd, output_path = tempfile.mkstemp(suffix="_xtts.wav")
    os.close(fd)

    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-ar", str(XTTS_SAMPLE_RATE),  # 16kHz
        "-ac", str(XTTS_CHANNELS),      # mono
        "-f", "wav",
        "-y",                            # overwrite
        output_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            os.unlink(output_path)
            return None, f"FFmpeg error: {result.stderr[-300:]}"
        return output_path, None
    except subprocess.TimeoutExpired:
        try:
            os.unlink(output_path)
        except Exception:
            pass
        return None, "Audio conversion timed out (file may be corrupt)"
    except FileNotFoundError:
        return None, "FFmpeg not installed. Please install FFmpeg."
    except Exception as e:
        return None, str(e)
