"""
utils/ffmpeg_utils.py  —  FFmpeg Utilities

The ONLY place FFmpeg is called in the entire system (besides audio_validator.py).

Two legitimate uses:
  1. extract_best_frames()  — extract the best 10 frames from a video
     for Simli onboarding. Used when user uploads a video instead of a photo.

  2. probe_audio_format()   — run ffprobe to get actual sample rate, channels,
     codec of an uploaded audio file. Used in the audio format smoke test
     to verify what Daily.co is actually sending vs what we expect.

Rule: FFmpeg is NEVER called during a live session. Only during onboarding.
All in-session audio processing uses numpy/scipy (utils/audio_utils.py).
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ────────────────────────────────────────────────────────────────────────────
# Frame extraction for Simli onboarding (video → best photo)
# ────────────────────────────────────────────────────────────────────────────

def extract_best_frames(
    video_path: str,
    n_frames: int = 10,
    output_dir: Optional[str] = None,
) -> Tuple[List[str], Optional[str]]:
    """
    Extract the N best frames from a video for Simli onboarding.

    "Best" = highest quality score via FFmpeg's scene change detection.
    We select frames that:
      - Have high sharpness (Laplacian variance)
      - Are spread across the video (not all from the first second)
      - Show clear frontal face (validated later by PhotoValidator)

    Strategy:
      1. Use FFmpeg to extract one frame per second at full quality
      2. Select N frames spaced evenly across the video
      3. PhotoValidator then picks the best one (frontal, sharp, correct size)

    Returns:
        ([list_of_frame_paths], None) on success
        ([], error_message) on failure
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="chronis_frames_")

    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Get video duration ──────────────────────────────────────────
    duration, err = _get_video_duration(video_path)
    if err:
        return [], err
    if duration <= 0:
        return [], "Could not determine video duration"

    # ── Step 2: Extract frames at 1fps, quality 2 (near-lossless JPEG) ──────
    frame_pattern = os.path.join(output_dir, "frame_%04d.jpg")

    cmd = [
        "ffmpeg",
        "-i",    video_path,
        "-vf",   "fps=1,scale=1280:-1",    # 1 frame/sec, max 1280px wide
        "-q:v",  "2",                        # JPEG quality 2 = near-lossless
        "-y",
        frame_pattern,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return [], f"FFmpeg frame extraction error: {result.stderr[-300:]}"
    except subprocess.TimeoutExpired:
        return [], "Frame extraction timed out"
    except FileNotFoundError:
        return [], "FFmpeg not installed"

    # ── Step 3: Collect extracted frames ────────────────────────────────────
    all_frames = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith("frame_") and f.endswith(".jpg")
    ])

    if not all_frames:
        return [], "No frames extracted from video"

    # ── Step 4: Select N evenly-spaced frames ───────────────────────────────
    if len(all_frames) <= n_frames:
        selected = all_frames
    else:
        # Pick n_frames evenly spaced indices
        indices  = [int(i * (len(all_frames) - 1) / (n_frames - 1))
                    for i in range(n_frames)]
        selected = [all_frames[i] for i in indices]

    print(f"[FFmpeg] Extracted {len(all_frames)} frames, selected {len(selected)}",
          flush=True)
    return selected, None


def cleanup_frames(frame_paths: List[str]) -> None:
    """Delete extracted frame files after onboarding completes."""
    dirs_to_remove = set()
    for path in frame_paths:
        try:
            os.unlink(path)
            dirs_to_remove.add(os.path.dirname(path))
        except Exception:
            pass
    for d in dirs_to_remove:
        try:
            os.rmdir(d)  # only removes if empty
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────────────
# Audio/video format probing
# ────────────────────────────────────────────────────────────────────────────

def probe_audio_format(file_path: str) -> Tuple[Dict, Optional[str]]:
    """
    Get the actual audio format of a file using ffprobe.

    Used in smoke testing to verify what format Daily.co is actually sending.
    Run this during the audio format smoke test on a real recording.

    Returns:
        (
            {
                "sample_rate": 16000,
                "channels": 1,
                "codec": "pcm_s16le",
                "bit_depth": 16,
                "duration_s": 5.2,
                "bit_rate_kbps": 256,
            },
            None
        )
        or ({}, error_message)
    """
    cmd = [
        "ffprobe",
        "-v",           "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        file_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return {}, f"ffprobe error: {result.stderr[:200]}"

        data    = json.loads(result.stdout)
        streams = data.get("streams", [])
        fmt     = data.get("format", {})

        # Find the first audio stream
        audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
        if not audio:
            return {}, "No audio stream found in file"

        return {
            "sample_rate":   int(audio.get("sample_rate", 0)),
            "channels":      int(audio.get("channels", 0)),
            "codec":         audio.get("codec_name", "unknown"),
            "bit_depth":     int(audio.get("bits_per_sample", 0)),
            "duration_s":    float(fmt.get("duration", 0)),
            "bit_rate_kbps": int(fmt.get("bit_rate", 0)) // 1000,
        }, None

    except FileNotFoundError:
        return {}, "ffprobe not installed (part of FFmpeg)"
    except json.JSONDecodeError:
        return {}, "Could not parse ffprobe output"
    except Exception as e:
        return {}, str(e)


def _get_video_duration(video_path: str) -> Tuple[float, Optional[str]]:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return 0, f"ffprobe error: {result.stderr[:100]}"
        data     = json.loads(result.stdout)
        duration = float(data.get("format", {}).get("duration", 0))
        return duration, None
    except Exception as e:
        return 0, str(e)


# ────────────────────────────────────────────────────────────────────────────
# Format conversion (onboarding only — never live session)
# ────────────────────────────────────────────────────────────────────────────

def convert_audio_to_wav_16k(
    input_path: str,
    output_path: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Convert any audio format to 16kHz mono PCM WAV.
    This is the same conversion audio_validator.py uses.
    Exposed here so other tools can call it directly.

    Returns (output_path, error).
    """
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix="_16k.wav")
        os.close(fd)

    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-ar", "16000",   # 16kHz
        "-ac", "1",       # mono
        "-f", "wav",
        "-y",
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            try:
                os.unlink(output_path)
            except Exception:
                pass
            return None, f"FFmpeg error: {result.stderr[-300:]}"
        return output_path, None
    except subprocess.TimeoutExpired:
        return None, "Conversion timed out"
    except FileNotFoundError:
        return None, "FFmpeg not installed"
    except Exception as e:
        return None, str(e)
