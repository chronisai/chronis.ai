"""
validators/photo_validator.py  —  PhotoValidator

Validates a user-uploaded photo before sending it to Simli's 45-90 second
agent creation API.

Bad photos that pass here will cause Simli to fail 45 minutes later with
a cryptic error. Catching them here takes milliseconds and gives a clear,
actionable rejection message.

Uses OpenCV only (no mediapipe). The frontal-face Haar cascade naturally
rejects extreme yaw angles (> ~35°) because it can only detect near-frontal
faces by design. Eye detection provides a secondary quality signal.

Four checks (all must pass):
  1. Face detected         — cv2 frontal-face Haar cascade finds at least one face
  2. Eyes detectable       — at least one eye found inside the face region
                             (fails for extreme yaw, closed-eye photos, very dark shots)
  3. Laplacian var > 100   — image not blurry
  4. Min 512×512 px        — sufficient resolution for Simli

Why we removed mediapipe:
  mediapipe>=0.10.0 dropped mp.solutions on Python 3.11 Linux in later patch
  releases, causing an AttributeError at startup. The Haar cascade approach
  is dependency-free (bundled with opencv-python-headless), works on all
  platforms, and is fast enough for this use case.
"""

from typing import Dict

import cv2
import numpy as np


# ── Thresholds ────────────────────────────────────────────────────────────────
MIN_LAPLACIAN_VAR   = 100.0   # below this → blurry
MIN_DIMENSION_PX    = 512     # minimum of width and height
MIN_FACE_AREA_RATIO = 0.03    # face must be at least 3% of image area


class PhotoValidator:

    def __init__(self):
        # Both cascade files are bundled with opencv-python-headless — no download needed
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

        if self._face_cascade.empty():
            raise RuntimeError(
                "OpenCV frontal-face cascade not found. "
                "Ensure opencv-python-headless is installed correctly."
            )

    def validate(self, image_bytes: bytes) -> Dict:
        """
        Validate a photo for Simli agent creation.

        Returns:
            {"valid": True, "resolution": "WxH", "sharpness": float}
            {"valid": False, "reason": "Human-readable rejection message"}
        """
        # ── Decode image ──────────────────────────────────────────────────
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            return {
                "valid":  False,
                "reason": "Could not decode image. Please upload a JPG or PNG.",
            }

        h, w = img.shape[:2]

        # ── Check: Minimum resolution ──────────────────────────────────────
        if min(h, w) < MIN_DIMENSION_PX:
            return {
                "valid":  False,
                "reason": (
                    f"Photo resolution too low ({w}x{h}px). "
                    f"Please use a photo that is at least 512x512 pixels."
                ),
            }

        # ── Check: Blur (Laplacian variance) ──────────────────────────────
        # Run BEFORE face detection — skip cascade time on blurry images
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < MIN_LAPLACIAN_VAR:
            return {
                "valid":  False,
                "reason": (
                    f"Photo is too blurry (sharpness score: {lap_var:.0f}). "
                    f"Please use a sharper, well-lit photo."
                ),
            }

        # ── Check: Face detected ───────────────────────────────────────────
        # The frontal-face cascade only detects near-frontal faces (yaw < ~35 deg)
        # so extreme sideways photos fail here automatically.
        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )

        if len(faces) == 0:
            # Retry with relaxed params — catches smaller or darker faces
            faces = self._face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(60, 60),
            )

        if len(faces) == 0:
            return {
                "valid":  False,
                "reason": (
                    "No face found in the photo. "
                    "Please upload a clear, well-lit photo where you are "
                    "looking directly at the camera."
                ),
            }

        # Use the largest detected face
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])

        # ── Check: Face large enough ───────────────────────────────────────
        face_area_ratio = (fw * fh) / (w * h)
        if face_area_ratio < MIN_FACE_AREA_RATIO:
            return {
                "valid":  False,
                "reason": (
                    "Your face is too small in the photo. "
                    "Please move closer to the camera so your face fills more of the frame."
                ),
            }

        # ── Check: Eye detection (angle + quality proxy) ───────────────────
        # At least one eye must be detectable inside the face region.
        # Rejects faces turned sideways enough to hide both eyes (>~30 deg),
        # very dark shots, and sunglasses.
        face_roi = gray[y: y + fh, x: x + fw]
        # Only scan top 60% of face — eye cascade false-positives on mouth/chin
        eye_roi  = face_roi[: int(fh * 0.6), :]
        eyes     = self._eye_cascade.detectMultiScale(
            eye_roi,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20),
        )

        if len(eyes) == 0:
            return {
                "valid":  False,
                "reason": (
                    "Eyes not clearly visible in the photo. "
                    "Please look directly at the camera with both eyes open, "
                    "remove sunglasses, and ensure the photo is well-lit."
                ),
            }

        return {
            "valid":      True,
            "resolution": f"{w}x{h}",
            "sharpness":  round(lap_var, 1),
        }

    def close(self):
        """No-op — cascade classifiers have no persistent resources to release."""
        pass


# ── Module-level singleton ────────────────────────────────────────────────────
_validator = None

def get_photo_validator() -> PhotoValidator:
    global _validator
    if _validator is None:
        _validator = PhotoValidator()
    return _validator
