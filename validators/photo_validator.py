"""
validators/photo_validator.py  —  PhotoValidator

Validates a user-uploaded photo (or video frame) before sending to Simli.

Relaxed thresholds for video frames which are naturally lower quality
than dedicated portrait photos.

Three checks (all must pass):
  1. Face detected         — cv2 frontal-face Haar cascade
  2. Face large enough     — face must be ≥ 2% of image area
  3. Laplacian var > 30    — image not severely blurry

Eye detection removed: too aggressive on video frames, side lighting,
and many valid portrait orientations. Face presence is sufficient.
"""

from typing import Dict
import cv2
import numpy as np

MIN_LAPLACIAN_VAR   = 30.0    # lowered from 100 — video frames are softer
MIN_DIMENSION_PX    = 240     # lowered from 512 — video frames can be smaller
MIN_FACE_AREA_RATIO = 0.02    # face must be ≥ 2% of image area


class PhotoValidator:

    def __init__(self):
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self._face_cascade.empty():
            raise RuntimeError(
                "OpenCV frontal-face cascade not found. "
                "Ensure opencv-python-headless is installed correctly."
            )

    def validate(self, image_bytes: bytes) -> Dict:
        """
        Validate a photo or video frame for Simli agent creation.

        Returns:
            {"valid": True, "resolution": "WxH", "sharpness": float}
            {"valid": False, "reason": "Human-readable rejection message"}
        """
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            return {
                "valid":  False,
                "reason": "Could not decode image. Please upload a JPG, PNG, or try a different video.",
            }

        h, w = img.shape[:2]

        if min(h, w) < MIN_DIMENSION_PX:
            return {
                "valid":  False,
                "reason": (
                    f"Image resolution too low ({w}x{h}px). "
                    f"Please use a higher quality video or photo."
                ),
            }

        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if lap_var < MIN_LAPLACIAN_VAR:
            return {
                "valid":  False,
                "reason": (
                    f"Image is too blurry (sharpness: {lap_var:.0f}). "
                    f"Please use a clearer video or a well-lit photo."
                ),
            }

        # Try face detection — two passes with progressively relaxed params
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
        )
        if len(faces) == 0:
            faces = self._face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=2, minSize=(40, 40)
            )
        if len(faces) == 0:
            # Final attempt — equalize histogram to handle dark/uneven lighting
            eq = cv2.equalizeHist(gray)
            faces = self._face_cascade.detectMultiScale(
                eq, scaleFactor=1.05, minNeighbors=2, minSize=(40, 40)
            )

        if len(faces) == 0:
            return {
                "valid":  False,
                "reason": (
                    "No face detected. Please ensure the person's face is "
                    "clearly visible and facing the camera."
                ),
            }

        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        face_area_ratio = (fw * fh) / (w * h)

        if face_area_ratio < MIN_FACE_AREA_RATIO:
            return {
                "valid":  False,
                "reason": (
                    "Face is too small in the frame. "
                    "Please use a video where the person's face is more prominent."
                ),
            }

        return {
            "valid":      True,
            "resolution": f"{w}x{h}",
            "sharpness":  round(lap_var, 1),
        }

    def close(self):
        pass


_validator = None

def get_photo_validator() -> PhotoValidator:
    global _validator
    if _validator is None:
        _validator = PhotoValidator()
    return _validator
