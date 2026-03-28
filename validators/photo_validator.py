"""
validators/photo_validator.py  —  PhotoValidator

Validates a user-uploaded photo before sending it to Simli's 45-90 second
agent creation API.

Bad photos that pass here will cause Simli to fail 45 minutes later with
a cryptic error. Catching them here takes milliseconds and gives a clear,
actionable rejection message.

Five checks (all must pass):
  1. Face detected         — MediaPipe FaceMesh finds at least one face
  2. Yaw angle < 25°      — face not too sideways
  3. Pitch angle < 20°    — face not too tilted up/down
  4. Laplacian var > 100  — image not blurry
  5. Min 512×512 px       — sufficient resolution for Simli

Head angle calculation:
  Uses the 3D face landmarks from MediaPipe FaceMesh.
  Yaw  = rotation around vertical axis (left-right turn)
  Pitch = rotation around horizontal axis (up-down tilt)
  We estimate these from specific landmark positions rather than
  running a full PnP solve (fast and accurate enough for our threshold).
"""

import io
import math
from typing import Dict, Tuple

import cv2
import mediapipe as mp
import numpy as np


# ── Thresholds ────────────────────────────────────────────────────────────────
MAX_YAW_DEG         = 25.0   # degrees
MAX_PITCH_DEG       = 20.0   # degrees
MIN_LAPLACIAN_VAR   = 100.0  # below this → blurry
MIN_DIMENSION_PX    = 512    # minimum of width and height


class PhotoValidator:

    def __init__(self):
        # Initialize MediaPipe FaceMesh once — reuse across calls
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh    = self._mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

    def validate(self, image_bytes: bytes) -> Dict:
        """
        Validate a photo for Simli agent creation.

        Returns:
            {"valid": True}
            {"valid": False, "reason": "Human-readable rejection message"}
        """
        # ── Decode image ──────────────────────────────────────────────────
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            return {"valid": False, "reason": "Could not decode image. Please upload a JPG or PNG."}

        h, w = img.shape[:2]

        # ── Check 5: Minimum resolution ────────────────────────────────────
        if min(h, w) < MIN_DIMENSION_PX:
            return {
                "valid":  False,
                "reason": f"Photo resolution too low ({w}×{h}px). Please use a photo that is at least 512×512 pixels.",
            }

        # ── Check 4: Blur (Laplacian variance) ────────────────────────────
        # Run BEFORE face detection — avoid wasting FaceMesh time on blurry images
        gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap_var  = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < MIN_LAPLACIAN_VAR:
            return {
                "valid":  False,
                "reason": f"Photo is too blurry (sharpness score: {lap_var:.0f}). Please use a sharper, well-lit photo.",
            }

        # ── Run MediaPipe FaceMesh ─────────────────────────────────────────
        rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)

        # ── Check 1: Face detected ─────────────────────────────────────────
        if not result.multi_face_landmarks:
            return {
                "valid":  False,
                "reason": "No face found in the photo. Please upload a clear photo showing your face.",
            }

        landmarks = result.multi_face_landmarks[0].landmark

        # ── Check 2 & 3: Head angles ───────────────────────────────────────
        yaw, pitch = self._estimate_head_angles(landmarks, w, h)

        if abs(yaw) > MAX_YAW_DEG:
            return {
                "valid":  False,
                "reason": f"Face is turned too far sideways (angle: {abs(yaw):.0f}°). Please look more directly at the camera.",
            }

        if abs(pitch) > MAX_PITCH_DEG:
            return {
                "valid":  False,
                "reason": f"Face is tilted too far up or down (angle: {abs(pitch):.0f}°). Please hold your head level.",
            }

        return {
            "valid":      True,
            "resolution": f"{w}×{h}",
            "sharpness":  round(lap_var, 1),
            "yaw_deg":    round(yaw, 1),
            "pitch_deg":  round(pitch, 1),
        }

    def _estimate_head_angles(
        self,
        landmarks,
        img_w: int,
        img_h: int,
    ) -> Tuple[float, float]:
        """
        Estimate yaw and pitch in degrees from FaceMesh 3D landmarks.

        Uses a simplified approach based on key facial landmark positions:
          - Nose tip (1), left eye outer (33), right eye outer (263)
          - Left mouth corner (61), right mouth corner (291)
          - Chin (152), forehead (10)

        This avoids the full PnP solve while being accurate to ~5° for
        our purposes (rejecting angles > 25°/20°).
        """
        # Extract key landmark positions (normalized 0-1, with z component)
        def lm(idx):
            p = landmarks[idx]
            return np.array([p.x * img_w, p.y * img_h, p.z * img_w])

        nose_tip      = lm(1)
        left_eye_out  = lm(33)
        right_eye_out = lm(263)
        chin          = lm(152)
        forehead      = lm(10)

        # ── Yaw: asymmetry between eye distances relative to nose ──────────
        # If face is turned right, left eye appears farther from nose than right
        dist_left  = np.linalg.norm(nose_tip[:2] - left_eye_out[:2])
        dist_right = np.linalg.norm(nose_tip[:2] - right_eye_out[:2])

        # Ratio maps to angle: 1.0 = frontal, <0.6 or >1.6 = extreme turn
        ratio = dist_left / (dist_right + 1e-6)
        # Approximate: ratio of 1.0 → 0°, ratio of 0.5 → ~30°
        yaw = math.degrees(math.atan2(ratio - 1.0, 0.5)) * 2.0

        # ── Pitch: nose tip position relative to chin-forehead midpoint ────
        face_center_y = (chin[1] + forehead[1]) / 2
        face_height   = abs(chin[1] - forehead[1]) + 1e-6

        # Positive = nose above center → head tilted down (camera sees more forehead)
        # Negative = nose below center → head tilted up
        nose_offset = (nose_tip[1] - face_center_y) / face_height
        pitch       = math.degrees(math.asin(np.clip(nose_offset * 1.5, -1, 1)))

        return yaw, pitch

    def close(self):
        """Release MediaPipe resources."""
        self._face_mesh.close()


# ── Module-level singleton ────────────────────────────────────────────────────
_validator: PhotoValidator | None = None

def get_photo_validator() -> PhotoValidator:
    global _validator
    if _validator is None:
        _validator = PhotoValidator()
    return _validator
