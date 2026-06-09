"""
MediaPipe FaceMesh — head pose extraction only.

Expression coefficients are NOT extracted here; they come from expression.py.
Pose (pitch / yaw / roll) is passed through to LivePortrait as a camera-space
retargeting hint so the output head orientation tracks the live feed.

Eye openness (EAR) is also returned so the caller can decide whether to pass
it to the retargeting_eyes slider.
"""

import math
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python as _mp_python
from mediapipe.tasks.python import vision as _mp_vision
import numpy as np

_MODEL_PATH = Path(__file__).parent / "face_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

def _ensure_model() -> str:
    if not _MODEL_PATH.exists():
        print(f"Downloading face landmarker model to {_MODEL_PATH} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return str(_MODEL_PATH)

# 6-point correspondence used for solvePnP head-pose estimation.
# World coords in mm (standard OpenCV face model).
_MODEL_POINTS = np.array([
    (  0.0,    0.0,    0.0),   # Nose tip         → landmark 4
    (  0.0, -330.0,  -65.0),   # Chin             → landmark 152
    (-225.0,  170.0, -135.0),  # L eye L corner   → landmark 33
    ( 225.0,  170.0, -135.0),  # R eye R corner   → landmark 263
    (-150.0, -150.0, -125.0),  # L mouth corner   → landmark 61
    ( 150.0, -150.0, -125.0),  # R mouth corner   → landmark 291
], dtype=np.float64)

_LANDMARK_IDS = [4, 152, 33, 263, 61, 291]

# Landmarks for eye-aspect-ratio
_LEFT_EYE_TOP, _LEFT_EYE_BOT   = 159, 145
_LEFT_EYE_L,   _LEFT_EYE_R     =  33, 133
_RIGHT_EYE_TOP, _RIGHT_EYE_BOT = 386, 374
_RIGHT_EYE_L,   _RIGHT_EYE_R   = 263, 362


def _ear(lm, top_i: int, bot_i: int, left_i: int, right_i: int) -> float:
    """Eye Aspect Ratio, normalised to [0, 1] (1 = fully open baseline)."""
    v = abs(lm[top_i].y - lm[bot_i].y)
    h = abs(lm[left_i].x - lm[right_i].x) + 1e-6
    return min(v / h * 6.0, 1.0)   # ×6 scales typical EAR (~0.15-0.45) to ~[0,1]


class FacePoseEstimator:
    def __init__(self, max_faces: int = 1, refine_landmarks: bool = True) -> None:
        options = _mp_vision.FaceLandmarkerOptions(
            base_options=_mp_python.BaseOptions(model_asset_path=_ensure_model()),
            running_mode=_mp_vision.RunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        self._detector = _mp_vision.FaceLandmarker.create_from_options(options)

    def extract_pose(self, frame: np.ndarray) -> Optional[dict]:
        """
        Returns dict with keys:
            pitch, yaw, roll   — degrees, camera-space Euler angles
            retargeting_eyes   — mean EAR [0, 1]
        Returns None if no face detected.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]

        # Build 2-D image points for solvePnP
        image_points = np.array([
            (lm[i].x * w, lm[i].y * h) for i in _LANDMARK_IDS
        ], dtype=np.float64)

        focal = w
        cam_matrix = np.array([
            [focal, 0,     w / 2],
            [0,     focal, h / 2],
            [0,     0,     1    ],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        ok, rvec, _ = cv2.solvePnP(
            _MODEL_POINTS, image_points, cam_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None

        rmat, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = _rotation_matrix_to_euler(rmat)

        # Reject degenerate solvePnP solutions (gimbal lock / face not fully visible)
        if abs(pitch) > 45 or abs(yaw) > 45 or abs(roll) > 45:
            return None

        eye_ear = (_ear(lm, _LEFT_EYE_TOP, _LEFT_EYE_BOT, _LEFT_EYE_L, _LEFT_EYE_R) +
                   _ear(lm, _RIGHT_EYE_TOP, _RIGHT_EYE_BOT, _RIGHT_EYE_L, _RIGHT_EYE_R)) / 2.0

        return {
            "pitch":           pitch,
            "yaw":             yaw,
            "roll":            roll,
            "retargeting_eyes": round(eye_ear, 3),
        }

    def close(self) -> None:
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _rotation_matrix_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """
    Decomposes a rotation matrix to (pitch, yaw, roll) in degrees.
    Convention: X = pitch, Y = yaw, Z = roll.
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.degrees(math.atan2( R[2, 1], R[2, 2]))
        yaw   = math.degrees(math.atan2(-R[2, 0], sy))
        roll  = math.degrees(math.atan2( R[1, 0], R[0, 0]))
    else:
        pitch = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
        yaw   = math.degrees(math.atan2(-R[2, 0], sy))
        roll  = 0.0

    return pitch, yaw, roll
