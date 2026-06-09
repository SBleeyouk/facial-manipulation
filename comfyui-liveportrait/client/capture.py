"""Webcam capture and baseline frame selection."""

import cv2
import numpy as np
from typing import Optional


class WebcamCapture:
    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480) -> None:
        self._cap = cv2.VideoCapture(device_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open webcam device {device_index}")

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()


def select_baseline_frame(
    frames: list[np.ndarray],
    pose_extractor,
) -> tuple[np.ndarray, int]:
    """
    Pick the most frontal / neutral frame from a list.

    Score = |yaw| + |pitch|.  Lowest score wins.
    Returns (frame, index).
    """
    best_frame = frames[0]
    best_idx = 0
    best_score = float("inf")

    for i, frame in enumerate(frames):
        pose = pose_extractor.extract_pose(frame)
        if pose is None:
            continue
        score = abs(pose["yaw"]) + abs(pose["pitch"])
        if score < best_score:
            best_score = score
            best_frame = frame
            best_idx = i

    return best_frame, best_idx


def save_frame(frame: np.ndarray, path: str) -> None:
    cv2.imwrite(path, frame)
