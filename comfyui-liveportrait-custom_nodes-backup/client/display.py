"""Side-by-side display: raw webcam frame (left) | retargeted output (right)."""

from typing import Optional

import cv2
import numpy as np

WINDOW_NAME = "LivePortrait Retargeting"

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.7
_THICKNESS  = 2
_WHITE      = (255, 255, 255)
_BLACK      = (0, 0, 0)
_CYAN       = (255, 220, 0)

_PLACEHOLDER_COLOR = (40, 40, 40)


class Display:
    def __init__(self, target_width: int = 640, target_height: int = 480) -> None:
        self._w = target_width
        self._h = target_height
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, target_width * 2, target_height)

    def show(
        self,
        webcam_frame: np.ndarray,
        output_frame: Optional[np.ndarray],
        expression_label: str = "neutral",
        fps_hint: Optional[float] = None,
    ) -> None:
        left  = self._resize(webcam_frame)
        right = self._resize(output_frame) if output_frame is not None else self._placeholder()

        self._put_label(left,  "RAW",                       top_left=True)
        self._put_label(right, f"EXPR: {expression_label}", top_left=True)

        if fps_hint is not None:
            self._put_label(left, f"{fps_hint:.1f} fps", top_left=False)

        canvas = np.hstack([left, right])
        self._draw_separator(canvas)

        cv2.imshow(WINDOW_NAME, canvas)

    def poll_key(self, wait_ms: int = 1) -> int:
        """Return the pressed key code, or -1."""
        return cv2.waitKey(wait_ms) & 0xFF

    def destroy(self) -> None:
        cv2.destroyWindow(WINDOW_NAME)

    # ------------------------------------------------------------------

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        return cv2.resize(frame, (self._w, self._h), interpolation=cv2.INTER_LINEAR)

    def _placeholder(self) -> np.ndarray:
        img = np.full((self._h, self._w, 3), _PLACEHOLDER_COLOR, dtype=np.uint8)
        _centered_text(img, "Waiting for ComfyUI...", _WHITE)
        return img

    def _draw_separator(self, canvas: np.ndarray) -> None:
        cv2.line(canvas, (self._w, 0), (self._w, self._h), _WHITE, 1)

    @staticmethod
    def _put_label(img: np.ndarray, text: str, top_left: bool) -> None:
        x = 10
        y = 25 if top_left else img.shape[0] - 10
        # Shadow
        cv2.putText(img, text, (x + 1, y + 1), _FONT, _FONT_SCALE, _BLACK, _THICKNESS + 1)
        cv2.putText(img, text, (x, y),         _FONT, _FONT_SCALE, _CYAN,  _THICKNESS)


def _centered_text(img: np.ndarray, text: str, color) -> None:
    h, w = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, _FONT, _FONT_SCALE, _THICKNESS)
    x = (w - tw) // 2
    y = (h + th) // 2
    cv2.putText(img, text, (x, y), _FONT, _FONT_SCALE, color, _THICKNESS)
