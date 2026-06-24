"""Side-by-side display: raw webcam frame (left) | retargeted output (right).

The output panel can additionally render a FACS-style AU overlay (arrows +
AU codes anchored to the deepfaked face) and a "degree of expression" gauge
that tracks overall coefficient strength. Both are driven by data computed
in main.py — display.py only knows how to draw, not how to detect faces.
"""

import math
import time
from typing import Optional

import cv2
import numpy as np

from expression import AU_MAP, expression_intensity, normalized_strength

WINDOW_NAME = "LivePortrait Retargeting"

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.7
_THICKNESS  = 2
_WHITE      = (255, 255, 255)
_BLACK      = (0, 0, 0)
_CYAN       = (255, 220, 0)

_PLACEHOLDER_COLOR = (40, 40, 40)

# -- "Elegant minimal" palette (BGR) ---------------------------------------
_AU_BASE      = (225, 205, 235)   # dim pastel lilac
_AU_BRIGHT    = (255, 235, 255)   # near-white pink highlight

_GAUGE_WIDTH    = 48
_GAUGE_MARGIN   = 22
_GAUGE_TRACK_W  = 8
_GAUGE_BG       = (38, 32, 40)
_GAUGE_LOW      = (235, 205, 175)   # calm pastel periwinkle
_GAUGE_HIGH     = (150, 165, 250)   # warm coral-pink

# Direction name -> base unit vector (image coords, y-down). Mirrored for
# the "l" side in _side_vector so e.g. "out_up" fans outward on both cheeks.
_BASE_VECS = {
    "up":       (0.0, -1.0),
    "down":     (0.0,  1.0),
    "out_up":   (1.0, -1.0),
    "out_down": (1.0,  1.0),
    "out":      (1.0,  0.0),
    "in":       (-1.0, 0.0),
}

# AU_MAP "anchor" name -> (landmark name, side) pairs to draw arrows at.
_ANCHOR_GROUPS = {
    "mouth_corners": (("mouth_l", "l"), ("mouth_r", "r")),
    "brows":         (("brow_l", "l"), ("brow_r", "r")),
    "eyes":          (("eye_l", "l"), ("eye_r", "r")),
    "eye_r":         (("eye_r", "r"),),
    "mouth_center":  (("mouth_center", "r"),),
}


class Display:
    def __init__(self, target_width: int = 640, target_height: int = 480) -> None:
        self._w = target_width
        self._h = target_height
        self._t0 = time.time()

        track_h = max(self._h - 2 * _GAUGE_MARGIN, 1)
        grad = np.zeros((track_h, 1, 3), dtype=np.float32)
        for i in range(track_h):
            t = 1.0 - i / max(track_h - 1, 1)   # row 0 (top) = max intensity colour
            grad[i, 0] = _lerp_color(_GAUGE_LOW, _GAUGE_HIGH, t)
        self._gauge_gradient = grad.astype(np.uint8)
        self._gauge_track_h  = track_h

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, target_width * 2 + _GAUGE_WIDTH, target_height)

    def show(
        self,
        webcam_frame: np.ndarray,
        output_frame: Optional[np.ndarray],
        expression_label: str = "neutral",
        fps_hint: Optional[float] = None,
        expr_coeffs: Optional[dict] = None,
        au_anchors: Optional[dict] = None,
    ) -> None:
        left  = self._resize(webcam_frame)
        right = self._resize(output_frame) if output_frame is not None else self._placeholder()

        if output_frame is not None and expr_coeffs and au_anchors:
            sx = self._w / output_frame.shape[1]
            sy = self._h / output_frame.shape[0]
            self._draw_au_overlay(right, expr_coeffs, au_anchors, sx, sy)

        self._put_label(left,  "RAW",                       top_left=True)
        self._put_label(right, f"EXPR: {expression_label}", top_left=True)

        if fps_hint is not None:
            self._put_label(left, f"{fps_hint:.1f} fps", top_left=False)

        canvas = np.hstack([left, right])
        self._draw_separator(canvas)

        intensity = expression_intensity(expr_coeffs) if expr_coeffs else 0.0
        canvas = np.hstack([canvas, self._render_gauge(intensity)])

        cv2.imshow(WINDOW_NAME, canvas)

    def poll_key(self, wait_ms: int = 1) -> int:
        """Return the pressed key code, or -1."""
        return cv2.waitKey(wait_ms) & 0xFF

    def destroy(self) -> None:
        cv2.destroyWindow(WINDOW_NAME)

    # ------------------------------------------------------------------
    # base frame helpers
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

    # ------------------------------------------------------------------
    # AU overlay ("Elegant minimal": soft pastel arrows that glow/pulse
    # in proportion to how strongly each coefficient is driven)
    # ------------------------------------------------------------------

    def _draw_au_overlay(self, img: np.ndarray, coeffs: dict, anchors: dict, sx: float, sy: float) -> None:
        elapsed = time.time() - self._t0
        for name, value in coeffs.items():
            if not value:
                continue
            meta = AU_MAP.get(name)
            if meta is None:
                continue
            strength = normalized_strength(name, value)
            if strength < 0.04:
                continue

            variant = meta["pos"] if value > 0 else meta.get("neg", meta["pos"])
            au_code, _au_name, direction = variant

            for anchor_name, side in _ANCHOR_GROUPS.get(meta["anchor"], ()):
                point = anchors.get(anchor_name)
                if point is None:
                    continue
                origin = (int(point[0] * sx), int(point[1] * sy))
                vec = self._side_vector(direction, side)
                self._draw_au_arrow(img, origin, vec, strength, au_code, elapsed)

    @staticmethod
    def _side_vector(direction: str, side: str) -> tuple[float, float]:
        dx, dy = _BASE_VECS.get(direction, (0.0, -1.0))
        if side == "l":
            dx = -dx
        norm = math.hypot(dx, dy) or 1.0
        return dx / norm, dy / norm

    def _draw_au_arrow(
        self, img: np.ndarray, origin: tuple[int, int], vec: tuple[float, float],
        strength: float, au_code: str, elapsed: float,
    ) -> None:
        pulse  = 0.7 + 0.3 * math.sin(elapsed * (1.5 + strength * 3.0))
        length = int(10 + 22 * strength * pulse)
        color  = _lerp_color(_AU_BASE, _AU_BRIGHT, strength)
        tip = (int(origin[0] + vec[0] * length), int(origin[1] + vec[1] * length))

        self._glow(img, origin, int(6 + 8 * strength), color, 0.18 + 0.22 * strength)
        cv2.arrowedLine(img, origin, tip, color, 1, cv2.LINE_AA, tipLength=0.45)

        label_pos = (tip[0] + int(vec[0] * 4) - 14, tip[1] + int(vec[1] * 4) + 4)
        cv2.putText(img, au_code, label_pos, _FONT, 0.36, _BLACK, 2, cv2.LINE_AA)
        cv2.putText(img, au_code, label_pos, _FONT, 0.36, color, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Degree-of-expression gauge: a glassy vertical capsule with a soft
    # gradient fill and a glowing marker that rises/falls with intensity.
    # ------------------------------------------------------------------

    def _render_gauge(self, intensity: float) -> np.ndarray:
        intensity = float(np.clip(intensity, 0.0, 1.0))
        gauge = np.full((self._h, _GAUGE_WIDTH, 3), _GAUGE_BG, dtype=np.uint8)

        cx = _GAUGE_WIDTH // 2
        top = _GAUGE_MARGIN
        track_h = self._gauge_track_h
        half_w = _GAUGE_TRACK_W // 2

        track = gauge[top:top + track_h, cx - half_w:cx + half_w]
        dim = (self._gauge_gradient.astype(np.float32) * 0.22).astype(np.uint8)
        track[:] = dim

        fill_rows = int(round(intensity * track_h))
        if fill_rows > 0:
            track[track_h - fill_rows:] = self._gauge_gradient[track_h - fill_rows:]

        cap_color_top = tuple(int(c) for c in dim[0, 0])
        cap_color_bot = tuple(int(c) for c in dim[-1, 0])
        cv2.circle(gauge, (cx, top), half_w, cap_color_top, -1, cv2.LINE_AA)
        cv2.circle(gauge, (cx, top + track_h), half_w, cap_color_bot, -1, cv2.LINE_AA)

        marker_y = top + track_h - fill_rows
        elapsed = time.time() - self._t0
        pulse = 0.6 + 0.4 * math.sin(elapsed * (2.2 + intensity * 4.0))
        color = _lerp_color(_GAUGE_LOW, _GAUGE_HIGH, intensity)
        radius = int(4 + 4 * intensity * pulse) + 2

        self._glow(gauge, (cx, marker_y), radius + 7, color, 0.30 + 0.25 * intensity)
        cv2.circle(gauge, (cx, marker_y), radius, color, -1, cv2.LINE_AA)
        cv2.circle(gauge, (cx, marker_y), radius, _WHITE, 1, cv2.LINE_AA)

        pct = f"{int(round(intensity * 100))}%"
        (tw, _th), _ = cv2.getTextSize(pct, _FONT, 0.38, 1)
        cv2.putText(gauge, pct, (max((_GAUGE_WIDTH - tw) // 2, 0), self._h - 6),
                    _FONT, 0.38, _WHITE, 1, cv2.LINE_AA)
        return gauge

    @staticmethod
    def _glow(img: np.ndarray, center: tuple[int, int], radius: int, color, alpha: float) -> None:
        x, y = center
        h, w = img.shape[:2]
        x0, x1 = max(x - radius, 0), min(x + radius, w)
        y0, y1 = max(y - radius, 0), min(y + radius, h)
        if x1 <= x0 or y1 <= y0:
            return
        roi = img[y0:y1, x0:x1]
        overlay = roi.copy()
        cv2.circle(overlay, (x - x0, y - y0), radius, color, -1, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def _lerp_color(c0, c1, t: float):
    t = max(0.0, min(1.0, t))
    return tuple(int(c0[i] + (c1[i] - c0[i]) * t) for i in range(3))


def _centered_text(img: np.ndarray, text: str, color) -> None:
    h, w = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, _FONT, _FONT_SCALE, _THICKNESS)
    x = (w - tw) // 2
    y = (h + th) // 2
    cv2.putText(img, text, (x, y), _FONT, _FONT_SCALE, color, _THICKNESS)
