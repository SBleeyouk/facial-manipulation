"""
Expression preset → ExpressionEditor coefficient mapping.

Key names match the INPUT_TYPES of the ExpressionEditor node
(ComfyUI-AdvancedLivePortrait).  Run: python main.py --validate
to print the actual accepted inputs from your installed nodes.

ExpressionEditor parameters:
  smile    [-0.3 .. 1.3]   smile intensity
  eyebrow  [-10  .. 15]    raise (positive) or furrow (negative)
  blink    [-20  .. 5]     close eyes (positive) or widen (negative)
  aaa      [-30  .. 120]   mouth open (jaw drop)
  eee      [-20  .. 15]    "eee" mouth shape
  woo      [-20  .. 15]    "woo" / pucker shape
  wink     [0    .. 25]    one-eye wink
  pupil_x  [-15  .. 15]    horizontal gaze
  pupil_y  [-15  .. 15]    vertical gaze
"""

EXPRESSIONS: dict[str, dict[str, float]] = {
    "neutral": {},

    "happy": {
        "smile":   1.0,
        "eyebrow": 5.0,
        "blink":   -2.0,   # slightly wider eyes
    },

    "sad": {
        "smile":   -0.3,
        "eyebrow": -5.0,
        "blink":   3.0,    # droopy eyelids
    },

    "angry": {
        "eyebrow": -8.0,
        "blink":   -3.0,   # narrowed eyes
    },

    "surprised": {
        "eyebrow": 10.0,
        "blink":   -10.0,  # wide eyes
        "aaa":     60.0,   # open mouth
    },
}

ALL_EXPRESSION_KEYS: set[str] = set().union(*EXPRESSIONS.values())

# Coefficient -> (min, max), taken from the ExpressionEditor docstring above.
# Used to normalise raw coefficient values to a [0, 1] strength for display.
COEFFICIENT_RANGES: dict[str, tuple[float, float]] = {
    "smile":   (-0.3, 1.3),
    "eyebrow": (-10.0, 15.0),
    "blink":   (-20.0, 5.0),
    "aaa":     (-30.0, 120.0),
    "eee":     (-20.0, 15.0),
    "woo":     (-20.0, 15.0),
    "wink":    (0.0, 25.0),
}

# Coefficient -> FACS Action Unit(s) it drives, for the on-screen overlay.
# "anchor" names match the points returned by FacePoseEstimator.extract_au_anchors().
# pos/neg describe the AU + arrow direction for positive vs. negative coefficient
# values (several sliders flip meaning across zero, e.g. eyebrow raise/furrow).
AU_MAP: dict[str, dict] = {
    "smile":   {"anchor": "mouth_corners", "pos": ("AU12", "Lip Corner Puller", "out_up"),
                                            "neg": ("AU15", "Lip Corner Depressor", "out_down")},
    "eyebrow": {"anchor": "brows",         "pos": ("AU1+2", "Brow Raiser", "up"),
                                            "neg": ("AU4", "Brow Lowerer", "down")},
    "blink":   {"anchor": "eyes",          "pos": ("AU43", "Eyes Closed", "down"),
                                            "neg": ("AU5", "Upper Lid Raiser", "up")},
    "wink":    {"anchor": "eye_r",         "pos": ("AU46", "Wink", "down")},
    "aaa":     {"anchor": "mouth_center",  "pos": ("AU26/27", "Jaw Drop", "down")},
    "eee":     {"anchor": "mouth_corners", "pos": ("AU20", "Lip Stretcher", "out")},
    "woo":     {"anchor": "mouth_corners", "pos": ("AU18", "Lip Pucker", "in")},
}


def normalized_strength(name: str, value: float) -> float:
    """Coefficient value -> [0, 1] strength relative to its full range."""
    lo, hi = COEFFICIENT_RANGES.get(name, (-1.0, 1.0))
    span = max(abs(lo), abs(hi)) or 1.0
    return min(abs(value) / span, 1.0)


def expression_intensity(coeffs: dict[str, float]) -> float:
    """
    Overall "degree of expression" in [0, 1] for the current coefficient set.

    Uses the strongest single coefficient rather than an average: a single
    extreme AU (e.g. aaa=60 for "surprised") should read as high intensity
    even though most of the other sliders stay at zero.
    """
    if not coeffs:
        return 0.0
    return max(normalized_strength(k, v) for k, v in coeffs.items())


def get_expression(prompt: str) -> dict[str, float]:
    return EXPRESSIONS.get(prompt, {})


def list_expressions() -> list[str]:
    return list(EXPRESSIONS.keys())


def get_full_expression(prompt: str) -> dict[str, float]:
    base = {k: 0.0 for k in ALL_EXPRESSION_KEYS}
    base.update(EXPRESSIONS.get(prompt, {}))
    return base
