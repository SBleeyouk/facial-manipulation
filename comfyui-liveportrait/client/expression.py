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


def get_expression(prompt: str) -> dict[str, float]:
    return EXPRESSIONS.get(prompt, {})


def list_expressions() -> list[str]:
    return list(EXPRESSIONS.keys())


def get_full_expression(prompt: str) -> dict[str, float]:
    base = {k: 0.0 for k in ALL_EXPRESSION_KEYS}
    base.update(EXPRESSIONS.get(prompt, {}))
    return base
