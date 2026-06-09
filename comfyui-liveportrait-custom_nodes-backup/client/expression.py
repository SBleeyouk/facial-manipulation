"""
Expression preset → coefficient mapping.

Key names must match the INPUT_TYPES of the LivePortraitProcess node installed in
your ComfyUI environment.  Run:  python main.py --validate
to print the actual accepted inputs; then update these dicts accordingly.
"""

EXPRESSIONS: dict[str, dict[str, float]] = {
    "neutral": {},

    "sad": {
        "brow_inner_up":  0.5,
        "eye_squint_l":   0.3,
        "eye_squint_r":   0.3,
        "mouth_frown_l":  0.6,
        "mouth_frown_r":  0.6,
        "mouth_down":     0.7,
    },

    "angry": {
        "brow_down_l":   0.8,
        "brow_down_r":   0.8,
        "lip_tighten":   0.6,
        "nose_sneer_l":  0.4,
        "nose_sneer_r":  0.4,
    },

    "surprised": {
        "eye_wide_l":        0.9,
        "eye_wide_r":        0.9,
        "brow_outer_up_l":   0.7,
        "brow_outer_up_r":   0.7,
        "mouth_open":        0.7,
    },

    "happy": {
        "mouth_smile_l":   0.8,
        "mouth_smile_r":   0.8,
        "cheek_squint_l":  0.5,
        "cheek_squint_r":  0.5,
    },
}

# All possible coefficient keys across all expressions (used for zeroing defaults)
ALL_EXPRESSION_KEYS: set[str] = set().union(*EXPRESSIONS.values())


def get_expression(prompt: str) -> dict[str, float]:
    """Return coefficient dict for the given expression name (defaults to neutral)."""
    return EXPRESSIONS.get(prompt, {})


def list_expressions() -> list[str]:
    return list(EXPRESSIONS.keys())


def get_full_expression(prompt: str) -> dict[str, float]:
    """Return a dict with every known key, zeroing out keys not in this expression."""
    base = {k: 0.0 for k in ALL_EXPRESSION_KEYS}
    base.update(EXPRESSIONS.get(prompt, {}))
    return base
