"""
Mutates liveportrait_base.json with per-frame values before each API call.

Importable as a standalone module — TouchDesigner scripts can call
WorkflowBuilder directly without running the full main.py loop.

Parameter mapping
-----------------
Expression coefficients from expression.py are injected into ExpressionEditor
(node EXPR_NODE_ID).  Key names match ExpressionEditor's INPUT_TYPES directly.

Pose values (pitch / yaw / roll from landmarks.py) are also injected into
ExpressionEditor as rotate_pitch / rotate_yaw / rotate_roll.

Eye openness (EAR) from landmarks.py is injected into AdvancedLivePortrait
(node EYE_NODE_ID) as retargeting_eyes.
"""

import copy
import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Node IDs — must match liveportrait_base.json
SOURCE_NODE_ID  = "1"
DRIVING_NODE_ID = "2"
EXPR_NODE_ID    = "9"   # ExpressionEditor — expression + head pose
EYE_NODE_ID     = "10"  # AdvancedLivePortrait — retargeting_eyes

# Maps expression.py keys → ExpressionEditor input names (1:1 here).
EXPRESSION_PARAM_MAP: dict[str, str] = {
    "smile":   "smile",
    "eyebrow": "eyebrow",
    "blink":   "blink",
    "aaa":     "aaa",
    "eee":     "eee",
    "woo":     "woo",
    "wink":    "wink",
}

_DEFAULT_WORKFLOW_PATH = Path(__file__).parent.parent / "workflows" / "liveportrait_base.json"


class WorkflowBuilder:
    def __init__(self, workflow_path: Optional[Path] = None) -> None:
        path = workflow_path or _DEFAULT_WORKFLOW_PATH
        with open(path) as f:
            self._base: dict = json.load(f)
        log.debug("Loaded workflow template from %s", path)

    def build(
        self,
        source_image: str,
        driving_image: str,
        expression: dict[str, float],
        pose: Optional[dict] = None,
    ) -> dict:
        """
        Return a workflow dict ready to POST to ComfyUI /prompt.

        Parameters
        ----------
        source_image  : filename from ComfyUIClient.upload_image() for baseline
        driving_image : filename from ComfyUIClient.upload_image() for current frame
        expression    : coefficient dict from expression.get_full_expression()
        pose          : dict with keys pitch, yaw, roll, retargeting_eyes
                        (from landmarks.py).  Pass None to use workflow defaults.
        """
        wf = copy.deepcopy(self._base)

        # Inject image paths
        wf[SOURCE_NODE_ID]["inputs"]["image"]  = source_image
        wf[DRIVING_NODE_ID]["inputs"]["image"] = driving_image

        # Inject expression coefficients into ExpressionEditor
        expr_inputs = wf[EXPR_NODE_ID]["inputs"]
        for expr_key, node_key in EXPRESSION_PARAM_MAP.items():
            if node_key in expr_inputs:
                expr_inputs[node_key] = float(expression.get(expr_key, 0.0))

        # Inject head pose into ExpressionEditor (clamped to node's [-20, 20] range)
        if pose is not None:
            for src_key, dst_key in (
                ("pitch", "rotate_pitch"),
                ("yaw",   "rotate_yaw"),
                ("roll",  "rotate_roll"),
            ):
                if dst_key in expr_inputs:
                    val = max(-20.0, min(20.0, float(pose.get(src_key, 0.0))))
                    expr_inputs[dst_key] = round(val, 4)

            # Eye openness → AdvancedLivePortrait retargeting_eyes
            eye_inputs = wf[EYE_NODE_ID]["inputs"]
            if "retargeting_eyes" in eye_inputs:
                eye_inputs["retargeting_eyes"] = round(
                    float(pose.get("retargeting_eyes", 0.0)), 3
                )

        return wf

    @classmethod
    def from_default(cls) -> "WorkflowBuilder":
        return cls()
