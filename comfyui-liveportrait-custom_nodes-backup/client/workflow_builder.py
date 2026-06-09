"""
Mutates liveportrait_base.json with per-frame values before each API call.

Importable as a standalone module — TouchDesigner scripts can call
WorkflowBuilder directly without running the full main.py loop.

Parameter mapping
-----------------
Expression coefficients from expression.py are injected into node EXPR_NODE_ID
using EXPRESSION_PARAM_MAP.  The map is (expression_key → node_input_key);
edit it here if --validate reveals that your installed nodes use different names.

Pose values (pitch / yaw / roll from landmarks.py) are injected into
POSE_NODE_ID as rotate_pitch / rotate_yaw / rotate_roll.

Eye openness from landmarks.py is injected as retargeting_eyes.
retargeting_lip is ALWAYS set to 0.0 (expression channel owns lip control).
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
EXPR_NODE_ID    = "4"   # LivePortraitProcess — receives both expression and pose
POSE_NODE_ID    = "4"

# Maps expression.py coefficient keys → node input parameter names.
# Update if --validate shows your node uses different names.
EXPRESSION_PARAM_MAP: dict[str, str] = {
    "brow_inner_up":    "brow_inner_up",
    "brow_outer_up_l":  "brow_outer_up_l",
    "brow_outer_up_r":  "brow_outer_up_r",
    "brow_down_l":      "brow_down_l",
    "brow_down_r":      "brow_down_r",
    "eye_wide_l":       "eye_wide_l",
    "eye_wide_r":       "eye_wide_r",
    "eye_squint_l":     "eye_squint_l",
    "eye_squint_r":     "eye_squint_r",
    "cheek_squint_l":   "cheek_squint_l",
    "cheek_squint_r":   "cheek_squint_r",
    "nose_sneer_l":     "nose_sneer_l",
    "nose_sneer_r":     "nose_sneer_r",
    "mouth_smile_l":    "mouth_smile_l",
    "mouth_smile_r":    "mouth_smile_r",
    "mouth_frown_l":    "mouth_frown_l",
    "mouth_frown_r":    "mouth_frown_r",
    "mouth_open":       "mouth_open",
    "mouth_down":       "mouth_down",
    "lip_tighten":      "lip_tighten",
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
        source_image  : filename returned by ComfyUIClient.upload_image() for baseline
        driving_image : filename returned by ComfyUIClient.upload_image() for current frame
        expression    : coefficient dict from expression.get_full_expression()
        pose          : dict with keys pitch, yaw, roll, retargeting_eyes (from landmarks.py)
                        Pass None to leave pose at workflow defaults.
        """
        wf = copy.deepcopy(self._base)

        # Inject image paths
        wf[SOURCE_NODE_ID]["inputs"]["image"]  = source_image
        wf[DRIVING_NODE_ID]["inputs"]["image"] = driving_image

        # Inject expression coefficients
        expr_inputs = wf[EXPR_NODE_ID]["inputs"]
        for expr_key, node_key in EXPRESSION_PARAM_MAP.items():
            if node_key in expr_inputs:
                expr_inputs[node_key] = float(expression.get(expr_key, 0.0))
            else:
                log.debug("Node input '%s' not found in workflow — skipping", node_key)

        # Block lip retargeting — expression channel owns it
        expr_inputs["retargeting_lip"] = 0.0

        # Inject pose passthrough
        if pose is not None:
            pose_inputs = wf[POSE_NODE_ID]["inputs"]
            for src_key, dst_key in (
                ("pitch", "rotate_pitch"),
                ("yaw",   "rotate_yaw"),
                ("roll",  "rotate_roll"),
            ):
                if dst_key in pose_inputs:
                    pose_inputs[dst_key] = round(float(pose.get(src_key, 0.0)), 4)

            if "retargeting_eyes" in pose_inputs:
                pose_inputs["retargeting_eyes"] = round(
                    float(pose.get("retargeting_eyes", 0.0)), 3
                )

        return wf

    # ------------------------------------------------------------------
    # TouchDesigner-friendly convenience constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_default(cls) -> "WorkflowBuilder":
        return cls()
