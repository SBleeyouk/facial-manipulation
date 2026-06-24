"""
One-shot LivePortrait -> LTX-2.3 video pipeline.

Unlike main.py (continuous webcam loop + live preview), this captures the
webcam exactly once to get a neutral "baseline" face, generates a single
emotion-edited still image from it (same LivePortrait nodes as the realtime
pipeline), and prepares a ComfyUI workflow that turns those two images into
one video: baseline -> generated emotion -> baseline.

The LTX-2.3 / video-stitch stage is not submitted over the API. LTX-2.3 is
wired into the workflow as a ComfyUI *subgraph* (see ComfyUI/blueprints/),
and subgraph expansion happens client-side in the ComfyUI web frontend, not
on the backend's /prompt endpoint. So this script does everything up to
that point — capture, upload, patch the workflow — and asks you to load it
in the browser and click Queue once.

Usage
-----
  python client/oneshot_pipeline.py --expr happy
  python client/oneshot_pipeline.py --expr sad --baseline-frames 60
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from capture        import WebcamCapture, select_baseline_frame, save_frame
from comfyui_client  import ComfyUIClient
from expression      import list_expressions
from landmarks       import FacePoseEstimator

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("oneshot")

REPO_ROOT     = Path(__file__).parent.parent.parent
COMFYUI_DIR   = REPO_ROOT / "ComfyUI"
SOURCE_WORKFLOW = (
    Path(__file__).parent.parent / "workflows" / "liveportrait_to_ltx_oneshot.json"
)
TARGET_WORKFLOW = (
    COMFYUI_DIR / "user" / "default" / "workflows" / "liveportrait_to_ltx_oneshot.json"
)

EMOTION_NODE_ID  = "10"   # LPEmotionPreset
CLIP_A_NODE_ID   = "100"  # Clip A: baseline -> emotion
CLIP_B_NODE_ID   = "101"  # Clip B: emotion -> baseline

REQUIRED_MODELS = [
    ("checkpoints",   "ltx-2.3-22b-distilled-fp8.safetensors"),
    ("text_encoders", "gemma_3_12B_it_fp4_mixed.safetensors"),
]


def check_ltx_models() -> bool:
    """Warn (don't fail) if the LTX-2.3 weights aren't in place yet."""
    all_present = True
    for subdir, filename in REQUIRED_MODELS:
        path = COMFYUI_DIR / "models" / subdir / filename
        if not path.exists():
            all_present = False
            log.warning("Missing model file: %s", path)
    if not all_present:
        log.warning(
            "LTX-2.3 weights are not installed yet. The workflow will still be "
            "prepared, but clicking Queue in ComfyUI will fail until these files "
            "are downloaded — see README.md for download links and where to put them."
        )
    return all_present


def capture_baseline(cam: WebcamCapture, pose_estimator: FacePoseEstimator, n_frames: int):
    log.info("Capturing %d frames to select baseline…", n_frames)
    frames = []
    for _ in range(n_frames):
        f = cam.read()
        if f is not None:
            frames.append(f)
        time.sleep(1 / 30)

    if not frames:
        raise RuntimeError("Webcam returned no frames during baseline capture.")

    best, idx = select_baseline_frame(frames, pose_estimator)
    log.info("Selected frame %d/%d as baseline (most frontal)", idx + 1, len(frames))
    return best


def patch_workflow(expr: str) -> Path:
    """Set the emotion preset + clip prompts in the workflow, write the patched copy."""
    workflow = json.loads(SOURCE_WORKFLOW.read_text())

    by_id = {str(n["id"]): n for n in workflow["nodes"]}

    by_id[EMOTION_NODE_ID]["widgets_values"][0] = expr

    by_id[CLIP_A_NODE_ID]["widgets_values"][0] = (
        f"A calm, sleeping face gently wakes and its expression smoothly shifts "
        f"into a {expr} expression. Natural, subtle, photorealistic motion, "
        f"no camera movement."
    )
    by_id[CLIP_B_NODE_ID]["widgets_values"][0] = (
        f"An expressive {expr} face calms and smoothly settles back into a "
        f"relaxed sleeping expression. Natural, subtle, photorealistic motion, "
        f"no camera movement."
    )

    TARGET_WORKFLOW.parent.mkdir(parents=True, exist_ok=True)
    TARGET_WORKFLOW.write_text(json.dumps(workflow, indent=2))
    return TARGET_WORKFLOW


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot LivePortrait -> LTX-2.3 video pipeline")
    parser.add_argument("--expr", default="sad", choices=list_expressions(),
                        help="Emotion to morph into (default: sad)")
    parser.add_argument("--baseline-frames", type=int, default=30, metavar="N",
                        help="Frames to sample when selecting the baseline face")
    parser.add_argument("--device", type=int, default=0, help="Webcam device index")
    parser.add_argument("--host", default="127.0.0.1", help="ComfyUI host")
    parser.add_argument("--port", type=int, default=8188, help="ComfyUI port")
    args = parser.parse_args()

    check_ltx_models()

    client = ComfyUIClient(host=args.host, port=args.port)
    client._check_http_reachable(timeout=5)

    log.info("Opening webcam device %d…", args.device)
    cam = WebcamCapture(device_index=args.device)
    pose_estimator = FacePoseEstimator()

    try:
        log.info("=== BASELINE CAPTURE — look at the camera, neutral/sleeping face ===")
        baseline_frame = capture_baseline(cam, pose_estimator, args.baseline_frames)
    finally:
        cam.release()
        pose_estimator.close()

    baseline_filename = client.upload_image(baseline_frame, name="baseline.png")
    log.info("Baseline uploaded as '%s'", baseline_filename)

    workflow_path = patch_workflow(args.expr)
    log.info("Workflow prepared: %s", workflow_path)

    print(
        f"\n{'─'*60}\n"
        f"  Done. Next steps:\n"
        f"  1. Open http://{args.host}:{args.port} in your browser\n"
        f"  2. Open Workflow -> liveportrait_to_ltx_oneshot\n"
        f"  3. Click Queue\n"
        f"  Output video lands in ComfyUI/output/liveportrait_ltx_oneshot/\n"
        f"{'─'*60}\n"
    )


if __name__ == "__main__":
    main()
