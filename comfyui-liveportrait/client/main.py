"""
Entry point: webcam loop + ComfyUI orchestration.

Usage
-----
  python main.py                       # default expression: sad
  python main.py --expr happy          # start with a different expression
  python main.py --baseline-frames 60  # longer baseline capture window
  python main.py --validate            # print LivePortrait node schemas and exit
  python main.py --device 1            # use second webcam

Keyboard controls (window must be focused)
  s → sad | a → angry | u → surprised | h → happy | n → neutral
  b → re-capture baseline
  q → quit

OSC (port 9000)
  /expression [string]    e.g.  /expression angry
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

# OSC is optional; graceful fallback if python-osc not installed
try:
    from pythonosc.dispatcher import Dispatcher
    from pythonosc.server import ThreadingOSCUDPServer
    _HAS_OSC = True
except ImportError:
    _HAS_OSC = False

import cv2
import numpy as np
from dotenv import load_dotenv

# Client-package imports (all sibling modules)
sys.path.insert(0, str(Path(__file__).parent))
from capture          import WebcamCapture, select_baseline_frame, save_frame
from comfyui_client   import ComfyUIClient
from display          import Display
from expression       import get_expression, get_full_expression, list_expressions
from landmarks        import FacePoseEstimator
from workflow_builder import WorkflowBuilder

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

# ------------------------------------------------------------------
# Config defaults (overridden by .env / CLI)
# ------------------------------------------------------------------
COMFYUI_HOST = os.getenv("COMFYUI_HOST", "127.0.0.1")
COMFYUI_PORT = int(os.getenv("COMFYUI_PORT", "8188"))
OSC_PORT     = int(os.getenv("OSC_PORT", "9000"))

KEY_MAP = {
    ord("s"): "sad",
    ord("a"): "angry",
    ord("u"): "surprised",
    ord("h"): "happy",
    ord("n"): "neutral",
}

BASELINE_DIR = Path(tempfile.gettempdir()) / "lp_baselines"
BASELINE_DIR.mkdir(exist_ok=True)

# Class-type name prefixes that belong to stock ComfyUI (not custom nodes).
# Used by --validate to filter the "what IS installed" fallback list.
_BUILTIN_NODE_PREFIXES = frozenset({
    "KSampler", "Load", "Save", "Preview", "Image", "VAE", "CLIP",
    "Checkpoint", "LoRA", "ControlNet", "Upscale", "Latent", "Mask",
    "Conditioning", "EmptyLatent", "Efficient", "Note", "PrimitiveNode",
    "Reroute", "BasicScheduler", "ModelSamplingDiscrete",
})


# ------------------------------------------------------------------
# Validate mode
# ------------------------------------------------------------------

def run_validate(host: str, port: int) -> None:
    """Print INPUT_TYPES of all installed LivePortrait nodes then exit."""
    client = ComfyUIClient(host, port)
    log.info("Fetching node schemas from ComfyUI…")
    try:
        client._check_http_reachable(timeout=5)
        nodes = client.list_liveportrait_nodes()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if not nodes:
        print("\nNo LivePortrait nodes found in ComfyUI.\n")
        print("Install these two packages via ComfyUI Manager, then restart ComfyUI:\n")
        print("  1. comfyui-liveportraitkj")
        print("  2. ComfyUI-AdvancedLivePortrait\n")
        print("Steps:")
        print("  • Open http://127.0.0.1:8188 in a browser")
        print("  • Click 'Manager' → 'Install Custom Nodes'")
        print("  • Search for each package name above and install")
        print("  • Restart ComfyUI, then re-run: python client/main.py --validate\n")

        all_nodes = client.get_node_info()
        custom = [k for k in all_nodes if k not in _BUILTIN_NODE_PREFIXES]
        if custom:
            print(f"Custom nodes currently installed ({len(custom)}):")
            for n in sorted(custom):
                print(f"  {n}")
        else:
            print("No custom nodes detected at all — ComfyUI Manager may not be installed either.")
        sys.exit(1)

    for class_type, info in nodes.items():
        print(f"\n{'='*60}")
        print(f"  {class_type}")
        print(f"{'='*60}")
        input_types = info.get("input", {})
        for group, params in input_types.items():
            print(f"\n  [{group}]")
            for name, spec in params.items():
                print(f"    {name}: {spec}")

    print("\nUpdate EXPRESSION_PARAM_MAP in workflow_builder.py if any names differ.")
    sys.exit(0)


# ------------------------------------------------------------------
# Baseline capture
# ------------------------------------------------------------------

def capture_baseline(
    cam: WebcamCapture,
    pose_estimator: FacePoseEstimator,
    n_frames: int,
) -> Path:
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

    path = BASELINE_DIR / "baseline.png"
    save_frame(best, str(path))
    return path


# ------------------------------------------------------------------
# OSC listener (optional, background thread)
# ------------------------------------------------------------------

def start_osc_listener(expression_state: list, port: int = 9000) -> None:
    """expression_state is a single-element list so we can mutate it from the OSC thread."""
    if not _HAS_OSC:
        log.warning("python-osc not installed — OSC listener disabled (pip install python-osc)")
        return

    dispatcher = Dispatcher()

    def handler(address, *args):
        if args:
            expr = str(args[0]).strip().lower()
            if expr in list_expressions():
                expression_state[0] = expr
                log.info("[OSC] expression → %s", expr)
            else:
                log.warning("[OSC] unknown expression '%s' (valid: %s)", expr, list_expressions())

    dispatcher.map("/expression", handler)

    try:
        server = ThreadingOSCUDPServer(("0.0.0.0", port), dispatcher)
        t = threading.Thread(target=server.serve_forever, daemon=True, name="osc-listener")
        t.start()
        log.info("OSC listener on port %d  (/expression [string])", port)
    except Exception as exc:
        log.warning("Could not start OSC listener on port %d: %s", port, exc)


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LivePortrait real-time retargeting client")
    parser.add_argument("--expr",             default="sad",  help="Starting expression")
    parser.add_argument("--baseline-frames",  type=int, default=30, metavar="N",
                        help="Frames to sample when selecting baseline")
    parser.add_argument("--device",           type=int, default=0,  help="Webcam device index")
    parser.add_argument("--host",             default=COMFYUI_HOST,  help="ComfyUI host")
    parser.add_argument("--port",             type=int, default=COMFYUI_PORT, help="ComfyUI port")
    parser.add_argument("--validate",         action="store_true",
                        help="Print LivePortrait node INPUT_TYPES and exit")
    args = parser.parse_args()

    if args.validate:
        run_validate(args.host, args.port)
        return

    if args.expr not in list_expressions():
        print(f"Unknown expression '{args.expr}'. Valid: {list_expressions()}", file=sys.stderr)
        sys.exit(1)

    # --- initialise subsystems ---
    log.info("Opening webcam device %d…", args.device)
    cam = WebcamCapture(device_index=args.device)

    pose_estimator   = FacePoseEstimator()
    workflow_builder = WorkflowBuilder()

    client = ComfyUIClient(host=args.host, port=args.port)
    log.info("Connecting to ComfyUI at %s:%d…", args.host, args.port)
    client.connect()

    display = Display()

    # shared state
    expression_state: list = [args.expr]   # mutable via OSC / keyboard
    start_osc_listener(expression_state, port=OSC_PORT)

    # --- baseline ---
    log.info("=== BASELINE CAPTURE — look at the camera, face forward ===")
    baseline_path = capture_baseline(cam, pose_estimator, args.baseline_frames)
    baseline_filename = client.upload_image(cv2.imread(str(baseline_path)), name="baseline.png")
    log.info("Baseline uploaded as '%s'", baseline_filename)

    # --- main loop ---
    log.info("Starting main loop.  Keys: s=sad a=angry u=surprised h=happy n=neutral b=baseline q=quit")

    last_submit_time = 0.0
    submit_interval  = 1 / 15   # cap at ~15 workflow submissions/sec
    fps_timer        = time.time()
    frame_count      = 0

    try:
        while True:
            loop_start = time.time()

            # 1. Capture
            frame = cam.read()
            if frame is None:
                log.warning("Webcam read failed, retrying…")
                time.sleep(0.05)
                continue

            # 2. Pose extraction (non-blocking, fast)
            pose = pose_estimator.extract_pose(frame)

            # 3. Submit workflow (rate-limited, non-blocking)
            now = time.time()
            if now - last_submit_time >= submit_interval:
                try:
                    driving_filename = client.upload_image(frame, name="driving.png")
                    expr_full = get_full_expression(expression_state[0])
                    workflow  = workflow_builder.build(
                        source_image  = baseline_filename,
                        driving_image = driving_filename,
                        expression    = expr_full,
                        pose          = pose,
                    )
                    client.submit_workflow(workflow)
                    last_submit_time = now
                except Exception as exc:
                    log.warning("Workflow submit failed: %s", exc)

            # 4. Display (never blocks on ComfyUI)
            output_frame = client.get_latest_frame()

            frame_count += 1
            elapsed = now - fps_timer
            fps = frame_count / elapsed if elapsed > 0 else 0.0

            display.show(frame, output_frame, expression_state[0], fps_hint=fps)

            # 5. Keyboard
            key = display.poll_key(wait_ms=1)
            if key == ord("q"):
                break
            elif key in KEY_MAP:
                expression_state[0] = KEY_MAP[key]
                log.info("Expression → %s", expression_state[0])
            elif key == ord("b"):
                log.info("Re-capturing baseline…")
                baseline_path    = capture_baseline(cam, pose_estimator, args.baseline_frames)
                baseline_filename = client.upload_image(
                    cv2.imread(str(baseline_path)), name="baseline.png"
                )
                log.info("New baseline uploaded as '%s'", baseline_filename)

    except KeyboardInterrupt:
        log.info("Interrupted")
    finally:
        cam.release()
        pose_estimator.close()
        client.close()
        display.destroy()
        log.info("Goodbye.")


if __name__ == "__main__":
    main()
