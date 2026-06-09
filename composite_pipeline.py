#!/usr/bin/env python3
"""
composite_pipeline.py

Captures webcam frames and runs each through two parallel paths:
  Path A (LivePortrait) — ComfyUI HTTP API at localhost:8188
  Path B (FluxRT)       — subprocess call to FluxRT/

Composites the results (LivePortrait face over FluxRT background) and
saves N frames as an MP4 at 1 fps (CPU-friendly default).

Usage:
  python composite_pipeline.py [--frames 30] [--prompt "..."] [--output out.mp4]
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python not found.  pip install opencv-python")
    sys.exit(1)

# ─── paths ────────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).parent
COMFYUI_DIR   = BASE_DIR / "ComfyUI"
FLUXRT_DIR    = BASE_DIR / "FluxRT"
WORKFLOW_PATH = (
    BASE_DIR
    / "ComfyUI"
    / "user"
    / "default"
    / "workflows"
    / "liveportrait_realtime_webcam.json"
)
COMFYUI_URL = "http://localhost:8188"
CLIENT_ID   = str(uuid.uuid4())

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ─── preflight ────────────────────────────────────────────────────────────────

def _comfyui_running() -> bool:
    try:
        urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=3)
        return True
    except Exception:
        return False


def _fluxrt_installed() -> bool:
    for name in ("run_pipeline.py", "main.py", "pipeline.py", "infer.py"):
        if (FLUXRT_DIR / name).exists():
            return True
    return False


def run_preflight() -> tuple[bool, bool]:
    """Check dependencies; return (comfyui_ok, fluxrt_ok)."""
    comfyui_ok = _comfyui_running() and WORKFLOW_PATH.exists()
    fluxrt_ok  = _fluxrt_installed()

    if not _comfyui_running():
        log.warning("ComfyUI not reachable at %s — Path A (LivePortrait) will be skipped.", COMFYUI_URL)
    elif not WORKFLOW_PATH.exists():
        log.warning("Workflow JSON missing at %s — Path A will be skipped.", WORKFLOW_PATH)
    else:
        log.info("ComfyUI  OK  (%s)", COMFYUI_URL)
        log.info("Workflow OK  (%s)", WORKFLOW_PATH)

    if not fluxrt_ok:
        log.warning(
            "FluxRT entry point not found in %s — Path B will be skipped.\n"
            "  Clone with: git clone https://github.com/tensorforger/FluxRT %s",
            FLUXRT_DIR, FLUXRT_DIR,
        )
    else:
        log.info("FluxRT   OK  (%s)", FLUXRT_DIR)

    return comfyui_ok, fluxrt_ok


# ─── ComfyUI / LivePortrait (Path A) ─────────────────────────────────────────

def _upload_image(img_bgr: np.ndarray, filename: str) -> str | None:
    """Upload a BGR frame to ComfyUI /upload/image; return server-side filename."""
    _, png_buf = cv2.imencode(".png", img_bgr)
    boundary = "----CompositeBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + png_buf.tobytes() + f"\r\n--{boundary}--\r\n".encode()

    try:
        req = urllib.request.Request(
            f"{COMFYUI_URL}/upload/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=15)
        return json.loads(resp.read()).get("name")
    except Exception as exc:
        log.error("ComfyUI upload error: %s", exc)
        return None


def _build_api_prompt(src_filename: str) -> dict:
    """
    Build a ComfyUI API-format prompt dict that replicates the LivePortrait
    webcam workflow but feeds a static uploaded image instead of live webcam.

    Graph:
      LoadImage (src) ──┐
                        ├─► ExpressionEditor ──► AdvancedLivePortrait ──► SaveImage
      LoadImage (drv) ──┘        ▲
      LPEmotionPreset ───────────┘
    """
    return {
        # Node 1 — source appearance image
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": src_filename, "upload": "image"},
        },
        # Node 2 — driver / motion image (same frame on CPU; fine for verification)
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": src_filename, "upload": "image"},
        },
        # Node 10 — emotion preset (matches original workflow)
        "10": {
            "class_type": "LPEmotionPreset",
            "inputs": {"emotion": "surprised"},
        },
        # Node 5 — expression editor
        "5": {
            "class_type": "ExpressionEditor",
            "inputs": {
                "src_image":    ["1",  0],
                "smile":        ["10", 0],
                "eyebrow":      ["10", 1],
                "blink":        ["10", 2],
                "aaa":          ["10", 3],
                "eee":          ["10", 4],
                "woo":          ["10", 5],
                "wink":         ["10", 6],
                "rotate_pitch": 0, "rotate_yaw": 0, "rotate_roll": 0,
                "pupil_x": 0,  "pupil_y": 0,
                "src_ratio": 1, "sample_ratio": 1,
                "sample_parts": "All",
                "crop_factor":  1.7,
            },
        },
        # Node 6 — AdvancedLivePortrait
        "6": {
            "class_type": "AdvancedLivePortrait",
            "inputs": {
                "src_images":       ["5", 0],
                "motion_link":      ["5", 1],
                "driving_images":   ["2", 0],
                "retargeting_eyes": 0,
                "retargeting_mouth": 0,
                "crop_factor":      1.7,
                "turn_on":          True,
                "tracking_src_vid": True,
                "animate_without_vid": False,
                "command":          "",
            },
        },
        # Node 11 — SaveImage (captures output for /history download)
        "11": {
            "class_type": "SaveImage",
            "inputs": {
                "images":          ["6", 0],
                "filename_prefix": "composite_lp",
            },
        },
    }


def _queue_prompt(api_prompt: dict) -> str | None:
    """POST to /prompt; return prompt_id or None on failure."""
    payload = json.dumps({"prompt": api_prompt, "client_id": CLIENT_ID}).encode()
    try:
        req = urllib.request.Request(
            f"{COMFYUI_URL}/prompt",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read())
        if "error" in data:
            log.error("ComfyUI prompt error: %s", data["error"])
            return None
        return data.get("prompt_id")
    except Exception as exc:
        log.error("ComfyUI queue error: %s", exc)
        return None


def _download_output(prompt_id: str, timeout: int = 300) -> np.ndarray | None:
    """
    Poll /history/{prompt_id} until done; download and return the first output
    image as a BGR ndarray.  timeout is generous for CPU inference.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(
                f"{COMFYUI_URL}/history/{prompt_id}", timeout=5
            )
            history = json.loads(resp.read())
            if prompt_id in history:
                for _node_id, out in history[prompt_id].get("outputs", {}).items():
                    if "images" in out:
                        img_info = out["images"][0]
                        params = urllib.parse.urlencode(
                            {
                                "filename":  img_info["filename"],
                                "subfolder": img_info.get("subfolder", ""),
                                "type":      img_info.get("type", "output"),
                            }
                        )
                        img_resp = urllib.request.urlopen(
                            f"{COMFYUI_URL}/view?{params}", timeout=15
                        )
                        buf = np.frombuffer(img_resp.read(), dtype=np.uint8)
                        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception as exc:
            log.debug("Polling /history: %s", exc)
        time.sleep(3)

    log.error("ComfyUI timed out (prompt %s, %ds)", prompt_id, timeout)
    return None


def run_liveportrait(frame: np.ndarray) -> np.ndarray | None:
    """Path A entry point: upload frame → queue → poll → return result."""
    fname = f"cmp_src_{uuid.uuid4().hex[:8]}.png"
    server_name = _upload_image(frame, fname)
    if server_name is None:
        return None

    prompt_id = _queue_prompt(_build_api_prompt(server_name))
    if prompt_id is None:
        return None

    return _download_output(prompt_id)


# ─── FluxRT (Path B) ──────────────────────────────────────────────────────────

def _fluxrt_entry() -> Path | None:
    for name in ("run_pipeline.py", "main.py", "pipeline.py", "infer.py"):
        p = FLUXRT_DIR / name
        if p.exists():
            return p
    return None


def run_fluxrt(frame: np.ndarray, prompt: str) -> np.ndarray | None:
    """
    Path B entry point: write frame to a temp file, call FluxRT via subprocess,
    load and return the output image.

    Expects FluxRT's entry script to accept:
      --input  <path>   input image
      --output <path>   output image path
      --prompt <text>   scene description
    """
    entry = _fluxrt_entry()
    if entry is None:
        log.error("FluxRT entry point not found in %s", FLUXRT_DIR)
        return None

    tmp_in  = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_out = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    in_path, out_path = tmp_in.name, tmp_out.name
    tmp_in.close()
    tmp_out.close()

    try:
        cv2.imwrite(in_path, frame)
        result = subprocess.run(
            [
                sys.executable, str(entry),
                "--input",  in_path,
                "--output", out_path,
                "--prompt", prompt,
            ],
            capture_output=True,
            text=True,
            timeout=300,          # CPU inference can be slow
            cwd=str(FLUXRT_DIR),
        )
        if result.returncode != 0:
            log.error("FluxRT exited %d — stderr: %s", result.returncode, result.stderr[-800:])
            return None
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            log.error("FluxRT produced no output file at %s", out_path)
            return None
        img = cv2.imread(out_path)
        if img is None:
            log.error("cv2.imread could not decode FluxRT output at %s", out_path)
        return img
    except subprocess.TimeoutExpired:
        log.error("FluxRT subprocess timed out")
        return None
    except Exception as exc:
        log.error("FluxRT unexpected error: %s", exc)
        return None
    finally:
        for p in (in_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


# ─── compositing ──────────────────────────────────────────────────────────────

def _face_mask(img: np.ndarray) -> np.ndarray:
    """
    Detect the face in img and return a soft alpha mask (float32, 0–1, HxW).
    Falls back to centre-third crop when no face is detected.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    mask = np.zeros((h, w), dtype=np.uint8)
    if len(faces) == 0:
        # fallback: centre region
        mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 255
    else:
        for (x, y, fw, fh) in faces:
            pad_x = int(fw * 0.25)
            pad_y = int(fh * 0.35)
            x0, y0 = max(0, x - pad_x),      max(0, y - pad_y)
            x1, y1 = min(w, x + fw + pad_x), min(h, y + fh + pad_y)
            mask[y0:y1, x0:x1] = 255

    # soften edges so the blend looks natural
    mask = cv2.GaussianBlur(mask, (61, 61), 0)
    return mask.astype(np.float32) / 255.0


def composite_frames(face_img: np.ndarray, bg_img: np.ndarray) -> np.ndarray:
    """
    Alpha-blend face_img (LivePortrait) over bg_img (FluxRT) using a
    Haar-cascade face mask on face_img.
    """
    h, w = face_img.shape[:2]
    bg = cv2.resize(bg_img, (w, h))

    alpha = _face_mask(face_img)
    alpha3 = np.stack([alpha, alpha, alpha], axis=2)

    blended = (
        face_img.astype(np.float32) * alpha3
        + bg.astype(np.float32) * (1.0 - alpha3)
    )
    return blended.astype(np.uint8)


# ─── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Composite LivePortrait (ComfyUI) + FluxRT pipeline"
    )
    p.add_argument(
        "--frames", type=int, default=30,
        help="Number of frames to capture (default: 30)",
    )
    p.add_argument(
        "--prompt", type=str,
        default="a cinematic portrait scene, dramatic lighting",
        help="FluxRT scene description prompt",
    )
    p.add_argument(
        "--output", type=str, default="composite_output.mp4",
        help="Output video filename (default: composite_output.mp4)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    comfyui_ok, fluxrt_ok = run_preflight()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log.error("Cannot open webcam (device index 0)")
        sys.exit(1)

    ret, test_frame = cap.read()
    if not ret:
        log.error("Cannot read from webcam")
        cap.release()
        sys.exit(1)

    h, w     = test_frame.shape[:2]
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(args.output, fourcc, 1, (w, h))  # 1 fps

    print(
        f"\n{'─'*60}\n"
        f"  Frames : {args.frames}\n"
        f"  Output : {args.output}  (1 fps, CPU-safe)\n"
        f"  Prompt : {args.prompt}\n"
        f"  Path A (LivePortrait) : {'enabled' if comfyui_ok else 'DISABLED'}\n"
        f"  Path B (FluxRT)       : {'enabled' if fluxrt_ok else 'DISABLED'}\n"
        f"{'─'*60}\n"
    )

    frames_written = 0

    for i in range(1, args.frames + 1):
        ret, frame = cap.read()
        if not ret:
            log.warning("Frame %d/%d: webcam read failed — skipping", i, args.frames)
            continue

        face_img = frame.copy()   # fallback: raw frame
        bg_img   = frame.copy()   # fallback: raw frame
        lp_tag   = "SKIP"
        fx_tag   = "SKIP"

        # ── Path A: LivePortrait ──────────────────────────────────────────────
        if comfyui_ok:
            try:
                result = run_liveportrait(frame)
                if result is not None:
                    face_img = result
                    lp_tag   = "✓"
                else:
                    lp_tag = "FAIL"
            except Exception as exc:
                log.error("LivePortrait unhandled exception on frame %d: %s", i, exc)
                lp_tag = "ERR"

        # ── Path B: FluxRT ────────────────────────────────────────────────────
        if fluxrt_ok:
            try:
                result = run_fluxrt(frame, args.prompt)
                if result is not None:
                    bg_img = result
                    fx_tag = "✓"
                else:
                    fx_tag = "FAIL"
            except Exception as exc:
                log.error("FluxRT unhandled exception on frame %d: %s", i, exc)
                fx_tag = "ERR"

        # ── Composite ─────────────────────────────────────────────────────────
        try:
            composited = composite_frames(face_img, bg_img)
            comp_tag   = "✓"
        except Exception as exc:
            log.error("Composite failed on frame %d: %s", i, exc)
            composited = frame
            comp_tag   = "ERR"

        writer.write(composited)
        frames_written += 1

        print(
            f"Frame {i:>{len(str(args.frames))}}/{args.frames}: "
            f"LivePortrait {lp_tag}  FluxRT {fx_tag}  Composite {comp_tag}"
        )

    cap.release()
    writer.release()
    print(f"\nDone. {frames_written} frame(s) saved → {args.output}")


if __name__ == "__main__":
    main()
