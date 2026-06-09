"""
ComfyUI WebSocket + REST client.

Thread model:
  • submit_workflow() / upload_image() — callable from any thread (uses requests, thread-safe)
  • _ws_listener()                     — background daemon thread; writes _latest_frame
  • get_latest_frame()                 — callable from any thread (read under lock)

TouchDesigner interface
-----------------------
  from comfyui_client import ComfyUIClient
  client = ComfyUIClient()
  client.connect()
  client.submit_workflow(workflow_dict)   # fire-and-forget
  frame = client.get_latest_frame()       # np.ndarray (BGR) | None
"""

import base64
import io
import json
import logging
import threading
import time
import uuid
from typing import Callable, Optional

import cv2
import numpy as np
import requests
import websocket       # websocket-client package

log = logging.getLogger(__name__)


class ComfyUIClient:
    OUTPUT_NODE_ID = "6"   # must match SaveImage node id in liveportrait_base.json

    def __init__(self, host: str = "127.0.0.1", port: int = 8188) -> None:
        self._host = host
        self._port = port
        self._client_id = str(uuid.uuid4())
        self._base_url = f"http://{host}:{port}"
        self._ws_url   = f"ws://{host}:{port}/ws?clientId={self._client_id}"

        self._ws: Optional[websocket.WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None

        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        self._frame_callbacks: list[Callable[[np.ndarray], None]] = []

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self, timeout: float = 10.0) -> None:
        """Open WebSocket and verify server is reachable. Raises on failure."""
        self._check_http_reachable(timeout)

        self._ws = websocket.WebSocketApp(
            self._ws_url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open,
        )
        self._ws_thread = threading.Thread(
            target=self._ws.run_forever,
            kwargs={"ping_interval": 30, "ping_timeout": 10},
            daemon=True,
            name="comfyui-ws",
        )
        self._ws_thread.start()
        log.info("WebSocket thread started  client_id=%s", self._client_id)

    def close(self) -> None:
        if self._ws:
            self._ws.close()

    # ------------------------------------------------------------------
    # Image upload
    # ------------------------------------------------------------------

    def upload_image(
        self,
        frame: np.ndarray,
        name: str = "frame.png",
        image_type: str = "input",
        overwrite: bool = True,
    ) -> str:
        """
        Upload a BGR numpy frame to ComfyUI's input folder.
        Returns the filename ComfyUI assigned (use directly in LoadImage node).
        """
        ok, buf = cv2.imencode(".png", frame)
        if not ok:
            raise RuntimeError("cv2.imencode failed")

        resp = requests.post(
            f"{self._base_url}/upload/image",
            files={"image": (name, buf.tobytes(), "image/png")},
            data={"type": image_type, "overwrite": str(overwrite).lower()},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        filename = data["name"]
        log.debug("Uploaded %s → %s", name, filename)
        return filename

    # ------------------------------------------------------------------
    # Workflow submission
    # ------------------------------------------------------------------

    def submit_workflow(self, workflow: dict) -> str:
        """
        POST the workflow to /prompt.  Fire-and-forget; returns prompt_id.
        The WebSocket listener will store the output when execution finishes.
        """
        payload = {"prompt": workflow, "client_id": self._client_id}
        resp = requests.post(
            f"{self._base_url}/prompt",
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()
        prompt_id: str = resp.json()["prompt_id"]
        log.debug("Queued prompt_id=%s", prompt_id)
        return prompt_id

    # ------------------------------------------------------------------
    # Frame retrieval
    # ------------------------------------------------------------------

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return self._latest_frame

    def add_frame_callback(self, fn: Callable[[np.ndarray], None]) -> None:
        """Called on the WS thread each time a new output frame arrives."""
        self._frame_callbacks.append(fn)

    # ------------------------------------------------------------------
    # Validation helper
    # ------------------------------------------------------------------

    def get_node_info(self, class_type: Optional[str] = None) -> dict:
        """
        Fetch /object_info (all nodes) or /object_info/{class_type}.
        Used by --validate flag.
        """
        url = f"{self._base_url}/object_info"
        if class_type:
            url += f"/{class_type}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def list_liveportrait_nodes(self) -> dict:
        """Return all installed nodes whose class_type contains 'liveportrait' (case-insensitive)."""
        all_nodes = self.get_node_info()
        return {
            k: v for k, v in all_nodes.items()
            if "liveportrait" in k.lower()
        }

    # ------------------------------------------------------------------
    # WebSocket callbacks (background thread)
    # ------------------------------------------------------------------

    def _on_open(self, ws) -> None:
        log.info("ComfyUI WebSocket connected")

    def _on_message(self, ws, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        msg_type = msg.get("type")

        if msg_type == "executed":
            data   = msg.get("data", {})
            node   = data.get("node")
            output = data.get("output", {})

            if node == self.OUTPUT_NODE_ID and "images" in output:
                for img_info in output["images"]:
                    frame = self._fetch_output_image(img_info)
                    if frame is not None:
                        with self._frame_lock:
                            self._latest_frame = frame
                        for cb in self._frame_callbacks:
                            try:
                                cb(frame)
                            except Exception:
                                log.exception("Frame callback raised")

        elif msg_type == "execution_error":
            log.error("ComfyUI execution error: %s", msg.get("data"))

        elif msg_type == "status":
            q = msg.get("data", {}).get("status", {}).get("exec_info", {})
            remaining = q.get("queue_remaining", "?")
            log.debug("Queue remaining: %s", remaining)

    def _on_error(self, ws, error) -> None:
        log.warning("WebSocket error: %s", error)

    def _on_close(self, ws, code, reason) -> None:
        log.info("WebSocket closed  code=%s  reason=%s", code, reason)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_output_image(self, img_info: dict) -> Optional[np.ndarray]:
        filename  = img_info.get("filename", "")
        subfolder = img_info.get("subfolder", "")
        img_type  = img_info.get("type", "output")

        url = (
            f"{self._base_url}/view"
            f"?filename={filename}&subfolder={subfolder}&type={img_type}"
        )
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            arr = np.frombuffer(resp.content, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception:
            log.exception("Failed to fetch output image %s", filename)
            return None

    def _check_http_reachable(self, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = requests.get(f"{self._base_url}/system_stats", timeout=2)
                if r.status_code == 200:
                    log.info("ComfyUI reachable at %s", self._base_url)
                    return
            except requests.ConnectionError:
                time.sleep(0.5)
        raise ConnectionError(
            f"\nComfyUI not reachable at {self._base_url} after {timeout}s.\n"
            "\nStart ComfyUI first (in a separate terminal):\n"
            "  cd /path/to/ComfyUI\n"
            "  python main.py --listen\n"
            "\nThen re-run this script."
        )
