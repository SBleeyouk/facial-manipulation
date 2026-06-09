# ComfyUI LivePortrait Real-Time Retargeting

Real-time facial expression retargeting pipeline.
Webcam → MediaPipe pose → ComfyUI LivePortrait → display / TouchDesigner.

---

## Architecture

```
[Webcam]
   │
   ├─ MediaPipe FaceMesh ──► pose (pitch/yaw/roll, eye EAR)
   │
   ├─ expression.py ────────► coefficient dict (preset or OSC-driven)
   │
   └─ workflow_builder.py ──► mutated ComfyUI workflow JSON
                                        │
                               [ComfyUI Server :8188]
                                 LoadImage (baseline)
                                 LoadImage (driving frame)
                                 LivePortraitProcess
                                 LivePortraitComposite
                                 SaveImage
                                        │
                              WebSocket ──► output frame (numpy BGR)
                                        │
                            display.py  or  TouchDesigner
```

---

## Requirements

### Python (client machine)
```
Python 3.10+
pip install -r requirements.txt
```

### ComfyUI (can be the same machine or a networked GPU box)

1. Clone ComfyUI and install custom nodes via **ComfyUI Manager**:
   - `comfyui-liveportraitkj` — KJNodes LivePortrait implementation
   - `ComfyUI-AdvancedLivePortrait` — per-blendshape expression control
   - `ComfyUI-VideoHelperSuite` (VHS) — frame I/O utilities

2. Place LivePortrait model weights in `ComfyUI/models/liveportrait/`:
   ```
   appearance_feature_extractor.safetensors
   motion_extractor.safetensors
   spade_generator.safetensors
   warping_module.safetensors
   landmark.onnx  (optional, for stitching)
   ```
   Download from: https://huggingface.co/KwaiVGI/LivePortrait

3. Start ComfyUI in server mode:
   ```bash
   cd ComfyUI
   python main.py --listen
   ```

---

## Quick start

```bash
cp .env.example .env
# edit .env if ComfyUI runs on a different host/port

# Validate that your installed LivePortrait nodes have the expected input names:
python client/main.py --validate

# Run:
python client/main.py --expr sad
```

**Keyboard controls** (click the display window first):

| Key | Action |
|-----|--------|
| `s` | sad |
| `a` | angry |
| `u` | surprised |
| `h` | happy |
| `n` | neutral |
| `b` | re-capture baseline |
| `q` | quit |

---

## Node parameter validation

If `--validate` shows that your installed `LivePortraitProcess` node uses
**different input names** than the defaults in `client/workflow_builder.py`,
edit `EXPRESSION_PARAM_MAP` at the top of that file.

Example: if the node calls the field `"eye_widen_left"` instead of `"eye_wide_l"`:
```python
EXPRESSION_PARAM_MAP = {
    ...
    "eye_wide_l": "eye_widen_left",   # ← right side is the actual node input key
    ...
}
```

The same applies to any blendshape key the node does not expose — those entries
are silently skipped, which is safe.

---

## TouchDesigner integration

### Option 1 — Import Python modules directly

TouchDesigner's Python interpreter can import `workflow_builder` and
`comfyui_client` as standalone modules.  Add the `client/` folder to your
TD project's `sys.path`:

```python
# In a TD Script DAT (Execute / DAT Execute)
import sys
sys.path.insert(0, "/path/to/comfyui-liveportrait/client")

from workflow_builder import WorkflowBuilder
from comfyui_client   import ComfyUIClient

client  = ComfyUIClient(host="127.0.0.1", port=8188)
client.connect()
builder = WorkflowBuilder()
```

Then in your render/frame callback:
```python
workflow = builder.build(
    source_image  = "baseline.png",    # filename from a prior upload
    driving_image = "driving.png",     # upload each frame first
    expression    = {"mouth_smile_l": 0.8, "mouth_smile_r": 0.8},
    pose          = {"pitch": 0, "yaw": 0, "roll": 0, "retargeting_eyes": 0.6},
)
client.submit_workflow(workflow)   # fire-and-forget
frame = client.get_latest_frame() # np.ndarray BGR | None
```

### Option 2 — Web Client DAT (REST only, no output image)

1. Add a **Web Client DAT**, set Base URL to `http://127.0.0.1:8188`.
2. In a Script DAT build the workflow JSON (copy from `liveportrait_base.json`,
   substitute image filenames).
3. POST to `/prompt`:
   ```python
   import json, td
   payload = json.dumps({"prompt": workflow_dict, "client_id": "my-td-client"})
   op("WebClient1").request("/prompt", method="POST",
       headers={"Content-Type": "application/json"}, body=payload)
   ```

### Option 3 — WebSocket DAT

1. Add a **WebSocket DAT**, connect to `ws://127.0.0.1:8188/ws?clientId=td-client`.
2. Parse incoming JSON in a DAT Execute callback:
   ```python
   import json
   msg = json.loads(dat.text)
   if msg.get("type") == "executed":
       # msg["data"]["output"]["images"][0]["filename"] is the output filename
       # Fetch via http://127.0.0.1:8188/view?filename=...
       pass
   ```

### OSC expression switching (TouchDesigner → Python client)

From any TD CHOP or DAT send an OSC message to UDP port 9000:
```
/expression  angry
```

The Python client will switch expression on the next loop iteration.
No Python code changes needed.

---

## File map

```
comfyui-liveportrait/
├── workflows/
│   └── liveportrait_base.json    # ComfyUI API-format workflow template
├── client/
│   ├── main.py                   # CLI entry point
│   ├── capture.py                # OpenCV webcam + baseline selection
│   ├── landmarks.py              # MediaPipe FaceMesh → pose (pitch/yaw/roll, EAR)
│   ├── expression.py             # Expression preset → coefficient dict
│   ├── comfyui_client.py         # WebSocket + REST client
│   ├── workflow_builder.py       # Workflow JSON mutation (TD-importable)
│   └── display.py                # Side-by-side OpenCV window
├── .env.example
├── requirements.txt
└── README.md
```

---

## Design notes

- **Decoupled latency**: webcam capture and display run at webcam frame rate
  (~30 fps). ComfyUI submissions are rate-limited to 15/sec. The display
  shows the _most recently returned_ ComfyUI frame without waiting.
- **Baseline uploaded once**: at session start the best frontal frame is
  uploaded and its filename reused every loop. Re-uploading on every frame
  would saturate ComfyUI's input directory.
- **OSC thread-safety**: `expression_state` is a single-element list; OSC
  and keyboard handlers both write `expression_state[0]`; Python's GIL makes
  this safe for a single str assignment.
- **retargeting_lip is always 0**: the expression channel owns lip shape.
  MediaPipe eye EAR drives `retargeting_eyes` only.
