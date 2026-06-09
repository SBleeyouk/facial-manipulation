# facial-manipulation

Real-time facial expression retargeting using ComfyUI + LivePortrait.

```
Webcam → MediaPipe pose → ComfyUI LivePortrait → display / TouchDesigner
```

---

## Prerequisites

- Python 3.10+
- GPU recommended (CPU works but is slow)
- Git with LFS (for model weights submodule)

---

## 1. Clone (with submodules)

```bash
git clone --recurse-submodules https://github.com/SBleeyouk/facial-manipulation.git
cd facial-manipulation
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

This pulls both `ComfyUI/` and `temp_pretrained_weights/` (LivePortrait model weights from HuggingFace).

---

## 2. Set up ComfyUI

```bash
cd ComfyUI
python -m venv comfyvenv
source comfyvenv/bin/activate        # Windows: comfyvenv\Scripts\activate
pip install -r requirements.txt
```

### Install custom nodes (via ComfyUI Manager or manually)

| Node | Purpose |
|------|---------|
| `ComfyUI-AdvancedLivePortrait` | Per-blendshape expression control |
| `comfyui-liveportraitkj` | KJNodes LivePortrait implementation |
| `ComfyUI-VideoHelperSuite` | Frame I/O utilities |

Install through the ComfyUI Manager UI, or clone each into `ComfyUI/custom_nodes/`.

### Link model weights

The weights are already present in `temp_pretrained_weights/` (HuggingFace submodule).
Symlink or copy them into ComfyUI's model directory:

```bash
mkdir -p ComfyUI/models/liveportrait
cp temp_pretrained_weights/*.safetensors ComfyUI/models/liveportrait/
cp temp_pretrained_weights/*.onnx        ComfyUI/models/liveportrait/   2>/dev/null || true
```

Expected files:

```
ComfyUI/models/liveportrait/
├── appearance_feature_extractor.safetensors
├── motion_extractor.safetensors
├── spade_generator.safetensors
├── warping_module.safetensors
└── landmark.onnx   (optional, for stitching)
```

---

## 3. Start ComfyUI

```bash
cd ComfyUI
source comfyvenv/bin/activate
python main.py --listen
```

ComfyUI will be available at `http://localhost:8188`.  
Leave this terminal running.

---

## 4. Run the LivePortrait client

```bash
cd comfyui-liveportrait
python -m venv myvenv
source myvenv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env if ComfyUI runs on a different host/port

# Validate that your installed nodes match the expected input names:
python client/main.py --validate

# Run with an expression:
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

## 5. Run the composite pipeline

`composite_pipeline.py` runs two parallel paths — LivePortrait (face) + FluxRT (background) — and composites them into a video.

```bash
# from the repo root, with ComfyUI already running
python composite_pipeline.py

# options:
python composite_pipeline.py --frames 60 --prompt "dramatic cinematic lighting" --output out.mp4
```

| Flag | Default | Description |
|------|---------|-------------|
| `--frames` | `30` | Number of frames to capture |
| `--prompt` | `"a cinematic portrait scene..."` | FluxRT scene description |
| `--output` | `composite_output.mp4` | Output video path |

> **Note:** FluxRT is an optional second path. If `FluxRT/` is not present the pipeline still runs LivePortrait-only.

---

## File map

```
facial-manipulation/
├── ComfyUI/                        # ComfyUI server (submodule)
├── temp_pretrained_weights/        # LivePortrait weights (HuggingFace submodule)
├── comfyui-liveportrait/           # Real-time client pipeline
│   ├── client/
│   │   ├── main.py                 # CLI entry point
│   │   ├── capture.py              # Webcam + baseline selection
│   │   ├── landmarks.py            # MediaPipe → pose (pitch/yaw/roll, EAR)
│   │   ├── expression.py           # Expression preset → coefficient dict
│   │   ├── comfyui_client.py       # WebSocket + REST client
│   │   ├── workflow_builder.py     # Workflow JSON builder (TD-importable)
│   │   └── display.py              # Side-by-side OpenCV window
│   ├── workflows/
│   │   └── liveportrait_base.json  # ComfyUI API-format workflow template
│   ├── .env.example
│   └── requirements.txt
├── comfyui-liveportrait-custom_nodes-backup/
├── composite_pipeline.py           # LivePortrait + FluxRT composite pipeline
└── README.md
```
