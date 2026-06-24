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

## 6. One-shot LivePortrait → LTX-2.3 video

A second, separate pipeline for when you don't want a real-time loop: capture the webcam **once**, generate **one** emotion-edited still image, then turn it into a single video that goes baseline (sleeping) face → generated emotion → back to the baseline face. It reuses the same LivePortrait expression nodes as the real-time pipeline; it does not touch or replace it.

### Download the LTX-2.3 weights

The video stage uses LTX-2.3 (Lightricks), wired in as a ComfyUI workflow blueprint (`ComfyUI/blueprints/First-Last-Frame to Video (LTX-2.3).json`). These files are **not included** — download them and place them as follows:

| File | Place in | Source |
|------|----------|--------|
| `ltx-2.3-22b-distilled-fp8.safetensors` | `ComfyUI/models/checkpoints/` | https://huggingface.co/Lightricks/LTX-2.3-fp8 |
| `gemma_3_12B_it_fp4_mixed.safetensors` | `ComfyUI/models/text_encoders/` | https://huggingface.co/Comfy-Org/ltx-2 (split_files/text_encoders) |

`oneshot_pipeline.py` checks for these on startup and warns (but doesn't block) if they're missing.

### Run it

```bash
cd comfyui-liveportrait
source myvenv/bin/activate    # same venv as the real-time client

python client/oneshot_pipeline.py --expr happy
```

This will:
1. Open the webcam once and capture a short burst of frames, picking the most frontal one as the baseline ("sleeping") face — no continuous capture, no live preview.
2. Upload that baseline image to ComfyUI.
3. Patch `workflows/liveportrait_to_ltx_oneshot.json` with the chosen `--expr` (same presets as the real-time pipeline: `happy/sad/angry/surprised/neutral`) and write it into `ComfyUI/user/default/workflows/` so it shows up in the ComfyUI UI's workflow list.

### Finish in the browser

Open `http://127.0.0.1:8188`, load the **liveportrait_to_ltx_oneshot** workflow, and click **Queue**.

This last step is manual by design: LTX-2.3 is wired in as a ComfyUI *subgraph*, and subgraph expansion happens in the browser frontend before the graph is sent to the server — there's no reliable way to submit it directly over the REST API. Everything before this step (webcam capture, upload, patching the workflow) is fully automated by `oneshot_pipeline.py`.

The graph does, in order: `LoadImage(baseline)` → `ExpressionEditor` + `AdvancedLivePortrait` (generates the emotion still, same as the real-time pipeline) → two LTX-2.3 "First-Last-Frame to Video" clips (baseline→emotion, then emotion→baseline) → `GetVideoComponents` + `ImageBatch` (concatenates the two clips' frames in time) → `CreateVideo` → `SaveVideo`. Output lands in `ComfyUI/output/liveportrait_ltx_oneshot/`.

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
│   │   ├── display.py              # Side-by-side OpenCV window
│   │   └── oneshot_pipeline.py     # One-shot capture → LivePortrait → LTX-2.3 video
│   ├── workflows/
│   │   ├── liveportrait_base.json  # ComfyUI API-format workflow template
│   │   └── liveportrait_to_ltx_oneshot.json  # One-shot UI workflow (LivePortrait + LTX-2.3)
│   ├── .env.example
│   └── requirements.txt
├── comfyui-liveportrait-custom_nodes-backup/
├── composite_pipeline.py           # LivePortrait + FluxRT composite pipeline
└── README.md
```
