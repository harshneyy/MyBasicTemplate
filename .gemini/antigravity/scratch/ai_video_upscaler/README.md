# 🎬 AI Video Upscaler

A Python-based AI video upscaling application that enhances low-resolution videos (360p/480p) to higher resolutions (720p/1080p) using **Real-ESRGAN** — running entirely on CPU.

---

## Project Structure

```
ai_video_upscaler/
├── main.py                  # ← CLI entry point
├── model_handler.py         # ← AI model loading & inference
├── video_processor.py       # ← Frame extraction & reconstruction
├── utils.py                 # ← Helper functions
├── generate_test_video.py   # ← Create a synthetic test video
├── requirements.txt
├── weights/                 # ← Auto-downloaded model weights
└── tests/
    └── test_upscaler.py     # ← Unit + integration tests
```

---

## Installation

### 1. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2. Install PyTorch (CPU-only — smaller download)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install the remaining dependencies
```bash
pip install -r requirements.txt
```

> **Note**: On first run, model weights (~65 MB for Real-ESRGAN x4plus) are
> downloaded automatically to the `weights/` folder.

---

## Usage

### Basic upscale (4×, default)
```bash
python main.py --input my_video.mp4
# Output: my_video_enhanced.mp4
```

### Custom scale and output path
```bash
python main.py --input clip.mp4 --output clip_4k.mp4 --scale 4
```

### Quick test — process only the first 10 frames
```bash
python main.py --input clip.mp4 --max-frames 10
```

### Generate a side-by-side comparison video
```bash
python main.py --input clip.mp4 --compare
# Produces: clip_enhanced.mp4 + clip_enhanced_comparison.mp4
```

### All options
```
python main.py --help

  --input   VIDEO     Input video path (required)
  --output  VIDEO     Output video path (default: <input>_enhanced.mp4)
  --scale   {2,4}     Upscale factor (default: 4)
  --max-frames N      Process only first N frames
  --compare           Also produce a side-by-side comparison video
  --tile-size PX      Tile size for tiled inference (default: 256, 0=disabled)
  --device  {cpu,cuda} PyTorch device (default: cpu)
  --verbose           Enable DEBUG logging
```

---

## Testing

### Generate a synthetic test video
```bash
python generate_test_video.py --output tests/test_input.mp4 --frames 30
```

### Run the test suite
```bash
python -m pytest tests/test_upscaler.py -v
```
Tests use a **mock model** (bicubic resize) — no real weights needed.

### End-to-end smoke test (real model)
```bash
python main.py --input tests/test_input.mp4 --output tests/test_output.mp4 \
               --scale 4 --max-frames 5
```

---

## How It Works

```
Input Video
    │
    ▼
extract_frames()          ← OpenCV reads one frame at a time (low RAM)
    │
    ▼  (for each frame)
model.enhance_frame()     ← Real-ESRGAN tiled inference (tile=256px)
    │
    ▼
reconstruct_video()       ← cv2.VideoWriter streams to disk
    │
    ▼
Enhanced Video  ✅
```

### Memory optimisation
- Frames are processed **one at a time** — no full-video buffering.
- **Tiled inference** (default tile=256px) caps per-frame GPU/RAM usage.
  - Reduce `--tile-size` further if you hit OOM on very large frames.

### Model fallback
| Priority | Backend | Package |
|----------|---------|---------|
| 1st | Real-ESRGAN x4plus/x2plus | `realesrgan` |
| 2nd | OpenCV LapSRN (DNN-SR) | `opencv-contrib-python` |

---

## Requirements

| Package | Purpose |
|---------|---------|
| `opencv-python` | Video I/O, frame processing |
| `torch` / `torchvision` | Model inference (CPU) |
| `realesrgan` / `basicsr` | Real-ESRGAN model |
| `Pillow` | Image format conversion |
| `tqdm` | Progress bar |
| `numpy` | Array operations |

---

## Tips

- **CPU speed**: Expect ~5–30 seconds per frame on CPU (depends on resolution & tile size).
- **Smaller tile = less RAM, slightly slower** — tune `--tile-size` to your machine.
- **Short test first**: Always run `--max-frames 5` before processing a full video.
- **FFmpeg codec**: If your player can't open the output, re-mux it:
  ```bash
  ffmpeg -i clip_enhanced.mp4 -c:v libx264 -crf 18 clip_enhanced_h264.mp4
  ```
