"""
tests/test_upscaler.py — Unit and integration tests for ai_video_upscaler.

Runs without GPU or real model weights by using a lightweight mock SR model.
Includes a synthetic video generator so no real video file is required.

Run:
    cd /home/harshney/.gemini/antigravity/scratch/ai_video_upscaler
    python -m pytest tests/test_upscaler.py -v
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from PIL import Image

# Ensure the parent package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    bgr_to_pil,
    create_comparison_frame,
    ensure_output_path,
    get_video_info,
    pil_to_bgr,
)
from video_processor import extract_frames, process_video, reconstruct_video


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic_video(path: str, width=320, height=180, fps=24, n_frames=10) -> None:
    """Write a coloured gradient synthetic video for testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(n_frames):
        # Gradient that changes each frame so we can verify ordering
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = int(i / n_frames * 255)  # Blue channel
        frame[:, :, 1] = np.linspace(0, 255, width, dtype=np.uint8)  # Green gradient
        frame[:, :, 2] = np.linspace(255, 0, height, dtype=np.uint8).reshape(-1, 1)
        writer.write(frame)
    writer.release()


class MockModelHandler:
    """A ModelHandler substitute that simply bicubic-upscales frames.

    No weights, no network — purely for testing the pipeline.
    """

    def __init__(self, scale: int = 4):
        self.scale = scale
        self.backend = "mock"
        self._model = True  # truthy so enhance_frame() won't raise

    def load_model(self):
        pass  # nothing to do

    def enhance_frame(self, pil_img: Image.Image) -> Image.Image:
        """Return a bicubic-upscaled version of the input image."""
        w, h = pil_img.size
        return pil_img.resize((w * self.scale, h * self.scale), Image.BICUBIC)


# ---------------------------------------------------------------------------
# utils.py tests
# ---------------------------------------------------------------------------

class TestUtils:
    def test_bgr_to_pil_shape(self):
        """BGR→PIL conversion should transpose channels correctly."""
        bgr = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        pil = bgr_to_pil(bgr)
        assert pil.mode == "RGB"
        assert pil.size == (200, 100)  # PIL size is (width, height)

    def test_pil_to_bgr_shape(self):
        """PIL→BGR round-trip should restore original array dimensions."""
        rgb = Image.fromarray(
            np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8), mode="RGB"
        )
        bgr = pil_to_bgr(rgb)
        assert bgr.shape == (100, 200, 3)

    def test_roundtrip(self):
        """bgr→pil→bgr should be a lossless round-trip."""
        original = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        reconstructed = pil_to_bgr(bgr_to_pil(original))
        np.testing.assert_array_equal(original, reconstructed)

    def test_get_video_info(self, tmp_path):
        """get_video_info should return correct width, height, fps."""
        video_path = str(tmp_path / "src.mp4")
        make_synthetic_video(video_path, width=320, height=180, fps=24, n_frames=3)
        cap = cv2.VideoCapture(video_path)
        info = get_video_info(cap)
        cap.release()
        assert info["width"] == 320
        assert info["height"] == 180
        assert abs(info["fps"] - 24.0) < 1.0  # allow ±1 fps rounding
        assert info["total_frames"] == 3

    def test_ensure_output_path(self):
        """ensure_output_path should append '_enhanced' before the extension."""
        assert ensure_output_path("videos/clip.mp4") == "videos/clip_enhanced.mp4"
        assert ensure_output_path("clip.avi") == "clip_enhanced.avi"

    def test_create_comparison_frame(self):
        """Comparison frame should be twice as wide as enhanced + 2px divider."""
        orig = np.zeros((100, 200, 3), dtype=np.uint8)
        enhanced = np.zeros((400, 800, 3), dtype=np.uint8)
        cmp = create_comparison_frame(orig, enhanced)
        assert cmp.shape == (400, 800 * 2 + 2, 3)


# ---------------------------------------------------------------------------
# video_processor.py tests
# ---------------------------------------------------------------------------

class TestVideoProcessor:
    def test_extract_frames_count(self, tmp_path):
        """extract_frames should yield exactly N frames for an N-frame video."""
        video_path = str(tmp_path / "src.mp4")
        make_synthetic_video(video_path, n_frames=10)
        frames = list(extract_frames(video_path))
        assert len(frames) == 10

    def test_extract_frames_max_frames(self, tmp_path):
        """extract_frames with max_frames=5 should stop after 5 frames."""
        video_path = str(tmp_path / "src.mp4")
        make_synthetic_video(video_path, n_frames=10)
        frames = list(extract_frames(video_path, max_frames=5))
        assert len(frames) == 5

    def test_extract_frames_index(self, tmp_path):
        """Frame indices should be sequential starting at 0."""
        video_path = str(tmp_path / "src.mp4")
        make_synthetic_video(video_path, n_frames=5)
        indices = [idx for idx, _ in extract_frames(video_path)]
        assert indices == list(range(5))

    def test_extract_frames_missing_file(self):
        """extract_frames should raise FileNotFoundError for bad paths."""
        with pytest.raises(FileNotFoundError):
            list(extract_frames("/nonexistent/video.mp4"))

    def test_reconstruct_video_creates_file(self, tmp_path):
        """reconstruct_video should create a valid .mp4 file."""
        out = str(tmp_path / "out.mp4")
        writer = reconstruct_video(out, fps=24, width=320, height=180)
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        for _ in range(5):
            writer.write(frame)
        writer.release()
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0


# ---------------------------------------------------------------------------
# Integration test — full pipeline with mock model
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_process_video_resolution(self, tmp_path):
        """Output video should be scale× larger than the input video."""
        scale = 4
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")

        make_synthetic_video(input_path, width=80, height=60, fps=10, n_frames=3)

        model = MockModelHandler(scale=scale)
        result = process_video(
            input_path=input_path,
            output_path=output_path,
            model=model,
            scale=scale,
            max_frames=3,
        )

        assert result["frames_processed"] == 3
        assert result["input_resolution"] == (80, 60)
        assert result["output_resolution"] == (80 * scale, 60 * scale)
        assert Path(output_path).exists()

    def test_process_video_comparison(self, tmp_path):
        """With make_comparison=True, a comparison video should also exist."""
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        cmp_path = str(tmp_path / "cmp.mp4")

        make_synthetic_video(input_path, width=80, height=60, fps=10, n_frames=3)
        model = MockModelHandler(scale=4)

        result = process_video(
            input_path=input_path,
            output_path=output_path,
            model=model,
            scale=4,
            max_frames=3,
            make_comparison=True,
            comparison_path=cmp_path,
        )

        assert Path(cmp_path).exists()
        assert result["comparison_path"] == cmp_path
