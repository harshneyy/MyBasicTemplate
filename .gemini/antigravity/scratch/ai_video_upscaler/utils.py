"""
utils.py — Helper utilities for the AI Video Upscaler.

Provides:
  - Logging setup
  - OpenCV ↔ PIL image conversion
  - Video metadata extraction
  - Side-by-side comparison frame builder
"""

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure and return the root logger.

    Args:
        verbose: If True, set level to DEBUG; otherwise INFO.

    Returns:
        Configured Logger instance.
    """
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%H:%M:%S"

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Suppress noisy third-party loggers
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("basicsr").setLevel(logging.WARNING)
    logging.getLogger("realesrgan").setLevel(logging.WARNING)

    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image conversion helpers
# ---------------------------------------------------------------------------

def bgr_to_pil(frame: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR ndarray to a PIL RGB Image.

    OpenCV reads frames as BGR by default; PIL and the SR model expect RGB.

    Args:
        frame: H×W×3 uint8 numpy array in BGR colour order.

    Returns:
        PIL Image in RGB mode.
    """
    # cv2.cvtColor is the fastest in-place conversion
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    """Convert a PIL RGB Image back to an OpenCV BGR ndarray.

    Args:
        img: PIL Image in RGB mode.

    Returns:
        H×W×3 uint8 numpy array in BGR colour order.
    """
    rgb_arr = np.array(img)
    return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Video metadata
# ---------------------------------------------------------------------------

def get_video_info(cap: cv2.VideoCapture) -> dict:
    """Extract basic metadata from an open cv2.VideoCapture object.

    Args:
        cap: An already-opened VideoCapture.

    Returns:
        Dictionary with keys: width, height, fps, total_frames.
    """
    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,   # default 25 if unknown
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }


# ---------------------------------------------------------------------------
# Comparison frame
# ---------------------------------------------------------------------------

def create_comparison_frame(
    original: np.ndarray,
    enhanced: np.ndarray,
) -> np.ndarray:
    """Build a side-by-side BGR frame: original (resized) | enhanced.

    The original frame is upscaled with bilinear interpolation so both
    panels share the same dimensions, making quality differences obvious.

    Args:
        original: Low-resolution BGR frame.
        enhanced: Upscaled BGR frame (output of the SR model).

    Returns:
        Side-by-side BGR frame with a 2-pixel divider line.

    Raises:
        ValueError: If the enhanced frame has zero dimensions.
    """
    h, w = enhanced.shape[:2]

    if h == 0 or w == 0:
        raise ValueError(f"Enhanced frame has invalid dimensions: {w}×{h}")

    # Resize original to match enhanced resolution (bilinear = fast)
    orig_resized = cv2.resize(original, (w, h), interpolation=cv2.INTER_LINEAR)

    # Draw labels on each panel
    _draw_label(orig_resized, "Original (bilinear)")
    _draw_label(enhanced, "AI Enhanced (Real-ESRGAN)")

    # Thin white divider between panels
    divider = np.ones((h, 2, 3), dtype=np.uint8) * 255

    return np.hstack([orig_resized, divider, enhanced])


def _draw_label(frame: np.ndarray, text: str) -> None:
    """Overlay a semi-transparent label on the top-left of *frame* in place."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, frame.shape[1] / 1280)   # scale text with resolution
    thickness = max(1, int(scale * 2))
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    # Dark background rectangle for readability
    pad = 8
    cv2.rectangle(
        frame,
        (0, 0),
        (tw + pad * 2, th + baseline + pad * 2),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(
        frame,
        text,
        (pad, th + pad),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def ensure_output_path(input_path: str, suffix: str = "_enhanced") -> str:
    """Derive a sensible output path from an input video path.

    Example:
        "videos/clip.mp4"  →  "videos/clip_enhanced.mp4"

    Args:
        input_path: Path to the source video file.
        suffix: String appended before the file extension.

    Returns:
        String path for the output file.
    """
    p = Path(input_path)
    return str(p.with_stem(p.stem + suffix))
