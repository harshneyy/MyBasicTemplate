"""
video_processor.py — Frame extraction and video reconstruction pipeline.

Provides:
  - extract_frames()    : Generator yielding (index, BGR frame) from a video file.
  - reconstruct_video() : Writes a sequence of BGR frames to an MP4 file.
  - process_video()     : End-to-end pipeline: extract → enhance → reconstruct.
"""

import logging
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from model_handler import ModelHandler
from utils import bgr_to_pil, create_comparison_frame, get_video_info, pil_to_bgr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(
    video_path: str,
    max_frames: Optional[int] = None,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """Yield (frame_index, BGR_frame) tuples from a video file.

    Uses OpenCV's VideoCapture for portable, dependency-free decoding.
    Frames are returned one at a time — no full-video buffering.

    Args:
        video_path:  Path to the source video file.
        max_frames:  If set, stop after yielding this many frames.
                     Useful for quick smoke-tests.

    Yields:
        (int, np.ndarray): Zero-based frame index and BGR frame array.

    Raises:
        FileNotFoundError: If *video_path* does not exist.
        RuntimeError:      If OpenCV cannot open the file.
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video or read error

            yield frame_idx, frame
            frame_idx += 1

            if max_frames is not None and frame_idx >= max_frames:
                logger.debug(f"Reached max_frames limit ({max_frames}).")
                break
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Video reconstruction
# ---------------------------------------------------------------------------

def reconstruct_video(
    output_path: str,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    """Create and return an open cv2.VideoWriter for *output_path*.

    Caller is responsible for writing frames and releasing the writer.

    Args:
        output_path: Destination path for the output video file.
        fps:         Frames per second (should match source video).
        width:       Frame width in pixels.
        height:      Frame height in pixels.

    Returns:
        An initialised cv2.VideoWriter (not yet written to).

    Raises:
        RuntimeError: If the writer cannot be initialised.
    """
    # mp4v is broadly compatible; use avc1 for smaller files if available
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create VideoWriter for '{output_path}'. "
            "Ensure the output directory exists and the codec is available."
        )

    logger.debug(
        f"VideoWriter opened: {output_path} @ {fps:.2f} fps, {width}×{height}"
    )
    return writer


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def process_video(
    input_path: str,
    output_path: str,
    model: ModelHandler,
    scale: int = 4,
    max_frames: Optional[int] = None,
    make_comparison: bool = False,
    comparison_path: Optional[str] = None,
) -> dict:
    """Full upscaling pipeline: extract frames → enhance → reconstruct.

    Processes one frame at a time to keep memory usage constant regardless
    of video length.

    Args:
        input_path:       Path to the source (low-resolution) video.
        output_path:      Destination path for the enhanced video.
        model:            A loaded ModelHandler instance.
        scale:            Upscaling factor (used to compute output resolution).
        max_frames:       Limit processing to the first N frames (optional).
        make_comparison:  If True, also write a side-by-side comparison video.
        comparison_path:  Path for the comparison video. Auto-derived if None.

    Returns:
        dict with keys:
            - frames_processed (int)
            - input_resolution  (tuple[int, int]) — (width, height)
            - output_resolution (tuple[int, int]) — (width, height)
            - output_path       (str)
            - comparison_path   (str | None)

    Raises:
        FileNotFoundError: If the input video does not exist.
        RuntimeError:      On codec or model failures.
    """
    # ── Read source metadata ──────────────────────────────────────────
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")
    info = get_video_info(cap)
    cap.release()

    src_w, src_h = info["width"], info["height"]
    fps = info["fps"]
    total = info["total_frames"]

    # If max_frames is set, cap the tqdm total displayed
    display_total = min(total, max_frames) if max_frames else total

    out_w = src_w * scale
    out_h = src_h * scale

    logger.info(
        f"Input : {input_path}  ({src_w}×{src_h}, {fps:.2f} fps, "
        f"{total} frames)"
    )
    logger.info(
        f"Output: {output_path}  (target {out_w}×{out_h}, scale ×{scale})"
    )

    # ── Open output writers ───────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    enhanced_writer = reconstruct_video(output_path, fps, out_w, out_h)

    cmp_path: Optional[str] = None
    cmp_writer: Optional[cv2.VideoWriter] = None

    if make_comparison:
        cmp_path = comparison_path or _derive_comparison_path(output_path)
        Path(cmp_path).parent.mkdir(parents=True, exist_ok=True)
        # Comparison frame is twice as wide (side by side) + 2px divider
        cmp_writer = reconstruct_video(cmp_path, fps, out_w * 2 + 2, out_h)
        logger.info(f"Comparison video: {cmp_path}")

    # ── Process frame by frame ────────────────────────────────────────
    frames_processed = 0
    actual_out_w, actual_out_h = out_w, out_h  # will be updated from first frame

    try:
        pbar = tqdm(
            total=display_total,
            desc="Upscaling",
            unit="frame",
            dynamic_ncols=True,
            colour="cyan",
        )

        for idx, bgr_frame in extract_frames(input_path, max_frames):
            # Convert to PIL → run SR model → convert back to BGR
            pil_frame = bgr_to_pil(bgr_frame)
            enhanced_pil = model.enhance_frame(pil_frame)
            enhanced_bgr = pil_to_bgr(enhanced_pil)

            # Update actual output dimensions from first real frame
            if frames_processed == 0:
                actual_out_h, actual_out_w = enhanced_bgr.shape[:2]
                if (actual_out_w, actual_out_h) != (out_w, out_h):
                    logger.warning(
                        f"Actual output size {actual_out_w}×{actual_out_h} "
                        f"differs from expected {out_w}×{out_h}."
                    )

            # Write enhanced frame
            enhanced_writer.write(enhanced_bgr)

            # Write comparison frame if requested
            if cmp_writer is not None:
                cmp_frame = create_comparison_frame(bgr_frame, enhanced_bgr)
                cmp_writer.write(cmp_frame)

            frames_processed += 1
            pbar.update(1)

        pbar.close()

    finally:
        enhanced_writer.release()
        if cmp_writer is not None:
            cmp_writer.release()

    logger.info(
        f"Done! Processed {frames_processed} frame(s). "
        f"Output saved to: {output_path}"
    )

    return {
        "frames_processed": frames_processed,
        "input_resolution": (src_w, src_h),
        "output_resolution": (actual_out_w, actual_out_h),
        "output_path": output_path,
        "comparison_path": cmp_path,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _derive_comparison_path(output_path: str) -> str:
    """Append '_comparison' to an output path stem."""
    p = Path(output_path)
    return str(p.with_stem(p.stem + "_comparison"))
