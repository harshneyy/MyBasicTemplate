"""
main.py — CLI entry point for the AI Video Upscaler.

Usage:
    python main.py --input INPUT_VIDEO [OPTIONS]

Examples:
    # Basic upscale (4× by default)
    python main.py --input my_clip.mp4

    # 2× upscale, only first 30 frames (quick test)
    python main.py --input my_clip.mp4 --scale 2 --max-frames 30

    # 4× upscale with side-by-side comparison video
    python main.py --input my_clip.mp4 --compare

    # Custom output path + verbose logging
    python main.py --input my_clip.mp4 --output enhanced.mp4 --verbose
"""

import argparse
import sys
import time
import logging
from pathlib import Path

from model_handler import ModelHandler
from utils import ensure_output_path, setup_logging
from video_processor import process_video


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai_upscaler",
        description=(
            "AI Video Upscaler — enhance low-resolution videos using "
            "Real-ESRGAN (or OpenCV DNN-SR fallback) on CPU."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Required ─────────────────────────────────────────────────────
    parser.add_argument(
        "--input", "-i",
        required=True,
        metavar="VIDEO",
        help="Path to the input (low-resolution) video file.",
    )

    # ── Optional ─────────────────────────────────────────────────────
    parser.add_argument(
        "--output", "-o",
        metavar="VIDEO",
        default=None,
        help=(
            "Path for the enhanced output video. "
            "Defaults to <input_name>_enhanced.mp4 in the same directory."
        ),
    )
    parser.add_argument(
        "--scale", "-s",
        type=int,
        choices=[2, 4],
        default=4,
        help="Super-resolution upscale factor: 2× or 4× (default: 4).",
    )
    parser.add_argument(
        "--max-frames", "-n",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N frames. Useful for quick tests.",
    )
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help=(
            "Also produce a side-by-side comparison video "
            "(<output_name>_comparison.mp4)."
        ),
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        metavar="PX",
        help=(
            "Tile size for Real-ESRGAN tiled inference (default: 256). "
            "Smaller values reduce RAM usage at the cost of slight speed loss. "
            "Set to 0 to disable tiling."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="PyTorch device for Real-ESRGAN (default: cpu).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Parse arguments, run the upscaling pipeline, and return an exit code."""
    parser = build_parser()
    args = parser.parse_args()

    # ── Logging ──────────────────────────────────────────────────────
    logger = setup_logging(args.verbose)

    # ── Input validation ─────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    if input_path.suffix.lower() not in {
        ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"
    }:
        logger.warning(
            f"Unrecognised file extension '{input_path.suffix}'. "
            "Attempting to open anyway…"
        )

    # ── Derive output path ────────────────────────────────────────────
    output_path = args.output or ensure_output_path(str(input_path))

    # ── Print summary banner ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("   🎬  AI Video Upscaler")
    print("=" * 60)
    print(f"   Input  : {input_path}")
    print(f"   Output : {output_path}")
    print(f"   Scale  : {args.scale}×")
    if args.max_frames:
        print(f"   Frames : first {args.max_frames} only")
    print(f"   Device : {args.device.upper()}")
    print(f"   Tiles  : {args.tile_size or 'disabled'}")
    print("=" * 60 + "\n")

    # ── Load AI model ─────────────────────────────────────────────────
    logger.info("Loading super-resolution model…")
    model = ModelHandler(
        scale=args.scale,
        device=args.device,
        tile_size=args.tile_size,
        tile_pad=10,
    )

    try:
        model.load_model()
    except RuntimeError as exc:
        logger.error(f"Model loading failed: {exc}")
        return 2

    logger.info(f"Active backend: {model.backend}")

    # ── Run the pipeline ──────────────────────────────────────────────
    t_start = time.perf_counter()

    try:
        result = process_video(
            input_path=str(input_path),
            output_path=output_path,
            model=model,
            scale=args.scale,
            max_frames=args.max_frames,
            make_comparison=args.compare,
        )
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1
    except RuntimeError as exc:
        logger.error(f"Processing failed: {exc}")
        return 3

    elapsed = time.perf_counter() - t_start

    # ── Final report ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("   ✅  Upscaling Complete!")
    print("=" * 60)
    in_w, in_h = result["input_resolution"]
    out_w, out_h = result["output_resolution"]
    n = result["frames_processed"]
    fps_throughput = n / elapsed if elapsed > 0 else 0

    print(f"   Frames processed : {n}")
    print(f"   Input resolution : {in_w}×{in_h}")
    print(f"   Output resolution: {out_w}×{out_h}")
    print(f"   Time elapsed     : {elapsed:.1f}s  ({fps_throughput:.2f} fps)")
    print(f"   Output saved to  : {result['output_path']}")
    if result.get("comparison_path"):
        print(f"   Comparison video : {result['comparison_path']}")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
