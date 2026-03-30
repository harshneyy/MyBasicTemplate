"""
generate_test_video.py — Convenience script to create a synthetic low-res test
video without needing a real video file.

Usage:
    python generate_test_video.py [--output PATH] [--frames N] [--width W] [--height H]
"""

import argparse
import numpy as np
import cv2


def generate(output: str, width: int, height: int, fps: float, n_frames: int) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (width, height))

    print(f"Generating synthetic {width}×{height} test video ({n_frames} frames)…")

    for i in range(n_frames):
        # Animated gradient pattern — each frame is visually distinct
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Horizontal blue gradient shifting over time
        offset = int(i / n_frames * 255)
        frame[:, :, 0] = np.roll(
            np.linspace(0, 255, width, dtype=np.uint8), offset
        )

        # Vertical green gradient
        frame[:, :, 1] = np.linspace(0, 200, height, dtype=np.uint8).reshape(-1, 1)

        # Diagonal red pattern
        for y in range(height):
            for x in range(0, width, 20):
                if (x + y + i * 4) % 40 < 20:
                    frame[y, x : x + 20, 2] = 180

        writer.write(frame)

    writer.release()
    print(f"✅ Saved to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic test video.")
    parser.add_argument("--output", default="tests/test_input.mp4")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--fps", type=float, default=24.0)
    args = parser.parse_args()

    generate(args.output, args.width, args.height, args.fps, args.frames)
