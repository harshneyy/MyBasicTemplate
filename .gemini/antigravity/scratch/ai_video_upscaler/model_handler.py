"""
model_handler.py — AI Super-Resolution model loading and inference.

Strategy:
  1. Primary: Real-ESRGAN via the `realesrgan` pip package.
     - Weights are auto-downloaded on first use (~65 MB for x4plus).
     - Tile-based inference (tile=256) keeps peak RAM low on CPU.
  2. Fallback: OpenCV DNN Super-Resolution (EDSR_x4 / LapSRN_x4).
     - Zero extra dependencies — only needs opencv-python.
     - Requires manually downloading a pre-trained .pb model file.
     - Activated automatically if `realesrgan` is not installed.

Usage:
    handler = ModelHandler(scale=4)
    handler.load_model()
    enhanced_pil = handler.enhance_frame(original_pil)
"""

import logging
import os
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Where weights are cached (relative to this file — in the weights/ folder)
WEIGHTS_DIR = Path(__file__).parent / "weights"

# Real-ESRGAN pretrained model names mapped to their HuggingFace download URLs
_REALESRGAN_MODELS = {
    4: {
        "name": "RealESRGAN_x4plus",
        "url": (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/"
            "v0.1.0/RealESRGAN_x4plus.pth"
        ),
    },
    2: {
        "name": "RealESRGAN_x2plus",
        "url": (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/"
            "v0.2.1/RealESRGAN_x2plus.pth"
        ),
    },
}

# OpenCV DNN-SR fallback model URLs (LapSRN is small and CPU-fast)
_OPENCV_MODELS = {
    4: {
        "name": "LapSRN_x4",
        "algo": "LapSRN",
        "url": (
            "https://raw.githubusercontent.com/fannymonori/"
            "TF-LapSRN/master/export/LapSRN_x4.pb"
        ),
    },
    2: {
        "name": "LapSRN_x2",
        "algo": "LapSRN",
        "url": (
            "https://raw.githubusercontent.com/fannymonori/"
            "TF-LapSRN/master/export/LapSRN_x2.pb"
        ),
    },
}


# ---------------------------------------------------------------------------
# ModelHandler
# ---------------------------------------------------------------------------

class ModelHandler:
    """Loads a super-resolution model and runs frame-level inference.

    Automatically selects Real-ESRGAN if available, otherwise falls back
    to OpenCV's built-in DNN super-resolution module.

    Args:
        scale:      Upscaling factor — 2 or 4 (default 4).
        device:     PyTorch device string ('cpu' or 'cuda'). Only used
                    for the Real-ESRGAN backend.
        tile_size:  Tile width/height for Real-ESRGAN tiled inference.
                    Smaller tiles use less RAM. Set 0 to disable tiling.
        tile_pad:   Padding overlap between tiles (avoids seam artefacts).
    """

    def __init__(
        self,
        scale: int = 4,
        device: str = "cpu",
        tile_size: int = 256,
        tile_pad: int = 10,
    ) -> None:
        if scale not in (2, 4):
            raise ValueError(f"Unsupported scale factor: {scale}. Choose 2 or 4.")

        self.scale = scale
        self.device = device
        self.tile_size = tile_size
        self.tile_pad = tile_pad

        self._model = None          # the loaded model object
        self._backend: str = ""     # "realesrgan" | "opencv"

        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Download weights (if needed) and initialise the SR model.

        Tries Real-ESRGAN first. Falls back to OpenCV DNN-SR if the
        `realesrgan` package is not installed.
        """
        if self._try_load_realesrgan():
            logger.info("✓ Real-ESRGAN model loaded (backend: realesrgan)")
            self._backend = "realesrgan"
        else:
            logger.warning(
                "realesrgan package not found — falling back to OpenCV DNN-SR."
            )
            self._load_opencv_sr()
            logger.info("✓ OpenCV DNN-SR model loaded (backend: opencv)")
            self._backend = "opencv"

    def enhance_frame(self, pil_img: Image.Image) -> Image.Image:
        """Run super-resolution inference on a single PIL RGB frame.

        Args:
            pil_img: Input low-resolution frame as a PIL RGB Image.

        Returns:
            Enhanced high-resolution frame as a PIL RGB Image.

        Raises:
            RuntimeError: If load_model() has not been called.
        """
        if self._model is None:
            raise RuntimeError("Call load_model() before enhance_frame().")

        if self._backend == "realesrgan":
            return self._enhance_realesrgan(pil_img)
        else:
            return self._enhance_opencv(pil_img)

    @property
    def backend(self) -> str:
        """Return the active backend name ('realesrgan' or 'opencv')."""
        return self._backend

    # ------------------------------------------------------------------
    # Real-ESRGAN backend
    # ------------------------------------------------------------------

    def _try_load_realesrgan(self) -> bool:
        """Attempt to import and initialise Real-ESRGAN.

        Returns:
            True if successful, False if the package is not installed.
        """
        try:
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model_info = _REALESRGAN_MODELS[self.scale]
            weight_path = WEIGHTS_DIR / f"{model_info['name']}.pth"

            # Download weights if not already cached
            if not weight_path.exists():
                logger.info(
                    f"Downloading {model_info['name']} weights (~65 MB)…"
                )
                self._download(model_info["url"], weight_path)

            # Build the underlying RRDB network (same architecture for x2/x4)
            num_feat = 64
            num_block = 23
            net = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=num_feat,
                num_block=num_block,
                num_grow_ch=32,
                scale=self.scale,
            )

            self._model = RealESRGANer(
                scale=self.scale,
                model_path=str(weight_path),
                model=net,
                tile=self.tile_size,
                tile_pad=self.tile_pad,
                pre_pad=0,
                half=False,     # half-precision only helps on GPU; keep False for CPU
                device=torch.device(self.device),
            )
            return True

        except ImportError:
            return False

    def _enhance_realesrgan(self, pil_img: Image.Image) -> Image.Image:
        """Run Real-ESRGAN inference on a PIL image.

        The RealESRGANer.enhance() method expects a BGR numpy array.
        """
        import cv2

        # PIL RGB → BGR
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # enhance() returns (output_bgr, _) where output_bgr is uint8
        output_bgr, _ = self._model.enhance(bgr, outscale=self.scale)

        # BGR → RGB → PIL
        rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    # ------------------------------------------------------------------
    # OpenCV DNN-SR fallback backend
    # ------------------------------------------------------------------

    def _load_opencv_sr(self) -> None:
        """Initialise OpenCV's DNN super-resolution module (LapSRN)."""
        try:
            import cv2
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
        except AttributeError:
            raise RuntimeError(
                "cv2.dnn_superres not available. "
                "Install opencv-contrib-python: pip install opencv-contrib-python"
            )

        model_info = _OPENCV_MODELS[self.scale]
        weight_path = WEIGHTS_DIR / f"{model_info['name']}.pb"

        if not weight_path.exists():
            logger.info(f"Downloading {model_info['name']} weights…")
            self._download(model_info["url"], weight_path)

        sr.readModel(str(weight_path))
        sr.setModel(model_info["algo"].lower(), self.scale)
        # Use OpenCL acceleration if available (transparent to caller)
        sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self._model = sr

    def _enhance_opencv(self, pil_img: Image.Image) -> Image.Image:
        """Run OpenCV DNN-SR inference on a PIL image."""
        import cv2

        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        output_bgr = self._model.upsample(bgr)
        rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _download(url: str, dest: Path) -> None:
        """Download *url* to *dest* with a simple progress indicator."""

        def _reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 / total_size)
                mb = downloaded / 1_048_576
                total_mb = total_size / 1_048_576
                print(
                    f"\r  {pct:5.1f}%  {mb:.1f}/{total_mb:.1f} MB",
                    end="",
                    flush=True,
                )

        try:
            urllib.request.urlretrieve(url, dest, _reporthook)
            print()  # newline after progress
            logger.info(f"Saved to {dest}")
        except Exception as e:
            # Remove partial download
            if dest.exists():
                dest.unlink()
            raise RuntimeError(f"Failed to download {url}: {e}") from e
