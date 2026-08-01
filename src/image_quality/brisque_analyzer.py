"""BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) scoring.

Uses ``pyiqa`` (IQA-PyTorch) as the backing implementation. See the project
README / task discussion for the dependency comparison that led to this
choice over ``image-quality``, the OpenCV quality module, and ``piq``.

Empirically verified input contract for ``pyiqa``'s BRISQUE metric:
    - A file path (``str``) is accepted directly.
    - A raw BGR ``numpy`` array (as returned by ``cv2.imread``) is NOT
      accepted and raises ``ValueError: Unsupported source type``.
    - A ``torch`` tensor shaped (1, 3, H, W), dtype float32, values in
      [0, 1], channel order RGB, is accepted and produces identical scores
      to the file-path input.
This module therefore scores from the file path when available, and falls
back to explicit tensor construction for in-memory arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from .utils import get_logger

logger = get_logger(__name__)


class BrisqueAnalyzer:
    """Lazily-loaded, reusable BRISQUE scorer.

    Model loading is expensive (a few seconds), so a single analyzer
    instance should be created once and reused across all images in a run.
    """

    def __init__(self, device: str = "cpu") -> None:
        self._device = device
        self._metric = None

    def _ensure_loaded(self) -> None:
        if self._metric is not None:
            return
        try:
            import pyiqa
        except ImportError as exc:
            raise ImportError(
                "pyiqa is required for BRISQUE scoring but is not installed. "
                "Install it with: pip install pyiqa torch torchvision"
            ) from exc

        try:
            self._metric = pyiqa.create_metric("brisque", device=self._device)
            logger.debug("Loaded pyiqa BRISQUE metric on device=%s", self._device)
        except Exception as exc:
            raise RuntimeError("Failed to load the pyiqa BRISQUE metric.") from exc

    def score_path(self, image_path: Path) -> float:
        """Compute the BRISQUE score directly from an image file path.

        Args:
            image_path: Path to a readable image file.

        Returns:
            BRISQUE score (lower is better quality).

        Raises:
            FileNotFoundError: If the path does not exist.
            RuntimeError: If scoring fails.
        """

        if not Path(image_path).is_file():
            raise FileNotFoundError(f"Image not found for BRISQUE scoring: {image_path}")

        self._ensure_loaded()
        try:
            score = self._metric(str(image_path))
            return float(score)
        except Exception as exc:
            raise RuntimeError(f"BRISQUE scoring failed for {image_path}.") from exc

    def score_array(self, image_bgr: np.ndarray) -> float:
        """Compute the BRISQUE score from an in-memory BGR array.

        Args:
            image_bgr: BGR image array as returned by ``cv2.imread``.

        Returns:
            BRISQUE score (lower is better quality).

        Raises:
            ValueError: If the image is empty.
            RuntimeError: If scoring fails.
        """

        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Cannot score an empty image array.")

        self._ensure_loaded()
        try:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            tensor = tensor.to(self._device)
            score = self._metric(tensor)
            return float(score)
        except Exception as exc:
            raise RuntimeError("BRISQUE scoring failed for in-memory image array.") from exc


_default_analyzer: Optional[BrisqueAnalyzer] = None


def get_default_analyzer(device: str = "cpu") -> BrisqueAnalyzer:
    """Return a process-wide shared ``BrisqueAnalyzer`` instance."""

    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = BrisqueAnalyzer(device=device)
    return _default_analyzer
