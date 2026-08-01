"""Reference-free image quality metrics used by the quality gate.

Each function takes a BGR ``numpy`` array (as returned by ``cv2.imread``) and
returns a single scalar metric. Functions are pure and have no side effects,
which keeps them independently unit-testable and reusable outside the agent.
"""

from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to single-channel grayscale.

    Args:
        image: BGR image array of shape (H, W, 3).

    Returns:
        Grayscale image array of shape (H, W).

    Raises:
        ValueError: If the image is empty or has an unsupported shape.
    """

    if image is None or image.size == 0:
        raise ValueError("Cannot convert an empty image to grayscale.")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape for grayscale conversion: {image.shape}")


def compute_blur_score(image: np.ndarray) -> float:
    """Compute a blur score using the variance of the Laplacian.

    Lower values indicate a blurrier image; higher values indicate sharper
    edges and finer detail.

    Args:
        image: BGR image array.

    Returns:
        Variance of the Laplacian of the grayscale image.
    """

    gray = to_grayscale(image)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness(image: np.ndarray) -> float:
    """Compute mean brightness on a 0-255 scale.

    Args:
        image: BGR image array.

    Returns:
        Mean pixel intensity of the grayscale image.
    """

    gray = to_grayscale(image)
    return float(np.mean(gray))


def compute_contrast(image: np.ndarray) -> float:
    """Compute contrast as the standard deviation of pixel intensities.

    Args:
        image: BGR image array.

    Returns:
        Standard deviation of the grayscale image's pixel intensities.
    """

    gray = to_grayscale(image)
    return float(np.std(gray))
