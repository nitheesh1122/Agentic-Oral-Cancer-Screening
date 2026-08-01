"""Image preprocessing pipeline applied to images that pass the quality gate.

Pipeline order: resize -> Gaussian noise reduction -> CLAHE contrast
enhancement -> normalize. Each step is a small, independently testable
function; ``quality_agent`` composes them.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import QualityConfig


def resize_image(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize an image to the target (width, height).

    Args:
        image: BGR image array.
        target_size: Desired (width, height) in pixels.

    Returns:
        Resized BGR image array.

    Raises:
        ValueError: If the image is empty.
    """

    if image is None or image.size == 0:
        raise ValueError("Cannot resize an empty image.")
    width, height = target_size
    interpolation = cv2.INTER_AREA if image.shape[0] > height else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def reduce_noise(image: np.ndarray, kernel_size: tuple[int, int]) -> np.ndarray:
    """Apply Gaussian blur for noise reduction.

    Args:
        image: BGR image array.
        kernel_size: Odd-valued (width, height) Gaussian kernel size.

    Returns:
        Denoised BGR image array.
    """

    if image is None or image.size == 0:
        raise ValueError("Cannot denoise an empty image.")
    return cv2.GaussianBlur(image, kernel_size, sigmaX=0)


def apply_clahe(image: np.ndarray, clip_limit: float, tile_grid_size: tuple[int, int]) -> np.ndarray:
    """Apply CLAHE contrast enhancement on the luminance channel only.

    Working in LAB color space and enhancing only the L channel avoids
    amplifying color-channel noise, which plain per-channel CLAHE in BGR
    would introduce.

    Args:
        image: BGR image array.
        clip_limit: CLAHE contrast-limiting threshold.
        tile_grid_size: CLAHE tile grid size.

    Returns:
        Contrast-enhanced BGR image array.
    """

    if image is None or image.size == 0:
        raise ValueError("Cannot enhance an empty image.")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Scale pixel values from [0, 255] uint8 to [0, 1] float32.

    Args:
        image: BGR uint8 image array.

    Returns:
        BGR float32 image array with values in [0, 1].
    """

    if image is None or image.size == 0:
        raise ValueError("Cannot normalize an empty image.")
    return image.astype(np.float32) / 255.0


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """Convert a [0, 1] float32 image back to a [0, 255] uint8 image.

    Standard image codecs (JPEG/PNG) require uint8 data, so persisted
    files are round-tripped back from the normalized array rather than
    written directly in float form.

    Args:
        image: BGR float32 image array with values in [0, 1].

    Returns:
        BGR uint8 image array.
    """

    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def run_preprocessing_pipeline(image: np.ndarray, config: QualityConfig) -> tuple[np.ndarray, np.ndarray]:
    """Run the full accepted-image preprocessing pipeline.

    Args:
        image: Original BGR image array (as read from disk).
        config: Quality configuration providing sizes and thresholds.

    Returns:
        A tuple of (normalized_float_image, persistable_uint8_image), where
        the latter is the round-tripped, disk-writable version of the former.
    """

    resized = resize_image(image, config.target_size)
    denoised = reduce_noise(resized, config.gaussian_kernel_size)
    enhanced = apply_clahe(denoised, config.clahe_clip_limit, config.clahe_tile_grid_size)
    normalized = normalize_image(enhanced)
    persistable = denormalize_image(normalized)
    return normalized, persistable


def build_comparison_image(original: np.ndarray, processed: np.ndarray) -> np.ndarray:
    """Build a labeled side-by-side "Original | Processed" comparison image.

    Args:
        original: Original BGR image array.
        processed: Processed (uint8) BGR image array.

    Returns:
        A single BGR image array with both frames placed side by side,
        each resized to a common display height and labeled.
    """

    if original is None or original.size == 0 or processed is None or processed.size == 0:
        raise ValueError("Cannot build a comparison image from empty inputs.")

    display_height = 480
    label_band_height = 30

    def _prepare_panel(img: np.ndarray, label: str) -> np.ndarray:
        aspect_ratio = img.shape[1] / img.shape[0]
        display_width = max(1, int(display_height * aspect_ratio))
        panel = cv2.resize(img, (display_width, display_height), interpolation=cv2.INTER_AREA)
        labeled = cv2.copyMakeBorder(
            panel, label_band_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30)
        )
        cv2.putText(
            labeled,
            label,
            (10, int(label_band_height * 0.7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return labeled

    left_panel = _prepare_panel(original, "Original")
    right_panel = _prepare_panel(processed, "Processed")
    divider = np.full((left_panel.shape[0], 4, 3), (255, 255, 255), dtype=np.uint8)
    return np.hstack((left_panel, divider, right_panel))
