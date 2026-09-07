"""Unit tests for reference-free quality metrics (src/image_quality/metrics.py)."""

from __future__ import annotations

import numpy as np
import pytest

from src.image_quality.metrics import (
    compute_blur_score,
    compute_brightness,
    compute_contrast,
    to_grayscale,
)


def _solid_bgr(value: int, shape=(50, 50, 3)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


def _checkerboard_bgr(shape=(50, 50, 3)) -> np.ndarray:
    image = np.zeros(shape, dtype=np.uint8)
    image[::2, ::2] = 255
    image[1::2, 1::2] = 255
    return image


class TestToGrayscale:
    def test_converts_bgr_to_2d(self):
        gray = to_grayscale(_solid_bgr(100))
        assert gray.ndim == 2

    def test_passes_through_already_grayscale(self):
        gray_in = np.full((10, 10), 128, dtype=np.uint8)
        assert to_grayscale(gray_in) is gray_in

    def test_raises_on_empty_image(self):
        with pytest.raises(ValueError):
            to_grayscale(np.array([]))

    def test_raises_on_unsupported_shape(self):
        with pytest.raises(ValueError):
            to_grayscale(np.zeros((10, 10, 4), dtype=np.uint8))


class TestComputeBlurScore:
    def test_flat_image_has_zero_variance(self):
        assert compute_blur_score(_solid_bgr(128)) == pytest.approx(0.0, abs=1e-6)

    def test_high_frequency_pattern_scores_higher_than_flat(self):
        flat_score = compute_blur_score(_solid_bgr(128))
        sharp_score = compute_blur_score(_checkerboard_bgr())
        assert sharp_score > flat_score


class TestComputeBrightness:
    def test_matches_known_solid_value(self):
        assert compute_brightness(_solid_bgr(200)) == pytest.approx(200.0, abs=1e-6)

    def test_black_image_is_zero(self):
        assert compute_brightness(_solid_bgr(0)) == pytest.approx(0.0, abs=1e-6)


class TestComputeContrast:
    def test_flat_image_has_zero_contrast(self):
        assert compute_contrast(_solid_bgr(128)) == pytest.approx(0.0, abs=1e-6)

    def test_checkerboard_has_positive_contrast(self):
        assert compute_contrast(_checkerboard_bgr()) > 0.0
