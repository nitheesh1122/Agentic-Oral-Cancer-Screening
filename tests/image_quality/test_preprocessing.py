"""Unit tests for the preprocessing pipeline (src/image_quality/preprocessing.py)."""

from __future__ import annotations

import numpy as np
import pytest

from src.image_quality.config import QualityConfig
from src.image_quality.preprocessing import (
    apply_clahe,
    build_comparison_image,
    denormalize_image,
    normalize_image,
    reduce_noise,
    resize_image,
    run_preprocessing_pipeline,
)


def _bgr(shape=(100, 80, 3), value=128) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


class TestResizeImage:
    def test_resizes_to_target_dimensions(self):
        result = resize_image(_bgr(), target_size=(64, 64))
        assert result.shape[:2] == (64, 64)

    def test_raises_on_empty_image(self):
        with pytest.raises(ValueError):
            resize_image(np.array([]), (64, 64))


class TestReduceNoise:
    def test_preserves_shape(self):
        image = _bgr()
        result = reduce_noise(image, kernel_size=(5, 5))
        assert result.shape == image.shape


class TestApplyClahe:
    def test_preserves_shape_and_dtype(self):
        image = _bgr()
        result = apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8))
        assert result.shape == image.shape
        assert result.dtype == np.uint8


class TestNormalizeRoundTrip:
    def test_normalize_then_denormalize_recovers_original(self):
        image = _bgr(value=200)
        normalized = normalize_image(image)
        assert normalized.dtype == np.float32
        assert normalized.max() <= 1.0
        recovered = denormalize_image(normalized)
        assert np.array_equal(recovered, image)


class TestRunPreprocessingPipeline:
    def test_produces_expected_output_shapes_and_dtypes(self):
        config = QualityConfig(target_size=(64, 64))
        normalized, persistable = run_preprocessing_pipeline(_bgr(), config)
        assert normalized.shape[:2] == (64, 64)
        assert normalized.dtype == np.float32
        assert persistable.shape[:2] == (64, 64)
        assert persistable.dtype == np.uint8


class TestBuildComparisonImage:
    def test_raises_on_empty_input(self):
        with pytest.raises(ValueError):
            build_comparison_image(np.array([]), _bgr())

    def test_produces_wider_image_than_either_input(self):
        original = _bgr(shape=(100, 80, 3))
        processed = _bgr(shape=(64, 64, 3))
        combined = build_comparison_image(original, processed)
        assert combined.shape[1] > max(original.shape[1], processed.shape[1])
