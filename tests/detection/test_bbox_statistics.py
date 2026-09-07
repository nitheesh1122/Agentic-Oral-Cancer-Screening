"""Unit tests for bounding-box statistics (src/detection/bbox_statistics.py)."""

from __future__ import annotations

import pytest

from src.detection.annotation_validation import YoloBox
from src.detection.bbox_statistics import compute_bbox_statistics, compute_per_image_box_counts


def _box(w, h, x=0.5, y=0.5) -> YoloBox:
    return YoloBox(class_id=0, x_center=x, y_center=y, width=w, height=h)


class TestComputeBboxStatistics:
    def test_empty_list_returns_zeroed_stats(self):
        stats = compute_bbox_statistics([])
        assert stats.count == 0
        assert stats.width.mean == 0.0

    def test_basic_percentiles(self):
        boxes = [_box(0.2, 0.2), _box(0.4, 0.4), _box(0.6, 0.6)]
        stats = compute_bbox_statistics(boxes)
        assert stats.count == 3
        assert stats.width.p50 == 0.4
        assert stats.width.p0 == 0.2
        assert stats.width.p100 == 0.6

    def test_flags_tiny_boxes(self):
        boxes = [_box(0.01, 0.01)]  # area = 0.0001, below default 0.001 threshold
        stats = compute_bbox_statistics(boxes)
        assert stats.tiny_boxes == 1

    def test_flags_near_full_image_boxes(self):
        boxes = [_box(0.99, 0.99)]  # area ~ 0.98, above 0.95 threshold
        stats = compute_bbox_statistics(boxes)
        assert stats.near_full_image_boxes == 1

    def test_flags_extreme_aspect_ratio(self):
        boxes = [_box(0.5, 0.05)]  # aspect ratio 10, above default 5.0
        stats = compute_bbox_statistics(boxes)
        assert stats.extreme_aspect_ratio_boxes == 1

    def test_normal_box_not_flagged(self):
        boxes = [_box(0.3, 0.3)]
        stats = compute_bbox_statistics(boxes)
        assert stats.tiny_boxes == 0
        assert stats.near_full_image_boxes == 0
        assert stats.extreme_aspect_ratio_boxes == 0


class TestComputePerImageBoxCounts:
    def test_buckets_correctly(self):
        result = compute_per_image_box_counts([0, 1, 1, 2, 3, 4])
        assert result.zero == 1
        assert result.one == 2
        assert result.two == 1
        assert result.three_or_more == 2

    def test_min_max_mean_median(self):
        result = compute_per_image_box_counts([1, 1, 3])
        assert result.min_annotations == 1
        assert result.max_annotations == 3
        assert result.mean_annotations == pytest.approx(5 / 3)
        assert result.median_annotations == 1

    def test_empty_input(self):
        result = compute_per_image_box_counts([])
        assert result.total_images == 0
        assert result.mean_annotations == 0.0
        assert result.median_annotations == 0.0
