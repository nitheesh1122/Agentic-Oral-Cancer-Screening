"""Unit tests for structural dataset validation (src/detection/dataset_validation.py)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.detection.dataset_validation import (
    discover_split_pairs,
    validate_detection_dataset,
    validate_image,
)


def _write_image(path: Path, size=(50, 50)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((*size, 3), 100, dtype=np.uint8))


def _write_label(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestValidateImage:
    def test_valid_image(self, tmp_path: Path):
        path = tmp_path / "a.jpg"
        _write_image(path)
        record = validate_image(path)
        assert record.readable
        assert record.width == 50 and record.height == 50

    def test_missing_file(self, tmp_path: Path):
        record = validate_image(tmp_path / "missing.jpg")
        assert not record.exists
        assert not record.readable

    def test_zero_byte_file(self, tmp_path: Path):
        path = tmp_path / "empty.jpg"
        path.write_bytes(b"")
        record = validate_image(path)
        assert record.exists
        assert not record.readable
        assert "zero-byte" in record.error

    def test_corrupt_file(self, tmp_path: Path):
        path = tmp_path / "corrupt.jpg"
        path.write_bytes(b"not a real image")
        record = validate_image(path)
        assert not record.readable
        assert "decode" in record.error

    def test_unsupported_extension(self, tmp_path: Path):
        path = tmp_path / "a.gif"
        path.write_bytes(b"GIF89a")
        record = validate_image(path)
        assert not record.readable
        assert "unsupported extension" in record.error


class TestDiscoverSplitPairs:
    def test_matched_pair_is_valid(self, tmp_path: Path):
        images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
        _write_image(images_dir / "img1.jpg")
        _write_label(labels_dir / "img1.txt", "0 0.5 0.5 0.2 0.2\n")

        report = discover_split_pairs(images_dir, labels_dir, "train")

        assert len(report.pairs) == 1
        assert len(report.valid_pairs) == 1
        assert not report.missing_labels
        assert not report.orphan_labels

    def test_missing_label_is_flagged(self, tmp_path: Path):
        images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
        _write_image(images_dir / "img1.jpg")

        report = discover_split_pairs(images_dir, labels_dir, "train")

        assert len(report.missing_labels) == 1

    def test_orphan_label_is_flagged(self, tmp_path: Path):
        images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
        images_dir.mkdir(parents=True)
        _write_label(labels_dir / "img1.txt", "0 0.5 0.5 0.2 0.2\n")

        report = discover_split_pairs(images_dir, labels_dir, "train")

        assert len(report.orphan_labels) == 1

    def test_empty_label_file_is_tracked_but_still_counted_as_a_valid_trainable_pair(self, tmp_path: Path):
        # An empty label file (zero rows) is a legitimate negative/background
        # sample in this dataset (e.g. merged's "negative (no box)" images),
        # not a malformed annotation -- so it's valid for training while
        # still being reported separately via `empty_labels` for review.
        images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
        _write_image(images_dir / "img1.jpg")
        _write_label(labels_dir / "img1.txt", "")

        report = discover_split_pairs(images_dir, labels_dir, "train")

        assert len(report.empty_labels) == 1
        assert len(report.invalid_labels) == 0
        assert len(report.valid_pairs) == 1

    def test_invalid_label_is_flagged_and_excluded_from_valid_pairs(self, tmp_path: Path):
        images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
        _write_image(images_dir / "img1.jpg")
        _write_label(labels_dir / "img1.txt", "5 0.5 0.5 0.2 0.2\n")

        report = discover_split_pairs(images_dir, labels_dir, "train")

        assert len(report.invalid_labels) == 1
        assert len(report.valid_pairs) == 0

    def test_raises_if_images_dir_missing(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            discover_split_pairs(tmp_path / "no_images", tmp_path / "labels", "train")


class TestValidateDetectionDataset:
    def test_validates_all_requested_splits(self, tmp_path: Path):
        for split in ("train", "val", "test"):
            _write_image(tmp_path / "images" / split / "img1.jpg")
            _write_label(tmp_path / "labels" / split / "img1.txt", "0 0.5 0.5 0.2 0.2\n")

        reports = validate_detection_dataset(tmp_path)

        assert set(reports.keys()) == {"train", "val", "test"}
        for report in reports.values():
            assert len(report.valid_pairs) == 1
