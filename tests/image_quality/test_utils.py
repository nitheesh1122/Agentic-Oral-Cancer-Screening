"""Unit tests for I/O and discovery helpers (src/image_quality/utils.py)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.image_quality.utils import (
    DiscoveredImage,
    discover_images,
    ensure_output_directories,
    load_image,
    save_image,
    write_quality_report,
)


def _write_dummy_jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((20, 20, 3), 100, dtype=np.uint8)
    cv2.imwrite(str(path), image)


class TestDiscoverImages:
    def test_discovers_images_grouped_by_class_subfolder(self, tmp_path: Path):
        _write_dummy_jpg(tmp_path / "ClassA" / "a1.jpg")
        _write_dummy_jpg(tmp_path / "ClassA" / "a2.jpg")
        _write_dummy_jpg(tmp_path / "ClassB" / "b1.jpg")

        items = discover_images(tmp_path, (".jpg",))

        assert len(items) == 3
        assert {i.class_name for i in items} == {"ClassA", "ClassB"}

    def test_ignores_unsupported_extensions(self, tmp_path: Path):
        _write_dummy_jpg(tmp_path / "ClassA" / "a1.jpg")
        (tmp_path / "ClassA" / "notes.txt").parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "ClassA" / "notes.txt").write_text("not an image")

        items = discover_images(tmp_path, (".jpg",))

        assert len(items) == 1

    def test_raises_if_raw_dir_missing(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            discover_images(tmp_path / "does_not_exist", (".jpg",))

    def test_raises_if_no_class_subfolders(self, tmp_path: Path):
        with pytest.raises(ValueError):
            discover_images(tmp_path, (".jpg",))

    def test_raises_if_no_images_found(self, tmp_path: Path):
        (tmp_path / "ClassA").mkdir()
        with pytest.raises(ValueError):
            discover_images(tmp_path, (".jpg",))


class TestLoadImage:
    def test_loads_valid_image(self, tmp_path: Path):
        path = tmp_path / "valid.jpg"
        _write_dummy_jpg(path)
        image = load_image(path)
        assert image.shape[:2] == (20, 20)

    def test_raises_on_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_image(tmp_path / "missing.jpg")

    def test_raises_on_corrupt_image(self, tmp_path: Path):
        path = tmp_path / "corrupt.jpg"
        path.write_bytes(b"this is not a valid image file")
        with pytest.raises(ValueError):
            load_image(path)


class TestSaveImage:
    def test_saves_and_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "nested" / "out.jpg"
        image = np.full((10, 10, 3), 50, dtype=np.uint8)
        save_image(image, path)
        assert path.is_file()


class TestEnsureOutputDirectories:
    def test_creates_per_class_and_comparison_dirs(self, tmp_path: Path):
        processed_dir = tmp_path / "processed"
        comparisons_dir = tmp_path / "comparisons"
        ensure_output_directories(processed_dir, comparisons_dir, iter(["A", "B"]))
        assert (processed_dir / "A").is_dir()
        assert (processed_dir / "B").is_dir()
        assert comparisons_dir.is_dir()


class TestWriteQualityReport:
    def test_writes_csv_with_records(self, tmp_path: Path):
        csv_path = tmp_path / "reports" / "quality_report.csv"
        write_quality_report([{"Filename": "a.jpg", "Decision": "Accepted"}], csv_path)
        assert csv_path.is_file()
        content = csv_path.read_text()
        assert "a.jpg" in content
        assert "Accepted" in content
