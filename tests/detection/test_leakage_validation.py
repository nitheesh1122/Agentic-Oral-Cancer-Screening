"""Unit tests for leakage checks (src/detection/leakage_validation.py)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.detection.leakage_validation import (
    find_cross_split_duplicates,
    find_patient_split_conflicts,
    hash_file,
)


def _write_image(path: Path, value: int, size=(20, 20)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((*size, 3), value, dtype=np.uint8))


class TestHashFile:
    def test_identical_content_same_hash(self, tmp_path: Path):
        a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
        _write_image(a, 50)
        _write_image(b, 50)
        assert hash_file(a) == hash_file(b)

    def test_different_content_different_hash(self, tmp_path: Path):
        a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
        _write_image(a, 50)
        _write_image(b, 200)
        assert hash_file(a) != hash_file(b)


class TestFindCrossSplitDuplicates:
    def test_no_leakage_when_all_images_unique(self, tmp_path: Path):
        train_dir, val_dir = tmp_path / "train", tmp_path / "val"
        _write_image(train_dir / "a.jpg", 10)
        _write_image(val_dir / "b.jpg", 20)

        report = find_cross_split_duplicates({"train": train_dir, "val": val_dir})

        assert not report.has_leakage
        assert report.total_images_hashed == 2

    def test_detects_same_content_across_splits(self, tmp_path: Path):
        train_dir, val_dir = tmp_path / "train", tmp_path / "val"
        _write_image(train_dir / "a.jpg", 77)
        _write_image(val_dir / "b.jpg", 77)  # same pixel content, different filename

        report = find_cross_split_duplicates({"train": train_dir, "val": val_dir})

        assert report.has_leakage
        assert len(report.duplicate_groups) == 1
        assert report.duplicate_groups[0].splits == {"train", "val"}

    def test_duplicate_within_same_split_is_not_flagged_as_leakage(self, tmp_path: Path):
        train_dir = tmp_path / "train"
        _write_image(train_dir / "a.jpg", 77)
        _write_image(train_dir / "b.jpg", 77)

        report = find_cross_split_duplicates({"train": train_dir})

        assert not report.has_leakage


class TestFindPatientSplitConflicts:
    def _row(self, task, source, patient, status, split):
        return {"task": task, "source_dataset": source, "patient_id": patient, "status": status, "split": split}

    def test_no_conflict_when_patient_confined_to_one_split(self):
        rows = [
            self._row("detection", "CODE", "P1", "INCLUDED", "train"),
            self._row("detection", "CODE", "P1", "INCLUDED", "train"),
            self._row("detection", "CODE", "P2", "INCLUDED", "val"),
        ]
        report = find_patient_split_conflicts(rows, task="detection")
        assert not report.has_leakage

    def test_conflict_when_same_patient_in_two_splits(self):
        rows = [
            self._row("detection", "CODE", "P1", "INCLUDED", "train"),
            self._row("detection", "CODE", "P1", "INCLUDED", "val"),
        ]
        report = find_patient_split_conflicts(rows, task="detection")
        assert report.has_leakage
        assert ("CODE", "P1") in report.conflicting_patients

    def test_blank_patient_id_is_skipped_not_flagged(self):
        rows = [
            self._row("detection", "merged", "", "INCLUDED", "train"),
            self._row("detection", "merged", "", "INCLUDED", "val"),
        ]
        report = find_patient_split_conflicts(rows, task="detection")
        assert not report.has_leakage
        assert report.total_patients_checked == 0

    def test_excluded_status_rows_are_ignored(self):
        rows = [
            self._row("detection", "CODE", "P1", "EXCLUDED", "train"),
            self._row("detection", "CODE", "P1", "INCLUDED", "val"),
        ]
        report = find_patient_split_conflicts(rows, task="detection")
        assert not report.has_leakage

    def test_other_task_rows_are_ignored(self):
        rows = [
            self._row("classification", "CODE", "P1", "INCLUDED", "train"),
            self._row("detection", "CODE", "P1", "INCLUDED", "val"),
        ]
        report = find_patient_split_conflicts(rows, task="detection")
        assert not report.has_leakage
