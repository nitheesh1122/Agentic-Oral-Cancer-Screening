"""Structural validation for the YOLO detection dataset (images, labels, pairing).

Operates read-only against ``DATA_SETS/PROCESSED/detection_yolo`` (or any
directory following the same ``images/<split>/`` + ``labels/<split>/``
layout). Never modifies or deletes anything under the dataset root.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2

from .annotation_validation import LabelValidationResult, parse_yolo_label_text

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(slots=True)
class ImageRecord:
    """Validation outcome for a single image file."""

    path: Path
    exists: bool
    readable: bool
    width: int | None
    height: int | None
    error: str | None = None


@dataclass(slots=True)
class PairRecord:
    """One image/label pair and its validation outcome."""

    split: str
    stem: str
    image_path: Path | None
    label_path: Path | None
    image: ImageRecord | None = None
    label: LabelValidationResult | None = None

    @property
    def has_image(self) -> bool:
        return self.image_path is not None

    @property
    def has_label(self) -> bool:
        return self.label_path is not None


@dataclass(slots=True)
class SplitValidationReport:
    """Aggregated validation results for one split (train/val/test)."""

    split: str
    pairs: list[PairRecord] = field(default_factory=list)

    @property
    def missing_labels(self) -> list[PairRecord]:
        return [p for p in self.pairs if p.has_image and not p.has_label]

    @property
    def orphan_labels(self) -> list[PairRecord]:
        return [p for p in self.pairs if p.has_label and not p.has_image]

    @property
    def unreadable_images(self) -> list[PairRecord]:
        return [p for p in self.pairs if p.image is not None and not p.image.readable]

    @property
    def invalid_labels(self) -> list[PairRecord]:
        return [p for p in self.pairs if p.label is not None and not p.label.is_valid]

    @property
    def empty_labels(self) -> list[PairRecord]:
        return [p for p in self.pairs if p.label is not None and p.label.is_empty]

    @property
    def valid_pairs(self) -> list[PairRecord]:
        return [
            p
            for p in self.pairs
            if p.has_image
            and p.has_label
            and p.image is not None
            and p.image.readable
            and p.label is not None
            and p.label.is_valid
        ]


def validate_image(path: Path) -> ImageRecord:
    """Validate that an image file exists, is non-empty, and decodes cleanly.

    Args:
        path: Path to the image file.

    Returns:
        An ``ImageRecord`` describing the outcome. Never raises -- failures
        are captured in the record's ``error`` field so batch validation can
        continue past individual bad files.
    """

    if not path.is_file():
        return ImageRecord(path=path, exists=False, readable=False, width=None, height=None, error="file not found")

    if path.stat().st_size == 0:
        return ImageRecord(path=path, exists=True, readable=False, width=None, height=None, error="zero-byte file")

    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        return ImageRecord(
            path=path, exists=True, readable=False, width=None, height=None,
            error=f"unsupported extension: {path.suffix}",
        )

    image = cv2.imread(str(path))
    if image is None or image.size == 0:
        return ImageRecord(
            path=path, exists=True, readable=False, width=None, height=None,
            error="failed to decode (corrupt or unsupported encoding)",
        )

    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        return ImageRecord(path=path, exists=True, readable=False, width=width, height=height, error="invalid dimensions")

    return ImageRecord(path=path, exists=True, readable=True, width=width, height=height)


def discover_split_pairs(images_dir: Path, labels_dir: Path, split: str) -> SplitValidationReport:
    """Discover and validate every image/label pair in one split.

    Args:
        images_dir: Directory containing image files for this split (flat).
        labels_dir: Directory containing corresponding ``.txt`` label files.
        split: Split name (``"train"``, ``"val"``, or ``"test"``) for reporting.

    Returns:
        A ``SplitValidationReport`` with one ``PairRecord`` per stem discovered
        on either side (image and/or label).

    Raises:
        FileNotFoundError: If ``images_dir`` does not exist.
    """

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_files = {
        f.stem: f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    }
    label_files = {f.stem: f for f in labels_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt"} if labels_dir.is_dir() else {}

    all_stems = sorted(set(image_files) | set(label_files))
    report = SplitValidationReport(split=split)

    for stem in all_stems:
        image_path = image_files.get(stem)
        label_path = label_files.get(stem)
        pair = PairRecord(split=split, stem=stem, image_path=image_path, label_path=label_path)

        if image_path is not None:
            pair.image = validate_image(image_path)
        if label_path is not None:
            pair.label = parse_yolo_label_text(label_path.read_text(encoding="utf-8", errors="replace"))

        report.pairs.append(pair)

    return report


def validate_detection_dataset(dataset_root: Path, splits: tuple[str, ...] = ("train", "val", "test")) -> dict[str, SplitValidationReport]:
    """Validate the full detection dataset across all splits.

    Args:
        dataset_root: Root of the ``detection_yolo`` tree (contains ``images/`` and ``labels/``).
        splits: Split subdirectory names to validate.

    Returns:
        Mapping of split name to its ``SplitValidationReport``.
    """

    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"

    return {
        split: discover_split_pairs(images_root / split, labels_root / split, split)
        for split in splits
    }
