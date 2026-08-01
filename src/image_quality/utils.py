"""Shared I/O, logging, and discovery helpers for the image quality module.

Kept self-contained (no dependency on the repository's root-level
``config.py``/``logger.py``) so the module can be lifted into another
project unchanged when the dataset grows to 9,201 images.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pandas as pd

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_configured = False


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a module-scoped logger, configuring the root handler once.

    Args:
        name: Logger name, typically ``__name__`` of the caller.
        level: Logging level name (e.g. "INFO", "DEBUG").

    Returns:
        A configured ``logging.Logger`` instance.
    """

    global _configured
    if not _configured:
        logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=_LOG_FORMAT)
        _configured = True
    return logging.getLogger(name)


logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DiscoveredImage:
    """A single discovered raw image and its class label.

    Attributes:
        path: Absolute path to the raw image file.
        class_name: Name of the immediate parent folder under the raw
            directory, treated as the classification label.
    """

    path: Path
    class_name: str


def discover_images(raw_dir: Path, supported_extensions: tuple[str, ...]) -> list[DiscoveredImage]:
    """Recursively discover images under each class subfolder of ``raw_dir``.

    Class subfolders and filenames are never hardcoded: every immediate
    subdirectory of ``raw_dir`` is treated as a class, and every file within
    it matching ``supported_extensions`` is discovered automatically.

    Args:
        raw_dir: Root directory containing one subfolder per class.
        supported_extensions: Case-insensitive extensions to include.

    Returns:
        A sorted list of ``DiscoveredImage`` entries.

    Raises:
        FileNotFoundError: If ``raw_dir`` does not exist.
        ValueError: If no class subfolders or no images are found.
    """

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")

    extensions = {ext.lower() for ext in supported_extensions}
    class_dirs = sorted(p for p in raw_dir.iterdir() if p.is_dir())
    if not class_dirs:
        raise ValueError(f"No class subfolders found under: {raw_dir}")

    discovered: list[DiscoveredImage] = []
    for class_dir in class_dirs:
        for file_path in sorted(class_dir.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                discovered.append(DiscoveredImage(path=file_path, class_name=class_dir.name))

    if not discovered:
        raise ValueError(f"No images with extensions {sorted(extensions)} found under: {raw_dir}")

    return discovered


def ensure_output_directories(
    processed_dir: Path, comparisons_dir: Path, class_names: Iterator[str]
) -> None:
    """Create processed/comparison directories, including per-class subfolders.

    Args:
        processed_dir: Root directory for accepted, preprocessed images.
        comparisons_dir: Directory for side-by-side comparison images.
        class_names: Class folder names discovered from the raw dataset.
    """

    comparisons_dir.mkdir(parents=True, exist_ok=True)
    for class_name in set(class_names):
        (processed_dir / class_name).mkdir(parents=True, exist_ok=True)


def load_image(path: Path) -> np.ndarray:
    """Read an image file into a BGR ``numpy`` array.

    Args:
        path: Path to the image file.

    Returns:
        BGR image array.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file cannot be decoded as an image.
    """

    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    image = cv2.imread(str(path))
    if image is None or image.size == 0:
        raise ValueError(f"Unable to decode image (corrupt or unsupported format): {path}")
    return image


def save_image(image: np.ndarray, path: Path) -> None:
    """Write a BGR ``numpy`` array to disk, creating parent folders as needed.

    Args:
        image: BGR image array to persist.
        path: Destination file path.

    Raises:
        RuntimeError: If the encoder fails to write the file.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), image)
    if not success:
        raise RuntimeError(f"Failed to write image to disk: {path}")


def write_quality_report(records: list[dict], csv_path: Path) -> None:
    """Write per-image quality records to a CSV report.

    Args:
        records: List of row dictionaries matching the report schema.
        csv_path: Destination CSV file path.
    """

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame.from_records(records)
    dataframe.to_csv(csv_path, index=False)
