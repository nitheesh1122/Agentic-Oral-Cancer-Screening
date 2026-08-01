"""Configuration for the Image Quality Assessment Agent.

All paths and thresholds used by the agent are centralized here so the module
has no hardcoded values elsewhere and can be re-pointed at the future
9,201-image dataset by changing only this file (or the corresponding
environment variables).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


def _project_root() -> Path:
    """Return the repository root (two levels above this file)."""

    return Path(__file__).resolve().parents[2]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float value for environment variable {name!r}.") from exc


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid int value for environment variable {name!r}.") from exc


def _env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default)))


@dataclass(frozen=True, slots=True)
class QualityConfig:
    """Typed configuration for image discovery, quality gating, and preprocessing.

    Attributes:
        dataset_root: Root of the dataset tree.
        raw_dir: Directory containing raw class subfolders (auto-discovered).
        processed_dir: Directory where accepted, preprocessed images are written.
        reports_dir: Directory for CSV and comparison-image reports.
        comparisons_dir: Directory for side-by-side original/processed images.
        csv_report_path: Output path of the quality report CSV.
        supported_extensions: Case-insensitive image file extensions to discover.
        target_size: (width, height) accepted images are resized to.
        brisque_max: Maximum acceptable BRISQUE score (lower score = better quality).
        blur_min: Minimum acceptable variance-of-Laplacian blur score.
        brightness_min: Minimum acceptable mean brightness (0-255 scale).
        brightness_max: Maximum acceptable mean brightness (0-255 scale).
        contrast_min: Minimum acceptable contrast (grayscale standard deviation).
        gaussian_kernel_size: Kernel size for Gaussian noise reduction.
        clahe_clip_limit: CLAHE contrast-limiting threshold.
        clahe_tile_grid_size: CLAHE tile grid size.
        brisque_device: Torch device used for BRISQUE inference ("cpu" or "cuda").
        log_level: Logging level name for this module.
    """

    dataset_root: Path = field(default_factory=lambda: _project_root() / "dataset")
    raw_dir: Path = field(default_factory=lambda: _project_root() / "dataset" / "raw")
    processed_dir: Path = field(default_factory=lambda: _project_root() / "dataset" / "processed")
    reports_dir: Path = field(default_factory=lambda: _project_root() / "dataset" / "reports")
    comparisons_dir: Path = field(
        default_factory=lambda: _project_root() / "dataset" / "reports" / "comparisons"
    )
    csv_report_path: Path = field(
        default_factory=lambda: _project_root() / "dataset" / "reports" / "quality_report.csv"
    )

    supported_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    target_size: tuple[int, int] = (640, 640)

    brisque_max: float = 45.0
    blur_min: float = 80.0
    brightness_min: float = 40.0
    brightness_max: float = 220.0
    contrast_min: float = 20.0

    gaussian_kernel_size: tuple[int, int] = (5, 5)
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)

    brisque_device: str = "cpu"
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "QualityConfig":
        """Build configuration from defaults overridden by environment variables."""

        root = _project_root()
        return cls(
            dataset_root=_env_path("QUALITY_DATASET_ROOT", root / "dataset"),
            raw_dir=_env_path("QUALITY_RAW_DIR", root / "dataset" / "raw"),
            processed_dir=_env_path("QUALITY_PROCESSED_DIR", root / "dataset" / "processed"),
            reports_dir=_env_path("QUALITY_REPORTS_DIR", root / "dataset" / "reports"),
            comparisons_dir=_env_path(
                "QUALITY_COMPARISONS_DIR", root / "dataset" / "reports" / "comparisons"
            ),
            csv_report_path=_env_path(
                "QUALITY_CSV_REPORT_PATH", root / "dataset" / "reports" / "quality_report.csv"
            ),
            target_size=(
                _env_int("QUALITY_TARGET_WIDTH", 640),
                _env_int("QUALITY_TARGET_HEIGHT", 640),
            ),
            brisque_max=_env_float("QUALITY_BRISQUE_MAX", 45.0),
            blur_min=_env_float("QUALITY_BLUR_MIN", 80.0),
            brightness_min=_env_float("QUALITY_BRIGHTNESS_MIN", 40.0),
            brightness_max=_env_float("QUALITY_BRIGHTNESS_MAX", 220.0),
            contrast_min=_env_float("QUALITY_CONTRAST_MIN", 20.0),
            brisque_device=os.getenv("QUALITY_BRISQUE_DEVICE", "cpu"),
            log_level=os.getenv("QUALITY_LOG_LEVEL", "INFO"),
        )


DEFAULT_CONFIG = QualityConfig.from_env()
