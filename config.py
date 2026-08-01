"""Application configuration for the oral cancer screening framework.

This module intentionally exposes a small, typed configuration surface for the
initial research prototype. The settings are placeholders and can be extended as
the implementation matures.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Configuration values used across the placeholder pipeline.

    Attributes:
        project_title: Human-readable project name.
        log_level: Default logging level name.
        image_quality_threshold: Placeholder threshold for image quality checks.
        confidence_threshold: Placeholder threshold for confidence-aware outputs.
    """

    project_title: str = "Agentic Oral Cancer Screening Framework"
    log_level: str = "INFO"
    image_quality_threshold: float = 0.5
    confidence_threshold: float = 0.75


DEFAULT_CONFIG = AppConfig()


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Return the process-wide application configuration.

    The current implementation reads environment variables only for optional
    tuning. TODO: replace this with a structured settings system when the
    research prototype needs experiment-level overrides.
    """

    try:
        return AppConfig(
            project_title=os.getenv("PROJECT_TITLE", DEFAULT_CONFIG.project_title),
            log_level=os.getenv("LOG_LEVEL", DEFAULT_CONFIG.log_level),
            image_quality_threshold=float(
                os.getenv("IMAGE_QUALITY_THRESHOLD", str(DEFAULT_CONFIG.image_quality_threshold))
            ),
            confidence_threshold=float(
                os.getenv("CONFIDENCE_THRESHOLD", str(DEFAULT_CONFIG.confidence_threshold))
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid application configuration environment values.") from exc