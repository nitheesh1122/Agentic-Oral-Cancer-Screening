"""Logging helpers for the oral cancer screening framework."""

from __future__ import annotations

import logging
from typing import Final

from config import get_config

_LOGGER_FORMAT: Final[str] = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging() -> None:
    """Configure application-wide logging.

    TODO: route structured logs to file, telemetry, or experiment tracking when
    the implementation expands beyond the initial placeholder pipeline.
    """

    config = get_config()
    level = getattr(logging, config.log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format=_LOGGER_FORMAT)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module or component."""

    configure_logging()
    return logging.getLogger(name)