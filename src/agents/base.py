"""Shared base class for all placeholder agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Any

from config import AppConfig, get_config
from logger import get_logger
from utils import build_placeholder_result


class BaseAgent(ABC):
    """Common initialization, logging, and placeholder behavior for agents."""

    def __init__(self, name: str, config: AppConfig | None = None, logger: logging.Logger | None = None) -> None:
        self.name = name
        self.config = config if config is not None else get_config()
        self.logger = logger if logger is not None else get_logger(f"src.agents.{name.lower()}")
        self.logger.debug("Initialized %s with placeholder configuration.", self.name)

    def initialize(self) -> None:
        """Perform lightweight startup checks for the agent.

        TODO: add real dependency loading, model warm-up, and health checks.
        """

        try:
            self.logger.debug("%s initialization completed.", self.name)
        except Exception as exc:  # pragma: no cover - defensive guard
            raise RuntimeError(f"Failed to initialize {self.name}.") from exc

    def placeholder_output(self, **payload: Any) -> dict[str, Any]:
        """Build a consistent placeholder result for downstream stages."""

        return build_placeholder_result(self.name, **payload)

    @abstractmethod
    def process(self, payload: Any = None) -> dict[str, Any]:
        """Process input data and return a placeholder result."""
