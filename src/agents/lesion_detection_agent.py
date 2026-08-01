"""Lesion detection agent."""

from __future__ import annotations

from typing import Any

from .base import BaseAgent


class LesionDetectionAgent(BaseAgent):
    """Detect candidate oral lesions from image inputs."""

    def __init__(self) -> None:
        super().__init__(name="LesionDetectionAgent")

    def process(self, payload: Any = None) -> dict[str, Any]:
        """Return a placeholder detection result.

        TODO: connect a detection model and bounding-box post-processing.
        """

        try:
            self.logger.debug("Running placeholder lesion detection.")
            return self.placeholder_output(stage="detection", input_type=type(payload).__name__)
        except Exception as exc:
            self.logger.error("Lesion detection failed: %s", exc)
            raise RuntimeError("Lesion detection failed.") from exc
