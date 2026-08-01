"""Image quality assessment agent."""

from __future__ import annotations

from typing import Any

from .base import BaseAgent


class ImageQualityAgent(BaseAgent):
    """Assess whether an oral image is suitable for downstream analysis."""

    def __init__(self) -> None:
        super().__init__(name="ImageQualityAgent")

    def process(self, payload: Any = None) -> dict[str, Any]:
        """Return a placeholder quality assessment result.

        TODO: implement blur, exposure, and framing checks.
        """

        try:
            self.logger.debug("Running placeholder image quality assessment.")
            return self.placeholder_output(stage="image_quality", input_type=type(payload).__name__)
        except Exception as exc:
            self.logger.error("Image quality assessment failed: %s", exc)
            raise RuntimeError("Image quality assessment failed.") from exc
