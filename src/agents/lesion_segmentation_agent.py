"""Lesion segmentation agent."""

from __future__ import annotations

from typing import Any

from .base import BaseAgent


class LesionSegmentationAgent(BaseAgent):
    """Generate a placeholder lesion mask from detection outputs."""

    def __init__(self) -> None:
        super().__init__(name="LesionSegmentationAgent")

    def process(self, payload: Any = None) -> dict[str, Any]:
        """Return a placeholder segmentation result.

        TODO: integrate a segmentation model and mask refinement logic.
        """

        try:
            self.logger.debug("Running placeholder lesion segmentation.")
            return self.placeholder_output(stage="segmentation", input_type=type(payload).__name__)
        except Exception as exc:
            self.logger.error("Lesion segmentation failed: %s", exc)
            raise RuntimeError("Lesion segmentation failed.") from exc
