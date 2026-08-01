"""Recommendation agent."""

from __future__ import annotations

from typing import Any

from .base import BaseAgent


class RecommendationAgent(BaseAgent):
    """Generate clinician-facing placeholder recommendations."""

    def __init__(self) -> None:
        super().__init__(name="RecommendationAgent")

    def process(self, payload: Any = None) -> dict[str, Any]:
        """Return a placeholder recommendation payload.

        TODO: generate templated, evidence-linked, and confidence-aware output.
        """

        try:
            self.logger.debug("Running placeholder recommendation generation.")
            return self.placeholder_output(stage="recommendation", input_type=type(payload).__name__)
        except Exception as exc:
            self.logger.error("Recommendation generation failed: %s", exc)
            raise RuntimeError("Recommendation generation failed.") from exc
