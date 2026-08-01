"""Vision-language agent."""

from __future__ import annotations

from typing import Any

from .base import BaseAgent


class VisionLanguageAgent(BaseAgent):
    """Interpret multimodal evidence using a placeholder VLM workflow."""

    def __init__(self) -> None:
        super().__init__(name="VisionLanguageAgent")

    def process(self, payload: Any = None) -> dict[str, Any]:
        """Return a placeholder multimodal interpretation.

        TODO: connect image-text prompting and structured observation parsing.
        """

        try:
            self.logger.debug("Running placeholder vision-language reasoning.")
            return self.placeholder_output(stage="vision_language", input_type=type(payload).__name__)
        except Exception as exc:
            self.logger.error("Vision-language reasoning failed: %s", exc)
            raise RuntimeError("Vision-language reasoning failed.") from exc
