"""Dashboard agent."""

from __future__ import annotations

from typing import Any

from .base import BaseAgent


class DashboardAgent(BaseAgent):
    """Prepare placeholder dashboard state for clinical review."""

    def __init__(self) -> None:
        super().__init__(name="DashboardAgent")

    def process(self, payload: Any = None) -> dict[str, Any]:
        """Return a placeholder dashboard payload.

        TODO: connect the presentation layer to the actual application UI.
        """

        try:
            self.logger.debug("Running placeholder dashboard assembly.")
            return self.placeholder_output(stage="dashboard", input_type=type(payload).__name__)
        except Exception as exc:
            self.logger.error("Dashboard assembly failed: %s", exc)
            raise RuntimeError("Dashboard assembly failed.") from exc
