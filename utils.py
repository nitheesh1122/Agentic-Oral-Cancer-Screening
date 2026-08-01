"""Shared utility helpers for the research prototype."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_placeholder_result(agent_name: str, **payload: Any) -> dict[str, Any]:
    """Create a typed placeholder result for a pipeline stage.

    TODO: replace this with stage-specific structured outputs once the model
    implementations are available.
    """

    result: dict[str, Any] = {
        "agent": agent_name,
        "status": "placeholder",
    }
    result.update(payload)
    return result


def ensure_parent_directory(path: str | Path) -> Path:
    """Ensure the parent directory of a path exists and return the path.

    This helper is intentionally conservative and does not create files.
    """

    try:
        resolved_path = Path(path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        return resolved_path
    except OSError as exc:
        raise OSError(f"Unable to prepare directory for path: {path}") from exc


def safe_text(value: Any) -> str:
    """Convert arbitrary input into a safe string representation."""

    try:
        return "" if value is None else str(value)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ValueError("Unable to convert value to text.") from exc