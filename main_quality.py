"""Standalone entry point for the Image Quality Assessment Agent.

Runs the agent end to end against ``dataset/raw`` and writes processed
images, comparison images, and the quality report CSV under ``dataset/``.
The returned structured outputs are the exact handoff contract for the
future YOLOv11 Lesion Detection Agent.
"""

from __future__ import annotations

from src.image_quality import ImageQualityAssessmentAgent


def main() -> int:
    """Run the Image Quality Assessment Agent pipeline.

    Returns:
        Process exit code: 0 on success, 1 on failure.
    """

    agent = ImageQualityAssessmentAgent()
    try:
        agent.run()
        return 0
    except (ImportError, FileNotFoundError, ValueError) as exc:
        print(f"Pipeline failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
