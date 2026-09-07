"""Unit tests for ImageQualityAssessmentAgent (src/image_quality/quality_agent.py).

BRISQUE scoring is mocked (via a fake BrisqueAnalyzer) so these tests exercise
the agent's own logic -- threshold decisions, preprocessing wiring, output
schema -- without loading the real pyiqa model.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.image_quality.config import QualityConfig
from src.image_quality.quality_agent import ImageQualityAssessmentAgent
from src.image_quality.utils import DiscoveredImage


class _FakeBrisqueAnalyzer:
    def __init__(self, score: float) -> None:
        self._score = score

    def score_path(self, image_path: Path) -> float:
        return self._score


def _write_test_image(path: Path, *, sharp: bool = True, size=(200, 200)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if sharp:
        image = np.zeros((*size, 3), dtype=np.uint8)
        image[::2, ::2] = 255
        image[1::2, 1::2] = 255
    else:
        image = np.full((*size, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _make_agent(tmp_path: Path, brisque_score: float = 10.0, **overrides) -> ImageQualityAssessmentAgent:
    config = QualityConfig(
        dataset_root=tmp_path,
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        reports_dir=tmp_path / "reports",
        comparisons_dir=tmp_path / "reports" / "comparisons",
        csv_report_path=tmp_path / "reports" / "quality_report.csv",
        **overrides,
    )
    agent = ImageQualityAssessmentAgent(config=config)
    agent.brisque_analyzer = _FakeBrisqueAnalyzer(brisque_score)
    return agent


class TestAssessQuality:
    def test_accepts_when_all_thresholds_pass(self, tmp_path: Path):
        agent = _make_agent(tmp_path)
        decision, proceed, reasons = agent.assess_quality(
            brisque_score=10.0, blur_score=200.0, brightness=128.0, contrast=50.0
        )
        assert decision == "Accepted"
        assert proceed is True
        assert reasons == []

    def test_rejects_on_high_brisque(self, tmp_path: Path):
        agent = _make_agent(tmp_path)
        decision, proceed, reasons = agent.assess_quality(
            brisque_score=999.0, blur_score=200.0, brightness=128.0, contrast=50.0
        )
        assert decision == "Rejected"
        assert proceed is False
        assert any("BRISQUE" in r for r in reasons)

    def test_rejects_on_low_blur(self, tmp_path: Path):
        agent = _make_agent(tmp_path)
        _, proceed, reasons = agent.assess_quality(
            brisque_score=10.0, blur_score=1.0, brightness=128.0, contrast=50.0
        )
        assert proceed is False
        assert any("Blur" in r for r in reasons)

    def test_rejects_on_out_of_range_brightness(self, tmp_path: Path):
        agent = _make_agent(tmp_path)
        _, proceed, reasons = agent.assess_quality(
            brisque_score=10.0, blur_score=200.0, brightness=5.0, contrast=50.0
        )
        assert proceed is False
        assert any("Brightness" in r for r in reasons)

    def test_accumulates_multiple_failure_reasons(self, tmp_path: Path):
        agent = _make_agent(tmp_path)
        _, proceed, reasons = agent.assess_quality(
            brisque_score=999.0, blur_score=1.0, brightness=5.0, contrast=1.0
        )
        assert proceed is False
        assert len(reasons) == 4


class TestProcessSingleImage:
    def test_accepted_image_produces_processed_file_and_full_output_schema(self, tmp_path: Path):
        image_path = tmp_path / "raw" / "ClassA" / "sharp.jpg"
        _write_test_image(image_path, sharp=True)
        agent = _make_agent(tmp_path, brisque_score=5.0)

        record, output = agent.process_single_image(DiscoveredImage(path=image_path, class_name="ClassA"))

        assert record["Decision"] == "Accepted"
        assert output["proceed"] is True
        assert Path(output["processed_image"]).is_file()
        assert set(output.keys()) == {
            "filename", "class", "processed_image", "brisque_score",
            "blur_score", "brightness", "contrast", "quality_decision", "proceed",
        }

    def test_rejected_image_has_no_processed_file(self, tmp_path: Path):
        image_path = tmp_path / "raw" / "ClassA" / "flat.jpg"
        _write_test_image(image_path, sharp=False)
        agent = _make_agent(tmp_path, brisque_score=5.0)

        record, output = agent.process_single_image(DiscoveredImage(path=image_path, class_name="ClassA"))

        assert record["Decision"] == "Rejected"
        assert output["proceed"] is False
        assert output["processed_image"] == ""

    def test_missing_image_is_handled_without_raising(self, tmp_path: Path):
        agent = _make_agent(tmp_path)
        missing = tmp_path / "raw" / "ClassA" / "does_not_exist.jpg"

        record, output = agent.process_single_image(DiscoveredImage(path=missing, class_name="ClassA"))

        assert "Invalid Image" in record["Decision"]
        assert output["proceed"] is False
        assert output["brisque_score"] is None

    def test_corrupt_image_is_handled_without_raising(self, tmp_path: Path):
        corrupt = tmp_path / "raw" / "ClassA" / "corrupt.jpg"
        corrupt.parent.mkdir(parents=True, exist_ok=True)
        corrupt.write_bytes(b"not a real image")
        agent = _make_agent(tmp_path)

        record, output = agent.process_single_image(DiscoveredImage(path=corrupt, class_name="ClassA"))

        assert "Invalid Image" in record["Decision"]
        assert output["proceed"] is False
