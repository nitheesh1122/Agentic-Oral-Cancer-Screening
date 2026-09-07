"""Unit tests for BrisqueAnalyzer (src/image_quality/brisque_analyzer.py).

Mocks pyiqa's metric loading so these tests run fast and do not require the
real BRISQUE model weights. Real-model verification is covered separately by
tests/smoke_quality.py against the actual dataset.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from src.image_quality.brisque_analyzer import BrisqueAnalyzer


class _FakeMetric:
    """Stand-in for pyiqa's callable metric object."""

    def __init__(self, return_value: float = 42.0) -> None:
        self.return_value = return_value
        self.calls: list = []

    def __call__(self, arg):
        self.calls.append(arg)
        return self.return_value


@pytest.fixture
def fake_pyiqa(monkeypatch):
    fake_metric = _FakeMetric()
    fake_module = types.SimpleNamespace(create_metric=lambda name, device: fake_metric)
    monkeypatch.setitem(sys.modules, "pyiqa", fake_module)
    return fake_metric


class TestScorePath:
    def test_returns_metric_output_as_float(self, tmp_path: Path, fake_pyiqa):
        image_path = tmp_path / "image.jpg"
        image_path.write_bytes(b"fake bytes -- never decoded, mocked metric")

        analyzer = BrisqueAnalyzer(device="cpu")
        score = analyzer.score_path(image_path)

        assert score == pytest.approx(42.0)
        assert fake_pyiqa.calls == [str(image_path)]

    def test_raises_on_missing_file(self, tmp_path: Path, fake_pyiqa):
        analyzer = BrisqueAnalyzer(device="cpu")
        with pytest.raises(FileNotFoundError):
            analyzer.score_path(tmp_path / "missing.jpg")

    def test_loads_metric_only_once(self, tmp_path: Path, fake_pyiqa, monkeypatch):
        image_path = tmp_path / "image.jpg"
        image_path.write_bytes(b"fake")

        load_count = {"n": 0}

        def counting_create_metric(name, device):
            load_count["n"] += 1
            return fake_pyiqa

        monkeypatch.setitem(
            sys.modules, "pyiqa", types.SimpleNamespace(create_metric=counting_create_metric)
        )

        analyzer = BrisqueAnalyzer(device="cpu")
        analyzer.score_path(image_path)
        analyzer.score_path(image_path)

        assert load_count["n"] == 1


class TestScoreArray:
    def test_returns_metric_output_as_float(self, fake_pyiqa):
        analyzer = BrisqueAnalyzer(device="cpu")
        image = np.full((20, 20, 3), 128, dtype=np.uint8)
        score = analyzer.score_array(image)
        assert score == pytest.approx(42.0)

    def test_raises_on_empty_array(self, fake_pyiqa):
        analyzer = BrisqueAnalyzer(device="cpu")
        with pytest.raises(ValueError):
            analyzer.score_array(np.array([]))

    def test_raises_on_none(self, fake_pyiqa):
        analyzer = BrisqueAnalyzer(device="cpu")
        with pytest.raises(ValueError):
            analyzer.score_array(None)


class TestEnsureLoaded:
    def test_raises_import_error_when_pyiqa_unavailable(self, tmp_path: Path, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyiqa", None)
        image_path = tmp_path / "image.jpg"
        image_path.write_bytes(b"fake")

        analyzer = BrisqueAnalyzer(device="cpu")
        with pytest.raises(ImportError):
            analyzer.score_path(image_path)
