"""Tests for the detector ground-truth scoring script.

Only the pure metric core is exercised here (no video, no cv2, no
ultralytics) so the test runs in any environment and on CI.  End-to-end runs
against a real recorded clip belong to an offline validation pass
(`python scripts/validate_detector.py --video ... --labels ...`).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.validate_detector import compute_accuracy, load_labels


def test_perfect_predictions():
    gt = {0: 3, 1: 5, 2: 2}
    metrics = compute_accuracy({0: 3, 1: 5, 2: 2}, gt)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["within_tolerance"] == 1.0
    assert metrics["bias"] == 0.0


def test_constant_overcount_bias():
    gt = {0: 2, 1: 4, 2: 6}
    metrics = compute_accuracy({0: 4, 1: 6, 2: 8}, gt)  # over by 2 everywhere
    assert metrics["mae"] == 2.0
    assert metrics["bias"] == 2.0
    assert metrics["within_tolerance"] == 0.0  # tolerance 1 default


def test_tolerance_parameter():
    gt = {0: 2, 1: 4}
    metrics = compute_accuracy({0: 3, 1: 3}, gt, tolerance=1)
    assert metrics["within_tolerance"] == 1.0
    strict = compute_accuracy({0: 3, 1: 3}, gt, tolerance=0)
    assert strict["within_tolerance"] == 0.0


def test_empty_labels():
    metrics = compute_accuracy({}, {})
    assert metrics["frames"] == 0
    assert metrics["mae"] is None


def test_load_labels_int_and_split():
    tmp = Path(tempfile.mkdtemp())
    label_file = tmp / "labels.json"
    label_file.write_text(json.dumps({"0": 4, "5": {"north": 1, "south": 2}}), encoding="utf-8")
    labels = load_labels(label_file)
    assert labels[0]["north"] == 4
    assert labels[5]["south"] == 2
    assert sum(labels[5].values()) == 3


def test_missing_video_exits_cleanly(monkeypatch):
    from scripts.validate_detector import main

    monkeypatch.setattr(sys, "argv", ["validate_detector", "--video", "missing.mp4", "--labels", "missing.json"])
    exit_code = main()
    assert exit_code == 2
