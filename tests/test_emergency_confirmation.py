"""Tests for the Phase 3 vision-emergency temporal confirmation (in VehicleDetector).

Default confirm_vision_frames/0 passes raw detections straight through (backward-
compatible). With confirmation enabled, a single emergency frame is NOT enough;
several consecutive detections are required, and a grace window holds the
confirmation across occasional missed frames.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from vision.detector import VehicleDetector


def _zeros():
    return np.zeros((10, 10, 3), dtype=np.uint8)


def _stub_detector(confirm=3, grace=5):
    """Create a bare detector with temporal confirmation but no real model."""
    det = VehicleDetector.__new__(VehicleDetector)
    det._confirm_vision_frames = confirm
    det._vision_grace_frames = grace
    det._vision_detect_streak = 0
    det._vision_confirmed = False
    det._vision_grace = 0
    det._ambulance_mode = "none"
    det._enrich_cross_dataset = False
    det._gps_cache = {"emergency": False}
    det._gps_last_poll_ts = 0.0
    det._model = type("FakeModel", (), {"predict": lambda self, **kw: [None]})()
    det._check_gps_emergency = lambda: {"emergency": False}
    det._attach_fusion = lambda out, hint: out
    # Default: no vehicles detected (override per-test via detect_emergency=...)
    det._vehicles_from_coco = lambda r: ([], False)
    det._merge_ambulance_detections = lambda f, v, e: (v, e)
    return det


def _evidence_detector(confirm=3, grace=5):
    """Return (detector, evidence) where setting evidence.emergency toggles the
    per-frame raw emergency detection returned by detect()."""
    det = _stub_detector(confirm, grace)

    class Evidence:
        emergency = False

    evidence = Evidence()

    det._vehicles_from_coco = lambda r: (
        ([{"class": "ambulance", "confidence": 0.9, "bbox": [0, 0, 10, 10]}] if evidence.emergency else []),
        evidence.emergency,
    )
    return det, evidence


def test_single_frame_does_not_confirm():
    det, evidence = _evidence_detector(confirm=3, grace=0)
    evidence.emergency = True
    result = det.detect(_zeros())
    assert result["vision_emergency"] is False, "Single frame must not confirm emergency"


def test_three_consecutive_frames_confirm():
    det, evidence = _evidence_detector(confirm=3, grace=0)
    evidence.emergency = True
    det.detect(_zeros())
    det.detect(_zeros())
    result = det.detect(_zeros())
    assert result["vision_emergency"] is True, "Three consecutive detections should confirm"


def test_interruption_prevents_confirmation():
    det, evidence = _evidence_detector(confirm=3, grace=0)
    evidence.emergency = True
    det.detect(_zeros())
    det.detect(_zeros())
    evidence.emergency = False
    det.detect(_zeros())  # miss resets the streak
    evidence.emergency = True
    for _ in range(2):
        result = det.detect(_zeros())
    assert result["vision_emergency"] is False, "Missed frame before 3 was reached must restart the streak"


def test_missed_frame_within_grace_keeps_confirmed():
    det, evidence = _evidence_detector(confirm=1, grace=2)
    evidence.emergency = True
    det.detect(_zeros())  # confirms immediately (confirm_vision_frames=1)
    assert det._vision_confirmed is True

    evidence.emergency = False
    det.detect(_zeros())  # miss: grace 2 -> 1
    assert det._vision_confirmed is True, "Grace window should hold confirmation"
    det.detect(_zeros())  # miss: grace 1 -> 0
    assert det._vision_confirmed is True, "Still within grace"
    det.detect(_zeros())  # miss: grace exhausted -> deconfirm
    assert det._vision_confirmed is False, "Should deconfirm after grace expires"


def test_detection_resets_grace_on_reappearance():
    det, evidence = _evidence_detector(confirm=1, grace=2)
    evidence.emergency = True
    det.detect(_zeros())  # confirm
    assert det._vision_confirmed is True

    evidence.emergency = False
    det.detect(_zeros())  # miss: grace 2 -> 1
    evidence.emergency = True
    result = det.detect(_zeros())  # reappearance resets grace to 2
    assert result["vision_emergency"] is True, "Detection reappearance should restore confirmation"
    assert det._vision_grace == 2, "Grace should reset to full on reappearance"


def test_confirm_zero_passthrough():
    det, evidence = _evidence_detector(confirm=0, grace=0)
    evidence.emergency = True
    result = det.detect(_zeros())
    assert result["vision_emergency"] is True, "With confirm=0, raw detection passes through"


def test_bare_detector_compat():
    det = VehicleDetector.__new__(VehicleDetector)
    det._model = type("M", (), {"predict": lambda self, **kw: [None]})()
    det._enrich_cross_dataset = False
    det._check_gps_emergency = lambda: {"emergency": False}
    det._attach_fusion = lambda out, h: out
    det._vehicles_from_coco = lambda r: ([], False)
    det._merge_ambulance_detections = lambda f, v, e: (v, False)
    result = det.detect(_zeros())
    assert result["vision_emergency"] is False


if __name__ == "__main__":
    test_single_frame_does_not_confirm()
    test_three_consecutive_frames_confirm()
    test_interruption_prevents_confirmation()
    test_missed_frame_within_grace_keeps_confirmed()
    test_detection_resets_grace_on_reappearance()
    test_confirm_zero_passthrough()
    test_bare_detector_compat()
    print("PASS test_emergency_confirmation")
