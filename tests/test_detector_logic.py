from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.detector import VehicleDetector


def _make_detector_for_merge(mode: str):
    detector = VehicleDetector.__new__(VehicleDetector)
    detector._ambulance_mode = mode
    detector._ambulance_custom = object()
    detector._ambulance_aux = object()
    detector._ambulance_conf = 0.3
    detector._ambulance_world_conf = 0.05
    detector._world_model = None
    detector._world_weights = ""
    return detector


def test_merge_preserves_existing_emergency_flag_on_model_failure():
    detector = _make_detector_for_merge("custom")

    def fake_run_ambulance_model(frame, vehicles, model, conf):
        return vehicles, False

    detector._run_ambulance_model = fake_run_ambulance_model
    detector._run_yolo_world = lambda frame, vehicles: (vehicles, False)

    vehicles, emergency = detector._merge_ambulance_detections(
        frame=None,
        vehicles=[{"class": "ambulance", "confidence": 0.8, "bbox": [0, 0, 10, 10]}],
        emergency=True,
    )

    assert emergency is True
    assert len(vehicles) == 1


def test_merge_can_set_emergency_when_aux_finds_ambulance():
    detector = _make_detector_for_merge("aux_weights")

    calls = {"n": 0}

    def fake_run_ambulance_model(frame, vehicles, model, conf):
        calls["n"] += 1
        if calls["n"] == 1:
            return vehicles, False
        found = vehicles + [{"class": "ambulance", "confidence": 0.7, "bbox": [1, 1, 5, 5]}]
        return found, True

    detector._run_ambulance_model = fake_run_ambulance_model
    detector._run_yolo_world = lambda frame, vehicles: (vehicles, False)

    vehicles, emergency = detector._merge_ambulance_detections(
        frame=None,
        vehicles=[{"class": "car", "confidence": 0.9, "bbox": [0, 0, 10, 10]}],
        emergency=False,
    )

    assert emergency is True
    assert any(v["class"] == "ambulance" for v in vehicles)


if __name__ == "__main__":
    test_merge_preserves_existing_emergency_flag_on_model_failure()
    test_merge_can_set_emergency_when_aux_finds_ambulance()
    print("PASS test_detector_logic")
