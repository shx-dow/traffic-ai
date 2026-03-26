from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.detector import VehicleDetector


def _make_detector(mode: str):
    detector = VehicleDetector.__new__(VehicleDetector)
    detector._ambulance_mode = mode
    detector._ambulance_custom = object()
    detector._ambulance_aux = object()
    detector._ambulance_conf = 0.3
    detector._ambulance_world_conf = 0.05
    detector._world_model = None
    detector._world_weights = ""
    return detector


def test_custom_mode_uses_custom_model_result():
    detector = _make_detector("custom")

    def fake_custom(frame, vehicles, model, conf):
        merged = vehicles + [{"class": "ambulance", "confidence": 0.8, "bbox": [1.0, 2.0, 3.0, 4.0]}]
        return merged, True

    detector._run_ambulance_model = fake_custom
    detector._run_yolo_world = lambda frame, vehicles: (vehicles, False)

    vehicles, emergency = detector._merge_ambulance_detections(
        frame=None,
        vehicles=[{"class": "car", "confidence": 0.9, "bbox": [0.0, 0.0, 10.0, 10.0]}],
        emergency=False,
    )

    assert emergency is True
    assert any(v["class"] == "ambulance" for v in vehicles)


def test_custom_mode_falls_back_to_yoloworld_when_custom_missing():
    detector = _make_detector("custom")
    detector._ambulance_custom = None

    detector._run_ambulance_model = lambda frame, vehicles, model, conf: (vehicles, False)

    def fake_world(frame, vehicles):
        merged = vehicles + [{"class": "ambulance", "confidence": 0.7, "bbox": [5.0, 6.0, 7.0, 8.0]}]
        return merged, True

    detector._run_yolo_world = fake_world

    vehicles, emergency = detector._merge_ambulance_detections(
        frame=None,
        vehicles=[{"class": "bus", "confidence": 0.5, "bbox": [0.0, 0.0, 9.0, 9.0]}],
        emergency=False,
    )

    assert emergency is True
    assert any(v["class"] == "ambulance" for v in vehicles)


def test_custom_merge_keeps_fire_truck_as_emergency_class():
    detector = _make_detector("custom")

    def fake_custom(frame, vehicles, model, conf):
        merged = vehicles + [{"class": "fire_truck", "confidence": 0.78, "bbox": [1.0, 2.0, 3.0, 4.0]}]
        return merged, True

    detector._run_ambulance_model = fake_custom
    detector._run_yolo_world = lambda frame, vehicles: (vehicles, False)

    vehicles, emergency = detector._merge_ambulance_detections(
        frame=None,
        vehicles=[{"class": "car", "confidence": 0.9, "bbox": [0.0, 0.0, 10.0, 10.0]}],
        emergency=False,
    )

    assert emergency is True
    assert any(v["class"] == "fire_truck" for v in vehicles)


if __name__ == "__main__":
    test_custom_mode_uses_custom_model_result()
    test_custom_mode_falls_back_to_yoloworld_when_custom_missing()
    test_custom_merge_keeps_fire_truck_as_emergency_class()
    print("PASS test_custom_ambulance")
