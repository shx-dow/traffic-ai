from __future__ import annotations

import os
import sys

import numpy as np

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


def test_normalize_emergency_label_supports_fire_services():
    detector = _make_detector_for_merge("custom")

    assert detector._normalize_emergency_label("fire truck") == "fire_truck"
    assert detector._normalize_emergency_label("fire_engine") == "fire_truck"
    assert detector._normalize_emergency_label("ambulance") == "ambulance"


def test_detect_reports_gps_and_vision_emergency_separately():
    detector = VehicleDetector.__new__(VehicleDetector)
    detector._enrich_cross_dataset = False
    detector._model = type(
        "FakeModel",
        (),
        {"predict": lambda self, source, conf, verbose: [object()]},
    )()
    detector._vehicles_from_coco = lambda raw_result: ([{"class": "car", "confidence": 0.9, "bbox": [0, 0, 10, 10]}], False)
    detector._merge_ambulance_detections = lambda frame, vehicles, emergency: (vehicles, emergency)
    detector._check_gps_emergency = lambda: {"emergency": True, "vehicle_id": "AMB_1"}
    detector._attach_fusion = lambda out, video_source_hint: out

    out = detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))

    assert out["vision_emergency"] is False
    assert out["gps_emergency"] is True
    assert out["emergency"] is True


def test_detect_aggregates_vision_and_gps_emergency():
    detector = VehicleDetector.__new__(VehicleDetector)
    detector._enrich_cross_dataset = False
    detector._model = type(
        "FakeModel",
        (),
        {"predict": lambda self, source, conf, verbose: [object()]},
    )()
    detector._vehicles_from_coco = lambda raw_result: ([{"class": "ambulance", "confidence": 0.9, "bbox": [0, 0, 10, 10]}], True)
    detector._merge_ambulance_detections = lambda frame, vehicles, emergency: (vehicles, emergency)
    detector._check_gps_emergency = lambda: {"emergency": True, "vehicle_id": "AMB_1"}
    detector._attach_fusion = lambda out, video_source_hint: out

    out = detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))

    assert out["vision_emergency"] is True
    assert out["gps_emergency"] is True
    assert out["emergency"] is True


if __name__ == "__main__":
    test_merge_preserves_existing_emergency_flag_on_model_failure()
    test_merge_can_set_emergency_when_aux_finds_ambulance()
    test_normalize_emergency_label_supports_fire_services()
    test_detect_reports_gps_and_vision_emergency_separately()
    test_detect_aggregates_vision_and_gps_emergency()
    print("PASS test_detector_logic")
