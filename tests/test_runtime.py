from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic.counter import LaneCounter
from logic.runtime import select_corridor_lane


def test_select_corridor_lane_prefers_ambulance_lane():
    counter = LaneCounter(frame_width=1280, frame_height=720)
    vehicles = [
        {"class": "car", "confidence": 0.9, "bbox": [620, 560, 700, 640]},
        {"class": "ambulance", "confidence": 0.8, "bbox": [1000, 330, 1080, 390]},
    ]
    lane_counts = {"north": 4, "south": 6, "east": 2, "west": 7}

    lane = select_corridor_lane(
        vehicles=vehicles,
        lane_counts=lane_counts,
        lane_counter=counter,
        fallback_lane="north",
    )

    assert lane == "east", f"Expected ambulance lane east, got {lane}"


def test_select_corridor_lane_falls_back_to_density():
    counter = LaneCounter(frame_width=1280, frame_height=720)
    vehicles = [
        {"class": "car", "confidence": 0.8, "bbox": [620, 560, 700, 640]},
    ]
    lane_counts = {"north": 1, "south": 5, "east": 2, "west": 3}

    lane = select_corridor_lane(
        vehicles=vehicles,
        lane_counts=lane_counts,
        lane_counter=counter,
        fallback_lane="west",
    )

    assert lane == "south", f"Expected densest lane south, got {lane}"


def test_select_corridor_lane_uses_fallback_when_no_data():
    counter = LaneCounter(frame_width=1280, frame_height=720)

    lane = select_corridor_lane(
        vehicles=[],
        lane_counts={},
        lane_counter=counter,
        fallback_lane="west",
    )

    assert lane == "west", f"Expected fallback lane west, got {lane}"


if __name__ == "__main__":
    test_select_corridor_lane_prefers_ambulance_lane()
    test_select_corridor_lane_falls_back_to_density()
    test_select_corridor_lane_uses_fallback_when_no_data()
    print("PASS test_runtime")
