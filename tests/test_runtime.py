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

    lane, source = select_corridor_lane(
        vehicles=vehicles,
        lane_counts=lane_counts,
        lane_counter=counter,
        fallback_lane="north",
    )

    assert lane == "east", f"Expected ambulance lane east, got {lane}"
    assert source == "vision_bbox"


def test_select_corridor_lane_supports_fire_truck_lane_selection():
    counter = LaneCounter(frame_width=1280, frame_height=720)
    vehicles = [
        {"class": "fire_truck", "confidence": 0.8, "bbox": [620, 80, 700, 160]},
    ]
    lane_counts = {"north": 1, "south": 5, "east": 2, "west": 3}

    lane, source = select_corridor_lane(
        vehicles=vehicles,
        lane_counts=lane_counts,
        lane_counter=counter,
        fallback_lane="west",
    )

    assert lane == "north", f"Expected fire truck lane north, got {lane}"
    assert source == "vision_bbox"


def test_select_corridor_lane_uses_fallback_when_no_data():
    counter = LaneCounter(frame_width=1280, frame_height=720)

    lane, source = select_corridor_lane(
        vehicles=[],
        lane_counts={},
        lane_counter=counter,
        fallback_lane="west",
    )

    assert lane == "west", f"Expected fallback lane west, got {lane}"
    assert source == "assumed_fallback"


def test_select_corridor_lane_prefers_last_corridor_when_ambulance_missing():
    counter = LaneCounter(frame_width=1280, frame_height=720)
    lane_counts = {"north": 1, "south": 7, "east": 2, "west": 3}

    lane, source = select_corridor_lane(
        vehicles=[{"class": "car", "confidence": 0.9, "bbox": [620, 560, 700, 640]}],
        lane_counts=lane_counts,
        lane_counter=counter,
        fallback_lane="north",
        last_corridor_lane="east",
    )

    assert lane == "east", f"Expected sticky last corridor east, got {lane}"
    assert source == "sticky_last"


if __name__ == "__main__":
    test_select_corridor_lane_prefers_ambulance_lane()
    test_select_corridor_lane_supports_fire_truck_lane_selection()
    test_select_corridor_lane_uses_fallback_when_no_data()
    test_select_corridor_lane_prefers_last_corridor_when_ambulance_missing()
    print("PASS test_runtime")
