from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic.emergency import apply_emergency_override
from logic.signal import SignalController


def test_congestion_scores_include_waiting_factor():
    ctrl = SignalController()
    ctrl._lane_wait_cycles["north"] = 3
    scores = ctrl.calculate_congestion_scores({"north": 2, "south": 1, "east": 0, "west": 0})
    assert scores["north"] > scores["south"]


def test_score_based_next_lane_prefers_high_congestion():
    ctrl = SignalController()
    ctrl._lane_wait_cycles["south"] = 1
    next_lane = ctrl.choose_next_lane("north", {"north": 1, "south": 5, "east": 1, "west": 1})
    assert next_lane == "south"


def test_emergency_corridor_override_is_lane_specific():
    state = apply_emergency_override({"north": "GREEN", "south": "RED", "east": "RED", "west": "RED"}, True, "south")
    assert state["south"] == "GREEN"
    assert state["north"] == "RED"


if __name__ == "__main__":
    test_congestion_scores_include_waiting_factor()
    test_score_based_next_lane_prefers_high_congestion()
    test_emergency_corridor_override_is_lane_specific()
    print("PASS test_congestion")
