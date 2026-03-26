from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic.baseline_signal import BaselineSignalController
from logic.signal import SignalController
from logic.simulation import run_signal_simulation


def test_simulation_emergency_activation_and_resume():
    controller = SignalController()
    events = [
        {"lane_counts": {"north": 3, "south": 4, "east": 1, "west": 0}, "emergency": False},
        {"lane_counts": {"north": 2, "south": 6, "east": 0, "west": 1}, "emergency": True, "corridor_lane": "south"},
        {"lane_counts": {"north": 2, "south": 5, "east": 1, "west": 1}, "emergency": True, "corridor_lane": "south"},
        {"lane_counts": {"north": 3, "south": 3, "east": 2, "west": 1}, "emergency": False},
    ]

    history = run_signal_simulation(controller, events, fps=30)

    assert history[1]["mode"] == "EMERGENCY"
    assert history[1]["green_lanes"] == ["south"]
    assert history[2]["mode"] == "EMERGENCY"
    assert history[3]["mode"] == "ADAPTIVE"


def test_simulation_baseline_vs_adaptive_allocates_differently():
    events = []
    for _ in range(200):
        events.append({"lane_counts": {"north": 1, "south": 12, "east": 1, "west": 1}, "emergency": False})

    adaptive = SignalController()
    baseline = BaselineSignalController(green_seconds=20)

    adaptive_history = run_signal_simulation(adaptive, events, fps=30)
    baseline_history = run_signal_simulation(baseline, events, fps=30)

    adaptive_south = sum(1 for frame in adaptive_history if "south" in frame["green_lanes"])
    baseline_south = sum(1 for frame in baseline_history if "south" in frame["green_lanes"])

    assert adaptive_south > baseline_south, (
        f"Expected adaptive to prioritize south more often. adaptive={adaptive_south}, baseline={baseline_south}"
    )


def test_simulation_wait_cycles_only_advance_when_a_phase_completes():
    controller = SignalController()
    events = [
        {"lane_counts": {"north": 1, "south": 5, "east": 0, "west": 0}, "emergency": False}
        for _ in range(10)
    ]

    run_signal_simulation(controller, events, fps=30)

    assert all(value == 0 for value in controller._lane_wait_cycles.values()), controller._lane_wait_cycles


if __name__ == "__main__":
    test_simulation_emergency_activation_and_resume()
    test_simulation_baseline_vs_adaptive_allocates_differently()
    test_simulation_wait_cycles_only_advance_when_a_phase_completes()
    print("PASS test_simulation")
