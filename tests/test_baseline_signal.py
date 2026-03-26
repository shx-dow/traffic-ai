from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic.baseline_signal import BaselineSignalController


def test_baseline_green_times_are_constant():
    signal = BaselineSignalController(green_seconds=18)
    counts = {"north": 50, "south": 1, "east": 10, "west": 2}
    times = signal.calculate_green_times(counts)

    assert times == {"north": 18, "south": 18, "east": 18, "west": 18}


def test_baseline_switches_after_fixed_time():
    signal = BaselineSignalController(green_seconds=20)
    fps = 30

    before = signal.should_switch_lane("north", {}, frame_counter=(20 * fps) - 1, fps=fps)
    at = signal.should_switch_lane("north", {}, frame_counter=20 * fps, fps=fps)

    assert before is False
    assert at is True


def test_baseline_cycles_in_fixed_order():
    signal = BaselineSignalController(green_seconds=20)
    assert signal.choose_next_lane("north", {}) == "south"
    assert signal.choose_next_lane("south", {}) == "east"
    assert signal.choose_next_lane("east", {}) == "west"
    assert signal.choose_next_lane("west", {}) == "north"


def test_baseline_resumes_to_baseline_after_emergency():
    signal = BaselineSignalController(green_seconds=20)
    signal.override_for_emergency("north")
    signal.resume_adaptive()
    assert signal.mode == "BASELINE"


if __name__ == "__main__":
    test_baseline_green_times_are_constant()
    test_baseline_switches_after_fixed_time()
    test_baseline_cycles_in_fixed_order()
    test_baseline_resumes_to_baseline_after_emergency()
    print("PASS test_baseline_signal")
