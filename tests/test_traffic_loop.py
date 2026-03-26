from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic.traffic_loop import build_signal_summary, is_balanced


def test_build_signal_summary_all_green_and_single_green():
    all_green = {"north": "GREEN", "south": "GREEN", "east": "GREEN", "west": "GREEN"}
    one_green = {"north": "GREEN", "south": "RED", "east": "RED", "west": "RED"}

    assert build_signal_summary(all_green) == "ALL_GREEN"
    assert build_signal_summary(one_green) == "north"


def test_is_balanced_uses_gap_threshold():
    assert is_balanced([10.0, 9.2, 2.0, 1.0], gap=1.0) is True
    assert is_balanced([10.0, 6.0, 2.0, 1.0], gap=1.0) is False


if __name__ == "__main__":
    test_build_signal_summary_all_green_and_single_green()
    test_is_balanced_uses_gap_threshold()
    print("PASS test_traffic_loop")
