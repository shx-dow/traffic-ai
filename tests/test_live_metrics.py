from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic.live_metrics import LiveMetricsTracker


def test_live_metrics_accumulates_wait_throughput_and_emergency():
    tracker = LiveMetricsTracker(fps=10)

    tracker.update(
        lane_counts={"north": 4, "south": 2, "east": 0, "west": 0},
        green_lanes=["north"],
        mode="ADAPTIVE",
    )
    tracker.update(
        lane_counts={"north": 0, "south": 3, "east": 1, "west": 0},
        green_lanes=["south"],
        mode="EMERGENCY",
    )

    snap = tracker.snapshot()
    assert snap["frames"] == 2
    assert snap["throughput_score"] == 7
    assert snap["max_queue"] == 4
    assert abs(float(snap["avg_wait_seconds"]) - 0.075) < 1e-9
    assert abs(float(snap["emergency_duration_seconds"]) - 0.1) < 1e-9


def test_live_metrics_rejects_invalid_fps():
    try:
        LiveMetricsTracker(fps=0)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for non-positive fps")


if __name__ == "__main__":
    test_live_metrics_accumulates_wait_throughput_and_emergency()
    test_live_metrics_rejects_invalid_fps()
    print("PASS test_live_metrics")
