from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic.metrics import compare_metrics, compute_metrics
from scripts.run_benchmark import build_scenario
from logic.baseline_signal import BaselineSignalController
from logic.signal import SignalController
from logic.simulation import run_signal_simulation


def test_compute_metrics_basic_fields():
    history = [
        {
            "mode": "ADAPTIVE",
            "green_lanes": ["north"],
            "lane_counts": {"north": 5, "south": 2, "east": 0, "west": 0},
        },
        {
            "mode": "EMERGENCY",
            "green_lanes": ["south"],
            "lane_counts": {"north": 3, "south": 4, "east": 1, "west": 0},
        },
    ]
    metrics = compute_metrics(history, fps=2)

    assert metrics["frames"] == 2
    assert metrics["fps"] == 2
    assert metrics["throughput_score"] >= 1
    assert metrics["emergency_duration_seconds"] == 0.5
    assert "north" in metrics["avg_wait_seconds_per_lane"]


def test_compare_metrics_reduction_and_gain():
    baseline = {"avg_wait_seconds_overall": 10.0, "throughput_score": 100.0}
    adaptive = {"avg_wait_seconds_overall": 8.0, "throughput_score": 120.0}
    cmp = compare_metrics(baseline, adaptive)

    assert round(cmp["wait_reduction_pct"], 2) == 20.0
    assert round(cmp["throughput_gain_pct"], 2) == 20.0


def test_benchmark_scenario_shows_adaptive_gain():
    scenario = build_scenario(1800)
    baseline = BaselineSignalController(green_seconds=20)
    adaptive = SignalController()

    baseline_metrics = compute_metrics(run_signal_simulation(baseline, scenario, fps=30), fps=30)
    adaptive_metrics = compute_metrics(run_signal_simulation(adaptive, scenario, fps=30), fps=30)
    cmp = compare_metrics(baseline_metrics, adaptive_metrics)

    assert cmp["wait_reduction_pct"] > 0.0
    assert cmp["throughput_gain_pct"] > 0.0


if __name__ == "__main__":
    test_compute_metrics_basic_fields()
    test_compare_metrics_reduction_and_gain()
    test_benchmark_scenario_shows_adaptive_gain()
    print("PASS test_metrics")
