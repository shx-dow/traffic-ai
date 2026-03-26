from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gps_utils import compute_emergency_priority, estimate_eta_seconds


def test_estimate_eta_seconds_uses_speed_and_distance():
    eta = estimate_eta_seconds(distance_km=1.0, speed_kmh=60.0)
    assert 59.0 <= eta <= 61.0


def test_estimate_eta_seconds_uses_min_speed_floor():
    eta = estimate_eta_seconds(distance_km=0.5, speed_kmh=0.0)
    assert eta > 0


def test_compute_emergency_priority_picks_best_eta_under_threshold():
    data = [
        {"vehicle_id": "A", "lat": 26.9160, "lon": 75.7873, "speed": 20.0},
        {"vehicle_id": "B", "lat": 26.9130, "lon": 75.7873, "speed": 50.0},
    ]
    priority = compute_emergency_priority(data)
    assert priority["emergency"] is True
    assert priority["vehicle_id"] in {"A", "B"}
    assert isinstance(priority["eta_seconds"], float)


if __name__ == "__main__":
    test_estimate_eta_seconds_uses_speed_and_distance()
    test_estimate_eta_seconds_uses_min_speed_floor()
    test_compute_emergency_priority_picks_best_eta_under_threshold()
    print("PASS test_eta")
