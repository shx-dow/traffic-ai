# tests/test_logic.py
# Run with: python tests/test_logic.py

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic.counter import LaneCounter
from logic.signal  import SignalController

def test_count_per_lane_basic():
    """A single vehicle must be counted exactly once in exactly one lane."""
    counter = LaneCounter(frame_width=1280, frame_height=720)
    vehicles = [{'class': 'car', 'confidence': 0.91,
                 'bbox': [100, 50, 200, 150]}]
    counts = counter.count_per_lane(vehicles)

    # Exactly 1 vehicle counted in total
    assert sum(counts.values()) == 1, \
        f"Expected total=1, got {sum(counts.values())}"

    # All 4 keys always present
    assert all(k in counts for k in ['north', 'south', 'east', 'west']), \
        "Missing lane keys in output"

    # Values are integers
    assert all(isinstance(v, int) for v in counts.values()), \
        "Count values must be integers"

    print("PASS test_count_per_lane_basic")


def test_green_times_proportional():
    """Busiest lane must always get the most green time."""
    signal = SignalController()
    counts = {'north': 20, 'south': 5, 'east': 10, 'west': 2}
    times  = signal.calculate_green_times(counts)

    assert times['north'] > times['east'], \
        "North (20 cars) must beat East (10 cars)"
    assert times['east']  > times['south'], \
        "East (10 cars) must beat South (5 cars)"
    assert times['south'] > times['west'], \
        "South (5 cars) must beat West (2 cars)"

    print("PASS test_green_times_proportional")


def test_green_times_clamped():
    """No lane must ever go below MIN or above MAX green time."""
    signal = SignalController()
    counts = {'north': 20, 'south': 5, 'east': 10, 'west': 2}
    times  = signal.calculate_green_times(counts)

    for lane, t in times.items():
        assert t >= signal.MIN_GREEN, \
            f"{lane}: {t}s is below MIN_GREEN ({signal.MIN_GREEN})"
        assert t <= signal.MAX_GREEN, \
            f"{lane}: {t}s exceeds MAX_GREEN ({signal.MAX_GREEN})"
        assert isinstance(t, int), \
            f"{lane}: green time must be int, got {type(t)}"

    print("PASS test_green_times_clamped")


def test_zero_traffic_no_crash():
    """Zero vehicles must not cause ZeroDivisionError and must return MIN for all lanes."""
    signal = SignalController()
    counts = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
    times  = signal.calculate_green_times(counts)   # must not crash

    assert all(t == signal.MIN_GREEN for t in times.values()), \
        f"Expected all MIN_GREEN ({signal.MIN_GREEN}), got {times}"

    print("PASS test_zero_traffic_no_crash")


def test_emergency_override_and_resume():
    """Emergency override must set mode, force one GREEN, and resume cleanly."""
    signal = SignalController()

    # Activate emergency
    state = signal.override_for_emergency('north')

    assert signal.mode    == 'EMERGENCY', \
        f"Expected mode=EMERGENCY, got {signal.mode}"
    assert state['north'] == 'GREEN', \
        "Corridor lane must be GREEN"
    assert state['south'] == 'RED', \
        "Non-corridor lane must be RED"
    assert state['east']  == 'RED', \
        "Non-corridor lane must be RED"
    assert state['west']  == 'RED', \
        "Non-corridor lane must be RED"

    # Resume adaptive
    signal.resume_adaptive()
    assert signal.mode == 'ADAPTIVE', \
        f"Expected mode=ADAPTIVE after resume, got {signal.mode}"

    print("PASS test_emergency_override_and_resume")


if __name__ == '__main__':
    print("Running Issue #2 unit tests...\n")
    test_count_per_lane_basic()
    test_green_times_proportional()
    test_green_times_clamped()
    test_zero_traffic_no_crash()
    test_emergency_override_and_resume()
    print("\nAll 5 tests passed. Issue #2 is ready for integration.")
