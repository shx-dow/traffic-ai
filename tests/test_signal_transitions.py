"""Tests for the Phase 2 signal phase machine (YELLOW/ALL_RED clearance,
safe emergency preemption, and recovery transitions).

These complement the existing backward-compatible tests in test_logic.py.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic.signal import SignalController


LANES = ("north", "south", "east", "west")


def _collect_state(ctrl, active_lane, fps=10, max_frames=60):
    """Drive tick() until is_transitioning becomes False and return the collected states."""
    states = []
    for _ in range(max_frames):
        ctrl.tick(fps)
        states.append(dict(ctrl.get_current_signal_state(active_lane)))
        if not ctrl.is_transitioning and ctrl.mode == "ADAPTIVE":
            break
    return states


def _green_lane(state):
    """Return which lane is GREEN or None."""
    for lane, s in state.items():
        if s == "GREEN":
            return lane
    return None


def test_normal_lane_switch_follows_yellow_all_red_sequence():
    ctrl = SignalController()
    ctrl.begin_lane_transition(prev_lane="north", next_lane="south", fps=10)

    states = _collect_state(ctrl, active_lane="south", fps=10, max_frames=60)
    assert any(_green_lane(s) is None for s in states), "Should have an ALL_RED phase"
    assert any(
        _green_lane(s) is None and any(v == "YELLOW" for v in s.values()) for s in states
    ), "Should have a YELLOW phase for the previous lane"
    final = states[-1]
    assert _green_lane(final) == "south", f"Final green should be south, got {final}"


def test_emergency_preemption_clears_before_giving_corridor_green():
    ctrl = SignalController()
    ctrl.request_emergency_preemption("east", fps=10, current_green_lane="north")

    assert ctrl.mode == "EMERGENCY"
    assert ctrl.emergency_state == "CLEARING"
    # First call shows yellow on north
    first = ctrl.get_current_signal_state("north")
    assert first["north"] == "YELLOW"
    assert first["east"] == "RED"

    states = _collect_state(ctrl, active_lane="north", fps=10)
    corridor_green = [_green_lane(s) for s in states]
    assert "east" in corridor_green, "Corridor lane should eventually turn GREEN"
    # Corridor should not appear green on the first tick
    first_green_idx = corridor_green.index("east")
    assert first_green_idx > 0, "Corridor should not turn green immediately"
    # All-green should never happen
    for s in states:
        greens = [l for l, v in s.items() if v == "GREEN"]
        assert len(greens) <= 1, f"Multiple greens in same frame: {s}"


def test_recovery_transitions_back_to_adaptive():
    ctrl = SignalController()
    ctrl.override_for_emergency("south")
    assert ctrl.mode == "EMERGENCY"
    ctrl.begin_recovery(fps=10)

    states = _collect_state(ctrl, active_lane="south", fps=10)
    final_mode = ctrl.mode
    final_green = _green_lane(states[-1])
    assert final_mode == "ADAPTIVE", f"Expected ADAPTIVE after recovery, got {final_mode}"
    assert final_green is not None, "Should be in GREEN phase after recovery"


def test_emergency_max_duration_triggers_auto_recovery():
    ctrl = SignalController()
    ctrl.override_for_emergency("west")
    ctrl.tick(fps=1)
    # Run many ticks until max duration is exceeded
    for _ in range(int(ctrl.EMERGENCY_MAX_DURATION * 1 + 5)):
        ctrl.tick(fps=1)
    assert ctrl.mode == "ADAPTIVE" or ctrl.emergency_state == "RECOVERY", (
        "Should auto-recover after EMERGENCY_MAX_DURATION"
    )


def test_no_two_greens_simultaneously():
    ctrl = SignalController()
    ctrl.begin_lane_transition(prev_lane="north", next_lane="east", fps=10)
    for _ in range(60):
        state = ctrl.get_current_signal_state("east")
        greens = [l for l, s in state.items() if s == "GREEN"]
        assert len(greens) <= 1, f"Multiple greens in same frame: {state}"
        ctrl.tick(10)


def test_baseline_inherits_phase_machine():
    from logic.baseline_signal import BaselineSignalController

    ctrl = BaselineSignalController(green_seconds=20)
    ctrl.begin_lane_transition(prev_lane="north", next_lane="south", fps=10)
    assert ctrl.is_transitioning
    ctrl.tick(fps=10)
    state = ctrl.get_current_signal_state("south")
    assert any(v in ("YELLOW", "RED") for v in state.values())


if __name__ == "__main__":
    test_normal_lane_switch_follows_yellow_all_red_sequence()
    test_emergency_preemption_clears_before_giving_corridor_green()
    test_recovery_transitions_back_to_adaptive()
    test_emergency_max_duration_triggers_auto_recovery()
    test_no_two_greens_simultaneously()
    test_baseline_inherits_phase_machine()
    print("PASS test_signal_transitions")