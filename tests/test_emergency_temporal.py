"""P1-C: emergency preemption temporal contract tests.

Assert the end-to-end timing a real corridor must guarantee, driven through the
FalconBridge against the deterministic synthetic emergency scenario:

  - preemption is PROMPT: corridor GREEN within YELLOW_TIME + ALL_RED_TIME
    (never immediate, never absent) after the ambulance first appears
  - corridor clearance equals prompt preemption (green well before arrival)
  - safety: during PREEMPTION only the corridor lane is GREEN; during
    CLEARING/RECOVERY no lane is GREEN; no step ever has two greens
  - bounded preemption (EMERGENCY_MAX_DURATION auto-recovery)
  - recovery: controller returns to ADAPTIVE and recovery is fast
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic.signal import SignalController
from sumo_demo.harness.bridge import FalconBridge, StepRecord, make_controller
from sumo_demo.harness.sinks import NullSignalSink
from sumo_demo.harness.traffic import Scenario, ScenarioTrafficSource

LANES = ("north", "south", "east", "west")

# Controller timings under test (logic/signal.py).
YELLOW_TIME = 3.0
ALL_RED_TIME = 2.0
EMERGENCY_MAX_DURATION = 30.0
RECOVERY_DURATION = YELLOW_TIME + ALL_RED_TIME

# A scenario where the ambulance appears early so tests run in far fewer steps.
_TEMPORAL_EMERGENCY = Scenario(
    name="temporal_emergency",
    flows_per_hour={"north": 400, "south": 360, "east": 300, "west": 260},
    emergency_lane="south",
    emergency_window=(60, 90),
)


def _run(scenario: Scenario = _TEMPORAL_EMERGENCY, seed: int = 7, steps: int = 160):
    bridge = FalconBridge(
        source=ScenarioTrafficSource(scenario, seed=seed),
        sink=NullSignalSink(),
        fps=1,
    )
    return bridge.run(make_controller("adaptive"), total_steps=steps)


def _greens(rec: StepRecord) -> List[str]:
    return [l for l in LANES if rec.signal_state.get(l) == "GREEN"]


def _emergency_blocks(records: List[StepRecord]) -> List[Tuple[int, int, str, str]]:
    """Collapse consecutive EMERGENCY steps into (start, end, emergency_state, corridor)."""
    blocks: List[Tuple[int, int, str, str]] = []
    for rec in records:
        if rec.mode != "EMERGENCY":
            continue
        corridor = rec.signal_state.get("south")  # only meaningful per-run scenario selection
        if blocks and blocks[-1][1] + 1 == rec.step:
            start, _end, state, _corr = blocks[-1]
            blocks[-1] = (start, rec.step, state, _corr)
        else:
            blocks.append((rec.step, rec.step, rec.emergency_state, corridor))
    return blocks


def test_preemption_is_prompt_but_not_immediate():
    result = _run()
    hooks = result.emergency_hooks
    assert hooks, "expected at least one emergency hook"
    hook = hooks[0]
    ttp = hook["time_to_preemption_s"]
    # Prompt: corridor green after current lane is cleared through
    # YELLOW_TIME + ALL_RED_TIME, never instant.
    assert ttp is not None and 0 < ttp <= YELLOW_TIME + ALL_RED_TIME, \
        f"expected prompt preemption, got {ttp}"
    # Not immediate: must clear current green first.
    assert ttp >= ALL_RED_TIME, f"preemption must clear via YELLOW+ALL_RED, got {ttp}"


def test_corridor_clearance_matches_preemption():
    result = _run()
    hook = result.emergency_hooks[0]
    assert hook["corridor_clearance_s"] == hook["time_to_preemption_s"]
    # Clearance (green before arrival) must be at least the prompt window.
    assert hook["corridor_clearance_s"] >= ALL_RED_TIME


def test_preemption_bounded_by_max_duration():
    result = _run()
    for hook in result.emergency_hooks:
        if hook["preemption_duration_s"] is not None:
            assert hook["preemption_duration_s"] <= EMERGENCY_MAX_DURATION, \
                f"preemption exceeded max duration: {hook}"


def test_only_corridor_green_during_preemption():
    result = _run()
    corridor = _TEMPORAL_EMERGENCY.emergency_lane
    for rec in result.records:
        if rec.mode == "EMERGENCY" and rec.emergency_state == "PREEMPTION":
            greens = _greens(rec)
            assert greens == [corridor], \
                f"step {rec.step}: expected only {corridor} green, got {greens}"


def test_no_green_during_clearing_and_recovery():
    result = _run()
    for rec in result.records:
        if rec.mode == "EMERGENCY" and rec.emergency_state in ("CLEARING", "RECOVERY"):
            assert _greens(rec) == [], \
                f"step {rec.step} ({rec.emergency_state}): lane must be red during clearance, got {_greens(rec)}"


def test_never_two_greens():
    result = _run()
    for rec in result.records:
        assert len(_greens(rec)) <= 1, f"step {rec.step}: simultaneous greens {_greens(rec)}"


def test_recovers_to_adaptive_and_corridor_invariant():
    result = _run()
    assert any(rec.mode == "EMERGENCY" for rec in result.records)
    hook = result.emergency_hooks[-1]
    assert hook["recovery_time_s"] is not None
    assert hook["recovery_time_s"] <= RECOVERY_DURATION, \
        f"recovery should be fast, got {hook['recovery_time_s']}s"


def test_corridor_lane_matches_emergency_scenario():
    result = _run()
    for rec in result.records:
        if rec.mode == "EMERGENCY" and rec.emergency_state == "PREEMPTION":
            assert rec.emergency_lane == _TEMPORAL_EMERGENCY.emergency_lane


def test_no_emergency_without_scenario():
    no_emergency = ScenarioTrafficSource(Scenario(
        name="medium_no_emergency",
        flows_per_hour={"north": 400, "south": 360, "east": 300, "west": 260},
    ), seed=7)
    bridge = FalconBridge(source=no_emergency, sink=NullSignalSink(), fps=1)
    result = bridge.run(make_controller("adaptive"), total_steps=200)
    assert result.emergency_hooks == []
    assert all(rec.mode != "EMERGENCY" for rec in result.records)


def test_baseline_resumes_mode_after_recovery():
    """Regression: after emergency recovery a baseline controller must return
    to BASELINE mode (not silently become ADAPTIVE)."""
    bridge = FalconBridge(
        source=ScenarioTrafficSource(_TEMPORAL_EMERGENCY, seed=3),
        sink=NullSignalSink(),
        fps=1,
    )
    result = bridge.run(make_controller("baseline", green_seconds=10), total_steps=200)
    post_recovery = [r for r in result.records if r.step >= 100 and r.mode != "EMERGENCY"]
    assert post_recovery, "no post-recovery records seen"
    assert all(r.mode == "BASELINE" for r in post_recovery), \
        f"baseline must stay BASELINE, got modes {set(r.mode for r in post_recovery)}"


def test_baseline_resumes_rotation_after_recovery():
    """Regression: after emergency recovery the baseline controller must keep
    cycling through lanes instead of being pinned on one lane forever."""
    bridge = FalconBridge(
        source=ScenarioTrafficSource(_TEMPORAL_EMERGENCY, seed=3),
        sink=NullSignalSink(),
        fps=1,
    )
    result = bridge.run(make_controller("baseline", green_seconds=10), total_steps=300)
    post_recovery = [r for r in result.records if r.step >= 160 and r.mode == "BASELINE"]
    greens = [
        [l for l in LANES if r.signal_state.get(l) == "GREEN"]
        for r in post_recovery
    ]
    unique_greens = {g[0] for g in greens if g}
    assert len(unique_greens) >= 3, \
        f"baseline must rotate after recovery, only saw lanes {unique_greens}"


if __name__ == "__main__":
    test_preemption_is_prompt_but_not_immediate()
    test_corridor_clearance_matches_preemption()
    test_preemption_bounded_by_max_duration()
    test_only_corridor_green_during_preemption()
    test_no_green_during_clearing_and_recovery()
    test_never_two_greens()
    test_recovers_to_adaptive_and_corridor_invariant()
    test_corridor_lane_matches_emergency_scenario()
    test_no_emergency_without_scenario()
    test_baseline_resumes_mode_after_recovery()
    test_baseline_resumes_rotation_after_recovery()
    print("PASS test_emergency_temporal")