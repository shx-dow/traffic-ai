"""Corner-case emergency matrix for SignalController.

Deterministic edge sequences that stress the preemption contract beyond the
nominal temporal suite: preemption requested mid-transition, corridor equal to
the currently-green lane, re-trigger while recovering, step-0 and final-step
ambulance passes, and two corridors arriving back-to-back.  Every sequence
must preserve the invariant: at most one GREEN, no GREEN during
CLEARING/RECOVERY, and during PREEMPTION exactly the corridor is GREEN.
"""
from __future__ import annotations

import pytest

from logic.signal import SignalController
from sumo_demo.harness import FalconBridge, SyntheticSignalSink
from sumo_demo.harness.traffic import TrafficSnapshot, TrafficSource

LANES = ("north", "south", "east", "west")
CORRIDOR = "south"


def _scripted(plan, total):
    class ScriptedSource(TrafficSource):
        def steps(self, _unused):
            for step in range(min(total, len(plan))):
                counts, emergency_lane = plan[step]
                yield TrafficSnapshot(step=step, lane_counts=counts, emergency_lane=emergency_lane)

    return ScriptedSource()


def _quiet(steps):
    return {lane: 0 for lane in LANES}


def _green_lanes(records, corridor_for_step):
    for r in records:
        greens = [lane for lane, state in r.signal_state.items() if state == "GREEN"]
        if len(greens) > 1:
            pytest.fail(f"two greens at step {r.step}: {r.signal_state}")

        if r.mode == "EMERGENCY" and r.emergency_state in ("CLEARING", "RECOVERY"):
            assert not greens, f"green during {r.emergency_state} at step {r.step}"

        if r.mode == "EMERGENCY" and r.emergency_state == "PREEMPTION":
            assert len(greens) == 1, f"corridor must be exclusively green at step {r.step}"
            assert greens[0] == corridor_for_step(r.step), (
                f"expected corridor {corridor_for_step(r.step)} green, got {greens} at step {r.step}"
            )
    return records


def _corridor_resolver(hooks):
    borders = sorted(h["emergency_start_step"] for h in hooks)
    def resolver(step):
        active = [i for i, b in enumerate(borders) if b <= step]
        return hooks[max(active)]["emergency_corridor_lane"] if active else CORRIDOR
    return resolver


def _run(plan, total):
    sink = SyntheticSignalSink(service_rate=1.0)
    result = FalconBridge(source=_scripted(plan, total), sink=sink, fps=1).run(
        SignalController(), total_steps=total
    )
    return _green_lanes(result.records, _corridor_resolver(result.emergency_hooks)), result.emergency_hooks


def test_preemption_requested_mid_transition():
    """Emergency arrives while a normal GREEN->YELLOW transition is in mid-flight."""
    plan = [(_quiet(t), None) for t in range(8)]
    plan[4] = (_quiet(4), CORRIDOR)  # arrives while phase may still be clearing
    _run(plan, total=60)


def test_preemption_corridor_equals_green_lane():
    """Emergency on the approach that is already green: no conflicting flash."""
    plan = [(_quiet(t), None) for t in range(2)]
    plan.append(({"north": 0, "south": 1, "east": 0, "west": 0}, None))
    plan.append(({"north": 0, "south": 1, "east": 0, "west": 0}, None))
    plan.append(({"north": 0, "south": 1, "east": 0, "west": 0}, CORRIDOR))
    _run(plan, total=40)


def test_emergency_at_step_zero():
    records, hooks = _run([(_quiet(0), CORRIDOR)] + [(_quiet(t), CORRIDOR) for t in range(1, 20)], total=20)
    assert hooks and hooks[-1]["time_to_preemption_s"] is not None


def test_emergency_at_last_step():
    plan = [(_quiet(t), None) for t in range(59)]
    plan.append((_quiet(59), CORRIDOR))
    records, hooks = _run(plan, total=60)
    assert hooks, "final-step emergency must still be recorded by the bridge"
    assert hooks[-1]["emergency_start_step"] == 59


def test_retrigger_during_recovery():
    """A second ambulance while the first event is still recovering is ignored
    by the bridge but must never produce a second green region."""
    plan = [(_quiet(t), None) for t in range(10)]
    plan += [(_quiet(t), CORRIDOR) for t in range(10, 15)]
    plan += [(_quiet(t), None) for t in range(15, 25)]  # recovery window
    plan.append((_quiet(25), CORRIDOR))  # re-trigger
    _run(plan, total=60)


def test_two_corridors_sequential():
    """Two corridors back-to-back each open their own safe preemption event."""
    plan = [(_quiet(t), None) for t in range(2)]
    plan.append((_quiet(2), "east"))
    plan += [(_quiet(t), None) for t in range(3, 20)]
    plan.append((_quiet(20), CORRIDOR))
    sink = SyntheticSignalSink(service_rate=1.0)
    bridge = FalconBridge(source=_scripted(plan, 60), sink=sink, fps=1)
    result = bridge.run(SignalController(), total_steps=60)
    hooks = result.emergency_hooks
    assert len(hooks) == 2
    assert [h["emergency_corridor_lane"] for h in hooks] == ["east", CORRIDOR]
    _green_lanes(result.records, _corridor_resolver(hooks))


def test_controller_level_retrigger_preserves_single_green():
    """Direct: request while PREEMPTION active, for an *different* corridor."""
    ctrl = SignalController()
    ctrl.request_emergency_preemption(corridor_lane="east", fps=1, current_green_lane="north")
    for _ in range(10):
        ctrl.tick(1)
    ctrl.request_emergency_preemption(corridor_lane="west", fps=1, current_green_lane="east")
    for _ in range(8):
        state = ctrl.get_current_signal_state("east")
        greens = [lane for lane, s in state.items() if s == "GREEN"]
        assert len(greens) <= 1
        if ctrl.emergency_state == "PREEMPTION":
            assert greens == ["west"]
        ctrl.tick(1)
