"""Property-based invariant testing for SignalController under Hypothesis.

Fuzzes the state machine with arbitrary (and sometimes hostile) demand plans:
random per-lane arrival counts per step plus random emergency windows on a
random corridor, driving each of the four controller arms through FalconBridge.
The safety contract must hold on *every* generated plan:
  - never two approaches GREEN simultaneously
  - during emergency CLEARING/RECOVERY no approach is GREEN
  - during emergency PREEMPTION exactly the corridor approach is GREEN
  - invariants survive repeated emergency re-triggering after recovery
"""
from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.strategies import composite

from logic.signal import SignalController
from sumo_demo.harness import FalconBridge, SyntheticSignalSink, make_controller
from sumo_demo.harness.traffic import TrafficSnapshot, TrafficSource

LANES = ("north", "south", "east", "west")


class PlanSource(TrafficSource):
    """Yields a prebuilt plan of (counts, emergency_lane) snapshots."""

    def __init__(self, plan):
        self._plan = list(plan)

    def steps(self, total_steps):
        for step in range(min(total_steps, len(self._plan))):
            counts, emergency_lane = self._plan[step]
            yield TrafficSnapshot(step=step, lane_counts=counts, emergency_lane=emergency_lane)


@composite
def flow_plan(draw):
    """A random, emergency-heavy demand plan for one full simulation run."""
    width = draw(st.integers(60, 240))
    counts = draw(
        st.lists(
            st.fixed_dictionaries({lane: st.integers(0, 30) for lane in LANES}),
            min_size=width,
            max_size=width,
        )
    )
    emergency_active = draw(st.lists(st.booleans(), min_size=width, max_size=width))
    corridor = draw(st.sampled_from(LANES))
    plan = [
        (counts[i], corridor if emergency_active[i] else None)
        for i in range(width)
    ]
    return plan, corridor


def _check_contract(records, corridor):
    assert records, "bridge must produce records"
    for r in records:
        greens = [lane for lane, state in r.signal_state.items() if state == "GREEN"]
        assert len(greens) <= 1, f"two greens at step {r.step}: {r.signal_state}"

        if r.mode == "EMERGENCY" and r.emergency_state in ("CLEARING", "RECOVERY"):
            assert not greens, f"green during {r.emergency_state} at step {r.step}"

        if r.mode == "EMERGENCY" and r.emergency_state == "PREEMPTION":
            assert greens == [corridor], (
                f"preemption must green only the corridor {corridor}, got {greens} "
                f"at step {r.step} (active {r.active_lane})"
            )


@settings(
    max_examples=25,
    deadline=30000,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
@given(plan=flow_plan())
def test_safety_invariants_hold_for_all_controllers(plan):
    plan_data, corridor = plan
    for mode in ("adaptive", "actuated", "fusion", "baseline"):
        sink = SyntheticSignalSink(service_rate=1.0)
        bridge = FalconBridge(source=PlanSource(plan_data), sink=sink, fps=1)
        result = bridge.run(make_controller(mode), total_steps=len(plan_data))
        _check_contract(result.records, corridor)


@settings(
    max_examples=30,
    deadline=30000,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
@given(plan=flow_plan())
def test_controller_can_be_reused_after_each_distinct_emergency(plan):
    """Every emergency window that opens while the controller is nominal must
    reach PREEMPTION and eventually return the controller to normal mode."""
    plan_data, corridor = plan
    controller = SignalController()
    sink = SyntheticSignalSink(service_rate=1.0)
    result = FalconBridge(source=PlanSource(plan_data), sink=sink, fps=1).run(
        controller, total_steps=len(plan_data)
    )
    for hook in result.emergency_hooks:
        assert hook["emergency_corridor_lane"] == corridor
    # Closed hooks always record a preemption time; the trailing hook is only
    # allowed to stay open if the plan ends while an emergency is in flight.
    closed, tail = result.emergency_hooks[:-1], result.emergency_hooks[-1:]
    for hook in closed:
        assert hook["time_to_preemption_s"] is not None
    last_emergency_idx = max((i for i, e in enumerate(plan_data) if e[1] is not None), default=-1)
    if last_emergency_idx >= 0 and last_emergency_idx + 6 <= len(plan_data):
        assert tail[0]["time_to_preemption_s"] is not None
        assert controller.mode == "ADAPTIVE"
