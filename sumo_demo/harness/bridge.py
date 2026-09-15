"""FalconBridge: drives Falcon's SignalController from a TrafficSource and
pushes the resulting signal states into a SignalSink (queue model today, SUMO
later).

This is the single place that owns the *driving loop* — how a controller's
frame-by-frame decisions become signal states — so the same code runs whether
the traffic comes from the synthetic source or, later, a TraCI source.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .traffic import TrafficSource
from .sinks import SignalSink
from .metrics import MetricsReport


@dataclass
class StepRecord:
    step: int
    mode: str
    phase: str
    emergency_state: str
    active_lane: str
    signal_state: Dict[str, str]
    arrivals: Dict[str, int]
    emergency_lane: Optional[str]


@dataclass
class BridgeResult:
    records: List[StepRecord] = field(default_factory=list)
    emergency_hooks: List[Dict[str, int]] = field(default_factory=list)

    def report(self, scenario: str, controller: str, sink: SignalSink, total_steps: int) -> MetricsReport:
        emergency = {}
        if self.emergency_hooks:
            last = self.emergency_hooks[-1]
            emergency = {
                k: v
                for k, v in last.items()
                if k
                in (
                    "time_to_preemption_s",
                    "preemption_duration_s",
                    "corridor_clearance_s",
                    "recovery_time_s",
                    "emergency_corridor_wait_s",
                )
            }
        return sink.report(scenario, controller, total_steps, **emergency)


def make_controller(mode: str, **kwargs):
    """Build a Falcon controller: 'adaptive' (SignalController) or 'baseline'
    (BaselineSignalController, fixed-time)."""
    from logic.signal import SignalController
    from logic.baseline_signal import BaselineSignalController

    if mode == "baseline":
        return BaselineSignalController(**kwargs)
    return SignalController()


class FalconBridge:
    """Run a controller against a traffic source, recording every step."""

    def __init__(
        self,
        source: TrafficSource,
        sink: SignalSink,
        fps: int = 1,
    ):
        self.source = source
        self.sink = sink
        self.fps = fps
        self.emergency_hooks: List[Dict[str, int]] = []

    def run(self, controller, total_steps: int) -> BridgeResult:
        active_lane = "north"
        frame_counter = 0
        pending_lane: Optional[str] = None
        in_transition = False
        emergency_hook_open = False
        preemption_hit = False
        recovery_begin_step: Optional[int] = None

        records: List[StepRecord] = []
        for snap in self.source.steps(total_steps):
            step = snap.step
            lane_counts = snap.lane_counts
            emergency_lane = snap.emergency_lane

            # The controller consumes the *observed* approach occupancy (queue),
            # exactly as it would from a camera ROI in the real pipeline. The
            # synthetic sink exposes measured queues; a TraCI sink would too.
            observed = getattr(self.sink, "observed_counts", None)
            controller_counts = observed() if callable(observed) else lane_counts

            # ---------- emergency entry ----------
            if emergency_lane and controller.mode != "EMERGENCY" and not emergency_hook_open:
                emergency_hook = {
                    "emergency_corridor_lane": emergency_lane,
                    "emergency_start_step": step,
                    "time_to_preemption_s": None,
                    "preemption_duration_s": None,
                    "corridor_clearance_s": None,
                    "recovery_time_s": None,
                }
                emergency_hook_open = True
                preemption_hit = False
                recovery_begin_step = None
                controller.request_emergency_preemption(
                    corridor_lane=emergency_lane,
                    fps=self.fps,
                    current_green_lane=active_lane,
                )
                pending_lane = emergency_lane
                in_transition = True

            # ---------- emergency exit (ambulance passed) ----------
            if not emergency_lane and controller.mode == "EMERGENCY" and controller.emergency_state == "PREEMPTION":
                if recovery_begin_step is None:
                    recovery_begin_step = step
                controller.begin_recovery(fps=self.fps)
                in_transition = True

            # ---------- drive the controller ----------
            if controller.mode == "EMERGENCY":
                controller.tick(self.fps)

                if emergency_hook_open:
                    corridor = emergency_hook["emergency_corridor_lane"]
                    if (
                        not preemption_hit
                        and corridor
                        and controller.emergency_state == "PREEMPTION"
                        and _corridor_green(controller, corridor)
                    ):
                        emergency_hook["time_to_preemption_s"] = step - emergency_hook["emergency_start_step"]
                        emergency_hook["corridor_clearance_s"] = emergency_hook["time_to_preemption_s"]
                        preemption_hit = True

                    if preemption_hit and emergency_lane is None:
                        emergency_hook["preemption_duration_s"] = step - (
                            emergency_hook["emergency_start_step"] + emergency_hook["corridor_clearance_s"]
                        )

                    if controller.mode != "EMERGENCY":
                        if recovery_begin_step is not None:
                            emergency_hook["recovery_time_s"] = step - recovery_begin_step
                        self.emergency_hooks.append(emergency_hook)
                        emergency_hook_open = False
                        in_transition = False
                        pending_lane = None

            elif in_transition:
                controller.tick(self.fps)
                if controller.mode != "EMERGENCY" and not controller.is_transitioning:
                    active_lane = pending_lane or active_lane
                    pending_lane = None
                    frame_counter = 0
                    in_transition = False
            else:
                pending = pending_lane
                should_switch = controller.should_switch_lane(
                    active_lane=active_lane,
                    lane_counts=controller_counts,
                    frame_counter=frame_counter,
                    fps=self.fps,
                )
                if pending is None and should_switch:
                    controller.record_cycle(active_lane, controller_counts)
                    next_lane = controller.choose_next_lane(active_lane, controller_counts)
                    if next_lane != active_lane:
                        controller.begin_lane_transition(active_lane, next_lane, fps=self.fps)
                        pending_lane = next_lane
                        in_transition = True
                    else:
                        frame_counter = 0
                else:
                    frame_counter += 1

            signal_state = dict(controller.get_current_signal_state(active_lane))

            records.append(
                StepRecord(
                    step=step,
                    mode=controller.mode,
                    phase=controller.phase,
                    emergency_state=controller.emergency_state,
                    active_lane=active_lane,
                    signal_state=signal_state,
                    arrivals=dict(lane_counts),
                    emergency_lane=emergency_lane,
                )
            )
            self.sink.on_step(step, signal_state, lane_counts)

        if emergency_hook_open:
            self.emergency_hooks.append(emergency_hook)

        return BridgeResult(records=records, emergency_hooks=self.emergency_hooks)


def _corridor_green(controller, corridor: str) -> bool:
    state = controller.get_current_signal_state(corridor)
    return state.get(corridor) == "GREEN"