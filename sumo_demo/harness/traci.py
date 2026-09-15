"""TraCI source + sink for the P1-A harness (SUMO backend).

Drop-in for the synthetic `ScenarioTrafficSource` / `SyntheticSignalSink`
when SUMO is installed: the source polls SUMO's per-approach occupancy and
emergency vehicles, and the sink pushes Falcon's signal state back into the
traffic light controller via `setRedYellowGreenState`.

Both types import `traci` lazily so the harness keeps working without SUMO
installed (imports are guarded and degrade to None / empty).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Generator, List, Optional

from .traffic import TrafficSnapshot, TrafficSource
from .sinks import SignalSink
from .metrics import MetricsReport

# --------------------------------------------------------------------------- #
# SUMO runtime discovery (mirrors sumo_demo.real_traci_runner._import_runtime)
# --------------------------------------------------------------------------- #

def _import_runtime():
    try:
        import traci  # type: ignore
        import sumolib  # type: ignore
        del sumolib
        return traci
    except Exception:
        return None


def _sumo_binary(gui: bool = False) -> Optional[Path]:
    from ..sumo_path import ensure_sumo_home

    home = ensure_sumo_home()
    if not home:
        return None
    binary = "sumo-gui.exe" if gui else "sumo.exe"
    candidate = Path(home) / "bin" / binary
    return candidate if candidate.exists() else None


# --------------------------------------------------------------------------- #
# B1 intersection wiring
# --------------------------------------------------------------------------- #

# Approaches keyed like Falcon lanes; value = SUMO lane carrying traffic
# toward the controlled intersection B1 from that direction.
APPROACH_LANES = {
    "north": "B0B1_0",
    "south": "B2B1_0",
    "west": "A1B1_0",
    "east": "C1B1_0",
}

# B1 tlLogic controls 16 links (getControlledLanes order == linkIndex order).
# Each approach owns 4 consecutive link indices (right/straight/left/u-turn):
#   B2B1 -> linkIndex 0-3   (south approach)
#   C1B1 -> linkIndex 4-7   (east approach)
#   B0B1 -> linkIndex 8-11  (north approach)
#   A1B1 -> linkIndex 12-15 (west approach)
LINK_RANGES = {
    "south": (0, 4),
    "east": (4, 8),
    "north": (8, 12),
    "west": (12, 16),
}

_FAULT_LINK_RANGE = (0, 16)


def build_state_string(falcon_state: Dict[str, str], num_links: int = 16) -> str:
    """Translate a Falcon signal_state into a SUMO red/green/yellow state string.

    Only ONE approach may be GREEN at a time (Falcon's invariant), so the
    string is 'G'*same-approach-links when green, all 'y' for the transition
    lane during YELLOW, else 'r' — never two greens in one step.
    """
    if num_links <= 0:
        num_links = 16
    state = ["r"] * num_links

    green_lane = next((l for l, s in falcon_state.items() if s == "GREEN"), None)
    yellow_lane = next((l for l, s in falcon_state.items() if s == "YELLOW"), None)

    if green_lane is not None:
        start, end = LINK_RANGES.get(green_lane, _FAULT_LINK_RANGE)
        for i in range(start, min(end, num_links)):
            state[i] = "G"
    elif yellow_lane is not None:
        start, end = LINK_RANGES.get(yellow_lane, _FAULT_LINK_RANGE)
        for i in range(start, min(end, num_links)):
            state[i] = "y"

    return "".join(state)


def approach_for_lane(lane_id: str) -> Optional[str]:
    """Map a SUMO lane/edge id back to a Falcon approach (B1 grid)."""
    lowered = lane_id.lower()
    for approach, lane in APPROACH_LANES.items():
        if lowered.startswith(lane.lower().split("_")[0]):
            return approach
    return None


# --------------------------------------------------------------------------- #
# TraCI traffic source
# --------------------------------------------------------------------------- #

class TraciTrafficSource(TrafficSource):
    """Reads per-approach occupancy + emergency vehicles from a live SUMO run.

    `steps()` advances the simulation by one step and yields a snapshot whose
    `lane_counts` is the number of vehicles currently on each approach lane
    (`getLastStepVehicleIDs`), and whose `emergency_lane` is the approach an
    configured emergency vehicle is on (or None).
    """

    def __init__(
        self,
        traci,
        tls_id: str = "B1",
        emergency_ids: Optional[List[str]] = None,
    ):
        self.traci = traci
        self.tls_id = tls_id
        self.emergency_ids = emergency_ids or ["ambulance_0"]
        self._num_links = 0

    @property
    def num_links(self) -> int:
        if not self._num_links:
            try:
                self._num_links = len(self.traci.trafficlight.getControlledLanes(self.tls_id))
            except Exception:
                self._num_links = 16
        return self._num_links or 16

    def _lane_counts(self) -> Dict[str, int]:
        counts = {approach: 0 for approach in APPROACH_LANES}
        for approach, lane in APPROACH_LANES.items():
            try:
                counts[approach] = len(self.traci.lane.getLastStepVehicleIDs(lane))
            except Exception:
                pass
        return counts

    def _emergency_lane(self) -> Optional[str]:
        try:
            ids = self.traci.vehicle.getIDList()
        except Exception:
            return None
        for vid in ids:
            if vid in self.emergency_ids:
                try:
                    lane = self.traci.vehicle.getLaneID(vid)
                except Exception:
                    lane = ""
                approach = approach_for_lane(lane or "")
                if approach is not None:
                    return approach
        return None

    def steps(self, total_steps: int) -> Generator[TrafficSnapshot, None, None]:
        traci = self.traci
        for step in range(total_steps):
            try:
                traci.simulationStep()
            except Exception:
                break
            yield TrafficSnapshot(
                step=step,
                lane_counts=self._lane_counts(),
                emergency_lane=self._emergency_lane(),
            )


# --------------------------------------------------------------------------- #
# TraCI signal sink
# --------------------------------------------------------------------------- #

class TraciSignalSink(SignalSink):
    """Applies Falcon's signal state to the SUMO traffic light controller and
    records simple occupancy/wait metrics fed back through `observed_counts()`.

    This is the camera-ROI analogue over SUMO: `observed_counts()` returns the
    number of halting vehicles per approach (what a detector at the stop line
    would report), so the controller consumes real simulated occupancy instead
    of the synthetic queue model.
    """

    def __init__(self, traci, tls_id: str = "B1"):
        self.traci = traci
        self.tls_id = tls_id
        self._nums: Dict[str, float] = {l: 0.0 for l in APPROACH_LANES}
        self._queue_total: Dict[str, float] = {l: 0.0 for l in APPROACH_LANES}
        self._queue_samples = 0
        self._max_queue = 0
        self._served = 0

    def on_step(self, step: int, signal_state: Dict[str, str], arrivals: Dict[str, int]) -> None:
        try:
            num_links = len(self.traci.trafficlight.getControlledLanes(self.tls_id))
        except Exception:
            num_links = 16
        state_str = build_state_string(signal_state, num_links)
        try:
            self.traci.trafficlight.setRedYellowGreenState(self.tls_id, state_str)
        except Exception:
            pass

        total = 0
        for approach, lane in APPROACH_LANES.items():
            try:
                halting = self.traci.lane.getLastStepHaltingNumber(lane)
            except Exception:
                halting = 0
            self._nums[approach] = float(halting)
            self._queue_total[approach] += float(halting)
            total += halting
        self._queue_samples += 1
        if total > self._max_queue:
            self._max_queue = int(total)
        try:
            self._served = int(self.traci.simulation.getEndedNumber())
        except Exception:
            pass

    def observed_counts(self) -> Dict[str, int]:
        """Halting vehicles per approach — what a camera ROI would report."""
        return {l: int(self._nums.get(l, 0)) for l in APPROACH_LANES}

    def report(self, scenario: str, controller: str, total_steps: int, **emergency) -> MetricsReport:
        total_wait = 0.0
        try:
            for vid in self.traci.vehicle.getIDList():
                total_wait += self.traci.vehicle.getAccumulatedWaitingTime(vid)
        except Exception:
            pass
        avg_wait = (total_wait / self._served) if self._served else 0.0
        avg_queue = (
            sum(self._queue_total.values()) / self._queue_samples if self._queue_samples else 0.0
        )
        args = dict(
            scenario=scenario,
            controller=controller,
            total_steps=total_steps,
            avg_wait_s=round(avg_wait, 2),
            max_queue=int(self._max_queue),
            avg_queue=round(avg_queue, 2),
            vehicles_served=int(self._served),
            vehicles_arrived=int(self._served),
        )
        if emergency:
            args.update({k: v for k, v in emergency.items() if v is not None})
        return MetricsReport(**args)