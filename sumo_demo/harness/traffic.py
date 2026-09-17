"""Traffic sources for the P1-A harness.

The harness drives Falcon's SignalController from a TrafficSource.  Today the
source is synthetic (Poisson arrivals per approach); the design mirrors SUMO's
TraCI interface so a `sumo_demo.harness.traci` source can be dropped in later
without touching the bridge or the controller.
"""
from __future__ import annotations

import math
import random
from collections.abc import Generator
from dataclasses import dataclass

LANES = ("north", "south", "east", "west")


@dataclass(frozen=True)
class TrafficSnapshot:
    """One simulation step of external traffic state.

    `lane_counts` is arrivals during this step per approach lane (SUMO's
    `lane.getLastStepVehicleIDs` semantics).  `emergency_lane` marks an
    approaching emergency vehicle on that approach (SUMO: a vehicle running an
    ambulance route), or None.
    """
    step: int
    lane_counts: dict[str, int]
    emergency_lane: str | None = None


class TrafficSource:
    """Abstract traffic source.  `steps()` yields one snapshot per second."""

    def steps(self, total_steps: int) -> Generator[TrafficSnapshot, None, None]:
        raise NotImplementedError


@dataclass(frozen=True)
class Scenario:
    """Named scenario = per-approach demand (veh/hour) + optional events.

    arrivals follow a Poisson process; `surge_window` scales all flows during a
    window (e.g. 3x) to model a surge; `emergency_window` places an ambulance on
    `emergency_lane`.
    """
    name: str
    flows_per_hour: dict[str, float]
    surge_window: tuple[int, int] | None = None
    surge_multiplier: float = 3.0
    emergency_lane: str | None = None
    emergency_window: tuple[int, int] | None = None


SCENARIOS = {
    "low": Scenario(
        name="low",
        flows_per_hour={"north": 200, "south": 180, "east": 160, "west": 140},
    ),
    "medium": Scenario(
        name="medium",
        flows_per_hour={"north": 400, "south": 360, "east": 300, "west": 260},
    ),
    "heavy": Scenario(
        name="heavy",
        flows_per_hour={"north": 720, "south": 650, "east": 540, "west": 420},
    ),
    "surge": Scenario(
        name="surge",
        flows_per_hour={"north": 400, "south": 360, "east": 300, "west": 260},
        surge_window=(240, 420),
        surge_multiplier=3.0,
    ),
    "emergency": Scenario(
        name="emergency",
        flows_per_hour={"north": 400, "south": 360, "east": 300, "west": 260},
        emergency_lane="south",
        emergency_window=(300, 420),
    ),
}


def _poisson(rng, rate_per_step: float) -> int:
    """One Poisson sample for arrivals during a single 1 second step."""
    if rate_per_step <= 0.0:
        return 0
    threshold = math.exp(-rate_per_step)
    k, product = 0, 1.0
    while True:
        product *= rng.random()
        if product <= threshold:
            return k
        k += 1


class ScenarioTrafficSource(TrafficSource):
    """Deterministic-seeded Poisson traffic generator for a Scenario."""

    def __init__(self, scenario: Scenario, seed: int = 42, step_duration: float = 1.0):
        self.scenario = scenario
        self.seed = seed
        self.step_duration = step_duration

    def steps(self, total_steps: int) -> Generator[TrafficSnapshot, None, None]:
        rng = random.Random(self.seed)
        per_second = {
            lane: flow / 3600.0 * self.step_duration
            for lane, flow in self.scenario.flows_per_hour.items()
        }
        for step in range(total_steps):
            surge = 1.0
            if self.scenario.surge_window:
                start, end = self.scenario.surge_window
                if start <= step < end:
                    surge = self.scenario.surge_multiplier
            counts: dict[str, int] = {}
            for lane in LANES:
                counts[lane] = _poisson(rng, per_second[lane] * surge)

            emergency_lane = None
            if self.scenario.emergency_window:
                start, end = self.scenario.emergency_window
                if start <= step < end:
                    emergency_lane = self.scenario.emergency_lane
            yield TrafficSnapshot(
                step=step,
                lane_counts=counts,
                emergency_lane=emergency_lane,
            )
