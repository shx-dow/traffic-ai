"""Traffic sources for the P1-A harness.

The harness drives Falcon's SignalController from a TrafficSource.  Today the
source is synthetic (Poisson arrivals per approach; an optional platoon/burst
demand model for correlation; and a log-normal headway renewal model that
clusters arrivals like empirical arterial platoons); the design mirrors SUMO's
TraCI interface so a `sumo_demo.harness.traci` source can be dropped in later
without touching the bridge or the controller.
"""
from __future__ import annotations

import math
import random
from collections.abc import Generator
from dataclasses import dataclass, replace

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


def emergency_corridor_variant(corridor: str) -> Scenario:
    """The emergency scenario with the ambulance corridor moved to `corridor`."""
    base = SCENARIOS["emergency"]
    if corridor not in LANES:
        raise ValueError(f"corridor must be one of {LANES}, got {corridor!r}")
    return replace(base, emergency_lane=corridor)


class ScenarioTrafficSource(TrafficSource):
    """Deterministic-seeded traffic generator for a Scenario.

    `demand_model` "poisson" yields independent Poisson arrivals per approach;
    "platoon" yields correlated bursts: each approach cycles through an active
    window (`platoon_on_s`) at elevated rate followed by an idle gap
    (`platoon_off_s`), with the peak rate scaled so the mean arrival rate equals
    the scenario flow.  Per-approach phase offsets decorrelate the bursts.
    "lognormal" draws inter-arrival headways from a log-normal renewal process:
    the symmetric shape of the log-normal concentrates mass on short headways
    with a long tail of long gaps, reproducing the platoon clusters empirical
    studies find at signalized approaches while preserving the scenario mean
    arrival rate.  Headways are floored at `min_headway_s`.
    """

    def __init__(
        self,
        scenario: Scenario,
        seed: int = 42,
        step_duration: float = 1.0,
        demand_model: str = "poisson",
        platoon_on_s: int = 8,
        platoon_off_s: int = 12,
        headway_cv: float = 2.0,
        min_headway_s: float = 0.4,
    ):
        if demand_model not in ("poisson", "platoon", "lognormal"):
            raise ValueError(f"unknown demand_model: {demand_model!r}")
        if headway_cv <= 0.0:
            raise ValueError(f"headway_cv must be positive, got {headway_cv!r}")
        if min_headway_s <= 0.0:
            raise ValueError(f"min_headway_s must be positive, got {min_headway_s!r}")
        self.scenario = scenario
        self.seed = seed
        self.step_duration = step_duration
        self.demand_model = demand_model
        self.platoon_on_s = max(1, int(platoon_on_s))
        self.platoon_off_s = max(1, int(platoon_off_s))
        self.headway_cv = headway_cv
        self.min_headway_s = min_headway_s

    def _arrivals(self, rng, per_second: dict[str, float], step: int) -> dict[str, int]:
        if self.demand_model == "poisson":
            return {lane: _poisson(rng, per_second[lane]) for lane in LANES}
        counts: dict[str, int] = {}
        period = self.platoon_on_s + self.platoon_off_s
        for lane in LANES:
            in_window = (step + rng.randint(0, period - 1)) % period < self.platoon_on_s
            if not in_window:
                counts[lane] = 0
                continue
            multiplier = period / self.platoon_on_s
            counts[lane] = _poisson(rng, per_second[lane] * multiplier)
        return counts

    def _lognormal_next_at(self, rng, rate_per_step: float) -> float:
        """One log-normal inter-arrival headway (seconds) for a renewal process.

        Parameterized by coefficient of variation so the mean headway equals
        `1 / rate_per_step`; floored at `min_headway_s`.
        """
        mean_h = 1.0 / rate_per_step
        sigma2 = math.log(1.0 + self.headway_cv ** 2)
        sigma = math.sqrt(sigma2)
        mu = math.log(mean_h) - sigma2 / 2.0
        return max(self.min_headway_s, math.exp(rng.gauss(mu, sigma)))

    def steps(self, total_steps: int) -> Generator[TrafficSnapshot, None, None]:
        rng = random.Random(self.seed)
        per_second = {
            lane: flow / 3600.0 * self.step_duration
            for lane, flow in self.scenario.flows_per_hour.items()
        }
        # Renewal state (next scheduled arrival time per approach, in steps).
        next_at: dict[str, float] = {}
        if self.demand_model == "lognormal":
            for lane in LANES:
                rate = per_second[lane]
                mean_h = 1.0 / rate if rate > 0.0 else float("inf")
                next_at[lane] = rng.random() * mean_h if math.isfinite(mean_h) else 0.0
        for step in range(total_steps):
            surge = 1.0
            if self.scenario.surge_window:
                start, end = self.scenario.surge_window
                if start <= step < end:
                    surge = self.scenario.surge_multiplier
            if self.demand_model == "poisson":
                counts = {
                    lane: _poisson(rng, per_second[lane] * surge) for lane in LANES
                }
            elif self.demand_model == "platoon":
                per_second_surge = {lane: v * surge for lane, v in per_second.items()}
                counts = self._arrivals(rng, per_second_surge, step)
            else:
                counts = {}
                for lane in LANES:
                    rate = per_second[lane] * surge
                    if rate <= 0.0:
                        counts[lane] = 0
                        continue
                    t = next_at[lane]
                    n = 0
                    while t < step + 1.0:
                        n += 1
                        t += self._lognormal_next_at(rng, rate)
                    next_at[lane] = t
                    counts[lane] = n

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
