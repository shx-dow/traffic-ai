from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


LANES = ("north", "south", "east", "west")


@dataclass
class SumoStepResult:
    step: int
    mode: str
    active_lane: str
    green_lanes: List[str]
    lane_counts: Dict[str, int]
    lane_scores: Dict[str, float]
    emergency_active: bool
    corridor_lane: str | None = None


@dataclass
class SumoBaselineController:
    green_seconds: int = 20
    lanes: tuple[str, ...] = LANES
    _current_index: int = 0
    _remaining: int = field(default=20, init=False)

    def __post_init__(self) -> None:
        self._remaining = int(self.green_seconds)

    def step(self, lane_counts: Dict[str, int]) -> SumoStepResult:
        del lane_counts
        self._remaining -= 1
        if self._remaining <= 0:
            self._current_index = (self._current_index + 1) % len(self.lanes)
            self._remaining = int(self.green_seconds)
        active = self.lanes[self._current_index]
        return SumoStepResult(
            step=0,
            mode="BASELINE",
            active_lane=active,
            green_lanes=[active],
            lane_counts={lane: 0 for lane in self.lanes},
            lane_scores={lane: 0.0 for lane in self.lanes},
            emergency_active=False,
        )


@dataclass
class SumoAdaptiveController:
    min_green: int = 10
    max_green: int = 60
    balance_gap: float = 2.5
    wait_weight: float = 2.0
    lanes: tuple[str, ...] = LANES
    _current_lane: str = "north"
    _frame_counter: int = 0
    _wait_cycles: Dict[str, int] = field(default_factory=lambda: {lane: 0 for lane in LANES})

    def calculate_scores(self, lane_counts: Dict[str, int]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for lane in self.lanes:
            scores[lane] = float(lane_counts.get(lane, 0)) + (self._wait_cycles[lane] * self.wait_weight)
        return scores

    def _green_target(self, scores: Dict[str, float]) -> int:
        total = sum(scores.values())
        if total <= 0:
            return self.min_green
        share = scores.get(self._current_lane, 0.0) / total
        return max(self.min_green, min(int(self.min_green + share * (self.max_green - self.min_green)), self.max_green))

    def step(self, lane_counts: Dict[str, int]) -> SumoStepResult:
        scores = self.calculate_scores(lane_counts)
        target = self._green_target(scores)
        self._frame_counter += 1

        best_other = max((lane for lane in self.lanes if lane != self._current_lane), key=lambda lane: scores[lane])
        switch = False
        if self._frame_counter >= target:
            switch = True
        elif scores[best_other] >= scores[self._current_lane] + self.balance_gap:
            switch = True

        if switch:
            for lane in self.lanes:
                if lane == self._current_lane:
                    self._wait_cycles[lane] = 0
                elif lane_counts.get(lane, 0) > 0:
                    self._wait_cycles[lane] += 1
                else:
                    self._wait_cycles[lane] = 0
            self._current_lane = best_other
            self._frame_counter = 0

        active = self._current_lane
        return SumoStepResult(
            step=0,
            mode="ADAPTIVE",
            active_lane=active,
            green_lanes=[active],
            lane_counts=dict(lane_counts),
            lane_scores=scores,
            emergency_active=False,
        )
