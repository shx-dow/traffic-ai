from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class IntersectionPlan:
    mode: str
    corridor_lane: Optional[str]
    hold_frames: int


class CorridorOrchestrator:
    def __init__(
        self,
        intersection_ids: List[str],
        *,
        preempt_hops: int = 2,
        latch_frames: int = 45,
    ) -> None:
        if not intersection_ids:
            raise ValueError("intersection_ids must not be empty")
        if preempt_hops < 0:
            raise ValueError("preempt_hops must be non-negative")
        if latch_frames <= 0:
            raise ValueError("latch_frames must be positive")

        self.intersection_ids = list(intersection_ids)
        self.preempt_hops = int(preempt_hops)
        self.latch_frames = int(latch_frames)
        self._hold_frames: Dict[str, int] = {node: 0 for node in self.intersection_ids}
        self._corridor_lane: Dict[str, Optional[str]] = {node: None for node in self.intersection_ids}

    def update(
        self,
        *,
        route: Optional[List[str]],
        position_index: Optional[int],
        ambulance_lane: Optional[str],
    ) -> Dict[str, IntersectionPlan]:
        targets = set()
        if route is not None and position_index is not None:
            start = max(0, int(position_index))
            end = min(len(route), start + self.preempt_hops + 1)
            targets = set(route[start:end])

        for node in self.intersection_ids:
            if node in targets:
                self._hold_frames[node] = self.latch_frames
                if ambulance_lane:
                    self._corridor_lane[node] = ambulance_lane
                continue

            if self._hold_frames[node] > 0:
                self._hold_frames[node] -= 1
                if self._hold_frames[node] == 0:
                    self._corridor_lane[node] = None

        return self.snapshot()

    def snapshot(self) -> Dict[str, IntersectionPlan]:
        plan: Dict[str, IntersectionPlan] = {}
        for node in self.intersection_ids:
            hold = self._hold_frames[node]
            lane = self._corridor_lane[node]
            plan[node] = IntersectionPlan(
                mode="EMERGENCY" if hold > 0 else "ADAPTIVE",
                corridor_lane=lane if hold > 0 else None,
                hold_frames=hold,
            )
        return plan
