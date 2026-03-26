from __future__ import annotations

from typing import Dict, Iterable

LANES = ("north", "south", "east", "west")


class LiveMetricsTracker:
    def __init__(self, fps: int) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.fps = int(fps)
        self.reset()

    def reset(self) -> None:
        self.frames = 0
        self.emergency_frames = 0
        self.total_wait_units = 0
        self.total_served_units = 0
        self.max_queue = 0

    def update(self, lane_counts: Dict[str, int], green_lanes: Iterable[str], mode: str) -> None:
        self.frames += 1
        if str(mode).upper() == "EMERGENCY":
            self.emergency_frames += 1

        active_green = {str(lane).lower() for lane in green_lanes}
        for lane in LANES:
            queue = int(lane_counts.get(lane, 0))
            if queue > self.max_queue:
                self.max_queue = queue
            if lane in active_green and queue > 0:
                self.total_served_units += queue
            elif queue > 0:
                self.total_wait_units += queue

    def snapshot(self) -> Dict[str, float | int]:
        frames = self.frames if self.frames > 0 else 1
        return {
            "frames": self.frames,
            "avg_wait_seconds": self.total_wait_units / (4 * self.fps),
            "throughput_score": self.total_served_units,
            "max_queue": self.max_queue,
            "emergency_duration_seconds": self.emergency_frames / self.fps,
            "emergency_ratio": self.emergency_frames / frames,
        }
