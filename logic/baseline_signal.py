from __future__ import annotations

from logic.signal import SignalController


class BaselineSignalController(SignalController):
    """Fixed-time controller. Inherits the safe phase machine (YELLOW/ALL_RED
    clearance) but allocates a constant green duration to every lane."""

    def __init__(self, green_seconds: int = 20):
        super().__init__()
        self.green_seconds = max(5, int(green_seconds))
        self.mode = "BASELINE"

    def calculate_green_times(self, lane_counts):
        return {lane: self.green_seconds for lane in self.lanes}

    def should_switch_lane(self, active_lane, lane_counts, frame_counter, fps):
        return frame_counter >= int(self.green_seconds * fps)

    def choose_next_lane(self, active_lane, lane_counts):
        index = self.lanes.index(active_lane)
        return self.lanes[(index + 1) % len(self.lanes)]

    def resume_adaptive(self):
        super().resume_adaptive()
        self.mode = "BASELINE"
