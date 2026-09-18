from __future__ import annotations

from logic.signal import SignalController


class ActuatedSignalController(SignalController):
    """Traffic-adaptive signal (gap-out) controller.

    Classic NEMA-style actuation: an approach keeps green for at least
    ``min_green_s`` seconds, extends while vehicles keep arriving (no
    measurable "gap"), gapping out once the approach has been idle for
    ``gap_s`` seconds, and hard max-outs at ``max_green_s``.

    Inherits the same safe phase machine (YELLOW/ALL_RED clearance and
    emergency preemption) as every other controller.  Lane order is a fixed
    round-robin cycle, exactly like the fixed-time baseline — only the *green
    hold* is demand responsive.
    """

    def __init__(
        self,
        min_green_s: int = 10,
        max_green_s: int = 30,
        gap_s: int = 3,
    ) -> None:
        super().__init__()
        self.min_green_s = max(1, int(min_green_s))
        self.max_green_s = max(self.min_green_s, int(max_green_s))
        self.gap_s = max(1, int(gap_s))
        self._gap_streak = 0
        self.mode = "ACTUATED"

    def calculate_green_times(self, lane_counts):
        """Actuation does not pre-allocate cycles; report the hold bounds."""
        return {
            lane: self.max_green_s
            for lane in self.lanes
        }

    def should_switch_lane(self, active_lane, lane_counts, frame_counter, fps):
        if frame_counter == 0:
            self._gap_streak = 0
        active_count = int(lane_counts.get(active_lane, 0))

        min_frames = int(self.min_green_s * fps)
        max_frames = int(self.max_green_s * fps)
        gap_frames = int(self.gap_s * fps)

        if frame_counter < min_frames:
            return False
        if frame_counter >= max_frames:
            return True
        if active_count == 0:
            self._gap_streak += 1
            if self._gap_streak >= gap_frames:
                return True
        else:
            self._gap_streak = 0
        return False

    def choose_next_lane(self, active_lane, lane_counts):
        index = self.lanes.index(active_lane)
        return self.lanes[(index + 1) % len(self.lanes)]

    def resume_adaptive(self):
        super().resume_adaptive()
        self._gap_streak = 0
        self.mode = "ACTUATED"
