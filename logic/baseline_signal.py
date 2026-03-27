from __future__ import annotations


class BaselineSignalController:
    def __init__(self, green_seconds: int = 20):
        self.green_seconds = max(5, int(green_seconds))
        self.mode = "BASELINE"
        self.lanes = ("north", "south", "east", "west")
        self.current_state = {lane: "RED" for lane in self.lanes}

    def calculate_green_times(self, lane_counts):
        return {lane: self.green_seconds for lane in self.lanes}

    def record_cycle(self, active_lane, lane_counts):
        return None

    def should_switch_lane(self, active_lane, lane_counts, frame_counter, fps):
        return frame_counter >= int(self.green_seconds * fps)

    def choose_next_lane(self, active_lane, lane_counts):
        index = self.lanes.index(active_lane)
        return self.lanes[(index + 1) % len(self.lanes)]

    def get_current_signal_state(self, active_lane):
        state = {lane: ("GREEN" if lane == active_lane else "RED") for lane in self.lanes}
        self.current_state = state
        return state

    def override_for_emergency(self, corridor_lane):
        self.mode = "EMERGENCY"
        return self.get_current_signal_state(corridor_lane)

    def resume_adaptive(self):
        self.mode = "BASELINE"
