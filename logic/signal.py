
class SignalController:
    """
    Converts per-lane vehicle counts into adaptive green signal durations.
    Manages RED/GREEN signal states and provides emergency override hook.

    Core formula:
        green_time = MIN + (count / total) * (MAX - MIN)
        clamped to [MIN_GREEN, MAX_GREEN]
    """

    def __init__(self):
        self.MIN_GREEN = 10
        self.MAX_GREEN = 60
        self.mode = 'ADAPTIVE'
        self.current_state = {
            'north': 'RED',
            'south': 'RED',
            'east':  'RED',
            'west':  'RED',
        }

    def calculate_green_times(self, lane_counts):
        """
        Proportionally allocates green time based on vehicle density.

        Input:  {'north': int, 'south': int, 'east': int, 'west': int}
        Output: {'north': int, 'south': int, 'east': int, 'west': int}
                 values are seconds, clamped to [MIN_GREEN, MAX_GREEN]
        """
        total = sum(lane_counts.values())

        if total == 0:
            return {lane: self.MIN_GREEN for lane in lane_counts}

        green_times = {}
        budget = self.MAX_GREEN - self.MIN_GREEN
        for lane, count in lane_counts.items():
            proportion = count / total
            raw = self.MIN_GREEN + proportion * budget
            green_times[lane] = max(self.MIN_GREEN, min(int(raw), self.MAX_GREEN))

        return green_times

    def get_current_signal_state(self, active_lane):
        """
        Returns RED/GREEN state for all lanes.
        Only active_lane is GREEN — all others are RED.
        """
        state = {
            lane: 'GREEN' if lane == active_lane else 'RED'
            for lane in ['north', 'south', 'east', 'west']
        }
        self.current_state = state
        return state

    def override_for_emergency(self, corridor_lane):
        """
        Called by emergency.py when ambulance is detected.
        Forces corridor_lane GREEN and suspends adaptive cycle.
        """
        self.mode = 'EMERGENCY'
        return self.get_current_signal_state(corridor_lane)

    def resume_adaptive(self):
        """
        Restore normal adaptive mode.
        """
        self.mode = 'ADAPTIVE'

