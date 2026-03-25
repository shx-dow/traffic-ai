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
        self.MIN_HOLD_SECONDS = 5
        self.SWITCH_GAP = 3
        self.CONGESTION_WAIT_WEIGHT = 2.0
        self.CONGESTION_BALANCE_GAP = 2.5
        self.MAX_WAIT_CYCLES = 4
        self.mode = 'ADAPTIVE'
        self.lanes = ('north', 'south', 'east', 'west')
        self._lane_wait_cycles = {lane: 0 for lane in self.lanes}
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

    def _normalize_counts(self, lane_counts):
        return {lane: int(lane_counts.get(lane, 0)) for lane in self.lanes}

    def _normalize_numeric(self, lane_counts):
        return {lane: float(lane_counts.get(lane, 0.0)) for lane in self.lanes}

    def calculate_congestion_scores(self, lane_counts):
        counts = self._normalize_numeric(lane_counts)
        scores = {}
        for lane in self.lanes:
            waiting_factor = self._lane_wait_cycles.get(lane, 0) * self.CONGESTION_WAIT_WEIGHT
            scores[lane] = float(counts[lane]) + float(waiting_factor)
        return scores

    def record_cycle(self, active_lane, lane_counts):
        counts = self._normalize_counts(lane_counts)
        for lane in self.lanes:
            if lane == active_lane:
                self._lane_wait_cycles[lane] = 0
                continue
            if counts[lane] > 0:
                self._lane_wait_cycles[lane] += 1
            else:
                self._lane_wait_cycles[lane] = 0

    def should_switch_lane(self, active_lane, lane_counts, frame_counter, fps):
        scores = self._normalize_numeric(lane_counts)
        green_times = self.calculate_green_times(scores)
        min_hold_frames = int(self.MIN_HOLD_SECONDS * fps)
        active_frames_target = int(green_times.get(active_lane, self.MIN_GREEN) * fps)
        active_score = scores.get(active_lane, 0.0)

        if frame_counter < min_hold_frames:
            return False

        if frame_counter >= active_frames_target:
            return True

        best_other_lane = None
        best_other_score = -1.0
        for lane, score in scores.items():
            if lane == active_lane:
                continue
            if score > best_other_score:
                best_other_lane = lane
                best_other_score = score

        if best_other_lane and best_other_score >= active_score + self.CONGESTION_BALANCE_GAP:
            return True

        if frame_counter >= int(self.MAX_GREEN * fps):
            return True

        return False

    def choose_next_lane(self, active_lane, lane_counts):
        scores = self._normalize_numeric(lane_counts)

        starved = [
            lane
            for lane in self.lanes
            if lane != active_lane
            and scores[lane] > 0
            and self._lane_wait_cycles[lane] >= self.MAX_WAIT_CYCLES
        ]
        if starved:
            return max(starved, key=lambda lane: (self._lane_wait_cycles[lane], scores[lane]))

        best_lane = max(self.lanes, key=lambda lane: scores[lane])
        if scores[best_lane] > 0:
            return best_lane

        active_index = self.lanes.index(active_lane)
        return self.lanes[(active_index + 1) % len(self.lanes)]

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

    def override_all_green(self):
        self.mode = 'EMERGENCY'
        state = {lane: 'GREEN' for lane in self.lanes}
        self.current_state = state
        return state

    def resume_adaptive(self):
        self.mode = 'ADAPTIVE'


