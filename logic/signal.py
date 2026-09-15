class SignalController:
    """
    Converts per-lane vehicle counts into adaptive green signal durations.
    Manages RED/GREEN/YELLOW/ALL_RED signal states with a phase machine and
    provides safe emergency override with clearance intervals.

    Core formula:
        green_time = MIN + (count / total) * (MAX - MIN)
        clamped to [MIN_GREEN, MAX_GREEN]

    Phase machine (normal control):
        GREEN -> YELLOW -> ALL_RED -> NEXT_GREEN

    Emergency control (safe transition, FHWA-style clearance):
        ANY NORMAL -> CLEARING (YELLOW/ALL_RED) -> PREEMPTION (corridor GREEN)
        -> RECOVERY (YELLOW/ALL_RED) -> ADAPTIVE

    Emergency preemption never jumps directly between two conflicting greens:
    the currently-greened approach is cleared through YELLOW + ALL_RED before the
    corridor approach turns green, and the corridor is cleared via YELLOW +
    ALL_RED during RECOVERY before normal adaptive control resumes.
    """

    def __init__(self):
        self.MIN_GREEN = 10
        self.MAX_GREEN = 60
        self.MIN_HOLD_SECONDS = 5
        self.SWITCH_GAP = 3
        self.CONGESTION_WAIT_WEIGHT = 2.0
        self.CONGESTION_BALANCE_GAP = 2.5
        self.SWITCH_GAP_RELATIVE = 0.3   # switch gap scales with the active lane's load
        self.GAP_EARLY_EXIT_MIN_SERVICE_FRAC = 0.7  # share of proportional green the
                                                    # active lane must serve before a
                                                    # raw gap-based early exit is allowed
        self.GAP_ACTIVE_CLEAR_THRESHOLD = 3.0      # active lane judged "essentially served"
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

        # --- Phase machine parameters (seconds) ---
        self.YELLOW_TIME = 3.0
        self.ALL_RED_TIME = 2.0
        self.MIN_RED_TIME = 2.0
        self.EMERGENCY_MAX_DURATION = 30.0      # auto-recover safety
        self.RECOVERY_DURATION = self.YELLOW_TIME + self.ALL_RED_TIME
        self.SWITCH_MARGIN = self.CONGESTION_BALANCE_GAP
        self.STARVATION_THRESHOLD = self.MAX_WAIT_CYCLES

        # --- Phase machine state ---
        self.phase = 'GREEN'                    # 'GREEN' | 'YELLOW' | 'ALL_RED'
        self.phase_remaining_frames = 0
        self.transition_next_lane = None        # lane that turns green next
        self.prev_green_lane = None

        # --- Emergency machine state ---
        self.emergency_state = 'NONE'           # 'NONE' | 'CLEARING' | 'PREEMPTION' | 'RECOVERY'
        self.emergency_corridor_lane = None
        self.emergency_frames = 0

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

        if best_other_lane and best_other_score >= active_score + self._effective_switch_gap(active_score):
            gap_exit_allowed = (
                active_score <= self.GAP_ACTIVE_CLEAR_THRESHOLD
                or frame_counter >= int(active_frames_target * self.GAP_EARLY_EXIT_MIN_SERVICE_FRAC)
            )
            if gap_exit_allowed:
                return True

        if frame_counter >= int(self.MAX_GREEN * fps):
            return True

        return False

    def _effective_switch_gap(self, active_score: float) -> float:
        """Balance gap that scales with the active lane's load.

        Under light load the absolute CONGESTION_BALANCE_GAP dominates (small
        counts: 2.5 vehicles is a meaningful difference).  Under heavy load the
        gap grows proportionally to the active lane's score so the controller
        does not flip lanes over noise when several approaches are saturated.
        """
        return max(self.CONGESTION_BALANCE_GAP, float(active_score) * self.SWITCH_GAP_RELATIVE)

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

    # ------------------------------------------------------------------ #
    # Phase machine API
    # ------------------------------------------------------------------ #

    def tick(self, fps: int) -> None:
        """Advance the phase machine by one frame (call once per processed frame)."""
        if self.phase_remaining_frames > 0:
            self.phase_remaining_frames -= 1
            if self.phase_remaining_frames == 0:
                self._advance_phase(int(fps))

        if self.mode == 'EMERGENCY' and self.emergency_state == 'PREEMPTION':
            self.emergency_frames += 1
            if self.emergency_frames >= int(self.EMERGENCY_MAX_DURATION * fps):
                self.begin_recovery(int(fps))

    def _advance_phase(self, fps: int) -> None:
        if self.mode == 'EMERGENCY':
            if self.emergency_state == 'CLEARING':
                if self.phase == 'YELLOW':
                    self.phase = 'ALL_RED'
                    self.phase_remaining_frames = int(self.ALL_RED_TIME * fps)
                else:  # ALL_RED -> corridor green
                    self.phase = 'GREEN'
                    self.emergency_state = 'PREEMPTION'
                    self.emergency_frames = 0
            elif self.emergency_state == 'RECOVERY':
                if self.phase == 'YELLOW':
                    self.phase = 'ALL_RED'
                    self.phase_remaining_frames = int(self.ALL_RED_TIME * fps)
                else:  # ALL_RED -> back to adaptive control (subclass-aware)
                    self.resume_adaptive()
            return

        # Normal control: advance a GREEN -> YELLOW -> ALL_RED -> NEXT_GREEN sequence.
        if self.phase == 'YELLOW':
            self.phase = 'ALL_RED'
            self.phase_remaining_frames = int(self.ALL_RED_TIME * fps)
        else:  # ALL_RED -> next green
            self.phase = 'GREEN'
            self.prev_green_lane = None
            self.transition_next_lane = None

    def begin_lane_transition(self, prev_lane: str, next_lane: str, fps: int) -> None:
        """Start a normal GREEN -> YELLOW -> ALL_RED -> NEXT_GREEN lane change."""
        if self.mode == 'EMERGENCY':
            return
        self.phase = 'YELLOW'
        self.prev_green_lane = prev_lane
        self.transition_next_lane = next_lane
        self.phase_remaining_frames = int(self.YELLOW_TIME * fps)

    @property
    def is_transitioning(self) -> bool:
        """True while a normal lane change is clearing (YELLOW/ALL_RED)."""
        return self.mode in ('ADAPTIVE', 'BASELINE') and self.phase != 'GREEN'

    # ------------------------------------------------------------------ #
    # Emergency control
    # ------------------------------------------------------------------ #

    def _transition_state(self):
        state = {lane: 'RED' for lane in ['north', 'south', 'east', 'west']}
        if self.phase == 'YELLOW':
            yellow_lane = self.prev_green_lane or self.emergency_corridor_lane
            if yellow_lane in state:
                state[yellow_lane] = 'YELLOW'
        self.current_state = state
        return state

    def request_emergency_preemption(
        self,
        corridor_lane: str,
        fps: int,
        current_green_lane: str | None = None,
    ):
        """
        Safely request emergency preemption: clear the current green through
        YELLOW + ALL_RED, then give the corridor approach GREEN.
        """
        self.mode = 'EMERGENCY'
        self.emergency_state = 'CLEARING'
        self.emergency_corridor_lane = corridor_lane
        self.prev_green_lane = current_green_lane or self.prev_green_lane
        self.transition_next_lane = corridor_lane
        self.phase = 'YELLOW'
        self.phase_remaining_frames = int(self.YELLOW_TIME * fps)
        self.emergency_frames = 0
        return self.get_current_signal_state(current_green_lane or corridor_lane)

    def begin_recovery(self, fps: int) -> None:
        """Safely release the corridor: YELLOW + ALL_RED, then adaptive control."""
        if self.mode != 'EMERGENCY' or self.emergency_state == 'RECOVERY':
            return
        self.emergency_state = 'RECOVERY'
        corridor = self.emergency_corridor_lane or self.prev_green_lane
        self.prev_green_lane = corridor
        if self.phase == 'ALL_RED':
            self.phase_remaining_frames = int(self.ALL_RED_TIME * fps)
        else:
            self.phase = 'YELLOW'
            self.phase_remaining_frames = int(self.YELLOW_TIME * fps)

    def override_for_emergency(self, corridor_lane):
        """
        Immediate emergency override (compat fast path):
        forces corridor_lane GREEN and suspends adaptive cycle.
        """
        self.mode = 'EMERGENCY'
        self.emergency_state = 'PREEMPTION'
        self.phase = 'GREEN'
        self.phase_remaining_frames = 0
        self.transition_next_lane = None
        self.prev_green_lane = None
        self.emergency_corridor_lane = corridor_lane
        self.emergency_frames = 0
        return self.get_current_signal_state(corridor_lane)

    def override_all_green(self):
        self.mode = 'EMERGENCY'
        self.emergency_state = 'PREEMPTION'
        self.phase = 'GREEN'
        self.phase_remaining_frames = 0
        self.transition_next_lane = None
        self.prev_green_lane = None
        self.emergency_corridor_lane = None
        self.emergency_frames = 0
        state = {lane: 'GREEN' for lane in self.lanes}
        self.current_state = state
        return state

    def resume_adaptive(self):
        self.mode = 'ADAPTIVE'
        self.emergency_state = 'NONE'
        self.phase = 'GREEN'
        self.phase_remaining_frames = 0
        self.transition_next_lane = None
        self.prev_green_lane = None
        self.emergency_corridor_lane = None
        self.emergency_frames = 0

    # ------------------------------------------------------------------ #
    # Outputs
    # ------------------------------------------------------------------ #

    def get_current_signal_state(self, active_lane):
        """
        Returns the current lane-wise signal state (RED/GREEN/YELLOW/ALL_RED).

        During a transition the previously-green approach is YELLOW then all
        approaches are RED (ALL_RED). During emergency preemption only the
        corridor approach is GREEN. Otherwise only `active_lane` is GREEN.
        """
        if self.mode == 'EMERGENCY':
            if self.emergency_state in ('CLEARING', 'RECOVERY'):
                return self._transition_state()
            corridor = self.emergency_corridor_lane or active_lane
            state = {lane: 'GREEN' if lane == corridor else 'RED' for lane in ['north', 'south', 'east', 'west']}
            self.current_state = state
            return state

        if self.phase != 'GREEN':
            return self._transition_state()

        state = {lane: 'GREEN' if lane == active_lane else 'RED' for lane in ['north', 'south', 'east', 'west']}
        self.current_state = state
        return state