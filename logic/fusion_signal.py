from __future__ import annotations

from logic.signal import SignalController


class FusionSignalController(SignalController):
    """Backlog + arrival-rate allocator.

    Identical phase/preemption safety to the adaptive controller, but the
    demand signal fed to the allocator combines the *observed queue*
    (backlog) with the current step's *arrival intensity* instead of queue
    alone.  The FalconBridge blends the two once ``uses_arrivals`` is set, so
    `choose_next_lane`/`should_switch_lane` never see a raw queue here — they
    see the fused demand.

    ``FUSION_WEIGHT`` trades how much the instantaneous arrival rate counts
    relative to standing queue.  Weight 0 degenerates to the plain adaptive
    controller; weight > 1 biases heavily toward freshly-arriving traffic.
    """

    uses_arrivals = True

    def __init__(self, fusion_weight: float = 1.0) -> None:
        super().__init__()
        self.FUSION_WEIGHT = float(fusion_weight)
        self.mode = "FUSION"

    def resume_adaptive(self):
        super().resume_adaptive()
        self.mode = "FUSION"
