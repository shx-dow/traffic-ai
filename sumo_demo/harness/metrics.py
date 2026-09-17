"""Queue model for the P1-A harness.

A light, deterministic discrete-time queue per approach lane.  Each second a
GREEN approach serves `service_rate` vehicles (a crude saturation flow) from its
queue and all approaches accumulate arrivals.  Accumulates the metrics the
evaluation needs:
    - average waiting time (Little's law: total queue-seconds / served)
    - throughput (vehicles served)
    - maximum / average queue length
    - emergency: time-to-preemption, corridor clearance, recovery, corridor wait
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MetricsReport:
    scenario: str
    controller: str
    total_steps: int
    avg_wait_s: float
    max_queue: int
    avg_queue: float
    vehicles_served: int
    vehicles_arrived: int
    time_to_preemption_s: int | None = None
    preemption_duration_s: int | None = None
    corridor_clearance_s: int | None = None
    recovery_time_s: int | None = None
    emergency_corridor_wait_s: int | None = None


@dataclass
class QueueModel:
    lanes: tuple = ("north", "south", "east", "west")
    service_rate: float = 0.5                # vehicles/sec served while GREEN
    queue: dict[str, float] = field(default_factory=dict)
    served: dict[str, int] = field(default_factory=dict)
    arrived: dict[str, int] = field(default_factory=dict)
    queue_seconds: float = 0.0
    max_queue: int = 0
    queue_samples: int = 0
    queue_total: float = 0.0

    def __post_init__(self):
        for lane in self.lanes:
            self.queue.setdefault(lane, 0.0)
            self.served.setdefault(lane, 0)
            self.arrived.setdefault(lane, 0)

    def step(self, arrivals: dict[str, int], signal_state: dict[str, str]) -> None:
        total_queue = 0.0
        for lane in self.lanes:
            self.queue[lane] += float(arrivals.get(lane, 0))
            self.arrived[lane] += int(arrivals.get(lane, 0))
            if signal_state.get(lane) == "GREEN" and self.queue[lane] > 0:
                served_now = min(self.queue[lane], self.service_rate)
                self.queue[lane] -= served_now
                self.served[lane] += served_now
                if self.queue[lane] < 0.0:
                    self.queue[lane] = 0.0
            total_queue += self.queue[lane]
        self.queue_seconds += total_queue
        self.queue_total += total_queue
        self.queue_samples += 1
        peak = int(total_queue)
        if peak > self.max_queue:
            self.max_queue = peak

    def queue_snapshot(self) -> dict[str, int]:
        return {lane: int(self.queue[lane]) for lane in self.lanes}

    def report(self, scenario: str, controller: str, total_steps: int) -> MetricsReport:
        total_served = float(sum(self.served.values()))
        total_arrived = sum(self.arrived.values())
        avg_wait = (self.queue_seconds / total_served) if total_served else 0.0
        avg_queue = (self.queue_total / self.queue_samples) if self.queue_samples else 0.0
        return MetricsReport(
            scenario=scenario,
            controller=controller,
            total_steps=total_steps,
            avg_wait_s=round(avg_wait, 2),
            max_queue=int(self.max_queue),
            avg_queue=round(avg_queue, 2),
            vehicles_served=int(total_served),
            vehicles_arrived=int(total_arrived),
        )


def summarize(
    scenario: str,
    controller: str,
    total_steps: int,
    queue_model: QueueModel,
    *,
    emergency_events: dict[str, int] | None = None,
) -> MetricsReport:
    """Wrap a QueueModel report with emergency timing events.

    `emergency_events` keys: time_to_preemption_s, preemption_duration_s,
    corridor_clearance_s, recovery_time_s, emergency_corridor_wait_s.
    """
    report = queue_model.report(scenario, controller, total_steps)
    if emergency_events:
        return MetricsReport(
            **{**report.__dict__, **emergency_events}
        )
    return report
