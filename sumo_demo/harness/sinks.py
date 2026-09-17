"""Signal state sinks.

A sink receives the Falcon controller's intended signal state each second so the
harness can measure it (synthetic queue model) or hand it to an external
simulator.  The synthetic sink is what runs today; a SUMO/TraCI sink can be
dropped in later with the same interface.
"""
from __future__ import annotations

from .metrics import MetricsReport, QueueModel, summarize


class SignalSink:
    """Receives the controller's signal state each step."""

    def on_step(self, step: int, signal_state: dict[str, str], arrivals: dict[str, int]) -> None:
        raise NotImplementedError

    def report(self, scenario: str, controller: str, total_steps: int, **emergency) -> MetricsReport:
        raise NotImplementedError


class SyntheticSignalSink(SignalSink):
    """Drives a QueueModel from the controller's signal states."""

    def __init__(self, service_rate: float = 0.5):
        self.queue_model = QueueModel(service_rate=service_rate)

    def on_step(self, step: int, signal_state: dict[str, str], arrivals: dict[str, int]) -> None:
        self.queue_model.step(arrivals, signal_state)

    def observed_counts(self) -> dict[str, int]:
        """Current queue length per approach — what a camera ROI would report."""
        return self.queue_model.queue_snapshot()

    def report(self, scenario: str, controller: str, total_steps: int, **emergency) -> MetricsReport:
        return summarize(scenario, controller, total_steps, self.queue_model, emergency_events=emergency or None)

    def reset(self) -> None:
        self.queue_model = QueueModel(service_rate=self.queue_model.service_rate)


class NullSignalSink(SignalSink):
    """Fallback sink that measures nothing (e.g. pure smoke runs)."""

    def on_step(self, step: int, signal_state: dict[str, str], arrivals: dict[str, int]) -> None:
        pass

    def report(self, scenario: str, controller: str, total_steps: int, **emergency) -> MetricsReport:
        return summarize(
            scenario,
            controller,
            total_steps,
            QueueModel(),
        )
