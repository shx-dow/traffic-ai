"""evaluate.py — run baseline vs adaptive over all scenarios and print comparison.

Usage:  python -m sumo_demo.evaluate [--steps 900]
"""
from __future__ import annotations

import argparse

from .harness import (
    SCENARIOS,
    FalconBridge,
    MetricsReport,
    ScenarioTrafficSource,
    SyntheticSignalSink,
    make_controller,
)


def _run(scenario_name: str, mode: str, steps: int, seed: int, service_rate: float) -> MetricsReport:
    scenario = SCENARIOS[scenario_name]
    controller = make_controller(mode)
    sink = SyntheticSignalSink(service_rate=service_rate)
    bridge = FalconBridge(
        source=ScenarioTrafficSource(scenario, seed=seed),
        sink=sink,
        fps=1,
    )
    result = bridge.run(controller, total_steps=steps)
    return result.report(scenario.name, mode.upper(), sink, steps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs adaptive across scenarios")
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--service-rate", type=float, default=1.0)
    parser.add_argument("--scenario", default=None, help="Run only this scenario (default: all)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenarios = [args.scenario] if args.scenario else sorted(SCENARIOS.keys())
    header = (
        f"{'Scenario':<12} {'Controller':<10} {'AvgWait':>8} {'MaxQ':>5} "
        f"{'AvgQ':>6} {'Served':>7} {'Arrived':>8}"
    )
    print(header)
    print("-" * len(header))
    for name in scenarios:
        baseline = _run(name, "baseline", args.steps, args.seed, args.service_rate)
        adaptive = _run(name, "adaptive", args.steps, args.seed, args.service_rate)
        for rep in (baseline, adaptive):
            print(
                f"{rep.scenario:<12} {rep.controller:<10} "
                f"{rep.avg_wait_s:>7.1f}s {rep.max_queue:>5} "
                f"{rep.avg_queue:>6.1f} {rep.vehicles_served:>7} {rep.vehicles_arrived:>8}"
            )
            _print_emergency(rep)
        if adaptive.avg_wait_s > 0 and baseline.avg_wait_s > 0:
            delta = (baseline.avg_wait_s - adaptive.avg_wait_s) / baseline.avg_wait_s * 100
            print(f"             {'D%':<10} {delta:>+6.1f}%")
        print()
    return 0


def _print_emergency(rep: MetricsReport) -> None:
    """Print emergency corridor timing metrics, when present."""
    fields = (
        ("t_preempt", "time_to_preemption_s"),
        ("t_clear", "corridor_clearance_s"),
        ("t_preempt_dur", "preemption_duration_s"),
        ("t_recover", "recovery_time_s"),
    )
    pieces = []
    for label, name in fields:
        value = getattr(rep, name, None)
        if value is not None:
            pieces.append(f"{label}={value}s")
    if pieces:
        print(f"             {'EMERG':<10} {' '.join(pieces)}")


if __name__ == "__main__":
    raise SystemExit(main())
