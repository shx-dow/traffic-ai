"""run_baseline.py — run the fixed-time (baseline) controller in the harness.

Usage:  python -m sumo_demo.run_baseline [--scenario medium] [--steps 900]
"""
from __future__ import annotations

import argparse

from .config import SumoDemoConfig
from .harness import (
    SCENARIOS,
    FalconBridge,
    SyntheticSignalSink,
    ScenarioTrafficSource,
    make_controller,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-time baseline in the P1-A harness")
    parser.add_argument("--scenario", default="medium", choices=sorted(SCENARIOS))
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--green-seconds", type=int, default=20)
    parser.add_argument("--service-rate", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = SumoDemoConfig()
    del cfg
    scenario = SCENARIOS[args.scenario]
    controller = make_controller("baseline", green_seconds=args.green_seconds)
    sink = SyntheticSignalSink(service_rate=args.service_rate)
    bridge = FalconBridge(
        source=ScenarioTrafficSource(scenario, seed=args.seed),
        sink=sink,
        fps=1,
    )
    result = bridge.run(controller, total_steps=args.steps)
    report = result.report(scenario.name, "BASELINE", sink, args.steps)
    print(
        f"[BASELINE] {scenario.name}: "
        f"avg_wait={report.avg_wait_s}s "
        f"max_queue={report.max_queue} "
        f"avg_queue={report.avg_queue} "
        f"served={report.vehicles_served}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
