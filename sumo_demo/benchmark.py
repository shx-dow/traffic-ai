"""benchmark.py — P1-B multi-scenario benchmark (baseline vs adaptive).

Runs LOW/MEDIUM/HEAVY/SURGE/EMERGENCY x {baseline, adaptive} over multiple
seeds and aggregates:
  - avg_wait_s, max_queue, avg_queue, vehicles_served (mean +/- std)
  - per-scenario verdict (adaptive better / baseline better / tied) based on
    the mean average-wait delta
  - a JSON artifact under profiling/artifacts/benchmark_results.json

Usage:
    python -m sumo_demo.benchmark [--steps 900] [--seeds 5]
                                  [--scenario medium] [--out artifacts/...json]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

from .harness import (
    SCENARIOS,
    FalconBridge,
    MetricsReport,
    ScenarioTrafficSource,
    SyntheticSignalSink,
    make_controller,
)

ROOT = Path(__file__).resolve().parent.parent


def _run(
    scenario_name: str,
    mode: str,
    steps: int,
    seed: int,
    service_rate: float,
) -> MetricsReport:
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


def _average(name: str, mode: str, steps: int, seeds, service_rate: float) -> Dict[str, float]:
    reports = [_run(name, mode, steps, seed, service_rate) for seed in seeds]
    waits = [r.avg_wait_s for r in reports]
    maxqs = [r.max_queue for r in reports]
    avgqs = [r.avg_queue for r in reports]
    served = [r.vehicles_served for r in reports]

    def mean_std(values: List[float]) -> Tuple[float, float]:
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        return round(mean, 2), round(std, 2)

    def mean_only(values) -> float | None:
        present = [float(v) for v in values if v is not None]
        return round(statistics.fmean(present), 2) if present else None

    return {
        "avg_wait_s": waits,
        "avg_wait_mean": mean_std(waits)[0],
        "avg_wait_std": mean_std(waits)[1],
        "max_queue_mean": mean_std(maxqs)[0],
        "max_queue_std": mean_std(maxqs)[1],
        "avg_queue_mean": mean_std(avgqs)[0],
        "avg_queue_std": mean_std(avgqs)[1],
        "vehicles_served_mean": mean_std(served)[0],
        "vehicles_served_std": mean_std(served)[1],
        "time_to_preemption_s_mean": mean_only([r.time_to_preemption_s for r in reports]),
        "corridor_clearance_s_mean": mean_only([r.corridor_clearance_s for r in reports]),
        "recovery_time_s_mean": mean_only([r.recovery_time_s for r in reports]),
    }


def _verdict(baseline: Dict[str, float], adaptive: Dict[str, float], threshold: float = 1.0) -> str:
    base_wait = baseline["avg_wait_mean"]
    adapt_wait = adaptive["avg_wait_mean"]
    delta = (base_wait - adapt_wait) / base_wait * 100 if base_wait else 0.0
    if delta > threshold:
        return "ADAPTIVE_BETTER"
    if delta < -threshold:
        return "BASELINE_BETTER"
    return "TIED"


def _format_emergency_mean(rep: Dict[str, float]) -> str:
    """Human-readable emergency corridor timing line, if the scenario has one."""
    fields = (
        ("time_to_preemption_s_mean", "preempt"),
        ("corridor_clearance_s_mean", "clear"),
        ("recovery_time_s_mean", "recover"),
    )
    pieces = []
    for key, label in fields:
        value = rep.get(key)
        if value is not None:
            pieces.append(f"{label}={value}s")
    return "EMERG: " + "  ".join(pieces) if pieces else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P1-B multi-scenario benchmark")
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--service-rate", type=float, default=1.0)
    parser.add_argument("--scenario", default=None, help="Run only this scenario (default: all)")
    parser.add_argument("--out", default="profiling/artifacts/benchmark_results.json")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="delta % at which adaptive is declared the winner")
    return parser.parse_args()


def run_benchmark(scenarios, steps: int, seeds, service_rate: float, threshold: float = 1.0) -> Dict[str, Dict]:
    """Run every scenario x {baseline, adaptive} over seeds, aggregated."""
    results: Dict[str, Dict] = {}
    for name in scenarios:
        baseline = _average(name, "baseline", steps, seeds, service_rate)
        adaptive = _average(name, "adaptive", steps, seeds, service_rate)
        results[name] = {
            "baseline": baseline,
            "adaptive": adaptive,
            "verdict": _verdict(baseline, adaptive, threshold),
        }
    return results


def main() -> int:
    args = parse_args()
    scenarios = [args.scenario] if args.scenario else sorted(SCENARIOS.keys())

    results = run_benchmark(scenarios, args.steps, args.seeds, args.service_rate, args.threshold)

    print(f"Benchmark: {len(scenarios)} scenarios x {len(args.seeds)} seeds x "
          f"{{baseline, adaptive}} at {args.steps} steps each\n")
    header = (
        f"{'Scenario':<12} {'Controller':<10} {'AvgWait':>12} {'MaxQ':>10} "
        f"{'AvgQ':>10} {'Served':>12} {'Verdict':<16}"
    )
    print(header)
    print("-" * len(header))
    print("(mean +/- std over seeds)")
    print("-" * len(header))
    for name in scenarios:
        block = results[name]
        for label, key in (("baseline", "baseline"), ("adaptive", "adaptive")):
            rep = block[key]
            print(
                f"{name:<12} {label:<10} "
                f"{rep['avg_wait_mean']:>6.1f}s +/- {rep['avg_wait_std']:>4.1f} "
                f"{rep['max_queue_mean']:>5.1f} (+/- {rep['max_queue_std']:>4.1f}) "
                f"{rep['avg_queue_mean']:>6.1f} (+/- {rep['avg_queue_std']:>4.1f}) "
                f"{rep['vehicles_served_mean']:>7.0f} (+/- {rep['vehicles_served_std']:>4.0f}) "
                f"{block['verdict']:<16}"
            )
            em = _format_emergency_mean(rep)
            if em:
                print(f"{'':9} {'':10} {em}")
        print()

    out_path = (ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "p1b": "multi-scenario benchmark",
                "steps": args.steps,
                "seeds": args.seeds,
                "service_rate": args.service_rate,
                "threshold_pct": args.threshold,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote benchmark artifact to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())