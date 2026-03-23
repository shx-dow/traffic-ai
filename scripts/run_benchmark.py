from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from logic.baseline_signal import BaselineSignalController
from logic.metrics import compare_metrics, compute_metrics
from logic.signal import SignalController
from logic.simulation import run_signal_simulation


def build_scenario(total_frames: int) -> list[dict]:
    events = []
    first_phase_end = total_frames // 3
    second_phase_end = (2 * total_frames) // 3
    emergency_start = total_frames // 2
    emergency_end = emergency_start + max(30, total_frames // 10)

    for i in range(total_frames):
        emergency = emergency_start <= i < emergency_end

        if i < first_phase_end:
            lane_counts = {"north": 1, "south": 12, "east": 0, "west": 0}
        elif i < second_phase_end:
            lane_counts = {"north": 2, "south": 11, "east": 1, "west": 0}
        else:
            lane_counts = {"north": 10, "south": 2, "east": 0, "west": 0}

        events.append(
            {
                "lane_counts": lane_counts,
                "emergency": emergency,
                "corridor_lane": "south" if emergency else None,
            }
        )
    return events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline vs adaptive benchmark and store metrics")
    parser.add_argument("--frames", type=int, default=1800)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--baseline-green-seconds", type=int, default=20)
    parser.add_argument("--out", default="artifacts/metrics.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames < 900:
        print("Warning: frames < 900 may not show stable adaptive gains")
    scenario = build_scenario(args.frames)

    baseline = BaselineSignalController(green_seconds=args.baseline_green_seconds)
    adaptive = SignalController()

    baseline_history = run_signal_simulation(baseline, scenario, fps=args.fps)
    adaptive_history = run_signal_simulation(adaptive, scenario, fps=args.fps)

    baseline_metrics = compute_metrics(baseline_history, fps=args.fps)
    adaptive_metrics = compute_metrics(adaptive_history, fps=args.fps)
    comparison = compare_metrics(baseline_metrics, adaptive_metrics)

    out = {
        "scenario": {
            "frames": args.frames,
            "fps": args.fps,
            "baseline_green_seconds": args.baseline_green_seconds,
        },
        "baseline": baseline_metrics,
        "adaptive": adaptive_metrics,
        "comparison": comparison,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Wrote benchmark metrics to {out_path}")
    print(f"baseline_avg_wait={baseline_metrics['avg_wait_seconds_overall']:.2f}")
    print(f"adaptive_avg_wait={adaptive_metrics['avg_wait_seconds_overall']:.2f}")
    print(f"wait_reduction_pct={comparison['wait_reduction_pct']:.2f}")
    print(f"throughput_gain_pct={comparison['throughput_gain_pct']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
