from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from logic.orchestrator import CorridorOrchestrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-intersection emergency corridor demo")
    parser.add_argument("--frames", type=int, default=120, help="Number of simulation frames")
    parser.add_argument("--fps", type=int, default=15, help="Simulation fps")
    parser.add_argument("--preempt-hops", type=int, default=2, help="How many intersections ahead to pre-clear")
    parser.add_argument("--latch-seconds", type=float, default=3.0, help="How long an intersection stays in emergency mode")
    parser.add_argument("--out", default="artifacts/orchestrator_demo.json", help="Output JSON summary path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames <= 0 or args.fps <= 0:
        print("frames and fps must be positive", file=sys.stderr)
        return 1

    route = ["int_a", "int_b", "int_c", "int_d"]
    orchestrator = CorridorOrchestrator(
        intersection_ids=route,
        preempt_hops=args.preempt_hops,
        latch_frames=max(1, int(args.latch_seconds * args.fps)),
    )

    frame_events = []
    for frame in range(args.frames):
        position_index = min(len(route) - 1, frame // max(1, args.fps * 2))
        lane = "south" if position_index < 2 else "east"
        plan = orchestrator.update(route=route, position_index=position_index, ambulance_lane=lane)
        row = {
            "frame": frame,
            "position_index": position_index,
            "route": route,
            "plan": {
                node: {
                    "mode": p.mode,
                    "corridor_lane": p.corridor_lane,
                    "hold_frames": p.hold_frames,
                }
                for node, p in plan.items()
            },
        }
        frame_events.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "scenario": {
            "frames": args.frames,
            "fps": args.fps,
            "route": route,
            "preempt_hops": args.preempt_hops,
            "latch_seconds": args.latch_seconds,
        },
        "events": frame_events,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote orchestrator demo to {out_path}")
    print("Final intersection states:")
    final_plan = frame_events[-1]["plan"]
    for node in route:
        state = final_plan[node]
        print(f"  {node}: mode={state['mode']} lane={state['corridor_lane']} hold={state['hold_frames']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
