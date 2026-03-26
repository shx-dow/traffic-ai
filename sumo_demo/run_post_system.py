from __future__ import annotations

import argparse

from .config import SumoDemoConfig
from .real_traci_runner import run_real_post_system


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SUMO post-system adaptive simulation")
    parser.add_argument("--out", default="artifacts/sumo_post_system.csv")
    parser.add_argument("--gui", action="store_true", help="Use sumo-gui instead of headless sumo")
    parser.add_argument("--delay", type=int, default=80, help="GUI step delay in ms (GUI only)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = SumoDemoConfig()
    run_real_post_system(cfg, out_csv=args.out, gui=args.gui, gui_delay_ms=args.delay)
    print(f"Wrote post-system summary to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
