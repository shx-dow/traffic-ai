from __future__ import annotations

import argparse

from .config import SumoDemoConfig
from .traci_runner import run_real_sumo_pre_system


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SUMO pre-system baseline simulation")
    parser.add_argument("--out", default="artifacts/sumo_pre_system.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = SumoDemoConfig()
    run_real_sumo_pre_system(cfg, out_csv=args.out)
    print(f"Wrote pre-system summary to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
