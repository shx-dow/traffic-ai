from __future__ import annotations

import argparse

from .config import SumoDemoConfig
from .runner import run_pre_system, write_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SUMO pre-system baseline simulation")
    parser.add_argument("--out", default="artifacts/sumo_pre_system.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = SumoDemoConfig()
    history = run_pre_system(cfg)
    write_summary(history, args.out)
    print(f"Wrote pre-system summary to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
