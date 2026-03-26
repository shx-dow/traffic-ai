from __future__ import annotations

import argparse

from .config import SumoDemoConfig
from .runner import run_post_system, write_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SUMO post-system adaptive simulation")
    parser.add_argument("--out", default="artifacts/sumo_post_system.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = SumoDemoConfig()
    history = run_post_system(cfg)
    write_summary(history, args.out)
    print(f"Wrote post-system summary to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
