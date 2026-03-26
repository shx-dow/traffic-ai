from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sumo_demo.config import SumoDemoConfig
from sumo_demo.runner import run_post_system, run_pre_system


def test_sumo_pre_and_post_system_runs_generate_history():
    cfg = SumoDemoConfig(sim_steps=30, emergency_start_step=10, emergency_end_step=20)
    pre = run_pre_system(cfg)
    post = run_post_system(cfg)

    assert len(pre) == 30
    assert len(post) == 30
    assert any(row.emergency_active for row in post)
    assert all(row.mode == "BASELINE" for row in pre)
    assert all(row.mode == "ADAPTIVE" for row in post)


if __name__ == "__main__":
    test_sumo_pre_and_post_system_runs_generate_history()
    print("PASS test_sumo_demo")
