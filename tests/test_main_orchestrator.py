from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic.orchestrator import CorridorOrchestrator


def test_runtime_style_orchestrator_progression_increases_preclear_window():
    route = ["int_a", "int_b", "int_c", "int_d"]
    orchestrator = CorridorOrchestrator(route, preempt_hops=2, latch_frames=90)

    plan_start = orchestrator.update(route=route, position_index=0, ambulance_lane="north")
    assert plan_start["int_a"].mode == "EMERGENCY"
    assert plan_start["int_d"].mode == "ADAPTIVE"

    plan_mid = orchestrator.update(route=route, position_index=1, ambulance_lane="north")
    assert plan_mid["int_b"].mode == "EMERGENCY"
    assert plan_mid["int_d"].mode == "EMERGENCY"


if __name__ == "__main__":
    test_runtime_style_orchestrator_progression_increases_preclear_window()
    print("PASS test_main_orchestrator")
