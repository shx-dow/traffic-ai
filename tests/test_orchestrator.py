from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic.orchestrator import CorridorOrchestrator


def test_orchestrator_preclears_current_and_upcoming_nodes():
    route = ["int_a", "int_b", "int_c", "int_d"]
    o = CorridorOrchestrator(route, preempt_hops=2, latch_frames=5)

    plan = o.update(route=route, position_index=1, ambulance_lane="south")

    assert plan["int_b"].mode == "EMERGENCY"
    assert plan["int_c"].mode == "EMERGENCY"
    assert plan["int_d"].mode == "EMERGENCY"
    assert plan["int_a"].mode == "ADAPTIVE"
    assert plan["int_c"].corridor_lane == "south"


def test_orchestrator_latch_expires_without_updates():
    route = ["int_a", "int_b"]
    o = CorridorOrchestrator(route, preempt_hops=0, latch_frames=2)

    plan = o.update(route=route, position_index=0, ambulance_lane="east")
    assert plan["int_a"].mode == "EMERGENCY"
    assert plan["int_a"].corridor_lane == "east"

    plan = o.update(route=None, position_index=None, ambulance_lane=None)
    assert plan["int_a"].mode == "EMERGENCY"

    plan = o.update(route=None, position_index=None, ambulance_lane=None)
    assert plan["int_a"].mode == "ADAPTIVE"
    assert plan["int_a"].corridor_lane is None


if __name__ == "__main__":
    test_orchestrator_preclears_current_and_upcoming_nodes()
    test_orchestrator_latch_expires_without_updates()
    print("PASS test_orchestrator")
