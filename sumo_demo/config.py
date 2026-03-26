from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SumoDemoConfig:
    net_file: str = "sumo_demo/scenarios/single_intersection.net.xml"
    route_file: str = "sumo_demo/scenarios/single_intersection.rou.xml"
    sumocfg_file: str = "sumo_demo/scenarios/single_intersection.sumocfg"
    baseline_tls_id: str = "J0"
    adaptive_tls_id: str = "J0"
    emergency_vehicle_id: str = "ambulance_0"
    emergency_start_step: int = 200
    emergency_end_step: int = 520
    sim_steps: int = 900
    step_length: float = 1.0
