"""run_traci.py — drive Falcon controllers against real SUMO via the harness bridge.

When SUMO is installed, this starts a live SUMO process against the
3x3 signalized grid (B1 = the controlled intersection), generates a route file
for the chosen scenario, and runs the selected Falcon controller end-to-end
using the same FalconBridge that powers the synthetic harness.

When SUMO is not installed it degrades to the synthetic backend so the tool is
always runnable.

Usage:  python -m sumo_demo.run_traci [--scenario medium] [--controller adaptive]
                                      [--steps 900] [--gui]
"""
from __future__ import annotations

import argparse
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from .harness import (
    SCENARIOS,
    FalconBridge,
    ScenarioTrafficSource,
    SyntheticSignalSink,
    make_controller,
)
from .harness.traci import (
    TraciSignalSink,
    TraciTrafficSource,
    _import_runtime,
    _sumo_binary,
)

# B1 grid: Falcon approach -> SUMO route through the controlled intersection.
APPROACH_ROUTES = {
    "north": "north_south",
    "south": "south_north",
    "west": "west_east",
    "east": "east_west",
}
ROUTE_EDGES = {
    "north": "B0B1 B1B2",
    "south": "B2B1 B1B0",
    "west": "A1B1 B1C1",
    "east": "C1B1 B1A1",
}


def write_route_file(scenario, target: Path, end_step: int) -> None:
    """Generate a .rou.xml for a Scenario: Poisson-drawn individual vehicles so
    arrival timing matches the synthetic harness's Poisson process."""
    import random

    rng = random.Random(42)
    flows = scenario.flows_per_hour

    routes_node = ET.Element("routes")
    ET.SubElement(
        routes_node, "vType", id="car", vClass="passenger", accel="2.6",
        decel="4.5", sigma="0.5", length="5.0", minGap="2.5", maxSpeed="13.89",
        color="0.2,0.6,1.0",
    )
    ET.SubElement(
        routes_node, "vType", id="ambulance", vClass="emergency", accel="3.2",
        decel="4.5", sigma="0.2", length="6.0", minGap="2.0", maxSpeed="16.67",
        color="1.0,0.1,0.1",
    )
    for approach, edges in ROUTE_EDGES.items():
        ET.SubElement(routes_node, "route", id=APPROACH_ROUTES[approach], edges=edges)

    vehicle_index = 0
    for approach, flow in flows.items():
        rate = flow / 3600.0
        for step in range(end_step):
            surge = 1.0
            if scenario.surge_window:
                start, end = scenario.surge_window
                if start <= step < end:
                    surge = scenario.surge_multiplier
            p = rate * surge
            if p > 0 and rng.random() < p:
                ET.SubElement(
                    routes_node, "vehicle", id=f"v_{approach}_{vehicle_index}",
                    type="car", route=APPROACH_ROUTES[approach], depart=str(step),
                    departLane="best", departSpeed="max",
                )
                vehicle_index += 1

    if scenario.emergency_window and scenario.emergency_lane:
        start, _ = scenario.emergency_window
        ET.SubElement(
            routes_node, "vehicle", id="ambulance_0", type="ambulance",
            route=APPROACH_ROUTES[scenario.emergency_lane], depart=str(start),
            departLane="best", departSpeed="max",
        )

    ET.ElementTree(routes_node).write(target, encoding="utf-8", xml_declaration=True)


def write_sumocfg(net_path: Path, route_path: Path, target: Path, end_step: int) -> None:
    cfg = ET.Element("configuration")
    inp = ET.SubElement(cfg, "input")
    ET.SubElement(inp, "net-file", value=str(net_path.resolve()))
    ET.SubElement(inp, "route-files", value=str(route_path.resolve()))
    time_node = ET.SubElement(cfg, "time")
    ET.SubElement(time_node, "begin", value="0")
    ET.SubElement(time_node, "end", value=str(end_step))
    ET.SubElement(time_node, "step-length", value="1.0")
    ET.ElementTree(cfg).write(target, encoding="utf-8", xml_declaration=True)


def _run_synthetic(scenario_name, mode, steps, service_rate):
    scenario = SCENARIOS[scenario_name]
    sink = SyntheticSignalSink(service_rate=service_rate)
    bridge = FalconBridge(
        source=ScenarioTrafficSource(scenario, seed=42),
        sink=sink,
        fps=1,
    )
    result = bridge.run(make_controller(mode), total_steps=steps)
    return result.report(scenario.name, mode.upper(), sink, steps)


def _run_traci(scenario_name, mode, steps, gui):
    net_path = Path(__file__).parent / "scenarios" / "real" / "single_intersection.net.xml"
    if not net_path.exists():
        raise FileNotFoundError(f"Network file not found: {net_path}")

    scenario = SCENARIOS[scenario_name]
    traci = _import_runtime()
    binary = _sumo_binary(gui=gui)
    if traci is None or binary is None:
        raise RuntimeError("SUMO not available; use synthetic backend")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        route_path = tmp / f"{scenario_name}.rou.xml"
        cfg_path = tmp / f"{scenario_name}.sumocfg"
        write_route_file(scenario, route_path, steps)
        write_sumocfg(net_path, route_path, cfg_path, steps)

        cmd = [str(binary), "-c", str(cfg_path)]
        if gui:
            cmd.extend(["--start", "--quit-on-end", "false", "--delay", "50"])
        traci.start(cmd)
        try:
            sink = TraciSignalSink(traci, tls_id="B1")
            source = TraciTrafficSource(traci, tls_id="B1")
            bridge = FalconBridge(source=source, sink=sink, fps=1)
            result = bridge.run(make_controller(mode), total_steps=steps)
            return result.report(scenario.name, mode.upper(), sink, steps)
        finally:
            traci.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Falcon against real SUMO (or synthetic fallback)")
    parser.add_argument("--scenario", default="medium", choices=sorted(SCENARIOS))
    parser.add_argument("--controller", default="adaptive", choices=["adaptive", "baseline"])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--service-rate", type=float, default=1.0,
                        help="synthetic backend service rate (ignored for SUMO)")
    parser.add_argument("--gui", action="store_true", help="use sumo-gui")
    parser.add_argument("--backend", default="auto", choices=["auto", "synthetic", "traci"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    backend = args.backend
    if backend == "auto":
        traci = _import_runtime()
        binary = _sumo_binary(gui=args.gui)
        backend = "traci" if (traci is not None and binary is not None) else "synthetic"

    if backend == "traci":
        try:
            report = _run_traci(args.scenario, args.controller, args.steps, args.gui)
        except (RuntimeError, FileNotFoundError) as exc:
            print(f"[run_traci] TraCI backend unavailable ({exc}); falling back to synthetic")
            report = _run_synthetic(
                args.scenario, args.controller, args.steps, args.service_rate
            )
            backend = "synthetic"
    else:
        report = _run_synthetic(args.scenario, args.controller, args.steps, args.service_rate)

    print(
        f"[{report.controller}] {report.scenario} ({backend}): "
        f"avg_wait={report.avg_wait_s}s max_queue={report.max_queue} "
        f"avg_queue={report.avg_queue} served={report.vehicles_served}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
