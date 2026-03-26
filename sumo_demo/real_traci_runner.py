from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from .config import SumoDemoConfig
from .controller import SumoAdaptiveController, SumoBaselineController, SumoStepResult
from .sumo_path import ensure_sumo_home


def _import_runtime():
    ensure_sumo_home()
    try:
        import traci  # type: ignore
        import sumolib  # type: ignore
        del sumolib
        return traci
    except Exception:
        return None


def _write_csv(history: List[SumoStepResult], out_path: str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step", "mode", "active_lane", "emergency_active", "corridor_lane", "north", "south", "east", "west"],
        )
        writer.writeheader()
        for row in history:
            counts = row.lane_counts
            writer.writerow({
                "step": row.step,
                "mode": row.mode,
                "active_lane": row.active_lane,
                "emergency_active": row.emergency_active,
                "corridor_lane": row.corridor_lane or "",
                "north": counts.get("north", 0),
                "south": counts.get("south", 0),
                "east": counts.get("east", 0),
                "west": counts.get("west", 0),
            })


def _lane_counts_from_sumo(traci, tls_id: str) -> Dict[str, int]:
    counts = {"north": 0, "south": 0, "east": 0, "west": 0}
    if not tls_id:
        return counts
    try:
        lanes = traci.trafficlight.getControlledLanes(tls_id)
    except Exception:
        return counts

    for lane_id in lanes:
        lane_lower = lane_id.lower()
        if "b1b2" in lane_lower or "a1b1" in lane_lower:
            counts["north"] += len(traci.lane.getLastStepVehicleIDs(lane_id))
        elif "b2b1" in lane_lower or "c1b1" in lane_lower:
            counts["south"] += len(traci.lane.getLastStepVehicleIDs(lane_id))
        elif "b0b1" in lane_lower or "b1c1" in lane_lower:
            counts["east"] += len(traci.lane.getLastStepVehicleIDs(lane_id))
        elif "b1a1" in lane_lower or "c1b1" in lane_lower:
            counts["west"] += len(traci.lane.getLastStepVehicleIDs(lane_id))
    return counts


def _detect_tls_id(traci, preferred: str = "") -> str | None:
    try:
        ids = list(traci.trafficlight.getIDList())
    except Exception:
        return None
    if preferred and preferred in ids:
        return preferred
    return ids[0] if ids else None


def run_real_pre_system(cfg: SumoDemoConfig, *, out_csv: str | None = None) -> List[SumoStepResult]:
    traci = _import_runtime()
    if traci is None:
        from .traci_runner import run_pre_system
        return run_pre_system(cfg, out_csv=out_csv)

    sumo_binary = Path(ensure_sumo_home() or "") / "bin" / "sumo.exe"
    if not sumo_binary.exists():
        from .traci_runner import run_pre_system
        return run_pre_system(cfg, out_csv=out_csv)

    cmd = [str(sumo_binary), "-c", str(Path(cfg.sumocfg_file))]
    traci.start(cmd)
    controller = SumoBaselineController(green_seconds=20)
    history: List[SumoStepResult] = []
    try:
        tls_id = _detect_tls_id(traci, cfg.baseline_tls_id)
        step = 0
        while traci.simulation.getMinExpectedNumber() > 0 and step < cfg.sim_steps:
            traci.simulationStep()
            lane_counts = _lane_counts_from_sumo(traci, tls_id or cfg.baseline_tls_id)
            result = controller.step(lane_counts)
            result.step = step
            history.append(result)
            step += 1
    finally:
        traci.close()

    if out_csv:
        _write_csv(history, out_csv)
    return history



def run_real_post_system(cfg: SumoDemoConfig, *, out_csv: str | None = None) -> List[SumoStepResult]:
    traci = _import_runtime()
    if traci is None:
        from .traci_runner import run_post_system
        return run_post_system(cfg, out_csv=out_csv)

    sumo_binary = Path(ensure_sumo_home() or "") / "bin" / "sumo.exe"
    if not sumo_binary.exists():
        from .traci_runner import run_post_system
        return run_post_system(cfg, out_csv=out_csv)

    cmd = [str(sumo_binary), "-c", str(Path(cfg.sumocfg_file))]
    traci.start(cmd)
    controller = SumoAdaptiveController()
    history: List[SumoStepResult] = []
    try:
        tls_id = _detect_tls_id(traci, cfg.adaptive_tls_id)
        step = 0
        emergency_start = cfg.emergency_start_step
        emergency_end = cfg.emergency_end_step
        while traci.simulation.getMinExpectedNumber() > 0 and step < cfg.sim_steps:
            traci.simulationStep()
            emergency = emergency_start <= step < emergency_end
            lane_counts = _lane_counts_from_sumo(traci, tls_id or cfg.adaptive_tls_id)
            result = controller.step(lane_counts)
            result.step = step
            result.emergency_active = emergency
            if emergency:
                result.corridor_lane = result.active_lane
            history.append(result)
            step += 1
    finally:
        traci.close()

    if out_csv:
        _write_csv(history, out_csv)
    return history
