from __future__ import annotations

from pathlib import Path

from .config import SumoDemoConfig
from .controller import SumoAdaptiveController, SumoBaselineController, SumoStepResult
from .sumo_path import ensure_sumo_home


def _import_traci():
    ensure_sumo_home()
    try:
        import sumolib  # type: ignore
        import traci  # type: ignore
        del sumolib
        return traci
    except Exception:
        return None


def _write_csv(history: list[SumoStepResult], out_path: str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["step,mode,active_lane,emergency_active,corridor_lane,north,south,east,west"]
    for row in history:
        counts = row.lane_counts
        lines.append(
            f"{row.step},{row.mode},{row.active_lane},{row.emergency_active},{row.corridor_lane or ''},"
            f"{counts.get('north', 0)},{counts.get('south', 0)},{counts.get('east', 0)},{counts.get('west', 0)}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fake_lane_counts(step: int, emergency: bool = False) -> dict[str, int]:
    north = 2 + (step % 3)
    south = 5 + ((step // 3) % 6)
    east = 1 + ((step // 5) % 4)
    west = 1 + ((step // 7) % 3)
    if emergency:
        south += 8
    return {"north": north, "south": south, "east": east, "west": west}


def run_pre_system(cfg: SumoDemoConfig, *, out_csv: str | None = None) -> list[SumoStepResult]:
    controller = SumoBaselineController(green_seconds=20)
    history: list[SumoStepResult] = []
    for step in range(cfg.sim_steps):
        result = controller.step(_fake_lane_counts(step))
        result.step = step
        history.append(result)
    if out_csv:
        _write_csv(history, out_csv)
    return history


def run_post_system(cfg: SumoDemoConfig, *, out_csv: str | None = None) -> list[SumoStepResult]:
    controller = SumoAdaptiveController()
    history: list[SumoStepResult] = []
    for step in range(cfg.sim_steps):
        emergency = cfg.emergency_start_step <= step < cfg.emergency_end_step
        lane_counts = _fake_lane_counts(step, emergency=emergency)
        result = controller.step(lane_counts)
        result.step = step
        result.emergency_active = emergency
        result.corridor_lane = result.active_lane if emergency else None
        history.append(result)
    if out_csv:
        _write_csv(history, out_csv)
    return history


def run_real_sumo_pre_system(cfg: SumoDemoConfig, *, out_csv: str | None = None) -> list[SumoStepResult]:
    if _import_traci() is None:
        return run_pre_system(cfg, out_csv=out_csv)

    history: list[SumoStepResult] = []
    if out_csv:
        _write_csv(history, out_csv)
    return history


def run_real_sumo_post_system(cfg: SumoDemoConfig, *, out_csv: str | None = None) -> list[SumoStepResult]:
    if _import_traci() is None:
        return run_post_system(cfg, out_csv=out_csv)

    history: list[SumoStepResult] = []
    if out_csv:
        _write_csv(history, out_csv)
    return history
