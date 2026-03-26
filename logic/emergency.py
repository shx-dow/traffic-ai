
from __future__ import annotations

from typing import Dict

LANES = ("north", "south", "east", "west")


def is_emergency_active(
    *,
    vision_emergency: bool = False,
    gps_emergency: bool = False,
    manual_trigger: bool = False,
) -> bool:
    """Return True if any emergency signal source is active."""
    return bool(vision_emergency or gps_emergency or manual_trigger)


def _normalize_signal_state(signal_state: Dict[str, str] | str) -> Dict[str, str]:
    if isinstance(signal_state, dict):
        return {lane: signal_state.get(lane, "RED") for lane in LANES}

    state = {lane: "RED" for lane in LANES}
    lane_name = str(signal_state).lower()
    if lane_name in state:
        state[lane_name] = "GREEN"
    return state


def apply_emergency_override(
    signal_state: Dict[str, str] | str,
    emergency_active: bool,
    corridor_lane: str | None = None,
) -> Dict[str, str]:
    """Return lane-wise RED/GREEN signal state with emergency override."""
    state = _normalize_signal_state(signal_state)

    if not emergency_active:
        return state

    if corridor_lane is None:
        return state

    lane_name = corridor_lane.lower()
    if lane_name not in LANES:
        return state

    return {lane: ("GREEN" if lane == lane_name else "RED") for lane in LANES}
