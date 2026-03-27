from __future__ import annotations

from typing import Any, Dict, List

LANES = ("north", "south", "east", "west")


def _normalize_counts(lane_counts: Dict[str, int] | None) -> Dict[str, int]:
    lane_counts = lane_counts or {}
    return {lane: int(lane_counts.get(lane, 0)) for lane in LANES}


def run_signal_simulation(controller: Any, frame_events: List[Dict[str, Any]], fps: int = 30) -> List[Dict[str, Any]]:
    active_lane = "north"
    frame_counter = 0
    last_corridor_lane = active_lane
    history: List[Dict[str, Any]] = []

    for event in frame_events:
        lane_counts = _normalize_counts(event.get("lane_counts"))
        emergency = bool(event.get("emergency", False))
        corridor_lane = event.get("corridor_lane")

        if emergency and controller.mode != "EMERGENCY":
            if corridor_lane is None:
                corridor_lane = last_corridor_lane or active_lane
            controller.override_for_emergency(corridor_lane)
            last_corridor_lane = corridor_lane

        if not emergency and controller.mode == "EMERGENCY":
            controller.resume_adaptive()
            frame_counter = 0

        if controller.mode == "EMERGENCY":
            signal_states = dict(controller.current_state)
        else:
            signal_states = controller.get_current_signal_state(active_lane)

            should_switch = controller.should_switch_lane(
                active_lane=active_lane,
                lane_counts=lane_counts,
                frame_counter=frame_counter,
                fps=fps,
            )
            if should_switch:
                controller.record_cycle(active_lane, lane_counts)
                active_lane = controller.choose_next_lane(active_lane, lane_counts)
                frame_counter = 0
            else:
                frame_counter += 1

        green_lanes = [lane for lane, state in signal_states.items() if state == "GREEN"]
        history.append(
            {
                "mode": controller.mode,
                "active_lane": active_lane,
                "green_lanes": green_lanes,
                "lane_counts": lane_counts,
            }
        )

    return history
