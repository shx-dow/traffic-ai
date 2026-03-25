from __future__ import annotations

from typing import Dict, List

from logic.counter import LaneCounter

LANES = ("north", "south", "east", "west")


def select_corridor_lane(
    vehicles: List[Dict],
    lane_counts: Dict[str, int],
    lane_counter: LaneCounter,
    fallback_lane: str,
    last_corridor_lane: str | None = None,
) -> str:
    ambulance_votes = {lane: 0 for lane in LANES}

    for vehicle in vehicles:
        if vehicle.get("class") != "ambulance":
            continue
        bbox = vehicle.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        lane = lane_counter.assign_lane(bbox)
        if lane in ambulance_votes:
            ambulance_votes[lane] += 1

    if any(ambulance_votes.values()):
        return max(ambulance_votes.items(), key=lambda item: item[1])[0]

    if last_corridor_lane in LANES:
        return last_corridor_lane

    if lane_counts:
        return max(lane_counts.items(), key=lambda item: item[1])[0]

    return fallback_lane
