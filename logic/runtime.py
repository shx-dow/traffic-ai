from __future__ import annotations

from typing import Dict, List

from logic.counter import LaneCounter

LANES = ("north", "south", "east", "west")
CORRIDOR_CLASSES = frozenset({"ambulance", "fire_truck"})


def select_corridor_lane(
    vehicles: List[Dict],
    lane_counts: Dict[str, int],
    lane_counter: LaneCounter,
    fallback_lane: str,
    last_corridor_lane: str | None = None,
) -> tuple[str, str]:
    emergency_votes = {lane: 0 for lane in LANES}
    fallback = str(fallback_lane).lower()
    if fallback not in LANES:
        fallback = "north"

    for vehicle in vehicles:
        if vehicle.get("class") not in CORRIDOR_CLASSES:
            continue
        bbox = vehicle.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        lane = lane_counter.assign_lane(bbox)
        if lane in emergency_votes:
            emergency_votes[lane] += 1

    if any(emergency_votes.values()):
        return max(emergency_votes.items(), key=lambda item: item[1])[0], "vision_bbox"

    if last_corridor_lane in LANES:
        return str(last_corridor_lane), "sticky_last"

    return fallback, "assumed_fallback"
