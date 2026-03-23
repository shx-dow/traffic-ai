from __future__ import annotations

from typing import Dict, List

LANES = ("north", "south", "east", "west")


def compute_metrics(history: List[Dict], fps: int) -> Dict:
    if fps <= 0:
        raise ValueError("fps must be positive")

    lane_wait_units = {lane: 0 for lane in LANES}
    max_queue = {lane: 0 for lane in LANES}
    served_units = {lane: 0 for lane in LANES}
    emergency_frames = 0

    for frame in history:
        counts = frame.get("lane_counts", {})
        green = set(frame.get("green_lanes", []))
        mode = frame.get("mode", "ADAPTIVE")

        if mode == "EMERGENCY":
            emergency_frames += 1

        for lane in LANES:
            queue = int(counts.get(lane, 0))
            if queue > max_queue[lane]:
                max_queue[lane] = queue
            if queue > 0 and lane not in green:
                lane_wait_units[lane] += queue
            if lane in green and queue > 0:
                served_units[lane] += queue

    total_wait_units = sum(lane_wait_units.values())
    avg_wait_seconds_per_lane = {
        lane: lane_wait_units[lane] / fps for lane in LANES
    }
    throughput_score = sum(served_units.values())

    return {
        "frames": len(history),
        "fps": fps,
        "avg_wait_seconds_per_lane": avg_wait_seconds_per_lane,
        "avg_wait_seconds_overall": (total_wait_units / len(LANES)) / fps if history else 0.0,
        "max_queue_per_lane": max_queue,
        "throughput_score": throughput_score,
        "emergency_duration_seconds": emergency_frames / fps,
    }


def compare_metrics(baseline: Dict, adaptive: Dict) -> Dict:
    b_wait = float(baseline.get("avg_wait_seconds_overall", 0.0))
    a_wait = float(adaptive.get("avg_wait_seconds_overall", 0.0))
    b_thr = float(baseline.get("throughput_score", 0.0))
    a_thr = float(adaptive.get("throughput_score", 0.0))

    wait_reduction_pct = 0.0
    if b_wait > 0:
        wait_reduction_pct = ((b_wait - a_wait) / b_wait) * 100.0

    throughput_gain_pct = 0.0
    if b_thr > 0:
        throughput_gain_pct = ((a_thr - b_thr) / b_thr) * 100.0

    return {
        "wait_reduction_pct": wait_reduction_pct,
        "throughput_gain_pct": throughput_gain_pct,
    }
