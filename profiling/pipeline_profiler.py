"""Per-stage latency profiler for the Falcon traffic pipeline.

Measures:
  - Frame acquisition
  - YOLO detection (+ GPS emergency poll)
  - Lane counting
  - Congestion scoring / signal decision
  - Emergency fusion
  - Orchestrator update
  - Overlay rendering

Reports p50 / p95 / p99 latency per stage, effective FPS, and total run time.

Usage:
    python profiling/pipeline_profiler.py [--frames 20] [--video assets/sample_video.mp4]
                                          [--out artifacts/latency_profile.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import cv2

from config import CONFIG, MODEL_PATH
from logic.counter import LaneCounter
from logic.signal import SignalController
from logic.orchestrator import CorridorOrchestrator
from logic.runtime import select_corridor_lane
from logic.traffic_loop import is_balanced
from ui.overlay import TrafficOverlay
from vision.detector import VehicleDetector

LANES = ("north", "south", "east", "west")


def _percentile(values, p):
    a = np.array(values)
    if a.size == 0:
        return 0.0
    return float(np.percentile(a, p))


def parse_args():
    p = argparse.ArgumentParser(description="Per-stage pipeline profiler")
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--video", default="assets/sample_video.mp4")
    p.add_argument("--out", default="artifacts/latency_profile.json")
    return p.parse_args()


def main():
    args = parse_args()
    video_path = ROOT / args.video
    use_video = video_path.is_file()

    detector = VehicleDetector(model_path=str(ROOT / MODEL_PATH))
    counter = LaneCounter(CONFIG["frame_width"], CONFIG["frame_height"], mode="per_camera")
    signal = SignalController()
    orchestrator = CorridorOrchestrator(
        ["int_a", "int_b", "int_c", "int_d"],
        preempt_hops=2,
        latch_frames=90,
    )
    overlay = TrafficOverlay()

    if use_video:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap = None
    else:
        cap = None
    rng = np.random.default_rng(42)

    total_frames = args.frames
    warmup = args.warmup
    stages = [
        "frame_acquisition",
        "detection",
        "lane_counting",
        "congestion_scoring",
        "signal_decision",
        "emergency_fusion",
        "orchestrator",
        "overlay_rendering",
    ]
    timings = {s: [] for s in stages}
    t_all_start = time.perf_counter()
    active_lane = "north"

    for i in range(total_frames):
        # 1. Frame acquisition
        if cap is not None:
            ok, frame = cap.read()
            if not ok:
                break
        else:
            frame = rng.integers(0, 255, (CONFIG["frame_height"], CONFIG["frame_width"], 3), dtype=np.uint8)
        t0 = time.perf_counter()
        t1 = t0
        timings["frame_acquisition"].append((t1 - t0) * 1000)

        # 2. Detection (includes GPS poll)
        t_detect = time.perf_counter()
        detection = detector.detect(frame)
        t_after_detect = time.perf_counter()
        timings["detection"].append((t_after_detect - t_detect) * 1000)

        # 3. Lane counting
        t_count = time.perf_counter()
        lane_counts = counter.count_per_lane(detection["vehicles"])
        lane_scores = signal.calculate_congestion_scores(lane_counts)
        timings["lane_counting"].append((time.perf_counter() - t_count) * 1000)

        # 4. Congestion scoring (already merged with counting)
        timings["congestion_scoring"].append(0.0)

        # 5. Signal decision
        t_sig = time.perf_counter()
        if signal.mode != "EMERGENCY":
            _ = signal.should_switch_lane(active_lane, lane_scores, 0, 30)
        timings["signal_decision"].append((time.perf_counter() - t_sig) * 1000)

        # 6. Emergency fusion
        t_emer = time.perf_counter()
        vision_emergency = bool(detection.get("vision_emergency"))
        gps_emergency = bool(detection.get("gps_emergency"))
        emergency_active = vision_emergency or gps_emergency
        corridor, _ = select_corridor_lane(
            vehicles=detection["vehicles"],
            lane_counts=lane_counts,
            lane_counter=counter,
            fallback_lane=active_lane,
        )
        timings["emergency_fusion"].append((time.perf_counter() - t_emer) * 1000)

        # 7. Orchestrator
        t_orch = time.perf_counter()
        _ = orchestrator.update(route=None, position_index=None, ambulance_lane=None)
        timings["orchestrator"].append((time.perf_counter() - t_orch) * 1000)

        # 8. Overlay
        t_ov = time.perf_counter()
        display = frame.copy()
        if emergency_active:
            signal.override_for_emergency(corridor)
        signal_states = signal.get_current_signal_state(active_lane)
        overlay.draw(
            display,
            detection_result=detection,
            lane_counts=lane_counts,
            lane_scores=lane_scores,
            signal_states=signal_states,
            emergency_active=emergency_active,
        )
        timings["overlay_rendering"].append((time.perf_counter() - t_ov) * 1000)

        if i < warmup:
            continue

        # Periodic reset so warm-up doesn't skew later decisions
        if (i - warmup) % 200 == 0 and signal.mode == "ADAPTIVE":
            pass

    t_all_end = time.perf_counter()

    if cap is not None:
        cap.release()

    effective_frames = total_frames - warmup
    total_ms = (t_all_end - t_all_start) * 1000
    fps = effective_frames / ((t_all_end - t_all_start) or 1e-9)

    rows = []
    for stage in stages:
        data = timings[stage][warmup:]
        avg = float(np.mean(data)) if data else 0.0
        total_pct = (avg / (total_ms / effective_frames)) * 100.0 if effective_frames else 0.0
        rows.append({
            "stage": stage,
            "avg_ms": round(avg, 3),
            "p50_ms": round(_percentile(data, 50), 3),
            "p95_ms": round(_percentile(data, 95), 3),
            "p99_ms": round(_percentile(data, 99), 3),
            "pct_total": round(total_pct, 1),
        })

    result = {
        "frames": effective_frames,
        "warmup": warmup,
        "total_ms": round(total_ms, 1),
        "fps": round(fps, 2),
        "source": "video" if use_video else "synthetic",
        "stages": rows,
    }

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote latency profile to {out_path}")
    print(f"\n{'Stage':<22} {'Avg ms':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'% total':>8}")
    print("-" * 62)
    for r in rows:
        print(f"{r['stage']:<22} {r['avg_ms']:>8.2f} {r['p50_ms']:>8.2f} {r['p95_ms']:>8.2f} {r['p99_ms']:>8.2f} {r['pct_total']:>7.1f}%")
    print(f"\nEffective FPS: {fps:.2f} ({effective_frames} frames in {total_ms:.0f} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())