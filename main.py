
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import cv2

from config import (BASELINE_GREEN_SECONDS, CONFIG, DEFAULT_SIGNAL_MODE,
                    EMERGENCY_LATCH_SECONDS, MODEL_PATH)
from logic.baseline_signal import BaselineSignalController
from logic.counter import LaneCounter
from logic.live_metrics import LiveMetricsTracker
from logic.runtime import select_corridor_lane
from logic.signal import SignalController
from ui.overlay import TrafficOverlay
from utils.helpers import run_repo_pipeline
from vision.detector import VehicleDetector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI Traffic Management — main runtime")
    p.add_argument("--video-source", default=CONFIG["video_source"],
                   help="Video source (0 for webcam or path to video file)")
    p.add_argument("--model-path", default=CONFIG.get("model_path", MODEL_PATH),
                   help="Path to YOLO model weights")
    p.add_argument("--frame-width", type=int, default=CONFIG["frame_width"])
    p.add_argument("--frame-height", type=int, default=CONFIG["frame_height"])
    p.add_argument("--camera-lane", choices=("north", "south", "east", "west"), default=CONFIG.get("camera_lane", "north"), help="Lane represented by this camera in per-camera mode")
    p.add_argument("--top-down-view", action="store_true", help="Use center-split top-down lane assignment instead of per-camera mode")
    p.add_argument("--mode", choices=("adaptive", "baseline"), default=DEFAULT_SIGNAL_MODE)
    p.add_argument("--baseline-green-seconds", type=int, default=BASELINE_GREEN_SECONDS)
    p.add_argument("--headless", action="store_true",
                    help="Run without opening a display window (useful for CI)")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 means no limit)")
    p.add_argument("--save-output", action="store_true", default=CONFIG.get("save_output", False))
    p.add_argument("--output-path", default=CONFIG.get("output_path", "artifacts/demo_output.mp4"))
    p.add_argument("--run-pipeline", action="store_true", help="Run core repo validation checks before main loop")
    p.add_argument("--hide-kpi-hud", action="store_true", help="Disable the live KPI panel overlay")
    p.add_argument("--metrics-log-path", default=CONFIG.get("metrics_log_path", ""), help="Write live KPI snapshots to JSONL")
    p.add_argument("--metrics-log-every", type=int, default=int(CONFIG.get("metrics_log_every", 30)), help="Log KPIs every N frames")
    p.add_argument(
        "--disable-gps",
        action="store_true",
        help="Do not start gps_server and do not run GPS integration tests",
    )
    return p.parse_args()


def open_capture(source) -> cv2.VideoCapture:
    try:
        idx = int(source)
        cap = cv2.VideoCapture(idx)
    except Exception:
        cap = cv2.VideoCapture(str(source))
    return cap


def get_capture_dimensions(cap: cv2.VideoCapture, default_width: int, default_height: int) -> tuple[int, int]:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0:
        width = default_width
    if height <= 0:
        height = default_height
    return width, height


class GracefulExit:
    def __init__(self):
        self.exit = False

    def __call__(self, signum, frame):
        print(f"Received signal {signum}, exiting gracefully...")
        self.exit = True


def main() -> None:
    args = parse_args()

    gps_proc = None
    if args.run_pipeline:
        gps_proc = run_repo_pipeline(
            disable_gps=args.disable_gps,
        )

    # Setup graceful shutdown handlers
    ge = GracefulExit()
    signal.signal(signal.SIGINT, ge)
    try:
        signal.signal(signal.SIGTERM, ge)
    except Exception:
        pass

    cap = open_capture(args.video_source)
    if not cap.isOpened():
        print("Failed to open video source", args.video_source)
        sys.exit(1)

    frame_width, frame_height = get_capture_dimensions(cap, args.frame_width, args.frame_height)

    detector = VehicleDetector(model_path=args.model_path)
    counter_mode = "top_down" if args.top_down_view else str(CONFIG.get("counter_mode", "per_camera"))
    show_directional_counts = counter_mode == "top_down"
    counter = LaneCounter(frame_width, frame_height, mode=counter_mode, camera_lane=args.camera_lane)
    overlay = TrafficOverlay()

    if args.mode == "baseline":
        signal_ctrl = BaselineSignalController(green_seconds=args.baseline_green_seconds)
    else:
        signal_ctrl = SignalController()

    lanes = ["north", "south", "east", "west"]
    active_lane = lanes[0]
    frame_counter = 0
    processed_frames = 0
    emergency_hold_frames = 0
    last_corridor_lane = active_lane

    fps = 30
    metrics = LiveMetricsTracker(fps=fps)
    frame_delay_ms = int(1000 / fps)
    display_window = (not args.headless) and bool(CONFIG.get("display_window", True))
    show_kpi_hud = bool(CONFIG.get("show_kpi_hud", True)) and (not args.hide_kpi_hud)

    metrics_log_path = str(args.metrics_log_path or "").strip()
    metrics_log_every = max(1, int(args.metrics_log_every))
    metrics_log_file = None
    if metrics_log_path:
        log_path = Path(metrics_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_log_file = log_path.open("w", encoding="utf-8")

    writer = None
    if args.save_output:
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (frame_width, frame_height))
        if not writer.isOpened():
            print(f"Warning: failed to open video writer at {out_path}")
            writer = None

    print(f"Starting main loop in {args.mode} mode. Press 'q' in window to quit.")

    while not ge.exit and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of stream or cannot read frame")
            break

        detection = detector.detect(frame)

        lane_counts = counter.count_per_lane(detection["vehicles"])
        emergency_seen = bool(detection.get("emergency"))
        if emergency_seen:
            emergency_hold_frames = int(max(1, EMERGENCY_LATCH_SECONDS * fps))
        elif emergency_hold_frames > 0:
            emergency_hold_frames -= 1

        emergency_active = emergency_seen or emergency_hold_frames > 0

        if emergency_active and signal_ctrl.mode != "EMERGENCY":
            corridor = select_corridor_lane(
                vehicles=detection["vehicles"],
                lane_counts=lane_counts,
                lane_counter=counter,
                fallback_lane=active_lane,
                last_corridor_lane=last_corridor_lane,
            )
            last_corridor_lane = corridor
            signal_ctrl.override_for_emergency(corridor)

        if emergency_active and signal_ctrl.mode == "EMERGENCY" and emergency_seen:
            corridor = select_corridor_lane(
                vehicles=detection["vehicles"],
                lane_counts=lane_counts,
                lane_counter=counter,
                fallback_lane=last_corridor_lane,
                last_corridor_lane=last_corridor_lane,
            )
            if corridor != last_corridor_lane:
                last_corridor_lane = corridor
                signal_ctrl.override_for_emergency(corridor)

        if not emergency_active and signal_ctrl.mode == "EMERGENCY":
            signal_ctrl.resume_adaptive()
            frame_counter = 0

        if signal_ctrl.mode == "EMERGENCY":
            signal_states = signal_ctrl.current_state
        else:
            signal_states = signal_ctrl.get_current_signal_state(active_lane)
            signal_ctrl.record_cycle(active_lane, lane_counts)

            should_switch = signal_ctrl.should_switch_lane(
                active_lane=active_lane,
                lane_counts=lane_counts,
                frame_counter=frame_counter,
                fps=fps,
            )

            if should_switch:
                active_lane = signal_ctrl.choose_next_lane(active_lane, lane_counts)
                frame_counter = 0
            else:
                frame_counter += 1

        green_lanes = [lane for lane, state in signal_states.items() if state == "GREEN"]
        metrics.update(lane_counts=lane_counts, green_lanes=green_lanes, mode=signal_ctrl.mode)
        kpi_snapshot = metrics.snapshot()

        if metrics_log_file is not None and (processed_frames % metrics_log_every == 0):
            row = {
                "frame": processed_frames,
                "mode": signal_ctrl.mode,
                "emergency_active": emergency_active,
                "active_lane": active_lane,
                "lane_counts": lane_counts,
                **kpi_snapshot,
            }
            metrics_log_file.write(json.dumps(row) + "\n")

        display = overlay.draw(
            frame.copy(),
            detection_result=detection,
            lane_counts=lane_counts,
            signal_states=signal_states,
            emergency_active=emergency_active,
            kpi_snapshot=(kpi_snapshot if show_kpi_hud else None),
            show_directional_counts=show_directional_counts,
            camera_lane=args.camera_lane,
        )
        cv2.putText(display, f"Mode: {signal_ctrl.mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        if show_directional_counts:
            cv2.putText(display, f"Counts: N{lane_counts['north']} S{lane_counts['south']} E{lane_counts['east']} W{lane_counts['west']}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        else:
            approach_count = int(lane_counts.get(args.camera_lane, 0))
            cv2.putText(display, f"Approach {args.camera_lane.title()} Count: {approach_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if writer is not None:
            writer.write(display)

        processed_frames += 1
        if args.max_frames > 0 and processed_frames >= args.max_frames:
            print(f"Reached max frames: {args.max_frames}")
            break

        if display_window:
            cv2.imshow("Traffic System", display)
            key = cv2.waitKey(frame_delay_ms) & 0xFF
            if key == ord("q"):
                print("'q' pressed — exiting")
                break

        if args.headless:
            time.sleep(1.0 / fps)

    cap.release()
    if writer is not None:
        writer.release()

    if metrics_log_file is not None:
        metrics_log_file.close()

    if display_window:
        cv2.destroyAllWindows()

    if gps_proc is not None:
        print("Stopping gps_server.py...")
        gps_proc.terminate()
        try:
            gps_proc.wait(timeout=5)
        except Exception:
            gps_proc.kill()


if __name__ == "__main__":
    main()
