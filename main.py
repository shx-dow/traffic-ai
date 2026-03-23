
from __future__ import annotations

import argparse
import signal
import sys
import time
from typing import Dict

import cv2

from logic.counter import LaneCounter
from logic.runtime import select_corridor_lane
from logic.signal import SignalController
from utils.helpers import run_repo_pipeline
from vision.detector import VehicleDetector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI Traffic Management — main runtime")
    p.add_argument("--video-source", default=0,
                   help="Video source (0 for webcam or path to video file)")
    p.add_argument("--model-path", default="yolov8n.pt",
                   help="Path to YOLO model weights")
    p.add_argument("--frame-width", type=int, default=1280)
    p.add_argument("--frame-height", type=int, default=720)
    p.add_argument("--headless", action="store_true",
                   help="Run without opening a display window (useful for CI)")
    p.add_argument("--skip-pipeline", action="store_true", help="Skip repo validation steps")
    p.add_argument("--pipeline-synthetic-frames", type=int, default=3, help="Frames for detector --synthetic test")
    p.add_argument("--pipeline-headless-frames", type=int, default=10, help="Frames for run_headless_main.py")
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
    if not args.skip_pipeline:
        gps_proc = run_repo_pipeline(
            synthetic_frames=args.pipeline_synthetic_frames,
            headless_frames=args.pipeline_headless_frames,
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
    counter = LaneCounter(frame_width, frame_height)
    signal_ctrl = SignalController()

    lanes = ["north", "south", "east", "west"]
    active_index = 0
    frame_counter = 0
    green_times: Dict[str, int] = {lane: 15 for lane in lanes}

    fps = 30
    frame_delay_ms = int(1000 / fps)

    print("Starting main loop. Press 'q' in window to quit.")

    while not ge.exit and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of stream or cannot read frame")
            break

        detection = detector.detect(frame)

        lane_counts = counter.count_per_lane(detection["vehicles"])
        green_times = signal_ctrl.calculate_green_times(lane_counts)

        if detection.get("emergency") and signal_ctrl.mode != "EMERGENCY":
            corridor = select_corridor_lane(
                vehicles=detection["vehicles"],
                lane_counts=lane_counts,
                lane_counter=counter,
                fallback_lane=lanes[active_index],
            )
            signal_ctrl.override_for_emergency(corridor)

        if not detection.get("emergency") and signal_ctrl.mode == "EMERGENCY":
            signal_ctrl.resume_adaptive()

        if signal_ctrl.mode == "EMERGENCY":
            signal_states = signal_ctrl.current_state
        else:
            active_lane = lanes[active_index]
            signal_states = signal_ctrl.get_current_signal_state(active_lane)

            frames_needed = green_times.get(active_lane, 15) * fps
            if frame_counter >= frames_needed:
                active_index = (active_index + 1) % len(lanes)
                frame_counter = 0

            frame_counter += 1

        display = frame.copy()
        cv2.putText(display, f"Mode: {signal_ctrl.mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(display, f"Counts: N{lane_counts['north']} S{lane_counts['south']} E{lane_counts['east']} W{lane_counts['west']}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if not args.headless:
            cv2.imshow("Traffic System", display)
            key = cv2.waitKey(frame_delay_ms) & 0xFF
            if key == ord("q"):
                print("'q' pressed — exiting")
                break

        if args.headless:
            time.sleep(1.0 / fps)

    cap.release()
    if not args.headless:
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
