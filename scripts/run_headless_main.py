"""Headless runner for the traffic pipeline used in CI/local tests.

This script mimics the main frame loop but does not open any GUI windows.
It runs a fixed number of iterations using a black frame so you can observe
the detector -> counter -> signal interactions in a reproducible way.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Ensure repo root is on sys.path so `vision/` and `logic/` imports work.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from logic.counter import LaneCounter
from logic.signal import SignalController
from vision.detector import VehicleDetector

# logic.emergency in this repo contains only helper functions for demo.
# We'll simulate a minimal EmergencyCorridorManager used by main.py.


class EmergencyCorridorManager:
    def __init__(self, signal_controller):
        self.active = False
        self.signal = signal_controller

    def activate(self, corridors):
        # corridors is ignored in this simulation; pick the first as corridor
        self.active = True
        if corridors:
            self.signal.override_for_emergency(corridors[0])

    def deactivate(self):
        self.active = False
        self.signal.resume_adaptive()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless runner for the traffic pipeline")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames to run")
    return parser.parse_args()


def run(frames: int = 60) -> None:
    # Config-like defaults matching tests
    frame_width = 1280
    frame_height = 720

    detector = VehicleDetector("yolov8n.pt")
    counter = LaneCounter(frame_width, frame_height)
    signal = SignalController()
    emergency = EmergencyCorridorManager(signal)

    lanes = ["north", "south", "east", "west"]
    active_index = 0
    frame_counter = 0
    green_times = {lane: 15 for lane in lanes}

    for i in range(frames):
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        detection = detector.detect(frame)
        lane_counts = counter.count_per_lane(detection["vehicles"])  # uses our code
        green_times = signal.calculate_green_times(lane_counts)

        # Emergency activation simulated if detector flags emergency
        if detection.get("emergency") and not emergency.active:
            emergency.activate(["int_1"])  # simulated corridor ids

        if signal.mode == "EMERGENCY":
            signal_states = signal.current_state
        else:
            active_lane = lanes[active_index]
            signal_states = signal.get_current_signal_state(active_lane)

            frames_needed = green_times.get(active_lane, 15) * 30
            if frame_counter >= frames_needed:
                active_index = (active_index + 1) % len(lanes)
                frame_counter = 0

            frame_counter += 1

        # Print a concise summary per frame
        print(f"Frame {i+1}/{frames}: count={detection['count']}, lanes={lane_counts}, mode={signal.mode}")
        time.sleep(0.01)


if __name__ == "__main__":
    args = parse_args()
    run(args.frames)
