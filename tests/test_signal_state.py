from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic.signal_state import (SignalStateSensor, parse_roi_arg,
                                resolve_signal_reading)


def test_parse_roi_arg_parses_coordinates():
    roi = parse_roi_arg("10,20,30,40")
    assert roi == (10, 20, 30, 40)


def test_video_signal_sensor_detects_red():
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    frame[10:50, 10:50] = (0, 0, 255)
    sensor = SignalStateSensor(source="video", roi=(0, 0, 60, 60))

    reading = sensor.read(frame)
    assert reading is not None
    assert reading.state == "RED"
    assert reading.source == "video"


def test_video_signal_sensor_returns_none_when_no_signal_pixels():
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    sensor = SignalStateSensor(source="video", roi=(0, 0, 60, 60))

    reading = sensor.read(frame)
    assert reading is None


def test_video_signal_sensor_rejects_tiny_noise_blob():
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    frame[10:12, 10:12] = (0, 0, 255)
    sensor = SignalStateSensor(source="video", roi=(0, 0, 60, 60))

    reading = sensor.read(frame)
    assert reading is None


def test_effective_roi_clamps_to_frame_bounds():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    sensor = SignalStateSensor(source="video", roi=(-20, -5, 160, 140))

    roi = sensor.get_effective_roi(frame)
    assert roi == (0, 0, 100, 100)


def test_resolve_signal_reading_uses_controller_fallback_when_missing():
    reading = resolve_signal_reading(
        None,
        requested_source="video",
        fallback_mode="controller",
        signal_states={"north": "GREEN", "south": "RED", "east": "RED", "west": "RED"},
        camera_lane="north",
    )
    assert reading is not None
    assert reading.state == "GREEN"
    assert reading.source == "controller_fallback"


def test_resolve_signal_reading_respects_none_fallback_mode():
    reading = resolve_signal_reading(
        None,
        requested_source="video",
        fallback_mode="none",
        signal_states={"north": "GREEN"},
        camera_lane="north",
    )
    assert reading is None


if __name__ == "__main__":
    test_parse_roi_arg_parses_coordinates()
    test_video_signal_sensor_detects_red()
    test_video_signal_sensor_returns_none_when_no_signal_pixels()
    test_video_signal_sensor_rejects_tiny_noise_blob()
    test_effective_roi_clamps_to_frame_bounds()
    test_resolve_signal_reading_uses_controller_fallback_when_missing()
    test_resolve_signal_reading_respects_none_fallback_mode()
    print("PASS test_signal_state")
