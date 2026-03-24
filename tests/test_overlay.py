from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.overlay import TrafficOverlay


def test_overlay_draw_runs_and_marks_frame():
    overlay = TrafficOverlay()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    detection = {
        "vehicles": [
            {"class": "car", "confidence": 0.91, "bbox": [100.0, 120.0, 220.0, 260.0]},
            {"class": "ambulance", "confidence": 0.88, "bbox": [300.0, 180.0, 420.0, 320.0]},
        ],
        "emergency": True,
    }
    counts = {"north": 5, "south": 2, "east": 7, "west": 1}
    signals = {"north": "GREEN", "south": "RED", "east": "RED", "west": "RED"}

    out = overlay.draw(frame.copy(), detection, counts, signals, emergency_active=True)

    assert out.shape == frame.shape
    assert out.dtype == frame.dtype
    assert int(out.sum()) > 0


if __name__ == "__main__":
    test_overlay_draw_runs_and_marks_frame()
    print("PASS test_overlay")
