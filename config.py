"""Central configuration for the traffic AI prototype.

Keep shared constants here so detection, traffic logic, and overlay rendering
stay consistent across the project.
"""

from __future__ import annotations

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

SIGNAL_DECISION_THRESHOLDS = {
    "balanced_gap": 2,
    "heavy_lane_min_count": 5,
}

FRAME_DISPLAY_SIZE = (1280, 720)

MODEL_PATH = "yolov8n.pt"
VIDEO_SOURCE = "assets/sample_video.mp4"
