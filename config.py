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

# Real traffic videos: place `real_traffic.zip` in this folder (or extract it there).
# Test harness auto-extracts the zip once if no video files are found yet.
# Original dataset: UniDataPro real-time-traffic-video — license CC BY-NC-ND 4.0 where applicable.
REAL_TIME_TRAFFIC_ASSET_DIR = "assets/real_time_traffic"
REAL_TRAFFIC_ZIP_NAME = "real_traffic.zip"

# FindVehicle (text NER). CoNLL: FindVehicle_train.txt / FindVehicle_test.txt; jsonl optional.
# Source: https://github.com/GuanRunwei/FindVehicle
FINDVEHICLE_DIR = "assets/findvehicle"
FINDVEHICLE_TRAIN_TXT = "assets/findvehicle/FindVehicle_train.txt"
FINDVEHICLE_TEST_TXT = "assets/findvehicle/FindVehicle_test.txt"
FINDVEHICLE_TRAIN_JSONL = "assets/findvehicle/FindVehicle_train.jsonl"

# When True, VehicleDetector adds a ``fusion`` dict (video + FindVehicle ontology); required keys unchanged.
DETECTOR_ENRICH_CROSS_DATASET = True
VISION_TRAFFIC_DATASET_NAME = "real_time_traffic"
FINDVEHICLE_SCHEMA_NAME = "FindVehicle"

# Ambulance / emergency (COCO yolov8n has no ambulance class).
# - "yolo_world": second pass with YOLOWorld + text "ambulance" (needs CLIP in requirements.txt).
# - "aux_weights": custom YOLO .pt with an "ambulance" class at AMBULANCE_AUX_MODEL_PATH.
# - "none": only explicit ambulance if main model outputs that class name.
AMBULANCE_DETECTION_MODE = "yolo_world"
AMBULANCE_WORLD_MODEL = "yolov8s-worldv2.pt"
AMBULANCE_AUX_MODEL_PATH = "assets/models/ambulance.pt"
# YOLOWorld open-vocab can miss tight crops; lower = more recall (more false positives).
AMBULANCE_CONFIDENCE = 0.25
