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
# - "custom": custom trained YOLO model from assets/models/ambulance.pt (highest priority)
# - "yolo_world": second pass with YOLOWorld + text "ambulance" (fallback)
# - "aux_weights": legacy aux model path (deprecated)
# - "none": only explicit ambulance if main model outputs that class name.
AMBULANCE_DETECTION_MODE = "custom"  # Prioritize custom trained model
AMBULANCE_WORLD_MODEL = "yolov8s-worldv2.pt"
AMBULANCE_CUSTOM_MODEL_PATH = "assets/models/ambulance.pt"  # Custom trained ambulance model
AMBULANCE_AUX_MODEL_PATH = "assets/models/ambulance.pt"  # Legacy compatibility
# YOLOWorld open-vocab can miss tight crops; lower = more recall (more false positives).
# Custom model: 0.3 (higher precision), YOLOWorld fallback: 0.05 (higher recall)
AMBULANCE_CONFIDENCE = 0.3  # Default confidence for custom model
AMBULANCE_WORLD_CONFIDENCE = 0.05  # Lower threshold for YOLOWorld fallback

# GPS-based ambulance detection
CAMERA_LAT = 26.9124  # Camera latitude (Jaipur, India)
CAMERA_LON = 75.7873  # Camera longitude (Jaipur, India)
GPS_SERVER_URL = "http://localhost:8000"  # GPS server endpoint
EMERGENCY_DISTANCE_KM = 0.5  # Distance threshold for GPS emergency detection (0.5 km)
