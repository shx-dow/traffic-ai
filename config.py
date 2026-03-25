"""Central configuration — all shared constants live here."""

from __future__ import annotations

CONFIG = {
    "video_source": 0,
    "model_path": "yolov8n.pt",
    "frame_width": 1280,
    "frame_height": 720,
    "min_green_time": 10,
    "max_green_time": 60,
    "detection_confidence": 0.4,
    "display_window": True,
    "save_output": False,
    "output_path": "artifacts/demo_output.mp4",
    "show_kpi_hud": True,
    "metrics_log_path": "",
    "metrics_log_every": 30,
    "counter_mode": "per_camera",
    "camera_lane": "north",
    "signal_state_source": "none",
    "signal_state_api_url": "",
    "signal_state_roi": "",
    "signal_state_api_timeout": 0.4,
    "approach_roi": "",
    "queue_roi": "",
}

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
DEFAULT_SIGNAL_MODE = "adaptive"
BASELINE_GREEN_SECONDS = 20

REAL_TIME_TRAFFIC_ASSET_DIR = "assets/real_time_traffic"  # place real_traffic.zip here
REAL_TRAFFIC_ZIP_NAME = "real_traffic.zip"

# FindVehicle NER dataset — https://github.com/GuanRunwei/FindVehicle
FINDVEHICLE_DIR = "assets/findvehicle"
FINDVEHICLE_TRAIN_TXT = "assets/findvehicle/FindVehicle_train.txt"
FINDVEHICLE_TEST_TXT = "assets/findvehicle/FindVehicle_test.txt"
FINDVEHICLE_TRAIN_JSONL = "assets/findvehicle/FindVehicle_train.jsonl"

DETECTOR_ENRICH_CROSS_DATASET = False
VISION_TRAFFIC_DATASET_NAME = "real_time_traffic"
FINDVEHICLE_SCHEMA_NAME = "FindVehicle"

# "custom" | "yolo_world" | "aux_weights" | "none"
AMBULANCE_DETECTION_MODE = "custom"
AMBULANCE_WORLD_MODEL = "yolov8s-worldv2.pt"
AMBULANCE_CUSTOM_MODEL_PATH = "assets/models/ambulance.pt"
AMBULANCE_AUX_MODEL_PATH = "assets/models/ambulance_aux.pt"
AMBULANCE_CONFIDENCE = 0.3
AMBULANCE_WORLD_CONFIDENCE = 0.05  # lower threshold for YOLOWorld (higher recall)

# GPS emergency detection
CAMERA_LAT = 26.9124  # Jaipur, India
CAMERA_LON = 75.7873
GPS_SERVER_URL = "http://localhost:8000"
EMERGENCY_DISTANCE_KM = 0.5
EMERGENCY_ETA_SECONDS = 45
GPS_REQUEST_TIMEOUT_SECONDS = 0.5
EMERGENCY_LATCH_SECONDS = 3
