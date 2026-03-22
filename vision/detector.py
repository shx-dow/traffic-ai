"""YOLOv8-based vehicle detection — single-frame inference for the traffic pipeline."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

# COCO (yolov8n.pt) vehicle-related classes we report downstream.
VEHICLE_CLASSES = frozenset({"car", "truck", "bus", "motorcycle", "bicycle"})
# Not in stock COCO; supported for custom weights / future models.
EMERGENCY_CLASS_NAMES = frozenset({"ambulance"})

# Classes that appear in `vehicles` and `count` (vehicles + emergency types).
TRACKED_CLASSES = VEHICLE_CLASSES | EMERGENCY_CLASS_NAMES

CONFIDENCE_THRESHOLD = 0.4


class VehicleDetector:
    """Runs Ultralytics YOLO on BGR frames from OpenCV and returns a fixed contract dict."""

    def __init__(self, model_path: str = "yolov8n.pt") -> None:
        self._model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Input  : numpy array (one video frame from OpenCV, BGR uint8)
        Output : {
            'vehicles': [{'class': str, 'confidence': float, 'bbox': [x1,y1,x2,y2]}, ...],
            'count': int,
            'emergency': bool,
            'raw_result': Results,
        }
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy ndarray (OpenCV BGR image).")

        results: List[Results] = self._model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )
        raw_result: Results = results[0]

        vehicles: List[Dict[str, Any]] = []
        names = raw_result.names

        if raw_result.boxes is None or len(raw_result.boxes) == 0:
            return {
                "vehicles": [],
                "count": 0,
                "emergency": False,
                "raw_result": raw_result,
            }

        for box in raw_result.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = str(names[cls_id]).lower()
            if cls_name not in TRACKED_CLASSES:
                continue
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy()
            bbox = [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])]
            vehicles.append(
                {
                    "class": cls_name,
                    "confidence": conf,
                    "bbox": bbox,
                }
            )

        emergency = any(v["class"] in EMERGENCY_CLASS_NAMES for v in vehicles)

        return {
            "vehicles": vehicles,
            "count": len(vehicles),
            "emergency": emergency,
            "raw_result": raw_result,
        }
