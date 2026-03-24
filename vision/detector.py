from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

VEHICLE_CLASSES = frozenset({"car", "truck", "bus", "motorcycle", "bicycle"})
EMERGENCY_CLASS_NAMES = frozenset({"ambulance"})
TRACKED_CLASSES = VEHICLE_CLASSES | EMERGENCY_CLASS_NAMES
CONFIDENCE_THRESHOLD = 0.4

# Type alias for a single detection dict
Detection = Dict[str, Any]


class VehicleDetector:
    """Runs Ultralytics YOLO on BGR frames and returns a fixed-contract dict."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        *,
        enrich_cross_dataset: Optional[bool] = None,
        ambulance_mode: Optional[str] = None,
        ambulance_confidence: Optional[float] = None,
    ) -> None:
        from config import (AMBULANCE_AUX_MODEL_PATH, AMBULANCE_CONFIDENCE,
                            AMBULANCE_CUSTOM_MODEL_PATH,
                            AMBULANCE_DETECTION_MODE,
                            AMBULANCE_WORLD_CONFIDENCE, AMBULANCE_WORLD_MODEL,
                            DETECTOR_ENRICH_CROSS_DATASET)

        self._model = YOLO(model_path)
        self._logger = logging.getLogger(__name__)
        self._enrich_cross_dataset = enrich_cross_dataset if enrich_cross_dataset is not None else DETECTOR_ENRICH_CROSS_DATASET
        self._ambulance_mode = (ambulance_mode or AMBULANCE_DETECTION_MODE).lower()
        self._ambulance_conf = float(ambulance_confidence if ambulance_confidence is not None else AMBULANCE_CONFIDENCE)
        self._ambulance_world_conf = float(AMBULANCE_WORLD_CONFIDENCE)
        self._world_weights = AMBULANCE_WORLD_MODEL
        self._world_model: Any = None
        self._ambulance_custom: Optional[YOLO] = None
        self._ambulance_aux: Optional[YOLO] = None

        if self._ambulance_mode in ("custom", "aux_weights"):
            custom_path = Path(AMBULANCE_CUSTOM_MODEL_PATH)
            if custom_path.is_file():
                try:
                    self._ambulance_custom = YOLO(str(custom_path.resolve()))
                    self._logger.info("Loaded custom ambulance model: %s", custom_path)
                except Exception:
                    self._logger.exception("Failed to load custom ambulance model: %s", custom_path)

        if self._ambulance_mode == "aux_weights" and self._ambulance_custom is None:
            aux_path = Path(AMBULANCE_AUX_MODEL_PATH)
            if aux_path.is_file():
                self._ambulance_aux = YOLO(str(aux_path.resolve()))

    # Public API

    def detect(self, frame: np.ndarray, *, video_source_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Input  : BGR numpy array (OpenCV frame)
        Output : {vehicles, count, emergency, gps_emergency, raw_result, [fusion]}
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy ndarray (OpenCV BGR image).")

        raw_result: Results = self._model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        vehicles, emergency = self._vehicles_from_coco(raw_result)
        vehicles, emergency = self._merge_ambulance_detections(frame, vehicles, emergency)

        gps_emergency = self._check_gps_emergency()
        out = {
            "vehicles": vehicles,
            "count": len(vehicles),
            "emergency": emergency or gps_emergency,
            "gps_emergency": gps_emergency,
            "raw_result": raw_result,
        }
        return self._attach_fusion(out, video_source_hint)

    # Detection helpers

    def _vehicles_from_coco(self, raw_result: Results) -> Tuple[List[Detection], bool]:
        vehicles: List[Detection] = []
        if not raw_result.boxes:
            return vehicles, False
        for box in raw_result.boxes:
            cls_name = str(raw_result.names[int(box.cls[0].item())]).lower()
            if cls_name not in TRACKED_CLASSES:
                continue
            xyxy = box.xyxy[0].cpu().numpy()
            vehicles.append({
                "class": cls_name,
                "confidence": float(box.conf[0].item()),
                "bbox": [float(v) for v in xyxy],
            })
        return vehicles, any(v["class"] in EMERGENCY_CLASS_NAMES for v in vehicles)

    def _merge_ambulance_detections(
        self, frame: np.ndarray, vehicles: List[Detection], emergency: bool
    ) -> Tuple[List[Detection], bool]:
        if self._ambulance_mode == "none":
            return vehicles, emergency

        if self._ambulance_mode in ("custom", "aux_weights") and self._ambulance_custom:
            vehicles, found = self._run_ambulance_model(frame, vehicles, self._ambulance_custom, self._ambulance_conf)
            emergency = emergency or found

        if self._ambulance_mode == "aux_weights" and self._ambulance_aux and not emergency:
            vehicles, found = self._run_ambulance_model(frame, vehicles, self._ambulance_aux, self._ambulance_conf)
            emergency = emergency or found

        if self._ambulance_mode == "yolo_world" or (self._ambulance_mode == "custom" and not self._ambulance_custom and not emergency):
            vehicles, found = self._run_yolo_world(frame, vehicles)
            emergency = emergency or found

        return vehicles, emergency

    def _run_ambulance_model(
        self, frame: np.ndarray, vehicles: List[Detection], model: YOLO, conf: float
    ) -> Tuple[List[Detection], bool]:
        """Run any YOLO ambulance model and merge results, deduplicating by IoU."""
        try:
            result = model.predict(source=frame, conf=conf, verbose=False)[0]
        except Exception as e:
            self._logger.warning("Ambulance model prediction failed: %s", e)
            return vehicles, False

        ambulance_boxes = [
            {"class": "ambulance", "confidence": float(box.conf[0].item()), "bbox": [float(v) for v in box.xyxy[0].cpu().numpy()]}
            for box in (result.boxes or [])
            if "ambulance" in str(result.names[int(box.cls[0].item())]).lower()
        ]
        if not ambulance_boxes:
            return vehicles, False

        # Drop existing detections that overlap with confirmed ambulances
        filtered = [v for v in vehicles if not any(self._iou(v["bbox"], a["bbox"]) > 0.5 for a in ambulance_boxes)]
        return filtered + ambulance_boxes, True

    def _run_yolo_world(self, frame: np.ndarray, vehicles: List[Detection]) -> Tuple[List[Detection], bool]:
        try:
            from ultralytics import YOLOWorld
            if self._world_model is None:
                self._world_model = YOLOWorld(self._world_weights)
                self._world_model.set_classes(["ambulance"])
            result = self._world_model.predict(source=frame, conf=self._ambulance_world_conf, verbose=False)[0]
        except Exception:
            self._logger.debug("YOLOWorld unavailable or failed")
            return vehicles, False

        ambulance_boxes = [
            {"class": "ambulance", "confidence": float(box.conf[0].item()), "bbox": [float(v) for v in box.xyxy[0].cpu().numpy()]}
            for box in (result.boxes or [])
            if "ambulance" in str(result.names[int(box.cls[0].item())]).lower()
        ]
        if not ambulance_boxes:
            return vehicles, False

        filtered = [v for v in vehicles if not any(self._iou(v["bbox"], a["bbox"]) > 0.5 for a in ambulance_boxes)]
        return filtered + ambulance_boxes, True

    # Geometry

    @staticmethod
    def _iou(b1: List[float], b2: List[float]) -> float:
        xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
        if inter == 0:
            return 0.0
        union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
        return inter / union if union else 0.0

    # GPS emergency check

    def _check_gps_emergency(self) -> bool:
        try:
            import requests

            from config import GPS_REQUEST_TIMEOUT_SECONDS, GPS_SERVER_URL
            r = requests.get(f"{GPS_SERVER_URL}/check-ambulance", timeout=GPS_REQUEST_TIMEOUT_SECONDS)
            r.raise_for_status()
            return bool(r.json().get("emergency", False))
        except Exception as e:
            self._logger.debug("GPS emergency check failed: %s", e)
            return False

    # Cross-dataset fusion (optional)

    def _attach_fusion(self, out: Dict[str, Any], video_source_hint: Optional[str]) -> Dict[str, Any]:
        if not self._enrich_cross_dataset:
            return out
        try:
            from config import (FINDVEHICLE_SCHEMA_NAME,
                                VISION_TRAFFIC_DATASET_NAME)
            from data.coco_findvehicle_bridge import \
                attach_cross_dataset_fusion
            attach_cross_dataset_fusion(out, vision_dataset=VISION_TRAFFIC_DATASET_NAME, text_dataset=FINDVEHICLE_SCHEMA_NAME, video_source_hint=video_source_hint)
        except Exception as e:
            self._logger.debug("Cross-dataset fusion unavailable: %s", e)
        return out
