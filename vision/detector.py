from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

VEHICLE_CLASSES = frozenset({"car", "truck", "bus", "motorcycle", "bicycle"})
EMERGENCY_CLASS_NAMES = frozenset({"ambulance", "fire_truck"})
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
        self._gps_cache: Dict[str, Any] = {
            "emergency": False,
            "vehicle_id": None,
            "distance_km": None,
            "eta_seconds": None,
            "speed_kmh": None,
        }
        self._gps_last_poll_ts = 0.0

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
        Output : {vehicles, count, vision_emergency, gps_emergency, emergency, raw_result, [fusion]}
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy ndarray (OpenCV BGR image).")

        raw_result: Results = self._model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        vehicles, vision_emergency = self._vehicles_from_coco(raw_result)
        vehicles, vision_emergency = self._merge_ambulance_detections(frame, vehicles, vision_emergency)

        gps_status = self._check_gps_emergency()
        gps_emergency = bool(gps_status.get("emergency", False))
        out = {
            "vehicles": vehicles,
            "count": len(vehicles),
            "vision_emergency": vision_emergency,
            "gps_emergency": gps_emergency,
            "emergency": bool(vision_emergency or gps_emergency),
            "gps_priority": gps_status,
            "raw_result": raw_result,
        }
        return self._attach_fusion(out, video_source_hint)

    # Detection helpers

    def _vehicles_from_coco(self, raw_result: Results) -> Tuple[List[Detection], bool]:
        vehicles: List[Detection] = []
        if not raw_result.boxes:
            return vehicles, False
        for box in raw_result.boxes:
            raw_name = str(raw_result.names[int(box.cls[0].item())]).lower()
            normalized_emergency = self._normalize_emergency_label(raw_name)
            cls_name = normalized_emergency if normalized_emergency else raw_name
            if cls_name not in TRACKED_CLASSES and cls_name not in VEHICLE_CLASSES:
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

        emergency_boxes: List[Detection] = []
        for box in (result.boxes or []):
            cls_label = str(result.names[int(box.cls[0].item())]).lower()
            normalized = self._normalize_emergency_label(cls_label)
            if normalized is None:
                continue
            emergency_boxes.append(
                {
                    "class": normalized,
                    "confidence": float(box.conf[0].item()),
                    "bbox": [float(v) for v in box.xyxy[0].cpu().numpy()],
                }
            )

        if not emergency_boxes:
            return vehicles, False

        filtered = [v for v in vehicles if not any(self._iou(v["bbox"], e["bbox"]) > 0.5 for e in emergency_boxes)]
        return filtered + emergency_boxes, True

    def _run_yolo_world(self, frame: np.ndarray, vehicles: List[Detection]) -> Tuple[List[Detection], bool]:
        try:
            from ultralytics import YOLOWorld
            if self._world_model is None:
                self._world_model = YOLOWorld(self._world_weights)
                self._world_model.set_classes(["ambulance", "fire truck", "fire engine"])
            result = self._world_model.predict(source=frame, conf=self._ambulance_world_conf, verbose=False)[0]
        except Exception:
            self._logger.debug("YOLOWorld unavailable or failed")
            return vehicles, False

        emergency_boxes: List[Detection] = []
        for box in (result.boxes or []):
            cls_label = str(result.names[int(box.cls[0].item())]).lower()
            normalized = self._normalize_emergency_label(cls_label)
            if normalized is None:
                continue
            emergency_boxes.append(
                {
                    "class": normalized,
                    "confidence": float(box.conf[0].item()),
                    "bbox": [float(v) for v in box.xyxy[0].cpu().numpy()],
                }
            )

        if not emergency_boxes:
            return vehicles, False

        filtered = [v for v in vehicles if not any(self._iou(v["bbox"], e["bbox"]) > 0.5 for e in emergency_boxes)]
        return filtered + emergency_boxes, True

    @staticmethod
    def _normalize_emergency_label(label: str) -> str | None:
        value = str(label).lower().replace("_", " ").replace("-", " ").strip()
        compact = value.replace(" ", "")

        if "ambulance" in value:
            return "ambulance"
        if "fire" in value and ("truck" in value or "engine" in value or compact in {"firetruck", "fireengine"}):
            return "fire_truck"
        if compact in {"firetruck", "fireengine"}:
            return "fire_truck"
        return None

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

    def _check_gps_emergency(self) -> Dict[str, Any]:
        try:
            import requests

            from config import (GPS_POLL_INTERVAL_SECONDS,
                                GPS_REQUEST_TIMEOUT_SECONDS, GPS_SERVER_URL)

            now = time.monotonic()
            if now - self._gps_last_poll_ts < GPS_POLL_INTERVAL_SECONDS:
                return self._gps_cache

            r = requests.get(f"{GPS_SERVER_URL}/check-ambulance", timeout=GPS_REQUEST_TIMEOUT_SECONDS)
            r.raise_for_status()
            payload = r.json()
            self._gps_cache = {
                "emergency": bool(payload.get("emergency", False)),
                "vehicle_id": payload.get("vehicle_id"),
                "distance_km": payload.get("distance_km"),
                "eta_seconds": payload.get("eta_seconds"),
                "speed_kmh": payload.get("speed_kmh"),
            }
            self._gps_last_poll_ts = now
            return self._gps_cache
        except Exception as e:
            self._logger.debug("GPS emergency check failed: %s", e)
            return self._gps_cache

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
