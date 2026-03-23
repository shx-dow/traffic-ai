"""YOLOv8-based vehicle detection — single-frame inference for the traffic pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import logging
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

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        *,
        enrich_cross_dataset: bool | None = None,
        ambulance_mode: str | None = None,
        ambulance_confidence: float | None = None,
    ) -> None:
        self._model = YOLO(model_path)
        self._logger = logging.getLogger(__name__)
        if enrich_cross_dataset is None:
            from config import DETECTOR_ENRICH_CROSS_DATASET

            enrich_cross_dataset = DETECTOR_ENRICH_CROSS_DATASET
        self._enrich_cross_dataset = enrich_cross_dataset

        from config import (
            AMBULANCE_AUX_MODEL_PATH,
            AMBULANCE_CONFIDENCE,
            AMBULANCE_CUSTOM_MODEL_PATH,
            AMBULANCE_DETECTION_MODE,
            AMBULANCE_WORLD_CONFIDENCE,
            AMBULANCE_WORLD_MODEL,
        )

        self._ambulance_mode = (
            ambulance_mode if ambulance_mode is not None else AMBULANCE_DETECTION_MODE
        ).lower()
        self._ambulance_conf = float(
            AMBULANCE_CONFIDENCE if ambulance_confidence is None else ambulance_confidence
        )
        self._ambulance_world_conf = float(AMBULANCE_WORLD_CONFIDENCE)
        self._world_weights = AMBULANCE_WORLD_MODEL
        self._world_model: Any = None
        self._ambulance_custom: YOLO | None = None
        self._ambulance_aux: YOLO | None = None
        
        # Load custom ambulance model (highest priority)
        if self._ambulance_mode in ("custom", "aux_weights"):
            custom_path = Path(AMBULANCE_CUSTOM_MODEL_PATH)
            if custom_path.is_file():
                try:
                    self._ambulance_custom = YOLO(str(custom_path.resolve()))
                    self._logger.info("Loaded custom ambulance model: %s", custom_path)
                except Exception:
                    self._logger.exception("Failed to load custom ambulance model: %s", custom_path)
            else:
                self._logger.info("Custom ambulance model not found: %s", custom_path)
                
        # Legacy aux_weights support
        if self._ambulance_mode == "aux_weights":
            aux_path = Path(AMBULANCE_AUX_MODEL_PATH)
            if aux_path.is_file() and self._ambulance_custom is None:
                self._ambulance_aux = YOLO(str(aux_path.resolve()))

    def detect(
        self,
        frame: np.ndarray,
        *,
        video_source_hint: str | None = None,
    ) -> Dict[str, Any]:
        """
        Input  : numpy array (one video frame from OpenCV, BGR uint8)
        Output : {
            'vehicles': [{'class': str, 'confidence': float, 'bbox': [x1,y1,x2,y2]}, ...],
            'count': int,
            'emergency': bool,
            'raw_result': Results,
            'fusion': {...}   # optional when enrich_cross_dataset=True
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

        vehicles, emergency = self._vehicles_from_coco(raw_result)
        vehicles, emergency = self._merge_ambulance_detections(frame, vehicles, emergency)

        # Check GPS-based emergency
        gps_emergency = self._check_gps_emergency()
        final_emergency = emergency or gps_emergency

        out = {
            "vehicles": vehicles,
            "count": len(vehicles),
            "emergency": final_emergency,
            "gps_emergency": gps_emergency,
            "raw_result": raw_result,
        }
        return self._attach_fusion(out, video_source_hint)

    def _vehicles_from_coco(self, raw_result: Results) -> Tuple[List[Dict[str, Any]], bool]:
        vehicles: List[Dict[str, Any]] = []
        if raw_result.boxes is None or len(raw_result.boxes) == 0:
            return vehicles, False
        names = raw_result.names
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
        return vehicles, emergency

    def _merge_ambulance_detections(
        self,
        frame: np.ndarray,
        vehicles: List[Dict[str, Any]],
        emergency: bool,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        if self._ambulance_mode == "none":
            return vehicles, emergency
            
        # Priority 1: Custom trained ambulance model (if available)
        if self._ambulance_mode in ("custom", "aux_weights") and self._ambulance_custom is not None:
            vehicles, emergency = self._append_custom_ambulance(frame, vehicles, emergency)
            
        # Priority 2: Legacy aux_weights (if custom failed)
        if self._ambulance_mode == "aux_weights" and self._ambulance_aux is not None and not emergency:
            vehicles, emergency = self._append_aux_ambulance(frame, vehicles, emergency)
            
        # Priority 3: YOLOWorld fallback (only when explicitly requested or custom model not available)
        if self._ambulance_mode == "yolo_world" or (self._ambulance_mode == "custom" and self._ambulance_custom is None and not emergency):
            vehicles, emergency = self._append_yolo_world_ambulance(frame, vehicles, emergency)
            
        return vehicles, emergency

    def _append_aux_ambulance(
        self,
        frame: np.ndarray,
        vehicles: List[Dict[str, Any]],
        emergency: bool,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        if self._ambulance_aux is None:
            return vehicles, emergency
        ar = self._ambulance_aux.predict(
            source=frame,
            conf=self._ambulance_conf,
            verbose=False,
        )[0]
        if ar.boxes is None or len(ar.boxes) == 0:
            return vehicles, emergency
        names = ar.names
        for box in ar.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = str(names[cls_id]).lower()
            if "ambulance" not in cls_name:
                continue
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy()
            bbox = [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])]
            vehicles.append({"class": "ambulance", "confidence": conf, "bbox": bbox})
            emergency = True
        return vehicles, emergency

    def _append_custom_ambulance(
        self,
        frame: np.ndarray,
        vehicles: List[Dict[str, Any]],
        emergency: bool,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Append ambulance detections from custom trained model with duplicate avoidance."""
        if self._ambulance_custom is None:
            return vehicles, emergency
            
        try:
            ar = self._ambulance_custom.predict(
                source=frame,
                conf=self._ambulance_conf,
                verbose=False,
            )[0]
        except Exception as e:
            print(f"Custom ambulance model prediction failed: {e}")
            return vehicles, emergency
            
        if ar.boxes is None or len(ar.boxes) == 0:
            self._logger.debug("Custom ambulance model: no detections")
            return vehicles, emergency
            
        names = ar.names
        ambulance_boxes = []
        
        for box in ar.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = str(names[cls_id]).lower()
            if "ambulance" not in cls_name:
                continue
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy()
            bbox = [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])]
            ambulance_boxes.append({"class": "ambulance", "confidence": conf, "bbox": bbox})
            
        if not ambulance_boxes:
            self._logger.debug("Custom ambulance model: no ambulance detections")
            return vehicles, emergency
            
        # Remove overlapping vehicle detections (truck/car that might be ambulances)
        filtered_vehicles = []
        for vehicle in vehicles:
            should_keep = True
            for ambulance in ambulance_boxes:
                if self._bboxes_overlap(vehicle["bbox"], ambulance["bbox"], iou_threshold=0.5):
                    self._logger.info(
                        "Replacing %s with ambulance (IoU: %.2f)",
                        vehicle['class'],
                        self._calculate_iou(vehicle['bbox'], ambulance['bbox'])
                    )
                    should_keep = False
                    break
            if should_keep:
                filtered_vehicles.append(vehicle)
                
        # Add ambulance detections
        filtered_vehicles.extend(ambulance_boxes)
        
        self._logger.info(
            "Custom ambulance model: added %d ambulance detection(s)", len(ambulance_boxes)
        )
        return filtered_vehicles, True

    def _bboxes_overlap(self, bbox1: List[float], bbox2: List[float], iou_threshold: float = 0.5) -> bool:
        """Check if two bounding boxes overlap significantly."""
        return self._calculate_iou(bbox1, bbox2) > iou_threshold
        
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
            
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - inter_area
        
        if union_area == 0:
            return 0.0
            
        return inter_area / union_area

    def _append_yolo_world_ambulance(
        self,
        frame: np.ndarray,
        vehicles: List[Dict[str, Any]],
        emergency: bool,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        try:
            from ultralytics import YOLOWorld
        except ImportError:
            self._logger.debug("YOLOWorld import failed - CLIP not available")
            return vehicles, emergency
        try:
            if self._world_model is None:
                self._world_model = YOLOWorld(self._world_weights)
                self._world_model.set_classes(["ambulance"])
                print("Initialized YOLOWorld model for ambulance detection")
                
            # Use lower confidence for fallback detection
            conf_threshold = self._ambulance_world_conf
            wr = self._world_model.predict(
                source=frame,
                conf=conf_threshold,
                verbose=False,
            )[0]
        except Exception:
            self._logger.exception("YOLOWorld prediction failed")
            return vehicles, emergency
            
        if wr.boxes is None or len(wr.boxes) == 0:
            self._logger.debug(
                "YOLOWorld: no ambulance detections above %s confidence", conf_threshold
            )
            return vehicles, emergency
            
        names = wr.names
        ambulance_boxes = []
        
        for box in wr.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = str(names[cls_id]).lower()
            if "ambulance" not in cls_name:
                continue
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy()
            bbox = [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])]
            ambulance_boxes.append({"class": "ambulance", "confidence": conf, "bbox": bbox})
            
        if not ambulance_boxes:
            self._logger.debug("YOLOWorld: no ambulance detections")
            return vehicles, emergency
            
        # Remove overlapping vehicle detections (truck/car that might be ambulances)
        filtered_vehicles = []
        for vehicle in vehicles:
            should_keep = True
            for ambulance in ambulance_boxes:
                if self._bboxes_overlap(vehicle["bbox"], ambulance["bbox"], iou_threshold=0.5):
                    print(f"YOLOWorld: Replacing {vehicle['class']} with ambulance (IoU: {self._calculate_iou(vehicle['bbox'], ambulance['bbox']):.2f})")
                    should_keep = False
                    break
            if should_keep:
                filtered_vehicles.append(vehicle)
                
        # Add ambulance detections
        filtered_vehicles.extend(ambulance_boxes)
        
        self._logger.info(
            "YOLOWorld fallback: added %d ambulance detection(s)", len(ambulance_boxes)
        )
        return filtered_vehicles, True

    def _check_gps_emergency(self) -> bool:
        """
        Check for GPS-based ambulance emergency by calling the GPS server API.

        Returns:
            True if any ambulance is within emergency distance, False otherwise
        """
        try:
            import requests
            from config import GPS_REQUEST_TIMEOUT_SECONDS, GPS_SERVER_URL

            # Call the GPS server's check-ambulance endpoint
            response = requests.get(
                f"{GPS_SERVER_URL}/check-ambulance",
                timeout=GPS_REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()

            result = response.json()
            gps_emergency = result.get("emergency", False)

            return gps_emergency

        except (requests.exceptions.RequestException, ImportError) as e:
            # GPS server unavailable or network error - don't fail detection
            self._logger.debug("GPS emergency check failed: %s", e)
            return False
        except Exception as e:
            self._logger.exception("Unexpected error in GPS emergency check")
            return False

    def _attach_fusion(
        self,
        out: Dict[str, Any],
        video_source_hint: str | None,
    ) -> Dict[str, Any]:
        if not self._enrich_cross_dataset:
            return out
        from config import FINDVEHICLE_SCHEMA_NAME, VISION_TRAFFIC_DATASET_NAME

        try:
            from data.coco_findvehicle_bridge import attach_cross_dataset_fusion
        except Exception as e:
            # Cross-dataset fusion module not available in this environment.
            # Don't fail detection; return original output.
            self._logger.debug("Cross-dataset fusion unavailable: %s", e)
            return out

        try:
            attach_cross_dataset_fusion(
                out,
                vision_dataset=VISION_TRAFFIC_DATASET_NAME,
                text_dataset=FINDVEHICLE_SCHEMA_NAME,
                video_source_hint=video_source_hint,
            )
        except Exception:
            self._logger.exception("attach_cross_dataset_fusion failed")

        return out
