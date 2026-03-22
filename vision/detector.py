"""YOLO-based vehicle detection module.

Expected contract:
- Load a YOLOv8 model from Ultralytics
- Expose `detect(frame)` for a single video frame
- Return a list of structured detections like:
  {"class": int, "bbox": (x1, y1, x2, y2)}
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


Detection = Dict[str, Any]
BBox = Tuple[int, int, int, int]


def detect(frame: Any) -> List[Detection]:
    """Detect vehicles in a frame.

    This scaffold intentionally does not perform inference yet.
    """
    raise NotImplementedError("Implement YOLOv8 inference and return structured detections.")
