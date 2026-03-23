"""Vehicle counting and lane-splitting logic.

This module should:
- Count vehicles from detection results
- Split vehicles into left and right lanes using a simple midpoint rule
- Return lightweight summary data for downstream signal logic and overlay
"""

from __future__ import annotations

from typing import Any, Dict, List


def count_vehicles(detections: List[Dict[str, Any]], frame_width: int) -> Dict[str, int]:
    """Count vehicles and estimate lane distribution.

    `detections` should be `VehicleDetector.detect(frame)["vehicles"]` (list of
    dicts with keys class, confidence, bbox).
    """
    raise NotImplementedError("Implement midpoint-based lane counting here.")
