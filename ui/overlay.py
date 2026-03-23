"""Frame overlay rendering utilities.

This module should draw bounding boxes, lane counts, and the current signal
status onto the video frame before display.
"""

from __future__ import annotations

from typing import Any, Dict, List


def draw_overlay(frame: Any, detections: List[Dict[str, Any]], counts: Dict[str, int], signal_state: str) -> Any:
    """Return the frame updated with visible demo annotations.

    For boxes, prefer `VehicleDetector.detect(frame)["raw_result"]` (Ultralytics
    Results); `detections` may be the same `vehicles` list passed to counting.
    """
    raise NotImplementedError("Implement drawing of boxes, text, and signal state here.")
