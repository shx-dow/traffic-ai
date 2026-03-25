"""Frame overlay rendering utilities.

This module should draw bounding boxes, lane counts, and the current signal
status onto the video frame before display.
"""

from __future__ import annotations

from typing import Any, Dict, List

import cv2


class TrafficOverlay:
    def draw(
        self,
        frame: Any,
        detection_result: Dict[str, Any],
        lane_counts: Dict[str, int],
        signal_states: Dict[str, str],
        emergency_active: bool,
    ) -> Any:
        vehicles = detection_result.get("vehicles", []) if isinstance(detection_result, dict) else []
        self._draw_bounding_boxes(frame, vehicles)
        self._draw_lane_counts(frame, lane_counts)
        self._draw_signal_panel(frame, signal_states)
        if emergency_active:
            self._draw_emergency_banner(frame)
        return frame

    def _draw_bounding_boxes(self, frame: Any, vehicles: List[Dict[str, Any]]) -> None:
        for v in vehicles:
            bbox = v.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(n) for n in bbox]
            label = f"{v.get('class', 'obj')} {float(v.get('confidence', 0.0)):.2f}"
            color = (0, 0, 255) if str(v.get("class", "")).lower() == "ambulance" else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _draw_lane_counts(self, frame: Any, lane_counts: Dict[str, int]) -> None:
        h, w = frame.shape[:2]
        labels = {
            "north": (20, 30),
            "south": (20, h - 20),
            "east": (w - 180, 30),
            "west": (w - 180, h - 20),
        }
        for lane, pos in labels.items():
            count = int(lane_counts.get(lane, 0))
            text = f"{lane[0].upper()}: {count}"
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    def _draw_signal_panel(self, frame: Any, signal_states: Dict[str, str]) -> None:
        h, w = frame.shape[:2]
        x0 = w - 250
        y0 = 50
        panel_w = 220
        panel_h = 145

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        lanes = ["north", "south", "east", "west"]
        for i, lane in enumerate(lanes):
            state = str(signal_states.get(lane, "RED")).upper()
            color = (0, 255, 0) if state == "GREEN" else (0, 0, 255)
            y = y0 + 28 + (i * 28)
            cv2.putText(frame, f"{lane}: {state}", (x0 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    def _draw_emergency_banner(self, frame: Any) -> None:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        cv2.putText(frame, "EMERGENCY MODE", (w // 2 - 130, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)


def draw_overlay(frame: Any, detections: List[Dict[str, Any]], counts: Dict[str, int], signal_state: str) -> Any:
    """Backward-compatible wrapper around TrafficOverlay.

    For boxes, prefer `VehicleDetector.detect(frame)["raw_result"]` (Ultralytics
    Results); `detections` may be the same `vehicles` list passed to counting.
    """
    overlay = TrafficOverlay()
    return overlay.draw(
        frame,
        detection_result={"vehicles": detections},
        lane_counts=counts,
        signal_states={"north": signal_state, "south": "RED", "east": "RED", "west": "RED"},
        emergency_active=(signal_state == "EMERGENCY"),
    )
