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
        lane_scores: Dict[str, float] | None = None,
        signal_states: Dict[str, str] | None = None,
        emergency_active: bool = False,
        kpi_snapshot: Dict[str, Any] | None = None,
        show_directional_counts: bool = True,
        camera_lane: str | None = None,
        ui_mode: str = "debug",
        per_camera_mode: bool = False,
    ) -> Any:
        mode = str(ui_mode).lower()
        vehicles = detection_result.get("vehicles", []) if isinstance(detection_result, dict) else []
        self._draw_bounding_boxes(frame, vehicles, show_labels=(mode != "demo"))
        if mode == "debug":
            self._draw_lane_counts(frame, lane_counts, lane_scores=lane_scores, show_directional_counts=show_directional_counts, camera_lane=camera_lane)
        self._draw_signal_panel(frame, signal_states or {}, per_camera_mode=per_camera_mode, camera_lane=camera_lane)
        if kpi_snapshot:
            self._draw_kpi_panel(frame, kpi_snapshot)
        if emergency_active:
            self._draw_emergency_banner(frame)
        return frame

    def _draw_bounding_boxes(self, frame: Any, vehicles: List[Dict[str, Any]], *, show_labels: bool) -> None:
        for v in vehicles:
            bbox = v.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(n) for n in bbox]
            label = f"{v.get('class', 'obj')} {float(v.get('confidence', 0.0)):.2f}"
            cls = str(v.get("class", "")).lower()
            color = (0, 0, 255) if cls in {"ambulance", "fire_truck"} else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if show_labels:
                cv2.putText(frame, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _draw_lane_counts(
        self,
        frame: Any,
        lane_counts: Dict[str, int],
        *,
        lane_scores: Dict[str, float] | None,
        show_directional_counts: bool,
        camera_lane: str | None,
    ) -> None:
        if not show_directional_counts:
            lane = str(camera_lane or "north").lower()
            approach_count = int(lane_counts.get(lane, 0))
            lane_score = float((lane_scores or {}).get(lane, 0.0))
            cv2.putText(
                frame,
                f"Approach {lane.title()}: {approach_count}  Score: {lane_score:.1f}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )
            return

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

    def _draw_signal_panel(self, frame: Any, signal_states: Dict[str, str], *, per_camera_mode: bool, camera_lane: str | None) -> None:
        h, w = frame.shape[:2]
        x0 = w - 250
        y0 = 50
        panel_w = 220
        panel_h = 145 if not per_camera_mode else 82

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        if per_camera_mode:
            lane = str(camera_lane or "north").lower()
            state = str(signal_states.get(lane, "RED")).upper()
            color = (0, 255, 0) if state == "GREEN" else (0, 0, 255)
            cv2.putText(frame, "Signal State", (x0 + 10, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2)
            cv2.putText(frame, f"{lane}: {state}", (x0 + 10, y0 + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            return

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

    def _draw_kpi_panel(self, frame: Any, kpi_snapshot: Dict[str, Any]) -> None:
        h, _ = frame.shape[:2]
        x0 = 20
        y0 = max(95, h - 185)
        panel_w = 290
        panel_h = 150

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        wait_s = float(kpi_snapshot.get("avg_wait_seconds", 0.0))
        throughput = int(kpi_snapshot.get("throughput_score", 0))
        max_queue = int(kpi_snapshot.get("max_queue", 0))
        emer_s = float(kpi_snapshot.get("emergency_duration_seconds", 0.0))

        lines = [
            "Live KPIs",
            f"Avg wait: {wait_s:.1f}s",
            f"Throughput: {throughput}",
            f"Max queue: {max_queue}",
            f"Emergency time: {emer_s:.1f}s",
        ]
        for i, text in enumerate(lines):
            y = y0 + 24 + (i * 24)
            color = (255, 255, 255) if i == 0 else (220, 255, 220)
            scale = 0.62 if i == 0 else 0.58
            cv2.putText(frame, text, (x0 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

    def draw_signal_roi(self, frame: Any, roi: tuple[int, int, int, int], label: str = "Signal ROI") -> None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
        cv2.putText(frame, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)

    def draw_counting_roi(self, frame: Any, roi: tuple[int, int, int, int], label: str, color: tuple[int, int, int]) -> None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


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
