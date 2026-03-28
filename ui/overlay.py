"""Frame overlay rendering utilities.

This module should draw bounding boxes, lane counts, and the current signal
status onto the video frame before display.
"""

from __future__ import annotations

from typing import Any, Dict, List

import cv2


class TrafficOverlay:
    @staticmethod
    def _draw_panel(
        frame: Any,
        *,
        x: int,
        y: int,
        w: int,
        h: int,
        color: tuple[int, int, int] = (20, 20, 20),
        alpha: float = 0.68,
    ) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    @staticmethod
    def _draw_text(
        frame: Any,
        text: str,
        org: tuple[int, int],
        *,
        scale: float,
        color: tuple[int, int, int],
        thickness: int = 2,
        shadow_color: tuple[int, int, int] = (15, 15, 15),
    ) -> None:
        cv2.putText(
            frame,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            shadow_color,
            thickness + 3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

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
            self._draw_emergency_banner(frame, active=True)
        elif mode == "debug":
            self._draw_emergency_banner(frame, active=False)
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
                self._draw_text(frame, label, (x1, max(18, y1 - 8)), scale=0.5, color=color, thickness=1)

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
            self._draw_text(
                frame,
                f"Approach {lane.title()}: {approach_count}  Score: {lane_score:.1f}",
                (20, 30),
                scale=0.7,
                color=(255, 240, 120),
                thickness=2,
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
            self._draw_text(frame, text, pos, scale=0.7, color=(255, 240, 120), thickness=2)

    def _draw_signal_panel(self, frame: Any, signal_states: Dict[str, str], *, per_camera_mode: bool, camera_lane: str | None) -> None:
        h, w = frame.shape[:2]
        x0 = w - 270
        y0 = 26
        panel_w = 240
        panel_h = 150 if not per_camera_mode else 90

        self._draw_panel(frame, x=x0, y=y0, w=panel_w, h=panel_h, color=(26, 32, 38), alpha=0.7)
        self._draw_text(frame, "Signal State", (x0 + 12, y0 + 28), scale=0.62, color=(235, 235, 235), thickness=2)

        if per_camera_mode:
            lane = str(camera_lane or "north").lower()
            state = str(signal_states.get(lane, "RED")).upper()
            color = (80, 255, 120) if state == "GREEN" else (80, 110, 255)
            self._draw_text(frame, f"{lane}: {state}", (x0 + 12, y0 + 62), scale=0.72, color=color, thickness=2)
            return

        lanes = ["north", "south", "east", "west"]
        for i, lane in enumerate(lanes):
            state = str(signal_states.get(lane, "RED")).upper()
            color = (80, 255, 120) if state == "GREEN" else (80, 110, 255)
            y = y0 + 60 + (i * 22)
            self._draw_text(frame, f"{lane}: {state}", (x0 + 12, y), scale=0.54, color=color, thickness=2)

    def _draw_emergency_banner(self, frame: Any, *, active: bool = True) -> None:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        color = (0, 0, 255) if active else (45, 90, 45)
        cv2.rectangle(overlay, (0, 0), (w, 42), color, -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        label = "EMERGENCY MODE ON" if active else "EMERGENCY MODE OFF"
        self._draw_text(frame, label, (w // 2 - 155, 28), scale=0.85, color=(255, 255, 255), thickness=2)

    def _draw_kpi_panel(self, frame: Any, kpi_snapshot: Dict[str, Any]) -> None:
        h, _ = frame.shape[:2]
        x0 = 20
        y0 = max(95, h - 185)
        panel_w = 290
        panel_h = 150

        self._draw_panel(frame, x=x0, y=y0, w=panel_w, h=panel_h, color=(20, 20, 20), alpha=0.6)

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
            self._draw_text(frame, text, (x0 + 12, y), scale=scale, color=color, thickness=2 if i == 0 else 1)

    def draw_status_card(
        self,
        frame: Any,
        lines: List[tuple[str, tuple[int, int, int], float]],
        *,
        x: int = 12,
        y: int = 12,
        width: int = 560,
    ) -> None:
        if not lines:
            return

        x0 = x
        y0 = y
        panel_w = width
        line_gap = 30
        panel_h = 20 + (line_gap * len(lines))
        self._draw_panel(frame, x=x0, y=y0, w=panel_w, h=panel_h, color=(24, 28, 34), alpha=0.58)

        for idx, (text, color, scale) in enumerate(lines):
            y = y0 + 28 + (idx * line_gap)
            self._draw_text(frame, text, (x0 + 12, y), scale=scale, color=color, thickness=2)

    def draw_signal_roi(self, frame: Any, roi: tuple[int, int, int, int], label: str = "Signal ROI") -> None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
        self._draw_text(frame, label, (x1, max(18, y1 - 6)), scale=0.55, color=(255, 200, 0), thickness=2)

    def draw_counting_roi(self, frame: Any, roi: tuple[int, int, int, int], label: str, color: tuple[int, int, int]) -> None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        self._draw_text(frame, label, (x1, max(18, y1 - 6)), scale=0.55, color=color, thickness=2)


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
