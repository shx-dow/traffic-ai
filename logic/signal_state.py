from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class SignalStateReading:
    state: str
    source: str
    confidence: float


class SignalStateSensor:
    def __init__(
        self,
        *,
        source: str = "none",
        api_url: str = "",
        roi: Optional[Tuple[int, int, int, int]] = None,
        timeout_seconds: float = 0.4,
        min_color_pixels: int = 24,
        min_blob_area: float = 18.0,
        min_confidence: float = 0.6,
    ) -> None:
        self.source = str(source).lower()
        self.api_url = str(api_url or "").strip()
        self.roi = roi
        self.timeout_seconds = float(timeout_seconds)
        self.min_color_pixels = int(min_color_pixels)
        self.min_blob_area = float(min_blob_area)
        self.min_confidence = float(min_confidence)

    def read(self, frame: np.ndarray | None = None) -> SignalStateReading | None:
        if self.source == "api":
            return self._read_from_api()
        if self.source == "video":
            if frame is None:
                return None
            return self._read_from_frame(frame)
        return None

    def _read_from_api(self) -> SignalStateReading | None:
        if not self.api_url:
            return None
        try:
            import requests

            r = requests.get(self.api_url, timeout=self.timeout_seconds)
            r.raise_for_status()
            payload: Dict[str, Any] = r.json()
            state = str(payload.get("state", "")).upper()
            if state not in {"RED", "YELLOW", "GREEN"}:
                return None
            conf = float(payload.get("confidence", 1.0))
            return SignalStateReading(state=state, source="api", confidence=max(0.0, min(1.0, conf)))
        except Exception:
            return None

    def _read_from_frame(self, frame: np.ndarray) -> SignalStateReading | None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.get_effective_roi(frame)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        bright_mask = cv2.inRange(v_channel, 170, 255)

        red1 = cv2.inRange(hsv, (0, 80, 80), (12, 255, 255))
        red2 = cv2.inRange(hsv, (168, 80, 80), (180, 255, 255))
        yellow = cv2.inRange(hsv, (16, 90, 90), (40, 255, 255))
        green = cv2.inRange(hsv, (40, 80, 80), (95, 255, 255))

        red_mask = cv2.bitwise_and(cv2.bitwise_or(red1, red2), bright_mask)
        yellow_mask = cv2.bitwise_and(yellow, bright_mask)
        green_mask = cv2.bitwise_and(green, bright_mask)

        red_score = self._mask_score(red_mask)
        yellow_score = self._mask_score(yellow_mask)
        green_score = self._mask_score(green_mask)

        total = red_score + yellow_score + green_score
        if total < self.min_color_pixels:
            return None

        best = max(("RED", red_score), ("YELLOW", yellow_score), ("GREEN", green_score), key=lambda x: x[1])
        confidence = best[1] / total if total > 0 else 0.0
        if confidence < self.min_confidence:
            return None
        return SignalStateReading(state=best[0], source="video", confidence=confidence)

    def get_effective_roi(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.roi or (0, 0, max(1, w // 6), max(1, h // 4))
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(x1 + 1, min(w, int(x2)))
        y2 = max(y1 + 1, min(h, int(y2)))
        return x1, y1, x2, y2

    def _mask_score(self, mask: np.ndarray) -> float:
        colored = float(np.count_nonzero(mask))
        if colored < self.min_color_pixels:
            return 0.0

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0

        best_area = max(float(cv2.contourArea(c)) for c in contours)
        if best_area < self.min_blob_area:
            return 0.0
        return colored


def parse_roi_arg(roi_value: str) -> Optional[Tuple[int, int, int, int]]:
    raw = str(roi_value or "").strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be in format x1,y1,x2,y2")
    x1, y1, x2, y2 = [int(p) for p in parts]
    return x1, y1, x2, y2
