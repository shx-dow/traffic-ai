from __future__ import annotations

from typing import Optional, Tuple


def parse_rect_roi(value: str) -> Optional[Tuple[int, int, int, int]]:
    raw = str(value or "").strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be in x1,y1,x2,y2 format")
    x1, y1, x2, y2 = [int(p) for p in parts]
    return x1, y1, x2, y2
