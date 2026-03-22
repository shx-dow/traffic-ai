"""Shared helper utilities.

Use this module for small reusable helpers such as geometry, formatting, and
frame normalization routines.
"""

from __future__ import annotations

from typing import Tuple


def clamp(value: int, minimum: int, maximum: int) -> int:
    """Clamp an integer to the given inclusive range."""
    return max(minimum, min(value, maximum))


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """Compute the center point of a bounding box."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) // 2, (y1 + y2) // 2
