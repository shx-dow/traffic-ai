"""Traffic signal decision logic.

This module should determine which lane receives green time based on the
current vehicle counts and configured thresholds.
"""

from __future__ import annotations

from typing import Dict


def decide_signal(counts: Dict[str, int]) -> str:
    """Return one of: LEFT_GREEN, RIGHT_GREEN, BALANCED."""
    raise NotImplementedError("Implement the signal decision policy here.")
