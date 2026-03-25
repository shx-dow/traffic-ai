from __future__ import annotations

from typing import Dict, Iterable


def build_signal_summary(signal_states: Dict[str, str]) -> str:
    greens = [lane for lane, state in signal_states.items() if str(state).upper() == "GREEN"]
    if len(greens) == 4:
        return "ALL_GREEN"
    if not greens:
        return "ALL_RED"
    return ",".join(greens)


def is_balanced(scores: Iterable[float], gap: float) -> bool:
    values = list(scores)
    if len(values) < 2:
        return True
    top = max(values)
    second = sorted(values, reverse=True)[1]
    return (top - second) < gap
