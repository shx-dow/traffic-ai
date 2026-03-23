
from __future__ import annotations


def is_emergency_active() -> bool:
    """Return whether the emergency override is active."""
    raise NotImplementedError("Implement emergency simulation or trigger handling here.")


def apply_emergency_override(signal_state: str, emergency_active: bool) -> str:
    """Return ALL_GREEN when emergency mode is active, otherwise pass through."""
    if emergency_active:
        return "ALL_GREEN"
    return signal_state
