"""Minimal stub for cross-dataset fusion used by `vision.detector`.

This lightweight implementation only attaches a simple `fusion` entry to
the `out` dict so downstream code can exercise the fusion path during
integration tests. It intentionally avoids any heavyweight I/O or
third-party dependencies.
"""
from typing import Any, Dict
from datetime import datetime


def attach_cross_dataset_fusion(
    out: Dict[str, Any],
    *,
    vision_dataset: str,
    text_dataset: str,
    video_source_hint: str | None = None,
) -> None:
    """Attach a minimal fusion summary into the `out` dict.

    This function mutates `out` in-place and returns None. It is a
    noop beyond adding a diagnostic entry; real implementations will
    perform cross-dataset lookups and merge richer metadata.
    """
    out.setdefault("fusion", {})
    out["fusion"]["attached_at"] = datetime.utcnow().isoformat() + "Z"
    out["fusion"]["vision_dataset"] = vision_dataset
    out["fusion"]["text_dataset"] = text_dataset
    out["fusion"]["video_source_hint"] = video_source_hint
