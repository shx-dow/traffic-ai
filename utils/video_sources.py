"""Locate traffic video files on disk (e.g. under assets/real_time_traffic)."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


def ensure_real_traffic_extracted(root: Path, zip_name: str = "real_traffic.zip") -> bool:
    """
    If ``root / zip_name`` exists and no playable video exists under ``root``, extract the zip into ``root``.

    Returns True if extraction was run. Large archives may take several minutes.
    """
    root = Path(root)
    zpath = root / zip_name
    if not zpath.is_file() or zpath.stat().st_size < 1000:
        return False
    if first_video_under(root) is not None:
        return False
    if not zipfile.is_zipfile(zpath):
        print(f"Warning: not a valid zip file: {zpath}", file=sys.stderr)
        return False
    print(
        f"Extracting {zpath.name} into {root} (first-time setup; may take a while) …",
        file=sys.stderr,
        flush=True,
    )
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(root)
    print("Extraction finished.", file=sys.stderr, flush=True)
    return True


def first_video_under(root: Path, *, max_depth: int = 8) -> Path | None:
    """
    Return the first video file found under `root` (sorted paths, depth-limited).

    Large archives often unpack one or more subfolders; shallow search avoids
    scanning the whole tree on huge datasets.
    """
    root = Path(root)
    if not root.is_dir():
        return None

    def depth(p: Path) -> int:
        try:
            return len(p.relative_to(root).parts)
        except ValueError:
            return 999

    candidates: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if depth(p) > max_depth:
            continue
        candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda x: str(x).lower())
    return candidates[0]
