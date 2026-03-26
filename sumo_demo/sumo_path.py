from __future__ import annotations

import os
from pathlib import Path


def ensure_sumo_home() -> str | None:
    existing = os.environ.get("SUMO_HOME")
    if existing and Path(existing).exists():
        return existing

    candidates = [
        r"C:\Program Files (x86)\Eclipse\Sumo",
        r"C:\Program Files\Eclipse\Sumo",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            os.environ["SUMO_HOME"] = candidate
            sumo_bin = Path(candidate) / "tools"
            if str(sumo_bin) not in os.environ.get("PYTHONPATH", ""):
                os.environ["PYTHONPATH"] = f"{sumo_bin};{os.environ.get('PYTHONPATH', '')}".strip(";")
            return candidate
    return None
