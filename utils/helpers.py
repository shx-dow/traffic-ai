"""Shared helper utilities — geometry, formatting, frame normalization, pipeline runner."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from typing import Tuple


def clamp(value: int, minimum: int, maximum: int) -> int:
    """Clamp an integer to the given inclusive range."""
    return max(minimum, min(value, maximum))


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """Compute the center point of a bounding box."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) // 2, (y1 + y2) // 2


def _port_open(host: str, port: int, timeout_s: float = 0.25) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_s)
        try:
            sock.connect((host, port))
            return True
        except OSError:
            return False


def _run_cmd(cmd: list[str], *, cwd: str) -> int:
    return subprocess.run(cmd, cwd=cwd).returncode


def run_repo_pipeline(*, disable_gps: bool):
    """
    Run repo-wide validations/tests/scripts that are safe for local/CI usage.

    Returns:
        A running gps_server process if we started it, otherwise None.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    python = sys.executable

    print("\n=== Repo pipeline: compile + core tests ===")
    subprocess.run([python, "-m", "compileall", "."], cwd=repo_root, check=False)
    _run_cmd([python, "tests/test_logic.py"], cwd=repo_root)
    _run_cmd([python, "tests/test_baseline_signal.py"], cwd=repo_root)
    _run_cmd([python, "tests/test_runtime.py"], cwd=repo_root)
    _run_cmd([python, "tests/test_simulation.py"], cwd=repo_root)
    _run_cmd([python, "tests/test_detector_logic.py"], cwd=repo_root)

    gps_proc = None
    gps_host = "localhost"
    gps_port = 8000

    if not disable_gps:
        if not _port_open(gps_host, gps_port):
            print("\n=== Starting gps_server.py (needed for GPS tests) ===")
            gps_proc = subprocess.Popen([python, "gps_server.py"], cwd=repo_root)

            for _ in range(20):
                if _port_open(gps_host, gps_port, timeout_s=0.15):
                    break
                time.sleep(0.25)

        _run_cmd([python, "tests/test_gps_integration.py"], cwd=repo_root)
    else:
        print("\n=== GPS disabled: skipping gps_server + tests/test_gps_integration.py ===")

    print("\n=== Pipeline finished. Starting real-time main loop ===")
    return gps_proc
