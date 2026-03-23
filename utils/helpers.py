"""Shared helper utilities.

Use this module for small reusable helpers such as geometry, formatting, and
frame normalization routines.
"""

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
    # Inherit stdio so user can see progress; return code drives success/failure.
    return subprocess.run(cmd, cwd=cwd).returncode


def run_repo_pipeline(*, synthetic_frames: int, headless_frames: int, disable_gps: bool):
    """
    Run repo-wide validations/tests/scripts that are safe for local/CI usage.

    Returns:
        A running gps_server process if we started it, otherwise None.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ../..
    python = sys.executable

    print("\n=== Repo pipeline: compile + unit tests + headless smoke ===")
    subprocess.run([python, "-m", "compileall", "."], cwd=repo_root, check=False)
    _run_cmd([python, "tests/test_logic.py"], cwd=repo_root)

    # Detector synthetic smoke test (no camera/video required).
    _run_cmd(
        [python, "tests/test_detector.py", "--synthetic", "--max-frames", str(synthetic_frames)],
        cwd=repo_root,
    )

    gps_proc = None
    gps_host = "localhost"
    gps_port = 8000

    if not disable_gps:
        # Start GPS server if not already running.
        if not _port_open(gps_host, gps_port):
            print("\n=== Starting gps_server.py (needed for GPS tests) ===")
            gps_proc = subprocess.Popen([python, "gps_server.py"], cwd=repo_root)

            # Wait a moment for health endpoint.
            for _ in range(20):
                if _port_open(gps_host, gps_port, timeout_s=0.15):
                    break
                time.sleep(0.25)

        # GPS integration tests (requires server).
        _run_cmd([python, "tests/test_gps_integration.py"], cwd=repo_root)
    else:
        print("\n=== GPS disabled: skipping gps_server + tests/test_gps_integration.py ===")

    # Headless pipeline runner (validates detector->counter->signal loop).
    rc = _run_cmd(
        [python, "scripts/run_headless_main.py", "--frames", str(headless_frames)],
        cwd=repo_root,
    )
    if rc != 0:
        print("run_headless_main.py failed with --frames; retrying without --frames...")
        _run_cmd([python, "scripts/run_headless_main.py"], cwd=repo_root)

    # Dataset extract is optional (only if zip exists).
    zip_path = os.path.join(repo_root, "assets", "real_time_traffic", "real_traffic.zip")
    if os.path.isfile(zip_path):
        _run_cmd([python, "scripts/extract_real_traffic.py"], cwd=repo_root)
    else:
        print("\n=== real_traffic.zip not found: skipping scripts/extract_real_traffic.py ===")

    # FindVehicle analytics depends on a `data.findvehicle` module which may not exist
    # in this repo snapshot.
    findvehicle_py = os.path.join(repo_root, "data", "findvehicle.py")
    findvehicle_pkg_dir = os.path.join(repo_root, "data", "findvehicle")
    if os.path.exists(findvehicle_py) or os.path.isdir(findvehicle_pkg_dir):
        _run_cmd([python, "scripts/analyze_findvehicle.py"], cwd=repo_root)
    else:
        print("\n=== data.findvehicle module not found: skipping scripts/analyze_findvehicle.py ===")

    # Ambulance validation: use a bundled sample image if present.
    sample_img = os.path.join(
        repo_root,
        "assets",
        "ambulance_dataset",
        "carsdataset",
        "ambulance",
        "0AL642VPYDA4.jpg",
    )
    if os.path.isfile(sample_img):
        _run_cmd(
            [python, "scripts/validate_ambulance.py", "--image", sample_img, "--ambulance-conf", "0.15"],
            cwd=repo_root,
        )
    else:
        print("\n=== Sample ambulance image not found: skipping scripts/validate_ambulance.py ===")

    print("\n=== Pipeline finished. Starting real-time main loop ===")
    return gps_proc
