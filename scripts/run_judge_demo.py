from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path


def _port_open(host: str, port: int, timeout_s: float = 0.25) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_s)
        try:
            sock.connect((host, port))
            return True
        except OSError:
            return False


def _wait_for_port(host: str, port: int, timeout_s: float = 8.0) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        if _port_open(host, port):
            return True
        time.sleep(0.2)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-command judge demo runner")
    parser.add_argument("--video-source", default="assets/sample_video.mp4")
    parser.add_argument("--camera-lane", choices=("north", "south", "east", "west"), default="north")
    parser.add_argument("--approach-roi", default="")
    parser.add_argument("--queue-roi", default="")
    parser.add_argument("--max-frames", type=int, default=900)
    parser.add_argument("--ui-mode", choices=("demo", "debug"), default="demo")
    parser.add_argument("--signal-state-source", choices=("none", "video", "api"), default="none")
    parser.add_argument("--signal-state-roi", default="")
    parser.add_argument("--with-ambulance-sim", action="store_true")
    parser.add_argument("--with-orchestrator", action="store_true")
    parser.add_argument("--with-benchmark", action="store_true")
    parser.add_argument("--with-report", action="store_true")
    parser.add_argument("--metrics-log-path", default="artifacts/live_metrics_demo.jsonl")
    parser.add_argument("--output-path", default="artifacts/demo_output.mp4")
    parser.add_argument("--report-path", default="artifacts/demo_report.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    python = sys.executable

    gps_proc = None
    sim_proc = None
    try:
        print("Starting gps_server.py...")
        gps_proc = subprocess.Popen([python, "gps_server.py"], cwd=root)
        if not _wait_for_port("localhost", 8000):
            print("gps_server did not start on port 8000")
            return 1

        if args.with_ambulance_sim:
            print("Starting simulate_ambulance.py...")
            sim_proc = subprocess.Popen([python, "simulate_ambulance.py"], cwd=root)

        cmd = [
            python,
            "main.py",
            "--mode",
            "adaptive",
            "--video-source",
            args.video_source,
            "--camera-lane",
            args.camera_lane,
            "--ui-mode",
            args.ui_mode,
            "--emergency-source",
            "fusion",
            "--max-frames",
            str(args.max_frames),
            "--save-output",
            "--output-path",
            args.output_path,
            "--metrics-log-path",
            args.metrics_log_path,
            "--metrics-log-every",
            "10",
            "--signal-state-source",
            args.signal_state_source,
        ]

        if args.approach_roi:
            cmd.extend(["--approach-roi", args.approach_roi])
        if args.queue_roi:
            cmd.extend(["--queue-roi", args.queue_roi])
        if args.signal_state_roi:
            cmd.extend(["--signal-state-roi", args.signal_state_roi])

        print("Running main demo runtime...")
        rc = subprocess.run(cmd, cwd=root).returncode
        if rc != 0:
            print(f"main.py exited with code {rc}")
            return rc

        if args.with_orchestrator:
            print("Running orchestrator demo artifact...")
            subprocess.run(
                [python, "scripts/demo_orchestrator.py", "--frames", "120", "--fps", "15", "--preempt-hops", "2", "--latch-seconds", "3"],
                cwd=root,
                check=False,
            )

        if args.with_benchmark:
            print("Running baseline vs adaptive benchmark...")
            subprocess.run([python, "scripts/run_benchmark.py", "--frames", "1800", "--out", "artifacts/metrics.json"], cwd=root, check=False)

        if args.with_report:
            print("Generating demo report...")
            subprocess.run(
                [
                    python,
                    "scripts/generate_demo_report.py",
                    "--metrics-log",
                    args.metrics_log_path,
                    "--benchmark",
                    "artifacts/metrics.json",
                    "--orchestrator",
                    "artifacts/orchestrator_demo.json",
                    "--output-video",
                    args.output_path,
                    "--out",
                    args.report_path,
                ],
                cwd=root,
                check=False,
            )

        print("Demo complete. Artifacts:")
        print(f"  - {args.output_path}")
        print(f"  - {args.metrics_log_path}")
        if args.with_orchestrator:
            print("  - artifacts/orchestrator_demo.json")
        if args.with_benchmark:
            print("  - artifacts/metrics.json")
        if args.with_report:
            print(f"  - {args.report_path}")
        return 0
    finally:
        if sim_proc is not None:
            sim_proc.terminate()
            try:
                sim_proc.wait(timeout=3)
            except Exception:
                sim_proc.kill()

        if gps_proc is not None:
            gps_proc.terminate()
            try:
                gps_proc.wait(timeout=3)
            except Exception:
                gps_proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
