#!/usr/bin/env python3
"""
Run VehicleDetector on a video file or webcam.

Prints per-frame vehicle count and emergency flag (single-line updates to limit I/O),
then reports average FPS and which COCO vehicle classes were seen.

Usage:
  python tests/test_detector.py --source path/to/video.mp4
  python tests/test_detector.py --source 0          # default webcam
  python tests/test_detector.py                     # tries assets/sample.mp4 then camera 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.detector import VehicleDetector

# Classes the Day-1 brief asks to confirm on real traffic footage.
EXPECTED_SUBSET = frozenset({"car", "truck", "bus", "motorcycle"})


def open_capture(source: str | None) -> cv2.VideoCapture:
    """Open a video file, numeric camera index, or default fallbacks."""
    if source is not None and source.isdigit():
        return cv2.VideoCapture(int(source))
    if source:
        p = Path(source)
        cap = cv2.VideoCapture(str(p.expanduser()))
        if cap.isOpened():
            return cap
        raise FileNotFoundError(f"Could not open video source: {source}")

    for candidate in (
        ROOT / "assets" / "sample_traffic.mp4",
        ROOT / "assets" / "sample.mp4",
    ):
        if candidate.is_file():
            cap = cv2.VideoCapture(str(candidate))
            if cap.isOpened():
                return cap

    return cv2.VideoCapture(0)


def run_synthetic(max_frames: int) -> int:
    """Benchmark detector FPS on random BGR frames (no camera or file required)."""
    detector = VehicleDetector()
    frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    seen_classes: set[str] = set()
    t0 = time.perf_counter()
    for i in range(max_frames):
        result = detector.detect(frame)
        for v in result["vehicles"]:
            seen_classes.add(v["class"])
        fps = (i + 1) / (time.perf_counter() - t0)
        print(
            f"\rframe={i + 1} count={result['count']} emergency={result['emergency']} inst_fps={fps:.1f}   ",
            end="",
            flush=True,
        )
    elapsed = time.perf_counter() - t0
    avg_fps = max_frames / elapsed if elapsed > 0 else 0.0
    print()
    print(f"Synthetic benchmark: {max_frames} frames in {elapsed:.2f}s → {avg_fps:.1f} FPS")
    print(f"Vehicle classes seen: {sorted(seen_classes)}")
    if avg_fps < 10.0 and max_frames > 5:
        print(f"Warning: average FPS ({avg_fps:.1f}) is below 10.", file=sys.stderr)
    return 0


def run(
    source: str | None,
    max_frames: int,
    width: int | None,
    height: int | None,
) -> int:
    cap = open_capture(source)
    if not cap.isOpened():
        print("Error: could not open any video source. Pass --source /path/to/video.mp4", file=sys.stderr)
        return 1

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    detector = VehicleDetector()
    seen_classes: set[str] = set()
    t0 = time.perf_counter()
    frames = 0

    try:
        while frames < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            result = detector.detect(frame)
            count = result["count"]
            emergency = result["emergency"]
            for v in result["vehicles"]:
                seen_classes.add(v["class"])

            frames += 1
            elapsed = time.perf_counter() - t0
            fps = frames / elapsed if elapsed > 0 else 0.0
            print(
                f"\rframe={frames} count={count} emergency={emergency} inst_fps={fps:.1f}   ",
                end="",
                flush=True,
            )
    finally:
        cap.release()

    elapsed = time.perf_counter() - t0
    avg_fps = frames / elapsed if elapsed > 0 else 0.0

    print()
    print(f"Frames processed: {frames}")
    print(f"Elapsed: {elapsed:.2f}s  Average FPS: {avg_fps:.1f}")
    print(f"Vehicle classes seen: {sorted(seen_classes)}")
    missing = EXPECTED_SUBSET - seen_classes
    if missing:
        print(
            f"Note: expected demo classes not seen in this clip (normal for short/random video): "
            f"{sorted(missing)}",
        )
    else:
        print("Confirmed detections include car, truck, bus, and motorcycle.")

    if avg_fps < 10.0 and frames > 5:
        print(
            f"Warning: average FPS ({avg_fps:.1f}) is below 10. Try smaller resolution, GPU, or fewer logs.",
            file=sys.stderr,
        )

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="VehicleDetector integration test / demo")
    parser.add_argument(
        "--source",
        default=None,
        help="Video file path or camera index (e.g. 0). Default: assets/sample*.mp4 or webcam.",
    )
    parser.add_argument("--max-frames", type=int, default=500, help="Stop after this many frames")
    parser.add_argument("--width", type=int, default=None, help="Optional capture width")
    parser.add_argument("--height", type=int, default=None, help="Optional capture height")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Ignore video; benchmark on random frames (FPS check without camera/file).",
    )
    args = parser.parse_args()

    if args.synthetic:
        raise SystemExit(run_synthetic(max_frames=args.max_frames))

    raise SystemExit(
        run(
            source=args.source,
            max_frames=args.max_frames,
            width=args.width,
            height=args.height,
        )
    )


if __name__ == "__main__":
    main()
