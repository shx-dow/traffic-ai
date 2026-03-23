
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

import config as app_config
from utils.video_sources import (ensure_real_traffic_extracted,
                                 first_video_under)
from vision.detector import VehicleDetector

# Classes the Day-1 brief asks to confirm on real traffic footage.
EXPECTED_SUBSET = frozenset({"car", "truck", "bus", "motorcycle"})


def open_capture(
    source: str | None,
) -> tuple[cv2.VideoCapture | None, str | None, str | None]:
    """
    Open a video file, numeric camera index, or default fallbacks.

    Returns (capture, None, video_source_hint) on success, or (None, error_message, None).
    """
    hint_synthetic = "Or run without camera/file: python tests/test_detector.py --synthetic"

    if source is not None and source.isdigit():
        cap = cv2.VideoCapture(int(source))
        if cap.isOpened():
            return cap, None, f"webcam:{source}"
        cap.release()
        return None, (
            f"Camera index {source} failed to open.\n"
            "  On macOS: System Settings → Privacy & Security → Camera → enable your terminal (or Cursor).\n"
            f"  {hint_synthetic}"
        ), None

    if source:
        p = Path(source).expanduser()
        if not p.is_file():
            return None, (
                f"Video file not found: {source}\n"
                "  Pass a real path to an .mp4 (the docs use /path/to/traffic.mp4 only as an example).\n"
                f"  {hint_synthetic}"
            ), None
        cap = cv2.VideoCapture(str(p))
        if cap.isOpened():
            return cap, None, str(p.resolve())
        cap.release()
        return None, (
            f"OpenCV could not read: {p}\n"
            "  Check the file is a supported format and not corrupted.\n"
            f"  {hint_synthetic}"
        ), None

    for candidate in (
        ROOT / "assets" / "sample_traffic.mp4",
        ROOT / "assets" / "sample.mp4",
    ):
        if candidate.is_file():
            cap = cv2.VideoCapture(str(candidate))
            if cap.isOpened():
                return cap, None, str(candidate.resolve())
            cap.release()

    hf_root = ROOT / app_config.REAL_TIME_TRAFFIC_ASSET_DIR
    ensure_real_traffic_extracted(hf_root, app_config.REAL_TRAFFIC_ZIP_NAME)
    hf_video = first_video_under(hf_root)
    if hf_video is not None:
        cap = cv2.VideoCapture(str(hf_video))
        if cap.isOpened():
            return cap, None, str(hf_video.resolve())
        cap.release()

    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        return cap, None, "webcam:0"
    cap.release()
    return None, (
        "No usable default source (no sample video, no video under assets/real_time_traffic/, webcam unavailable).\n"
        "  Put real_traffic.zip in assets/real_time_traffic/ (it auto-extracts on first run), or run:\n"
        "  python scripts/extract_real_traffic.py\n"
        "  See assets/real_time_traffic/DATASET.txt\n"
        f"  {hint_synthetic}\n"
        "  Or: --source /absolute/path/to/your/video.mp4"
    ), None


def run_synthetic(max_frames: int) -> int:
    """Benchmark detector FPS on random BGR frames (no camera or file required)."""
    detector = VehicleDetector()
    frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    seen_classes: set[str] = set()
    t0 = time.perf_counter()
    for i in range(max_frames):
        result = detector.detect(frame, video_source_hint="synthetic_noise")
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
    cap, open_err, video_hint = open_capture(source)
    if cap is None:
        print(f"Error: {open_err}", file=sys.stderr)
        return 1

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    detector = VehicleDetector()
    seen_classes: set[str] = set()
    t0 = time.perf_counter()
    frames = 0
    result: dict | None = None
    emergency_announced = False

    try:
        while frames < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            result = detector.detect(frame, video_source_hint=video_hint)
            count = result["count"]
            emergency = result["emergency"]
            for v in result["vehicles"]:
                seen_classes.add(v["class"])

            frames += 1
            elapsed = time.perf_counter() - t0
            fps = frames / elapsed if elapsed > 0 else 0.0
            if emergency and not emergency_announced:
                print(
                    "\n*** EMERGENCY: ambulance detected (YOLOWorld or aux weights) ***",
                    flush=True,
                )
                emergency_announced = True
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
    if result and result.get("emergency"):
        print("Note: emergency=True on last processed frame (ambulance present in that frame).")
    missing = EXPECTED_SUBSET - seen_classes
    if missing:
        print(
            f"Note: expected demo classes not seen in this clip (normal for short/random video): "
            f"{sorted(missing)}",
        )
    else:
        print("Confirmed detections include car, truck, bus, and motorcycle.")

    if result and "fusion" in result:
        fus = result["fusion"]
        print()
        print("Cross-dataset output (real_time_traffic video + FindVehicle ontology):")
        print(f"  {fus.get('summary', '')}")
        print(f"  vision: {fus['vision']['dataset']} | source: {fus['vision'].get('video_source_hint')}")
        print(f"  text schema: {fus['text_schema']['dataset']} ({fus['text_schema'].get('entity_family', '')})")
        if fus.get("per_vehicle"):
            ex = fus["per_vehicle"][0]
            print(
                f"  example alignment: COCO '{ex.get('coco_class')}' → FindVehicle stems {ex.get('findvehicle_vehicle_type_stems', [])[:4]}…",
            )

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
