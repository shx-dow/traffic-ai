from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import pytest


def check_video_file():
    """Check if video file exists and is readable. Falls back to sample_video.mp4."""
    video_path = ROOT / "assets" / "real_time_traffic" / "output.mp4"
    fallback_path = ROOT / "assets" / "sample_video.mp4"
    print("=" * 60)
    print("REAL_TIME_TRAFFIC DATASET CHECK")
    print("=" * 60)

    if not video_path.exists():
        if fallback_path.exists():
            print(
                f"⚠️  real_time_traffic/output.mp4 not found — using fallback: {fallback_path}"
            )
            video_path = fallback_path
        else:
            print(f"❌ Video file NOT found: {video_path}")
            return False, None

    print(f"✅ Video file found: {video_path}")
    file_size_gb = video_path.stat().st_size / (1024**3)
    print(f"   File size: {file_size_gb:.2f} GB")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = frame_count / fps if fps > 0 else 0

    print(f"\n📊 Video Properties:")
    print(f"   FPS: {fps}")
    print(f"   Frames: {frame_count}")
    print(f"   Resolution: {width}x{height}")
    print(f"   Duration: {duration_sec:.1f}s ({duration_sec / 60:.1f}m)")

    ret, frame = cap.read()
    if ret:
        print(f"\n✅ Frame reading: SUCCESS  shape={frame.shape}")
    else:
        print("\n❌ Frame reading: FAILED")
        cap.release()
        return False, None

    cap.release()
    return True, str(video_path)


@pytest.fixture
def video_path() -> str:
    is_valid, path = check_video_file()
    if not is_valid or not path:
        pytest.skip("No readable traffic video found for integration test")
    return path


def test_detection_on_video(video_path: str):
    """Run detection on sample frames from the video."""
    print("\n" + "=" * 60)
    print("RUNNING DETECTION TEST")
    print("=" * 60)

    from vision.detector import VehicleDetector

    cap = cv2.VideoCapture(video_path)
    detector = VehicleDetector(ambulance_mode="custom")

    frame_count = 0
    total_vehicles = 0
    frame_with_vehicles = 0
    emergency_count = 0

    frame_interval = 30
    test_frames = 5

    while frame_count < test_frames * frame_interval:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            result = detector.detect(frame)
            vehicles = result["count"]
            emergency = result["emergency"]

            total_vehicles += vehicles
            if vehicles > 0:
                frame_with_vehicles += 1
            if emergency:
                emergency_count += 1

            print(f"\n📹 Frame {frame_count}:")
            print(f"   Vehicles detected: {vehicles}")
            print(f"   Emergency: {emergency}")
            if result["vehicles"]:
                classes = ", ".join(set(v["class"] for v in result["vehicles"]))
                print(f"   Classes: {classes}")

        frame_count += 1

    cap.release()

    print("\n" + "=" * 60)
    print("DETECTION RESULTS SUMMARY")
    print("=" * 60)
    print(f"Frames tested: {test_frames}")
    print(f"Frames with vehicles: {frame_with_vehicles}")
    print(f"Total vehicles detected: {total_vehicles}")
    print(f"Average per frame: {total_vehicles / test_frames:.1f}")
    print(f"Emergency frames: {emergency_count}")

    if total_vehicles > 0:
        print("\n✅ REAL_TIME_TRAFFIC DATASET: WORKING CORRECTLY!")
    else:
        print("\n⚠️  No vehicles detected in test frames")


def main():
    print("\n🔍 Testing real_time_traffic dataset...\n")

    is_valid, video_path = check_video_file()

    if is_valid and video_path:
        test_detection_on_video(video_path)
    else:
        print("\n❌ Dataset check FAILED - video not accessible")
        print("\nTo fix:")
        print("  1. Check if assets/real_time_traffic/output.mp4 exists")
        print("  2. Extract real_traffic.zip: python scripts/extract_real_traffic.py")


if __name__ == "__main__":
    main()
