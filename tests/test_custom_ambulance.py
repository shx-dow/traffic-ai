from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
from vision.detector import VehicleDetector


def test_ambulance_detection():
    """Test ambulance detection with custom model priority."""

    # Test image path — resolved relative to project root
    test_image = ROOT / "assets" / "ambulance_dataset" / "carsdataset" / "ambulance" / "0AL642VPYDA4.jpg"

    if not test_image.exists():
        print(f"Test image not found: {test_image}")
        return

    frame = cv2.imread(str(test_image))
    if frame is None:
        print("Failed to load test image")
        return

    print("Testing ambulance detection pipeline...")
    print("=" * 50)

    detector = VehicleDetector(ambulance_mode="custom")
    result = detector.detect(frame)

    print(f"Emergency detected: {result['emergency']}")
    print(f"Total vehicles: {result['count']}")

    print("\nVehicle detections:")
    for vehicle in result['vehicles']:
        print(f"  {vehicle['class']}: conf={vehicle['confidence']:.3f}, bbox={vehicle['bbox']}")

    ambulances = [v for v in result['vehicles'] if v['class'] == 'ambulance']
    if ambulances:
        print(f"\n✅ Ambulance detected! ({len(ambulances)} instance(s))")
        for amb in ambulances:
            print(f"  Confidence: {amb['confidence']:.3f}")
    else:
        print("\n❌ No ambulance detected")


if __name__ == "__main__":
    test_ambulance_detection()
