#!/usr/bin/env python3
"""
Test script for custom ambulance model integration.

Once you have trained your ambulance model, place it at:
assets/models/ambulance.pt

This script demonstrates the improved ambulance detection pipeline.
"""

from pathlib import Path
import cv2
from vision.detector import VehicleDetector

def test_ambulance_detection():
    """Test ambulance detection with custom model priority."""

    # Test image path (update this to your test image)
    test_image = "/Users/soumaydhrub/Downloads/carsdataset/ambulance/0AL642VPYDA4.jpg"

    if not Path(test_image).exists():
        print(f"Test image not found: {test_image}")
        return

    # Load image
    frame = cv2.imread(test_image)
    if frame is None:
        print("Failed to load test image")
        return

    print("Testing ambulance detection pipeline...")
    print("=" * 50)

    # Test with custom model (will fallback to YOLOWorld if custom not available)
    detector = VehicleDetector(ambulance_mode="custom")
    result = detector.detect(frame)

    print(f"Emergency detected: {result['emergency']}")
    print(f"Total vehicles: {result['count']}")

    print("\nVehicle detections:")
    for vehicle in result['vehicles']:
        print(f"  {vehicle['class']}: conf={vehicle['confidence']:.3f}, bbox={vehicle['bbox']}")

    # Check for ambulance specifically
    ambulances = [v for v in result['vehicles'] if v['class'] == 'ambulance']
    if ambulances:
        print(f"\n✅ Ambulance detected! ({len(ambulances)} instance(s))")
        for amb in ambulances:
            print(f"  Confidence: {amb['confidence']:.3f}")
    else:
        print("\n❌ No ambulance detected")

if __name__ == "__main__":
    test_ambulance_detection()