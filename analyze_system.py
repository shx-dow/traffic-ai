#!/usr/bin/env python3
"""
Analyze the traffic detection system capabilities.

This script analyzes the existing VehicleDetector system and explains its capabilities
without modifying any core logic.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def check_vehicle_detection():
    """Check if vehicle detection is available."""
    try:
        from ultralytics import YOLO
        # Try to load the model
        model = YOLO("yolov8n.pt")
        return True, "YOLOv8 model loads successfully"
    except Exception as e:
        return False, f"YOLOv8 unavailable: {e}"

def check_yolo_world():
    """Check if YOLOWorld is available."""
    try:
        from ultralytics import YOLOWorld
        return True, "YOLOWorld available for open-vocabulary detection"
    except ImportError:
        return False, "YOLOWorld not installed (CLIP dependency missing)"
    except Exception as e:
        return False, f"YOLOWorld error: {e}"

def check_custom_ambulance_model():
    """Check if custom ambulance model exists."""
    model_path = ROOT / "assets" / "models" / "ambulance.pt"
    if model_path.exists():
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            return True, f"Custom ambulance model loaded from {model_path}"
        except Exception as e:
            return False, f"Custom model exists but failed to load: {e}"
    else:
        return False, f"Custom model not found at {model_path}"

def check_emergency_support():
    """Check if emergency detection is supported."""
    # This is always supported in the current system
    return True, "Emergency flag supported via ambulance detection"

def run_sample_test():
    """Run a quick test on a sample image if available."""
    # Look for sample images in common locations
    sample_paths = [
        ROOT / "assets" / "sample_image.jpg",
        ROOT / "assets" / "sample_image.png",
        ROOT / "images" / "train" / "sample.jpg",  # Check if any image exists
    ]

    # Find first existing image
    test_image = None
    for path in sample_paths:
        if path.exists():
            test_image = path
            break

    # If no sample image, try to find any image in the project (prefer traffic-related)
    if not test_image:
        traffic_keywords = ['car', 'vehicle', 'traffic', 'ambulance', 'truck', 'bus']
        for ext in ['.jpg', '.png', '.jpeg']:
            for file in ROOT.rglob(f'*{ext}'):
                filename = str(file).lower()
                if any(keyword in filename for keyword in traffic_keywords):
                    test_image = file
                    break
            if test_image:
                break

    # Last resort: any image file
    if not test_image:
        for ext in ['.jpg', '.png', '.jpeg']:
            for file in ROOT.rglob(f'*{ext}'):
                if 'ambulance' not in str(file).lower():  # Avoid test images that might be confusing
                    test_image = file
                    break
            if test_image:
                break

    if not test_image:
        return None, "No sample image found for testing"

    try:
        import cv2
        from vision.detector import VehicleDetector

        # Load image
        frame = cv2.imread(str(test_image))
        if frame is None:
            return None, f"Failed to load image: {test_image}"

        # Run detection
        detector = VehicleDetector()
        result = detector.detect(frame)

        # Format results
        classes = [v['class'] for v in result['vehicles']]
        class_counts = {}
        for cls in classes:
            class_counts[cls] = class_counts.get(cls, 0) + 1

        class_summary = ", ".join([f"{cls}: {count}" for cls, count in class_counts.items()]) if class_counts else "None detected"

        return {
            'image': str(test_image),
            'total_vehicles': result['count'],
            'classes': class_summary,
            'emergency': result['emergency']
        }, None

    except Exception as e:
        return None, f"Test failed: {e}"

def print_capabilities():
    """Print system capabilities."""
    print("=" * 60)
    print("TRAFFIC DETECTION SYSTEM CAPABILITIES")
    print("=" * 60)

    # Check each capability
    capabilities = [
        ("Vehicle Detection", check_vehicle_detection()),
        ("YOLOWorld Support", check_yolo_world()),
        ("Custom Ambulance Model", check_custom_ambulance_model()),
        ("Emergency Flag Support", check_emergency_support()),
        ("Lane Detection", (False, "Not implemented")),
        ("Vehicle Tracking", (False, "Not implemented")),
        ("Traffic Light Detection", (False, "Not implemented")),
        ("Speed Estimation", (False, "Not implemented")),
    ]

    for name, (available, details) in capabilities:
        status = "YES" if available else "NO"
        print(f"• {name}: {status}")
        if details:
            print(f"  └─ {details}")

    print()

def print_output_format():
    """Print output format explanation."""
    print("=" * 60)
    print("OUTPUT FORMAT EXPLANATION")
    print("=" * 60)

    print("""
The VehicleDetector.detect() method returns a dictionary with:

{
    "vehicles": [
        {
            "class": "car|truck|bus|motorcycle|bicycle|ambulance",
            "confidence": 0.85,  // Float between 0-1
            "bbox": [x1, y1, x2, y2]  // Bounding box coordinates
        },
        ...
    ],
    "count": 3,  // Total number of vehicles detected
    "emergency": false  // True if ambulance detected
}

• vehicles: List of detected objects with class name, confidence score, and bounding box
• count: Total number of vehicles (same as len(vehicles))
• emergency: Boolean flag indicating if emergency vehicle (ambulance) was detected
""")

def print_testing_guide():
    """Print testing guide."""
    print("=" * 60)
    print("HOW TO TEST THE SYSTEM")
    print("=" * 60)

    print("""
1. TEST VEHICLE DETECTION:
   python tests/test_detector.py --source assets/sample_video.mp4
   python tests/test_detector.py --source 0  # Webcam

2. TEST AMBULANCE DETECTION:
   python scripts/validate_ambulance.py --image path/to/ambulance.jpg
   python scripts/validate_ambulance.py --image path/to/ambulance.jpg --ambulance-conf 0.15

3. RUN YOLO DIRECTLY:
   yolo task=detect mode=predict model=yolov8n.pt source=assets/sample_video.mp4

4. ANALYZE FINDVEHICLE DATA:
   python scripts/analyze_findvehicle.py --jsonl assets/findvehicle/sample.jsonl

5. RUN THIS ANALYSIS:
   python analyze_system.py
""")

def main():
    """Main analysis function."""
    print("🔍 Analyzing Traffic Detection System...\n")

    # Print capabilities
    print_capabilities()

    # Print output format
    print_output_format()

    # Print testing guide
    print_testing_guide()

    # Run sample test
    print("=" * 60)
    print("SAMPLE TEST RESULTS")
    print("=" * 60)

    test_result, error = run_sample_test()
    if test_result:
        print(f"✅ Test Image: {test_result['image']}")
        print(f"📊 Total Vehicles: {test_result['total_vehicles']}")
        print(f"🏷️  Detected Classes: {test_result['classes']}")
        print(f"🚨 Emergency Detected: {test_result['emergency']}")
    else:
        print(f"❌ Sample Test: {error}")
        print("💡 To run tests, ensure you have sample images in assets/ or images/ directories")

    print("\n" + "=" * 60)
    print("🎯 SYSTEM ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()