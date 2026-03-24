"""Analyze and report traffic detection system capabilities."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check_vehicle_detection():
    try:
        from ultralytics import YOLO
        YOLO("yolov8n.pt")
        return True, "YOLOv8 model loads successfully"
    except Exception as e:
        return False, f"YOLOv8 unavailable: {e}"


def check_yolo_world():
    try:
        from ultralytics import YOLOWorld  # noqa: F401
        return True, "YOLOWorld available for open-vocabulary detection"
    except ImportError:
        return False, "YOLOWorld not installed (CLIP dependency missing)"
    except Exception as e:
        return False, f"YOLOWorld error: {e}"


def check_custom_ambulance_model():
    model_path = ROOT / "assets" / "models" / "ambulance.pt"
    if model_path.exists():
        try:
            from ultralytics import YOLO
            YOLO(str(model_path))
            return True, f"Custom ambulance model loaded from {model_path}"
        except Exception as e:
            return False, f"Custom model exists but failed to load: {e}"
    return False, f"Custom model not found at {model_path}"


def run_sample_test():
    """Run a quick detection test on the first image found in assets."""
    test_image = None
    for ext in (".jpg", ".png", ".jpeg"):
        for f in ROOT.rglob(f"*{ext}"):
            test_image = f
            break
        if test_image:
            break

    if not test_image:
        return None, "No sample image found"

    try:
        import cv2

        from vision.detector import VehicleDetector

        frame = cv2.imread(str(test_image))
        if frame is None:
            return None, f"Failed to load image: {test_image}"

        detector = VehicleDetector()
        result = detector.detect(frame)

        classes = [v["class"] for v in result["vehicles"]]
        class_counts: dict = {}
        for cls in classes:
            class_counts[cls] = class_counts.get(cls, 0) + 1

        class_summary = ", ".join(f"{c}: {n}" for c, n in class_counts.items()) or "None detected"
        return {"image": str(test_image), "total_vehicles": result["count"], "classes": class_summary, "emergency": result["emergency"]}, None
    except Exception as e:
        return None, f"Test failed: {e}"


def main():
    print("🔍 Analyzing Traffic Detection System...\n")
    print("=" * 60)
    print("SYSTEM CAPABILITIES")
    print("=" * 60)

    capabilities = [
        ("Vehicle Detection", check_vehicle_detection()),
        ("YOLOWorld Support", check_yolo_world()),
        ("Custom Ambulance Model", check_custom_ambulance_model()),
        ("Emergency Flag Support", (True, "Supported via ambulance detection")),
        ("Lane Detection", (False, "Not implemented")),
        ("Vehicle Tracking", (False, "Not implemented")),
    ]

    for name, (available, details) in capabilities:
        print(f"• {name}: {'YES' if available else 'NO'}")
        if details:
            print(f"  └─ {details}")

    print("\n" + "=" * 60)
    print("SAMPLE TEST RESULTS")
    print("=" * 60)

    test_result, error = run_sample_test()
    if test_result:
        print(f"✅ Image: {test_result['image']}")
        print(f"   Vehicles: {test_result['total_vehicles']}")
        print(f"   Classes: {test_result['classes']}")
        print(f"   Emergency: {test_result['emergency']}")
    else:
        print(f"❌ {error}")

    print("\n" + "=" * 60)
    print("HOW TO TEST")
    print("=" * 60)
    print("  python tests/test_detector.py --source assets/sample_video.mp4")
    print("  python scripts/validate_ambulance.py --image path/to/ambulance.jpg")
    print("  python scripts/analyze_findvehicle.py --jsonl assets/findvehicle/sample.jsonl")


if __name__ == "__main__":
    main()
