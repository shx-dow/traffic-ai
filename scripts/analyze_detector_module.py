"""Comprehensive analysis of vision/detector.py and tests/test_detector.py."""
from __future__ import annotations

import sys
from inspect import signature
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check_requirement(name: str, condition: bool, details: str = "") -> bool:
    status = "✅ PASS" if condition else "❌ FAIL"
    print(f"{status}: {name}")
    if details:
        print(f"   └─ {details}")
    return condition


def analyze_vision_detector() -> bool:
    print("=" * 70)
    print("VISION/DETECTOR.PY - CRITICAL MODULE ANALYSIS")
    print("=" * 70)

    all_pass = True

    print("\n📋 REQUIREMENT 1: VehicleDetector Class Implementation")
    print("-" * 70)
    try:
        from vision.detector import VehicleDetector
        all_pass &= check_requirement("VehicleDetector class exists", True, "Class successfully imported")
    except Exception as e:
        all_pass &= check_requirement("VehicleDetector class exists", False, f"Import failed: {e}")
        return all_pass

    print("\n📋 REQUIREMENT 2: __init__ Method")
    print("-" * 70)
    try:
        sig = signature(VehicleDetector.__init__)
        params = list(sig.parameters.keys())
        all_pass &= check_requirement("Has __init__ method", True, f"Parameters: {params}")
        all_pass &= check_requirement("Accepts model_path parameter", "model_path" in params)
        # _model is set on instances, not the class — instantiate to verify
        import numpy as np
        _tmp = VehicleDetector(model_path="yolov8n.pt")
        all_pass &= check_requirement("Loads YOLO model", hasattr(_tmp, "_model"))
    except Exception as e:
        all_pass &= check_requirement("__init__ method", False, str(e))

    print("\n📋 REQUIREMENT 3: detect() Method")
    print("-" * 70)
    try:
        sig = signature(VehicleDetector.detect)
        params = list(sig.parameters.keys())
        all_pass &= check_requirement("Has detect method", "detect" in dir(VehicleDetector), f"Signature: detect({', '.join(params)})")
        all_pass &= check_requirement("Accepts frame parameter", "frame" in params)
        return_annotation = sig.return_annotation
        # Accept Dict[str, Any], dict, or any annotation containing "dict"
        ann_str = str(return_annotation).lower()
        all_pass &= check_requirement("Returns dict", "dict" in ann_str, f"Return type: {return_annotation}")
    except Exception as e:
        all_pass &= check_requirement("detect() method", False, str(e))

    print("\n📋 REQUIREMENT 4: Output Contract (Critical)")
    print("-" * 70)
    try:
        import numpy as np
        from vision.detector import VehicleDetector

        detector = VehicleDetector(model_path="yolov8n.pt")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(dummy_frame)

        all_pass &= check_requirement("Has 'vehicles' key", "vehicles" in result, f"Type: {type(result.get('vehicles'))}")
        all_pass &= check_requirement("Has 'count' key", "count" in result, f"Type: {type(result.get('count'))}")
        all_pass &= check_requirement("Has 'emergency' key", "emergency" in result, f"Type: {type(result.get('emergency'))}")
        all_pass &= check_requirement("Has 'raw_result' key", "raw_result" in result)
        all_pass &= check_requirement("vehicles is list", isinstance(result.get("vehicles"), list))
        all_pass &= check_requirement("count is integer", isinstance(result.get("count"), int), f"Value: {result.get('count')}")
        all_pass &= check_requirement("emergency is boolean", isinstance(result.get("emergency"), bool), f"Value: {result.get('emergency')}")
    except Exception as e:
        all_pass &= check_requirement("Output contract", False, str(e))

    print("\n📋 REQUIREMENT 5: Vehicle Classes Detection")
    print("-" * 70)
    try:
        from vision.detector import VEHICLE_CLASSES, TRACKED_CLASSES
        expected = {"car", "truck", "bus", "motorcycle", "bicycle"}
        all_pass &= check_requirement("VEHICLE_CLASSES defined", len(VEHICLE_CLASSES) > 0, f"Classes: {', '.join(sorted(VEHICLE_CLASSES))}")
        all_pass &= check_requirement("Includes required classes", expected.issubset(VEHICLE_CLASSES), f"Missing: {expected - VEHICLE_CLASSES}")
        all_pass &= check_requirement("TRACKED_CLASSES includes emergency", "ambulance" in TRACKED_CLASSES)
    except Exception as e:
        all_pass &= check_requirement("Vehicle classes", False, str(e))

    print("\n📋 REQUIREMENT 6: Confidence Threshold")
    print("-" * 70)
    try:
        from vision.detector import CONFIDENCE_THRESHOLD
        all_pass &= check_requirement("CONFIDENCE_THRESHOLD defined", CONFIDENCE_THRESHOLD is not None, f"Value: {CONFIDENCE_THRESHOLD}")
        all_pass &= check_requirement("Threshold is reasonable", 0.3 <= CONFIDENCE_THRESHOLD <= 0.5, f"Current: {CONFIDENCE_THRESHOLD}")
    except Exception as e:
        all_pass &= check_requirement("Confidence threshold", False, str(e))

    print("\n📋 REQUIREMENT 7: Ambulance/Emergency Detection")
    print("-" * 70)
    try:
        from vision.detector import EMERGENCY_CLASS_NAMES
        all_pass &= check_requirement("EMERGENCY_CLASS_NAMES defined", len(EMERGENCY_CLASS_NAMES) > 0, f"Classes: {EMERGENCY_CLASS_NAMES}")
        all_pass &= check_requirement("Includes 'ambulance' class", "ambulance" in EMERGENCY_CLASS_NAMES)
        detector_code = (ROOT / "vision" / "detector.py").read_text()
        all_pass &= check_requirement("Ambulance detection implemented", "_merge_ambulance_detections" in detector_code)
    except Exception as e:
        all_pass &= check_requirement("Ambulance detection", False, str(e))

    return all_pass


def analyze_test_detector() -> bool:
    print("\n" + "=" * 70)
    print("TESTS/TEST_DETECTOR.PY - TESTING MODULE ANALYSIS")
    print("=" * 70)

    all_pass = True
    test_file = ROOT / "tests" / "test_detector.py"

    all_pass &= check_requirement("test_detector.py exists", test_file.exists(), f"Location: {test_file}")
    if not test_file.exists():
        return all_pass

    content = test_file.read_text()
    all_pass &= check_requirement("Uses argparse", "argparse" in content)
    all_pass &= check_requirement("Accepts --source argument", "--source" in content)
    all_pass &= check_requirement("Supports --synthetic mode", "--synthetic" in content)
    all_pass &= check_requirement("Uses OpenCV VideoCapture", "VideoCapture" in content)
    all_pass &= check_requirement("Has fallback logic", "real_time_traffic" in content or "webcam" in content)
    all_pass &= check_requirement("Measures FPS", "fps" in content or "FPS" in content)
    all_pass &= check_requirement("Uses timing functions", "time.time()" in content or "time.perf_counter()" in content)

    return all_pass


def test_real_detection() -> bool:
    print("\n" + "=" * 70)
    print("LIVE TEST: Real Detection on Sample Frame")
    print("=" * 70)

    all_pass = True
    try:
        import numpy as np
        from vision.detector import VehicleDetector

        print("\n🔧 Initializing VehicleDetector...")
        detector = VehicleDetector(model_path="yolov8n.pt")
        all_pass &= check_requirement("VehicleDetector initialized", detector is not None)

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        all_pass &= check_requirement("Detection executed", result is not None)

        if result:
            vehicles = result.get("vehicles", [])
            count = result.get("count", 0)
            emergency = result.get("emergency", False)
            all_pass &= check_requirement("Returns correct count", count == len(vehicles), f"count={count}, len(vehicles)={len(vehicles)}")
            all_pass &= check_requirement("Emergency flag is boolean", isinstance(emergency, bool), f"Value: {emergency}")
    except Exception as e:
        all_pass &= check_requirement("Live detection test", False, str(e))

    return all_pass


def main():
    print("\n🔍 COMPREHENSIVE SYSTEM ANALYSIS - VEHICLE DETECTION MODULE")
    print("=" * 70)

    r1 = analyze_vision_detector()
    r2 = analyze_test_detector()
    r3 = test_real_detection()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if r1 and r2 and r3:
        print("\n✅ ALL REQUIREMENTS MET — system ready.")
    else:
        print("\n⚠️  SOME REQUIREMENTS NOT MET — review ❌ FAIL items above.")

    print("\nNext steps:")
    print("  python tests/test_detector.py --source assets/real_time_traffic/output.mp4")
    print("  python scripts/validate_ambulance.py --image <ambulance_image>")


if __name__ == "__main__":
    main()
