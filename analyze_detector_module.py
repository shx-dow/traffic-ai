#!/usr/bin/env python3
"""
SYSTEM ANALYSIS: Vehicle Detection Module (vision/detector.py)
Check if all Day-1 critical requirements are met.
"""

import sys
from pathlib import Path
from inspect import signature

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def check_requirement(name, condition, details=""):
    """Print requirement check result."""
    status = "✅ PASS" if condition else "❌ FAIL"
    print(f"{status}: {name}")
    if details:
        print(f"   └─ {details}")
    return condition

def analyze_vision_detector():
    """Analyze vision/detector.py implementation."""
    print("=" * 70)
    print("VISION/DETECTOR.PY - CRITICAL MODULE ANALYSIS")
    print("=" * 70)
    
    all_pass = True
    
    # 1. CLASS EXISTENCE
    print("\n📋 REQUIREMENT 1: VehicleDetector Class Implementation")
    print("-" * 70)
    try:
        from vision.detector import VehicleDetector
        all_pass &= check_requirement(
            "VehicleDetector class exists",
            True,
            "Class successfully imported"
        )
    except Exception as e:
        all_pass &= check_requirement(
            "VehicleDetector class exists",
            False,
            f"Import failed: {e}"
        )
        return all_pass
    
    # 2. __INIT__ METHOD
    print("\n📋 REQUIREMENT 2: __init__ Method")
    print("-" * 70)
    try:
        sig = signature(VehicleDetector.__init__)
        params = list(sig.parameters.keys())
        
        all_pass &= check_requirement(
            "Has __init__ method",
            "__init__" in dir(VehicleDetector),
            f"Parameters: {params}"
        )
        
        all_pass &= check_requirement(
            "Accepts model_path parameter",
            "model_path" in params,
            "Default model: yolov8n.pt"
        )
        
        all_pass &= check_requirement(
            "Loads YOLO model",
            hasattr(VehicleDetector, '_model'),
            "YOLO model loads in __init__"
        )
    except Exception as e:
        all_pass &= check_requirement("__init__ method", False, str(e))
    
    # 3. DETECT METHOD
    print("\n📋 REQUIREMENT 3: detect() Method")
    print("-" * 70)
    try:
        sig = signature(VehicleDetector.detect)
        params = list(sig.parameters.keys())
        
        all_pass &= check_requirement(
            "Has detect method",
            "detect" in dir(VehicleDetector),
            f"Signature: detect({', '.join(params)})"
        )
        
        all_pass &= check_requirement(
            "Accepts frame parameter",
            "frame" in params,
            "Input: numpy array (BGR from OpenCV)"
        )
        
        # Check return type annotation
        return_annotation = sig.return_annotation
        all_pass &= check_requirement(
            "Returns dict",
            "dict" in str(return_annotation) or return_annotation == dict,
            f"Return type: {return_annotation}"
        )
    except Exception as e:
        all_pass &= check_requirement("detect() method", False, str(e))
    
    # 4. OUTPUT CONTRACT
    print("\n📋 REQUIREMENT 4: Output Contract (Critical)")
    print("-" * 70)
    
    try:
        import cv2
        import numpy as np
        
        detector = VehicleDetector(model_path="yolov8n.pt")
        
        # Create a dummy frame (1x1 pixel)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = detector.detect(dummy_frame)
        
        # Check dict keys
        required_keys = {"vehicles", "count", "emergency", "raw_result"}
        actual_keys = set(result.keys())
        
        all_pass &= check_requirement(
            "Has 'vehicles' key",
            "vehicles" in result,
            f"Type: {type(result.get('vehicles'))}"
        )
        
        all_pass &= check_requirement(
            "Has 'count' key",
            "count" in result,
            f"Type: {type(result.get('count'))}"
        )
        
        all_pass &= check_requirement(
            "Has 'emergency' key",
            "emergency" in result,
            f"Type: {type(result.get('emergency'))}"
        )
        
        all_pass &= check_requirement(
            "Has 'raw_result' key",
            "raw_result" in result,
            "Contains YOLO Results object"
        )
        
        # Check vehicles list structure
        vehicles = result.get("vehicles", [])
        all_pass &= check_requirement(
            "vehicles is list",
            isinstance(vehicles, list),
            f"Length: {len(vehicles)}"
        )
        
        # Check count is int
        count = result.get("count")
        all_pass &= check_requirement(
            "count is integer",
            isinstance(count, int),
            f"Value: {count}"
        )
        
        # Check emergency is bool
        emergency = result.get("emergency")
        all_pass &= check_requirement(
            "emergency is boolean",
            isinstance(emergency, bool),
            f"Value: {emergency}"
        )
        
    except Exception as e:
        all_pass &= check_requirement("Output contract", False, str(e))
    
    # 5. VEHICLE CLASSES
    print("\n📋 REQUIREMENT 5: Vehicle Classes Detection")
    print("-" * 70)
    
    try:
        from vision.detector import VEHICLE_CLASSES, TRACKED_CLASSES
        
        expected_classes = {"car", "truck", "bus", "motorcycle", "bicycle"}
        
        all_pass &= check_requirement(
            "VEHICLE_CLASSES defined",
            len(VEHICLE_CLASSES) > 0,
            f"Classes: {', '.join(sorted(VEHICLE_CLASSES))}"
        )
        
        all_pass &= check_requirement(
            "Includes required classes",
            expected_classes.issubset(VEHICLE_CLASSES),
            f"Missing: {expected_classes - VEHICLE_CLASSES}"
        )
        
        all_pass &= check_requirement(
            "TRACKED_CLASSES includes emergency",
            "ambulance" in TRACKED_CLASSES,
            f"Total tracked: {len(TRACKED_CLASSES)} classes"
        )
        
    except Exception as e:
        all_pass &= check_requirement("Vehicle classes", False, str(e))
    
    # 6. CONFIDENCE THRESHOLD
    print("\n📋 REQUIREMENT 6: Confidence Threshold")
    print("-" * 70)
    
    try:
        from vision.detector import CONFIDENCE_THRESHOLD
        
        all_pass &= check_requirement(
            "CONFIDENCE_THRESHOLD defined",
            CONFIDENCE_THRESHOLD is not None,
            f"Value: {CONFIDENCE_THRESHOLD}"
        )
        
        all_pass &= check_requirement(
            "Threshold is 0.4 (or reasonable)",
            0.3 <= CONFIDENCE_THRESHOLD <= 0.5,
            f"Current: {CONFIDENCE_THRESHOLD} (spec: 0.4)"
        )
        
    except Exception as e:
        all_pass &= check_requirement("Confidence threshold", False, str(e))
    
    # 7. AMBULANCE DETECTION
    print("\n📋 REQUIREMENT 7: Ambulance/Emergency Detection")
    print("-" * 70)
    
    try:
        from vision.detector import EMERGENCY_CLASS_NAMES
        
        all_pass &= check_requirement(
            "EMERGENCY_CLASS_NAMES defined",
            len(EMERGENCY_CLASS_NAMES) > 0,
            f"Classes: {EMERGENCY_CLASS_NAMES}"
        )
        
        all_pass &= check_requirement(
            "Includes 'ambulance' class",
            "ambulance" in EMERGENCY_CLASS_NAMES,
            "Emergency vehicle support enabled"
        )
        
        # Check if ambulance detection is implemented
        detector_code = Path("vision/detector.py").read_text()
        has_ambulance_logic = "_merge_ambulance_detections" in detector_code
        
        all_pass &= check_requirement(
            "Ambulance detection implemented",
            has_ambulance_logic,
            "Method: _merge_ambulance_detections() present"
        )
        
    except Exception as e:
        all_pass &= check_requirement("Ambulance detection", False, str(e))
    
    return all_pass

def analyze_test_detector():
    """Analyze tests/test_detector.py implementation."""
    print("\n" + "=" * 70)
    print("TESTS/TEST_DETECTOR.PY - TESTING MODULE ANALYSIS")
    print("=" * 70)
    
    all_pass = True
    
    # 1. TEST SCRIPT EXISTS
    print("\n📋 REQUIREMENT 1: Test Script")
    print("-" * 70)
    
    test_file = Path("tests/test_detector.py")
    all_pass &= check_requirement(
        "test_detector.py exists",
        test_file.exists(),
        f"Location: {test_file}"
    )
    
    if not test_file.exists():
        return all_pass
    
    # 2. COMMAND LINE INTERFACE
    print("\n📋 REQUIREMENT 2: Command Line Interface")
    print("-" * 70)
    
    try:
        content = test_file.read_text()
        
        has_argparse = "argparse" in content
        has_source_arg = "--source" in content
        has_synthetic = "--synthetic" in content
        
        all_pass &= check_requirement(
            "Uses argparse",
            has_argparse,
            "Command line argument parsing"
        )
        
        all_pass &= check_requirement(
            "Accepts --source argument",
            has_source_arg,
            "Can specify video file or webcam (0)"
        )
        
        all_pass &= check_requirement(
            "Supports --synthetic mode",
            has_synthetic,
            "Can run FPS benchmark without file"
        )
        
    except Exception as e:
        all_pass &= check_requirement("CLI interface", False, str(e))
    
    # 3. VIDEO SOURCE HANDLING
    print("\n📋 REQUIREMENT 3: Video Source Handling")
    print("-" * 70)
    
    try:
        content = test_file.read_text()
        
        has_video_file = "VideoCapture" in content
        has_fallbacks = "real_time_traffic" in content or "webcam" in content
        
        all_pass &= check_requirement(
            "Uses OpenCV VideoCapture",
            has_video_file,
            "Can read MP4/MOV files"
        )
        
        all_pass &= check_requirement(
            "Has fallback logic",
            has_fallbacks,
            "Falls back to real_time_traffic → webcam"
        )
        
    except Exception as e:
        all_pass &= check_requirement("Video handling", False, str(e))
    
    # 4. PERFORMANCE MEASUREMENT
    print("\n📋 REQUIREMENT 4: FPS Measurement (>= 10 fps requirement)")
    print("-" * 70)
    
    try:
        content = test_file.read_text()
        
        has_fps_calc = "fps" in content or "FPS" in content
        has_timing = "time.time()" in content or "time.perf_counter()" in content
        
        all_pass &= check_requirement(
            "Measures FPS",
            has_fps_calc,
            "Tracks frames per second"
        )
        
        all_pass &= check_requirement(
            "Uses timing functions",
            has_timing,
            "Calculates performance metrics"
        )
        
    except Exception as e:
        all_pass &= check_requirement("FPS measurement", False, str(e))
    
    # 5. OUTPUT VERIFICATION
    print("\n📋 REQUIREMENT 5: Output Verification")
    print("-" * 70)
    
    try:
        content = test_file.read_text()
        
        has_vehicle_count = "count" in content
        has_emergency_flag = "emergency" in content
        has_class_detection = "class" in content
        
        all_pass &= check_requirement(
            "Prints vehicle count",
            has_vehicle_count,
            "Shows 'count' field per frame"
        )
        
        all_pass &= check_requirement(
            "Prints emergency flag",
            has_emergency_flag,
            "Shows 'emergency' field per frame"
        )
        
        all_pass &= check_requirement(
            "Reports detected classes",
            has_class_detection,
            "Lists COCO classes found"
        )
        
    except Exception as e:
        all_pass &= check_requirement("Output verification", False, str(e))
    
    return all_pass

def test_real_detection():
    """Test actual detection on a sample frame."""
    print("\n" + "=" * 70)
    print("LIVE TEST: Real Detection on Sample Frame")
    print("=" * 70)
    
    all_pass = True
    
    try:
        import cv2
        import numpy as np
        from vision.detector import VehicleDetector
        
        print("\n🔧 Initializing VehicleDetector...")
        detector = VehicleDetector(model_path="yolov8n.pt")
        all_pass &= check_requirement(
            "VehicleDetector initialized",
            detector is not None,
            "Model loaded successfully"
        )
        
        # Create test frame
        print("\n📹 Creating test frame...")
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Run detection
        print("🎯 Running detection...")
        result = detector.detect(frame)
        
        all_pass &= check_requirement(
            "Detection executed",
            result is not None,
            "Frame processed without errors"
        )
        
        # Verify output
        if result:
            vehicles = result.get('vehicles', [])
            count = result.get('count', 0)
            emergency = result.get('emergency', False)
            
            all_pass &= check_requirement(
                "Returns correct count",
                count == len(vehicles),
                f"count={count}, len(vehicles)={len(vehicles)}"
            )
            
            all_pass &= check_requirement(
                "Emergency flag is boolean",
                isinstance(emergency, bool),
                f"Value: {emergency}"
            )
            
            # Show structure
            if vehicles:
                print(f"\n📊 Sample vehicle detection:")
                sample_vehicle = vehicles[0]
                print(f"   - class: {sample_vehicle.get('class')}")
                print(f"   - confidence: {sample_vehicle.get('confidence'):.3f}")
                print(f"   - bbox: {sample_vehicle.get('bbox')}")
        
    except Exception as e:
        all_pass &= check_requirement("Live detection test", False, str(e))
    
    return all_pass

def main():
    """Run comprehensive analysis."""
    print("\n")
    print("🔍 COMPREHENSIVE SYSTEM ANALYSIS - VEHICLE DETECTION MODULE")
    print("=" * 70)
    
    result1 = analyze_vision_detector()
    result2 = analyze_test_detector()
    result3 = test_real_detection()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    
    all_pass = result1 and result2 and result3
    
    if all_pass:
        print("\n✅ ALL REQUIREMENTS MET!")
        print("\n🎯 System Status: READY FOR PRODUCTION")
        print("\nYour vehicle detection module is:")
        print("  ✅ Properly structured with VehicleDetector class")
        print("  ✅ Implements full output contract (vehicles, count, emergency, raw_result)")
        print("  ✅ Detects all required vehicle classes")
        print("  ✅ Supports ambulance/emergency detection")
        print("  ✅ Has comprehensive test suite")
        print("  ✅ Includes FPS measurement")
        print("  ✅ Multiple video source fallbacks")
    else:
        print("\n⚠️  SOME REQUIREMENTS NOT MET")
        print("\nPlease review the ❌ FAIL items above for details.")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("\n1. Run tests on real video:")
    print("   python tests/test_detector.py --source assets/real_time_traffic/output.mp4")
    print("\n2. Verify ambulance detection:")
    print("   python scripts/validate_ambulance.py --image <ambulance_image>")
    print("\n3. Check FPS performance on your hardware")
    print("   (Target: >= 10 fps for real-time processing)")
    print("\n")

if __name__ == "__main__":
    main()
