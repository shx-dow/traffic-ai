"""
Test script for GPS-based ambulance detection integration.

This script tests the complete integration between:
1. GPS server API
2. VehicleDetector GPS emergency checking
3. Emergency detection logic
"""

import numpy as np
import requests
import time
from vision.detector import VehicleDetector

def test_gps_server():
    """Test GPS server endpoints."""
    print("🧪 Testing GPS Server...")

    GPS_SERVER_URL = "http://localhost:8000"

    try:
        # Test health endpoint
        response = requests.get(f"{GPS_SERVER_URL}/health")
        response.raise_for_status()
        print("✅ GPS server is healthy")

        # Test check-ambulance endpoint (should return no emergency initially)
        response = requests.get(f"{GPS_SERVER_URL}/check-ambulance")
        response.raise_for_status()
        result = response.json()
        assert result["emergency"] == False
        print("✅ Check-ambulance endpoint working (no emergency)")

        # Send test ambulance location (close to camera for emergency)
        test_location = {
            "vehicle_id": "AMB_TEST_001",
            "lat": 26.9124,  # Very close to camera (26.9124, 75.7873)
            "lon": 75.7873,
            "speed": 45.0
        }

        response = requests.post(f"{GPS_SERVER_URL}/update-location", json=test_location)
        response.raise_for_status()
        print("✅ Ambulance location updated")

        # Wait a moment for processing
        time.sleep(0.5)

        # Test check-ambulance endpoint (should now return emergency)
        response = requests.get(f"{GPS_SERVER_URL}/check-ambulance")
        response.raise_for_status()
        result = response.json()
        assert result["emergency"] == True
        print("✅ Emergency detected correctly!")

        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ GPS server test failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_vehicle_detector_integration():
    """Test VehicleDetector GPS integration."""
    print("\n🧪 Testing VehicleDetector GPS Integration...")

    try:
        # Create a mock frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Initialize detector
        detector = VehicleDetector()

        # Test detection (GPS server should be running)
        result = detector.detect(frame)

        # Check output format
        required_keys = {"vehicles", "count", "emergency"}
        assert all(key in result for key in required_keys), f"Missing keys: {required_keys - set(result.keys())}"

        print("✅ VehicleDetector output format correct")
        print(f"   Emergency status: {result['emergency']}")
        print(f"   Vehicle count: {result['count']}")

        return True

    except Exception as e:
        print(f"❌ VehicleDetector integration test failed: {e}")
        return False

def test_distance_calculation():
    """Test distance calculation utility."""
    print("\n🧪 Testing Distance Calculation...")

    try:
        from gps_utils import calculate_distance, CAMERA_LAT, CAMERA_LON

        # Test distance from camera to itself (should be 0)
        distance = calculate_distance(CAMERA_LAT, CAMERA_LON, CAMERA_LAT, CAMERA_LON)
        assert distance < 0.001, f"Distance to self should be ~0, got {distance}"

        print(f"Distance (self): {distance:.3f} km")

        # Test distance to a point ~1km away
        test_lat, test_lon = 26.9214, 75.7873  # Approximately 1km north
        distance = calculate_distance(CAMERA_LAT, CAMERA_LON, test_lat, test_lon)
        assert 0.8 <= distance <= 1.2, f"Distance should be ~1km, got {distance}"

        print(f"Distance (~1km): {distance:.3f} km")
        return True

    except Exception as e:
        print(f"❌ Distance calculation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚑 GPS Ambulance Detection - Integration Tests")
    print("=" * 50)

    # Test components
    tests = [
        test_distance_calculation,
        test_gps_server,
        test_vehicle_detector_integration,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! GPS integration is working correctly.")
        print("\n📱 To test with mobile GPS:")
        print("   1. Open mobile_gps_tracker.html in your mobile browser")
        print("   2. Allow location access")
        print("   3. Click 'Start Tracking'")
        print("   4. Run: python tests/test_detector.py --source your_video.mp4")
        print("\n🔧 To test with simulation:")
        print("   1. Run: python simulate_ambulance.py")
        print("   2. Run: python tests/test_detector.py --source your_video.mp4")
    else:
        print("❌ Some tests failed. Check GPS server is running:")
        print("   uvicorn gps_server:app --reload")

if __name__ == "__main__":
    main()