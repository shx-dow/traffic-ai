from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import requests

from vision.detector import VehicleDetector
from gps_utils import calculate_distance, CAMERA_LAT, CAMERA_LON


def test_gps_server():
    """Test GPS server endpoints."""
    print("🧪 Testing GPS Server...")

    gps_server_url = "http://localhost:8000"

    try:
        response = requests.get(f"{gps_server_url}/health", timeout=5)
        response.raise_for_status()
        print("✅ GPS server is healthy")

        response = requests.get(f"{gps_server_url}/check-ambulance", timeout=5)
        response.raise_for_status()
        result = response.json()
        assert result["emergency"] is False
        print("✅ Check-ambulance endpoint working (no emergency)")

        test_location = {
            "vehicle_id": "AMB_TEST_001",
            "lat": 26.9124,
            "lon": 75.7873,
            "speed": 45.0,
        }

        response = requests.post(f"{gps_server_url}/update-location", json=test_location, timeout=5)
        response.raise_for_status()
        print("✅ Ambulance location updated")

        time.sleep(0.5)

        response = requests.get(f"{gps_server_url}/check-ambulance", timeout=5)
        response.raise_for_status()
        result = response.json()
        assert result["emergency"] is True
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
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detector = VehicleDetector()
        result = detector.detect(frame)

        required_keys = {"vehicles", "count", "emergency"}
        assert all(key in result for key in required_keys), \
            f"Missing keys: {required_keys - set(result.keys())}"

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
        distance = calculate_distance(CAMERA_LAT, CAMERA_LON, CAMERA_LAT, CAMERA_LON)
        assert distance < 0.001, f"Distance to self should be ~0, got {distance}"
        print(f"Distance (self): {distance:.3f} km")

        test_lat, test_lon = 26.9214, 75.7873
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

    tests = [
        test_distance_calculation,
        test_gps_server,
        test_vehicle_detector_integration,
    ]

    passed = sum(1 for t in tests if t())

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All tests passed!")
        print("\n📱 To test with mobile GPS:")
        print("   1. Open mobile_gps_tracker.html in your mobile browser")
        print("   2. Allow location access and click 'Start Tracking'")
        print("\n🔧 To test with simulation:")
        print("   python simulate_ambulance.py")
    else:
        print("❌ Some tests failed. Ensure GPS server is running:")
        print("   uvicorn gps_server:app --reload")


if __name__ == "__main__":
    main()
