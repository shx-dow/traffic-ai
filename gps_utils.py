
from __future__ import annotations

import math
from typing import Dict, List


# Camera location constants (Jaipur, India - can be configured)
CAMERA_LAT = 26.9124
CAMERA_LON = 75.7873

# Emergency distance threshold in kilometers
EMERGENCY_DISTANCE_KM = 1.0


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth using Haversine formula.

    Args:
        lat1: Latitude of first point in degrees
        lon1: Longitude of first point in degrees
        lat2: Latitude of second point in degrees
        lon2: Longitude of second point in degrees

    Returns:
        Distance in kilometers
    """
    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    # Earth's radius in kilometers
    earth_radius_km = 6371.0

    distance = earth_radius_km * c
    return distance


def check_gps_emergency(ambulance_data: List[Dict]) -> bool:
    """
    Check if any ambulance is within emergency distance of the camera.

    Args:
        ambulance_data: List of ambulance location dictionaries from GPS server

    Returns:
        True if any ambulance is within emergency distance, False otherwise
    """
    for ambulance in ambulance_data:
        try:
            lat = float(ambulance["lat"])
            lon = float(ambulance["lon"])
            vehicle_id = ambulance.get("vehicle_id", "unknown")

            distance = calculate_distance(CAMERA_LAT, CAMERA_LON, lat, lon)

            if distance < EMERGENCY_DISTANCE_KM:
                print(f"GPS ambulance detected: {vehicle_id} at {distance:.2f} km → emergency triggered")
                return True
            else:
                print(f"GPS ambulance: {vehicle_id} at {distance:.2f} km (not emergency)")

        except (KeyError, ValueError, TypeError) as e:
            print(f"Error processing ambulance data {ambulance}: {e}")
            continue

    return False


def get_camera_location() -> Dict[str, float]:
    """
    Get the camera location coordinates.

    Returns:
        Dictionary with lat and lon keys
    """
    return {
        "lat": CAMERA_LAT,
        "lon": CAMERA_LON
    }


# Example usage and testing
if __name__ == "__main__":
    # Test distance calculation
    # Jaipur to Delhi (approximately 280 km)
    jaipur_lat, jaipur_lon = 26.9124, 75.7873
    delhi_lat, delhi_lon = 28.7041, 77.1025

    distance = calculate_distance(jaipur_lat, jaipur_lon, delhi_lat, delhi_lon)
    print(f"Distance from Jaipur to Delhi: {distance:.2f} km")

    # Test emergency check with sample data
    test_ambulances = [
        {"vehicle_id": "AMB001", "lat": 26.9124, "lon": 75.7873},  # At camera
        {"vehicle_id": "AMB002", "lat": 26.9200, "lon": 75.7900},  # ~1.2 km away
        {"vehicle_id": "AMB003", "lat": 26.9050, "lon": 75.7800},  # ~0.8 km away (emergency)
    ]

    emergency = check_gps_emergency(test_ambulances)
    print(f"Emergency status: {emergency}")
