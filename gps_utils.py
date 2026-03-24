from __future__ import annotations

import math
from typing import Dict, List

from config import CAMERA_LAT, CAMERA_LON, EMERGENCY_DISTANCE_KM


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in km between two lat/lon points (Haversine)."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def check_gps_emergency(ambulance_data: List[Dict]) -> bool:
    """Return True if any ambulance in the list is within EMERGENCY_DISTANCE_KM of the camera."""
    for ambulance in ambulance_data:
        try:
            lat = float(ambulance["lat"])
            lon = float(ambulance["lon"])
            vid = ambulance.get("vehicle_id", "unknown")
            distance = calculate_distance(CAMERA_LAT, CAMERA_LON, lat, lon)
            if distance < EMERGENCY_DISTANCE_KM:
                print(f"GPS emergency: {vid} at {distance:.2f} km")
                return True
        except (KeyError, ValueError, TypeError) as e:
            print(f"Error processing ambulance data {ambulance}: {e}")
    return False


def get_camera_location() -> Dict[str, float]:
    return {"lat": CAMERA_LAT, "lon": CAMERA_LON}


if __name__ == "__main__":
    d = calculate_distance(26.9124, 75.7873, 28.7041, 77.1025)
    print(f"Jaipur to Delhi: {d:.2f} km")
