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


def estimate_eta_seconds(distance_km: float, speed_kmh: float, *, min_speed_kmh: float = 5.0) -> float:
    d = max(0.0, float(distance_km))
    s = max(float(min_speed_kmh), float(speed_kmh))
    return (d / s) * 3600.0


def compute_emergency_priority(ambulance_data: List[Dict]) -> Dict[str, float | bool | str | None]:
    best = {
        "emergency": False,
        "vehicle_id": None,
        "distance_km": None,
        "eta_seconds": None,
        "speed_kmh": None,
    }

    best_eta = float("inf")
    for ambulance in ambulance_data:
        try:
            lat = float(ambulance["lat"])
            lon = float(ambulance["lon"])
            speed = float(ambulance.get("speed", 0.0))
            vid = str(ambulance.get("vehicle_id", "unknown"))

            distance = calculate_distance(CAMERA_LAT, CAMERA_LON, lat, lon)
            eta = estimate_eta_seconds(distance, speed)

            if distance <= EMERGENCY_DISTANCE_KM and eta < best_eta:
                best_eta = eta
                best = {
                    "emergency": True,
                    "vehicle_id": vid,
                    "distance_km": distance,
                    "eta_seconds": eta,
                    "speed_kmh": speed,
                }
        except (KeyError, ValueError, TypeError):
            continue

    return best


def get_camera_location() -> Dict[str, float]:
    return {"lat": CAMERA_LAT, "lon": CAMERA_LON}


if __name__ == "__main__":
    d = calculate_distance(26.9124, 75.7873, 28.7041, 77.1025)
    print(f"Jaipur to Delhi: {d:.2f} km")
