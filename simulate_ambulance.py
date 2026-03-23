
import time
import requests
from typing import Tuple

from config import GPS_SERVER_URL, CAMERA_LAT, CAMERA_LON

# Starting location (~7 km south of camera)
START_LAT = 26.8500
START_LON = 75.7873

# Movement parameters
SPEED_KMH = 60.0  # Ambulance speed in km/h
UPDATE_INTERVAL_SECONDS = 2.0

# Vehicle ID for simulation
VEHICLE_ID = "AMB_SIM_001"


def calculate_next_position(current_lat: float, current_lon: float,
                          target_lat: float, target_lon: float,
                          speed_kmh: float, time_seconds: float) -> Tuple[float, float]:
    """Calculate next GPS position moving toward target at given speed."""
    from gps_utils import calculate_distance
    import math

    distance_to_target = calculate_distance(current_lat, current_lon, target_lat, target_lon)

    if distance_to_target < 0.01:
        return target_lat, target_lon

    lat1_rad = math.radians(current_lat)
    lat2_rad = math.radians(target_lat)
    dlon_rad = math.radians(target_lon - current_lon)

    y = math.sin(dlon_rad) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)
    bearing_rad = math.atan2(y, x)

    distance_to_move_km = min((speed_kmh / 3600.0) * time_seconds, distance_to_target)

    earth_radius_km = 6371.0
    lat1_rad = math.radians(current_lat)
    lon1_rad = math.radians(current_lon)

    new_lat_rad = math.asin(
        math.sin(lat1_rad) * math.cos(distance_to_move_km / earth_radius_km) +
        math.cos(lat1_rad) * math.sin(distance_to_move_km / earth_radius_km) * math.cos(bearing_rad)
    )
    new_lon_rad = lon1_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(distance_to_move_km / earth_radius_km) * math.cos(lat1_rad),
        math.cos(distance_to_move_km / earth_radius_km) - math.sin(lat1_rad) * math.sin(new_lat_rad)
    )

    return math.degrees(new_lat_rad), math.degrees(new_lon_rad)


def send_gps_update(vehicle_id: str, lat: float, lon: float, speed: float) -> bool:
    """Send GPS location update to the server. Returns True if successful."""
    try:
        payload = {"vehicle_id": vehicle_id, "lat": lat, "lon": lon, "speed": speed}
        response = requests.post(f"{GPS_SERVER_URL}/update-location", json=payload, timeout=5)
        response.raise_for_status()
        print(f"✓ GPS Update sent: {vehicle_id} @ ({lat:.4f}, {lon:.4f}) - {speed:.1f} km/h")
        return True
    except requests.exceptions.RequestException as e:
        print(f"✗ Failed to send GPS update: {e}")
        return False


def main():
    """Main simulation loop."""
    print("🚑 Starting Ambulance GPS Simulation")
    print(f"Vehicle ID: {VEHICLE_ID}")
    print(f"Starting position: ({START_LAT:.4f}, {START_LON:.4f})")
    print(f"Target position: ({CAMERA_LAT:.4f}, {CAMERA_LON:.4f})")
    print(f"Speed: {SPEED_KMH} km/h")
    print(f"Update interval: {UPDATE_INTERVAL_SECONDS} seconds")
    print("-" * 50)

    current_lat = START_LAT
    current_lon = START_LON

    step = 0

    try:
        while True:
            step += 1

            from gps_utils import calculate_distance
            distance_to_camera = calculate_distance(current_lat, current_lon, CAMERA_LAT, CAMERA_LON)
            print(f"Step {step}: Distance to camera: {distance_to_camera:.2f} km")

            success = send_gps_update(VEHICLE_ID, current_lat, current_lon, SPEED_KMH)
            if not success:
                print("Failed to send GPS update, retrying in next interval...")

            if distance_to_camera < 0.01:
                print("🎯 Ambulance has reached the camera location!")
                break

            current_lat, current_lon = calculate_next_position(
                current_lat, current_lon,
                CAMERA_LAT, CAMERA_LON,
                SPEED_KMH, UPDATE_INTERVAL_SECONDS
            )

            time.sleep(UPDATE_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped by user")

    print("Simulation ended.")


if __name__ == "__main__":
    main()
