
import time
import requests
from typing import Tuple

# GPS server configuration
GPS_SERVER_URL = "http://localhost:8000"

# Starting location (some distance from camera)
START_LAT = 26.8500  # ~7 km south of camera
START_LON = 75.7873

# Camera location (target)
from gps_utils import CAMERA_LAT, CAMERA_LON

# Movement parameters
SPEED_KMH = 60.0  # Ambulance speed in km/h
UPDATE_INTERVAL_SECONDS = 2.0

# Vehicle ID for simulation
VEHICLE_ID = "AMB_SIM_001"


def calculate_next_position(current_lat: float, current_lon: float,
                          target_lat: float, target_lon: float,
                          speed_kmh: float, time_seconds: float) -> Tuple[float, float]:
    """
    Calculate next GPS position based on current position, target, speed, and time.

    Args:
        current_lat: Current latitude
        current_lon: Current longitude
        target_lat: Target latitude
        target_lon: Target longitude
        speed_kmh: Speed in km/h
        time_seconds: Time interval in seconds

    Returns:
        Tuple of (new_lat, new_lon)
    """
    from gps_utils import calculate_distance
    import math

    # Calculate distance to target
    distance_to_target = calculate_distance(current_lat, current_lon, target_lat, target_lon)

    if distance_to_target < 0.01:  # Very close to target
        return target_lat, target_lon

    # Calculate bearing (direction) to target
    lat1_rad = math.radians(current_lat)
    lat2_rad = math.radians(target_lat)
    dlon_rad = math.radians(target_lon - current_lon)

    y = math.sin(dlon_rad) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)

    bearing_rad = math.atan2(y, x)

    # Calculate distance to move in this time interval
    distance_to_move_km = (speed_kmh / 3600.0) * time_seconds  # Convert speed to km per second

    # Don't overshoot the target
    distance_to_move_km = min(distance_to_move_km, distance_to_target)

    # Calculate new position
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

    new_lat = math.degrees(new_lat_rad)
    new_lon = math.degrees(new_lon_rad)

    return new_lat, new_lon


def send_gps_update(vehicle_id: str, lat: float, lon: float, speed: float) -> bool:
    """
    Send GPS location update to the server.

    Args:
        vehicle_id: Vehicle identifier
        lat: Latitude
        lon: Longitude
        speed: Speed in km/h

    Returns:
        True if successful, False otherwise
    """
    try:
        payload = {
            "vehicle_id": vehicle_id,
            "lat": lat,
            "lon": lon,
            "speed": speed
        }

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

            # Calculate distance to camera
            from gps_utils import calculate_distance
            distance_to_camera = calculate_distance(current_lat, current_lon, CAMERA_LAT, CAMERA_LON)

            print(f"Step {step}: Distance to camera: {distance_to_camera:.2f} km")

            # Send GPS update
            success = send_gps_update(VEHICLE_ID, current_lat, current_lon, SPEED_KMH)

            if not success:
                print("Failed to send GPS update, retrying in next interval...")

            # Check if we've reached the destination
            if distance_to_camera < 0.01:  # Within 10 meters
                print("🎯 Ambulance has reached the camera location!")
                break

            # Calculate next position
            current_lat, current_lon = calculate_next_position(
                current_lat, current_lon,
                CAMERA_LAT, CAMERA_LON,
                SPEED_KMH, UPDATE_INTERVAL_SECONDS
            )

            # Wait before next update
            time.sleep(UPDATE_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped by user")

    print("Simulation ended.")


if __name__ == "__main__":
    main()
