from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="GPS Ambulance Detection Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# vehicle_id → {lat, lon, speed, timestamp}
ambulance_locations: Dict[str, Dict] = {}


class AmbulanceLocation(BaseModel):
    vehicle_id: str
    lat: float
    lon: float
    speed: float


def _active_ambulances() -> Dict[str, Dict]:
    """Return ambulances updated within the last 5 minutes, pruning stale entries."""
    cutoff = datetime.now() - timedelta(minutes=5)
    active = {vid: d for vid, d in ambulance_locations.items() if d["timestamp"] > cutoff}
    ambulance_locations.clear()
    ambulance_locations.update(active)
    return active


@app.post("/update-location")
async def update_location(location: AmbulanceLocation) -> Dict[str, str]:
    ambulance_locations[location.vehicle_id] = {
        "lat": location.lat,
        "lon": location.lon,
        "speed": location.speed,
        "timestamp": datetime.now(),
    }
    print(f"Updated: {location.vehicle_id} ({location.lat:.4f}, {location.lon:.4f}) @ {location.speed:.1f} km/h")
    return {"status": "success", "message": f"Location updated for {location.vehicle_id}"}


@app.get("/ambulances")
async def get_ambulances() -> List[Dict]:
    active = _active_ambulances()
    return [
        {"vehicle_id": vid, "lat": d["lat"], "lon": d["lon"],
         "speed": d["speed"], "timestamp": d["timestamp"].isoformat()}
        for vid, d in active.items()
    ]


@app.get("/check-ambulance")
async def check_ambulance() -> Dict[str, object]:
    try:
        from config import EMERGENCY_DISTANCE_KM, EMERGENCY_ETA_SECONDS
        from gps_utils import CAMERA_LAT, CAMERA_LON, calculate_distance, estimate_eta_seconds

        active = _active_ambulances()
        emergency = False
        best = {
            "vehicle_id": None,
            "distance_km": None,
            "eta_seconds": None,
            "speed_kmh": None,
        }
        best_eta = float("inf")
        for vid, data in active.items():
            distance = calculate_distance(CAMERA_LAT, CAMERA_LON, data["lat"], data["lon"])
            speed = float(data.get("speed", 0.0))
            eta = estimate_eta_seconds(distance, speed)
            should_trigger = (distance < EMERGENCY_DISTANCE_KM) or (eta <= EMERGENCY_ETA_SECONDS)
            if should_trigger:
                print(f"EMERGENCY: {vid} at {distance:.2f} km, eta={eta:.1f}s")
                emergency = True
                if eta < best_eta:
                    best_eta = eta
                    best = {
                        "vehicle_id": vid,
                        "distance_km": distance,
                        "eta_seconds": eta,
                        "speed_kmh": speed,
                    }
            else:
                print(f"Monitoring: {vid} at {distance:.2f} km, eta={eta:.1f}s")
        return {"emergency": emergency, **best}
    except Exception as e:
        print(f"Error in check_ambulance: {e}")
        return {"emergency": False}


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
