"""
GPS-based ambulance detection server using FastAPI.

This server maintains real-time ambulance locations and provides endpoints
for updating locations and retrieving active ambulances.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="GPS Ambulance Detection Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for ambulance locations
# Key: vehicle_id, Value: dict with location data and timestamp
ambulance_locations: Dict[str, Dict] = {}


class AmbulanceLocation(BaseModel):
    """Model for ambulance location update."""
    vehicle_id: str
    lat: float
    lon: float
    speed: float


class AmbulanceData(BaseModel):
    """Model for ambulance data response."""
    vehicle_id: str
    lat: float
    lon: float
    speed: float
    timestamp: datetime


@app.post("/update-location")
async def update_location(location: AmbulanceLocation) -> Dict[str, str]:
    """
    Update the location of an ambulance.

    Args:
        location: Ambulance location data

    Returns:
        Success message
    """
    try:
        ambulance_locations[location.vehicle_id] = {
            "lat": location.lat,
            "lon": location.lon,
            "speed": location.speed,
            "timestamp": datetime.now()
        }
        print(f"Updated location for ambulance {location.vehicle_id}: "
              f"({location.lat:.4f}, {location.lon:.4f}) @ {location.speed:.1f} km/h")
        return {"status": "success", "message": f"Location updated for {location.vehicle_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update location: {str(e)}")


@app.get("/ambulances")
async def get_ambulances() -> List[Dict]:
    """
    Get all active ambulance locations.

    Returns:
        List of active ambulance data
    """
    try:
        # Clean up old locations (older than 5 minutes)
        cutoff_time = datetime.now() - timedelta(minutes=5)
        active_ambulances = {
            vehicle_id: data
            for vehicle_id, data in ambulance_locations.items()
            if data["timestamp"] > cutoff_time
        }

        # Update storage with only active ambulances
        ambulance_locations.clear()
        ambulance_locations.update(active_ambulances)

        result = [
            {
                "vehicle_id": vehicle_id,
                "lat": data["lat"],
                "lon": data["lon"],
                "speed": data["speed"],
                "timestamp": data["timestamp"].isoformat()
            }
            for vehicle_id, data in active_ambulances.items()
        ]

        print(f"Returning {len(result)} active ambulance locations")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve ambulances: {str(e)}")


@app.get("/check-ambulance")
async def check_ambulance() -> Dict[str, bool]:
    """
    Check if any ambulance is within emergency distance of the camera.

    Returns:
        Dictionary with emergency status
    """
    try:
        # Clean up old locations (older than 5 minutes)
        cutoff_time = datetime.now() - timedelta(minutes=5)
        active_ambulances = {
            vehicle_id: data
            for vehicle_id, data in ambulance_locations.items()
            if data["timestamp"] > cutoff_time
        }

        # Update storage with only active ambulances
        ambulance_locations.clear()
        ambulance_locations.update(active_ambulances)

        # Check for emergency (ambulance within 0.5 km)
        emergency = False
        for vehicle_id, data in active_ambulances.items():
            from gps_utils import calculate_distance, CAMERA_LAT, CAMERA_LON

            distance = calculate_distance(CAMERA_LAT, CAMERA_LON, data["lat"], data["lon"])

            if distance < 0.5:  # 0.5 km emergency threshold
                print(f"🚨 GPS EMERGENCY: Ambulance {vehicle_id} at {distance:.2f} km from camera!")
                emergency = True
            else:
                print(f"📍 Ambulance {vehicle_id} at {distance:.2f} km from camera (monitoring)")

        return {"emergency": emergency}

    except Exception as e:
        print(f"Error checking ambulance emergency: {e}")
        return {"emergency": False}


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    print("Starting GPS Ambulance Detection Server...")
    print("Endpoints:")
    print("  POST /update-location - Update ambulance location")
    print("  GET /ambulances - Get all active ambulances")
    print("  GET /check-ambulance - Check for emergency ambulances")
    print("  GET /health - Health check")
    uvicorn.run(app, host="0.0.0.0", port=8000)