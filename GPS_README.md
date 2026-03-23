# GPS-Based Ambulance Detection Extension

This extension adds GPS-based ambulance detection to your existing YOLO-based traffic AI system, allowing emergency detection **before** ambulances are visible in camera footage.

## 🚑 How It Works

The system now combines two detection methods:
- **Vision-based**: Detects ambulances visible in camera frames (existing functionality)
- **GPS-based**: Detects ambulances approaching within 1km of camera location

Final emergency status = `vision_emergency OR gps_emergency`

## 📁 New Files Added

### Core Components
- `gps_server.py` - FastAPI server for GPS location management
- `gps_utils.py` - Distance calculations and emergency logic
- `simulate_ambulance.py` - GPS simulation for testing

### Integration
- Modified `vision/detector.py` - Added GPS emergency checking
- Updated `config.py` - Added GPS configuration constants
- Updated `requirements.txt` - Added FastAPI, requests dependencies

### Testing
- `test_gps_integration.py` - Integration test script

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start GPS Server
```bash
uvicorn gps_server:app --reload
```

### 3. Run Ambulance Simulation (in another terminal)
```bash
python simulate_ambulance.py
```

### 4. Test Detection
```bash
python tests/test_detector.py --source assets/sample_video.mp4
```

## 📡 API Endpoints

### POST /update-location
Update ambulance GPS location:
```json
{
  "vehicle_id": "AMB123",
  "lat": 26.9124,
  "lon": 75.7873,
  "speed": 45.0
}
```

### GET /ambulances
Get all active ambulance locations:
```json
[
  {
    "vehicle_id": "AMB123",
    "lat": 26.9124,
    "lon": 75.7873,
    "speed": 45.0,
    "timestamp": "2024-01-01T12:00:00"
  }
]
```

## ⚙️ Configuration

Update these constants in `config.py`:
```python
CAMERA_LAT = 26.9124          # Your camera latitude
CAMERA_LON = 75.7873          # Your camera longitude
GPS_SERVER_URL = "http://localhost:8000"  # GPS server endpoint
EMERGENCY_DISTANCE_KM = 1.0   # Emergency trigger distance
```

## 🔧 Integration Details

### Modified Output Format
The `VehicleDetector.detect()` method now returns:
```python
{
  "vehicles": [...],           # Detected vehicles (unchanged)
  "count": 5,                  # Vehicle count (unchanged)
  "emergency": true,           # Final emergency status (vision OR gps)
  "gps_emergency": false,      # GPS-based emergency status
  "raw_result": ...,           # YOLO results (unchanged)
}
```

### Backward Compatibility
- Existing code continues to work unchanged
- GPS features are additive - no breaking changes
- If GPS server is unavailable, system gracefully falls back to vision-only detection

## 🧪 Testing

### Unit Tests
```bash
# Test GPS utilities
python gps_utils.py

# Test integration
python test_gps_integration.py
```

### Manual Testing
1. Start GPS server: `uvicorn gps_server:app --reload`
2. Run simulation: `python simulate_ambulance.py`
3. Monitor server logs for GPS updates
4. Check `/ambulances` endpoint for active locations

### Real-world Testing
```bash
# Test with video file
python tests/test_detector.py --source assets/real_time_traffic/output.mp4

# Test with camera (if available)
python tests/test_detector.py --source 0
```

## 📊 Distance Calculation

Uses Haversine formula for accurate GPS distance calculation:
```python
from gps_utils import calculate_distance

distance = calculate_distance(lat1, lon1, lat2, lon2)  # Returns km
```

## 🚨 Emergency Logic

Emergency is triggered when:
- Any ambulance is within `EMERGENCY_DISTANCE_KM` (default: 1.0 km) of camera
- Vision detects ambulance in frame

## 🔍 Debug Output

The system provides detailed logging:
```
GPS ambulance detected: AMB001 at 0.8 km → emergency triggered
GPS ambulance: AMB002 at 1.2 km (not emergency)
GPS emergency check failed: Connection refused  # Server offline
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   Ambulance     │───▶│   GPS Server    │
│   GPS Device    │    │  (FastAPI)      │
└─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐
│ VehicleDetector │◀───│   GPS Utils     │
│  (YOLO-based)   │    │ (Distance calc) │
└─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐
│   Emergency     │
│   Decision      │
└─────────────────┘
```

## 🚦 Production Deployment

### Server Deployment
```bash
# Production server (no reload)
uvicorn gps_server:app --host 0.0.0.0 --port 8000

# With SSL
uvicorn gps_server:app --ssl-keyfile key.pem --ssl-certfile cert.pem
```

### Monitoring
- GPS server health: `GET /health`
- Active ambulances: `GET /ambulances`
- Server logs show all GPS updates and emergency triggers

### Error Handling
- Network failures don't break vision detection
- Invalid GPS data is logged and skipped
- Server timeouts are handled gracefully (2-second timeout)

## 🔒 Security Considerations

- GPS server should be deployed behind firewall/VPN
- Consider authentication for production use
- Validate GPS coordinates (lat/lon ranges)
- Rate limiting for GPS update endpoints

## 📈 Performance

- GPS check adds ~10-50ms latency (network dependent)
- Minimal impact on vision processing pipeline
- Ambulance locations cached for 5 minutes
- Automatic cleanup of stale GPS data

## 🐛 Troubleshooting

### GPS Server Not Responding
```bash
# Check if server is running
curl http://localhost:8000/health

# Start server
uvicorn gps_server:app --reload
```

### No GPS Emergency Detection
- Verify ambulance coordinates are within 1km of camera
- Check GPS server connectivity
- Review server logs for error messages

### Integration Issues
```bash
# Test integration
python test_gps_integration.py

# Check detector output format
python -c "from vision.detector import VehicleDetector; d=VehicleDetector(); print(d.detect(np.zeros((480,640,3),dtype=np.uint8)).keys())"
```

## 🎯 Next Steps

1. **Real GPS Integration**: Connect to actual ambulance GPS devices
2. **Multiple Cameras**: Support multiple camera locations
3. **Route Prediction**: Predict ambulance arrival times
4. **Priority System**: Different emergency distances for different vehicle types
5. **Historical Tracking**: Store ambulance movement history

---

**Note**: This extension maintains full backward compatibility with your existing YOLO-based detection system while adding powerful GPS-based early warning capabilities.</content>
<parameter name="filePath">/Users/soumaydhrub/~/Library/Application Support/Cursor/traffic-ai/GPS_README.md