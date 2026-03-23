# 🚑 Real-Time GPS Ambulance Detection System

## Overview
Your traffic AI system now supports **real-time GPS-based ambulance detection** that works alongside your existing YOLO vision pipeline. Ambulances can be detected **before they appear in camera view** when within 0.5km of the camera location.

## 🏗️ System Architecture

```
Mobile App/Browser → GPS Server → VehicleDetector → Emergency Signal
       ↓                    ↓              ↓
   Real-time GPS       FastAPI REST     YOLO + GPS
   Updates (2s)        Endpoints        Integration
```

## 📡 API Endpoints

### POST /update-location
**Input:**
```json
{
  "vehicle_id": "AMB_001",
  "lat": 26.9124,
  "lon": 75.7873,
  "speed": 45.0
}
```

### GET /check-ambulance
**Output:**
```json
{
  "emergency": true
}
```

## 🔧 Integration with VehicleDetector

The `detect()` method now returns:
```python
{
  "vehicles": [...],     # YOLO detections
  "count": int,          # Vehicle count
  "emergency": bool      # vision_emergency OR gps_emergency
}
```

**Emergency Logic:**
- `emergency = True` if ambulance visible in camera OR ambulance GPS within 0.5km

## 🧪 Testing Guide

### Method 1: Browser GPS Testing
1. **Start GPS Server:**
   ```bash
   uvicorn gps_server:app --reload
   ```

2. **Open Mobile GPS Tracker:**
   - Open `mobile_gps_tracker.html` in your mobile browser
   - Allow location access
   - Click "Start Tracking"
   - GPS data sends every 2 seconds

3. **Test Detection:**
   ```bash
   python tests/test_detector.py --source assets/sample_video.mp4
   ```

### Method 2: Simulation Testing
1. **Start GPS Server:**
   ```bash
   uvicorn gps_server:app --reload
   ```

2. **Run Ambulance Simulation:**
   ```bash
   python simulate_ambulance.py
   ```
   - Simulates ambulance moving toward camera
   - Updates GPS every 2 seconds

3. **Test Detection:**
   ```bash
   python tests/test_detector.py --source assets/sample_video.mp4
   ```

### Method 3: API Testing
1. **Start GPS Server:**
   ```bash
   uvicorn gps_server:app --reload
   ```

2. **Send Test GPS Data:**
   ```bash
   curl -X POST http://localhost:8000/update-location \
     -H "Content-Type: application/json" \
     -d '{"vehicle_id": "AMB_TEST", "lat": 26.9124, "lon": 75.7873, "speed": 45.0}'
   ```

3. **Check Emergency Status:**
   ```bash
   curl http://localhost:8000/check-ambulance
   # Should return: {"emergency": true}
   ```

4. **Run Integration Test:**
   ```bash
   python test_gps_integration.py
   ```

## 📊 Expected Behavior

### When Ambulance is Far (>0.5km):
```
📍 Ambulance AMB_001 at 1.2 km from camera (monitoring)
Emergency status: False
```

### When Ambulance is Close (<0.5km):
```
🚨 GPS EMERGENCY: Ambulance AMB_001 at 0.3 km from camera!
Emergency status: True
```

### When Ambulance is Visible in Camera:
```
Custom ambulance model: added 1 ambulance detection(s)
Emergency status: True
```

## 📱 Mobile GPS Tracker Features

- **Real-time GPS tracking** using `navigator.geolocation`
- **Automatic data sending** every 2 seconds
- **Emergency alerts** when other ambulances are detected nearby
- **Visual status indicators** and location display
- **Vibration alerts** on mobile devices
- **Background operation** when app is minimized

## ⚙️ Configuration

**Camera Location** (config.py):
```python
CAMERA_LAT = 26.9124  # Jaipur, India
CAMERA_LON = 75.7873
EMERGENCY_DISTANCE_KM = 0.5  # Emergency threshold
```

**Server Configuration:**
```python
GPS_SERVER_URL = "http://localhost:8000"
```

## 🔍 Debug Information

The system provides detailed logging:
- GPS update confirmations
- Distance calculations
- Emergency triggers
- API call failures (graceful fallback)

## 🚦 Production Deployment

1. **Change server URL** in mobile tracker for remote access
2. **Add authentication** to GPS endpoints
3. **Enable HTTPS** for secure data transmission
4. **Configure firewall** to allow GPS server access
5. **Monitor server logs** for GPS activity

## 🛠️ Troubleshooting

### GPS Server Not Starting
```bash
# Check Python environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn gps_server:app --reload --host 0.0.0.0 --port 8000
```

### Mobile GPS Not Working
- Ensure HTTPS for geolocation API
- Check browser permissions
- Verify server accessibility from mobile network

### Emergency Not Triggering
- Verify ambulance coordinates are within 0.5km
- Check GPS server logs for update confirmations
- Test with `curl` commands first

## 📋 Files Modified/Created

- ✅ `gps_server.py` - FastAPI GPS server with /check-ambulance endpoint
- ✅ `vision/detector.py` - Integrated GPS emergency checking
- ✅ `config.py` - Updated emergency distance to 0.5km
- ✅ `mobile_gps_tracker.html` - Mobile GPS sender with real-time tracking
- ✅ `test_gps_integration.py` - Comprehensive integration tests

## 🎯 Success Criteria

- ✅ GPS server accepts location updates
- ✅ Emergency detection works within 0.5km radius
- ✅ VehicleDetector integrates GPS without breaking YOLO
- ✅ Output format unchanged for existing code
- ✅ Mobile GPS tracker sends real-time data
- ✅ Graceful fallback when GPS server unavailable

**Your traffic AI system now provides early warning for approaching ambulances! 🚑✨**