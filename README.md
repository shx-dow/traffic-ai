# AI Traffic Flow Optimizer & Emergency Green Corridor System

## Project Overview

This prototype demonstrates a real-time traffic management workflow built around
computer vision and simple decision logic. A video stream is analyzed to detect
vehicles, count traffic by lane, estimate density, and choose a signal state.
When an emergency is simulated or detected, the system can override the normal
flow and activate an all-green corridor for a short time.

## Features

- Real-time vehicle detection with YOLOv8
- Lane-aware vehicle counting and simple density estimation
- Dynamic traffic signal selection
- Emergency vehicle override for a green corridor demo
- Modular structure for team collaboration

## Architecture

Pipeline:

`Video -> Detection -> Counting -> Decision -> Emergency Override -> Overlay -> Display`

Module responsibilities:

- `main.py`: Coordinates the full runtime loop
- `vision/detector.py`: Runs YOLO inference and returns structured detections
- `logic/counter.py`: Counts vehicles and splits them across lanes
- `logic/signal.py`: Decides which lane gets the green signal
- `logic/emergency.py`: Simulates emergency activation and override behavior
- `ui/overlay.py`: Draws detections, counts, and signal state on the frame
- `utils/helpers.py`: Shared helper functions
- `config.py`: Centralized thresholds, classes, and display settings

## Setup Instructions

1. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the prototype:

   ```bash
   python main.py
   ```

3. Update the sample video path in `config.py` if needed.

## Demo Instructions

- The demo is designed to run on a sample video source.
- If emergency simulation is implemented, the automated system 
  should be working properly (or press the configured key during playback
  to trigger the green corridor override).
- Use the on-screen overlay to verify detection counts and signal decisions.

## Team Structure Suggestion

- Vision owner: `vision/detector.py`
- Traffic logic owner: `logic/counter.py` and `logic/signal.py`
- Emergency flow owner: `logic/emergency.py`
- UI/visualization owner: `ui/overlay.py`
- Integration owner: `main.py`, `config.py`, and launch flow