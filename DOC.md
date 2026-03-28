# Traffic AI Prototype Guide

This document explains the codebase in simple terms so it is easy to present to hackathon judges.

## One-line idea

The system watches traffic in real time, decides which lane should get green based on congestion, and gives priority to emergency vehicles.

## What problem it solves

- normal traffic gets stuck because signals are fixed
- emergency vehicles lose time at red lights
- traffic control should react to live conditions, not a static timer

## What the project does

The codebase supports three main layers:

1. live OpenCV + YOLO prototype
2. emergency priority logic
3. SUMO simulation for pre-system vs post-system comparison

## Simple architecture

### 1) Data acquisition

Files:
- `main.py`
- `vision/detector.py`

What happens:
- video frame enters the system
- YOLO detects vehicles
- the detector also checks emergency state from vision and GPS

### 2) Processing

Files:
- `logic/counter.py`
- `logic/signal.py`
- `logic/emergency.py`
- `logic/runtime.py`
- `logic/signal_state.py`
- `logic/live_metrics.py`

What happens:
- vehicles are counted per camera lane
- queue and approach ROIs estimate congestion
- congestion score decides signal timing
- emergency logic decides corridor lane and source
- live metrics track wait, throughput, and queue

### 3) Execution

Files:
- `ui/overlay.py`
- `main.py`

What happens:
- the chosen signal state is shown on screen
- the overlay shows counts, scores, and emergency status
- the system can save output video and metrics logs

## Current core behavior

### Per-camera first

The live prototype is designed for a real roadside camera.

In per-camera mode:
- one camera represents one approach
- the lane is configured with `--camera-lane`
- `--approach-roi` and `--queue-roi` make counting more realistic

### Score-based switching

The controller no longer uses only vehicle count.

It uses:
- approach count
- queue count
- wait history

This makes the system look smarter and more realistic to judges.

### Emergency handling

Emergency triggers can come from:
- manual key press `e`
- vision detection
- GPS location data
- fusion of vision and GPS

The code explains the source on screen and in logs.

## Important files to mention in a demo

- `main.py` - main runtime
- `vision/detector.py` - YOLO detection and emergency fusion
- `logic/counter.py` - lane counting and ROI logic
- `logic/signal.py` - adaptive signal controller
- `logic/emergency.py` - emergency override behavior
- `ui/overlay.py` - on-screen display
- `scripts/run_judge_demo.py` - one-command demo runner
- `scripts/generate_demo_report.py` - final report from artifacts

## How to explain the runtime flow

Say this in order:

1. A camera frame enters the system.
2. YOLO detects vehicles and possible emergency vehicles.
3. The counter groups vehicles into the configured camera approach.
4. The controller computes congestion scores.
5. The highest-scoring approach gets priority.
6. If an emergency is detected, the corridor lane is overridden.
7. The overlay shows why the system made that choice.

## What the overlay shows

In demo mode it shows:
- current mode
- approach count
- queue count
- congestion score
- current signal state
- emergency status
- decision reason

This is the best screen for judges because it is readable and explainable.

## SUMO layer

SUMO is an optional simulation layer.

Use it to show:
- pre-system baseline
- post-system adaptive control
- prototype corridor pre-clear story

Important note:
- SUMO is a demo and validation layer
- the live prototype is still the OpenCV + YOLO runtime

## How to run the main prototype

```bash
python scripts/run_judge_demo.py --video-source assets/sample_video.mp4 --camera-lane north --approach-roi 120,220,1180,700 --queue-roi 420,360,980,700 --with-ambulance-sim --with-orchestrator --with-benchmark --with-report
```

## How to run a quick smoke test

```bash
python scripts/run_judge_demo.py --video-source assets/sample_video.mp4 --camera-lane north --approach-roi 120,220,1180,700 --queue-roi 420,360,980,700 --headless --max-frames 120 --with-report
```

## How to run SUMO

Pre-system:

```bash
python -m sumo_demo.run_pre_system --gui --out artifacts/sumo_pre_system.csv
```

Post-system:

```bash
python -m sumo_demo.run_post_system --gui --out artifacts/sumo_post_system.csv
```

## What changed compared with a basic traffic-light project

- not a fixed timer
- not just vehicle count
- not a fake emergency flag
- not a single hardcoded flow

Instead, it has:
- per-camera ROI-based counting
- congestion scoring
- emergency source labeling
- baseline vs adaptive benchmark
- judge-ready report generation
- optional SUMO simulation

## Best judge story

Say this:

"We built a real-time traffic prototype that adapts signal timing using live camera input and gives emergency vehicles priority. The live system is single-intersection and camera-based, while the multi-node corridor is shown as a prototype pre-clear simulation. We also generate metrics, benchmark output, and a final report so the result is explainable and measurable."

## Honest scope statement

This is the safest way to describe the project:

- live runtime: single intersection
- camera input: yes
- emergency priority: yes
- multi-node corridor: prototype pre-clear simulation
- SUMO: optional validation/demo layer
