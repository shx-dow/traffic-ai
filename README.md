# AI Traffic Flow Optimizer and Emergency Green Corridor

## Scope

This repository implements the core logic for a single-intersection traffic controller:

- real-time vehicle detection
- lane-wise counting
- adaptive signal policy
- emergency override behavior
- baseline vs adaptive benchmark with metrics output

UI and multi-intersection orchestration are separate workstreams.

## Core modules

- `main.py`: runtime entrypoint, supports adaptive and baseline signal modes
- `vision/detector.py`: YOLO detection and emergency flag integration
- `logic/counter.py`: lane assignment and per-lane vehicle counts
- `logic/signal.py`: adaptive signal controller (fairness and anti-flap switching)
- `logic/baseline_signal.py`: fixed-cycle baseline controller for A/B comparison
- `logic/simulation.py`: deterministic frame-event simulation runner
- `logic/metrics.py`: metric computation and baseline/adaptive comparison
- `scripts/run_benchmark.py`: benchmark command that writes `artifacts/metrics.json`

## Setup

```bash
pip install -r requirements.txt
```

## Run

Adaptive mode:

```bash
python main.py --mode adaptive --video-source assets/sample_video.mp4
```

Per-camera mode is the default. Set which approach this camera represents:

```bash
python main.py --mode adaptive --camera-lane north --video-source assets/sample_video.mp4
```

Optional top-down fallback (center-split N/S/E/W heuristic):

```bash
python main.py --mode adaptive --top-down-view --video-source assets/sample_video.mp4
```

Baseline mode:

```bash
python main.py --mode baseline --baseline-green-seconds 20 --video-source assets/sample_video.mp4
```

Optional core preflight checks before runtime:

```bash
python main.py --run-pipeline --mode adaptive --video-source assets/sample_video.mp4
```

## Benchmark

Generate baseline vs adaptive metrics:

```bash
python scripts/run_benchmark.py
```

Output file:

- `artifacts/metrics.json`

## Multi-intersection demo

Run a mocked 4-intersection corridor pre-clear simulation:

```bash
python scripts/demo_orchestrator.py --frames 120 --fps 15 --preempt-hops 2 --latch-seconds 3
```

Output file:

- `artifacts/orchestrator_demo.json`

## Tests

Run core logic tests:

```bash
python tests/test_logic.py
python tests/test_baseline_signal.py
python tests/test_runtime.py
python tests/test_simulation.py
python tests/test_metrics.py
python tests/test_custom_ambulance.py
python tests/test_detector_logic.py
python tests/test_live_metrics.py
python tests/test_orchestrator.py
```
