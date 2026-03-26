# AI Traffic Flow Optimizer and Emergency Green Corridor

## Scope

This repository implements a judge-ready single-intersection traffic controller with a prototype multi-node pre-clear demo:

- real-time vehicle detection
- lane-wise counting
- adaptive signal policy
- emergency priority corridor behavior
- baseline vs adaptive benchmark with metrics output

The live runtime is single-node. Multi-intersection orchestration is presented as a prototype pre-clear artifact, not a full live route-graph deployment.

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

Per-camera ROI calibration (recommended for better queue estimation):

```bash
python main.py --mode adaptive --camera-lane north --approach-roi 120,220,1180,700 --queue-roi 420,360,980,700 --show-count-roi --video-source assets/sample_video.mp4
```

Use `--ui-mode demo` for judge-facing clean overlays (default), or `--ui-mode debug` for all diagnostics.

Emergency trigger modes:

```bash
python main.py --mode adaptive --emergency-source manual --video-source assets/sample_video.mp4
```

Press `e` to toggle a manual corridor override on the current approach.

Emergency detection supports ambulance and fire-service vehicles (`fire_truck`) via model label normalization.

One-command judge demo runner:

```bash
python scripts/run_judge_demo.py --video-source assets/sample_video.mp4 --camera-lane north --approach-roi 120,220,1180,700 --queue-roi 420,360,980,700 --with-ambulance-sim --with-orchestrator --with-benchmark
```

This generates demo artifacts under `artifacts/` including output video, runtime metrics, orchestrator summary, and benchmark metrics.

Add `--with-report` to auto-generate a final concise report for judges:

```bash
python scripts/run_judge_demo.py --video-source assets/sample_video.mp4 --camera-lane north --approach-roi 120,220,1180,700 --queue-roi 420,360,980,700 --with-ambulance-sim --with-orchestrator --with-benchmark --with-report
```

Or generate report directly from existing artifacts:

```bash
python scripts/generate_demo_report.py --metrics-log artifacts/live_metrics_demo.jsonl --benchmark artifacts/metrics.json --orchestrator artifacts/orchestrator_demo.json --output-video artifacts/demo_output.mp4 --out artifacts/demo_report.md
```

Optional top-down fallback (center-split N/S/E/W heuristic):

```bash
python main.py --mode adaptive --top-down-view --video-source assets/sample_video.mp4
```

Enable prototype route-aware corridor pre-clear in runtime (single-node view of a multi-intersection plan):

```bash
python main.py --mode adaptive --enable-orchestrator --orchestrator-route int_a,int_b,int_c,int_d --orchestrator-node-id int_a --orchestrator-preempt-hops 2 --video-source assets/sample_video.mp4
```

Optional observed traffic-light sensing (credibility overlay):

```bash
python main.py --mode adaptive --signal-state-source video --signal-state-roi 0,0,220,160 --video-source assets/sample_video.mp4
```

If sensing is unavailable, fallback to controller signal state (default):

```bash
python main.py --mode adaptive --signal-state-source video --signal-state-fallback controller --video-source assets/sample_video.mp4
```

Or via API source:

```bash
python main.py --mode adaptive --signal-state-source api --signal-state-api-url http://localhost:9000/signal-state --video-source assets/sample_video.mp4
```

GPS emergency prioritization now includes ETA, distance, corridor lane, and lane-source labeling in runtime overlays and metrics logs when gps_server is running.

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

Run a mocked 4-intersection prototype pre-clear simulation:

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

## Judge Demo Script (60-90 seconds)

Use this sequence during judging for a clean narrative:

1. **Problem statement (10s)**
   - "Static traffic signals create avoidable congestion and delay emergency vehicles."

2. **System overview (10s)**
   - "This system uses per-camera AI detection, congestion scoring, and emergency preemption."

3. **Live adaptive behavior (15-20s)**
   - Run:
     ```bash
     python scripts/run_judge_demo.py --video-source assets/sample_video.mp4 --camera-lane north --approach-roi 120,220,1180,700 --queue-roi 420,360,980,700 --with-orchestrator
     ```
   - Say: "Signal decisions are closed-loop and updated continuously from live congestion plus queue score."

4. **Emergency logic (15-20s)**
   - Press `e` to toggle manual emergency and show a forced corridor lane.
   - Say: "Emergency source can be manual, vision, GPS, or fusion; corridor pre-clear is a prototype artifact."

5. **Evidence and impact (15-20s)**
   - Run with benchmark/report for artifacts:
     ```bash
     python scripts/run_judge_demo.py --video-source assets/sample_video.mp4 --camera-lane north --approach-roi 120,220,1180,700 --queue-roi 420,360,980,700 --with-ambulance-sim --with-orchestrator --with-benchmark --with-report
     ```
   - Show:
     - `artifacts/demo_output.mp4`
     - `artifacts/live_metrics_demo.jsonl`
     - `artifacts/metrics.json`
     - `artifacts/demo_report.md`

### Judge talking points

- "Per-camera deployment is practical for real intersections."
- "Congestion scoring improves over count-only switching."
- "Emergency mode supports ambulance and fire-service detection."
- "We log explainable decisions and measurable KPI deltas."
