# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Implemented lane counting and signal control logic with headless runner and tests.
- Added GPS-based ambulance tracking and emergency status integration (`shx-02`).
- Added GPS server, utility scripts, and mobile tracker application files.
- Integrated `VehicleDetector` utilizing YOLOv8n and corresponding `test_detector` script.
- Added ambulance dataset YAML, scripts, and configuration updates.
- Tracked YOLO models (`yolov8n.pt` and `yolov8s-worldv2.pt`) with Git LFS.
- Initialized core scaffold and prototype structure with essential modules.
- **P1-A evaluation harness** (`sumo_demo/harness/`): deterministic synthetic
  traffic source, queue model, `FalconBridge` driving loop, and five named
  scenarios (LOW/MEDIUM/HEAVY/SURGE/EMERGENCY).
- **P1-A TraCI slot-in** (`sumo_demo/harness/traci.py`): SUMO-backed source and
  sink sharing the same `TrafficSource` / `SignalSink` interface as the synthetic
  backend. Runner (`sumo_demo/run_traci.py`) degrades to synthetic when SUMO is
  not installed.
- **P1-B multi-seed benchmark** (`sumo_demo/benchmark.py`): runs all five
  scenarios over configurable seeds, computes mean/std, verdict per scenario,
  and writes a JSON artifact to `profiling/artifacts/benchmark_results.json`.
- **P1-C emergency temporal contract tests** (`tests/test_emergency_temporal.py`):
  corridor green before arrival, exclusive preemption, bounded timing, and
  baseline recovery assertions.
- **FakeTraCI validation** (`tests/test_traci_loop.py`): drives the full bridge
  loop against a stubbed TraCI API without real SUMO.
- **Fair explicit baseline green:** `--baseline-green 13` (default 13 s) in the
  benchmark instead of the silent library default; benchmark numbers
  regenerated on this basis.
- **Controller arms:** actuated gap-out (`logic/actuated_signal.py`) and
  queue-plus-arrival fusion (`logic/fusion_signal.py`) controllers, selectable
  via `--controllers baseline adaptive actuated fusion`.
- **Ablation sweeps:** `--ablate` benchmark mode writing
  `profiling/artifacts/ablation_*.json` (switch gap, baseline green, fusion
  weight, green bounds).
- **Hypothesis invariant fuzzing** (`tests/test_invariants_fuzz.py`): safety
  contract checked over randomized demand + emergency plans for all four arms.
- **Emergency corner-case matrix** (`tests/test_emergency_edgecases.py`):
  preemption mid-transition, corridor == green lane, re-trigger during recovery,
  step-0 / final-step ambulance, sequential two-corridor runs.
- **Detector ground-truth scorer** (`scripts/validate_detector.py` +
  `tests/test_validate_detector.py`): MAE/RMSE/tolerance/bias vs labelled
  frame counts.
- **Lazy ultralytics import** in `vision/detector.py`: detector code imports
  and unit-tests without the torch/ultralytics stack (CI-light).
- **GitHub Actions CI** (`.github/workflows/ci.yml`): pytest + ruff.
- **Reproduce pipeline** (`Makefile` targets), `pyproject.toml`
  (ruff/mypy/pytest config), and `main.py --infer-every` / `--reconnect`.
- **Model-free DQN RL baseline** (`sumo_demo/rl_baseline.py` +
  `scripts/train_rl_baseline.py`): learns green-hold durations
  ({5,10,15,30} s) over a round-robin service order from a 9-dim state,
  with n-step returns and a target network; trained on held-out medium
  seeds and served via `--controllers ... rl --rl-weights <path>`.
- **30-seed statistical protocol**: benchmark seed range defaults to
  `42..71`, and every adaptive-vs-baseline comparison adds a matched
  paired t-test, Wilcoxon signed-rank test, 95% CI, and Cohen's dz.
- **Corridor sweep** (`--corridor-sweep`): emergency scenario replayed
  with the ambulance corridor on each of the four approaches.
- **Platoon demand model** (`--demand-model platoon`): correlated burst
  arrivals (on/off cycle with per-approach phase offsets) alongside the
  Poisson model.
- **Log-normal headway demand model** (`--demand-model lognormal`,
  `--headway-cv`): renewal arrivals with log-normal inter-arrival
  headways (cv default 2.0, 0.4 s floor) that cluster into platoons at
  signal-relevant timescales while preserving mean flow; adaptive-vs-
  baseline margins widen to 31-49% under this model. Raw per-seed CSV
  `results/raw/lognormal_results.csv` (750 rows).
- **Per-seed CSV export** (`--export-csv`): long-form (scenario,
  controller, seed, avg_wait_s) so every reported statistic is
  re-derivable.
- **Per-scenario DQN specialists** (`--controllers rl_special`): the same
  DQN architecture retrained per scenario via
  `scripts/train_rl_baseline.py --specialist-dir`, resolved at benchmark
  time from `profiling/artifacts/rl_specialists/rl_<scenario>.pt`; the
  held-out-medium model remains the `rl` generalist arm.
- **Arterial network harness** (`sumo_demo/harness/network.py`): a
  three-intersection line with Poisson end/cross sources, a travel-delay
  transfer model, and programmed westbound preemption progression
  (staggered detection every 8 s per node). Runner
  `sumo_demo/network_benchmark.py` plus contract suite
  `tests/test_network_preemption.py` (14 checks) and raw per-seed CSV
  `results/raw/network_results.csv` (arterial, 60 rows); adaptive vs
  baseline network-wide wait `21.1 -> 17.0` s, d=4.08 s, dz=4.20.
- **MIT LICENSE** and `rl = ["torch"]` optional dependency extra in
  `pyproject.toml`.

### Changed
- Refactored codebase to organize imports and enhance overall code readability across multiple files.
- Streamlined configuration, GPS server, and general codebase cleanup.
- Clarified dataset download instructions and updated links in the `README`.
- Generalized script parameters and updated configuration defaults for better stability.
- `sumo_demo/evaluate.py` now surfaces emergency corridor timing (time_to_preemption,
  corridor_clearance, recovery_time) when the scenario includes an emergency event.
- `sumo_demo/benchmark.py` aggregates emergency corridor metrics (mean over seeds)
  in the JSON artifact and prints them inline.
- `tests/test_detector.py` auto-falls back to the synthetic frame benchmark when
  no camera, video, or webcam is available — no longer reports FAIL in CI.
- Benchmark results re-aligned to the fair explicit 13 s baseline; the new
  controller arms land between the fixed-time and adaptive extremes.
- `sumo_demo/benchmark.py` reports all four arms and significance by
  default; the artifact and printed tables now include the actuated,
  fusion, and RL arms alongside baseline/adaptive.

### Fixed
- Fixed `.gitignore` to properly exclude environment, cache, and large unnecessary files.
- **Emergency baseline pinning bug** (`logic/signal.py` + `sumo_demo/harness/bridge.py`):
  after emergency recovery `_advance_phase` hardcoded `mode='ADAPTIVE'`, so a
  BaselineSignalController lost its identity; additionally `pending_lane` was never
  cleared, leaving the intersection pinned on one lane indefinitely. Both are
  fixed: `_advance_phase` now calls `resume_adaptive()` (subclass-aware), and the
  bridge clears `pending_lane` on recovery completion. Baseline emergency avg_wait
  improved from ~326 s to ~43 s as a result.
- Adaptive controller churn fix (`logic/signal.py`): gap-based early exit now
  requires either the active lane to be essentially cleared (`GAP_ACTIVE_CLEAR_THRESHOLD=3`)
  or 70% of the proportional green budget served (`GAP_EARLY_EXIT_MIN_SERVICE_FRAC=0.7`),
  with a relative gap floor (`SWITCH_GAP_RELATIVE=0.3`) so large queues do not
  trigger constant lane flips.
- **Mode-whitelist transition stall** (`logic/signal.py`): `is_transitioning`
  only recognized `ADAPTIVE`/`BASELINE`, which left the actuated and fusion arms
  pinned in YELLOW with zero throughput. Generalized to any non-emergency mode.
- Ruff lint clean repo, including a previously silent `Circle`/`Rectangle`
  annotation-only missing import.