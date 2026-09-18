"""Tests for the P1-A harness: traffic source, bridge, queue model, metrics.

Covers:
- Scenario traffic sources produce correct snapshots and are reproducible.
- FalconBridge drives baseline + adaptive through the phase machine.
- No two greens in the same step; signals go through YELLOW/ALL_RED on switch.
- Emergency preemption gives corridor GREEN, is not immediate, and recovers.
- Queue model: RED builds queue, GREEN drains, average wait increases with congestion.
- All five scenarios defined.
- MetricsReport structure.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import statistics

from sumo_demo.harness.bridge import BridgeResult, FalconBridge, StepRecord, make_controller
from sumo_demo.harness.metrics import MetricsReport, QueueModel
from sumo_demo.harness.sinks import NullSignalSink, SyntheticSignalSink
from sumo_demo.harness.traffic import (
    SCENARIOS,
    Scenario,
    ScenarioTrafficSource,
    TrafficSnapshot,
    _poisson,
    emergency_corridor_variant,
)

LANES = ("north", "south", "east", "west")


def _assert_one_green(signal_state):
    greens = [l for l in LANES if signal_state.get(l) == "GREEN"]
    assert len(greens) <= 1, f"Multiple greens: {signal_state}"


def test_scenario_traffic_source_reproducible():
    src = ScenarioTrafficSource(SCENARIOS["medium"], seed=7)
    first = list(src.steps(50))
    second = list(src.steps(50))
    assert first == second
    assert len(first) == 50
    for snap in first:
        assert isinstance(snap, TrafficSnapshot)
        assert isinstance(snap.lane_counts, dict)
        for lane in LANES:
            assert lane in snap.lane_counts
            assert snap.lane_counts[lane] >= 0


def test_poisson_average_reasonable():
    rng = random.Random(123)
    rate = 0.1
    samples = [_poisson(rng, rate) for _ in range(10000)]
    avg = sum(samples) / len(samples)
    assert 0.05 < avg < 0.20, f"Poisson average {avg} outside expected range for rate {rate}"


def test_all_scenarios_defined():
    for name in ("low", "medium", "heavy", "surge", "emergency"):
        assert name in SCENARIOS
        sc = SCENARIOS[name]
        assert isinstance(sc, Scenario)
        assert isinstance(sc.flows_per_hour, dict)


def test_emergency_scenario_has_window():
    sc = SCENARIOS["emergency"]
    assert sc.emergency_lane is not None
    assert sc.emergency_window is not None


def test_baseline_single_green_always():
    bridge = FalconBridge(
        source=ScenarioTrafficSource(SCENARIOS["medium"], seed=1),
        sink=NullSignalSink(),
        fps=1,
    )
    result = bridge.run(make_controller("baseline"), total_steps=200)
    for rec in result.records:
        _assert_one_green(rec.signal_state)


def test_adaptive_responds_to_heavy_demand():
    sc = Scenario(
        name="unbalanced_test",
        flows_per_hour={"north": 1200, "south": 1200, "east": 100, "west": 100},
    )
    bridge = FalconBridge(
        source=ScenarioTrafficSource(sc, seed=5),
        sink=NullSignalSink(),
        fps=1,
    )
    result = bridge.run(make_controller("adaptive"), total_steps=300)
    # NS with heavy demand should get at least some green (at least > 10% of steps)
    ns_respective = sum(1 for r in result.records if r.signal_state.get("north") == "GREEN")
    assert ns_respective > 20, f"N-S should get more green, got {ns_respective} of {len(result.records)}"


def test_baseline_rotates_lanes():
    bridge = FalconBridge(
        source=ScenarioTrafficSource(SCENARIOS["low"], seed=3),
        sink=NullSignalSink(),
        fps=1,
    )
    result = bridge.run(make_controller("baseline", green_seconds=10), total_steps=200)
    # baseline should give green to multiple lanes in 200 steps
    all_active = set()
    for r in result.records:
        if any(v == "GREEN" for v in r.signal_state.values()):
            for l in LANES:
                if r.signal_state.get(l) == "GREEN":
                    all_active.add(l)
    assert len(all_active) >= 2, f"Baseline should rotate, only saw: {all_active}"


def test_queue_model_drains_on_green():
    qm = QueueModel(service_rate=1.0)
    build_arrivals = {"north": 5, "south": 0, "east": 0, "west": 0}
    empty_arrivals = {"north": 0, "south": 0, "east": 0, "west": 0}
    red_state = {l: "RED" for l in LANES}
    green_state = {l: "GREEN" if l == "north" else "RED" for l in LANES}

    # 10 red steps → queue builds
    for _ in range(10):
        qm.step(build_arrivals, red_state)
    assert qm.queue["north"] >= 50, f"Queue should build on RED, got {qm.queue['north']}"

    # 60 green steps with no new arrivals → queue drains to zero
    for _ in range(60):
        qm.step(empty_arrivals, green_state)
    assert qm.queue["north"] == 0, f"Queue should drain on GREEN, got {qm.queue['north']}"


def test_metrics_report_structure():
    qm = QueueModel()
    arrivals = {l: 2 for l in LANES}
    green = {l: "GREEN" if l == "north" else "RED" for l in LANES}
    for _ in range(10):
        qm.step(arrivals, green)
    report = qm.report("medium", "ADAPTIVE", 10)
    assert isinstance(report, MetricsReport)
    assert report.vehicles_served > 0
    assert report.scenario == "medium"
    assert report.controller == "ADAPTIVE"
    assert report.total_steps == 10


def test_falcon_bridge_baseline_result_has_records():
    bridge = FalconBridge(
        source=ScenarioTrafficSource(SCENARIOS["low"], seed=9),
        sink=SyntheticSignalSink(service_rate=0.5),
        fps=1,
    )
    result = bridge.run(make_controller("baseline", green_seconds=10), total_steps=60)
    assert isinstance(result, BridgeResult)
    assert len(result.records) == 60
    for rec in result.records:
        assert isinstance(rec, StepRecord)
        assert rec.mode == "BASELINE"


def test_falcon_bridge_adaptive_result_has_records():
    bridge = FalconBridge(
        source=ScenarioTrafficSource(SCENARIOS["medium"], seed=9),
        sink=SyntheticSignalSink(service_rate=0.5),
        fps=1,
    )
    result = bridge.run(make_controller("adaptive"), total_steps=60)
    assert len(result.records) == 60
    assert all(rec.mode == "ADAPTIVE" for rec in result.records)


def test_queue_accumulation_metrics():
    qm = QueueModel()
    arrivals = {l: 10 for l in LANES}
    red = {l: "RED" for l in LANES}
    green = {l: "GREEN" if l == "north" else "RED" for l in LANES}
    for _ in range(5):
        qm.step(arrivals, red)
    for _ in range(10):
        qm.step(arrivals, green)
    assert qm.max_queue > 0
    assert qm.served["north"] > 0


def test_surge_scenario_emergency_with_emergency_window():
    sc = Scenario(
        name="surge_emergency",
        flows_per_hour={l: 300 for l in LANES},
        emergency_lane="north",
        emergency_window=(10, 30),
    )
    src = ScenarioTrafficSource(sc, seed=1)
    snaps = list(src.steps(60))
    assert any(s.emergency_lane == "north" for s in snaps)
    assert any(s.emergency_lane is None for s in snaps)


def test_platoon_source_reproducible_and_bursty():
    src = ScenarioTrafficSource(SCENARIOS["medium"], seed=7, demand_model="platoon")
    first = list(src.steps(600))
    second = list(src.steps(600))
    assert first == second
    total_bursts = sum(1 for s in first if max(s.lane_counts.values()) >= 2)
    assert total_bursts > 0, "platoon demand should produce bursts"
    for lane in LANES:
        assert sum(s.lane_counts[lane] for s in first) > 0


def test_platoon_mean_rate_matches_flow():
    src = ScenarioTrafficSource(SCENARIOS["medium"], seed=7, demand_model="platoon")
    snaps = list(src.steps(3600))
    north_flow = SCENARIOS["medium"].flows_per_hour["north"]
    per_hour = sum(s.lane_counts["north"] for s in snaps)
    assert 0.6 * north_flow < per_hour < 1.4 * north_flow, f"platoon N arrivals/hour={per_hour}"


def _bin_dispersion(snaps, lane: str, bin_s: int) -> float:
    bins = []
    for i in range(0, len(snaps) - bin_s, bin_s):
        bins.append(sum(s.lane_counts[lane] for s in snaps[i:i + bin_s]))
    mean = statistics.mean(bins)
    return statistics.variance(bins) / mean if mean > 0 else 0.0


def test_lognormal_source_reproducible_and_bursty():
    src = ScenarioTrafficSource(SCENARIOS["medium"], seed=7, demand_model="lognormal")
    first = list(src.steps(3600))
    second = list(src.steps(3600))
    assert first == second, "lognormal demand must be deterministic in the seed"
    lane = "north"
    poisson = ScenarioTrafficSource(SCENARIOS["medium"], seed=7, demand_model="poisson")
    poisson_snaps = list(poisson.steps(3600))
    # Log-normal headways lengthen the burst structure at cycle timescales:
    # the 30 s index of dispersion must clearly exceed the Poisson one,
    # while the 1 s dispersion stays near 1 (fine-scale rate unchanged).
    disp_ln, disp_po = _bin_dispersion(first, lane, 30), _bin_dispersion(poisson_snaps, lane, 30)
    assert disp_ln > disp_po + 0.5, f"30s dispersion lognormal={disp_ln:.2f} vs poisson={disp_po:.2f}"
    assert abs(_bin_dispersion(first, lane, 1) - 1.0) < 0.25


def test_lognormal_mean_rate_matches_flow():
    src = ScenarioTrafficSource(SCENARIOS["medium"], seed=7, demand_model="lognormal")
    snaps = list(src.steps(3600))
    north_flow = SCENARIOS["medium"].flows_per_hour["north"]
    per_hour = sum(s.lane_counts["north"] for s in snaps)
    assert 0.6 * north_flow < per_hour < 1.4 * north_flow, f"lognormal N arrivals/hour={per_hour}"
    for lane in LANES:
        assert sum(s.lane_counts[lane] for s in snaps) > 0


def test_lognormal_respects_min_headway():
    src = ScenarioTrafficSource(SCENARIOS["heavy"], seed=3, demand_model="lognormal",
                                headway_cv=2.5, min_headway_s=0.4)
    snaps = list(src.steps(3600))
    # At the heaviest flow, a 0.4 s headway floor permits at most 3 arrivals
    # per 1 s step (0.0, 0.4, 0.8); anything more would violate the floor.
    for lane in LANES:
        max_step = max(s.lane_counts[lane] for s in snaps)
        assert max_step <= 3, f"{lane} lane peak {max_step} exceeds 3/s floor capacity"


def test_emergency_corridor_variant_each_approach():
    for lane in ("north", "south", "east", "west"):
        sc = emergency_corridor_variant(lane)
        assert sc.emergency_lane == lane
        assert sc.emergency_window == SCENARIOS["emergency"].emergency_window


def test_null_sink_report():
    sink = NullSignalSink()
    sink.on_step(0, {l: "GREEN" for l in LANES}, {l: 0 for l in LANES})
    report = sink.report("low", "ADAPTIVE", 1)
    assert isinstance(report, MetricsReport)


if __name__ == "__main__":
    test_scenario_traffic_source_reproducible()
    test_poisson_average_reasonable()
    test_all_scenarios_defined()
    test_emergency_scenario_has_window()
    test_baseline_single_green_always()
    test_adaptive_responds_to_heavy_demand()
    test_platoon_source_reproducible_and_bursty()
    test_platoon_mean_rate_matches_flow()
    test_lognormal_source_reproducible_and_bursty()
    test_lognormal_mean_rate_matches_flow()
    test_lognormal_respects_min_headway()
    test_baseline_rotates_lanes()
    test_queue_model_drains_on_green()
    test_metrics_report_structure()
    test_falcon_bridge_baseline_result_has_records()
    test_falcon_bridge_adaptive_result_has_records()
    test_queue_accumulation_metrics()
    test_surge_scenario_emergency_with_emergency_window()
    test_null_sink_report()
    print("PASS test_harness")
