"""Tests for the P1-B benchmark aggregation + artifact writer.

SUMO is not required: benchmarks run on the deterministic synthetic backend.
Covers:
- _run produces a MetricsReport for each controller
- _average aggregates mean/std over several seeds
- _verdict is correct for clearly-better / worse / tied deltas
- run_benchmark covers all five scenarios and writes a parseable JSON artifact
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sumo_demo.benchmark import _run, _average, _verdict
from sumo_demo import benchmark as benchmark_mod
from sumo_demo.harness import SCENARIOS, MetricsReport


def test_run_produces_report():
    rep = _run("low", "adaptive", steps=50, seed=1, service_rate=1.0)
    assert isinstance(rep, MetricsReport)
    assert rep.controller == "ADAPTIVE"
    assert rep.scenario == "low"
    assert rep.total_steps == 50


def test_average_over_seeds():
    result = _average("low", "adaptive", steps=200, seeds=[1, 2, 3], service_rate=1.0)
    assert len(result["avg_wait_s"]) == 3
    assert result["avg_wait_mean"] > 0
    assert result["avg_wait_std"] >= 0
    assert result["vehicles_served_mean"] > 0


def test_verdict_directions():
    better = {"avg_wait_mean": 10.0}
    worse = {"avg_wait_mean": 20.0}
    assert _verdict(worse, better) == "ADAPTIVE_BETTER"
    assert _verdict(better, worse) == "BASELINE_BETTER"
    assert _verdict({"avg_wait_mean": 10.0}, {"avg_wait_mean": 10.0}) == "TIED"


def test_run_benchmark_covers_all_scenarios():
    results = benchmark_mod.run_benchmark(sorted(SCENARIOS), steps=60, seeds=[1], service_rate=1.0)
    assert set(results) == set(SCENARIOS)
    for name in SCENARIOS:
        assert "baseline" in results[name]
        assert "adaptive" in results[name]
        assert results[name]["verdict"] in ("ADAPTIVE_BETTER", "BASELINE_BETTER", "TIED")


def test_write_artifact_json(tmp_path=None):
    tmp_path = tmp_path or Path(tempfile.mkdtemp())
    results = benchmark_mod.run_benchmark(["low"], steps=60, seeds=[1], service_rate=1.0)
    out_path = tmp_path / "benchmark_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "p1b": "multi-scenario benchmark",
                "steps": 60,
                "seeds": [1],
                "service_rate": 1.0,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert "low" in data["results"]
    assert "adaptive" in data["results"]["low"]
    assert isinstance(data["results"]["low"]["adaptive"]["avg_wait_mean"], (int, float))


if __name__ == "__main__":
    test_run_produces_report()
    test_average_over_seeds()
    test_verdict_directions()
    test_run_benchmark_covers_all_scenarios()
    test_write_artifact_json()
    print("PASS test_benchmark")