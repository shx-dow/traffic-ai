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

from sumo_demo import benchmark as benchmark_mod
from sumo_demo.benchmark import _average, _run, _verdict
from sumo_demo.harness import SCENARIOS, MetricsReport


def test_run_produces_report():
    rep = _run("low", "adaptive", steps=50, seed=1, service_rate=1.0, green_seconds=13)
    assert isinstance(rep, MetricsReport)
    assert rep.controller == "ADAPTIVE"
    assert rep.scenario == "low"
    assert rep.total_steps == 50


def test_average_over_seeds():
    result = _average("low", "adaptive", steps=200, seeds=[1, 2, 3], service_rate=1.0, green_seconds=13)
    assert len(result["avg_wait_s"]) == 3
    assert result["avg_wait_mean"] > 0
    assert result["avg_wait_std"] >= 0
    assert result["vehicles_served_mean"] > 0


def test_verdict_directions():
    better = {"avg_wait_mean": 10.0}
    worse = {"avg_wait_mean": 20.0}
    assert _verdict(worse, better) == "CONTENDER_BETTER"
    assert _verdict(better, worse) == "REFERENCE_BETTER"
    assert _verdict({"avg_wait_mean": 10.0}, {"avg_wait_mean": 10.0}) == "TIED"


def test_paired_significance_rejects_no_difference():
    ref = {"avg_wait_s": [12.0, 14.0, 13.0, 12.5, 13.5]}
    cont = {"avg_wait_s": [6.0, 7.0, 5.5, 6.5, 6.0]}
    st = benchmark_mod.paired_significance(ref, cont)
    assert st["seeds"] == 5
    assert st["mean_diff_s"] > 0
    assert st["p"] < 0.05
    assert st["ci95_s"][0] > 0
    assert st["cohen_dz"] > 0


def test_paired_significance_tied_and_missing():
    same = {"avg_wait_s": [10.0] * 5}
    st = benchmark_mod.paired_significance(same, {"avg_wait_s": [10.0] * 5})
    assert st is not None and st["p"] == 1.0
    assert benchmark_mod.paired_significance({"avg_wait_s": None}, {"avg_wait_s": None}) is None
    assert benchmark_mod.paired_significance({"avg_wait_s": [1.0]}, {"avg_wait_s": [1.0, 2.0]}) is None


def test_t_crit_95_matches_table():
    assert abs(benchmark_mod._t_crit_95(4) - 2.7764451051977987) < 1e-3
    assert benchmark_mod._t_two_tailed_p(1.0, 4) > 0.3
    assert benchmark_mod._t_two_tailed_p(10.0, 4) < 0.001


def test_wilcoxon_rejects_consistent_shift():
    diffs = [6.0, 7.0, 5.5, 6.5, 6.0, 7.5, 5.0, 8.0]
    p = benchmark_mod._wilcoxon_p(diffs)
    assert p is not None and p < 0.05


def test_wilcoxon_tied_and_too_few_within():
    same = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    assert benchmark_mod._wilcoxon_p([0.0] * 6) == 1.0
    assert benchmark_mod._wilcoxon_p([1.0, -1.0, 1.0, -1.0]) is None
    assert benchmark_mod._wilcoxon_p(same) is not None


def test_paired_significance_includes_wilcoxon():
    ref = {"avg_wait_s": [12.0, 14.0, 13.0, 12.5, 13.5]}
    cont = {"avg_wait_s": [6.0, 7.0, 5.5, 6.5, 6.0]}
    st = benchmark_mod.paired_significance(ref, cont)
    assert st["wilcoxon_p"] is not None
    assert st["wilcoxon_p"] < 0.05


def test_run_benchmark_covers_all_scenarios():
    results = benchmark_mod.run_benchmark(
        sorted(SCENARIOS), steps=60, seeds=[1], service_rate=1.0, green_seconds=13
    )
    assert set(results) == set(SCENARIOS)
    for name in SCENARIOS:
        assert "baseline" in results[name]
        assert "adaptive" in results[name]
        assert results[name]["verdict"] in ("CONTENDER_BETTER", "REFERENCE_BETTER", "TIED")


def test_write_artifact_json(tmp_path=None):
    tmp_path = tmp_path or Path(tempfile.mkdtemp())
    results = benchmark_mod.run_benchmark(
        ["low"], steps=60, seeds=[1], service_rate=1.0, green_seconds=13
    )
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
