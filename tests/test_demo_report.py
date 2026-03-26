from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_demo_report import summarize_live_metrics


def test_summarize_live_metrics_handles_empty_rows():
    summary = summarize_live_metrics([])
    assert summary["frames"] == 0
    assert summary["avg_wait_seconds_mean"] is None


def test_summarize_live_metrics_extracts_key_fields():
    rows = [
        {
            "avg_wait_seconds": 3.0,
            "max_queue": 5,
            "throughput_score": 10,
            "decision_reason": "Hold",
            "emergency_source": "vision",
            "emergency_active": False,
        },
        {
            "avg_wait_seconds": 5.0,
            "max_queue": 7,
            "throughput_score": 14,
            "decision_reason": "Hold",
            "emergency_source": "vision",
            "emergency_active": True,
        },
    ]
    summary = summarize_live_metrics(rows)

    assert summary["frames"] == 2
    assert summary["emergency_frames"] == 1
    assert abs(float(summary["avg_wait_seconds_mean"]) - 4.0) < 1e-9
    assert summary["max_queue_peak"] == 7
    assert summary["throughput_final"] == 14
    assert summary["top_decision_reason"] == "Hold"
    assert summary["top_emergency_source"] == "vision"


if __name__ == "__main__":
    test_summarize_live_metrics_handles_empty_rows()
    test_summarize_live_metrics_extracts_key_fields()
    print("PASS test_demo_report")
