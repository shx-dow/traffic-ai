from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


LANES = ("north", "south", "east", "west")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a concise hackathon demo report from artifacts")
    parser.add_argument("--metrics-log", default="artifacts/live_metrics_demo.jsonl")
    parser.add_argument("--benchmark", default="artifacts/metrics.json")
    parser.add_argument("--orchestrator", default="artifacts/orchestrator_demo.json")
    parser.add_argument("--output-video", default="artifacts/demo_output.mp4")
    parser.add_argument("--out", default="artifacts/demo_report.md")
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return rows


def summarize_live_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "frames": 0,
            "emergency_frames": 0,
            "avg_wait_seconds_mean": None,
            "max_queue_peak": None,
            "throughput_final": None,
            "avg_lane_counts": None,
            "top_decision_reason": None,
            "top_emergency_source": None,
            "top_corridor_lane": None,
            "top_corridor_source": None,
        }

    emergency_frames = sum(1 for row in rows if bool(row.get("emergency_active")))
    waits = [float(row.get("avg_wait_seconds", 0.0)) for row in rows if "avg_wait_seconds" in row]
    queues = [int(row.get("max_queue", 0)) for row in rows if "max_queue" in row]
    final_throughput = rows[-1].get("throughput_score")

    lane_totals = {lane: 0.0 for lane in LANES}
    lane_frames = 0
    for row in rows:
        counts = row.get("lane_counts") or {}
        lane_frames += 1
        for lane in LANES:
            lane_totals[lane] += float(counts.get(lane, 0.0))
    avg_lane_counts = {lane: lane_totals[lane] / lane_frames for lane in LANES} if lane_frames else None

    reasons = Counter(str(row.get("decision_reason", "")) for row in rows if row.get("decision_reason"))
    sources = Counter(str(row.get("emergency_source", "")) for row in rows if row.get("emergency_source"))
    corridor_lanes = Counter(str(row.get("corridor_lane", "")) for row in rows if row.get("corridor_lane"))
    corridor_sources = Counter(str(row.get("corridor_lane_source", "")) for row in rows if row.get("corridor_lane_source"))

    return {
        "frames": len(rows),
        "emergency_frames": emergency_frames,
        "avg_wait_seconds_mean": mean(waits) if waits else None,
        "max_queue_peak": max(queues) if queues else None,
        "throughput_final": final_throughput,
        "avg_lane_counts": avg_lane_counts,
        "top_decision_reason": reasons.most_common(1)[0][0] if reasons else None,
        "top_emergency_source": sources.most_common(1)[0][0] if sources else None,
        "top_corridor_lane": corridor_lanes.most_common(1)[0][0] if corridor_lanes else None,
        "top_corridor_source": corridor_sources.most_common(1)[0][0] if corridor_sources else None,
    }


def summarize_benchmark(benchmark: Dict[str, Any] | None) -> Dict[str, Any]:
    if not benchmark:
        return {"wait_reduction_pct": None, "throughput_gain_pct": None}
    comp = benchmark.get("comparison", {})
    return {
        "wait_reduction_pct": comp.get("wait_reduction_pct"),
        "throughput_gain_pct": comp.get("throughput_gain_pct"),
    }


def summarize_orchestrator(orchestrator: Dict[str, Any] | None) -> Dict[str, Any]:
    if not orchestrator:
        return {"final_emergency_nodes": None, "route": None}
    events = orchestrator.get("events", [])
    route = orchestrator.get("scenario", {}).get("route")
    if not events:
        return {"final_emergency_nodes": 0, "route": route}
    final_plan = events[-1].get("plan", {})
    emergency_nodes = [node for node, state in final_plan.items() if str(state.get("mode", "")).upper() == "EMERGENCY"]
    return {
        "final_emergency_nodes": len(emergency_nodes),
        "route": route,
    }


def calculate_green_red_split(avg_lane_counts: Dict[str, float] | None) -> Dict[str, Any]:
    if not avg_lane_counts:
        return {"greens": None, "reds": None, "cycle_seconds": None}

    min_green = 10
    max_green = 60
    total = sum(float(avg_lane_counts.get(lane, 0.0)) for lane in LANES)

    if total <= 0:
        greens = {lane: min_green for lane in LANES}
    else:
        budget = max_green - min_green
        greens = {}
        for lane in LANES:
            raw = min_green + (float(avg_lane_counts.get(lane, 0.0)) / total) * budget
            greens[lane] = max(min_green, min(int(raw), max_green))

    cycle_seconds = sum(greens.values())
    reds = {lane: max(cycle_seconds - greens[lane], 0) for lane in LANES}
    return {"greens": greens, "reds": reds, "cycle_seconds": cycle_seconds}


def render_markdown(
    *,
    live: Dict[str, Any],
    benchmark: Dict[str, Any],
    orchestrator: Dict[str, Any],
    output_video: str,
    metrics_log: str,
    benchmark_path: str,
    orchestrator_path: str,
) -> str:
    def fmt(value: Any, digits: int = 2) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    signal_split = calculate_green_red_split(live.get("avg_lane_counts"))
    greens = signal_split["greens"]
    reds = signal_split["reds"]
    cycle_seconds = signal_split["cycle_seconds"]

    lines = [
        "# Demo Report",
        "",
        "## Single-Node Runtime Summary",
        f"- Frames processed: {fmt(live['frames'])}",
        f"- Emergency frames: {fmt(live['emergency_frames'])}",
        f"- Mean average wait (s): {fmt(live['avg_wait_seconds_mean'])}",
        f"- Peak queue: {fmt(live['max_queue_peak'])}",
        f"- Final throughput score: {fmt(live['throughput_final'])}",
        f"- Dominant decision reason: {fmt(live['top_decision_reason'])}",
        f"- Dominant emergency source: {fmt(live['top_emergency_source'])}",
        f"- Dominant corridor lane: {fmt(live['top_corridor_lane'])}",
        f"- Dominant corridor lane source: {fmt(live['top_corridor_source'])}",
        "",
        "## Recommended Green/Red Split",
    ]

    if greens and reds and cycle_seconds:
        lines.extend([
            f"- Cycle length from adaptive split: {fmt(cycle_seconds)} s",
            "",
            "| Lane | Green (s) | Red (s) |",
            "| --- | ---: | ---: |",
        ])
        for lane in LANES:
            lines.append(f"| {lane.title()} | {fmt(greens[lane])} | {fmt(reds[lane])} |")
        lines.append("")
    else:
        lines.extend([
            "- No traffic samples available to compute a green/red split.",
            "",
        ])

    lines.extend([
        "## Benchmark Summary",
        f"- Wait reduction vs baseline (%): {fmt(benchmark['wait_reduction_pct'])}",
        f"- Throughput gain vs baseline (%): {fmt(benchmark['throughput_gain_pct'])}",
        "",
        "## Prototype Pre-Clear Summary",
        f"- Route: {fmt(orchestrator['route'])}",
        f"- Emergency nodes in final state: {fmt(orchestrator['final_emergency_nodes'])}",
        "",
        "## Artifacts",
        f"- Demo video: `{output_video}`",
        f"- Runtime metrics log: `{metrics_log}`",
        f"- Benchmark metrics: `{benchmark_path}`",
        f"- Orchestrator output: `{orchestrator_path}`",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    metrics_path = Path(args.metrics_log)
    benchmark_path = Path(args.benchmark)
    orchestrator_path = Path(args.orchestrator)

    live_rows = _read_jsonl(metrics_path)
    live = summarize_live_metrics(live_rows)
    benchmark = summarize_benchmark(_read_json(benchmark_path))
    orchestrator = summarize_orchestrator(_read_json(orchestrator_path))

    content = render_markdown(
        live=live,
        benchmark=benchmark,
        orchestrator=orchestrator,
        output_video=args.output_video,
        metrics_log=str(metrics_path),
        benchmark_path=str(benchmark_path),
        orchestrator_path=str(orchestrator_path),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    print(f"Wrote demo report to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
