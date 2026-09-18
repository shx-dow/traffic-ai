"""benchmark.py — P1-B multi-scenario benchmark (baseline vs adaptive).

Runs LOW/MEDIUM/HEAVY/SURGE/EMERGENCY x {baseline, adaptive, actuated,
fusion, rl, rl_special} over multiple seeds and aggregates:
  - avg_wait_s, max_queue, avg_queue, vehicles_served (mean +/- std)
  - per-scenario verdict (adaptive better / baseline better / tied) based on
    the mean average-wait delta
  - a JSON artifact under profiling/artifacts/benchmark_results.json

The fixed-time baseline green is *explicit* (--baseline-green, default 13 s)
so the benchmark always matches the tuned baseline instead of silently
using the library default.

Usage:
    python -m sumo_demo.benchmark [--steps 600] [--seeds 42 43 44 45 46]
                                  [--baseline-green 13]
                                  [--controllers baseline adaptive]
                                  [--scenario medium] [--out artifacts/...json]
    python -m sumo_demo.benchmark --ablate switch_gap_relative \
        --ablate-values 0.1 0.3 0.5 --ablate-scenario medium
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from .harness import (
    SCENARIOS,
    FalconBridge,
    MetricsReport,
    ScenarioTrafficSource,
    SyntheticSignalSink,
    emergency_corridor_variant,
    make_controller,
)

ROOT = Path(__file__).resolve().parent.parent

# Parameter overrides supported by --ablate.  Keys are instance attributes set
# on the freshly-built controller before the run; the reference (base) value
# for each is captured on a bare constructor when the sweep is requested.
ABLATION_PARAMS: dict[str, tuple[str, str]] = {
    "switch_gap_relative": ("adaptive", "SWITCH_GAP_RELATIVE"),
    "gap_clear": ("adaptive", "GAP_ACTIVE_CLEAR_THRESHOLD"),
    "mg_min": ("adaptive", "MIN_GREEN"),
    "mg_max": ("adaptive", "MAX_GREEN"),
    "gap_early_exit": ("adaptive", "GAP_EARLY_EXIT_MIN_SERVICE_FRAC"),
    "baseline_green": ("baseline", "green_seconds"),
    "fusion_weight": ("fusion", "FUSION_WEIGHT"),
}


def _build_controller(
    mode: str,
    green_seconds: int,
    ablate: str | None,
    ablate_value: float | None,
    rl_weights: str | None = None,
):
    kwargs = {}
    if mode == "baseline":
        kwargs["green_seconds"] = green_seconds
    elif mode == "actuated":
        kwargs["min_green_s"] = 10
        kwargs["max_green_s"] = 30
        kwargs["gap_s"] = 3
    elif mode == "rl":
        kwargs["weights_path"] = rl_weights
    controller = make_controller(mode, **kwargs)

    if ablate:
        if ablate not in ABLATION_PARAMS:
            raise ValueError(f"unknown ablation param: {ablate} (choose from {sorted(ABLATION_PARAMS)})")
        ctrl_mode, attr = ABLATION_PARAMS[ablate]
        if ctrl_mode != mode:
            raise ValueError(f"--ablate {ablate} applies to controller '{ctrl_mode}', not '{mode}'")
        setattr(controller, attr, float(ablate_value) if "green_seconds" not in attr else int(ablate_value))
    return controller


def _run(
    scenario_name: str,
    mode: str,
    steps: int,
    seed: int,
    service_rate: float,
    green_seconds: int,
    ablate: str | None = None,
    ablate_value: float | None = None,
    rl_weights: str | None = None,
    demand_model: str = "poisson",
    platoon_on_s: int = 8,
    platoon_off_s: int = 12,
    corridor: str | None = None,
    rl_specialist_dir: str | None = None,
    headway_cv: float = 1.2,
) -> MetricsReport:
    if mode == "rl_special":
        key = "emergency" if corridor is not None else scenario_name
        weights = Path(rl_specialist_dir) / f"rl_{key}.pt"
        if not weights.is_file():
            raise FileNotFoundError(
                f"rl_special needs per-scenario weights at {weights} "
                f"(train with: python scripts/train_rl_baseline.py --scenario {key} "
                f"--out {weights})"
            )
        mode = "rl"
        rl_weights = str(weights)
    scenario = SCENARIOS[scenario_name]
    if corridor is not None:
        scenario = emergency_corridor_variant(corridor)
    controller = _build_controller(mode, green_seconds, ablate, ablate_value, rl_weights)
    sink = SyntheticSignalSink(service_rate=service_rate)
    bridge = FalconBridge(
        source=ScenarioTrafficSource(
            scenario,
            seed=seed,
            demand_model=demand_model,
            platoon_on_s=platoon_on_s,
            platoon_off_s=platoon_off_s,
            headway_cv=headway_cv,
        ),
        sink=sink,
        fps=1,
    )
    result = bridge.run(controller, total_steps=steps)
    return result.report(scenario.name, mode.upper(), sink, steps)


def _average(
    name: str,
    mode: str,
    steps: int,
    seeds,
    service_rate: float,
    green_seconds: int,
    ablate: str | None = None,
    ablate_value: float | None = None,
    rl_weights: str | None = None,
    demand_model: str = "poisson",
    platoon_on_s: int = 8,
    platoon_off_s: int = 12,
    corridor: str | None = None,
    rl_specialist_dir: str | None = None,
    headway_cv: float = 1.2,
) -> dict[str, float | None]:
    reports = [
        _run(
            name, mode, steps, seed, service_rate, green_seconds, ablate, ablate_value,
            rl_weights, demand_model, platoon_on_s, platoon_off_s, corridor,
            rl_specialist_dir, headway_cv,
        )
        for seed in seeds
    ]
    waits = [r.avg_wait_s for r in reports]
    maxqs = [r.max_queue for r in reports]
    avgqs = [r.avg_queue for r in reports]
    served = [r.vehicles_served for r in reports]

    def mean_std(values: list[float]) -> tuple[float, float]:
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        return round(mean, 2), round(std, 2)

    def mean_only(values) -> float | None:
        present = [float(v) for v in values if v is not None]
        return round(statistics.fmean(present), 2) if present else None

    return {
        "avg_wait_s": waits,
        "avg_wait_mean": mean_std(waits)[0],
        "avg_wait_std": mean_std(waits)[1],
        "max_queue_mean": mean_std(maxqs)[0],
        "max_queue_std": mean_std(maxqs)[1],
        "avg_queue_mean": mean_std(avgqs)[0],
        "avg_queue_std": mean_std(avgqs)[1],
        "vehicles_served_mean": mean_std(served)[0],
        "vehicles_served_std": mean_std(served)[1],
        "time_to_preemption_s_mean": mean_only([r.time_to_preemption_s for r in reports]),
        "corridor_clearance_s_mean": mean_only([r.corridor_clearance_s for r in reports]),
        "recovery_time_s_mean": mean_only([r.recovery_time_s for r in reports]),
    }


def _verdict(reference: dict[str, float | None], contender: dict[str, float | None], threshold: float = 1.0) -> str:
    base_wait = reference["avg_wait_mean"]
    cont_wait = contender["avg_wait_mean"]
    delta = (base_wait - cont_wait) / base_wait * 100 if base_wait else 0.0
    if delta > threshold:
        return "CONTENDER_BETTER"
    if delta < -threshold:
        return "REFERENCE_BETTER"
    return "TIED"


def _reg_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b) (Numerical Recipes).

    Used for Student-t tail probabilities without pulling in a scipy dep.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    def _gammln(xx: float) -> float:
        cof = [76.18009172947146, -86.50532032941677, 24.01409824083091,
               -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5]
        y = xx
        tmp = y + 5.5 - (y + 0.5) * math.log(y + 5.5)
        ser = 1.000000000190015
        for j in range(6):
            ser += cof[j] / (y + 1)
            y += 1
        return -tmp + math.log(2.5066282746310005 * ser / xx)

    def _betacf(aa: float, bb: float, xx: float) -> float:
        maxit, eps = 1000, 3.0e-12
        qab, qap, qam = aa + bb, aa + 1.0, aa - 1.0
        c, d, m2 = 1.0, 1.0 - qab * xx / qap, 0.0
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        h = d
        for m in range(1, maxit):
            m2 += 2
            aa_ = m * (bb - m) * xx / ((qam + m2) * (aa + m2))
            d = 1.0 + aa_ * d
            if abs(d) < 1e-30:
                d = 1e-30
            c = 1.0 + aa_ / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            h *= d * c
            aa_ = -(aa + m) * (qab + m) * xx / ((aa + m2) * (qap + m2))
            d = 1.0 + aa_ * d
            if abs(d) < 1e-30:
                d = 1e-30
            c = 1.0 + aa_ / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < eps and m > 3:
                break
        return h

    bt = math.exp(_gammln(a + b) - _gammln(a) - _gammln(b)
                  + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_two_tailed_p(t: float, df: int) -> float:
    """Two-sided p-value for a Student-t statistic with `df` degrees of freedom.

    Uses F(t) = 1 - (1/2) I_{df/(df+t^2)}(df/2, 1/2), so p = 2(1 - F(|t|)).
    """
    if not math.isfinite(t):
        return 0.0
    x = df / (df + t * t)
    return _reg_beta(df / 2.0, 0.5, x)


def _t_crit_95(df: int) -> float:
    """Two-tailed 95% critical value t(df, 0.975) via bisection on the CDF."""
    lo, hi = 0.0, 1.0
    while _t_two_tailed_p(hi, df) > 0.05:
        hi *= 2.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if _t_two_tailed_p(mid, df) > 0.05:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _wilcoxon_p(diffs: list[float]) -> float | None:
    """Two-sided Wilcoxon signed-rank p (normal approx. with tie correction).

    Returns None when fewer than 5 nonzero pairs, the practical minimum for
    the approximation to hold.
    """
    nonzero = [d for d in diffs if d != 0.0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    if n < 5:
        return None
    order = sorted(range(n), key=lambda i: abs(nonzero[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(abs(nonzero[order[j + 1]]) - abs(nonzero[order[i]])) < 1e-12:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    w_plus = sum(ranks[i] for i in range(n) if nonzero[i] > 0)
    mu = n * (n + 1) / 4.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0
    for r in set(ranks):
        g = ranks.count(r)
        variance -= g * (g - 1) * (g + 1) / 48.0
    if w_plus == mu or variance <= 0.0:
        return 1.0
    z = (w_plus - mu) / math.sqrt(max(variance, 1e-12))
    return 2.0 * (1.0 - _norm_cdf(abs(z)))


def paired_significance(
    reference: dict, contender: dict,
) -> dict | None:
    """Paired two-sided t-test, Wilcoxon signed-rank, 95% CI, and Cohen's dz.

    Uses the per-seed `avg_wait_s` series stored for each controller in the
    benchmark artifact (matched by seed order). Returns None when the series
    are missing or of unequal length.
    """
    base = reference.get("avg_wait_s")
    cont = contender.get("avg_wait_s")
    if not base or not cont:
        return None
    base = [float(v) for v in base]
    cont = [float(v) for v in cont]
    if len(base) != len(cont) or len(base) < 2:
        return None
    n = len(base)
    diffs = [b - c for b, c in zip(base, cont, strict=True)]
    md = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    df = n - 1
    if sd > 0:
        t = md / (sd / math.sqrt(n))
        p = _t_two_tailed_p(abs(t), df)
        tc = _t_crit_95(df)
        half = tc * sd / math.sqrt(n)
        dz = md / sd
    else:
        half = 0.0
        if md == 0:
            p, dz = 1.0, 0.0
        else:
            p, dz = 0.0, math.inf
    return {
        "seeds": n,
        "mean_diff_s": round(md, 2),
        "ci95_s": [round(md - half, 2), round(md + half, 2)],
        "p": round(p, 5),
        "wilcoxon_p": round(_wilcoxon_p(diffs), 5) if _wilcoxon_p(diffs) is not None else None,
        "cohen_dz": round(dz, 2),
    }


def report_significance(results: dict, reference: str = "baseline", contender: str = "adaptive") -> str:
    """Human-readable matched-test summary for reference vs contender."""
    lines = []
    for name in sorted(results):
        block = results[name]
        if reference not in block or contender not in block:
            continue
        st = paired_significance(block[reference], block[contender])
        if st is None:
            continue
        ci = st["ci95_s"]
        wilcoxon = f"  W={st['wilcoxon_p']:.5f}" if st["wilcoxon_p"] is not None else ""
        lines.append(
            f"{name:<12} paired t({st['seeds'] - 1}) d={st['mean_diff_s']:>6.2f}s "
            f"95% CI [{ci[0]:>6.2f},{ci[1]:>6.2f}]  p={st['p']:.5f}{wilcoxon}  dz={st['cohen_dz']:>4.1f}"
        )
    return "\n".join(lines) if lines else ""


def _format_emergency_mean(rep: dict[str, float | None]) -> str:
    """Human-readable emergency corridor timing line, if the scenario has one."""
    fields = (
        ("time_to_preemption_s_mean", "preempt"),
        ("corridor_clearance_s_mean", "clear"),
        ("recovery_time_s_mean", "recover"),
    )
    pieces = []
    for key, label in fields:
        value = rep.get(key)
        if value is not None:
            pieces.append(f"{label}={value}s")
    return "EMERG: " + "  ".join(pieces) if pieces else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P1-B multi-scenario benchmark")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 72)))
    parser.add_argument("--service-rate", type=float, default=1.0)
    parser.add_argument("--scenario", default=None, help="Run only this scenario (default: all)")
    parser.add_argument("--out", default="profiling/artifacts/benchmark_results.json")
    parser.add_argument(
        "--baseline-green",
        type=int,
        default=13,
        help="fixed-time baseline green per approach in seconds (tuned default 13)",
    )
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=["baseline", "adaptive"],
        choices=["baseline", "adaptive", "actuated", "fusion", "rl", "rl_special"],
        help="controllers to benchmark (verdict is only reported when the "
        "first two are 'baseline' and 'adaptive'; 'rl' = held-out-medium "
        "generalist, 'rl_special' = per-scenario specialist weights)",
    )
    parser.add_argument(
        "--rl-weights",
        default="profiling/artifacts/rl_policy.pt",
        help="trained DQN weights (torch state dict) for --controllers rl",
    )
    parser.add_argument(
        "--rl-specialist-dir",
        default="profiling/artifacts/rl_specialists",
        help="directory of per-scenario DQN weights for --controllers rl_special "
        "(expects rl_<scenario>.pt for every benchmarked scenario)",
    )
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="delta % at which adaptive is declared the winner")
    # --- experimental variants ---
    parser.add_argument("--demand-model", default="poisson",
                        choices=["poisson", "platoon", "lognormal"],
                        help="arrival process (platoon = correlated bursts; "
                        "lognormal = log-normal headway platoons)")
    parser.add_argument("--headway-cv", type=float, default=2.0,
                        help="log-normal headway coefficient of variation (lognormal demand)")
    parser.add_argument("--platoon-on", type=int, default=8, help="platoon active window, seconds")
    parser.add_argument("--platoon-off", type=int, default=12, help="platoon idle gap, seconds")
    parser.add_argument("--corridor-sweep", action="store_true",
                        help="run the emergency scenario for every approach corridor")
    parser.add_argument("--corridor-out", default="profiling/artifacts/corridor_results.json",
                        help="artifact path for --corridor-sweep")
    parser.add_argument("--export-csv", default=None, metavar="PATH",
                        help="also write a long-form per-seed CSV to PATH")
    # --- ablation sweep ---
    parser.add_argument("--ablate", default=None, choices=sorted(ABLATION_PARAMS),
                        help="sweep a controller parameter (writes ablation_results.json)")
    parser.add_argument("--ablate-values", type=float, nargs="+",
                        help="values to sweep for --ablate (default = param-specific)")
    parser.add_argument("--ablate-scenario", default="medium",
                        choices=sorted(SCENARIOS), help="scenario for the ablation sweep")
    parser.add_argument("--ablate-seeds", type=int, nargs="+", default=[42, 44],
                        help="seeds for the ablation sweep")
    parser.add_argument("--ablate-out", default="profiling/artifacts/ablation_results.json")
    return parser.parse_args()


def _ablate_default_values(param: str) -> list[float]:
    defaults = {
        "switch_gap_relative": [0.1, 0.3, 0.5],
        "gap_clear": [2.0, 3.0, 5.0],
        "mg_min": [5, 10, 15],
        "mg_max": [30, 60, 90],
        "gap_early_exit": [0.5, 0.7, 0.9],
        "baseline_green": [10, 13, 16, 20],
        "fusion_weight": [0.5, 1.0, 2.0],
    }
    return defaults[param]


def run_ablation(args: argparse.Namespace) -> int:
    mode, _attr = ABLATION_PARAMS[args.ablate]
    values = args.ablate_values or _ablate_default_values(args.ablate)

    ctrl_mode_label = mode.replace("_", " ").title()
    print(f"Ablation '{args.ablate}' on {ctrl_mode_label} | scenario={args.ablate_scenario} "
          f"seeds={args.ablate_seeds}\n")

    reference = _average(
        args.ablate_scenario, "baseline", args.steps, args.ablate_seeds,
        args.service_rate, args.baseline_green,
    )
    base_wait = reference["avg_wait_mean"]

    rows: list[dict] = []
    header = f"{'value':>10} {'avgWait':>10} {'+/-':>8} {'d_vs_baseline':>14}"
    print(header)
    print("-" * len(header))
    for value in values:
        rep = _average(
            args.ablate_scenario, mode, args.steps, args.ablate_seeds,
            args.service_rate, args.baseline_green,
            ablate=args.ablate, ablate_value=value,
        )
        wait = rep["avg_wait_mean"]
        d = (base_wait - wait) / base_wait * 100 if base_wait else 0.0
        rows.append({
            "param": args.ablate,
            "value": value,
            "controller": mode,
            "scenario": args.ablate_scenario,
            "seeds": args.ablate_seeds,
            "steps": args.steps,
            "avg_wait_mean": wait,
            "avg_wait_std": rep["avg_wait_std"],
            "max_queue_mean": rep["max_queue_mean"],
            "avg_queue_mean": rep["avg_queue_mean"],
            "vehicles_served_mean": rep["vehicles_served_mean"],
            "baseline_avg_wait_mean": base_wait,
            "delta_vs_baseline_pct": round(d, 2),
        })
        print(f"{value:>10g} {wait:>10.1f} +/- {rep['avg_wait_std']:>6.1f} "
              f"{d:+12.2f}%")

    out_path = (ROOT / args.ablate_out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "p1b_ablation": args.ablate,
                "baseline_green_s": args.baseline_green,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote ablation artifact to {out_path}")
    return 0


def run_benchmark(
    scenarios,
    steps: int,
    seeds,
    service_rate: float,
    green_seconds: int,
    controllers: list[str] = ("baseline", "adaptive"),
    threshold: float = 1.0,
    rl_weights: str | None = None,
    demand_model: str = "poisson",
    platoon_on_s: int = 8,
    platoon_off_s: int = 12,
    rl_specialist_dir: str | None = None,
    headway_cv: float = 1.2,
) -> dict:
    """Run every scenario x controllers over seeds, aggregated."""
    results: dict[str, dict] = {}
    for name in scenarios:
        block: dict = {}
        for ctrl in controllers:
            block[ctrl] = _average(
                name, ctrl, steps, seeds, service_rate, green_seconds,
                rl_weights=rl_weights, demand_model=demand_model,
                platoon_on_s=platoon_on_s, platoon_off_s=platoon_off_s,
                rl_specialist_dir=rl_specialist_dir, headway_cv=headway_cv,
            )
        if "baseline" in controllers and "adaptive" in controllers:
            block["verdict"] = _verdict(block["baseline"], block["adaptive"], threshold)
        results[name] = block
    return results


def run_corridor_sweep(
    steps: int,
    seeds,
    service_rate: float,
    green_seconds: int,
    controllers: list[str] = ("baseline", "adaptive"),
    rl_weights: str | None = None,
    demand_model: str = "poisson",
    platoon_on_s: int = 8,
    platoon_off_s: int = 12,
    rl_specialist_dir: str | None = None,
    headway_cv: float = 1.2,
) -> dict:
    """Run the emergency scenario with the ambulance on each approach."""
    from .harness.traffic import LANES

    results: dict[str, dict] = {}
    for corridor in LANES:
        block: dict = {}
        for ctrl in controllers:
            block[ctrl] = _average(
                "emergency", ctrl, steps, seeds, service_rate, green_seconds,
                rl_weights=rl_weights, demand_model=demand_model,
                platoon_on_s=platoon_on_s, platoon_off_s=platoon_off_s,
                corridor=corridor, rl_specialist_dir=rl_specialist_dir,
                headway_cv=headway_cv,
            )
        if "baseline" in controllers and "adaptive" in controllers:
            block["verdict"] = _verdict(block["baseline"], block["adaptive"])
        results[corridor] = block
    return results


def _export_csv(results: dict, seeds, out_path: Path) -> None:
    """Write a long-form per-seed CSV (scenario, controller, seed, avg_wait_s)."""
    import csv

    rows: list[tuple[str, str, int, float]] = []
    for scenario, block in sorted(results.items()):
        for controller, rep in block.items():
            if controller == "verdict":
                continue
            for seed, wait in zip(seeds, rep.get("avg_wait_s") or [], strict=False):
                rows.append((scenario, controller, int(seed), float(wait)))
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scenario", "controller", "seed", "avg_wait_s"])
        for row in sorted(rows, key=lambda r: (r[0], r[1], r[2])):
            writer.writerow(row)


def main() -> int:
    args = parse_args()

    if args.ablate:
        return run_ablation(args)

    if "rl" in args.controllers:
        rl_path = (ROOT / args.rl_weights).resolve()
        if not rl_path.is_file():
            print(f"error: --controllers rl requires trained weights at {rl_path} "
                  f"(run scripts/train_rl_baseline.py first)")
            return 2

    if "rl_special" in args.controllers:
        spec_dir = (ROOT / args.rl_specialist_dir).resolve()
        keys = ["emergency"] if args.corridor_sweep else sorted(SCENARIOS)
        missing = [f"rl_{k}.pt" for k in keys if not (spec_dir / f"rl_{k}.pt").is_file()]
        if missing:
            print(f"error: rl_special requires per-scenario weights in {spec_dir}; missing "
                  f"{', '.join(missing)} (train with scripts/train_rl_baseline.py)")
            return 2

    demand_kwargs = {
        "demand_model": args.demand_model,
        "platoon_on_s": args.platoon_on,
        "platoon_off_s": args.platoon_off,
        "headway_cv": args.headway_cv,
    }

    if args.corridor_sweep:
        results = run_corridor_sweep(
            args.steps, args.seeds, args.service_rate, args.baseline_green,
            args.controllers, args.rl_weights, rl_specialist_dir=args.rl_specialist_dir,
            **demand_kwargs,
        )
        print(f"Corridor sweep ({args.demand_model} demand): emergency corridor x "
              f"{len(args.seeds)} seeds x {args.controllers} at {args.steps} steps\n")
        header = f"{'Corridor':<9} {'Controller':<10} {'AvgWait':>12} {'Verdict':<18}"
        print(header)
        print("-" * len(header))
        for corridor in sorted(results):
            block = results[corridor]
            for ctrl in args.controllers:
                rep = block[ctrl]
                print(f"{corridor:<9} {ctrl:<10} {rep['avg_wait_mean']:>6.1f}s +/- {rep['avg_wait_std']:>4.1f} "
                      f"{block.get('verdict', ''):<18}")
                em = _format_emergency_mean(rep)
                if em:
                    print(f"{'':8} {'':10} {em}")
            print()
        sig = report_significance(results)
        if sig:
            print("Matched significance per corridor:")
            print(sig)
            print()
        out_path = (ROOT / args.corridor_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({
                "p1b": "emergency corridor sweep",
                "steps": args.steps,
                "seeds": args.seeds,
                "service_rate": args.service_rate,
                "baseline_green_s": args.baseline_green,
                "controllers": args.controllers,
                "demand_model": args.demand_model,
                "results": results,
            }, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote corridor sweep artifact to {out_path}")
        if args.export_csv:
            _export_csv(results, args.seeds, (ROOT / args.export_csv).resolve())
            print(f"Wrote per-seed CSV to {args.export_csv}")
        return 0

    scenarios = [args.scenario] if args.scenario else sorted(SCENARIOS.keys())

    results = run_benchmark(
        scenarios,
        args.steps,
        args.seeds,
        args.service_rate,
        args.baseline_green,
        args.controllers,
        args.threshold,
        args.rl_weights,
        rl_specialist_dir=args.rl_specialist_dir,
        **demand_kwargs,
    )

    if args.demand_model in ("platoon", "lognormal") and args.out == "profiling/artifacts/benchmark_results.json":
        args.out = f"profiling/artifacts/{args.demand_model}_results.json"

    print(f"Benchmark ({args.demand_model} demand): {len(scenarios)} scenarios x "
          f"{len(args.seeds)} seeds x {args.controllers} at {args.steps} steps each, "
          f"baseline green={args.baseline_green}s\n")
    header = (
        f"{'Scenario':<12} {'Controller':<10} {'AvgWait':>12} {'MaxQ':>10} "
        f"{'AvgQ':>10} {'Served':>12} {'Verdict':<18}"
    )
    print(header)
    print("-" * len(header))
    print("(mean +/- std over seeds)")
    print("-" * len(header))
    for name in scenarios:
        block = results[name]
        for ctrl in args.controllers:
            rep = block[ctrl]
            print(
                f"{name:<12} {ctrl:<10} "
                f"{rep['avg_wait_mean']:>6.1f}s +/- {rep['avg_wait_std']:>4.1f} "
                f"{rep['max_queue_mean']:>5.1f} (+/- {rep['max_queue_std']:>4.1f}) "
                f"{rep['avg_queue_mean']:>6.1f} (+/- {rep['avg_queue_std']:>4.1f}) "
                f"{rep['vehicles_served_mean']:>7.0f} (+/- {rep['vehicles_served_std']:>4.0f}) "
                f"{block.get('verdict', ''):<18}"
            )
            em = _format_emergency_mean(rep)
            if em:
                print(f"{'':9} {'':10} {em}")
        print()

    sig = report_significance(results)
    if sig:
        print("Matched significance (baseline vs adaptive, per-seed paired t-test):")
        print(sig)
        print()

    out_path = (ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "p1b": "multi-scenario benchmark",
                "steps": args.steps,
                "seeds": args.seeds,
                "service_rate": args.service_rate,
                "baseline_green_s": args.baseline_green,
                "controllers": args.controllers,
                "threshold_pct": args.threshold,
"demand_model": args.demand_model,
                "platoon_on_s": args.platoon_on,
                "platoon_off_s": args.platoon_off,
                "headway_cv": args.headway_cv,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote benchmark artifact to {out_path}")
    if args.export_csv:
        _export_csv(results, args.seeds, (ROOT / args.export_csv).resolve())
        print(f"Wrote per-seed CSV to {args.export_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
