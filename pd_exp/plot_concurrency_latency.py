#!/usr/bin/env python3
"""
Plot concurrency sweep results for the reviewer response.

Generates two figures:
  Figure 1 (TODO 1): Latency and throughput vs. concurrency — line plots showing
      how TTFT, TPOT, and RPS change as concurrency increases.  At low concurrency
      the scheduler's inherent TTFT/TPOT trade-off is clearly visible.

  Figure 2 (TODO 2): SLO-constrained throughput comparison — for multiple SLO
      targets, find the maximum concurrency (≈ throughput) where P99 TTFT and P99
      TPOT stay within budget, then compare schedulers via grouped bar charts.

Usage:
    # Single model
    python pd_exp/plot_concurrency_latency.py \
        pd_exp/outputs/concurrency_sweep_wildchat_Qwen3-8B_h200

    # Multiple models (pass multiple dirs)
    python pd_exp/plot_concurrency_latency.py \
        pd_exp/outputs/concurrency_sweep_wildchat_Qwen3-8B_h200 \
        pd_exp/outputs/concurrency_sweep_wildchat_Qwen3-30B-A3B_h200 \
        pd_exp/outputs/concurrency_sweep_wildchat_gemma-3-1b-it_h200 \
        --output-dir pd_exp/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter


# ── Styling ─────────────────────────────────────────────────────────────
SCHEDULER_DISPLAY = {
    "baseline": "CP (v1)",
    "pd_ratio": r"EB($\theta^{*}\!=\!0.8$)",
    "pd_ifr":   r"THETA",
}

SCHEDULER_COLORS = {
    "baseline": "#B23A3A",   # warm red  (consistent with v1)
    "pd_ratio": "#2A6F97",   # blue
    "pd_ifr":   "#2A6F97",   # blue      (consistent with Ours)
}

SCHEDULER_MARKERS = {
    "baseline": "o",
    "pd_ratio": "s",
    "pd_ifr":   "s",
}

# Only show CP vs THETA for rebuttal (EB(θ*=0.8) ≈ EB(k̂*), omit for clarity)
SCHEDULER_ORDER = ["baseline", "pd_ifr"]


def configure_style() -> None:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.size"] = 13
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 11
    plt.rcParams["axes.formatter.useoffset"] = False
    plt.rcParams["axes.formatter.use_mathtext"] = False


# ── Data loading ────────────────────────────────────────────────────────
def load_results(exp_dir: Path) -> Dict[str, Any]:
    """Load all results from a concurrency sweep experiment directory.

    Returns:
        {
            "model": str,
            "gpu_type": str,
            "config": dict,
            "results": {
                concurrency_int: {
                    scheduler_str: {metric: value, ...}
                }
            }
        }
    """
    config_path = exp_dir / "experiment_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No experiment_config.json in {exp_dir}")

    with open(config_path) as f:
        config = json.load(f)

    model = config.get("model", "unknown")
    model_short = model.split("/")[-1] if "/" in model else model
    gpu_type = config.get("gpu_type", "unknown")

    results: Dict[int, Dict[str, Dict[str, float]]] = {}

    for clients_dir in sorted(exp_dir.glob("clients_*")):
        try:
            num_clients = int(clients_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue

        results[num_clients] = {}

        for bench_file in sorted(clients_dir.glob("bench_*.json")):
            scheduler = bench_file.stem.replace("bench_", "")
            try:
                with open(bench_file) as f:
                    metrics = json.load(f)
                results[num_clients][scheduler] = metrics
            except (json.JSONDecodeError, OSError):
                print(f"  Warning: failed to read {bench_file}")

    return {
        "model": model_short,
        "gpu_type": gpu_type,
        "config": config,
        "results": results,
    }


# ── Figure 1: Latency & throughput vs concurrency ──────────────────────
def plot_latency_vs_concurrency(
    all_data: List[Dict[str, Any]],
    output_path: Path,
    highlight_concurrencies: List[int] = [64, 256],
) -> None:
    """Line plots: TTFT, TPOT, and RPS vs num_clients for each model."""

    n_models = len(all_data)
    metrics_info = [
        ("mean_ttft_ms", "Mean TTFT (s)", 1e-3, False),    # ms → s
        ("p99_ttft_ms",  "P99 TTFT (s)",  1e-3, False),
        ("mean_tpot_ms", "Mean TPOT (ms)", 1.0, False),
        ("p99_tpot_ms",  "P99 TPOT (ms)",  1.0, False),
        ("request_throughput", "Throughput (req/s)", 1.0, True),
    ]
    n_metrics = len(metrics_info)

    fig, axes = plt.subplots(
        n_metrics, n_models,
        figsize=(5 * n_models, 3.2 * n_metrics),
        squeeze=False,
    )

    for col, data in enumerate(all_data):
        results = data["results"]
        model_name = data["model"]
        concurrencies = sorted(results.keys())

        for row, (metric_key, ylabel, scale, higher_better) in enumerate(metrics_info):
            ax = axes[row, col]

            for scheduler in SCHEDULER_ORDER:
                values = []
                xs = []
                for c in concurrencies:
                    if scheduler in results[c] and metric_key in results[c][scheduler]:
                        xs.append(c)
                        values.append(results[c][scheduler][metric_key] * scale)

                if not values:
                    continue

                label = SCHEDULER_DISPLAY.get(scheduler, scheduler)
                ax.plot(
                    xs, values,
                    color=SCHEDULER_COLORS.get(scheduler, "gray"),
                    marker=SCHEDULER_MARKERS.get(scheduler, "o"),
                    markersize=6,
                    linewidth=2,
                    label=label,
                    zorder=3,
                )

            # Highlight specific concurrencies with vertical bands
            for hc in highlight_concurrencies:
                if hc in [int(x) for x in concurrencies]:
                    ax.axvline(
                        hc, color="#CCCCCC", linestyle="--",
                        linewidth=1, zorder=1,
                    )

            ax.set_xscale("log", base=2)
            ax.set_xticks(concurrencies)
            ax.set_xticklabels([str(c) for c in concurrencies], rotation=45, ha="right")
            ax.grid(True, alpha=0.3, zorder=0)

            if row == 0:
                ax.set_title(model_name, fontsize=14, fontweight="bold")
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == n_metrics - 1:
                ax.set_xlabel("Number of Clients")
            if row == 0 and col == n_models - 1:
                ax.legend(loc="upper left", framealpha=0.9)

    fig.suptitle(
        f"Latency and Throughput vs. Concurrency (WildChat, {all_data[0]['gpu_type'].upper()})",
        fontsize=15, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    for fmt in ["pdf", "png"]:
        out = output_path.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)


# ── Figure 2: SLO-constrained throughput ────────────────────────────────
def find_max_slo_compliant(
    results: Dict[int, Dict[str, Dict[str, float]]],
    scheduler: str,
    ttft_p99_limit_s: float,
    tpot_p99_limit_ms: float,
) -> Tuple[Optional[int], Optional[float]]:
    """Find the maximum num_clients where P99 TTFT ≤ limit AND P99 TPOT ≤ limit.

    Returns (max_clients, throughput_at_max_clients) or (None, None).
    """
    best_clients = None
    best_throughput = None

    for c in sorted(results.keys()):
        if scheduler not in results[c]:
            continue
        m = results[c][scheduler]
        p99_ttft_s = m.get("p99_ttft_ms", float("inf")) / 1000.0
        p99_tpot_ms = m.get("p99_tpot_ms", float("inf"))

        if p99_ttft_s <= ttft_p99_limit_s and p99_tpot_ms <= tpot_p99_limit_ms:
            best_clients = c
            best_throughput = m.get("request_throughput", 0)

    return best_clients, best_throughput


def plot_slo_constrained_throughput(
    all_data: List[Dict[str, Any]],
    output_path: Path,
    slo_targets: Optional[List[Tuple[str, float, float]]] = None,
) -> None:
    """Bar chart: max SLO-compliant throughput per scheduler, per SLO target.

    slo_targets: list of (name, ttft_p99_s, tpot_p99_ms)
    """
    if slo_targets is None:
        slo_targets = [
            ("Strict\n(TTFT<2s, TPOT<50ms)",    2.0,   50.0),
            ("Moderate\n(TTFT<5s, TPOT<100ms)",  5.0,  100.0),
            ("Relaxed\n(TTFT<10s, TPOT<200ms)", 10.0,  200.0),
            ("Loose\n(TTFT<30s, TPOT<300ms)",   30.0,  300.0),
        ]

    n_models = len(all_data)
    n_slos = len(slo_targets)
    n_schedulers = len(SCHEDULER_ORDER)

    fig, axes = plt.subplots(
        1, n_models,
        figsize=(5 * n_models, 4.5),
        squeeze=False,
    )

    bar_width = 0.8 / n_schedulers

    for col, data in enumerate(all_data):
        ax = axes[0, col]
        results = data["results"]
        model_name = data["model"]

        x = np.arange(n_slos)

        for i, scheduler in enumerate(SCHEDULER_ORDER):
            throughputs = []
            for slo_name, ttft_limit, tpot_limit in slo_targets:
                _, throughput = find_max_slo_compliant(
                    results, scheduler, ttft_limit, tpot_limit,
                )
                throughputs.append(throughput if throughput is not None else 0)

            label = SCHEDULER_DISPLAY.get(scheduler, scheduler)
            bars = ax.bar(
                x + i * bar_width - (n_schedulers - 1) * bar_width / 2,
                throughputs,
                bar_width * 0.9,
                color=SCHEDULER_COLORS.get(scheduler, "gray"),
                label=label,
                zorder=3,
            )

            # Add value labels on bars
            for bar, val in zip(bars, throughputs):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3,
                        f"{val:.1f}",
                        ha="center", va="bottom",
                        fontsize=8, fontweight="bold",
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [name for name, _, _ in slo_targets],
            fontsize=9,
        )
        ax.set_title(model_name, fontsize=14, fontweight="bold")
        ax.set_ylabel("Max SLO-Compliant\nThroughput (req/s)")
        ax.grid(True, axis="y", alpha=0.3, zorder=0)
        if col == n_models - 1:
            ax.legend(loc="upper left", framealpha=0.9)

    fig.suptitle(
        f"SLO-Constrained Maximum Throughput (WildChat, {all_data[0]['gpu_type'].upper()})",
        fontsize=15, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    for fmt in ["pdf", "png"]:
        out = output_path.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)


# ── Figure 3: Moderate-concurrency bar chart ────────────────────────────
def plot_moderate_concurrency_bars(
    all_data: List[Dict[str, Any]],
    output_path: Path,
    focus_concurrencies: List[int] = [64, 256, 2048],
) -> None:
    """Grouped bar chart at select concurrency levels — the clearest way to
    show the latency trade-off when queuing delay is minimal (64, 256)
    vs. heavy (2048).
    """
    n_models = len(all_data)
    metrics_info = [
        ("request_throughput", "Throughput\n(req/s)", 1.0),
        ("mean_ttft_ms",       "Mean TTFT\n(s)",     1e-3),
        ("mean_tpot_ms",       "Mean TPOT\n(ms)",    1.0),
    ]
    n_metrics = len(metrics_info)
    n_conc = len(focus_concurrencies)
    n_schedulers = len(SCHEDULER_ORDER)

    fig, axes = plt.subplots(
        n_metrics, n_models,
        figsize=(4.5 * n_models, 3 * n_metrics),
        squeeze=False,
    )

    bar_width = 0.8 / n_schedulers

    for col, data in enumerate(all_data):
        results = data["results"]
        model_name = data["model"]

        # Filter to available concurrencies
        available = [c for c in focus_concurrencies if c in results]
        x = np.arange(len(available))

        for row, (metric_key, ylabel, scale) in enumerate(metrics_info):
            ax = axes[row, col]

            for i, scheduler in enumerate(SCHEDULER_ORDER):
                values = []
                for c in available:
                    if scheduler in results[c] and metric_key in results[c][scheduler]:
                        values.append(results[c][scheduler][metric_key] * scale)
                    else:
                        values.append(0)

                label = SCHEDULER_DISPLAY.get(scheduler, scheduler)
                bars = ax.bar(
                    x + i * bar_width - (n_schedulers - 1) * bar_width / 2,
                    values,
                    bar_width * 0.9,
                    color=SCHEDULER_COLORS.get(scheduler, "gray"),
                    label=label if row == 0 else None,
                    zorder=3,
                )

                # Value labels
                for bar, val in zip(bars, values):
                    if val > 0:
                        label_text = f"{val:.1f}" if val >= 1 else f"{val:.2f}"
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height(),
                            label_text,
                            ha="center", va="bottom",
                            fontsize=7,
                        )

            ax.set_xticks(x)
            ax.set_xticklabels([str(c) for c in available])
            ax.grid(True, axis="y", alpha=0.3, zorder=0)

            if row == 0:
                ax.set_title(model_name, fontsize=14, fontweight="bold")
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == n_metrics - 1:
                ax.set_xlabel("Number of Clients")
            if row == 0 and col == n_models - 1:
                ax.legend(loc="best", framealpha=0.9)

    fig.suptitle(
        f"Latency Trade-off at Moderate Concurrency (WildChat, {all_data[0]['gpu_type'].upper()})",
        fontsize=15, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    for fmt in ["pdf", "png"]:
        out = output_path.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)


# ── Summary table ───────────────────────────────────────────────────────
def print_summary_table(all_data: List[Dict[str, Any]]) -> None:
    """Print a text summary of key metrics at each concurrency level."""
    for data in all_data:
        model = data["model"]
        results = data["results"]
        print(f"\n{'='*80}")
        print(f"Model: {model}  |  GPU: {data['gpu_type']}")
        print(f"{'='*80}")
        print(f"{'Clients':>8} {'Scheduler':<12} {'RPS':>8} "
              f"{'TTFT_mean':>10} {'TTFT_p99':>10} "
              f"{'TPOT_mean':>10} {'TPOT_p99':>10}")
        print(f"{'':>8} {'':>12} {'(req/s)':>8} "
              f"{'(s)':>10} {'(s)':>10} "
              f"{'(ms)':>10} {'(ms)':>10}")
        print("-" * 80)

        for c in sorted(results.keys()):
            for scheduler in SCHEDULER_ORDER:
                if scheduler not in results[c]:
                    continue
                m = results[c][scheduler]
                rps = m.get("request_throughput", 0)
                ttft_mean = m.get("mean_ttft_ms", 0) / 1000
                ttft_p99 = m.get("p99_ttft_ms", 0) / 1000
                tpot_mean = m.get("mean_tpot_ms", 0)
                tpot_p99 = m.get("p99_tpot_ms", 0)
                display = SCHEDULER_DISPLAY.get(scheduler, scheduler)
                print(f"{c:>8} {display:<12} {rps:>8.2f} "
                      f"{ttft_mean:>10.2f} {ttft_p99:>10.2f} "
                      f"{tpot_mean:>10.2f} {tpot_p99:>10.2f}")
            if c != max(results.keys()):
                print()


# ── Main ────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot concurrency sweep results (TODO 1 & TODO 2)."
    )
    parser.add_argument(
        "exp_dirs", nargs="+", type=Path,
        help="Experiment output directories "
             "(e.g., pd_exp/outputs/concurrency_sweep_wildchat_Qwen3-8B_h200)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for figures (default: first exp_dir parent / figures)",
    )
    parser.add_argument(
        "--focus-concurrencies", type=int, nargs="+", default=[64, 256, 2048],
        help="Concurrency levels for the moderate-concurrency bar chart",
    )
    parser.add_argument(
        "--slo-ttft", type=float, nargs="+", default=[2.0, 5.0, 10.0, 30.0],
        help="TTFT P99 SLO targets in seconds",
    )
    parser.add_argument(
        "--slo-tpot", type=float, nargs="+", default=[50.0, 100.0, 200.0, 300.0],
        help="TPOT P99 SLO targets in milliseconds",
    )
    parser.add_argument(
        "--no-summary", action="store_true",
        help="Skip printing summary table",
    )
    args = parser.parse_args()

    configure_style()

    # Load all experiment data
    all_data = []
    for exp_dir in args.exp_dirs:
        print(f"Loading: {exp_dir}")
        data = load_results(exp_dir)
        n_conc = len(data["results"])
        n_sched = max(len(v) for v in data["results"].values()) if data["results"] else 0
        print(f"  Model: {data['model']}, GPU: {data['gpu_type']}, "
              f"concurrencies: {n_conc}, schedulers: {n_sched}")
        all_data.append(data)

    if not all_data:
        print("No data loaded!")
        return

    # Output directory
    output_dir = args.output_dir or (args.exp_dirs[0].parent / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_type = all_data[0]["gpu_type"]

    # Summary table
    if not args.no_summary:
        print_summary_table(all_data)

    # Figure 1: latency vs concurrency (line plots)
    print("\n--- Figure 1: Latency vs Concurrency ---")
    plot_latency_vs_concurrency(
        all_data,
        output_dir / f"concurrency_latency_lines_{gpu_type}",
    )

    # Figure 2: moderate concurrency bar chart
    print("\n--- Figure 2: Moderate Concurrency Bars ---")
    plot_moderate_concurrency_bars(
        all_data,
        output_dir / f"moderate_concurrency_bars_{gpu_type}",
        focus_concurrencies=args.focus_concurrencies,
    )

    # Figure 3: SLO-constrained throughput
    print("\n--- Figure 3: SLO-Constrained Throughput ---")
    slo_targets = []
    slo_names = ["Strict", "Moderate", "Relaxed", "Loose"]
    for i, (ttft, tpot) in enumerate(zip(args.slo_ttft, args.slo_tpot)):
        name = slo_names[i] if i < len(slo_names) else f"SLO-{i+1}"
        name += f"\n(TTFT<{ttft:.0f}s, TPOT<{tpot:.0f}ms)"
        slo_targets.append((name, ttft, tpot))

    plot_slo_constrained_throughput(
        all_data,
        output_dir / f"slo_constrained_throughput_{gpu_type}",
        slo_targets=slo_targets,
    )

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
