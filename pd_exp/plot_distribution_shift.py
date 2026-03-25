#!/usr/bin/env python3
"""
Plot distribution shift experiment results.

Reads schedule stats from pd_ifr and pd_ratio runs, and generates:
  - Panel 1: θ* (k_ratio) over iteration — shows IFR adaptation vs fixed ratio
  - Panel 2: Sliding-window throughput (tokens/s) over time
  - Panel 3: Preemption count (memory safety indicator)

Usage:
    python pd_exp/plot_distribution_shift.py \
        pd_exp/outputs/distribution_shift_Qwen3-8B_20260325_120000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCHEDULER_DISPLAY = {
    "pd_ifr": "THETA (adaptive)",
    "pd_ratio": r"THETA (fixed $\theta^{*}\!=\!0.8$)",
}

SCHEDULER_COLORS = {
    "pd_ifr": "#2A6F97",
    "pd_ratio": "#B23A3A",
}


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


def load_stats(stats_path: Path) -> List[Dict[str, Any]]:
    """Load schedule stats JSON, handling possibly truncated files."""
    with open(stats_path) as f:
        content = f.read().strip()

    # Try parsing as-is
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "stats" in data:
            return data["stats"]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Try repairing truncated JSON (missing closing brackets)
    for suffix in ["]}", "]", "]}]}}"]:
        try:
            data = json.loads(content + suffix)
            if isinstance(data, dict) and "stats" in data:
                return data["stats"]
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Cannot parse stats file: {stats_path}")


def compute_sliding_throughput(
    stats: List[Dict[str, Any]],
    window_sec: float = 5.0,
) -> tuple[List[float], List[float]]:
    """Compute sliding-window throughput in tokens/s."""
    if not stats:
        return [], []

    timestamps = [s.get("timestamp", 0) for s in stats]
    tokens = [s.get("total_tokens", 0) for s in stats]

    t_out = []
    thr_out = []

    for i, t in enumerate(timestamps):
        # Find window start
        j = i
        while j > 0 and timestamps[j] > t - window_sec:
            j -= 1
        window_tokens = sum(tokens[j:i + 1])
        window_time = timestamps[i] - timestamps[j] if i > j else window_sec
        if window_time > 0:
            t_out.append(t)
            thr_out.append(window_tokens / window_time)

    return t_out, thr_out


def estimate_shift_iteration(
    stats: List[Dict[str, Any]],
    num_prompts_per_phase: int,
) -> Optional[int]:
    """Estimate the iteration index where the distribution shift occurs.

    Heuristic: track cumulative new requests scheduled.  The shift happens
    around the iteration where cumulative new_reqs ≈ num_prompts_per_phase.
    """
    cum_new = 0
    for i, s in enumerate(stats):
        cum_new += s.get("num_new_reqs", 0)
        if cum_new >= num_prompts_per_phase:
            return i
    return None


def plot_distribution_shift(
    exp_dir: Path,
    output_path: Path,
    window_sec: float = 5.0,
) -> None:
    """Generate the 3-panel distribution shift figure."""

    config_path = exp_dir / "experiment_config.json"
    with open(config_path) as f:
        config = json.load(f)

    num_prompts_per_phase = config.get("num_prompts_per_phase", 2000)
    window_size = config.get("ifr_window_size", 500)

    schedulers = ["pd_ifr", "pd_ratio"]
    all_stats = {}
    for sched in schedulers:
        stats_path = exp_dir / f"{sched}_stats.json"
        if stats_path.exists():
            all_stats[sched] = load_stats(stats_path)
            print(f"  {sched}: {len(all_stats[sched])} iterations")
        else:
            print(f"  {sched}: stats file not found, skipping")

    if not all_stats:
        print("No stats data found!")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

    # ── Panel 1: θ* (k_ratio) over iteration ──
    ax1 = axes[0]
    for sched, stats in all_stats.items():
        k_ratios = [s.get("k_ratio", None) for s in stats]
        iters = list(range(len(k_ratios)))

        # Filter out None values
        valid = [(i, k) for i, k in zip(iters, k_ratios) if k is not None]
        if valid:
            xi, yk = zip(*valid)
            label = SCHEDULER_DISPLAY.get(sched, sched)
            ax1.plot(
                xi, yk,
                color=SCHEDULER_COLORS.get(sched, "gray"),
                linewidth=1.5, alpha=0.8,
                label=label,
            )

    # Mark shift point
    shift_iter = None
    for sched, stats in all_stats.items():
        shift_iter = estimate_shift_iteration(stats, num_prompts_per_phase)
        if shift_iter is not None:
            break

    if shift_iter is not None:
        ax1.axvline(
            shift_iter, color="black", linestyle="--",
            linewidth=1.5, alpha=0.7, label="Distribution shift",
        )

    ax1.set_ylabel(r"$\theta^{*}$ (k_ratio)")
    ax1.set_title(
        r"$\theta^{*}$ Convergence Under Distribution Shift"
        f" (W={window_size})",
        fontweight="bold",
    )
    ax1.legend(loc="best", framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Sliding-window throughput ──
    ax2 = axes[1]
    for sched, stats in all_stats.items():
        t_vals, thr_vals = compute_sliding_throughput(stats, window_sec)
        if t_vals:
            label = SCHEDULER_DISPLAY.get(sched, sched)
            ax2.plot(
                t_vals, thr_vals,
                color=SCHEDULER_COLORS.get(sched, "gray"),
                linewidth=1.5, alpha=0.8,
                label=label,
            )

    # Mark shift time
    if shift_iter is not None and shift_iter < len(list(all_stats.values())[0]):
        first_stats = list(all_stats.values())[0]
        shift_time = first_stats[shift_iter].get("timestamp", 0)
        ax2.axvline(
            shift_time, color="black", linestyle="--",
            linewidth=1.5, alpha=0.7,
        )

    ax2.set_ylabel("Throughput (tokens/s)")
    ax2.set_title("Sliding-Window Throughput", fontweight="bold")
    ax2.legend(loc="best", framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Cumulative preemptions (memory safety) ──
    ax3 = axes[2]
    for sched, stats in all_stats.items():
        preemptions = [s.get("num_preempted_reqs", 0) for s in stats]
        cum_preempt = np.cumsum(preemptions)
        iters = list(range(len(cum_preempt)))

        label = SCHEDULER_DISPLAY.get(sched, sched)
        ax3.plot(
            iters, cum_preempt,
            color=SCHEDULER_COLORS.get(sched, "gray"),
            linewidth=1.5, alpha=0.8,
            label=label,
        )

    if shift_iter is not None:
        ax3.axvline(
            shift_iter, color="black", linestyle="--",
            linewidth=1.5, alpha=0.7,
        )

    ax3.set_xlabel("Scheduling Iteration")
    ax3.set_ylabel("Cumulative Preemptions")
    ax3.set_title("Memory Safety (Preemptions)", fontweight="bold")
    ax3.legend(loc="best", framealpha=0.9)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()

    for fmt in ["pdf", "png"]:
        out = output_path.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)


def print_summary(exp_dir: Path) -> None:
    """Print convergence summary statistics."""
    config_path = exp_dir / "experiment_config.json"
    with open(config_path) as f:
        config = json.load(f)

    num_prompts_per_phase = config.get("num_prompts_per_phase", 2000)

    for sched in ["pd_ifr", "pd_ratio"]:
        stats_path = exp_dir / f"{sched}_stats.json"
        if not stats_path.exists():
            continue

        stats = load_stats(stats_path)
        display = SCHEDULER_DISPLAY.get(sched, sched)

        # Find shift point
        shift_iter = estimate_shift_iteration(stats, num_prompts_per_phase)

        # Compute k_ratio stats before/after shift
        k_ratios = [s.get("k_ratio", None) for s in stats]
        k_ratios = [k for k in k_ratios if k is not None]

        if shift_iter and shift_iter < len(k_ratios):
            before = k_ratios[:shift_iter]
            after = k_ratios[shift_iter:]

            # Find convergence: when k_ratio stabilizes after shift
            # (within 5% of final value for 50 consecutive iterations)
            final_value = np.mean(after[-100:]) if len(after) > 100 else np.mean(after)
            convergence_iter = None
            for i in range(len(after)):
                if i + 50 <= len(after):
                    window = after[i:i + 50]
                    if all(abs(k - final_value) / max(final_value, 0.01) < 0.05
                           for k in window):
                        convergence_iter = i
                        break

            print(f"\n{display}:")
            print(f"  Shift at iteration: {shift_iter}")
            print(f"  θ* before shift: {np.mean(before):.4f} "
                  f"(std={np.std(before):.4f})")
            print(f"  θ* after shift (final): {final_value:.4f}")
            if convergence_iter is not None:
                print(f"  Convergence iterations after shift: {convergence_iter}")
            else:
                print(f"  Convergence: not reached within {len(after)} iterations")

            # Preemption stats
            preemptions = sum(s.get("num_preempted_reqs", 0) for s in stats)
            print(f"  Total preemptions: {preemptions}")
        else:
            print(f"\n{display}: shift point not detected")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot distribution shift experiment results."
    )
    parser.add_argument(
        "exp_dir", type=Path,
        help="Experiment output directory",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for figures (default: exp_dir parent / figures)",
    )
    parser.add_argument(
        "--window-sec", type=float, default=5.0,
        help="Sliding window size for throughput (seconds)",
    )
    args = parser.parse_args()

    configure_style()

    print(f"Loading: {args.exp_dir}")
    output_dir = args.output_dir or (args.exp_dir.parent / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print_summary(args.exp_dir)

    print("\n--- Generating figure ---")
    plot_distribution_shift(
        args.exp_dir,
        output_dir / "distribution_shift_convergence",
        window_sec=args.window_sec,
    )

    print(f"\nFigures saved to: {output_dir}")


if __name__ == "__main__":
    main()
