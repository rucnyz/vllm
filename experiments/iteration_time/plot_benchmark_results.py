"""
Plot benchmark results from benchmark_batch_combinations.py output.

Generates a figure with three curves:
1. Pure prefill: time vs prefill token count
2. Pure decode: time vs batch size (decode count)
3. Mixed batch: time vs total token count

Usage:
    python plot_benchmark_results.py --input results.json --output benchmark_plot.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(filepath: str) -> dict:
    """Load benchmark results from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def extract_data(results: list[dict]) -> tuple[dict, dict, dict, dict]:
    """
    Extract and organize data for plotting.

    Returns:
        (pure_prefill, pure_decode, mixed, mixed_by_decode)
        - pure_prefill, pure_decode, mixed: dicts with 'x', 'y', 'yerr' arrays
        - mixed_by_decode: dict mapping decode_count -> {x, y, yerr, prefill_sizes}
    """
    pure_prefill = {"x": [], "y": [], "yerr": []}
    pure_decode = {"x": [], "y": [], "yerr": []}
    mixed = {
        "x": [], "y": [], "yerr": [], "labels": [],
        "num_decode": [], "prefill_size": []
    }
    # Group mixed by decode count for slope analysis
    mixed_by_decode: dict[int, dict] = {}

    for r in results:
        num_decode = r["num_decode"]
        num_prefill = r["num_prefill"]
        prefill_size = r["prefill_chunk_size"]
        total_tokens = r["total_tokens"]
        mean_time = r["mean_time_ms"]
        std_time = r["std_time_ms"]

        if num_decode == 0 and num_prefill == 1:
            # Pure prefill
            pure_prefill["x"].append(prefill_size)
            pure_prefill["y"].append(mean_time)
            pure_prefill["yerr"].append(std_time)
        elif num_decode > 0 and num_prefill == 0:
            # Pure decode
            pure_decode["x"].append(num_decode)
            pure_decode["y"].append(mean_time)
            pure_decode["yerr"].append(std_time)
        elif num_decode > 0 and num_prefill >= 1:
            # Mixed batch
            mixed["x"].append(total_tokens)
            mixed["y"].append(mean_time)
            mixed["yerr"].append(std_time)
            mixed["labels"].append(f"{num_decode}D{prefill_size}P")
            mixed["num_decode"].append(num_decode)
            mixed["prefill_size"].append(prefill_size)

            # Group by decode count
            if num_decode not in mixed_by_decode:
                mixed_by_decode[num_decode] = {
                    "x": [], "y": [], "yerr": [], "prefill_sizes": []
                }
            mixed_by_decode[num_decode]["x"].append(total_tokens)
            mixed_by_decode[num_decode]["y"].append(mean_time)
            mixed_by_decode[num_decode]["yerr"].append(std_time)
            mixed_by_decode[num_decode]["prefill_sizes"].append(prefill_size)

    # Convert to numpy arrays and sort by x
    for data in [pure_prefill, pure_decode]:
        if data["x"]:
            indices = np.argsort(data["x"])
            data["x"] = np.array(data["x"])[indices]
            data["y"] = np.array(data["y"])[indices]
            data["yerr"] = np.array(data["yerr"])[indices]

    # Sort mixed data separately (has extra fields)
    if mixed["x"]:
        indices = np.argsort(mixed["x"])
        mixed["x"] = np.array(mixed["x"])[indices]
        mixed["y"] = np.array(mixed["y"])[indices]
        mixed["yerr"] = np.array(mixed["yerr"])[indices]
        mixed["labels"] = [mixed["labels"][i] for i in indices]
        mixed["num_decode"] = [mixed["num_decode"][i] for i in indices]
        mixed["prefill_size"] = [mixed["prefill_size"][i] for i in indices]

    # Sort each decode group by x
    for decode_count, data in mixed_by_decode.items():
        if data["x"]:
            indices = np.argsort(data["x"])
            data["x"] = np.array(data["x"])[indices]
            data["y"] = np.array(data["y"])[indices]
            data["yerr"] = np.array(data["yerr"])[indices]
            data["prefill_sizes"] = [data["prefill_sizes"][i] for i in indices]

    return pure_prefill, pure_decode, mixed, mixed_by_decode


def annotate_points(
    ax,
    x_vals,
    y_vals,
    labels: list[str],
    fontsize: int = 8,
    offset: tuple[int, int] = (5, 5),
    max_labels: int = 20,
):
    """Add text annotations to data points."""
    if len(labels) == 0:
        return

    # If too many points, only label a subset
    if len(labels) > max_labels:
        step = len(labels) // max_labels
        indices = list(range(0, len(labels), step))
    else:
        indices = list(range(len(labels)))

    for i in indices:
        ax.annotate(
            labels[i],
            (x_vals[i], y_vals[i]),
            textcoords="offset points",
            xytext=offset,
            fontsize=fontsize,
            alpha=0.8,
        )


def plot_results(
    pure_prefill: dict,
    pure_decode: dict,
    mixed: dict,
    config: dict,
    output_path: str,
    annotate: bool = True,
):
    """Create the benchmark plot."""
    fig, ax = plt.subplots(figsize=(14, 9))

    # Color scheme
    colors = {
        "prefill": "#2ecc71",  # Green
        "decode": "#3498db",   # Blue
        "mixed": "#e74c3c",    # Red
    }

    # Plot pure prefill
    if pure_prefill["x"].size > 0:
        ax.errorbar(
            pure_prefill["x"],
            pure_prefill["y"],
            yerr=pure_prefill["yerr"],
            fmt="o-",
            color=colors["prefill"],
            linewidth=2,
            markersize=8,
            capsize=4,
            label="Pure Prefill (1 request)",
        )

    # Plot pure decode
    if pure_decode["x"].size > 0:
        ax.errorbar(
            pure_decode["x"],
            pure_decode["y"],
            yerr=pure_decode["yerr"],
            fmt="s-",
            color=colors["decode"],
            linewidth=2,
            markersize=8,
            capsize=4,
            label="Pure Decode (N requests)",
        )

    # Plot mixed batch
    if mixed["x"].size > 0:
        ax.errorbar(
            mixed["x"],
            mixed["y"],
            yerr=mixed["yerr"],
            fmt="^-",
            color=colors["mixed"],
            linewidth=2,
            markersize=8,
            capsize=4,
            label="Mixed (N Decode + 1 Prefill)",
        )
        # Add annotations for mixed batch points
        if annotate and mixed.get("labels"):
            annotate_points(
                ax, mixed["x"], mixed["y"], mixed["labels"],
                fontsize=7, offset=(5, 5)
            )

    # Labels and title
    ax.set_xlabel("Total Tokens", fontsize=12)
    ax.set_ylabel("Execution Time (ms)", fontsize=12)

    model_name = config.get("model", "Unknown")
    decode_ctx = config.get("decode_context_lens", [])
    ctx_str = f", decode_ctx={decode_ctx}" if decode_ctx else ""
    ax.set_title(
        f"vLLM Single-Step Execution Time\n"
        f"Model: {model_name}{ctx_str}",
        fontsize=14,
    )

    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set axis to start from 0
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")

    return fig, ax


def plot_detailed_results(
    pure_prefill: dict,
    pure_decode: dict,
    mixed: dict,
    config: dict,
    output_path: str,
    annotate: bool = True,
):
    """Create a detailed 2x2 subplot figure."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors = {
        "prefill": "#2ecc71",
        "decode": "#3498db",
        "mixed": "#e74c3c",
    }

    # 1. Pure Prefill (top-left)
    ax1 = axes[0, 0]
    if pure_prefill["x"].size > 0:
        ax1.errorbar(
            pure_prefill["x"],
            pure_prefill["y"],
            yerr=pure_prefill["yerr"],
            fmt="o-",
            color=colors["prefill"],
            linewidth=2,
            markersize=8,
            capsize=4,
        )
    ax1.set_xlabel("Prefill Tokens", fontsize=11)
    ax1.set_ylabel("Time (ms)", fontsize=11)
    ax1.set_title("Pure Prefill", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    # 2. Pure Decode (top-right)
    ax2 = axes[0, 1]
    if pure_decode["x"].size > 0:
        ax2.errorbar(
            pure_decode["x"],
            pure_decode["y"],
            yerr=pure_decode["yerr"],
            fmt="s-",
            color=colors["decode"],
            linewidth=2,
            markersize=8,
            capsize=4,
        )
    ax2.set_xlabel("Batch Size (Decode Requests)", fontsize=11)
    ax2.set_ylabel("Time (ms)", fontsize=11)
    ax2.set_title("Pure Decode", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)

    # 3. Mixed Batch (bottom-left)
    ax3 = axes[1, 0]
    if mixed["x"].size > 0:
        ax3.errorbar(
            mixed["x"],
            mixed["y"],
            yerr=mixed["yerr"],
            fmt="^-",
            color=colors["mixed"],
            linewidth=2,
            markersize=8,
            capsize=4,
        )
        # Add annotations for mixed batch points
        if annotate and mixed.get("labels"):
            annotate_points(
                ax3, mixed["x"], mixed["y"], mixed["labels"],
                fontsize=7, offset=(5, 5)
            )
    ax3.set_xlabel("Total Tokens (Decode + Prefill)", fontsize=11)
    ax3.set_ylabel("Time (ms)", fontsize=11)
    ax3.set_title("Mixed Batch (N Decode + 1 Prefill)", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(left=0)
    ax3.set_ylim(bottom=0)

    # 4. All together (bottom-right)
    ax4 = axes[1, 1]
    if pure_prefill["x"].size > 0:
        ax4.errorbar(
            pure_prefill["x"],
            pure_prefill["y"],
            yerr=pure_prefill["yerr"],
            fmt="o-",
            color=colors["prefill"],
            linewidth=2,
            markersize=6,
            capsize=3,
            label="Pure Prefill",
        )
    if pure_decode["x"].size > 0:
        ax4.errorbar(
            pure_decode["x"],
            pure_decode["y"],
            yerr=pure_decode["yerr"],
            fmt="s-",
            color=colors["decode"],
            linewidth=2,
            markersize=6,
            capsize=3,
            label="Pure Decode",
        )
    if mixed["x"].size > 0:
        ax4.errorbar(
            mixed["x"],
            mixed["y"],
            yerr=mixed["yerr"],
            fmt="^-",
            color=colors["mixed"],
            linewidth=2,
            markersize=6,
            capsize=3,
            label="Mixed",
        )
        # Add annotations for mixed batch points in combined plot
        if annotate and mixed.get("labels"):
            annotate_points(
                ax4, mixed["x"], mixed["y"], mixed["labels"],
                fontsize=6, offset=(3, 3)
            )
    ax4.set_xlabel("Total Tokens", fontsize=11)
    ax4.set_ylabel("Time (ms)", fontsize=11)
    ax4.set_title("All Benchmarks Combined", fontsize=12, fontweight="bold")
    ax4.legend(loc="upper left", fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(left=0)
    ax4.set_ylim(bottom=0)

    # Overall title
    model_name = config.get("model", "Unknown")
    decode_ctx = config.get("decode_context_lens", [])
    fig.suptitle(
        f"vLLM Benchmark Results - {model_name}\n"
        f"Decode Context: {decode_ctx}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Detailed plot saved to {output_path}")

    return fig, axes


def plot_slope_analysis(
    pure_prefill: dict,
    pure_decode: dict,
    mixed_by_decode: dict[int, dict],
    config: dict,
    output_path: str,
):
    """
    Plot mixed batch data grouped by decode count to analyze slope changes.

    Each decode count is plotted as a separate line, allowing visualization
    of how the slope (time vs tokens) changes with different decode ratios.
    """
    fig, ax = plt.subplots(figsize=(14, 9))

    # Color map for different decode counts
    decode_counts = sorted(mixed_by_decode.keys())
    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(decode_counts) - 1, 1))
              for i in range(len(decode_counts))]

    # Plot pure prefill as reference
    prefill_slope = None
    if pure_prefill["x"].size > 0:
        ax.errorbar(
            pure_prefill["x"],
            pure_prefill["y"],
            yerr=pure_prefill["yerr"],
            fmt="o--",
            color="gray",
            linewidth=1.5,
            markersize=6,
            capsize=3,
            alpha=0.7,
            label="Pure Prefill (reference)",
        )
        # Calculate and annotate slope for pure prefill
        if len(pure_prefill["x"]) >= 2:
            prefill_slope, _ = np.polyfit(pure_prefill["x"], pure_prefill["y"], 1)
            last_x = pure_prefill["x"][-1]
            last_y = pure_prefill["y"][-1]
            ax.annotate(
                f"slope={prefill_slope:.4f}",
                (last_x, last_y),
                textcoords="offset points",
                xytext=(10, 0),
                fontsize=8,
                color="gray",
            )

    # Plot pure decode as reference
    decode_slope = None
    if pure_decode["x"].size > 0:
        ax.errorbar(
            pure_decode["x"],
            pure_decode["y"],
            yerr=pure_decode["yerr"],
            fmt="s--",
            color="black",
            linewidth=1.5,
            markersize=6,
            capsize=3,
            alpha=0.7,
            label="Pure Decode (reference)",
        )
        # Calculate and annotate slope for pure decode
        if len(pure_decode["x"]) >= 2:
            decode_slope, _ = np.polyfit(pure_decode["x"], pure_decode["y"], 1)
            last_x = pure_decode["x"][-1]
            last_y = pure_decode["y"][-1]
            ax.annotate(
                f"slope={decode_slope:.4f}",
                (last_x, last_y),
                textcoords="offset points",
                xytext=(10, 0),
                fontsize=8,
                color="black",
            )

    # Plot each decode count as a separate line
    for i, decode_count in enumerate(decode_counts):
        data = mixed_by_decode[decode_count]
        if len(data["x"]) == 0:
            continue

        ax.errorbar(
            data["x"],
            data["y"],
            yerr=data["yerr"],
            fmt="^-",
            color=colors[i],
            linewidth=2,
            markersize=8,
            capsize=4,
            label=f"{decode_count} Decode reqs",
        )

        # Calculate and display slope using linear regression
        if len(data["x"]) >= 2:
            x_arr = np.array(data["x"])
            y_arr = np.array(data["y"])
            slope, _ = np.polyfit(x_arr, y_arr, 1)
            # Add slope annotation at the end of the line
            last_x = x_arr[-1]
            last_y = y_arr[-1]
            ax.annotate(
                f"slope={slope:.4f}",
                (last_x, last_y),
                textcoords="offset points",
                xytext=(10, 0),
                fontsize=8,
                color=colors[i],
            )

    ax.set_xlabel("Total Tokens", fontsize=12)
    ax.set_ylabel("Execution Time (ms)", fontsize=12)

    model_name = config.get("model", "Unknown")
    ax.set_title(
        f"Slope Analysis: Effect of Decode Ratio on Execution Time\n"
        f"Model: {model_name}",
        fontsize=14,
    )

    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    # Save with slope suffix
    slope_path = output_path.replace(".png", "_slope_analysis.png")
    plt.savefig(slope_path, dpi=150, bbox_inches="tight")
    print(f"Slope analysis plot saved to {slope_path}")

    # Print slope summary
    print("\nSlope Summary (ms per token):")
    print("-" * 40)
    if prefill_slope is not None:
        print(f"  Pure Prefill:  slope = {prefill_slope:.6f} ms/token")
    if decode_slope is not None:
        print(f"  Pure Decode:   slope = {decode_slope:.6f} ms/token")
    print("-" * 40)
    for decode_count in decode_counts:
        data = mixed_by_decode[decode_count]
        if len(data["x"]) >= 2:
            x_arr = np.array(data["x"])
            y_arr = np.array(data["y"])
            slope, _ = np.polyfit(x_arr, y_arr, 1)
            print(f"  {decode_count:4d} Decode: slope = {slope:.6f} ms/token")

    return fig, ax


def extract_data_by_ratio(
    results: list[dict],
    target_ratios: list[float] = [0.25, 0.50, 0.75],
    tolerance: float = 0.10,
) -> dict[float, dict]:
    """
    Extract mixed batch data grouped by decode ratio.

    Args:
        results: List of benchmark results
        target_ratios: Target decode ratios to group by (e.g., 0.25 = 25%)
        tolerance: How close a point needs to be to a target ratio to be included

    Returns:
        Dict mapping target_ratio -> {x: [], y: [], yerr: [], labels: [], actual_ratios: []}
    """
    mixed_by_ratio: dict[float, dict] = {
        ratio: {"x": [], "y": [], "yerr": [], "labels": [], "actual_ratios": []}
        for ratio in target_ratios
    }

    for r in results:
        num_decode = r["num_decode"]
        num_prefill = r["num_prefill"]
        prefill_size = r["prefill_chunk_size"]
        total_tokens = r["total_tokens"]
        mean_time = r["mean_time_ms"]
        std_time = r["std_time_ms"]

        # Only process mixed batch results
        if num_decode > 0 and num_prefill >= 1:
            # Calculate decode ratio: decode tokens / total tokens
            decode_ratio = num_decode / total_tokens

            # Find closest target ratio within tolerance
            for target_ratio in target_ratios:
                if abs(decode_ratio - target_ratio) <= tolerance:
                    mixed_by_ratio[target_ratio]["x"].append(total_tokens)
                    mixed_by_ratio[target_ratio]["y"].append(mean_time)
                    mixed_by_ratio[target_ratio]["yerr"].append(std_time)
                    mixed_by_ratio[target_ratio]["labels"].append(
                        f"{num_decode}D{prefill_size}P"
                    )
                    mixed_by_ratio[target_ratio]["actual_ratios"].append(decode_ratio)
                    break  # Only assign to one ratio group

    # Convert to numpy arrays and sort by x
    for ratio, data in mixed_by_ratio.items():
        if data["x"]:
            indices = np.argsort(data["x"])
            data["x"] = np.array(data["x"])[indices]
            data["y"] = np.array(data["y"])[indices]
            data["yerr"] = np.array(data["yerr"])[indices]
            data["labels"] = [data["labels"][i] for i in indices]
            data["actual_ratios"] = [data["actual_ratios"][i] for i in indices]
        else:
            data["x"] = np.array([])
            data["y"] = np.array([])
            data["yerr"] = np.array([])

    return mixed_by_ratio


def plot_ratio_analysis(
    pure_prefill: dict,
    pure_decode: dict,
    mixed_by_ratio: dict[float, dict],
    config: dict,
    output_path: str,
    target_ratios: list[float] = [0.25, 0.50, 0.75],
):
    """
    Plot mixed batch data grouped by decode ratio (percentage).

    Each target ratio (e.g., 25%, 50%, 75%) is plotted as a separate line,
    showing how execution time changes with total tokens at fixed decode ratios.
    """
    fig, ax = plt.subplots(figsize=(14, 9))

    # Color map for different ratios
    ratio_colors = {
        0.25: "#3498db",  # Blue
        0.50: "#e74c3c",  # Red
        0.75: "#9b59b6",  # Purple
    }
    # Fallback colors for custom ratios
    cmap = plt.cm.tab10
    default_colors = [cmap(i) for i in range(10)]

    # Plot pure prefill as reference
    prefill_slope = None
    if pure_prefill["x"].size > 0:
        ax.errorbar(
            pure_prefill["x"],
            pure_prefill["y"],
            yerr=pure_prefill["yerr"],
            fmt="o--",
            color="gray",
            linewidth=1.5,
            markersize=6,
            capsize=3,
            alpha=0.7,
            label="Pure Prefill (0% decode)",
        )
        if len(pure_prefill["x"]) >= 2:
            prefill_slope, _ = np.polyfit(pure_prefill["x"], pure_prefill["y"], 1)
            last_x = pure_prefill["x"][-1]
            last_y = pure_prefill["y"][-1]
            ax.annotate(
                f"slope={prefill_slope:.4f}",
                (last_x, last_y),
                textcoords="offset points",
                xytext=(10, 0),
                fontsize=8,
                color="gray",
            )

    # Plot pure decode as reference (100% decode)
    decode_slope = None
    if pure_decode["x"].size > 0:
        ax.errorbar(
            pure_decode["x"],
            pure_decode["y"],
            yerr=pure_decode["yerr"],
            fmt="s--",
            color="black",
            linewidth=1.5,
            markersize=6,
            capsize=3,
            alpha=0.7,
            label="Pure Decode (100% decode)",
        )
        if len(pure_decode["x"]) >= 2:
            decode_slope, _ = np.polyfit(pure_decode["x"], pure_decode["y"], 1)
            last_x = pure_decode["x"][-1]
            last_y = pure_decode["y"][-1]
            ax.annotate(
                f"slope={decode_slope:.4f}",
                (last_x, last_y),
                textcoords="offset points",
                xytext=(10, 0),
                fontsize=8,
                color="black",
            )

    # Plot each ratio group
    for i, ratio in enumerate(sorted(target_ratios)):
        data = mixed_by_ratio.get(ratio, {"x": np.array([])})
        if len(data["x"]) == 0:
            continue

        color = ratio_colors.get(ratio, default_colors[i % len(default_colors)])
        percentage = int(ratio * 100)

        ax.errorbar(
            data["x"],
            data["y"],
            yerr=data["yerr"],
            fmt="^-",
            color=color,
            linewidth=2,
            markersize=8,
            capsize=4,
            label=f"{percentage}% Decode",
        )

        # Calculate and display slope
        if len(data["x"]) >= 2:
            x_arr = np.array(data["x"])
            y_arr = np.array(data["y"])
            slope, _ = np.polyfit(x_arr, y_arr, 1)
            last_x = x_arr[-1]
            last_y = y_arr[-1]
            ax.annotate(
                f"slope={slope:.4f}",
                (last_x, last_y),
                textcoords="offset points",
                xytext=(10, 0),
                fontsize=8,
                color=color,
            )

    ax.set_xlabel("Total Tokens", fontsize=12)
    ax.set_ylabel("Execution Time (ms)", fontsize=12)

    model_name = config.get("model", "Unknown")
    ax.set_title(
        f"Decode Ratio Analysis: Execution Time vs Total Tokens\n"
        f"Model: {model_name}",
        fontsize=14,
    )

    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    # Save with ratio suffix
    ratio_path = output_path.replace(".png", "_ratio_analysis.png")
    plt.savefig(ratio_path, dpi=150, bbox_inches="tight")
    print(f"Ratio analysis plot saved to {ratio_path}")

    # Print ratio summary
    print("\nDecode Ratio Analysis Summary:")
    print("-" * 50)
    if prefill_slope is not None:
        print(f"  Pure Prefill (0%):   slope = {prefill_slope:.6f} ms/token")
    for ratio in sorted(target_ratios):
        data = mixed_by_ratio.get(ratio, {"x": np.array([])})
        if len(data["x"]) >= 2:
            x_arr = np.array(data["x"])
            y_arr = np.array(data["y"])
            slope, _ = np.polyfit(x_arr, y_arr, 1)
            percentage = int(ratio * 100)
            avg_actual = np.mean(data["actual_ratios"]) * 100 if data["actual_ratios"] else ratio * 100
            print(f"  ~{percentage}% Decode (avg={avg_actual:.1f}%): slope = {slope:.6f} ms/token, "
                  f"{len(data['x'])} points")
    if decode_slope is not None:
        print(f"  Pure Decode (100%):  slope = {decode_slope:.6f} ms/token")

    return fig, ax


def plot_slope_vs_ratio(
    pure_prefill: dict,
    pure_decode: dict,
    mixed_by_ratio: dict[float, dict],
    config: dict,
    output_path: str,
    target_ratios: list[float],
):
    """
    Plot slope as a function of decode ratio - clean visualization for papers.

    This creates a simple line/scatter plot showing how the effective slope
    (ms per token) changes with decode ratio, clearly highlighting the
    non-monotonic behavior where high decode ratios can exceed 100% decode.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Collect slope data
    ratios = []
    slopes = []
    slope_errors = []

    # Add pure prefill (0% decode)
    if pure_prefill["x"].size > 0 and len(pure_prefill["x"]) >= 2:
        prefill_slope, _ = np.polyfit(pure_prefill["x"], pure_prefill["y"], 1)
        ratios.append(0.0)
        slopes.append(prefill_slope)
        slope_errors.append(0)  # Could compute confidence interval if needed

    # Add mixed ratios
    for ratio in sorted(target_ratios):
        data = mixed_by_ratio.get(ratio, {"x": np.array([])})
        if len(data["x"]) >= 2:
            x_arr = np.array(data["x"])
            y_arr = np.array(data["y"])
            slope, _ = np.polyfit(x_arr, y_arr, 1)
            ratios.append(ratio)
            slopes.append(slope)
            slope_errors.append(0)

    # Add pure decode (100% decode)
    if pure_decode["x"].size > 0 and len(pure_decode["x"]) >= 2:
        decode_slope, _ = np.polyfit(pure_decode["x"], pure_decode["y"], 1)
        ratios.append(1.0)
        slopes.append(decode_slope)
        slope_errors.append(0)

    # Convert to arrays
    ratios = np.array(ratios)
    slopes = np.array(slopes)

    # Plot main curve
    ax.plot(
        ratios * 100,
        slopes * 1000,  # Convert to μs/token for readability
        "o-",
        color="#2c3e50",
        linewidth=2,
        markersize=8,
        markerfacecolor="#3498db",
        markeredgecolor="#2c3e50",
        markeredgewidth=1.5,
        label="Measured slope",
    )

    # Add horizontal reference line for pure decode
    if pure_decode["x"].size > 0 and len(pure_decode["x"]) >= 2:
        ax.axhline(
            y=decode_slope * 1000,
            color="#e74c3c",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="Pure decode slope",
        )

    # Highlight the crossover region
    # Find where slope exceeds pure decode
    if len(slopes) > 0 and pure_decode["x"].size > 0 and len(pure_decode["x"]) >= 2:
        decode_slope_val = decode_slope * 1000
        for r, s in zip(ratios, slopes):
            if r < 1.0 and s * 1000 > decode_slope_val:
                ax.scatter(
                    [r * 100],
                    [s * 1000],
                    color="#e74c3c",
                    s=150,
                    zorder=5,
                    marker="o",
                    edgecolors="#c0392b",
                    linewidths=2,
                )

    # Labels and styling
    ax.set_xlabel("Decode Ratio (%)", fontsize=12)
    ax.set_ylabel("Slope (μs/token)", fontsize=12)

    model_name = config.get("model", "Unknown").split("/")[-1]  # Short name
    ax.set_title(
        f"Iteration Time Slope vs Decode Ratio\n({model_name})",
        fontsize=13,
    )

    ax.set_xlim(-5, 105)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.legend(loc="upper left", fontsize=10)

    # Add percentage labels on x-axis
    ax.set_xticks([0, 20, 40, 60, 80, 100])

    plt.tight_layout()

    # Save
    slope_ratio_path = output_path.replace(".png", "_slope_vs_ratio.png")
    plt.savefig(slope_ratio_path, dpi=300, bbox_inches="tight")
    print(f"Slope vs ratio plot saved to {slope_ratio_path}")

    # Also save as PDF for paper
    pdf_path = slope_ratio_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"PDF saved to {pdf_path}")

    return fig, ax


def plot_dual_gpu_analysis(
    h200_data: tuple,  # (pure_prefill, pure_decode, mixed_by_ratio)
    A6000_data: tuple,
    output_path: str,
    highlight_ratios: list[float] | None = None,
):
    """
    Combined 2x2 plot with H200 (left) and A6000 (right).
    Top row: slope vs ratio, Bottom row: time vs tokens.
    """
    if highlight_ratios is None:
        highlight_ratios = [0.20, 0.40, 0.60, 0.80, 0.90]

    # GPU-specific settings
    gpu_settings = {
        "H200": {"ylim": 200, "xlim": 9000, "slope_ymin": 15},
        "A6000": {"ylim": 250, "xlim": 4500, "slope_ymin": 20},
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 9),
                              gridspec_kw={'height_ratios': [0.5, 1.5]})

    # Color mapping
    color_prefill = "#1abc9c"
    color_decode = "#2c3e50"
    mixed_colors = ["#27ae60", "#f39c12", "#9b59b6", "#e74c3c", "#8b4513"]
    markers_list = ["^", "v", "D", "p", "h"]

    gpu_data = [("H200", h200_data), ("A6000", A6000_data)]

    for col, (gpu_name, (pure_prefill, pure_decode, mixed_by_ratio)) in enumerate(gpu_data):
        settings = gpu_settings[gpu_name]
        ax1 = axes[0, col]  # Top row
        ax2 = axes[1, col]  # Bottom row

        # ========== Top subplot: Slope vs Ratio ==========
        ratios = []
        slopes = []
        point_colors = []

        if pure_prefill["x"].size > 0 and len(pure_prefill["x"]) >= 2:
            prefill_slope, _ = np.polyfit(pure_prefill["x"], pure_prefill["y"], 1)
            ratios.append(0.0)
            slopes.append(prefill_slope)
            point_colors.append(color_prefill)

        for i, ratio in enumerate(highlight_ratios):
            data = mixed_by_ratio.get(ratio, {"x": np.array([])})
            if len(data["x"]) >= 2:
                x_arr = np.array(data["x"])
                y_arr = np.array(data["y"])
                slope, _ = np.polyfit(x_arr, y_arr, 1)
                ratios.append(ratio)
                slopes.append(slope)
                point_colors.append(mixed_colors[i % len(mixed_colors)])

        decode_slope = None
        if pure_decode["x"].size > 0 and len(pure_decode["x"]) >= 2:
            decode_slope, _ = np.polyfit(pure_decode["x"], pure_decode["y"], 1)
            ratios.append(1.0)
            slopes.append(decode_slope)
            point_colors.append(color_decode)

        ratios = np.array(ratios)
        slopes = np.array(slopes)

        # Plot connecting line
        ax1.plot(ratios * 100, slopes * 1000, "-", color="#888888", linewidth=1.5, zorder=1)

        # Plot each point with its color
        for r, s, c in zip(ratios, slopes, point_colors):
            ax1.scatter(r * 100, s * 1000, color=c, s=80, zorder=10, edgecolors="white", linewidths=1)

        # Reference line for pure decode
        if decode_slope is not None:
            ax1.axhline(y=decode_slope * 1000, color=color_decode, linestyle="--",
                       linewidth=1.5, alpha=0.7, label="Pure decode (100%)")

        ax1.set_xlabel("Decode Ratio (%)", fontsize=14)
        if col == 0:  # Only show ylabel on first column
            ax1.set_ylabel("Slope (μs/token)", fontsize=14)
        ax1.set_title(f"(a) Slope vs Decode Ratio", fontsize=14, fontweight="bold")
        ax1.set_xlim(-5, 105)
        ax1.set_ylim(settings["slope_ymin"], max(slopes) * 1000 * 1.05)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="lower right", fontsize=12)
        ax1.set_xticks([0, 20, 40, 60, 80, 100])
        ax1.tick_params(axis="both", labelsize=11)

        # ========== Bottom subplot: Time vs Tokens ==========
        line_width = 1.2
        marker_size = 5

        # Find max x for extending lines
        all_x = list(pure_prefill["x"]) + list(pure_decode["x"])
        for ratio in highlight_ratios:
            data = mixed_by_ratio.get(ratio, {"x": np.array([])})
            if len(data["x"]) > 0:
                all_x.extend(data["x"])
        max_x = max(all_x) if all_x else 10000

        # Plot pure prefill
        if pure_prefill["x"].size > 0:
            ax2.plot(pure_prefill["x"], pure_prefill["y"], "o-", color=color_prefill,
                    linewidth=line_width + 1, markersize=marker_size + 1,
                    label="Pure Prefill (0%)", zorder=10)
            if len(pure_prefill["x"]) >= 2:
                prefill_slope_val, _ = np.polyfit(pure_prefill["x"], pure_prefill["y"], 1)
                last_x, last_y = pure_prefill["x"][-1], pure_prefill["y"][-1]
                extend_x = np.array([last_x, max_x * 1.1])
                extend_y = np.array([last_y, last_y + prefill_slope_val * (max_x * 1.1 - last_x)])
                ax2.plot(extend_x, extend_y, "--", color=color_prefill, linewidth=1.0, alpha=0.7)
                ax2.annotate("0%", (last_x, last_y), textcoords="offset points",
                           xytext=(25, -20), fontsize=11, color=color_prefill, fontweight="bold",
                           arrowprops=dict(arrowstyle="->", color=color_prefill, lw=1.5))

        # Plot pure decode
        if pure_decode["x"].size > 0:
            ax2.plot(pure_decode["x"], pure_decode["y"], "s-", color=color_decode,
                    linewidth=line_width + 1, markersize=marker_size + 1,
                    label="Pure Decode (100%)", zorder=10)
            if len(pure_decode["x"]) >= 2:
                decode_slope_val, _ = np.polyfit(pure_decode["x"], pure_decode["y"], 1)
                last_x, last_y = pure_decode["x"][-1], pure_decode["y"][-1]
                extend_x = np.array([last_x, max_x * 1.1])
                extend_y = np.array([last_y, last_y + decode_slope_val * (max_x * 1.1 - last_x)])
                ax2.plot(extend_x, extend_y, "--", color=color_decode, linewidth=1.0, alpha=0.7)
                ax2.annotate("100%", (last_x, last_y), textcoords="offset points",
                           xytext=(15, 30), fontsize=11, color=color_decode, fontweight="bold",
                           arrowprops=dict(arrowstyle="->", color=color_decode, lw=1.5))

        # Plot mixed ratios
        for i, ratio in enumerate(highlight_ratios):
            data = mixed_by_ratio.get(ratio, {"x": np.array([])})
            if len(data["x"]) > 0:
                percentage = int(ratio * 100)
                color = mixed_colors[i % len(mixed_colors)]
                marker = markers_list[i % len(markers_list)]
                ax2.plot(data["x"], data["y"], f"{marker}-", color=color,
                        linewidth=line_width, markersize=marker_size, label=f"{percentage}% Decode")

        ax2.set_xlabel("Total Tokens", fontsize=14)
        if col == 0:  # Only show ylabel on first column
            ax2.set_ylabel("Execution Time (ms)", fontsize=14)
        ax2.set_title(f"(b) Execution Time vs Total Tokens", fontsize=14, fontweight="bold")
        ax2.legend(loc="upper left", fontsize=12, framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis="both", labelsize=11)
        ax2.set_xlim(0, settings["xlim"])
        ax2.set_ylim(0, settings["ylim"])

        # Add GPU name as column title
        axes[0, col].text(0.5, 1.15, gpu_name, transform=axes[0, col].transAxes,
                         fontsize=16, fontweight="bold", ha="center")

    plt.tight_layout()

    # Save
    dual_path = output_path.replace(".png", "_dual_gpu.png")
    plt.savefig(dual_path, dpi=300, bbox_inches="tight")
    print(f"Dual GPU plot saved to {dual_path}")

    pdf_path = dual_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"PDF saved to {pdf_path}")

    return fig, axes


def plot_combined_analysis(
    pure_prefill: dict,
    pure_decode: dict,
    mixed_by_ratio: dict[float, dict],
    config: dict,
    output_path: str,
    target_ratios: list[float],
    highlight_ratios: list[float] | None = None,
    gpu_name: str = "H200",
):
    """
    Combined plot with slope vs ratio (top) and time vs tokens (bottom).
    Best for papers - shows both the crossover phenomenon and raw data.
    """
    if highlight_ratios is None:
        highlight_ratios = [0.20, 0.40, 0.60, 0.80, 0.90]

    # GPU-specific settings
    gpu_settings = {
        "H200": {"ylim": 200, "xlim": 9000, "slope_ymin": 15},
        "A6000": {"ylim": 250, "xlim": 4500, "slope_ymin": 40},
    }
    settings = gpu_settings.get(gpu_name, {"ylim": 200, "xlim": 9000, "slope_ymin": 10})

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9),
                                     gridspec_kw={'height_ratios': [0.5, 1.5]})

    # ========== Top subplot: Slope vs Ratio ==========
    # Color mapping to match bottom plot
    color_prefill = "#1abc9c"
    color_decode = "#2c3e50"
    mixed_colors = ["#27ae60", "#f39c12", "#9b59b6", "#e74c3c", "#8b4513"]

    ratios = []
    slopes = []
    point_colors = []

    # Add pure prefill (0% decode)
    if pure_prefill["x"].size > 0 and len(pure_prefill["x"]) >= 2:
        prefill_slope, _ = np.polyfit(pure_prefill["x"], pure_prefill["y"], 1)
        ratios.append(0.0)
        slopes.append(prefill_slope)
        point_colors.append(color_prefill)

    # Add mixed ratios (only those in highlight_ratios)
    for i, ratio in enumerate(highlight_ratios):
        data = mixed_by_ratio.get(ratio, {"x": np.array([])})
        if len(data["x"]) >= 2:
            x_arr = np.array(data["x"])
            y_arr = np.array(data["y"])
            slope, _ = np.polyfit(x_arr, y_arr, 1)
            ratios.append(ratio)
            slopes.append(slope)
            point_colors.append(mixed_colors[i % len(mixed_colors)])

    # Add pure decode (100% decode)
    decode_slope = None
    if pure_decode["x"].size > 0 and len(pure_decode["x"]) >= 2:
        decode_slope, _ = np.polyfit(pure_decode["x"], pure_decode["y"], 1)
        ratios.append(1.0)
        slopes.append(decode_slope)
        point_colors.append(color_decode)

    ratios = np.array(ratios)
    slopes = np.array(slopes)

    # Plot connecting line (gray)
    ax1.plot(
        ratios * 100,
        slopes * 1000,
        "-",
        color="#888888",
        linewidth=1.5,
        zorder=1,
    )

    # Plot each point with its corresponding color
    for r, s, c in zip(ratios, slopes, point_colors):
        ax1.scatter(
            r * 100,
            s * 1000,
            color=c,
            s=80,
            zorder=10,
            edgecolors="white",
            linewidths=1,
        )

    # Add horizontal reference line for pure decode
    if decode_slope is not None:
        ax1.axhline(
            y=decode_slope * 1000,
            color=color_decode,
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="Pure decode (100%)",
        )

    ax1.set_xlabel("Decode Ratio (%)", fontsize=14)
    ax1.set_ylabel("Slope (μs/token)", fontsize=14)
    ax1.set_title("(a) Slope vs Decode Ratio", fontsize=14, fontweight="bold")
    ax1.set_xlim(-5, 105)
    ax1.set_ylim(settings["slope_ymin"], max(slopes) * 1000 * 1.05)  # Add 5% headroom above max
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=16)
    ax1.set_xticks([0, 20, 40, 60, 80, 100])
    ax1.tick_params(axis="both", labelsize=12)

    # ========== Bottom subplot: Time vs Tokens ==========
    line_width = 1.2
    marker_size = 5

    # Find max x across all data for extending lines
    all_x = []
    if pure_prefill["x"].size > 0:
        all_x.extend(pure_prefill["x"])
    if pure_decode["x"].size > 0:
        all_x.extend(pure_decode["x"])
    for ratio in highlight_ratios:
        data = mixed_by_ratio.get(ratio, {"x": np.array([])})
        if len(data["x"]) > 0:
            all_x.extend(data["x"])
    max_x = max(all_x) if all_x else 10000

    # Plot pure prefill - thicker line
    if pure_prefill["x"].size > 0:
        ax2.plot(
            pure_prefill["x"],
            pure_prefill["y"],
            "o-",
            color="#1abc9c",
            linewidth=line_width + 1,  # Thicker
            markersize=marker_size + 1,
            label="Pure Prefill (0%)",
            zorder=10,
        )
        # Fit line and extend as dashed line to edge
        if len(pure_prefill["x"]) >= 2:
            prefill_slope_val, _ = np.polyfit(
                pure_prefill["x"], pure_prefill["y"], 1
            )
            last_x = pure_prefill["x"][-1]
            last_y = pure_prefill["y"][-1]
            # Extend dashed line from actual endpoint using slope
            extend_x = np.array([last_x, max_x * 1.1])
            extend_y = np.array([last_y, last_y + prefill_slope_val * (max_x * 1.1 - last_x)])
            ax2.plot(
                extend_x, extend_y, "--",
                color="#1abc9c", linewidth=1.0, alpha=0.7, zorder=5
            )
            # Add arrow annotation for 0% at line end
            ax2.annotate(
                "0%",
                (last_x, last_y),
                textcoords="offset points",
                xytext=(10, 0),
                fontsize=11,
                color="#1abc9c",
                fontweight="bold",
                arrowprops=dict(
                    arrowstyle="->",
                    color="#1abc9c",
                    lw=1.5,
                ),
            )

    # Plot pure decode - thicker line
    if pure_decode["x"].size > 0:
        ax2.plot(
            pure_decode["x"],
            pure_decode["y"],
            "s-",
            color="#2c3e50",
            linewidth=line_width + 1,  # Thicker
            markersize=marker_size + 1,
            label="Pure Decode (100%)",
            zorder=10,
        )
        # Fit line and extend as dashed line to edge
        if len(pure_decode["x"]) >= 2:
            decode_slope_val, _ = np.polyfit(
                pure_decode["x"], pure_decode["y"], 1
            )
            last_x = pure_decode["x"][-1]
            last_y = pure_decode["y"][-1]
            # Extend dashed line from actual endpoint using slope
            extend_x = np.array([last_x, max_x * 1.1])
            extend_y = np.array([last_y, last_y + decode_slope_val * (max_x * 1.1 - last_x)])
            ax2.plot(
                extend_x, extend_y, "--",
                color="#2c3e50", linewidth=1.0, alpha=0.7, zorder=5
            )
            # Add arrow annotation for 100% at line end
            ax2.annotate(
                "100%",
                (last_x, last_y),
                textcoords="offset points",
                xytext=(15, 2),
                # xytext=(20, 40),
                fontsize=11,
                color="#2c3e50",
                fontweight="bold",
                arrowprops=dict(
                    arrowstyle="->",
                    color="#2c3e50",
                    lw=1.5,
                ),
            )

    # Plot highlighted ratios
    colors = ["#27ae60", "#f39c12", "#9b59b6", "#e74c3c", "#8b4513"]
    markers = ["^", "v", "D", "p", "h"]
    for i, ratio in enumerate(highlight_ratios):
        data = mixed_by_ratio.get(ratio, {"x": np.array([])})
        actual_ratio = ratio
        if len(data["x"]) == 0:
            available = [r for r in mixed_by_ratio.keys()
                        if len(mixed_by_ratio[r]["x"]) > 0]
            if available:
                closest = min(available, key=lambda r: abs(r - ratio))
                data = mixed_by_ratio[closest]
                actual_ratio = closest

        if len(data["x"]) > 0:
            percentage = int(actual_ratio * 100)
            color = colors[i % len(colors)]
            ax2.plot(
                data["x"],
                data["y"],
                f"{markers[i % len(markers)]}-",
                color=color,
                linewidth=line_width,
                markersize=marker_size,
                label=f"{percentage}% Decode",
            )

    ax2.set_xlabel("Total Tokens", fontsize=14)
    ax2.set_ylabel("Execution Time (ms)", fontsize=14)
    ax2.set_title("(b) Execution Time vs Total Tokens", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=12, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="both", labelsize=11)
    ax2.set_xlim(0, settings["xlim"])
    ax2.set_ylim(0, settings["ylim"])

    plt.tight_layout()

    # Save
    combined_path = output_path.replace(".png", "_combined.png")
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")
    print(f"Combined plot saved to {combined_path}")

    pdf_path = combined_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"PDF saved to {pdf_path}")

    return fig, (ax1, ax2)


def plot_ratio_analysis_simplified(
    pure_prefill: dict,
    pure_decode: dict,
    mixed_by_ratio: dict[float, dict],
    config: dict,
    output_path: str,
    highlight_ratios: list[float] | None = None,
    use_log_scale: bool = False,
):
    """
    Simplified ratio analysis plot for papers - shows only key curves.

    Shows: Pure prefill, Pure decode, and highlighted ratio(s) to
    clearly demonstrate the crossover phenomenon.

    Args:
        use_log_scale: If True, use log scale on both axes to separate curves
    """
    if highlight_ratios is None:
        highlight_ratios = [0.20, 0.40, 0.60, 0.80, 0.90]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Style settings for paper
    line_width = 1.2
    marker_size = 5

    # Plot pure prefill - with markers
    if pure_prefill["x"].size > 0:
        ax.plot(
            pure_prefill["x"],
            pure_prefill["y"],
            "o-",  # Line with circle markers
            color="#1abc9c",  # Teal - distinct from other colors
            linewidth=line_width,
            markersize=marker_size,
            label="Pure Prefill (0%)",
            zorder=10,  # Draw on top
        )
        # Add arrow annotation for 0% at line end
        if len(pure_prefill["x"]) >= 2:
            last_x = pure_prefill["x"][-1]
            last_y = pure_prefill["y"][-1]
            ax.annotate(
                "0%",
                (last_x, last_y),
                textcoords="offset points",
                xytext=(25, -20),
                fontsize=11,
                color="#1abc9c",
                fontweight="bold",
                arrowprops=dict(
                    arrowstyle="->",
                    color="#1abc9c",
                    lw=1.5,
                ),
            )

    # Plot pure decode - with markers
    if pure_decode["x"].size > 0:
        ax.plot(
            pure_decode["x"],
            pure_decode["y"],
            "s-",  # Line with square markers
            color="#2c3e50",  # Dark blue-gray
            linewidth=line_width,
            markersize=marker_size,
            label="Pure Decode (100%)",
            zorder=10,  # Draw on top
        )
        # Add arrow annotation for 100% at line end
        if len(pure_decode["x"]) >= 2:
            last_x = pure_decode["x"][-1]
            last_y = pure_decode["y"][-1]
            ax.annotate(
                "100%",
                (last_x, last_y),
                textcoords="offset points",
                xytext=(20, 45),  # To the right of the endpoint
                fontsize=11,
                color="#2c3e50",
                fontweight="bold",
                arrowprops=dict(
                    arrowstyle="->",
                    color="#2c3e50",
                    lw=1.5,
                ),
            )

    # Plot highlighted ratios - distinct colors for easy comparison
    # green, orange, purple, red, brown for 20%, 40%, 60%, 80%, 90%
    colors = ["#27ae60", "#f39c12", "#9b59b6", "#e74c3c", "#8b4513"]
    for i, ratio in enumerate(highlight_ratios):
        data = mixed_by_ratio.get(ratio, {"x": np.array([])})
        actual_ratio = ratio
        if len(data["x"]) == 0:
            # Try to find closest available ratio
            available = [r for r in mixed_by_ratio.keys()
                        if len(mixed_by_ratio[r]["x"]) > 0]
            if available:
                closest = min(available, key=lambda r: abs(r - ratio))
                data = mixed_by_ratio[closest]
                actual_ratio = closest

        if len(data["x"]) > 0:
            percentage = int(actual_ratio * 100)
            color = colors[i % len(colors)]
            markers = ["^", "v", "D", "p", "h"]
            ax.plot(
                data["x"],
                data["y"],
                f"{markers[i % len(markers)]}-",  # Line with markers
                color=color,
                linewidth=line_width,
                markersize=marker_size,
                label=f"{percentage}% Decode",
            )

    ax.set_xlabel("Total Tokens", fontsize=16)
    ax.set_ylabel("Execution Time (ms)", fontsize=16)

    ax.set_title(
        "H200",
        fontsize=18,
    )

    # Place legend outside or in best location
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    if use_log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    # Don't force axes to start from 0 - let matplotlib auto-scale to show differences

    # Ensure tick labels are readable
    ax.tick_params(axis="both", labelsize=10)

    plt.tight_layout()

    # Save
    simplified_path = output_path.replace(".png", "_ratio_simplified.png")
    plt.savefig(simplified_path, dpi=300, bbox_inches="tight")
    print(f"Simplified ratio plot saved to {simplified_path}")

    # PDF for paper
    pdf_path = simplified_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"PDF saved to {pdf_path}")

    return fig, ax


def plot_throughput(
    pure_prefill: dict,
    pure_decode: dict,
    mixed: dict,
    config: dict,
    output_path: str,
):
    """Create throughput plot (tokens/sec)."""
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = {
        "prefill": "#2ecc71",
        "decode": "#3498db",
        "mixed": "#e74c3c",
    }

    # Calculate throughput: tokens / time_ms * 1000 = tokens/sec
    def calc_throughput(x, y):
        return x / y * 1000

    # Plot pure prefill throughput
    if pure_prefill["x"].size > 0:
        throughput = calc_throughput(pure_prefill["x"], pure_prefill["y"])
        ax.plot(
            pure_prefill["x"],
            throughput,
            "o-",
            color=colors["prefill"],
            linewidth=2,
            markersize=8,
            label="Pure Prefill",
        )

    # Plot pure decode throughput
    if pure_decode["x"].size > 0:
        throughput = calc_throughput(pure_decode["x"], pure_decode["y"])
        ax.plot(
            pure_decode["x"],
            throughput,
            "s-",
            color=colors["decode"],
            linewidth=2,
            markersize=8,
            label="Pure Decode",
        )

    # Plot mixed throughput
    if mixed["x"].size > 0:
        throughput = calc_throughput(mixed["x"], mixed["y"])
        ax.plot(
            mixed["x"],
            throughput,
            "^-",
            color=colors["mixed"],
            linewidth=2,
            markersize=8,
            label="Mixed",
        )

    ax.set_xlabel("Total Tokens", fontsize=12)
    ax.set_ylabel("Throughput (tokens/sec)", fontsize=12)

    model_name = config.get("model", "Unknown")
    ax.set_title(
        f"vLLM Throughput\nModel: {model_name}",
        fontsize=14,
    )

    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    # Save with different filename
    throughput_path = output_path.replace(".png", "_throughput.png")
    plt.savefig(throughput_path, dpi=150, bbox_inches="tight")
    print(f"Throughput plot saved to {throughput_path}")

    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description="Plot benchmark results from benchmark_batch_combinations.py"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="results3.json",
        help="Input JSON file with benchmark results",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="benchmark_plot.png",
        help="Output image file",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Generate detailed 2x2 subplot figure",
    )
    parser.add_argument(
        "--throughput",
        action="store_true",
        help="Also generate throughput plot",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        default=True,
        help="Add labels (e.g., 128D1024P) to mixed batch points (default: True)",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Disable point labels on mixed batch",
    )
    parser.add_argument(
        "--slope-analysis",
        action="store_true",
        default=True,
        help="Generate slope analysis plot showing how decode ratio affects slope",
    )
    parser.add_argument(
        "--ratio-analysis",
        action="store_true",
        default=False,
        help="Generate ratio analysis plot (25%%, 50%%, 75%% decode ratio curves)",
    )
    parser.add_argument(
        "--target-ratios",
        type=str,
        default="0.25,0.50,0.75",
        help="Comma-separated target decode ratios for ratio analysis (default: 0.25,0.50,0.75)",
    )
    parser.add_argument(
        "--ratio-tolerance",
        type=float,
        default=0.10,
        help="Tolerance for matching decode ratios (default: 0.10 = ±10%%)",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use log scale on axes to better separate curves",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="H200",
        help="GPU name for title and axis limits (H200 or A6000)",
    )
    parser.add_argument(
        "--input-A6000",
        type=str,
        default=None,
        help="Second input JSON file for A6000 (enables dual GPU plot)",
    )
    args = parser.parse_args()

    # Handle annotate flag
    annotate = args.annotate and not args.no_annotate

    # Load data
    print(f"Loading results from {args.input}...")
    data = load_results(args.input)
    config = data.get("config", {})
    results = data.get("results", [])

    print(f"Found {len(results)} benchmark results")

    # Extract data
    pure_prefill, pure_decode, mixed, mixed_by_decode = extract_data(results)

    print(f"  Pure prefill: {len(pure_prefill['x'])} points")
    print(f"  Pure decode: {len(pure_decode['x'])} points")
    print(f"  Mixed batch: {len(mixed['x'])} points")
    print(f"  Mixed decode groups: {len(mixed_by_decode)} "
          f"({sorted(mixed_by_decode.keys())})")

    # Generate plots
    if args.detailed:
        plot_detailed_results(
            pure_prefill, pure_decode, mixed, config, args.output,
            annotate=annotate
        )
    else:
        plot_results(
            pure_prefill, pure_decode, mixed, config, args.output,
            annotate=annotate
        )

    if args.throughput:
        plot_throughput(pure_prefill, pure_decode, mixed, config, args.output)

    # Commented out - too cluttered for paper
    # if args.slope_analysis:
    #     plot_slope_analysis(
    #         pure_prefill, pure_decode, mixed_by_decode, config, args.output
    #     )

    if args.ratio_analysis:
        target_ratios = [float(x) for x in args.target_ratios.split(",")]
        mixed_by_ratio = extract_data_by_ratio(
            results, target_ratios=target_ratios, tolerance=args.ratio_tolerance
        )
        print(f"  Ratio groups: {len(mixed_by_ratio)} "
              f"({[f'{int(r*100)}%' for r in sorted(target_ratios)]})")

        # Check if dual GPU mode
        if args.input_A6000:
            print(f"\nLoading A6000 results from {args.input_A6000}...")
            A6000_data = load_results(args.input_A6000)
            A6000_results = A6000_data.get("results", [])
            A6000_prefill, A6000_decode, _, _ = extract_data(A6000_results)
            A6000_mixed_by_ratio = extract_data_by_ratio(
                A6000_results, target_ratios=target_ratios, tolerance=args.ratio_tolerance
            )
            print(f"  A6000 ratio groups: {len(A6000_mixed_by_ratio)}")

            # Generate dual GPU plot
            plot_dual_gpu_analysis(
                h200_data=(pure_prefill, pure_decode, mixed_by_ratio),
                A6000_data=(A6000_prefill, A6000_decode, A6000_mixed_by_ratio),
                output_path=args.output,
                highlight_ratios=[0.20, 0.40, 0.60, 0.80, 0.90],
            )
        else:
            # Single GPU plot
            plot_combined_analysis(
                pure_prefill, pure_decode, mixed_by_ratio, config, args.output,
                target_ratios=target_ratios,
                highlight_ratios=[0.20, 0.40, 0.60, 0.80, 0.90],
                gpu_name=args.gpu,
            )

    plt.show()


if __name__ == "__main__":
    main()
