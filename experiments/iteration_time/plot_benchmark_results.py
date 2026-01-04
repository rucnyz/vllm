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
        default="results.json",
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

    if args.slope_analysis:
        plot_slope_analysis(
            pure_prefill, pure_decode, mixed_by_decode, config, args.output
        )

    plt.show()


if __name__ == "__main__":
    main()
