#!/usr/bin/env python3
"""
Parse genai-bench results and plot k* sweep results.

Usage:
    python experiments/serve/plot_k_star_sweep.py \
        --input-dir ./experiment_results/k_star_sweep

Or manually specify results:
    python experiments/serve/plot_k_star_sweep.py \
        --csv ./experiment_results/k_star_sweep/results_summary.csv
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_genai_bench_log(log_file: Path) -> dict | None:
    """Parse genai-bench log output to extract metrics."""
    if not log_file.exists():
        return None

    content = log_file.read_text()
    metrics = {}

    # Common patterns in genai-bench output
    patterns = {
        "throughput": r"throughput[:\s]+([0-9.]+)",
        "tokens_per_second": r"tokens[/_]per[/_]second[:\s]+([0-9.]+)",
        "ttft_mean": r"ttft[_\s]*(mean|avg)[:\s]+([0-9.]+)",
        "time_to_first_token": r"time[_\s]*to[_\s]*first[_\s]*token[:\s]+([0-9.]+)",
        "tpot_mean": r"tpot[_\s]*(mean|avg)[:\s]+([0-9.]+)",
        "time_per_output_token": r"time[_\s]*per[_\s]*output[_\s]*token[:\s]+([0-9.]+)",
        "latency_mean": r"(e2e[_\s]*)?latency[_\s]*(mean|avg)[:\s]+([0-9.]+)",
        "requests_per_second": r"requests[/_]per[/_]second[:\s]+([0-9.]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            # Get the last group (the number)
            value = match.groups()[-1]
            try:
                metrics[key] = float(value)
            except ValueError:
                pass

    return metrics if metrics else None


def parse_genai_bench_json(json_file: Path) -> dict | None:
    """Parse genai-bench JSON results."""
    if not json_file.exists():
        return None

    try:
        with open(json_file) as f:
            data = json.load(f)

        metrics = {}

        # Navigate the JSON structure to find metrics
        def extract_metrics(obj, prefix=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_key = f"{prefix}_{k}" if prefix else k
                    if isinstance(v, (int, float)):
                        metrics[new_key.lower()] = v
                    elif isinstance(v, dict):
                        extract_metrics(v, new_key)

        extract_metrics(data)
        return metrics

    except Exception as e:
        print(f"Error parsing {json_file}: {e}")
        return None


def collect_results(input_dir: Path) -> pd.DataFrame:
    """Collect results from all k* experiment directories and baseline."""
    results = []

    # First: Check for baseline directory
    baseline_dir = input_dir / "baseline"
    if baseline_dir.exists():
        print(f"Processing BASELINE from {baseline_dir}")
        metrics = {"k_star": 0, "scheduler": "baseline"}

        log_file = baseline_dir / "benchmark.log"
        log_metrics = parse_genai_bench_log(log_file)
        if log_metrics:
            metrics.update(log_metrics)

        json_files = list(baseline_dir.rglob("*.json"))
        for jf in json_files:
            json_metrics = parse_genai_bench_json(jf)
            if json_metrics:
                for k, v in json_metrics.items():
                    if k not in metrics:
                        metrics[k] = v

        if len(metrics) > 2:
            results.append(metrics)
            print(f"  BASELINE results: {metrics}")
        else:
            print("  No metrics found for BASELINE")

    # Then: Find all k_star_XXX directories
    k_dirs = sorted(input_dir.glob("k_star_*"))

    for k_dir in k_dirs:
        # Extract k* value from directory name
        match = re.search(r"k_star_(\d+)", k_dir.name)
        if not match:
            continue

        k_star = int(match.group(1))
        print(f"Processing k*={k_star} from {k_dir}")

        metrics = {"k_star": k_star, "scheduler": "pd"}

        # Try to parse benchmark log
        log_file = k_dir / "benchmark.log"
        log_metrics = parse_genai_bench_log(log_file)
        if log_metrics:
            metrics.update(log_metrics)

        # Try to parse JSON results
        json_files = list(k_dir.rglob("*.json"))
        for jf in json_files:
            json_metrics = parse_genai_bench_json(jf)
            if json_metrics:
                # Don't overwrite existing metrics
                for k, v in json_metrics.items():
                    if k not in metrics:
                        metrics[k] = v

        if len(metrics) > 2:  # More than just k_star and scheduler
            results.append(metrics)
        else:
            print(f"  No metrics found for k*={k_star}")

    return pd.DataFrame(results)


def plot_results(df: pd.DataFrame, output_dir: Path):
    """Generate plots for k* sweep results."""
    if df.empty:
        print("No data to plot")
        return

    # Sort by k_star
    df = df.sort_values("k_star")

    # Separate baseline and P/D results
    baseline_df = df[df["k_star"] == 0] if "k_star" in df.columns else pd.DataFrame()
    pd_df = df[df["k_star"] > 0] if "k_star" in df.columns else df

    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("P/D Scheduler vs Baseline: Performance Comparison", fontsize=16)

    # Color scheme
    colors = {"throughput": "#2ecc71", "ttft": "#3498db", "tpot": "#9b59b6", "latency": "#e74c3c"}

    def add_baseline_line(ax, baseline_val, label_suffix=""):
        """Add horizontal baseline reference line."""
        if baseline_val is not None and not np.isnan(baseline_val):
            ax.axhline(y=baseline_val, color="gray", linestyle="--",
                      linewidth=2, alpha=0.8, label=f"Baseline: {baseline_val:.2f}{label_suffix}")

    # Plot 1: Throughput vs k*
    ax1 = axes[0, 0]
    throughput_col = None
    for col in ["throughput", "tokens_per_second", "requests_per_second"]:
        if col in df.columns and df[col].notna().any():
            throughput_col = col
            break

    if throughput_col:
        # Plot P/D results only (k* > 0)
        if not pd_df.empty and throughput_col in pd_df.columns:
            ax1.plot(pd_df["k_star"], pd_df[throughput_col], "-o",
                    color=colors["throughput"], linewidth=2, markersize=8,
                    label="P/D Scheduler")

        # Add baseline horizontal line
        if not baseline_df.empty and throughput_col in baseline_df.columns:
            baseline_val = baseline_df[throughput_col].iloc[0]
            add_baseline_line(ax1, baseline_val)

        ax1.set_xlabel("k* (switching threshold)", fontsize=12)
        ax1.set_ylabel(f"{throughput_col.replace('_', ' ').title()}", fontsize=12)
        ax1.set_title("Throughput vs k*", fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Mark the best k* for P/D
        if not pd_df.empty and throughput_col in pd_df.columns:
            best_idx = pd_df[throughput_col].idxmax()
            best_k = pd_df.loc[best_idx, "k_star"]
            best_val = pd_df.loc[best_idx, throughput_col]
            ax1.scatter([best_k], [best_val], color="red", s=150, zorder=5,
                       marker="*", label=f"Best k*={best_k}")
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, "No throughput data", ha="center", va="center",
                transform=ax1.transAxes, fontsize=12)

    # Plot 2: TTFT vs k*
    ax2 = axes[0, 1]
    ttft_col = None
    for col in df.columns:
        if "ttft" in col.lower() or "first_token" in col.lower():
            if df[col].notna().any():
                ttft_col = col
                break

    if ttft_col:
        if not pd_df.empty and ttft_col in pd_df.columns:
            ax2.plot(pd_df["k_star"], pd_df[ttft_col], "-o",
                    color=colors["ttft"], linewidth=2, markersize=8,
                    label="P/D Scheduler")

        if not baseline_df.empty and ttft_col in baseline_df.columns:
            baseline_val = baseline_df[ttft_col].iloc[0]
            add_baseline_line(ax2, baseline_val)

        ax2.set_xlabel("k* (switching threshold)", fontsize=12)
        ax2.set_ylabel("TTFT (ms)", fontsize=12)
        ax2.set_title("Time to First Token vs k*", fontsize=14)
        ax2.grid(True, alpha=0.3)

        # Mark the best (lowest) k*
        if not pd_df.empty and ttft_col in pd_df.columns:
            best_idx = pd_df[ttft_col].idxmin()
            best_k = pd_df.loc[best_idx, "k_star"]
            best_val = pd_df.loc[best_idx, ttft_col]
            ax2.scatter([best_k], [best_val], color="red", s=150, zorder=5,
                       marker="*", label=f"Best k*={best_k}")
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "No TTFT data", ha="center", va="center",
                transform=ax2.transAxes, fontsize=12)

    # Plot 3: TPOT vs k*
    ax3 = axes[1, 0]
    tpot_col = None
    for col in df.columns:
        if "tpot" in col.lower() or "per_output_token" in col.lower():
            if df[col].notna().any():
                tpot_col = col
                break

    if tpot_col:
        if not pd_df.empty and tpot_col in pd_df.columns:
            ax3.plot(pd_df["k_star"], pd_df[tpot_col], "-o",
                    color=colors["tpot"], linewidth=2, markersize=8,
                    label="P/D Scheduler")

        if not baseline_df.empty and tpot_col in baseline_df.columns:
            baseline_val = baseline_df[tpot_col].iloc[0]
            add_baseline_line(ax3, baseline_val)

        ax3.set_xlabel("k* (switching threshold)", fontsize=12)
        ax3.set_ylabel("TPOT (ms)", fontsize=12)
        ax3.set_title("Time per Output Token vs k*", fontsize=14)
        ax3.grid(True, alpha=0.3)

        # Mark the best (lowest) k*
        if not pd_df.empty and tpot_col in pd_df.columns:
            best_idx = pd_df[tpot_col].idxmin()
            best_k = pd_df.loc[best_idx, "k_star"]
            best_val = pd_df.loc[best_idx, tpot_col]
            ax3.scatter([best_k], [best_val], color="red", s=150, zorder=5,
                       marker="*", label=f"Best k*={best_k}")
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, "No TPOT data", ha="center", va="center",
                transform=ax3.transAxes, fontsize=12)

    # Plot 4: E2E Latency vs k*
    ax4 = axes[1, 1]
    latency_col = None
    for col in df.columns:
        if "latency" in col.lower() and "e2e" in col.lower():
            if df[col].notna().any():
                latency_col = col
                break
    if not latency_col:
        for col in df.columns:
            if "latency" in col.lower():
                if df[col].notna().any():
                    latency_col = col
                    break

    if latency_col:
        if not pd_df.empty and latency_col in pd_df.columns:
            ax4.plot(pd_df["k_star"], pd_df[latency_col], "-o",
                    color=colors["latency"], linewidth=2, markersize=8,
                    label="P/D Scheduler")

        if not baseline_df.empty and latency_col in baseline_df.columns:
            baseline_val = baseline_df[latency_col].iloc[0]
            add_baseline_line(ax4, baseline_val)

        ax4.set_xlabel("k* (switching threshold)", fontsize=12)
        ax4.set_ylabel("E2E Latency (ms)", fontsize=12)
        ax4.set_title("End-to-End Latency vs k*", fontsize=14)
        ax4.grid(True, alpha=0.3)

        # Mark the best (lowest) k*
        if not pd_df.empty and latency_col in pd_df.columns:
            best_idx = pd_df[latency_col].idxmin()
            best_k = pd_df.loc[best_idx, "k_star"]
            best_val = pd_df.loc[best_idx, latency_col]
            ax4.scatter([best_k], [best_val], color="red", s=150, zorder=5,
                       marker="*", label=f"Best k*={best_k}")
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, "No latency data", ha="center", va="center",
                transform=ax4.transAxes, fontsize=12)

    plt.tight_layout()

    # Save plots
    output_dir.mkdir(parents=True, exist_ok=True)

    png_file = output_dir / "k_star_sweep_results.png"
    plt.savefig(png_file, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {png_file}")

    pdf_file = output_dir / "k_star_sweep_results.pdf"
    plt.savefig(pdf_file, bbox_inches="tight")
    print(f"PDF saved to {pdf_file}")

    plt.show()


def plot_tradeoff(df: pd.DataFrame, output_dir: Path):
    """Plot throughput vs latency tradeoff with k* as parameter."""
    if df.empty:
        return

    # Find throughput and latency columns
    throughput_col = None
    for col in ["throughput", "tokens_per_second"]:
        if col in df.columns and df[col].notna().any():
            throughput_col = col
            break

    latency_col = None
    for col in df.columns:
        if "latency" in col.lower():
            if df[col].notna().any():
                latency_col = col
                break

    if not throughput_col or not latency_col:
        print("Cannot plot tradeoff: missing throughput or latency data")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create scatter plot with color based on k*
    scatter = ax.scatter(df[throughput_col], df[latency_col],
                        c=df["k_star"], cmap="viridis", s=100, alpha=0.8)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("k* (switching threshold)", fontsize=12)

    # Annotate each point with k* value
    for _, row in df.iterrows():
        ax.annotate(f'{int(row["k_star"])}',
                   (row[throughput_col], row[latency_col]),
                   textcoords="offset points", xytext=(5, 5),
                   fontsize=8, alpha=0.7)

    ax.set_xlabel(f"{throughput_col.replace('_', ' ').title()}", fontsize=12)
    ax.set_ylabel(f"{latency_col.replace('_', ' ').title()} (ms)", fontsize=12)
    ax.set_title("Throughput-Latency Tradeoff (varying k*)", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Highlight Pareto frontier
    df_sorted = df.sort_values(throughput_col)
    pareto_front = []
    min_latency = float("inf")
    for _, row in df_sorted.iterrows():
        if row[latency_col] < min_latency:
            pareto_front.append(row)
            min_latency = row[latency_col]

    if pareto_front:
        pareto_df = pd.DataFrame(pareto_front)
        ax.plot(pareto_df[throughput_col], pareto_df[latency_col],
               "r--", linewidth=2, alpha=0.5, label="Pareto frontier")
        ax.legend()

    plt.tight_layout()

    # Save
    png_file = output_dir / "k_star_tradeoff.png"
    plt.savefig(png_file, dpi=150, bbox_inches="tight")
    print(f"Tradeoff plot saved to {png_file}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot k* sweep results")
    parser.add_argument("--input-dir", type=str,
                       default="./experiment_results/k_star_sweep",
                       help="Directory containing k_star_XXX subdirectories")
    parser.add_argument("--csv", type=str,
                       help="Path to CSV results file (alternative to --input-dir)")
    parser.add_argument("--output-dir", type=str,
                       help="Output directory for plots (default: same as input-dir)")

    args = parser.parse_args()

    if args.csv:
        # Load from CSV
        df = pd.read_csv(args.csv)
        output_dir = Path(args.output_dir or Path(args.csv).parent)
    else:
        # Collect from experiment directories
        input_dir = Path(args.input_dir)
        df = collect_results(input_dir)
        output_dir = Path(args.output_dir or args.input_dir)

        # Save collected results
        if not df.empty:
            csv_file = output_dir / "k_star_sweep_results.csv"
            df.to_csv(csv_file, index=False)
            print(f"Results saved to {csv_file}")

    if df.empty:
        print("No results found!")
        return

    print("\nCollected results:")
    print(df.to_string(index=False))

    # Generate plots
    plot_results(df, output_dir)
    plot_tradeoff(df, output_dir)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for col in df.columns:
        if col == "k_star":
            continue
        if df[col].dtype in [np.float64, np.int64]:
            if "throughput" in col.lower() or "per_second" in col.lower():
                best_idx = df[col].idxmax()
                print(f"Best k* for {col}: {df.loc[best_idx, 'k_star']} "
                      f"(value: {df.loc[best_idx, col]:.2f})")
            else:
                best_idx = df[col].idxmin()
                print(f"Best k* for {col}: {df.loc[best_idx, 'k_star']} "
                      f"(value: {df.loc[best_idx, col]:.2f})")


if __name__ == "__main__":
    main()
