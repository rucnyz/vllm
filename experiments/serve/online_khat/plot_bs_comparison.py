#!/usr/bin/env python3
"""
Plot comparison of different batch sizes across scenarios.
For each scenario, select the k value with maximum throughput.
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Base directory
BASE_DIR = Path("/scratch/yuzhou/zwf/vllm/experiments/serve/online_khat")

# Batch sizes to analyze (in order)
BS_CONFIGS = [
    ("bs128_c256_n4000", 128),
    ("bs256_c512_n4000", 256),
    ("bs512_c1024_n4000", 512),
    ("bs768_c1024_n4000", 768),
    ("bs1024_c1024_n4000", 1024),
    ("bs1152_c1408_n4000", 1152),
    ("bs1408_c1536_n4000", 1408),
    ("bs1792_c2048_n4000", 1792),
]

# Scenarios to analyze
SCENARIOS = ["in128_out1024", "in512_out512", "in1024_out128"]
SCENARIO_LABELS = {
    "in128_out1024": "Short Prefix, Long Output (128→1024)",
    "in512_out512": "Balanced (512→512)",
    "in1024_out128": "Long Prefix, Short Output (1024→128)",
}

# Metrics to plot
METRICS = [
    ("output_throughput", "Output Throughput (tokens/s)"),
    ("request_throughput", "Request Throughput (req/s)"),
    ("mean_ttft_ms", "Mean TTFT (ms)"),
    ("mean_tpot_ms", "Mean TPOT (ms)"),
    ("mean_itl_ms", "Mean ITL (ms)"),
    ("p99_ttft_ms", "P99 TTFT (ms)"),
]


def load_bench_results(bench_dir):
    """Load all bench results from a directory and return dict with k values."""
    results = {}
    bench_files = glob.glob(str(bench_dir / "bench_*.json"))

    for bench_file in bench_files:
        filename = os.path.basename(bench_file)
        # Extract k value from filename
        if "baseline" in filename:
            k = "baseline"
        elif "fixed" in filename:
            # bench_fixed128.json -> k=128
            k = int(filename.replace("bench_fixed", "").replace(".json", ""))
        else:
            continue

        try:
            with open(bench_file, 'r') as f:
                data = json.load(f)
                results[k] = data
        except Exception as e:
            print(f"Error loading {bench_file}: {e}")

    return results


def find_best_k_for_throughput(results):
    """Find the k value that gives maximum output_throughput."""
    best_k = None
    best_throughput = -1

    for k, data in results.items():
        if k == "baseline":
            continue
        throughput = data.get("output_throughput", 0)
        if throughput > best_throughput:
            best_throughput = throughput
            best_k = k

    return best_k, best_throughput


def collect_data():
    """Collect data for all BS configs and scenarios."""
    data = {scenario: {"bs": [], "best_k": [], "metrics": {m[0]: [] for m in METRICS}}
            for scenario in SCENARIOS}

    baseline_data = {scenario: {"bs": [], "metrics": {m[0]: [] for m in METRICS}}
                     for scenario in SCENARIOS}

    for bs_dir, bs_value in BS_CONFIGS:
        for scenario in SCENARIOS:
            scenario_dir = BASE_DIR / bs_dir / scenario
            if not scenario_dir.exists():
                print(f"Warning: {scenario_dir} does not exist")
                continue

            results = load_bench_results(scenario_dir)
            if not results:
                print(f"Warning: No results found in {scenario_dir}")
                continue

            # Find best k
            best_k, _ = find_best_k_for_throughput(results)
            if best_k is None:
                print(f"Warning: No valid k found for {bs_dir}/{scenario}")
                continue

            # Get data for best k
            best_data = results[best_k]

            data[scenario]["bs"].append(bs_value)
            data[scenario]["best_k"].append(best_k)

            for metric, _ in METRICS:
                value = best_data.get(metric, np.nan)
                data[scenario]["metrics"][metric].append(value)

            # Also collect baseline data
            if "baseline" in results:
                baseline_data[scenario]["bs"].append(bs_value)
                for metric, _ in METRICS:
                    value = results["baseline"].get(metric, np.nan)
                    baseline_data[scenario]["metrics"][metric].append(value)

    return data, baseline_data


def plot_comparison(data, baseline_data):
    """Create comparison plots."""
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, len(SCENARIOS)))
    markers = ['o', 's', '^']

    for idx, (metric, metric_label) in enumerate(METRICS):
        ax = axes[idx]

        for i, scenario in enumerate(SCENARIOS):
            bs_values = data[scenario]["bs"]
            metric_values = data[scenario]["metrics"][metric]

            if bs_values and metric_values:
                ax.plot(bs_values, metric_values,
                       marker=markers[i], color=colors[i],
                       linewidth=2, markersize=8,
                       label=f"{SCENARIO_LABELS[scenario]} (best k)")

                # Plot baseline with dashed line
                if baseline_data[scenario]["bs"]:
                    ax.plot(baseline_data[scenario]["bs"],
                           baseline_data[scenario]["metrics"][metric],
                           marker=markers[i], color=colors[i],
                           linewidth=1, markersize=6,
                           linestyle='--', alpha=0.5,
                           label=f"{SCENARIO_LABELS[scenario]} (baseline)")

        ax.set_xlabel("Batch Size", fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(metric_label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks([128, 256, 512, 768, 1024, 1152, 1408, 1792])
        ax.tick_params(axis='x', rotation=45)

    # Create a single legend for the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
               ncol=2, fontsize=9)

    plt.suptitle("Performance Metrics vs Batch Size\n(Best k* for max throughput)",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    output_path = BASE_DIR / "bs_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_best_k_values(data):
    """Plot the best k values for each scenario."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(SCENARIOS)))
    markers = ['o', 's', '^']

    for i, scenario in enumerate(SCENARIOS):
        bs_values = data[scenario]["bs"]
        k_values = data[scenario]["best_k"]

        if bs_values and k_values:
            ax.plot(bs_values, k_values,
                   marker=markers[i], color=colors[i],
                   linewidth=2, markersize=8,
                   label=SCENARIO_LABELS[scenario])

    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Best k* Value", fontsize=12)
    ax.set_title("Optimal k* vs Batch Size (max throughput)", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([128, 256, 512, 768, 1024, 1152, 1408, 1792])
    ax.tick_params(axis='x', rotation=45)
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    output_path = BASE_DIR / "bs_best_k.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_throughput_only(data, baseline_data):
    """Create cleaner throughput-only comparison plot."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = {'best': plt.cm.tab10(0), 'baseline': plt.cm.tab10(1)}

    for idx, scenario in enumerate(SCENARIOS):
        ax = axes[idx]
        bs_values = data[scenario]["bs"]
        best_throughput = data[scenario]["metrics"]["output_throughput"]
        baseline_throughput = baseline_data[scenario]["metrics"]["output_throughput"]

        x = np.arange(len(bs_values))
        width = 0.35

        bars1 = ax.bar(x - width/2, best_throughput, width, label='Best k*', color='steelblue')
        bars2 = ax.bar(x + width/2, baseline_throughput, width, label='Baseline', color='coral', alpha=0.7)

        ax.set_xlabel("Batch Size", fontsize=11)
        ax.set_ylabel("Output Throughput (tokens/s)", fontsize=11)
        ax.set_title(SCENARIO_LABELS[scenario], fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(bs_values, rotation=45)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # Add improvement percentages on top of best bars
        for i, (best, base) in enumerate(zip(best_throughput, baseline_throughput)):
            if base > 0:
                improvement = (best - base) / base * 100
                ax.annotate(f'+{improvement:.1f}%',
                           xy=(x[i] - width/2, best),
                           ha='center', va='bottom',
                           fontsize=8, color='darkgreen')

    plt.suptitle("Output Throughput: Best k* vs Baseline", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = BASE_DIR / "bs_throughput_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_k_ratio(data):
    """Plot k/BS ratio for each scenario."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(SCENARIOS)))
    markers = ['o', 's', '^']

    for i, scenario in enumerate(SCENARIOS):
        bs_values = data[scenario]["bs"]
        k_values = data[scenario]["best_k"]
        ratios = [k / bs for k, bs in zip(k_values, bs_values)]

        ax.plot(bs_values, ratios,
               marker=markers[i], color=colors[i],
               linewidth=2, markersize=8,
               label=SCENARIO_LABELS[scenario])

    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("k*/BS Ratio", fontsize=12)
    ax.set_title("Optimal k*/BS Ratio vs Batch Size", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([128, 256, 512, 768, 1024, 1152, 1408, 1792])
    ax.tick_params(axis='x', rotation=45)
    ax.legend(loc='best', fontsize=10)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='k=BS')
    ax.set_ylim(0, 1.2)

    plt.tight_layout()
    output_path = BASE_DIR / "bs_k_ratio.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_combined_analysis(data, baseline_data):
    """Create a comprehensive 2x2 plot with key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(SCENARIOS)))
    markers = ['o', 's', '^']

    # Plot 1: Output Throughput
    ax = axes[0, 0]
    for i, scenario in enumerate(SCENARIOS):
        bs_values = data[scenario]["bs"]
        throughput = data[scenario]["metrics"]["output_throughput"]
        ax.plot(bs_values, throughput, marker=markers[i], color=colors[i],
               linewidth=2, markersize=8, label=SCENARIO_LABELS[scenario])
    ax.set_xlabel("Batch Size", fontsize=11)
    ax.set_ylabel("Output Throughput (tokens/s)", fontsize=11)
    ax.set_title("(a) Output Throughput (Best k*)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xticks([128, 256, 512, 768, 1024, 1152, 1408, 1792])
    ax.tick_params(axis='x', rotation=45)

    # Plot 2: Best k* value
    ax = axes[0, 1]
    for i, scenario in enumerate(SCENARIOS):
        bs_values = data[scenario]["bs"]
        k_values = data[scenario]["best_k"]
        ax.plot(bs_values, k_values, marker=markers[i], color=colors[i],
               linewidth=2, markersize=8, label=SCENARIO_LABELS[scenario])
    ax.plot([128, 1792], [128, 1792], 'k--', alpha=0.3, label='k*=BS')
    ax.set_xlabel("Batch Size", fontsize=11)
    ax.set_ylabel("Best k* Value", fontsize=11)
    ax.set_title("(b) Optimal k* Value", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xticks([128, 256, 512, 768, 1024, 1152, 1408, 1792])
    ax.tick_params(axis='x', rotation=45)

    # Plot 3: k/BS Ratio
    ax = axes[1, 0]
    for i, scenario in enumerate(SCENARIOS):
        bs_values = data[scenario]["bs"]
        k_values = data[scenario]["best_k"]
        ratios = [k / bs for k, bs in zip(k_values, bs_values)]
        ax.plot(bs_values, ratios, marker=markers[i], color=colors[i],
               linewidth=2, markersize=8, label=SCENARIO_LABELS[scenario])
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Batch Size", fontsize=11)
    ax.set_ylabel("k*/BS Ratio", fontsize=11)
    ax.set_title("(c) Optimal k*/BS Ratio", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xticks([128, 256, 512, 768, 1024, 1152, 1408, 1792])
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, 1.2)

    # Plot 4: Mean TTFT
    ax = axes[1, 1]
    for i, scenario in enumerate(SCENARIOS):
        bs_values = data[scenario]["bs"]
        ttft = data[scenario]["metrics"]["mean_ttft_ms"]
        ax.plot(bs_values, [t/1000 for t in ttft], marker=markers[i], color=colors[i],
               linewidth=2, markersize=8, label=SCENARIO_LABELS[scenario])
    ax.set_xlabel("Batch Size", fontsize=11)
    ax.set_ylabel("Mean TTFT (s)", fontsize=11)
    ax.set_title("(d) Mean Time to First Token (Best k*)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xticks([128, 256, 512, 768, 1024, 1152, 1408, 1792])
    ax.tick_params(axis='x', rotation=45)

    plt.suptitle("Performance Analysis: Batch Size Impact on Scheduling\n(k* selected for max throughput)",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = BASE_DIR / "bs_combined_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def print_summary(data):
    """Print summary table."""
    print("\n" + "="*80)
    print("Summary: Best k* and Output Throughput for each BS and Scenario")
    print("="*80)

    for scenario in SCENARIOS:
        print(f"\n{SCENARIO_LABELS[scenario]}:")
        print("-" * 60)
        print(f"{'BS':<10} {'Best k*':<10} {'Output Throughput':<20} {'k/BS Ratio':<15}")
        print("-" * 60)

        for i, bs in enumerate(data[scenario]["bs"]):
            k = data[scenario]["best_k"][i]
            throughput = data[scenario]["metrics"]["output_throughput"][i]
            ratio = k / bs if bs > 0 else 0
            print(f"{bs:<10} {k:<10} {throughput:<20.2f} {ratio:<15.3f}")


if __name__ == "__main__":
    print("Collecting data...")
    data, baseline_data = collect_data()

    print("\nPlotting comparison...")
    plot_comparison(data, baseline_data)

    print("\nPlotting best k values...")
    plot_best_k_values(data)

    print("\nPlotting throughput comparison...")
    plot_throughput_only(data, baseline_data)

    print("\nPlotting k ratio...")
    plot_k_ratio(data)

    print("\nPlotting combined analysis...")
    plot_combined_analysis(data, baseline_data)

    print_summary(data)
