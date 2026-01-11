#!/usr/bin/env python3
"""
Plot decode token ratio from vLLM scheduler statistics.

Usage:
    # Single file
    python plot_schedule_stats.py schedule_stats.json -o output

    # Directory with multiple files (D128_2048_c8.json, etc.)
    python plot_schedule_stats.py /path/to/schedule_stats/ -o output_dir
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_stats(filepath: str) -> tuple[list[dict], dict | None]:
    """Load schedule stats from JSON file.

    Returns: (stats_list, run_info or None)
    """
    try:
        with open(filepath) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  JSON decode error: {e}")
        print("  Attempting to repair truncated JSON...")
        data = repair_json(filepath)
        if data is None:
            return [], None

    # Handle different formats
    if isinstance(data, dict):
        if "schedule_stats" in data:
            # Format: {"run_info": {...}, "schedule_stats": [...]}
            return data["schedule_stats"], data.get("run_info")
        elif "stats" in data:
            # Format: {"stats": [...]}
            return data["stats"], None
        else:
            return [], None
    elif isinstance(data, list):
        return data, None
    else:
        return [], None


def repair_json(filepath: str) -> dict | None:
    """尝试修复不完整的 JSON 文件（如被截断的文件）"""
    with open(filepath, 'r') as f:
        content = f.read()

    # 尝试找到最后一个完整的 JSON 对象
    # 格式: {"stats": [...]}

    # 方法1: 找到最后一个 '},' 或 '}]' 并截断
    # 从后往前找最后一个完整的条目
    last_complete = content.rfind('},')
    if last_complete == -1:
        last_complete = content.rfind('}]')

    if last_complete == -1:
        print("  Could not find valid JSON structure")
        return None

    # 截断到最后一个完整条目
    truncated = content[:last_complete + 1]

    # 补全 JSON 结构
    if '"stats"' in truncated:
        # 需要补上 ]} 来闭合
        truncated = truncated.rstrip().rstrip(',') + '\n  ]\n}'
    else:
        truncated = truncated.rstrip().rstrip(',') + '\n]'

    try:
        data = json.loads(truncated)
        print("  Repaired JSON successfully")
        return data
    except json.JSONDecodeError as e:
        print(f"  Repair failed: {e}")
        return None


def analyze_stats(stats: list[dict]) -> dict:
    """Analyze stats and return summary."""
    if not stats:
        return {"error": "No data"}

    iterations = []
    decode_ratios = []

    for i, s in enumerate(stats):
        total = s.get("prefill_tokens", 0) + s.get("decode_tokens", 0)
        if total > 0:
            ratio = s["decode_tokens"] / total * 100
            iterations.append(i)
            decode_ratios.append(ratio)

    if not decode_ratios:
        return {"error": "No valid iterations"}

    decode_ratios = np.array(decode_ratios)

    return {
        "total_iterations": len(iterations),
        "mean_decode_ratio": float(np.mean(decode_ratios)),
        "std_decode_ratio": float(np.std(decode_ratios)),
        "iterations_100_decode": int(sum(1 for r in decode_ratios if r == 100)),
        "iterations_0_decode": int(sum(1 for r in decode_ratios if r == 0)),
        "iterations": iterations,
        "decode_ratios": decode_ratios,
    }


def plot_decode_ratio(stats: list[dict], output_prefix: str, title_suffix: str = ""):
    """Plot decode token ratio per iteration."""

    analysis = analyze_stats(stats)
    if "error" in analysis:
        print(f"  Error: {analysis['error']}")
        return analysis

    iterations = np.array(analysis["iterations"])
    decode_ratios = analysis["decode_ratios"]

    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    title_base = f"Decode Token Ratio{title_suffix}"

    # Plot 1: Scatter plot
    ax1 = axes[0]
    ax1.scatter(iterations, decode_ratios, s=2, color='#3498db', alpha=0.6)
    ax1.set_xlabel('Schedule Iteration', fontsize=11)
    ax1.set_ylabel('Decode Token Ratio (%)', fontsize=11)
    ax1.set_title(f'{title_base} (Scatter)', fontsize=12)
    ax1.set_ylim(-5, 105)
    ax1.axhline(y=100, color='green', linestyle='-', alpha=0.3, linewidth=1)
    ax1.axhline(y=0, color='red', linestyle='-', alpha=0.3, linewidth=1)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Line plot with moving average
    ax2 = axes[1]
    ax2.plot(iterations, decode_ratios, color='#3498db', linewidth=0.3, alpha=0.4, label='Raw')

    # Add moving average for trend
    window = min(100, len(decode_ratios) // 10)
    if window > 1:
        moving_avg = np.convolve(decode_ratios, np.ones(window)/window, mode='valid')
        ma_iterations = iterations[window-1:]
        ax2.plot(ma_iterations, moving_avg, color='#e74c3c', linewidth=2,
                 alpha=0.9, label=f'Moving Avg (window={window})')

    ax2.set_xlabel('Schedule Iteration', fontsize=11)
    ax2.set_ylabel('Decode Token Ratio (%)', fontsize=11)
    ax2.set_title(f'{title_base} with Moving Average', fontsize=12)
    ax2.set_ylim(-5, 105)
    ax2.axhline(y=100, color='green', linestyle='-', alpha=0.3, linewidth=1)
    ax2.axhline(y=0, color='red', linestyle='-', alpha=0.3, linewidth=1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right')

    plt.tight_layout()
    plt.show()

    # Save figure
    output_file = f"{output_prefix}_decode_ratio.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return analysis


def print_summary(name: str, analysis: dict):
    """Print summary for a single file."""
    if "error" in analysis:
        print(f"\n{name}: Error - {analysis['error']}")
        return

    print(f"\n{name}:")
    print(f"  Total iterations: {analysis['total_iterations']}")
    print(f"  Mean decode ratio: {analysis['mean_decode_ratio']:.1f}%")
    print(f"  Iterations with 100% decode: {analysis['iterations_100_decode']}")
    print(f"  Iterations with 0% decode (pure prefill): {analysis['iterations_0_decode']}")


def process_single_file(filepath: Path, output_dir: Path) -> dict:
    """Process a single stats file."""
    stats, run_info = load_stats(str(filepath))

    if not stats:
        return {"name": filepath.stem, "error": "No stats data"}

    # Generate output prefix
    output_prefix = str(output_dir / filepath.stem)

    # Create title suffix from run_info or filename
    if run_info:
        scenario = run_info.get("scenario", "")
        concurrency = run_info.get("concurrency", "")
        title_suffix = f" - {scenario} (c={concurrency})"
    else:
        title_suffix = f" - {filepath.stem}"

    analysis = plot_decode_ratio(stats, output_prefix, title_suffix)
    analysis["name"] = filepath.stem
    analysis["run_info"] = run_info

    return analysis


def process_directory(dir_path: Path, output_dir: Path):
    """Process all JSON files in directory."""
    # Find all matching JSON files (D*_c*.json pattern)
    json_files = sorted(dir_path.glob("D*_c*.json"))

    if not json_files:
        # Try all JSON files except summary.json
        json_files = sorted([f for f in dir_path.glob("*.json")
                            if f.name != "summary.json"])

    if not json_files:
        print(f"No JSON files found in {dir_path}")
        return

    print(f"Found {len(json_files)} JSON files to process")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for filepath in json_files:
        print(f"\nProcessing: {filepath.name}...", end=" ")
        result = process_single_file(filepath, output_dir)

        if "error" not in result:
            print(f"OK ({result['total_iterations']} iterations)")
            all_results.append(result)
        else:
            print(f"FAILED: {result['error']}")

    # Print all summaries
    print("\n" + "=" * 60)
    print("SUMMARY FOR ALL FILES")
    print("=" * 60)

    for result in all_results:
        print_summary(result["name"], result)

    # Print overall summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Name':<25} {'Iters':<10} {'Mean%':<10} {'100%':<10} {'0%':<10}")
    print("-" * 65)

    for r in all_results:
        if "error" not in r:
            print(f"{r['name']:<25} {r['total_iterations']:<10} "
                  f"{r['mean_decode_ratio']:<10.1f} "
                  f"{r['iterations_100_decode']:<10} "
                  f"{r['iterations_0_decode']:<10}")

    # Save combined summary
    summary_output = output_dir / "all_summaries.json"
    summary_data = []
    for r in all_results:
        summary_entry = {
            "name": r["name"],
            "run_info": r.get("run_info"),
            "total_iterations": r.get("total_iterations"),
            "mean_decode_ratio": r.get("mean_decode_ratio"),
            "std_decode_ratio": r.get("std_decode_ratio"),
            "iterations_100_decode": r.get("iterations_100_decode"),
            "iterations_0_decode": r.get("iterations_0_decode"),
        }
        summary_data.append(summary_entry)

    with open(summary_output, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved summary to: {summary_output}")

    print(f"\nPlots saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Plot decode token ratio per iteration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python plot_schedule_stats.py schedule_stats.json -o output

  # Directory with multiple files
  python plot_schedule_stats.py /path/to/schedule_stats/ -o output_dir
        """
    )
    parser.add_argument("input_path", help="Path to JSON file or directory")
    parser.add_argument("-o", "--output", default="output",
                        help="Output file prefix (single file) or directory (multiple files)")

    args = parser.parse_args()

    input_path = Path(args.input_path)

    if not input_path.exists():
        print(f"Error: Path not found: {input_path}")
        sys.exit(1)

    if input_path.is_dir():
        # Process directory
        output_dir = Path(args.output)
        process_directory(input_path, output_dir)
    else:
        # Process single file
        output_dir = Path(args.output).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        stats, run_info = load_stats(str(input_path))
        print(f"Loaded {len(stats)} schedule records")

        if run_info:
            title_suffix = f" - {run_info.get('scenario', '')} (c={run_info.get('concurrency', '')})"
        else:
            title_suffix = ""

        analysis = plot_decode_ratio(stats, args.output, title_suffix)
        print_summary(input_path.name, analysis)


if __name__ == "__main__":
    main()
