#!/usr/bin/env python3
"""
Plot decode token ratio from vLLM scheduler statistics.

Usage:
    python plot_schedule_stats.py schedule_stats.json -o output
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_stats(filepath: str) -> dict:
    """Load schedule stats from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_decode_ratio(stats: list[dict], output_prefix: str = "schedule"):
    """Plot decode token ratio per iteration."""

    if not stats:
        print("No statistics data found!")
        return

    # Calculate decode ratio for each iteration
    iterations = []
    decode_ratios = []

    for i, s in enumerate(stats):
        total = s["prefill_tokens"] + s["decode_tokens"]
        if total > 0:
            ratio = s["decode_tokens"] / total * 100
            iterations.append(i)
            decode_ratios.append(ratio)

    iterations = np.array(iterations)
    decode_ratios = np.array(decode_ratios)

    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Plot 1: Scatter plot
    ax1 = axes[0]
    ax1.scatter(iterations, decode_ratios, s=2, color='#3498db', alpha=0.6)
    ax1.set_xlabel('Schedule Iteration', fontsize=11)
    ax1.set_ylabel('Decode Token Ratio (%)', fontsize=11)
    ax1.set_title('Decode Token Ratio per Schedule Iteration (Scatter)', fontsize=12)
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
    ax2.set_title('Decode Token Ratio with Moving Average', fontsize=12)
    ax2.set_ylim(-5, 105)
    ax2.axhline(y=100, color='green', linestyle='-', alpha=0.3, linewidth=1)
    ax2.axhline(y=0, color='red', linestyle='-', alpha=0.3, linewidth=1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right')

    plt.tight_layout()

    # Save figure
    output_file = f"{output_prefix}_decode_ratio.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")

    plt.show()

    # Print summary
    print(f"\nSummary:")
    print(f"  Total iterations: {len(iterations)}")
    print(f"  Mean decode ratio: {np.mean(decode_ratios):.1f}%")
    print(f"  Iterations with 100% decode: {sum(1 for r in decode_ratios if r == 100)}")
    print(f"  Iterations with 0% decode (pure prefill): {sum(1 for r in decode_ratios if r == 0)}")


def main():
    parser = argparse.ArgumentParser(description="Plot decode token ratio per iteration")
    parser.add_argument("stats_file", help="Path to schedule_stats.json")
    parser.add_argument("-o", "--output", default="schedule", help="Output file prefix")

    args = parser.parse_args()

    if not Path(args.stats_file).exists():
        print(f"Error: File not found: {args.stats_file}")
        sys.exit(1)

    data = load_stats(args.stats_file)
    stats = data.get("stats", [])

    print(f"Loaded {len(stats)} schedule records")

    plot_decode_ratio(stats, args.output)


if __name__ == "__main__":
    main()
