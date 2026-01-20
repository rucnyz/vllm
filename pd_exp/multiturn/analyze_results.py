#!/usr/bin/env python3
"""
Analyze multi-turn benchmark results.

Usage:
    python pd_exp/multiturn/analyze_results.py pd_exp/outputs/multiturn_wildchat_*/
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_metrics_file(filepath: str) -> dict:
    """Parse metrics from benchmark log output."""
    metrics = {}
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Parse pandas DataFrame output format
        # Look for lines with metric names and values
        lines = content.strip().split('\n')
        for line in lines:
            # Match lines like "ttft_ms          123.45  ..."
            parts = line.split()
            if len(parts) >= 2:
                metric_name = parts[0]
                if metric_name in ['ttft_ms', 'tpot_ms', 'latency_ms',
                                   'input_num_tokens', 'output_num_tokens',
                                   'approx_cached_percent']:
                    try:
                        # Try to get mean value (usually second column)
                        metrics[metric_name] = float(parts[1])
                    except (ValueError, IndexError):
                        pass
    except Exception as e:
        print(f"Warning: Failed to parse {filepath}: {e}", file=sys.stderr)

    return metrics


def parse_results_json(filepath: str) -> dict:
    """Parse results from JSON output file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Warning: Failed to parse {filepath}: {e}", file=sys.stderr)
        return {}


def analyze_directory(result_dir: str) -> pd.DataFrame:
    """Analyze all results in a multi-turn benchmark output directory."""
    result_dir = Path(result_dir)

    results = []

    # Find all tb*/bs*/ directories
    for tb_dir in sorted(result_dir.glob('tb*')):
        tb_match = re.search(r'tb(\d+)', tb_dir.name)
        if not tb_match:
            continue
        tb = int(tb_match.group(1))

        for bs_dir in sorted(tb_dir.glob('bs*')):
            bs_match = re.search(r'bs(\d+)', bs_dir.name)
            if not bs_match:
                continue
            bs = int(bs_match.group(1))

            # Find scheduler results
            for scheduler in ['baseline', 'pd_kratio', 'pd_dynamic']:
                metrics_file = bs_dir / f'{scheduler}_metrics.txt'
                results_file = bs_dir / f'{scheduler}_results.json'
                bench_log = bs_dir / 'logs' / f'{scheduler}_bench.log'

                row = {
                    'tb': tb,
                    'bs': bs,
                    'scheduler': scheduler,
                }

                # Try to parse metrics
                if metrics_file.exists():
                    metrics = parse_metrics_file(str(metrics_file))
                    row.update(metrics)
                elif bench_log.exists():
                    # Try to parse from bench log
                    metrics = parse_metrics_file(str(bench_log))
                    row.update(metrics)

                # Check if we have any metrics
                if len(row) > 3:  # More than just tb, bs, scheduler
                    results.append(row)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


def print_comparison(df: pd.DataFrame):
    """Print scheduler comparison tables."""
    if df.empty:
        print("No results to display")
        return

    print("\n" + "=" * 80)
    print("Multi-Turn Benchmark Results (Scheduler Comparison)")
    print("=" * 80)

    # Key metrics to compare
    metrics = ['ttft_ms', 'tpot_ms', 'latency_ms', 'approx_cached_percent']
    available_metrics = [m for m in metrics if m in df.columns]

    if not available_metrics:
        print("\nNo recognized metrics found in results.")
        print("Available columns:", list(df.columns))
        return

    # Group by (tb, bs) and compare schedulers
    for (tb, bs), group in df.groupby(['tb', 'bs']):
        print(f"\n--- TB={tb}, BS={bs} ---")

        pivot = group.pivot(index='scheduler', columns=[], values=available_metrics)
        if isinstance(pivot, pd.Series):
            pivot = pivot.to_frame().T

        # Reorder schedulers
        scheduler_order = ['baseline', 'pd_kratio', 'pd_dynamic']
        pivot = pivot.reindex([s for s in scheduler_order if s in pivot.index])

        print(pivot.to_string())

        # Calculate improvement over baseline
        if 'baseline' in group['scheduler'].values:
            baseline = group[group['scheduler'] == 'baseline'].iloc[0]
            print("\nImprovement over baseline:")
            for scheduler in ['pd_kratio', 'pd_dynamic']:
                if scheduler in group['scheduler'].values:
                    row = group[group['scheduler'] == scheduler].iloc[0]
                    improvements = []
                    for metric in available_metrics:
                        if metric in baseline and metric in row:
                            base_val = baseline[metric]
                            new_val = row[metric]
                            if base_val > 0:
                                if metric == 'approx_cached_percent':
                                    # Higher is better for cache hit
                                    pct = (new_val - base_val) / base_val * 100
                                else:
                                    # Lower is better for latency
                                    pct = (base_val - new_val) / base_val * 100
                                improvements.append(f"{metric}: {pct:+.1f}%")
                    if improvements:
                        print(f"  {scheduler}: {', '.join(improvements)}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics by Scheduler")
    print("=" * 80)

    summary = df.groupby('scheduler')[available_metrics].agg(['mean', 'std', 'min', 'max'])
    print(summary.to_string())


def save_csv(df: pd.DataFrame, output_path: str):
    """Save results to CSV."""
    df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze multi-turn benchmark results')
    parser.add_argument('result_dir', type=str,
                        help='Path to multi-turn benchmark output directory')
    parser.add_argument('--csv', type=str, default=None,
                        help='Save results to CSV file')
    parser.add_argument('--json', action='store_true',
                        help='Output as JSON')

    args = parser.parse_args()

    if not os.path.isdir(args.result_dir):
        print(f"Error: {args.result_dir} is not a directory")
        sys.exit(1)

    df = analyze_directory(args.result_dir)

    if df.empty:
        print(f"No results found in {args.result_dir}")
        sys.exit(1)

    if args.json:
        print(df.to_json(orient='records', indent=2))
    else:
        print_comparison(df)

    if args.csv:
        save_csv(df, args.csv)


if __name__ == '__main__':
    main()
