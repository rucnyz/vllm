#!/usr/bin/env python3
"""
Analyze benchmark results to extract input/output length statistics.

This script analyzes the results from vllm bench serve to show:
- Actual input/output token distributions
- Whether the workload is prefill-heavy or decode-heavy

Usage:
    # Analyze a single benchmark result
    python pd_exp/analyze_benchmark_stats.py results/bench_baseline.json

    # Analyze all results in a grid search output directory
    python pd_exp/analyze_benchmark_stats.py pd_exp/outputs/grid_search_xxx/

Note: For detailed per-request statistics, run benchmark with --save-detailed flag.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np


def extract_json_field(file_path: str, field_name: str, is_array: bool = False):
    """Extract a specific field from a large JSON file without loading the entire file.

    This is a simple regex-based extraction that works for top-level fields.
    For arrays, it returns the parsed array. For scalars, it returns the value.
    """
    with open(file_path, 'r') as f:
        content = f.read()

    if is_array:
        # Match array field: "field_name": [...]
        pattern = rf'"{field_name}":\s*\[(.*?)\]'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            try:
                return json.loads('[' + match.group(1) + ']')
            except json.JSONDecodeError:
                return None
        return None
    else:
        # Match scalar field: "field_name": value
        pattern = rf'"{field_name}":\s*([^,\}}\]]+)'
        match = re.search(pattern, content)
        if match:
            value_str = match.group(1).strip()
            try:
                return json.loads(value_str)
            except json.JSONDecodeError:
                return value_str.strip('"')
        return None


def analyze_single_result_fast(result_path: str) -> dict:
    """Analyze a single benchmark result file using fast field extraction.

    This avoids loading the entire JSON file which can be 300MB+.
    """
    # Fields we need (all are near the beginning of the file)
    needed_scalar_fields = [
        'num_prompts', 'completed', 'total_input_tokens', 'total_output_tokens',
        'request_throughput', 'output_throughput', 'total_token_throughput'
    ]
    needed_array_fields = ['input_lens', 'output_lens']

    stats = {
        'file': os.path.basename(result_path),
    }

    # Read only the first portion of the file for scalar fields
    # These are all at the beginning
    with open(result_path, 'r') as f:
        # Read first 100KB which should contain all metadata
        header = f.read(100 * 1024)

    for field in needed_scalar_fields:
        pattern = rf'"{field}":\s*([^,\}}\]]+)'
        match = re.search(pattern, header)
        if match:
            value_str = match.group(1).strip()
            try:
                stats[field] = json.loads(value_str)
            except json.JSONDecodeError:
                stats[field] = None

    stats['num_prompts'] = stats.get('num_prompts', 0)
    stats['completed'] = stats.get('completed', 0)

    # For arrays (input_lens, output_lens), we need to find them
    # They appear early in the file after the scalar fields
    # Read more of the file to find them
    with open(result_path, 'r') as f:
        # Read first 2MB which should contain input_lens and output_lens
        content = f.read(2 * 1024 * 1024)

    has_detailed = False
    for field in needed_array_fields:
        pattern = rf'"{field}":\s*\[([\d,\s]+)\]'
        match = re.search(pattern, content)
        if match:
            try:
                arr = json.loads('[' + match.group(1) + ']')
                stats[field] = arr
                has_detailed = True
            except json.JSONDecodeError:
                stats[field] = None

    # Process input/output statistics
    if has_detailed:
        input_lens = [x for x in stats.get('input_lens', []) if x is not None and x > 0]
        output_lens = [x for x in stats.get('output_lens', []) if x is not None and x > 0]

        # Remove raw arrays from stats to save memory
        stats.pop('input_lens', None)
        stats.pop('output_lens', None)

        if input_lens:
            stats['input'] = {
                'mean': np.mean(input_lens),
                'median': np.median(input_lens),
                'std': np.std(input_lens),
                'min': np.min(input_lens),
                'max': np.max(input_lens),
                'p25': np.percentile(input_lens, 25),
                'p75': np.percentile(input_lens, 75),
                'p95': np.percentile(input_lens, 95),
            }

        if output_lens:
            stats['output'] = {
                'mean': np.mean(output_lens),
                'median': np.median(output_lens),
                'std': np.std(output_lens),
                'min': np.min(output_lens),
                'max': np.max(output_lens),
                'p25': np.percentile(output_lens, 25),
                'p75': np.percentile(output_lens, 75),
                'p95': np.percentile(output_lens, 95),
            }

        stats['detailed'] = True
    else:
        # Fall back to aggregate statistics
        total_input = stats.get('total_input_tokens', 0)
        total_output = stats.get('total_output_tokens', 0)
        completed = stats.get('completed', 1)

        if completed > 0:
            stats['input'] = {'mean': total_input / completed}
            stats['output'] = {'mean': total_output / completed}

        stats['detailed'] = False

    # Calculate ratio
    if 'input' in stats and 'output' in stats:
        input_mean = stats['input']['mean']
        output_mean = stats['output']['mean']
        if input_mean > 0:
            stats['output_input_ratio'] = output_mean / input_mean
            stats['workload_type'] = 'decode-heavy' if output_mean > input_mean else 'prefill-heavy'

    # Throughput metrics
    stats['throughput'] = {
        'request_throughput': stats.get('request_throughput', 0),
        'output_throughput': stats.get('output_throughput', 0),
        'total_token_throughput': stats.get('total_token_throughput', 0),
    }

    # Clean up intermediate fields (keep num_prompts and completed for display)
    fields_to_remove = ['total_input_tokens', 'total_output_tokens',
                        'request_throughput', 'output_throughput', 'total_token_throughput']
    for field in fields_to_remove:
        stats.pop(field, None)

    return stats


def analyze_single_result(result_path: str, fast: bool = True) -> dict:
    """Analyze a single benchmark result file.

    Args:
        result_path: Path to the JSON file
        fast: If True, use fast extraction (good for large files).
              If False, load entire JSON (slower but more reliable).
    """
    if fast:
        try:
            return analyze_single_result_fast(result_path)
        except Exception as e:
            print(f"Fast extraction failed for {result_path}, falling back to full load: {e}",
                  file=sys.stderr)

    # Fall back to full JSON load
    with open(result_path, 'r') as f:
        data = json.load(f)

    stats = {
        'file': os.path.basename(result_path),
        'num_prompts': data.get('num_prompts', 0),
        'completed': data.get('completed', 0),
    }

    # Check if detailed data is available
    has_detailed = 'input_lens' in data and 'output_lens' in data

    if has_detailed:
        input_lens = [x for x in data['input_lens'] if x is not None and x > 0]
        output_lens = [x for x in data['output_lens'] if x is not None and x > 0]

        if input_lens:
            stats['input'] = {
                'mean': np.mean(input_lens),
                'median': np.median(input_lens),
                'std': np.std(input_lens),
                'min': np.min(input_lens),
                'max': np.max(input_lens),
                'p25': np.percentile(input_lens, 25),
                'p75': np.percentile(input_lens, 75),
                'p95': np.percentile(input_lens, 95),
            }

        if output_lens:
            stats['output'] = {
                'mean': np.mean(output_lens),
                'median': np.median(output_lens),
                'std': np.std(output_lens),
                'min': np.min(output_lens),
                'max': np.max(output_lens),
                'p25': np.percentile(output_lens, 25),
                'p75': np.percentile(output_lens, 75),
                'p95': np.percentile(output_lens, 95),
            }

        stats['detailed'] = True
    else:
        # Fall back to aggregate statistics
        total_input = data.get('total_input_tokens', 0)
        total_output = data.get('total_output_tokens', 0)
        completed = data.get('completed', 1)

        if completed > 0:
            stats['input'] = {'mean': total_input / completed}
            stats['output'] = {'mean': total_output / completed}

        stats['detailed'] = False

    # Calculate ratio
    if 'input' in stats and 'output' in stats:
        input_mean = stats['input']['mean']
        output_mean = stats['output']['mean']
        if input_mean > 0:
            stats['output_input_ratio'] = output_mean / input_mean
            stats['workload_type'] = 'decode-heavy' if output_mean > input_mean else 'prefill-heavy'

    # Throughput metrics
    stats['throughput'] = {
        'request_throughput': data.get('request_throughput', 0),
        'output_throughput': data.get('output_throughput', 0),
        'total_token_throughput': data.get('total_token_throughput', 0),
    }

    return stats


def print_stats(stats: dict, verbose: bool = False):
    """Print statistics in a formatted way."""
    print(f"\n{'='*60}")
    print(f"File: {stats['file']}")
    print(f"{'='*60}")

    print(f"\nRequests: {stats['completed']}/{stats['num_prompts']} completed")

    if 'input' in stats:
        print(f"\n📥 Input Length Statistics:")
        inp = stats['input']
        if stats['detailed']:
            print(f"  Mean:   {inp['mean']:.1f} tokens")
            print(f"  Median: {inp['median']:.1f} tokens")
            print(f"  Std:    {inp['std']:.1f}")
            print(f"  Range:  [{inp['min']:.0f}, {inp['max']:.0f}]")
            if verbose:
                print(f"  P25:    {inp['p25']:.1f}")
                print(f"  P75:    {inp['p75']:.1f}")
                print(f"  P95:    {inp['p95']:.1f}")
        else:
            print(f"  Mean:   {inp['mean']:.1f} tokens (aggregate only)")

    if 'output' in stats:
        print(f"\n📤 Output Length Statistics:")
        out = stats['output']
        if stats['detailed']:
            print(f"  Mean:   {out['mean']:.1f} tokens")
            print(f"  Median: {out['median']:.1f} tokens")
            print(f"  Std:    {out['std']:.1f}")
            print(f"  Range:  [{out['min']:.0f}, {out['max']:.0f}]")
            if verbose:
                print(f"  P25:    {out['p25']:.1f}")
                print(f"  P75:    {out['p75']:.1f}")
                print(f"  P95:    {out['p95']:.1f}")
        else:
            print(f"  Mean:   {out['mean']:.1f} tokens (aggregate only)")

    if 'output_input_ratio' in stats:
        ratio = stats['output_input_ratio']
        workload = stats['workload_type']
        print(f"\n📊 Workload Analysis:")
        print(f"  Output/Input Ratio: {ratio:.2f}x")
        print(f"  Workload Type: {workload.upper()}")
        if ratio > 2:
            print(f"  → Strong decode-heavy (ratio > 2)")
        elif ratio > 1:
            print(f"  → Moderate decode-heavy (1 < ratio < 2)")
        elif ratio > 0.5:
            print(f"  → Balanced workload (0.5 < ratio < 1)")
        else:
            print(f"  → Prefill-heavy (ratio < 0.5)")

    if 'throughput' in stats:
        tp = stats['throughput']
        print(f"\n⚡ Throughput:")
        print(f"  Request:     {tp['request_throughput']:.2f} req/s")
        print(f"  Output:      {tp['output_throughput']:.2f} tok/s")
        print(f"  Total Token: {tp['total_token_throughput']:.2f} tok/s")

    if not stats['detailed']:
        print(f"\n⚠️  Note: Run benchmark with --save-detailed for per-request statistics")


def analyze_directory(dir_path: str, verbose: bool = False, fast: bool = True) -> list:
    """Analyze all benchmark results in a directory (recursively)."""
    results = []
    dir_path = Path(dir_path)

    # Find all bench_*.json files
    json_files = list(dir_path.rglob('bench_*.json'))
    total = len(json_files)

    print(f"Found {total} benchmark files to analyze...", file=sys.stderr)

    for i, json_file in enumerate(json_files, 1):
        try:
            print(f"\r[{i}/{total}] Processing {json_file.name}...", end='', file=sys.stderr)
            stats = analyze_single_result(str(json_file), fast=fast)
            stats['file'] = str(json_file.relative_to(dir_path))
            results.append(stats)
        except Exception as e:
            print(f"\nWarning: Failed to analyze {json_file}: {e}", file=sys.stderr)

    print(f"\nDone processing {len(results)} files.", file=sys.stderr)
    return results


def print_summary(results: list):
    """Print summary statistics across all results."""
    if not results:
        print("No results to summarize")
        return

    print(f"\n{'='*60}")
    print(f"SUMMARY ({len(results)} benchmark files)")
    print(f"{'='*60}")

    # Aggregate input/output statistics
    all_input_means = [r['input']['mean'] for r in results if 'input' in r]
    all_output_means = [r['output']['mean'] for r in results if 'output' in r]

    if all_input_means:
        print(f"\n📥 Input Length (across all benchmarks):")
        print(f"  Overall Mean: {np.mean(all_input_means):.1f} tokens")
        print(f"  Range: [{np.min(all_input_means):.1f}, {np.max(all_input_means):.1f}]")

    if all_output_means:
        print(f"\n📤 Output Length (across all benchmarks):")
        print(f"  Overall Mean: {np.mean(all_output_means):.1f} tokens")
        print(f"  Range: [{np.min(all_output_means):.1f}, {np.max(all_output_means):.1f}]")

    if all_input_means and all_output_means:
        overall_ratio = np.mean(all_output_means) / np.mean(all_input_means)
        print(f"\n📊 Overall Workload:")
        print(f"  Mean Output/Input Ratio: {overall_ratio:.2f}x")
        print(f"  Type: {'DECODE-HEAVY' if overall_ratio > 1 else 'PREFILL-HEAVY'}")

    # Throughput summary
    throughputs = [r['throughput']['request_throughput'] for r in results if 'throughput' in r]
    if throughputs:
        print(f"\n⚡ Throughput Summary:")
        print(f"  Mean: {np.mean(throughputs):.2f} req/s")
        print(f"  Range: [{np.min(throughputs):.2f}, {np.max(throughputs):.2f}]")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze benchmark results for input/output statistics')
    parser.add_argument('path', type=str,
                        help='Path to benchmark result JSON file or directory')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show detailed percentile statistics')
    parser.add_argument('--json', action='store_true',
                        help='Output results as JSON')
    parser.add_argument('--summary-only', action='store_true',
                        help='Only show summary (for directories)')
    parser.add_argument('--no-fast', action='store_true',
                        help='Disable fast extraction (use full JSON load)')

    args = parser.parse_args()

    path = Path(args.path)

    fast = not args.no_fast

    if path.is_file():
        stats = analyze_single_result(str(path), fast=fast)
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print_stats(stats, verbose=args.verbose)
    elif path.is_dir():
        results = analyze_directory(str(path), verbose=args.verbose, fast=fast)
        if not results:
            print(f"No benchmark result files found in {path}")
            sys.exit(1)

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            if not args.summary_only:
                for stats in results:
                    print_stats(stats, verbose=args.verbose)
            print_summary(results)
    else:
        print(f"Error: {path} does not exist")
        sys.exit(1)


if __name__ == '__main__':
    main()
