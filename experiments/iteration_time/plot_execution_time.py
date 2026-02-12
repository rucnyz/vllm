#!/usr/bin/env python3
"""
Plot execution time results from JSON file.

Usage:
    python plot_execution_time.py --input-json execution_time_results.json --output-dir ./plots
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def _get_model_short_name(model: str) -> str:
    """Extract short model name from full path (e.g. 'Qwen/Qwen3-4B' -> 'Qwen3-4B')."""
    return model.split("/")[-1]


def plot_results(input_json: str, output_dir: str, title: str | None = None, total_tokens_filter: list[int] | None = None):
    """Generate execution time plots from results."""
    with open(input_json, 'r') as f:
        data = json.load(f)

    results = data["results"]
    if total_tokens_filter:
        results = [r for r in results if r["total_tokens"] in total_tokens_filter]
    if title is None:
        model_name = _get_model_short_name(data.get("config", {}).get("model", ""))
        title = f"{model_name} (NVIDIA H200)" if model_name else "NVIDIA H200"
    os.makedirs(output_dir, exist_ok=True)

    def _save_png_and_pdf(fig, filename_png: str) -> None:
        """Save figure as PNG and PDF (same stem)."""
        out_png = os.path.join(output_dir, filename_png)
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {out_png}")

        stem, _ = os.path.splitext(filename_png)
        out_pdf = os.path.join(output_dir, f"{stem}.pdf")
        fig.savefig(out_pdf, bbox_inches='tight')
        print(f"Plot saved to {out_pdf}")

    # Group by decode percentage
    by_decode_pct = {}
    for r in results:
        pct = r["decode_percentage"]
        if pct not in by_decode_pct:
            by_decode_pct[pct] = {
                "total_tokens": [],
                "execution_time_ms": [],
                "execution_time_std": [],
            }
        by_decode_pct[pct]["total_tokens"].append(r["total_tokens"])
        by_decode_pct[pct]["execution_time_ms"].append(r["execution_time_ms"])
        by_decode_pct[pct]["execution_time_std"].append(r.get("execution_time_std", 0))

    all_tokens = sorted(set(r["total_tokens"] for r in results))
    token_to_idx = {t: i for i, t in enumerate(all_tokens)}

    colors = plt.cm.viridis(np.linspace(0, 1, len(by_decode_pct)))

    # Plot execution time - smaller width, larger fonts
    fig, ax = plt.subplots(figsize=(7, 5))

    for (pct, pct_data), color in zip(sorted(by_decode_pct.items()), colors):
        sorted_idx = np.argsort(pct_data["total_tokens"])
        tokens = np.array(pct_data["total_tokens"])[sorted_idx]
        y = np.array(pct_data["execution_time_ms"])[sorted_idx]
        yerr = np.array(pct_data["execution_time_std"])[sorted_idx]
        x = np.array([token_to_idx[t] for t in tokens])

        if pct == 0:
            label = "Pure Prefill"
        elif pct == 100:
            label = "Pure Decode"
        else:
            label = f"{pct}% Decode"

        if yerr.sum() > 0:
            ax.errorbar(x, y, yerr=yerr, marker='o', label=label, color=color,
                       capsize=4, linewidth=2.5, markersize=8)
        else:
            ax.plot(x, y, marker='o', label=label, color=color, linewidth=2.5, markersize=8)

    ax.set_xticks(range(len(all_tokens)))
    ax.set_xticklabels([str(t) for t in all_tokens], fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel("Total Tokens", fontsize=16)
    ax.set_ylabel("Execution Time (ms)", fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_png_and_pdf(fig, "execution_time.png")
    plt.close()

    # Plot marginal cost per token
    fig, ax = plt.subplots(figsize=(6, 5))

    for (pct, pct_data), color in zip(sorted(by_decode_pct.items()), colors):
        sorted_idx = np.argsort(pct_data["total_tokens"])
        tokens = np.array(pct_data["total_tokens"])[sorted_idx]
        times = np.array(pct_data["execution_time_ms"])[sorted_idx]
        x = np.array([token_to_idx[t] for t in tokens])

        # Calculate marginal cost (time per token)
        marginal_cost = times / tokens

        if pct == 0:
            label = "Pure Prefill"
        elif pct == 100:
            label = "Pure Decode"
        else:
            label = f"{pct}% Decode"

        ax.plot(x, marginal_cost, marker='o', label=label, color=color, linewidth=2.5, markersize=8)

    ax.set_xticks(range(len(all_tokens)))
    ax.set_xticklabels([str(t) for t in all_tokens], fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel("Total Tokens", fontsize=16)
    ax.set_ylabel("Marginal Cost (ms/token)", fontsize=16)
    ax.set_title(f"{title} - Marginal Cost", fontsize=18)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_png_and_pdf(fig, "marginal_cost.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot execution time results")
    parser.add_argument("--input-json", type=str, required=True, help="Input JSON file")
    parser.add_argument("--output-dir", type=str, default="./plots", help="Output directory")
    parser.add_argument("--title", type=str, default=None, help="Plot title (auto-detected from JSON if not set)")
    parser.add_argument("--total-tokens", type=str, default=None, help="Filter total_tokens (e.g. 1024,2048,4096)")
    args = parser.parse_args()

    total_tokens_filter = None
    if args.total_tokens:
        total_tokens_filter = [int(x) for x in args.total_tokens.split(",")]
    plot_results(args.input_json, args.output_dir, title=args.title, total_tokens_filter=total_tokens_filter)


if __name__ == "__main__":
    main()
