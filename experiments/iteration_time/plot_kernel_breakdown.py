#!/usr/bin/env python3
"""
Plot kernel breakdown results from JSON file.

Usage:
    python plot_kernel_breakdown.py --input-json gemm_attention_results.json --output-dir ./plots
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_results(input_json: str, output_dir: str, title: str = "NVIDIA RTX PRO 6000"):
    """Generate kernel breakdown plots from results."""
    with open(input_json, 'r') as f:
        data = json.load(f)

    results = data["results"]
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
                "gemm_time_ms": [],
                "attention_time_ms": [],
                "other_time_ms": [],
                "total_kernel_time_ms": [],
            }
        by_decode_pct[pct]["total_tokens"].append(r["total_tokens"])
        by_decode_pct[pct]["gemm_time_ms"].append(r["gemm_time_ms"])
        by_decode_pct[pct]["attention_time_ms"].append(r["attention_time_ms"])
        by_decode_pct[pct]["other_time_ms"].append(r.get("other_time_ms", 0))
        by_decode_pct[pct]["total_kernel_time_ms"].append(r.get("total_kernel_time_ms", 0))

    all_tokens = sorted(set(r["total_tokens"] for r in results))

    # Plot kernel breakdown
    fig, axes = plt.subplots(1, len(all_tokens), figsize=(4.5 * len(all_tokens), 6), sharey=True)
    if len(all_tokens) == 1:
        axes = [axes]

    bar_width = 0.7
    # Default blue-orange-green palette
    kernel_colors = {
        'GEMM': '#1f77b4',       # blue
        'Attention': '#ff7f0e',  # orange
        'Other': '#2ca02c',      # green
    }

    for ax, total_tok in zip(axes, all_tokens):
        # Get data for this total_tokens value
        decode_pcts = []
        gemm_vals = []
        attn_vals = []
        other_vals = []

        for pct in sorted(by_decode_pct.keys()):
            pct_data = by_decode_pct[pct]
            for i, tok in enumerate(pct_data["total_tokens"]):
                if tok == total_tok:
                    decode_pcts.append(pct)
                    gemm_vals.append(pct_data["gemm_time_ms"][i])
                    attn_vals.append(pct_data["attention_time_ms"][i])
                    other_vals.append(pct_data["other_time_ms"][i])
                    break

        if not decode_pcts:
            continue

        x = np.arange(len(decode_pcts))
        gemm_vals = np.array(gemm_vals)
        attn_vals = np.array(attn_vals)
        other_vals = np.array(other_vals)

        # Draw stacked bars
        ax.bar(x, gemm_vals, bar_width, label='GEMM', color=kernel_colors['GEMM'])
        ax.bar(x, attn_vals, bar_width, bottom=gemm_vals, label='Attention', color=kernel_colors['Attention'])
        ax.bar(x, other_vals, bar_width, bottom=gemm_vals + attn_vals, label='Other', color=kernel_colors['Other'])

        # Calculate attention top (top of attention bars)
        attn_tops = gemm_vals + attn_vals

        # Draw line connecting the points at attention bar tops
        ax.plot(x, attn_tops, color='#2D3436', linewidth=2, linestyle='-', zorder=5,
                marker='o', markersize=6, markerfacecolor='white', markeredgecolor='#2D3436', markeredgewidth=1.5)

        ax.set_xlabel("Decode %", fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{p}%" for p in decode_pcts], rotation=45, fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_title(f"Total Tokens = {total_tok}", fontsize=16)
        ax.grid(True, alpha=0.3, axis='y')

    axes[0].set_ylabel("Kernel Time (ms)", fontsize=16)
    axes[0].legend(loc='upper left', fontsize=14)

    fig.suptitle(title, fontsize=20, y=0.97)
    plt.tight_layout()
    _save_png_and_pdf(fig, "kernel_breakdown.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot kernel breakdown results")
    parser.add_argument("--input-json", type=str, required=True, help="Input JSON file")
    parser.add_argument("--output-dir", type=str, default="./plots", help="Output directory")
    parser.add_argument("--title", type=str, default="NVIDIA RTX PRO 6000", help="Plot title")
    args = parser.parse_args()

    plot_results(args.input_json, args.output_dir, args.title)


if __name__ == "__main__":
    main()
