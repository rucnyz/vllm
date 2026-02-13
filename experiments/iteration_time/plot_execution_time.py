#!/usr/bin/env python3
"""
Plot execution time results from JSON file.

Usage:
    python plot_execution_time.py --input-json execution_time_results.json --output-dir ./plots
"""

import argparse
import json
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np


def _get_gpu_name() -> str:
    """Auto-detect GPU name using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Get first GPU name and clean it up
            gpu_name = result.stdout.strip().split("\n")[0].strip()
            # Simplify common names
            if "RTX PRO 6000" in gpu_name:
                return "NVIDIA RTX PRO 6000"
            elif "H200" in gpu_name:
                return "NVIDIA H200"
            elif "H100" in gpu_name:
                return "NVIDIA H100"
            elif "A100" in gpu_name:
                return "NVIDIA A100"
            elif "RTX 4090" in gpu_name:
                return "NVIDIA RTX 4090"
            return gpu_name
    except Exception:
        pass
    return "GPU"  # Fallback


def _get_model_short_name(model: str) -> str:
    """Extract short model name from full path (e.g. 'Qwen/Qwen3-4B' -> 'Qwen3-4B')."""
    name = model.split("/")[-1]
    # Format specific model names
    name_mapping = {
        "gemma-3-1b-it": "Gemma-3-1B-IT",
        "gemma-3-4b-it": "Gemma-3-4B-IT",
        "gemma-3-12b-it": "Gemma-3-12B-IT",
        "gemma-3-27b-it": "Gemma-3-27B-IT",
    }
    return name_mapping.get(name.lower(), name)


def plot_results(
    input_json: str,
    output_dir: str,
    title: str | None = None,
    total_tokens_filter: list[int] | None = None,
    stop_on_incomplete: bool = False,
):
    """Generate execution time plots from results.

    Args:
        input_json: Path to input JSON file
        output_dir: Output directory for plots
        title: Plot title (auto-detected from JSON if not set)
        total_tokens_filter: List of total_tokens to include
        stop_on_incomplete: If True, stop at first total_tokens that doesn't
            have all decode percentages
    """
    with open(input_json, 'r') as f:
        data = json.load(f)

    results = data["results"]
    config = data.get("config", {})

    # Get expected decode percentages
    expected_decode_pcts = set(config.get("decode_percentages", [0, 20, 40, 60, 80, 100]))

    # Determine target total_tokens
    if total_tokens_filter:
        target_tokens = sorted(total_tokens_filter)
    else:
        target_tokens = sorted(set(r["total_tokens"] for r in results))

    # Apply stop_on_incomplete logic
    if stop_on_incomplete:
        valid_tokens = []
        for tt in target_tokens:
            available_pcts = set(
                r["decode_percentage"] for r in results if r["total_tokens"] == tt
            )
            if expected_decode_pcts.issubset(available_pcts):
                valid_tokens.append(tt)
            else:
                missing = expected_decode_pcts - available_pcts
                print(f"Stopping at total_tokens={tt}: missing decode percentages {sorted(missing)}")
                break
        target_tokens = valid_tokens
    else:
        # Just filter to tokens that exist in data
        available_tokens = set(r["total_tokens"] for r in results)
        target_tokens = [t for t in target_tokens if t in available_tokens]

    results = [r for r in results if r["total_tokens"] in target_tokens]

    if not results:
        print("No valid results to plot!")
        return

    print(f"Plotting with total_tokens: {sorted(set(r['total_tokens'] for r in results))}")

    if title is None:
        model_name = _get_model_short_name(config.get("model", ""))
        gpu_name = _get_gpu_name()
        title = f"{model_name} ({gpu_name})" if model_name else gpu_name
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
    parser.add_argument("--stop-on-incomplete", action="store_true",
                       help="Stop at first total_tokens missing any decode percentage")
    args = parser.parse_args()

    total_tokens_filter = None
    if args.total_tokens:
        total_tokens_filter = [int(x) for x in args.total_tokens.split(",")]
    plot_results(
        args.input_json,
        args.output_dir,
        title=args.title,
        total_tokens_filter=total_tokens_filter,
        stop_on_incomplete=args.stop_on_incomplete,
    )


if __name__ == "__main__":
    main()
