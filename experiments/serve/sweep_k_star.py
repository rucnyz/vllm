#!/usr/bin/env python3
"""
Sweep VLLM_PD_K_STAR from 1 to 128 (step=4) and measure throughput/latency.

Usage:
    python experiments/serve/sweep_k_star.py \
        --gpu 6 \
        --model Qwen/Qwen3-8B \
        --dataset ./experiments/serve/alpaca_prompts.csv \
        --output-dir ./experiment_results/k_star_sweep

This script:
1. Starts vLLM server with specific k* value
2. Runs genai-bench to measure performance
3. Collects results and generates plots
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def wait_for_server(port: int, timeout: int = 120) -> bool:
    """Wait for vLLM server to be ready."""
    import urllib.request
    import urllib.error

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            url = f"http://localhost:{port}/health"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    print(f"Server on port {port} is ready!")
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            pass
        time.sleep(2)
    return False


def start_vllm_server(
    k_star: int,
    gpu: int,
    model: str,
    port: int,
    max_num_seqs: int,
    api_key: str,
) -> subprocess.Popen:
    """Start vLLM server with specific k* value."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["VLLM_USE_PD_SCHEDULER"] = "1"
    env["VLLM_PD_ENABLE_DYNAMIC_KSTAR"] = "0"
    env["VLLM_PD_K_STAR"] = str(k_star)
    # Note: VLLM_PD_ALPHA_P/BETA_P/ALPHA_D/BETA_D are NOT needed when
    # VLLM_PD_K_STAR is explicitly set - the scheduler uses the fixed k* directly

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", model,
        "--port", str(port),
        "--max-num-seqs", str(max_num_seqs),
        "--gpu-memory-utilization", "0.9",
        "--api-key", api_key,
    ]

    print(f"Starting vLLM server with k*={k_star} on port {port}...")
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return process


def run_benchmark(
    port: int,
    model: str,
    dataset_path: str,
    output_dir: str,
    api_key: str,
    max_requests: int = 500,
    max_time: int = 120,
    concurrency: int = 128,
) -> dict | None:
    """Run genai-bench and return results."""
    cmd = [
        "genai-bench", "benchmark",
        "--api-backend", "vllm",
        "--api-key", api_key,
        "--api-base", f"http://localhost:{port}",
        "--api-model-name", model,
        "--model-tokenizer", model,
        "--task", "text-to-text",
        "--experiment-base-dir", output_dir,
        "--dataset-path", dataset_path,
        "--dataset-prompt-column", "prompt",
        "--max-time-per-run", str(max_time),
        "--max-requests-per-run", str(max_requests),
        "--num-concurrency", str(concurrency),
    ]

    print(f"Running benchmark: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Benchmark failed: {result.stderr}")
        return None

    print(result.stdout)
    return parse_benchmark_output(output_dir)


def parse_benchmark_output(output_dir: str) -> dict | None:
    """Parse genai-bench output to extract metrics."""
    # Find the latest results file
    output_path = Path(output_dir)

    # Look for JSON results files
    json_files = list(output_path.rglob("*.json"))
    if not json_files:
        print(f"No JSON results found in {output_dir}")
        return None

    # Get the most recent one
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)

    try:
        with open(latest_file) as f:
            data = json.load(f)

        # Extract key metrics (structure depends on genai-bench version)
        metrics = {}

        # Try to find throughput and latency metrics
        if isinstance(data, dict):
            # Common metric locations
            if "throughput" in data:
                metrics["throughput"] = data["throughput"]
            if "metrics" in data:
                m = data["metrics"]
                metrics["ttft_mean"] = m.get("ttft_mean", m.get("time_to_first_token_mean"))
                metrics["tpot_mean"] = m.get("tpot_mean", m.get("time_per_output_token_mean"))
                metrics["latency_mean"] = m.get("latency_mean", m.get("e2e_latency_mean"))
                metrics["throughput"] = m.get("throughput", m.get("tokens_per_second"))

            # Also check top level
            for key in ["ttft_mean", "tpot_mean", "latency_mean", "throughput",
                       "time_to_first_token_mean", "time_per_output_token_mean",
                       "e2e_latency_mean", "tokens_per_second"]:
                if key in data and key not in metrics:
                    metrics[key] = data[key]

        return metrics if metrics else data

    except Exception as e:
        print(f"Error parsing results: {e}")
        return None


def stop_server(process: subprocess.Popen):
    """Gracefully stop vLLM server."""
    if process.poll() is None:
        print("Stopping vLLM server...")
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def plot_results(results: list[dict], output_dir: str):
    """Generate plots for throughput and latency vs k*."""
    df = pd.DataFrame(results)

    if df.empty:
        print("No results to plot")
        return

    # Separate baseline and P/D results
    baseline_df = df[df["k_star"] == 0] if "k_star" in df.columns else pd.DataFrame()
    pd_df = df[df["k_star"] > 0] if "k_star" in df.columns else df

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("P/D Scheduler vs Baseline: Performance Comparison", fontsize=16)

    def add_baseline_line(ax, baseline_val, metric_name, color="gray"):
        """Add horizontal baseline reference line."""
        if baseline_val is not None:
            ax.axhline(y=baseline_val, color=color, linestyle="--",
                      linewidth=2, alpha=0.8, label=f"Baseline: {baseline_val:.2f}")

    # Plot 1: Throughput vs k*
    ax1 = axes[0, 0]
    if "throughput" in df.columns:
        # Plot P/D results
        if not pd_df.empty:
            ax1.plot(pd_df["k_star"], pd_df["throughput"], "b-o",
                    linewidth=2, markersize=8, label="P/D Scheduler")

        # Add baseline
        if not baseline_df.empty and "throughput" in baseline_df.columns:
            baseline_val = baseline_df["throughput"].iloc[0]
            add_baseline_line(ax1, baseline_val, "throughput")

        ax1.set_xlabel("k* (switching threshold)", fontsize=12)
        ax1.set_ylabel("Throughput (tokens/sec)", fontsize=12)
        ax1.set_title("Throughput vs k*", fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Mark the best k* for P/D
        if not pd_df.empty:
            best_idx = pd_df["throughput"].idxmax()
            best_k = pd_df.loc[best_idx, "k_star"]
            best_throughput = pd_df.loc[best_idx, "throughput"]
            ax1.scatter([best_k], [best_throughput], color="r", s=150, zorder=5,
                       marker="*", label=f"Best k*={best_k}")
        ax1.legend()

    # Plot 2: TTFT vs k*
    ax2 = axes[0, 1]
    ttft_col = next((c for c in df.columns if "ttft" in c.lower()), None)
    if ttft_col:
        if not pd_df.empty:
            ax2.plot(pd_df["k_star"], pd_df[ttft_col], "g-o",
                    linewidth=2, markersize=8, label="P/D Scheduler")

        if not baseline_df.empty and ttft_col in baseline_df.columns:
            baseline_val = baseline_df[ttft_col].iloc[0]
            add_baseline_line(ax2, baseline_val, "ttft")

        ax2.set_xlabel("k* (switching threshold)", fontsize=12)
        ax2.set_ylabel("TTFT (ms)", fontsize=12)
        ax2.set_title("Time to First Token vs k*", fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    # Plot 3: TPOT vs k*
    ax3 = axes[1, 0]
    tpot_col = next((c for c in df.columns if "tpot" in c.lower()), None)
    if tpot_col:
        if not pd_df.empty:
            ax3.plot(pd_df["k_star"], pd_df[tpot_col], "m-o",
                    linewidth=2, markersize=8, label="P/D Scheduler")

        if not baseline_df.empty and tpot_col in baseline_df.columns:
            baseline_val = baseline_df[tpot_col].iloc[0]
            add_baseline_line(ax3, baseline_val, "tpot")

        ax3.set_xlabel("k* (switching threshold)", fontsize=12)
        ax3.set_ylabel("TPOT (ms)", fontsize=12)
        ax3.set_title("Time per Output Token vs k*", fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend()

    # Plot 4: E2E Latency vs k*
    ax4 = axes[1, 1]
    latency_col = next((c for c in df.columns if "latency" in c.lower()), None)
    if latency_col:
        if not pd_df.empty:
            ax4.plot(pd_df["k_star"], pd_df[latency_col], "r-o",
                    linewidth=2, markersize=8, label="P/D Scheduler")

        if not baseline_df.empty and latency_col in baseline_df.columns:
            baseline_val = baseline_df[latency_col].iloc[0]
            add_baseline_line(ax4, baseline_val, "latency")

        ax4.set_xlabel("k* (switching threshold)", fontsize=12)
        ax4.set_ylabel("E2E Latency (ms)", fontsize=12)
        ax4.set_title("End-to-End Latency vs k*", fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.legend()

    plt.tight_layout()

    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_file = output_path / "k_star_sweep_results.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_file}")

    # Also save as PDF
    pdf_file = output_path / "k_star_sweep_results.pdf"
    plt.savefig(pdf_file, bbox_inches="tight")
    print(f"PDF saved to {pdf_file}")

    plt.show()


def start_baseline_server(
    gpu: int,
    model: str,
    port: int,
    max_num_seqs: int,
    api_key: str,
) -> subprocess.Popen:
    """Start baseline vLLM server (original scheduler, no P/D)."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["VLLM_USE_PD_SCHEDULER"] = "0"
    env["VLLM_PD_ENABLE_DYNAMIC_KSTAR"] = "0"

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", model,
        "--port", str(port),
        "--max-num-seqs", str(max_num_seqs),
        "--gpu-memory-utilization", "0.9",
        "--api-key", api_key,
    ]

    print(f"Starting BASELINE vLLM server (original scheduler) on port {port}...")
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return process


def main():
    parser = argparse.ArgumentParser(description="Sweep k* values and measure performance")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Model name")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--output-dir", type=str, default="./experiment_results/k_star_sweep",
                       help="Output directory for results")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--api-key", type=str, default="7355608", help="API key")
    parser.add_argument("--max-num-seqs", type=int, default=128, help="Max concurrent sequences")
    parser.add_argument("--k-start", type=int, default=1, help="Starting k* value")
    parser.add_argument("--k-end", type=int, default=128, help="Ending k* value")
    parser.add_argument("--k-step", type=int, default=4, help="Step size for k*")
    parser.add_argument("--max-requests", type=int, default=500, help="Max requests per benchmark")
    parser.add_argument("--max-time", type=int, default=120, help="Max time per benchmark (seconds)")
    parser.add_argument("--concurrency", type=int, default=128, help="Number of concurrent requests")
    parser.add_argument("--warmup-requests", type=int, default=50, help="Warmup requests before measurement")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline benchmark")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate k* values to test
    k_values = list(range(args.k_start, args.k_end + 1, args.k_step))
    # Make sure we include 1 and the endpoint
    if 1 not in k_values:
        k_values = [1] + k_values
    if args.k_end not in k_values:
        k_values.append(args.k_end)
    k_values = sorted(set(k_values))

    print(f"Will test k* values: {k_values}")
    print(f"Total experiments: {len(k_values)} + 1 baseline")

    results = []

    # ===== First: Run baseline (original vLLM scheduler) =====
    if not args.skip_baseline:
        print(f"\n{'='*60}")
        print("Testing BASELINE (original vLLM scheduler)")
        print(f"{'='*60}")

        exp_output_dir = output_dir / "baseline"
        exp_output_dir.mkdir(parents=True, exist_ok=True)

        process = start_baseline_server(
            gpu=args.gpu,
            model=args.model,
            port=args.port,
            max_num_seqs=args.max_num_seqs,
            api_key=args.api_key,
        )

        try:
            if not wait_for_server(args.port, timeout=180):
                print("Baseline server failed to start")
                stop_server(process)
            else:
                time.sleep(5)

                metrics = run_benchmark(
                    port=args.port,
                    model=args.model,
                    dataset_path=args.dataset,
                    output_dir=str(exp_output_dir),
                    api_key=args.api_key,
                    max_requests=args.max_requests,
                    max_time=args.max_time,
                    concurrency=args.concurrency,
                )

                if metrics:
                    metrics["k_star"] = 0  # Use 0 to represent baseline
                    metrics["scheduler"] = "baseline"
                    results.append(metrics)
                    print(f"BASELINE results: {metrics}")

        finally:
            stop_server(process)
            time.sleep(10)

    # ===== Then: Run P/D scheduler with different k* values =====
    for k_star in k_values:
        print(f"\n{'='*60}")
        print(f"Testing k* = {k_star}")
        print(f"{'='*60}")

        # Create experiment-specific output dir
        exp_output_dir = output_dir / f"k_star_{k_star:03d}"
        exp_output_dir.mkdir(parents=True, exist_ok=True)

        # Start server
        process = start_vllm_server(
            k_star=k_star,
            gpu=args.gpu,
            model=args.model,
            port=args.port,
            max_num_seqs=args.max_num_seqs,
            api_key=args.api_key,
        )

        try:
            # Wait for server to be ready
            if not wait_for_server(args.port, timeout=180):
                print(f"Server failed to start for k*={k_star}")
                stop_server(process)
                continue

            # Give it a moment to stabilize
            time.sleep(5)

            # Run benchmark
            metrics = run_benchmark(
                port=args.port,
                model=args.model,
                dataset_path=args.dataset,
                output_dir=str(exp_output_dir),
                api_key=args.api_key,
                max_requests=args.max_requests,
                max_time=args.max_time,
                concurrency=args.concurrency,
            )

            if metrics:
                metrics["k_star"] = k_star
                metrics["scheduler"] = "pd"
                results.append(metrics)
                print(f"k*={k_star} results: {metrics}")

        finally:
            stop_server(process)
            # Wait a bit before starting next server
            time.sleep(10)

    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_file = output_dir / "k_star_sweep_results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\nResults saved to {results_file}")

        # Also save as JSON
        json_file = output_dir / "k_star_sweep_results.json"
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"JSON results saved to {json_file}")

        # Generate plots
        plot_results(results, str(output_dir))

        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(results_df.to_string(index=False))

        if "throughput" in results_df.columns:
            best_idx = results_df["throughput"].idxmax()
            print(f"\nBest k* for throughput: {results_df.loc[best_idx, 'k_star']}")
            print(f"Best throughput: {results_df.loc[best_idx, 'throughput']:.2f} tokens/sec")
    else:
        print("No results collected!")


if __name__ == "__main__":
    main()
