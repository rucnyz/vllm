# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark GEMM time vs total tokens for different decode percentages.

This script measures GEMM kernel time for different batch compositions:
- Pure prefill (0% decode)
- Mixed batches (20%, 40%, 60%, 80% decode)
- Pure decode (100% decode)

Usage:
    # Step 1: Run benchmark to collect data
    python benchmark_gemm_by_decode_ratio.py run \
        --model Qwen/Qwen3-4B \
        --total-tokens 512,1024,2048,4096,8192 \
        --output-json gemm_by_decode_ratio.json

    # Step 2: Generate plot from collected data
    python benchmark_gemm_by_decode_ratio.py plot \
        --input-json gemm_by_decode_ratio.json \
        --output-png gemm_by_decode_ratio.png

    # Or run with nsys for accurate GPU kernel timing:
    VLLM_NVTX_SCOPES_FOR_PROFILING=1 nsys profile -o gemm_profile \
        python benchmark_gemm_by_decode_ratio.py run ...
"""

import os
import sys
from pathlib import Path

# Add vllm project root to path
VLLM_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(VLLM_ROOT))

import argparse
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import torch

# Check if using NVTX mode
USE_NVTX = os.environ.get("VLLM_NVTX_SCOPES_FOR_PROFILING") == "1"
if USE_NVTX:
    os.environ["VLLM_NVTX_SCOPES_FOR_PROFILING"] = "1"
    os.environ.pop("VLLM_CUSTOM_SCOPES_FOR_PROFILING", None)
    print("Using NVTX profiling scopes (use nsys for accurate GEMM timing)")
else:
    os.environ["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] = "1"
    print("Using PyTorch record_function profiling scopes")


@dataclass
class GEMMBenchmarkConfig:
    """Configuration for GEMM benchmark."""
    model: str = "Qwen/Qwen3-4B"
    dtype: str = "float16"
    # Total token counts to benchmark
    total_tokens_list: list[int] = field(
        default_factory=lambda: [512, 1024, 2048, 4096, 8192]
    )
    # Decode percentages to test
    decode_percentages: list[int] = field(
        default_factory=lambda: [0, 20, 40, 60, 80, 100]
    )
    decode_context_len: int = 256  # Context length for decode requests
    num_warmup: int = 3
    num_iterations: int = 10
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    gpu_memory_utilization: float = 0.9
    output_json: str = "gemm_by_decode_ratio.json"


@dataclass
class GEMMBenchmarkResult:
    """Result of a single GEMM benchmark."""
    decode_percentage: int
    total_tokens: int
    num_decode: int
    num_prefill_tokens: int
    mean_execute_time_ms: float
    std_execute_time_ms: float
    # Per-iteration times for detailed analysis
    iteration_times_ms: list[float] = field(default_factory=list)


class GPUTimer:
    """Accurate GPU timing using CUDA events."""
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        torch.cuda.synchronize()
        self.start_event.record()

    def stop(self) -> float:
        """Stop timing and return elapsed time in milliseconds."""
        self.end_event.record()
        torch.cuda.synchronize()
        return self.start_event.elapsed_time(self.end_event)


class GEMMBenchmark:
    """Benchmark for measuring GEMM time with different decode percentages."""

    def __init__(self, config: GEMMBenchmarkConfig):
        self.config = config
        self.engine_core = None
        self.results: list[GEMMBenchmarkResult] = []

    def setup(self):
        """Initialize EngineCore."""
        from vllm import SamplingParams
        from vllm.engine.arg_utils import EngineArgs
        from vllm.utils.torch_utils import set_default_torch_num_threads
        from vllm.v1.engine.core import EngineCore
        from vllm.v1.executor import Executor

        print(f"Initializing model: {self.config.model}")
        print(f"  dtype: {self.config.dtype}")
        print(f"  max_num_batched_tokens: {self.config.max_num_batched_tokens}")

        engine_args = EngineArgs(
            model=self.config.model,
            dtype=self.config.dtype,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            max_num_seqs=self.config.max_num_seqs,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            enforce_eager=True,
            block_size=16,
        )

        vllm_config = engine_args.create_engine_config()
        executor_class = Executor.get_class(vllm_config)

        with set_default_torch_num_threads(1):
            self.engine_core = EngineCore(
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=False,
            )

        print("Setup complete!")

    def _add_request(self, prompt_len: int, max_tokens: int = 1000) -> str:
        """Add a new request to the engine."""
        from vllm import SamplingParams
        from vllm.v1.engine import EngineCoreRequest

        req_id = f"req_{uuid.uuid4().hex[:8]}"
        prompt_token_ids = list(np.random.randint(100, 10000, size=prompt_len))

        request = EngineCoreRequest(
            request_id=req_id,
            prompt_token_ids=prompt_token_ids,
            mm_features=None,
            sampling_params=SamplingParams(max_tokens=max_tokens),
            pooling_params=None,
            eos_token_id=None,
            arrival_time=time.time(),
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        )
        self.engine_core.add_request(
            *self.engine_core.preprocess_add_request(request)
        )
        return req_id

    def _cleanup_all_requests(self):
        """Clean up all requests."""
        if self.engine_core is None:
            return

        running_ids = [req.request_id for req in self.engine_core.scheduler.running]
        waiting_ids = [req.request_id for req in self.engine_core.scheduler.waiting]
        all_ids = running_ids + waiting_ids

        if all_ids:
            self.engine_core.abort_requests(all_ids)

        # Drain remaining
        for _ in range(100):
            if not self.engine_core.scheduler.has_requests():
                break
            try:
                self.engine_core.step_fn()
            except Exception:
                break

        # Force clear
        self.engine_core.scheduler.running.clear()
        self.engine_core.scheduler.waiting.clear()
        self.engine_core.scheduler.requests.clear()

        try:
            input_batch = self.engine_core.model_executor.driver_worker.worker.model_runner.input_batch
            input_batch.clear()
        except Exception:
            pass

    def _prepare_decode_requests(self, num_decode: int, context_len: int) -> bool:
        """Prepare decode requests by completing their prefill phase."""
        if num_decode == 0:
            return True

        # Add requests
        for _ in range(num_decode):
            self._add_request(context_len, max_tokens=100000)

        # Run until all in decode phase
        tokens_to_process = context_len * num_decode
        max_steps = (tokens_to_process // self.config.max_num_batched_tokens) + 100

        for _ in range(max_steps):
            num_running = len(self.engine_core.scheduler.running)
            if num_running >= num_decode:
                all_in_decode = all(
                    req.num_computed_tokens >= req.num_prompt_tokens
                    for req in self.engine_core.scheduler.running
                )
                if all_in_decode:
                    return True
            if not self.engine_core.scheduler.has_requests():
                break
            try:
                self.engine_core.step_fn()
            except Exception as e:
                print(f"  Warning: Error during prefill: {e}")
                return False

        return False

    def _run_single_step(self) -> Optional[float]:
        """Run a single step and return execute time in ms."""
        if not self.engine_core.scheduler.has_requests():
            return None

        try:
            scheduler_output = self.engine_core.scheduler.schedule()
        except Exception as e:
            print(f"  Warning: Schedule failed: {e}")
            return None

        if scheduler_output.total_num_scheduled_tokens == 0:
            return None

        gpu_timer = GPUTimer()
        gpu_timer.start()
        model_output = self.engine_core.model_executor.execute_model(
            scheduler_output, non_block=False
        )
        if model_output is None:
            model_output = self.engine_core.model_executor.sample_tokens(None)
        execute_time = gpu_timer.stop()

        self.engine_core.scheduler.update_from_output(scheduler_output, model_output)

        return execute_time

    def run_benchmark(
        self,
        decode_percentage: int,
        total_tokens: int,
    ) -> Optional[GEMMBenchmarkResult]:
        """
        Run benchmark for specific decode percentage and total tokens.

        Args:
            decode_percentage: 0-100, percentage of tokens that are decode
            total_tokens: Total number of tokens in the batch
        """
        # Calculate batch composition
        num_decode = int(total_tokens * decode_percentage / 100)
        num_prefill_tokens = total_tokens - num_decode

        # Validate
        if num_decode > self.config.max_num_seqs:
            print(f"  Skipping: {num_decode} decode requests exceeds max_num_seqs")
            return None

        if total_tokens > self.config.max_num_batched_tokens:
            print(f"  Skipping: {total_tokens} tokens exceeds max_num_batched_tokens")
            return None

        desc = f"{decode_percentage}% decode ({num_decode}D + {num_prefill_tokens}P)"
        print(f"  Testing {desc}, total={total_tokens}...")

        iteration_times = []

        for i in range(self.config.num_warmup + self.config.num_iterations):
            self._cleanup_all_requests()

            # Prepare decode requests if needed
            if num_decode > 0:
                success = self._prepare_decode_requests(
                    num_decode, self.config.decode_context_len
                )
                if not success:
                    print(f"    Warning: Failed to prepare decode requests")
                    return None

            # Add prefill request if needed
            if num_prefill_tokens > 0:
                self._add_request(num_prefill_tokens, max_tokens=1)

            # Run one step and measure
            execute_time = self._run_single_step()

            if i >= self.config.num_warmup and execute_time is not None:
                iteration_times.append(execute_time)

        if not iteration_times:
            return None

        result = GEMMBenchmarkResult(
            decode_percentage=decode_percentage,
            total_tokens=total_tokens,
            num_decode=num_decode,
            num_prefill_tokens=num_prefill_tokens,
            mean_execute_time_ms=float(np.mean(iteration_times)),
            std_execute_time_ms=float(np.std(iteration_times)),
            iteration_times_ms=iteration_times,
        )

        print(f"    -> {result.mean_execute_time_ms:.3f}ms ± {result.std_execute_time_ms:.3f}ms")
        return result

    def run_all_benchmarks(self):
        """Run all benchmark configurations."""
        print(f"\n{'='*60}")
        print("GEMM BENCHMARK BY DECODE RATIO")
        print(f"{'='*60}")
        print(f"Total tokens: {self.config.total_tokens_list}")
        print(f"Decode percentages: {self.config.decode_percentages}")
        print(f"Decode context length: {self.config.decode_context_len}")

        # Global warmup
        print("\nGlobal warmup...")
        for _ in range(10):
            self._cleanup_all_requests()
            self._add_request(512, max_tokens=5)
            for _ in range(3):
                if not self.engine_core.scheduler.has_requests():
                    break
                try:
                    self.engine_core.step_fn()
                except Exception:
                    break
        self._cleanup_all_requests()
        torch.cuda.synchronize()
        print("Warmup complete.")

        # Run benchmarks
        for total_tokens in self.config.total_tokens_list:
            print(f"\n=== Total tokens: {total_tokens} ===")
            for decode_pct in self.config.decode_percentages:
                result = self.run_benchmark(decode_pct, total_tokens)
                if result:
                    self.results.append(result)

        print(f"\n{'='*60}")
        print(f"Completed {len(self.results)} benchmarks!")

    def save_results(self, filepath: str):
        """Save results to JSON file."""
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "config": asdict(self.config),
            "results": [asdict(r) for r in self.results],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {filepath}")

    def print_results(self):
        """Print results in table format."""
        print("\n" + "=" * 80)
        print("RESULTS: GEMM Time by Decode Ratio")
        print("=" * 80)
        print(f"{'Decode%':>8} {'TotalTok':>10} {'Decode':>8} {'Prefill':>8} {'Time(ms)':>12} {'Std':>8}")
        print("-" * 80)

        for r in sorted(self.results, key=lambda x: (x.decode_percentage, x.total_tokens)):
            print(f"{r.decode_percentage:>8} {r.total_tokens:>10} "
                  f"{r.num_decode:>8} {r.num_prefill_tokens:>8} "
                  f"{r.mean_execute_time_ms:>12.3f} {r.std_execute_time_ms:>8.3f}")

        print("=" * 80)


def plot_results(input_json: str, output_png: str):
    """Generate plot from benchmark results."""
    import matplotlib.pyplot as plt

    # Load data
    with open(input_json, "r") as f:
        data = json.load(f)

    results = data["results"]

    # Group by decode percentage
    by_decode_pct = {}
    for r in results:
        pct = r["decode_percentage"]
        if pct not in by_decode_pct:
            by_decode_pct[pct] = {"total_tokens": [], "time_ms": [], "std_ms": []}
        by_decode_pct[pct]["total_tokens"].append(r["total_tokens"])
        by_decode_pct[pct]["time_ms"].append(r["mean_execute_time_ms"])
        by_decode_pct[pct]["std_ms"].append(r["std_execute_time_ms"])

    # Get all unique total_tokens values and sort them
    all_tokens = sorted(set(r["total_tokens"] for r in results))
    # Create mapping from token count to index (for evenly spaced x-axis)
    token_to_idx = {t: i for i, t in enumerate(all_tokens)}

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color map for different decode percentages
    colors = plt.cm.viridis(np.linspace(0, 1, len(by_decode_pct)))

    # Plot each decode percentage as a line
    for (pct, pct_data), color in zip(sorted(by_decode_pct.items()), colors):
        # Sort by total tokens
        sorted_idx = np.argsort(pct_data["total_tokens"])
        tokens = np.array(pct_data["total_tokens"])[sorted_idx]
        y = np.array(pct_data["time_ms"])[sorted_idx]
        yerr = np.array(pct_data["std_ms"])[sorted_idx]

        # Convert tokens to evenly spaced indices
        x = np.array([token_to_idx[t] for t in tokens])

        # Determine label
        if pct == 0:
            label = "Pure Prefill"
        elif pct == 100:
            label = "Pure Decode"
        else:
            label = f"{pct}% Decode"

        ax.errorbar(x, y, yerr=yerr, marker='o', label=label, color=color,
                    capsize=3, capthick=1, linewidth=2, markersize=6)

    # Set x-axis ticks to show token counts at evenly spaced positions
    ax.set_xticks(range(len(all_tokens)))
    ax.set_xticklabels([str(t) for t in all_tokens])

    ax.set_xlabel("Total Tokens", fontsize=12)
    ax.set_ylabel("Execute Time (ms)", fontsize=12)
    ax.set_title("GEMM Time vs Total Tokens by Decode Percentage", fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_png}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GEMM time by decode ratio"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run subcommand
    run_parser = subparsers.add_parser("run", help="Run benchmarks")
    run_parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    run_parser.add_argument("--dtype", type=str, default="float16")
    run_parser.add_argument(
        "--total-tokens", type=str, default="512,1024,2048,4096,8192",
        help="Comma-separated list of total token counts"
    )
    run_parser.add_argument(
        "--decode-percentages", type=str, default="0,20,40,60,80,100",
        help="Comma-separated list of decode percentages"
    )
    run_parser.add_argument(
        "--decode-context-len", type=int, default=256,
        help="Context length for decode requests"
    )
    run_parser.add_argument("--num-warmup", type=int, default=3)
    run_parser.add_argument("--num-iterations", type=int, default=10)
    run_parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    run_parser.add_argument("--max-num-seqs", type=int, default=512)
    run_parser.add_argument("--output-json", type=str, default="gemm_by_decode_ratio.json")

    # Plot subcommand
    plot_parser = subparsers.add_parser("plot", help="Generate plot from results")
    plot_parser.add_argument("--input-json", type=str, required=True)
    plot_parser.add_argument("--output-png", type=str, default="gemm_by_decode_ratio.png")

    args = parser.parse_args()

    if args.command == "run":
        config = GEMMBenchmarkConfig(
            model=args.model,
            dtype=args.dtype,
            total_tokens_list=[int(x) for x in args.total_tokens.split(",")],
            decode_percentages=[int(x) for x in args.decode_percentages.split(",")],
            decode_context_len=args.decode_context_len,
            num_warmup=args.num_warmup,
            num_iterations=args.num_iterations,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            output_json=args.output_json,
        )

        benchmark = GEMMBenchmark(config)
        benchmark.setup()
        benchmark.run_all_benchmarks()
        benchmark.print_results()
        benchmark.save_results(args.output_json)

    elif args.command == "plot":
        plot_results(args.input_json, args.output_png)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
