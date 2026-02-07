#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark GEMM and Attention time separately using PyTorch Profiler.

This script measures GEMM kernel time and Attention kernel time separately
for different batch compositions (decode percentages) using torch.profiler.

Usage:
    CUDA_VISIBLE_DEVICES=3 python benchmark_gemm_attention_v2.py run \
        --model Qwen/Qwen3-4B \
        --total-tokens 256,512,1024,2048 \
        --decode-percentages 0,20,40,60,80,100 \
        --output-json gemm_attention_results.json

    python benchmark_gemm_attention_v2.py plot \
        --input-json gemm_attention_results.json \
        --output-dir ./plots
"""

import argparse
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity

# Add vllm project root to path
VLLM_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(VLLM_ROOT))


@dataclass
class KernelTimes:
    """Kernel timing breakdown for a single step."""
    gemm_time_ms: float = 0.0
    attention_time_ms: float = 0.0
    other_time_ms: float = 0.0
    total_kernel_time_ms: float = 0.0


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    decode_percentage: int
    total_tokens: int
    num_decode: int
    num_prefill_tokens: int
    gemm_time_ms: float
    attention_time_ms: float
    other_time_ms: float
    total_kernel_time_ms: float
    # Standard deviations
    gemm_time_std: float = 0.0
    attention_time_std: float = 0.0


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark."""
    model: str = "Qwen/Qwen3-4B"
    dtype: str = "float16"
    total_tokens_list: list[int] = field(
        default_factory=lambda: [256, 512, 1024, 2048]
    )
    decode_percentages: list[int] = field(
        default_factory=lambda: [0, 20, 40, 60, 80, 100]
    )
    decode_context_len: int = 32
    num_warmup: int = 3
    num_iterations: int = 5
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 5000
    output_json: str = "gemm_attention_results.json"


def classify_kernel(name: str) -> str:
    """Classify a CUDA kernel as GEMM, Attention, or Other."""
    name_lower = name.lower()

    # Attention patterns (check first as some attention uses cutlass)
    attn_patterns = ['flash', 'fmha', 'flashattn', 'attention']
    if any(p in name_lower for p in attn_patterns):
        return 'attention'

    # GEMM patterns
    gemm_patterns = ['nvjet', 'gemm', 'xmma', 'cublas', 'matmul', 'ampere']
    if any(p in name_lower for p in gemm_patterns):
        return 'gemm'

    return 'other'


def extract_kernel_times_from_profiler(prof) -> KernelTimes:
    """Extract GEMM and Attention kernel times from torch.profiler."""
    gemm_time_us = 0
    attention_time_us = 0
    other_time_us = 0

    for event in prof.key_averages():
        # Use self_device_time_total for actual CUDA kernel execution time
        # (self_cuda_time_total is deprecated in newer PyTorch versions)
        cuda_time = getattr(event, 'self_device_time_total', 0)
        if cuda_time is None or cuda_time <= 0:
            continue

        name = event.key
        category = classify_kernel(name)

        if category == 'gemm':
            gemm_time_us += cuda_time
        elif category == 'attention':
            attention_time_us += cuda_time
        else:
            other_time_us += cuda_time

    return KernelTimes(
        gemm_time_ms=gemm_time_us / 1000,
        attention_time_ms=attention_time_us / 1000,
        other_time_ms=other_time_us / 1000,
        total_kernel_time_ms=(gemm_time_us + attention_time_us + other_time_us) / 1000,
    )


class GEMMAttentionBenchmark:
    """Benchmark for measuring GEMM and Attention time separately."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.engine_core = None
        self.results: list[BenchmarkResult] = []

    def setup(self):
        """Initialize EngineCore."""
        from vllm import SamplingParams
        from vllm.engine.arg_utils import EngineArgs
        from vllm.utils.torch_utils import set_default_torch_num_threads
        from vllm.v1.engine.core import EngineCore
        from vllm.v1.executor import Executor

        print(f"Initializing model: {self.config.model}")

        engine_args = EngineArgs(
            model=self.config.model,
            dtype=self.config.dtype,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            max_num_seqs=self.config.max_num_seqs,
            gpu_memory_utilization=0.9,
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

        for _ in range(100):
            if not self.engine_core.scheduler.has_requests():
                break
            try:
                self.engine_core.step_fn()
            except Exception:
                break

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

        for _ in range(num_decode):
            self._add_request(context_len, max_tokens=100000)

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
                print(f"  Error during prefill: {e}")
                return False

        return False

    def _run_single_step_with_profiler(self) -> Optional[KernelTimes]:
        """Run a single step and profile kernel times."""
        if not self.engine_core.scheduler.has_requests():
            return None

        try:
            scheduler_output = self.engine_core.scheduler.schedule()
        except Exception as e:
            print(f"  Schedule failed: {e}")
            return None

        if scheduler_output.total_num_scheduled_tokens == 0:
            return None

        # Profile the execute_model call
        with profile(
            activities=[ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
        ) as prof:
            torch.cuda.synchronize()
            model_output = self.engine_core.model_executor.execute_model(
                scheduler_output, non_block=False
            )
            if model_output is None:
                model_output = self.engine_core.model_executor.sample_tokens(None)
            torch.cuda.synchronize()

        self.engine_core.scheduler.update_from_output(scheduler_output, model_output)

        return extract_kernel_times_from_profiler(prof)

    def run_benchmark(
        self,
        decode_percentage: int,
        total_tokens: int,
    ) -> Optional[BenchmarkResult]:
        """Run benchmark for specific decode percentage and total tokens."""
        num_decode = int(total_tokens * decode_percentage / 100)
        num_prefill_tokens = total_tokens - num_decode

        if num_decode > self.config.max_num_seqs:
            print(f"  Skipping: {num_decode} decode requests exceeds max_num_seqs")
            return None

        desc = f"{decode_percentage}% decode ({num_decode}D + {num_prefill_tokens}P)"
        print(f"  Testing {desc}, total={total_tokens}...")

        gemm_times = []
        attention_times = []
        other_times = []

        for i in range(self.config.num_warmup + self.config.num_iterations):
            self._cleanup_all_requests()

            # Prepare decode requests
            if num_decode > 0:
                success = self._prepare_decode_requests(
                    num_decode, self.config.decode_context_len
                )
                if not success:
                    print(f"    Warning: Failed to prepare decode requests")
                    return None

            # Add prefill request
            if num_prefill_tokens > 0:
                self._add_request(num_prefill_tokens, max_tokens=1)

            # Run one step with profiling
            kernel_times = self._run_single_step_with_profiler()

            if i >= self.config.num_warmup and kernel_times is not None:
                gemm_times.append(kernel_times.gemm_time_ms)
                attention_times.append(kernel_times.attention_time_ms)
                other_times.append(kernel_times.other_time_ms)

        if not gemm_times:
            return None

        result = BenchmarkResult(
            decode_percentage=decode_percentage,
            total_tokens=total_tokens,
            num_decode=num_decode,
            num_prefill_tokens=num_prefill_tokens,
            gemm_time_ms=float(np.mean(gemm_times)),
            attention_time_ms=float(np.mean(attention_times)),
            other_time_ms=float(np.mean(other_times)),
            total_kernel_time_ms=float(np.mean(gemm_times) + np.mean(attention_times) + np.mean(other_times)),
            gemm_time_std=float(np.std(gemm_times)),
            attention_time_std=float(np.std(attention_times)),
        )

        print(f"    GEMM: {result.gemm_time_ms:.3f}±{result.gemm_time_std:.3f}ms, "
              f"Attn: {result.attention_time_ms:.3f}±{result.attention_time_std:.3f}ms, "
              f"Other: {result.other_time_ms:.3f}ms")
        return result

    def run_all_benchmarks(self):
        """Run all benchmark configurations."""
        print(f"\n{'='*60}")
        print("GEMM/Attention Benchmark")
        print(f"{'='*60}")
        print(f"Total tokens: {self.config.total_tokens_list}")
        print(f"Decode percentages: {self.config.decode_percentages}")

        # Global warmup
        print("\nGlobal warmup...")
        for _ in range(5):
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
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {filepath}")


def plot_results(input_json: str, output_dir: str):
    """Generate GEMM and Attention time plots from results."""
    import matplotlib.pyplot as plt

    with open(input_json, 'r') as f:
        data = json.load(f)

    results = data["results"]
    os.makedirs(output_dir, exist_ok=True)

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
                "gemm_std": [],
                "attention_std": [],
            }
        by_decode_pct[pct]["total_tokens"].append(r["total_tokens"])
        by_decode_pct[pct]["gemm_time_ms"].append(r["gemm_time_ms"])
        by_decode_pct[pct]["attention_time_ms"].append(r["attention_time_ms"])
        by_decode_pct[pct]["other_time_ms"].append(r.get("other_time_ms", 0))
        by_decode_pct[pct]["total_kernel_time_ms"].append(r.get("total_kernel_time_ms", 0))
        by_decode_pct[pct]["gemm_std"].append(r.get("gemm_time_std", 0))
        by_decode_pct[pct]["attention_std"].append(r.get("attention_time_std", 0))

    all_tokens = sorted(set(r["total_tokens"] for r in results))
    token_to_idx = {t: i for i, t in enumerate(all_tokens)}

    colors = plt.cm.viridis(np.linspace(0, 1, len(by_decode_pct)))

    def make_plot(time_key: str, std_key: str, title: str, ylabel: str, output_file: str):
        fig, ax = plt.subplots(figsize=(10, 6))

        for (pct, pct_data), color in zip(sorted(by_decode_pct.items()), colors):
            sorted_idx = np.argsort(pct_data["total_tokens"])
            tokens = np.array(pct_data["total_tokens"])[sorted_idx]
            y = np.array(pct_data[time_key])[sorted_idx]
            yerr = np.array(pct_data[std_key])[sorted_idx] if std_key else None
            x = np.array([token_to_idx[t] for t in tokens])

            if pct == 0:
                label = "Pure Prefill"
            elif pct == 100:
                label = "Pure Decode"
            else:
                label = f"{pct}% Decode"

            if yerr is not None and yerr.sum() > 0:
                ax.errorbar(x, y, yerr=yerr, marker='o', label=label, color=color,
                           capsize=3, linewidth=2, markersize=6)
            else:
                ax.plot(x, y, marker='o', label=label, color=color, linewidth=2, markersize=6)

        ax.set_xticks(range(len(all_tokens)))
        ax.set_xticklabels([str(t) for t in all_tokens])
        ax.set_xlabel("Total Tokens", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, output_file), dpi=150, bbox_inches='tight')
        print(f"Plot saved to {os.path.join(output_dir, output_file)}")
        plt.close()

    make_plot("gemm_time_ms", "gemm_std",
              "GEMM Time vs Total Tokens by Decode Percentage",
              "GEMM Time (ms)", "gemm_time.png")
    make_plot("attention_time_ms", "attention_std",
              "Attention Time vs Total Tokens by Decode Percentage",
              "Attention Time (ms)", "attention_time.png")

    # Plot total kernel time
    make_plot("total_kernel_time_ms", None,
              "Total Kernel Time vs Total Tokens by Decode Percentage",
              "Total Kernel Time (ms)", "total_kernel_time.png")

    # Plot kernel breakdown (stacked bar chart)
    def plot_kernel_breakdown():
        fig, axes = plt.subplots(1, len(all_tokens), figsize=(4 * len(all_tokens), 6), sharey=True)
        if len(all_tokens) == 1:
            axes = [axes]

        bar_width = 0.8
        kernel_colors = {'GEMM': '#2ecc71', 'Attention': '#e74c3c', 'Other': '#3498db'}

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

            ax.bar(x, gemm_vals, bar_width, label='GEMM', color=kernel_colors['GEMM'])
            ax.bar(x, attn_vals, bar_width, bottom=gemm_vals, label='Attention', color=kernel_colors['Attention'])
            ax.bar(x, other_vals, bar_width, bottom=gemm_vals + attn_vals, label='Other', color=kernel_colors['Other'])

            ax.set_xlabel("Decode %", fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{p}%" for p in decode_pcts], rotation=45)
            ax.set_title(f"Total Tokens = {total_tok}", fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')

        axes[0].set_ylabel("Kernel Time (ms)", fontsize=12)
        axes[-1].legend(loc='upper left', fontsize=10)

        fig.suptitle("Kernel Time Breakdown by Decode Percentage", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "kernel_breakdown.png"), dpi=150, bbox_inches='tight')
        print(f"Plot saved to {os.path.join(output_dir, 'kernel_breakdown.png')}")
        plt.close()

    plot_kernel_breakdown()


def main():
    parser = argparse.ArgumentParser(description="Benchmark GEMM and Attention separately")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    run_parser = subparsers.add_parser("run", help="Run benchmarks")
    run_parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    run_parser.add_argument("--dtype", type=str, default="float16")
    run_parser.add_argument("--total-tokens", type=str, default="256,512,1024,2048")
    run_parser.add_argument("--decode-percentages", type=str, default="0,20,40,60,80,100")
    run_parser.add_argument("--decode-context-len", type=int, default=256)
    run_parser.add_argument("--num-warmup", type=int, default=3)
    run_parser.add_argument("--num-iterations", type=int, default=5)
    run_parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    run_parser.add_argument("--max-num-seqs", type=int, default=5000)
    run_parser.add_argument("--output-json", type=str, default="gemm_attention_results.json")

    plot_parser = subparsers.add_parser("plot", help="Generate plots from results")
    plot_parser.add_argument("--input-json", type=str, required=True)
    plot_parser.add_argument("--output-dir", type=str, default="./plots")

    args = parser.parse_args()

    if args.command == "run":
        config = BenchmarkConfig(
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

        benchmark = GEMMAttentionBenchmark(config)
        benchmark.setup()
        benchmark.run_all_benchmarks()
        benchmark.save_results(args.output_json)

    elif args.command == "plot":
        plot_results(args.input_json, args.output_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
