#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark GEMM and Attention time separately using nsys profiling.

This script measures GEMM kernel time and Attention kernel time separately
for different batch compositions (decode percentages).

Usage:
    # Step 1: Run benchmark to collect nsys profiles
    CUDA_VISIBLE_DEVICES=3 python benchmark_gemm_attention_separate.py run \
        --model Qwen/Qwen3-4B \
        --total-tokens 256,512,1024,2048 \
        --decode-percentages 0,20,40,60,80,100 \
        --output-dir ./nsys_profiles

    # Step 2: Parse nsys profiles and generate plots
    python benchmark_gemm_attention_separate.py plot \
        --input-dir ./nsys_profiles \
        --output-dir ./plots
"""

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Add vllm project root to path
VLLM_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(VLLM_ROOT))


@dataclass
class KernelTimes:
    """Kernel timing breakdown for a single benchmark."""
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
    decode_context_len: int = 256
    num_warmup: int = 2
    num_iterations: int = 3
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 5000
    output_dir: str = "./nsys_profiles"


def parse_nsys_sqlite(sqlite_path: str) -> KernelTimes:
    """
    Parse nsys sqlite output to extract GEMM and Attention kernel times.

    GEMM kernels: contain 'gemm', 'cutlass', 'cublas', 'xmma' in name
    Attention kernels: contain 'flash', 'attn', 'fmha', 'attention' in name
    """
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    kernel_data = []

    # Try to get kernel names by joining with StringIds table
    try:
        cursor.execute("""
            SELECT s.value as name, SUM(k.end - k.start) as total_time
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.demangledName = s.id
            GROUP BY k.demangledName
        """)
        kernel_data = cursor.fetchall()
    except sqlite3.OperationalError:
        pass

    if not kernel_data:
        # Try with shortName
        try:
            cursor.execute("""
                SELECT s.value as name, SUM(k.end - k.start) as total_time
                FROM CUPTI_ACTIVITY_KIND_KERNEL k
                JOIN StringIds s ON k.shortName = s.id
                GROUP BY k.shortName
            """)
            kernel_data = cursor.fetchall()
        except sqlite3.OperationalError:
            pass

    if not kernel_data:
        # Try direct column if it's already a string
        try:
            cursor.execute("""
                SELECT demangledName as name, SUM(end - start) as total_time
                FROM CUPTI_ACTIVITY_KIND_KERNEL
                WHERE typeof(demangledName) = 'text'
                GROUP BY demangledName
            """)
            kernel_data = cursor.fetchall()
        except sqlite3.OperationalError:
            pass

    conn.close()

    gemm_time_ns = 0
    attention_time_ns = 0
    other_time_ns = 0

    # Patterns for GEMM kernels (linear layer matrix multiplications)
    # nvjet_hsh = NVIDIA JIT matrix multiply kernels
    # cutlass = NVIDIA CUTLASS GEMM (but NOT FlashAttn which also uses cutlass)
    gemm_patterns = ['nvjet', 'gemm', 'xmma', 'cublas', 'matmul', 'sm80_xmma', 'ampere']
    # Patterns for Attention kernels
    attn_patterns = ['flash', 'fmha', 'FlashAttn', 'attention']

    for name, total_time in kernel_data:
        if name is None:
            continue
        name_lower = name.lower()

        # Check attention first (FlashAttn uses cutlass but is attention)
        is_attn = any(p.lower() in name_lower for p in attn_patterns)
        is_gemm = any(p.lower() in name_lower for p in gemm_patterns) and not is_attn

        if is_attn:
            attention_time_ns += total_time
        elif is_gemm:
            gemm_time_ns += total_time
        else:
            other_time_ns += total_time

    # Convert ns to ms
    return KernelTimes(
        gemm_time_ms=gemm_time_ns / 1e6,
        attention_time_ms=attention_time_ns / 1e6,
        other_time_ms=other_time_ns / 1e6,
        total_kernel_time_ms=(gemm_time_ns + attention_time_ns + other_time_ns) / 1e6,
    )


def run_single_benchmark_with_nsys(
    model: str,
    dtype: str,
    num_decode: int,
    num_prefill_tokens: int,
    decode_context_len: int,
    num_warmup: int,
    num_iterations: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    output_prefix: str,
) -> Optional[KernelTimes]:
    """
    Run a single benchmark configuration with nsys profiling.
    """
    # Create a temporary Python script for this specific benchmark
    script_content = f'''
import os
import sys
import time
import uuid
import numpy as np
import torch

# Disable profiling scopes to avoid overhead
os.environ.pop("VLLM_NVTX_SCOPES_FOR_PROFILING", None)
os.environ.pop("VLLM_CUSTOM_SCOPES_FOR_PROFILING", None)

from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.torch_utils import set_default_torch_num_threads
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor import Executor

def add_request(engine_core, prompt_len, max_tokens=1000):
    req_id = f"req_{{uuid.uuid4().hex[:8]}}"
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
    engine_core.add_request(*engine_core.preprocess_add_request(request))
    return req_id

def cleanup_all_requests(engine_core):
    running_ids = [req.request_id for req in engine_core.scheduler.running]
    waiting_ids = [req.request_id for req in engine_core.scheduler.waiting]
    all_ids = running_ids + waiting_ids
    if all_ids:
        engine_core.abort_requests(all_ids)
    for _ in range(100):
        if not engine_core.scheduler.has_requests():
            break
        try:
            engine_core.step_fn()
        except Exception:
            break
    engine_core.scheduler.running.clear()
    engine_core.scheduler.waiting.clear()
    engine_core.scheduler.requests.clear()
    try:
        input_batch = engine_core.model_executor.driver_worker.worker.model_runner.input_batch
        input_batch.clear()
    except Exception:
        pass

def prepare_decode_requests(engine_core, num_decode, context_len, max_num_batched_tokens):
    if num_decode == 0:
        return True
    for _ in range(num_decode):
        add_request(engine_core, context_len, max_tokens=100000)
    tokens_to_process = context_len * num_decode
    max_steps = (tokens_to_process // max_num_batched_tokens) + 100
    for _ in range(max_steps):
        num_running = len(engine_core.scheduler.running)
        if num_running >= num_decode:
            all_in_decode = all(
                req.num_computed_tokens >= req.num_prompt_tokens
                for req in engine_core.scheduler.running
            )
            if all_in_decode:
                return True
        if not engine_core.scheduler.has_requests():
            break
        try:
            engine_core.step_fn()
        except Exception as e:
            print(f"Error during prefill: {{e}}", file=sys.stderr)
            return False
    return False

def run_single_step(engine_core):
    if not engine_core.scheduler.has_requests():
        return None
    try:
        scheduler_output = engine_core.scheduler.schedule()
    except Exception as e:
        print(f"Schedule failed: {{e}}", file=sys.stderr)
        return None
    if scheduler_output.total_num_scheduled_tokens == 0:
        return None
    torch.cuda.synchronize()
    model_output = engine_core.model_executor.execute_model(scheduler_output, non_block=False)
    if model_output is None:
        model_output = engine_core.model_executor.sample_tokens(None)
    torch.cuda.synchronize()
    engine_core.scheduler.update_from_output(scheduler_output, model_output)
    return scheduler_output.total_num_scheduled_tokens

# Initialize engine
engine_args = EngineArgs(
    model="{model}",
    dtype="{dtype}",
    max_num_batched_tokens={max_num_batched_tokens},
    max_num_seqs={max_num_seqs},
    gpu_memory_utilization=0.9,
    enable_chunked_prefill=True,
    enable_prefix_caching=False,
    enforce_eager=True,
    block_size=16,
)
vllm_config = engine_args.create_engine_config()
executor_class = Executor.get_class(vllm_config)

with set_default_torch_num_threads(1):
    engine_core = EngineCore(
        vllm_config=vllm_config,
        executor_class=executor_class,
        log_stats=False,
    )

# Warmup
for _ in range({num_warmup}):
    cleanup_all_requests(engine_core)
    add_request(engine_core, 512, max_tokens=5)
    for _ in range(3):
        if not engine_core.scheduler.has_requests():
            break
        try:
            engine_core.step_fn()
        except Exception:
            break
cleanup_all_requests(engine_core)
torch.cuda.synchronize()

# Run benchmark iterations
num_decode = {num_decode}
num_prefill_tokens = {num_prefill_tokens}
decode_context_len = {decode_context_len}

for i in range({num_iterations}):
    cleanup_all_requests(engine_core)

    # Prepare decode requests
    if num_decode > 0:
        success = prepare_decode_requests(engine_core, num_decode, decode_context_len, {max_num_batched_tokens})
        if not success:
            print("Failed to prepare decode requests", file=sys.stderr)
            sys.exit(1)

    # Add prefill request
    if num_prefill_tokens > 0:
        add_request(engine_core, num_prefill_tokens, max_tokens=1)

    # Run step and sync
    tokens = run_single_step(engine_core)

    if tokens is None:
        print("No tokens scheduled", file=sys.stderr)
        sys.exit(1)

print("Benchmark completed successfully")
'''

    # Write script to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name

    try:
        # Run with nsys - only capture regions marked by cudaProfilerStart/Stop
        nsys_output = f"{output_prefix}"
        cmd = [
            "nsys", "profile",
            "-o", nsys_output,
            "--force-overwrite", "true",
            "--export", "sqlite",
            "-c", "cudaProfilerApi",
            "-t", "cuda",
            "python", script_path
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            print(f"nsys failed: {result.stderr}")
            return None

        # Parse the sqlite output
        sqlite_path = f"{nsys_output}.sqlite"
        if not os.path.exists(sqlite_path):
            print(f"sqlite file not found: {sqlite_path}")
            return None

        kernel_times = parse_nsys_sqlite(sqlite_path)
        return kernel_times

    finally:
        os.unlink(script_path)


def run_benchmarks_simple(config: BenchmarkConfig):
    """
    Run benchmarks using a simpler approach - profile each configuration separately.
    """
    os.makedirs(config.output_dir, exist_ok=True)
    results = []

    for total_tokens in config.total_tokens_list:
        print(f"\n=== Total tokens: {total_tokens} ===")

        for decode_pct in config.decode_percentages:
            num_decode = int(total_tokens * decode_pct / 100)
            num_prefill_tokens = total_tokens - num_decode

            if num_decode > config.max_num_seqs:
                print(f"  Skipping {decode_pct}% decode: {num_decode} exceeds max_num_seqs")
                continue

            print(f"  Testing {decode_pct}% decode ({num_decode}D + {num_prefill_tokens}P)...")

            output_prefix = os.path.join(
                config.output_dir,
                f"profile_t{total_tokens}_d{decode_pct}"
            )

            kernel_times = run_single_benchmark_with_nsys(
                model=config.model,
                dtype=config.dtype,
                num_decode=num_decode,
                num_prefill_tokens=num_prefill_tokens,
                decode_context_len=config.decode_context_len,
                num_warmup=config.num_warmup,
                num_iterations=config.num_iterations,
                max_num_batched_tokens=config.max_num_batched_tokens,
                max_num_seqs=config.max_num_seqs,
                output_prefix=output_prefix,
            )

            if kernel_times:
                result = BenchmarkResult(
                    decode_percentage=decode_pct,
                    total_tokens=total_tokens,
                    num_decode=num_decode,
                    num_prefill_tokens=num_prefill_tokens,
                    gemm_time_ms=kernel_times.gemm_time_ms / config.num_iterations,
                    attention_time_ms=kernel_times.attention_time_ms / config.num_iterations,
                    other_time_ms=kernel_times.other_time_ms / config.num_iterations,
                    total_kernel_time_ms=kernel_times.total_kernel_time_ms / config.num_iterations,
                )
                results.append(result)
                print(f"    GEMM: {result.gemm_time_ms:.3f}ms, "
                      f"Attention: {result.attention_time_ms:.3f}ms, "
                      f"Other: {result.other_time_ms:.3f}ms")
            else:
                print(f"    Failed to get kernel times")

    # Save results
    results_path = os.path.join(config.output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump({
            "config": asdict(config),
            "results": [asdict(r) for r in results],
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


def plot_results(input_dir: str, output_dir: str):
    """Generate GEMM and Attention time plots from results."""
    import matplotlib.pyplot as plt

    # Load results
    results_path = os.path.join(input_dir, "results.json")
    with open(results_path, 'r') as f:
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
                "total_time_ms": [],
            }
        by_decode_pct[pct]["total_tokens"].append(r["total_tokens"])
        by_decode_pct[pct]["gemm_time_ms"].append(r["gemm_time_ms"])
        by_decode_pct[pct]["attention_time_ms"].append(r["attention_time_ms"])
        by_decode_pct[pct]["other_time_ms"].append(r["other_time_ms"])
        by_decode_pct[pct]["total_time_ms"].append(r["total_kernel_time_ms"])

    # Get all unique total_tokens values
    all_tokens = sorted(set(r["total_tokens"] for r in results))
    token_to_idx = {t: i for i, t in enumerate(all_tokens)}

    # Color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(by_decode_pct)))

    def make_plot(time_key: str, title: str, ylabel: str, output_file: str):
        fig, ax = plt.subplots(figsize=(10, 6))

        for (pct, pct_data), color in zip(sorted(by_decode_pct.items()), colors):
            sorted_idx = np.argsort(pct_data["total_tokens"])
            tokens = np.array(pct_data["total_tokens"])[sorted_idx]
            y = np.array(pct_data[time_key])[sorted_idx]
            x = np.array([token_to_idx[t] for t in tokens])

            if pct == 0:
                label = "Pure Prefill"
            elif pct == 100:
                label = "Pure Decode"
            else:
                label = f"{pct}% Decode"

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

    # Generate plots
    make_plot("gemm_time_ms", "GEMM Time vs Total Tokens by Decode Percentage",
              "GEMM Time (ms)", "gemm_time.png")
    make_plot("attention_time_ms", "Attention Time vs Total Tokens by Decode Percentage",
              "Attention Time (ms)", "attention_time.png")
    make_plot("total_time_ms", "Total Kernel Time vs Total Tokens by Decode Percentage",
              "Total Kernel Time (ms)", "total_kernel_time.png")

    # Also create a combined stacked bar chart for one token count
    if results:
        # Pick the largest token count with most data
        token_counts = {}
        for r in results:
            t = r["total_tokens"]
            token_counts[t] = token_counts.get(t, 0) + 1
        best_token = max(token_counts.keys(), key=lambda t: token_counts[t])

        fig, ax = plt.subplots(figsize=(10, 6))

        # Filter results for this token count
        filtered = [r for r in results if r["total_tokens"] == best_token]
        filtered.sort(key=lambda r: r["decode_percentage"])

        x = range(len(filtered))
        labels = [f"{r['decode_percentage']}%" for r in filtered]
        gemm = [r["gemm_time_ms"] for r in filtered]
        attn = [r["attention_time_ms"] for r in filtered]
        other = [r["other_time_ms"] for r in filtered]

        ax.bar(x, gemm, label='GEMM', color='steelblue')
        ax.bar(x, attn, bottom=gemm, label='Attention', color='coral')
        ax.bar(x, other, bottom=[g+a for g, a in zip(gemm, attn)], label='Other', color='gray')

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Decode Percentage", fontsize=12)
        ax.set_ylabel("Time (ms)", fontsize=12)
        ax.set_title(f"Kernel Time Breakdown at {best_token} Total Tokens", fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "kernel_breakdown.png"), dpi=150, bbox_inches='tight')
        print(f"Plot saved to {os.path.join(output_dir, 'kernel_breakdown.png')}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Benchmark GEMM and Attention separately")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run subcommand
    run_parser = subparsers.add_parser("run", help="Run benchmarks with nsys")
    run_parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    run_parser.add_argument("--dtype", type=str, default="float16")
    run_parser.add_argument("--total-tokens", type=str, default="256,512,1024,2048")
    run_parser.add_argument("--decode-percentages", type=str, default="0,20,40,60,80,100")
    run_parser.add_argument("--decode-context-len", type=int, default=256)
    run_parser.add_argument("--num-warmup", type=int, default=2)
    run_parser.add_argument("--num-iterations", type=int, default=3)
    run_parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    run_parser.add_argument("--max-num-seqs", type=int, default=5000)
    run_parser.add_argument("--output-dir", type=str, default="./nsys_profiles")

    # Plot subcommand
    plot_parser = subparsers.add_parser("plot", help="Generate plots from results")
    plot_parser.add_argument("--input-dir", type=str, required=True)
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
            output_dir=args.output_dir,
        )
        run_benchmarks_simple(config)

    elif args.command == "plot":
        plot_results(args.input_dir, args.output_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
