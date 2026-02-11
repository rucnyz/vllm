# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark script for measuring vLLM full step execution times.

This script measures the FULL step execution time including:
- scheduler.schedule()
- model_executor.execute_model()
- model_executor.sample_tokens()
- scheduler.update_from_output()

Unlike benchmark_batch_combinations.py which only measures execute_model(),
this script captures the complete iteration overhead.

Usage:
    python benchmark_batch_combinations_full_step.py \
        --model Qwen/Qwen3-4B \
        --prefill-sizes 256,512,1024,2048,4096 \
        --decode-context-lens 256 \
        --decode-counts 1,2,4,8,16,32,64 \
        --num-warmup 5 \
        --num-iterations 20 \
        --output-json results_full_step.json
"""

import sys
from pathlib import Path

# Add vllm project root to path so we can import vllm modules
VLLM_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(VLLM_ROOT))

import argparse
import json
import time
import uuid
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from tqdm import tqdm

from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.torch_utils import set_default_torch_num_threads
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor import Executor


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark."""
    model: str = "facebook/opt-125m"
    dtype: str = "float16"
    # For mixed batch tests
    prefill_chunk_sizes: list[int] = field(
        default_factory=lambda: [512, 1024, 2048, 4096]
    )
    decode_counts: list[int] = field(
        default_factory=lambda: [0, 1, 2, 4, 8, 16, 32]
    )
    decode_context_lens: list[int] = field(
        default_factory=lambda: [512, 1024, 2048, 4096]
    )  # Context lengths for decode requests
    # For pure tests (separate ranges for better comparison)
    pure_prefill_sizes: list[int] | None = None  # If None, use prefill_chunk_sizes
    pure_decode_counts: list[int] | None = None  # If None, use decode_counts
    num_warmup: int = 5
    num_iterations: int = 20
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 256
    block_size: int = 16
    gpu_memory_utilization: float = 0.9
    output_json: str | None = None
    enforce_eager: bool = True


@dataclass
class BenchmarkResult:
    """Result of a single benchmark measurement."""
    description: str
    num_decode: int
    num_prefill: int
    prefill_chunk_size: int
    decode_context_len: int  # Context length for decode requests
    total_tokens: int
    # Full step timing (schedule + execute + update)
    mean_time_ms: float
    std_time_ms: float
    throughput_tokens_per_sec: float
    # Breakdown timing
    mean_schedule_time_ms: float
    std_schedule_time_ms: float
    mean_execute_time_ms: float
    std_execute_time_ms: float
    mean_update_time_ms: float
    std_update_time_ms: float


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


class CPUTimer:
    """CPU timing using time.perf_counter for non-GPU operations."""

    def __init__(self):
        self._start_time = 0.0

    def start(self):
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop timing and return elapsed time in milliseconds."""
        return (time.perf_counter() - self._start_time) * 1000


@dataclass
class StepTimingBreakdown:
    """Timing breakdown for a single step."""
    schedule_time_ms: float
    execute_time_ms: float
    update_time_ms: float
    total_time_ms: float
    num_decode: int
    num_prefill: int
    total_tokens: int


class BatchBenchmark:
    """Main benchmark class that measures full step execution time."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.engine_core: EngineCore | None = None
        self.results: list[BenchmarkResult] = []

    def setup(self):
        """Initialize EngineCore."""
        print(f"Initializing model: {self.config.model}")
        print(f"  dtype: {self.config.dtype}")
        print(f"  max_num_batched_tokens: {self.config.max_num_batched_tokens}")
        print(f"  max_num_seqs: {self.config.max_num_seqs}")
        print(f"  gpu_memory_utilization: {self.config.gpu_memory_utilization}")

        engine_args = EngineArgs(
            model=self.config.model,
            dtype=self.config.dtype,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            max_num_seqs=self.config.max_num_seqs,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            enforce_eager=self.config.enforce_eager,
            block_size=self.config.block_size,
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
        assert self.engine_core is not None

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

    def _run_single_step_with_breakdown(self) -> StepTimingBreakdown | None:
        """
        Run a single step and measure time with breakdown.

        Returns:
            StepTimingBreakdown or None if nothing scheduled
        """
        assert self.engine_core is not None

        cpu_timer = CPUTimer()
        gpu_timer = GPUTimer()

        # 1. Schedule (CPU-bound)
        cpu_timer.start()
        scheduler_output = self.engine_core.scheduler.schedule()
        schedule_time = cpu_timer.stop()

        if scheduler_output.total_num_scheduled_tokens == 0:
            return None

        # Count decode vs prefill
        num_decode = 0
        num_prefill = 0
        for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
            if num_tokens == 1:
                num_decode += 1
            else:
                num_prefill += 1

        total_tokens = scheduler_output.total_num_scheduled_tokens

        # 2. Execute model (GPU-bound)
        gpu_timer.start()
        model_output = self.engine_core.model_executor.execute_model(
            scheduler_output, non_block=False
        )
        # If execute_model returns None, we need to call sample_tokens
        if model_output is None:
            model_output = self.engine_core.model_executor.sample_tokens(None)
        execute_time = gpu_timer.stop()

        # 3. Update scheduler state (CPU-bound)
        cpu_timer.start()
        self.engine_core.scheduler.update_from_output(scheduler_output, model_output)
        update_time = cpu_timer.stop()

        total_time = schedule_time + execute_time + update_time

        return StepTimingBreakdown(
            schedule_time_ms=schedule_time,
            execute_time_ms=execute_time,
            update_time_ms=update_time,
            total_time_ms=total_time,
            num_decode=num_decode,
            num_prefill=num_prefill,
            total_tokens=total_tokens,
        )

    def _cleanup_all_requests(self):
        """Clean up all requests."""
        if self.engine_core is None:
            return

        # Get all request IDs (both running and waiting are list[Request])
        running_ids = [req.request_id for req in self.engine_core.scheduler.running]
        waiting_ids = [req.request_id for req in self.engine_core.scheduler.waiting]
        all_ids = running_ids + waiting_ids

        if all_ids:
            self.engine_core.abort_requests(all_ids)

        # Drain remaining
        while self.engine_core.scheduler.has_requests():
            try:
                self.engine_core.step_fn()
            except Exception:
                break

    def _prepare_decode_requests(
        self, num_decode: int, context_len: int
    ) -> tuple[list[str], bool]:
        """
        Prepare decode requests by completing their prefill phase.

        Returns:
            (list of request IDs, success flag)
        """
        if num_decode == 0:
            return [], True

        assert self.engine_core is not None

        # Check if we can even schedule this many requests
        if num_decode > self.config.max_num_seqs:
            return [], False

        # Add requests with very large max_tokens to avoid early completion
        req_ids = []
        for _ in range(num_decode):
            req_id = self._add_request(context_len, max_tokens=100000)
            req_ids.append(req_id)

        # Run until all requests have completed prefill
        # (in running state with num_computed_tokens >= context_len)
        tokens_to_process = context_len * num_decode
        max_steps = (tokens_to_process // self.config.max_num_batched_tokens) + 100
        for _ in range(max_steps):
            num_running = len(self.engine_core.scheduler.running)
            if num_running >= num_decode:
                # Check that all running requests have completed prefill
                all_in_decode = all(
                    req.num_computed_tokens >= req.num_prompt_tokens
                    for req in self.engine_core.scheduler.running
                )
                if all_in_decode:
                    break
            # Check if there are requests to process before calling step_fn
            if not self.engine_core.scheduler.has_requests():
                break
            try:
                self.engine_core.step_fn()
            except (KeyError, RuntimeError, AssertionError) as e:
                # Request may have been cleaned up unexpectedly or scheduler assertion failed
                print(f"  Warning: Error during prefill preparation: {e}")
                return req_ids, False

        # Verify we have enough decode requests ready
        num_running = len(self.engine_core.scheduler.running)
        num_in_decode = sum(
            1 for req in self.engine_core.scheduler.running
            if req.num_computed_tokens >= req.num_prompt_tokens
        )

        if num_in_decode < num_decode:
            return req_ids, False

        return req_ids, True

    def _global_warmup(self):
        """
        Run a global warmup to ensure GPU is in hot state.
        This ensures fair comparison between all benchmarks.
        """
        assert self.engine_core is not None

        # Run several iterations of prefill + decode to warm up GPU
        num_warmup_iters = 20
        warmup_prefill_size = max(self.config.prefill_chunk_sizes)

        for _ in range(num_warmup_iters):
            self._cleanup_all_requests()

            # Run a prefill
            self._add_request(warmup_prefill_size, max_tokens=10)
            for _ in range(5):  # Run a few steps
                if not self.engine_core.scheduler.has_requests():
                    break
                try:
                    self.engine_core.step_fn()
                except (KeyError, RuntimeError, AssertionError):
                    break

        self._cleanup_all_requests()
        torch.cuda.synchronize()

    def run_pure_decode_benchmark(
        self,
        num_decode: int,
        decode_context_len: int,
    ) -> BenchmarkResult | None:
        """Benchmark pure decode (N requests, 1 token each)."""
        if num_decode == 0:
            return None

        description = f"{num_decode}D(ctx={decode_context_len})"
        timings: list[StepTimingBreakdown] = []

        # Warmup + measurement
        for i in range(self.config.num_warmup + self.config.num_iterations):
            self._cleanup_all_requests()

            # Prepare decode requests with fixed context length
            _, success = self._prepare_decode_requests(num_decode, decode_context_len)
            if not success:
                print(f"  Warning: Could not prepare {num_decode} decode requests "
                      f"(max_num_seqs={self.config.max_num_seqs})")
                return None

            # Run one step and measure
            breakdown = self._run_single_step_with_breakdown()

            if i >= self.config.num_warmup and breakdown and breakdown.num_decode > 0:
                timings.append(breakdown)

        if not timings:
            return None

        return self._create_result(description, num_decode, 0, 0, decode_context_len, timings)

    def run_pure_prefill_benchmark(
        self,
        prefill_chunk_size: int,
    ) -> BenchmarkResult | None:
        """Benchmark pure prefill (1 request with given chunk size)."""
        description = f"1P({prefill_chunk_size})"
        timings: list[StepTimingBreakdown] = []

        for i in range(self.config.num_warmup + self.config.num_iterations):
            self._cleanup_all_requests()

            # Pre-measurement warmup: run some steps to match the state
            # of mixed benchmarks which run _prepare_decode_requests
            self._add_request(256, max_tokens=5)
            for _ in range(3):
                if not self.engine_core.scheduler.has_requests():
                    break
                try:
                    self.engine_core.step_fn()
                except (KeyError, RuntimeError, AssertionError):
                    break
            self._cleanup_all_requests()

            # Add a new prefill request
            self._add_request(prefill_chunk_size, max_tokens=1)

            # Run one step (should be prefill)
            breakdown = self._run_single_step_with_breakdown()

            if i >= self.config.num_warmup and breakdown and breakdown.total_tokens > 0:
                timings.append(breakdown)

        if not timings:
            return None

        return self._create_result(description, 0, 1, prefill_chunk_size, 0, timings)

    def run_mixed_benchmark(
        self,
        num_decode: int,
        prefill_chunk_size: int,
        decode_context_len: int,
    ) -> BenchmarkResult | None:
        """Benchmark mixed batch (N decode + 1 prefill)."""
        if num_decode == 0:
            return self.run_pure_prefill_benchmark(prefill_chunk_size)

        description = f"{num_decode}D(ctx={decode_context_len})+1P({prefill_chunk_size})"
        total_tokens = num_decode + prefill_chunk_size

        if total_tokens > self.config.max_num_batched_tokens:
            return None

        timings: list[StepTimingBreakdown] = []

        for i in range(self.config.num_warmup + self.config.num_iterations):
            self._cleanup_all_requests()

            # Prepare decode requests first
            _, success = self._prepare_decode_requests(num_decode, decode_context_len)
            if not success:
                print(f"  Warning: Could not prepare {num_decode} decode requests "
                      f"(max_num_seqs={self.config.max_num_seqs})")
                return None

            # Add a new prefill request
            self._add_request(prefill_chunk_size, max_tokens=1)

            # Run one step - scheduler should schedule both decode and prefill
            breakdown = self._run_single_step_with_breakdown()

            # Verify we got the expected batch composition
            if (i >= self.config.num_warmup and breakdown and
                breakdown.num_decode >= num_decode and breakdown.num_prefill >= 1):
                timings.append(breakdown)

        if not timings:
            print(f"  Warning: No valid measurements for {description}")
            return None

        return self._create_result(
            description, num_decode, 1, prefill_chunk_size, decode_context_len, timings
        )

    def _create_result(
        self,
        description: str,
        num_decode: int,
        num_prefill: int,
        prefill_chunk_size: int,
        decode_context_len: int,
        timings: list[StepTimingBreakdown],
    ) -> BenchmarkResult:
        """Create a BenchmarkResult from timing breakdowns."""
        total_tokens = num_decode + (prefill_chunk_size if num_prefill > 0 else 0)

        total_times = [t.total_time_ms for t in timings]
        schedule_times = [t.schedule_time_ms for t in timings]
        execute_times = [t.execute_time_ms for t in timings]
        update_times = [t.update_time_ms for t in timings]

        mean_total = float(np.mean(total_times))
        throughput = (total_tokens / mean_total) * 1000 if mean_total > 0 else 0

        return BenchmarkResult(
            description=description,
            num_decode=num_decode,
            num_prefill=num_prefill,
            prefill_chunk_size=prefill_chunk_size,
            decode_context_len=decode_context_len,
            total_tokens=total_tokens,
            mean_time_ms=mean_total,
            std_time_ms=float(np.std(total_times)),
            throughput_tokens_per_sec=throughput,
            mean_schedule_time_ms=float(np.mean(schedule_times)),
            std_schedule_time_ms=float(np.std(schedule_times)),
            mean_execute_time_ms=float(np.mean(execute_times)),
            std_execute_time_ms=float(np.std(execute_times)),
            mean_update_time_ms=float(np.mean(update_times)),
            std_update_time_ms=float(np.std(update_times)),
        )

    def run_all_benchmarks(self):
        """Run all benchmark configurations."""
        print(f"\n{'='*60}")
        print("RUNNING FULL STEP BENCHMARKS")
        print(f"{'='*60}")

        # Determine which sizes to use for pure tests
        pure_prefill_sizes = (
            self.config.pure_prefill_sizes
            if self.config.pure_prefill_sizes
            else self.config.prefill_chunk_sizes
        )
        pure_decode_counts = (
            self.config.pure_decode_counts
            if self.config.pure_decode_counts
            else self.config.decode_counts
        )

        print(f"\nPure prefill sizes: {pure_prefill_sizes}")
        print(f"Pure decode counts: {pure_decode_counts}")
        print(f"Mixed prefill sizes: {self.config.prefill_chunk_sizes}")
        print(f"Mixed decode counts: {self.config.decode_counts}")

        # Global warmup to ensure GPU is in hot state
        print("\n0. Global GPU Warmup...")
        self._global_warmup()
        print("   Warmup complete.")

        # 1. Pure decode benchmarks (varying context lengths)
        print("\n1. Pure Decode Benchmarks:")
        for decode_ctx_len in self.config.decode_context_lens:
            for num_decode in pure_decode_counts:
                if num_decode > 0:
                    print(f"   Testing {num_decode}D(ctx={decode_ctx_len})...")
                    result = self.run_pure_decode_benchmark(num_decode, decode_ctx_len)
                    if result:
                        self.results.append(result)
                        self._print_result_summary(result)

        # 2. Pure prefill benchmarks
        print("\n2. Pure Prefill Benchmarks:")
        for prefill_size in pure_prefill_sizes:
            print(f"   Testing 1P({prefill_size})...")
            result = self.run_pure_prefill_benchmark(prefill_size)
            if result:
                self.results.append(result)
                self._print_result_summary(result)

        # 3. Mixed benchmarks: N decode + 1 prefill (varying context lengths)
        print("\n3. Mixed Benchmarks (Decode + Prefill):")
        for decode_ctx_len in self.config.decode_context_lens:
            for num_decode in self.config.decode_counts:
                if num_decode > 0:
                    for prefill_size in self.config.prefill_chunk_sizes:
                        if num_decode + prefill_size <= self.config.max_num_batched_tokens:
                            print(f"   Testing {num_decode}D(ctx={decode_ctx_len})+1P({prefill_size})...")
                            result = self.run_mixed_benchmark(num_decode, prefill_size, decode_ctx_len)
                            if result:
                                self.results.append(result)
                                self._print_result_summary(result)

        print(f"\n{'='*60}")
        print(f"Completed {len(self.results)} benchmarks!")
        print(f"{'='*60}")

    def _print_result_summary(self, result: BenchmarkResult):
        """Print a single result summary with breakdown."""
        print(f"      -> Total: {result.mean_time_ms:.3f}ms ± {result.std_time_ms:.3f}ms "
              f"(sched: {result.mean_schedule_time_ms:.3f}ms, "
              f"exec: {result.mean_execute_time_ms:.3f}ms, "
              f"update: {result.mean_update_time_ms:.3f}ms)")

    def print_results(self):
        """Print results in table format."""
        print("\n" + "=" * 160)
        print(f"FULL STEP BENCHMARK RESULTS - Model: {self.config.model}")
        print(f"Decode context lengths: {self.config.decode_context_lens}")
        print("=" * 160)
        print(f"{'Config':<40} {'Decode':>6} {'Prefill':>7} {'Ctx':>6} {'Tokens':>6} "
              f"{'Total(ms)':>10} {'Sched(ms)':>10} {'Exec(ms)':>10} {'Update(ms)':>11} {'Tok/s':>10}")
        print("-" * 160)

        for r in sorted(self.results, key=lambda x: (x.decode_context_len, x.total_tokens)):
            print(f"{r.description:<40} {r.num_decode:>6} {r.num_prefill:>7} "
                  f"{r.decode_context_len:>6} {r.total_tokens:>6} "
                  f"{r.mean_time_ms:>10.3f} {r.mean_schedule_time_ms:>10.3f} "
                  f"{r.mean_execute_time_ms:>10.3f} {r.mean_update_time_ms:>11.3f} "
                  f"{r.throughput_tokens_per_sec:>10.0f}")

        print("=" * 160)

        # Print overhead analysis
        self._print_overhead_analysis()

    def _print_overhead_analysis(self):
        """Print analysis of scheduling overhead."""
        print("\n" + "=" * 80)
        print("OVERHEAD ANALYSIS: Scheduling vs Execution Time")
        print("=" * 80)

        for r in sorted(self.results, key=lambda x: (x.decode_context_len, x.total_tokens)):
            if r.mean_time_ms > 0:
                sched_pct = (r.mean_schedule_time_ms / r.mean_time_ms) * 100
                exec_pct = (r.mean_execute_time_ms / r.mean_time_ms) * 100
                update_pct = (r.mean_update_time_ms / r.mean_time_ms) * 100
                overhead_pct = sched_pct + update_pct
                print(f"  {r.description:<40}: "
                      f"sched={sched_pct:5.1f}%, exec={exec_pct:5.1f}%, "
                      f"update={update_pct:5.1f}%, total_overhead={overhead_pct:5.1f}%")

        print("=" * 80)

    def save_results(self, filepath: str):
        """Save results to JSON file."""
        # Create parent directory if it doesn't exist
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "config": asdict(self.config),
            "results": [asdict(r) for r in self.results],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {filepath}")


def add_cli_args(parser: argparse.ArgumentParser):
    """Add CLI arguments."""
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="Model to benchmark",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="Model dtype",
    )
    parser.add_argument(
        "--prefill-sizes",
        type=str,
        default="512,1024,2048,4096",
        help="Comma-separated list of prefill chunk sizes",
    )
    parser.add_argument(
        "--decode-counts",
        type=str,
        default="0,1,2,4,8,16,32",
        help="Comma-separated list of decode request counts",
    )
    parser.add_argument(
        "--decode-context-lens",
        type=str,
        default="512,1024,2048,4096",
        help="Comma-separated list of context lengths for decode requests",
    )
    parser.add_argument(
        "--pure-prefill-sizes",
        type=str,
        default=None,
        help="Comma-separated list of prefill sizes for PURE prefill tests "
             "(default: same as --prefill-sizes). Use larger values for comparison.",
    )
    parser.add_argument(
        "--pure-decode-counts",
        type=str,
        default=None,
        help="Comma-separated list of decode counts for PURE decode tests "
             "(default: same as --decode-counts). Use larger values for comparison.",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=5,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=20,
        help="Number of measurement iterations",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=16384,
        help="Maximum number of batched tokens",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="Maximum number of sequences",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (0.0 to 1.0)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save results in JSON format",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=True,
        help="Enforce eager mode (disable CUDA graphs)",
    )


def main(args: argparse.Namespace | None = None):
    """Main entry point."""
    if args is None:
        parser = argparse.ArgumentParser(
            description="Benchmark vLLM full step execution (schedule + execute + update)"
        )
        add_cli_args(parser)
        args = parser.parse_args()

    # Parse pure test sizes (optional, defaults to mixed sizes)
    pure_prefill_sizes = None
    if args.pure_prefill_sizes:
        pure_prefill_sizes = [int(x) for x in args.pure_prefill_sizes.split(",")]

    pure_decode_counts = None
    if args.pure_decode_counts:
        pure_decode_counts = [int(x) for x in args.pure_decode_counts.split(",")]

    config = BenchmarkConfig(
        model=args.model,
        dtype=args.dtype,
        prefill_chunk_sizes=[int(x) for x in args.prefill_sizes.split(",")],
        decode_counts=[int(x) for x in args.decode_counts.split(",")],
        decode_context_lens=[int(x) for x in args.decode_context_lens.split(",")],
        pure_prefill_sizes=pure_prefill_sizes,
        pure_decode_counts=pure_decode_counts,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        output_json=args.output_json,
        enforce_eager=args.enforce_eager,
    )

    benchmark = BatchBenchmark(config)
    benchmark.setup()
    benchmark.run_all_benchmarks()
    benchmark.print_results()

    if args.output_json:
        benchmark.save_results(args.output_json)


if __name__ == "__main__":
    main()
