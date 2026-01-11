#!/usr/bin/env python3
"""
Find optimal k_hat using online server throughput measurement.

This script sweeps through k values by starting a vLLM server for each k,
measuring throughput using HTTP requests, and finding the k that maximizes
throughput.

Unlike run_k_sweep.py (offline), this uses the serve mode which is closer
to production deployment.

Usage:
    # Full sweep from k=1 to N
    python experiments/find_khat_serve.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset alpaca \
        --num-requests 500 \
        --batch-size 32

    # Specific k values
    python experiments/find_khat_serve.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset alpaca \
        --k-values 1 5 10 15 20 25 30

    # Adaptive sweep (dense near expected optimal)
    python experiments/find_khat_serve.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset alpaca \
        --adaptive-sweep \
        --expected-kstar 15
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

import aiohttp
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pd_experiment_utils import (
    DEFAULT_OUTPUT_DIR,
    load_dataset_prompts,
    plot_throughput_curve,
)


@dataclass
class ThroughputResult:
    """Results from a single k value throughput test."""
    k: int
    elapsed_time: float
    num_completed: int
    num_errors: int
    total_output_tokens: int
    throughput_tokens_per_sec: float
    throughput_requests_per_sec: float


async def send_chat_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> dict:
    """Send a single chat completion request."""
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                return {"error": f"HTTP {response.status}: {error_text}"}
            return await response.json()
    except Exception as e:
        return {"error": str(e)}


async def run_throughput_test(
    base_url: str,
    model: str,
    prompts: list[str],
    max_tokens: int,
    concurrency: int,
    num_warmup: int = 50,
) -> tuple[float, int, int, int]:
    """
    Run throughput test against server.

    Returns: (elapsed_time, num_completed, num_errors, total_output_tokens)
    """
    connector = aiohttp.TCPConnector(limit=concurrency)
    timeout = aiohttp.ClientTimeout(total=600)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_request(prompt):
            async with semaphore:
                return await send_chat_request(session, base_url, model, prompt, max_tokens)

        # Warmup
        if num_warmup > 0:
            warmup_prompts = prompts[:num_warmup]
            warmup_tasks = [bounded_request(p) for p in warmup_prompts]
            for coro in asyncio.as_completed(warmup_tasks):
                await coro

        # Actual test
        test_prompts = prompts[num_warmup:]
        tasks = [bounded_request(p) for p in test_prompts]

        start_time = time.perf_counter()

        results = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)

        elapsed = time.perf_counter() - start_time

    # Process results
    num_errors = 0
    total_output_tokens = 0

    for result in results:
        if "error" in result:
            num_errors += 1
            continue
        if "choices" in result and len(result["choices"]) > 0:
            usage = result.get("usage", {})
            total_output_tokens += usage.get("completion_tokens", 0)

    num_completed = len(results) - num_errors

    return elapsed, num_completed, num_errors, total_output_tokens


def wait_for_server(base_url: str, timeout: int = 300) -> bool:
    """Wait for server to become ready."""
    import requests
    start = time.time()
    health_url = base_url.replace("/v1", "/health")

    while time.time() - start < timeout:
        try:
            resp = requests.get(health_url, timeout=5)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)

    return False


def start_server(
    model: str,
    port: int,
    k: int,
    batch_size: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
) -> subprocess.Popen:
    """Start vLLM server with given k value."""
    env = os.environ.copy()
    env["VLLM_USE_PD_SCHEDULER"] = "1"
    env["VLLM_PD_ENABLE_DYNAMIC_KSTAR"] = "0"  # Fixed k mode
    env["VLLM_PD_K_STAR_DYNAMIC"] = str(k)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--max-num-seqs", str(batch_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--trust-remote-code",
    ]

    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    return process


def stop_server(process: subprocess.Popen):
    """Stop vLLM server gracefully."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def generate_adaptive_k_values(N: int, expected_kstar: int) -> list[int]:
    """
    Generate k values with adaptive sampling.
    Dense sampling around expected k*, sparse elsewhere.
    """
    # Dense region: expected_kstar/2 to 2*expected_kstar
    k_star_region_min = max(1, expected_kstar // 2)
    k_star_region_max = min(N, expected_kstar * 2)

    # Sparse sampling outside k* region (step=5)
    sparse_low = list(range(1, k_star_region_min, 5))
    sparse_high = list(range(k_star_region_max + 5, N + 1, 5))

    # Dense sampling near k* (step=1 or 2)
    step = 1 if k_star_region_max - k_star_region_min <= 20 else 2
    dense = list(range(k_star_region_min, k_star_region_max + 1, step))

    # Combine and sort
    k_values = sparse_low + dense + sparse_high
    k_values.append(expected_kstar)
    k_values.append(1)
    k_values.append(N)

    return sorted(set(k_values))


def run_single_k_test(
    model: str,
    port: int,
    k: int,
    batch_size: int,
    prompts: list[str],
    max_output_tokens: int,
    num_warmup: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
) -> ThroughputResult | None:
    """
    Run throughput test for a single k value.
    Starts server, runs test, stops server.
    """
    base_url = f"http://localhost:{port}/v1"

    print(f"\n  Starting server with k={k}...")
    process = start_server(
        model, port, k, batch_size,
        gpu_memory_utilization, tensor_parallel_size
    )

    try:
        # Wait for server to be ready
        if not wait_for_server(base_url, timeout=300):
            print(f"  ERROR: Server failed to start for k={k}")
            return None

        print(f"  Server ready. Running throughput test...")

        # Run test
        elapsed, completed, errors, output_tokens = asyncio.run(
            run_throughput_test(
                base_url=base_url,
                model=model,
                prompts=prompts,
                max_tokens=max_output_tokens,
                concurrency=batch_size,
                num_warmup=num_warmup,
            )
        )

        throughput_tokens = output_tokens / elapsed if elapsed > 0 else 0
        throughput_reqs = completed / elapsed if elapsed > 0 else 0

        result = ThroughputResult(
            k=k,
            elapsed_time=elapsed,
            num_completed=completed,
            num_errors=errors,
            total_output_tokens=output_tokens,
            throughput_tokens_per_sec=throughput_tokens,
            throughput_requests_per_sec=throughput_reqs,
        )

        print(f"  k={k}: {throughput_tokens:.1f} tokens/s, "
              f"{throughput_reqs:.2f} req/s, "
              f"{completed} completed, {errors} errors")

        return result

    finally:
        print(f"  Stopping server...")
        stop_server(process)
        # Give OS time to release port
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(
        description='Find optimal k_hat using online server throughput')

    # Model
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.85)

    # Dataset
    parser.add_argument('--dataset', type=str, default='alpaca',
                        choices=['alpaca', 'sharegpt', 'lmsys'])
    parser.add_argument('--sharegpt-path', type=str,
                        default='./ShareGPT_V3_unfiltered_cleaned_split.json')
    parser.add_argument('--max-input-tokens', type=int, default=32000)

    # Test parameters
    parser.add_argument('--num-requests', type=int, default=500,
                        help='Number of test requests per k value')
    parser.add_argument('--num-warmup', type=int, default=50,
                        help='Number of warmup requests per k value')
    parser.add_argument('--max-output-tokens', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Max batch size (N)')
    parser.add_argument('--port', type=int, default=8000)

    # k sweep parameters
    parser.add_argument('--k-values', type=int, nargs='+', default=None,
                        help='Specific k values to test')
    parser.add_argument('--k-step', type=int, default=1,
                        help='Step size for k sweep')
    parser.add_argument('--adaptive-sweep', action='store_true',
                        help='Use adaptive sampling around expected k*')
    parser.add_argument('--expected-kstar', type=int, default=None,
                        help='Expected k* for adaptive sweep')

    # Output
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Find k_hat using Online Server Throughput")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size N: {args.batch_size}")
    print(f"Requests per k: {args.num_requests}")
    print(f"Warmup per k: {args.num_warmup}")
    print("=" * 60)

    # Load tokenizer and prompts
    print(f"\nLoading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    max_samples = args.num_requests + args.num_warmup + 100
    prompts, input_lengths = load_dataset_prompts(
        args.dataset, tokenizer, max_samples,
        sharegpt_path=args.sharegpt_path,
        max_input_tokens=args.max_input_tokens,
    )

    print(f"Loaded {len(prompts)} prompts")
    print(f"Input length: mean={np.mean(input_lengths):.1f}, "
          f"std={np.std(input_lengths):.1f}")

    # Determine k values to test
    if args.k_values is not None:
        k_values = sorted(set(args.k_values))
    elif args.adaptive_sweep:
        expected = args.expected_kstar or args.batch_size // 2
        k_values = generate_adaptive_k_values(args.batch_size, expected)
    else:
        k_values = list(range(1, args.batch_size + 1, args.k_step))

    # Validate k values
    k_values = [k for k in k_values if 1 <= k <= args.batch_size]

    print(f"\nk values to test: {k_values}")
    print(f"Total tests: {len(k_values)}")
    print(f"Estimated time: {len(k_values) * 5}+ minutes (server startup overhead)")

    # Run sweep
    throughput_by_k = {}
    results_list = []
    best_k = 1
    best_throughput = 0.0

    for i, k in enumerate(k_values):
        print(f"\n{'='*60}")
        print(f"Testing k={k} ({i+1}/{len(k_values)})")
        print(f"{'='*60}")

        result = run_single_k_test(
            model=args.model,
            port=args.port,
            k=k,
            batch_size=args.batch_size,
            prompts=prompts,
            max_output_tokens=args.max_output_tokens,
            num_warmup=args.num_warmup,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
        )

        if result is not None:
            # Use tokens/sec as primary metric
            throughput = result.throughput_tokens_per_sec
            throughput_by_k[k] = throughput
            results_list.append(result)

            if throughput > best_throughput:
                best_throughput = throughput
                best_k = k

    k_hat = best_k

    # Save results
    results = {
        'test_type': 'find_khat_serve',
        'model': args.model,
        'dataset': args.dataset,
        'batch_size_N': args.batch_size,
        'num_requests_per_k': args.num_requests,
        'num_warmup_per_k': args.num_warmup,
        'k_values_tested': k_values,
        'k_hat': k_hat,
        'throughput_khat_tokens_per_sec': best_throughput,
        'throughput_by_k': {str(k): v for k, v in throughput_by_k.items()},
        'detailed_results': [
            {
                'k': r.k,
                'elapsed_time': r.elapsed_time,
                'num_completed': r.num_completed,
                'num_errors': r.num_errors,
                'total_output_tokens': r.total_output_tokens,
                'throughput_tokens_per_sec': r.throughput_tokens_per_sec,
                'throughput_requests_per_sec': r.throughput_requests_per_sec,
            }
            for r in results_list
        ],
    }

    json_path = os.path.join(args.output_dir, f"{args.dataset}_find_khat_serve.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {json_path}")

    # Plot throughput curve
    plot_throughput_curve(
        throughput_by_k,
        k_star=None,  # We don't know analytical k* here
        k_hat=k_hat,
        output_dir=args.output_dir,
        dataset_name=f"{args.dataset}_serve",
    )

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Empirical k_hat: {k_hat}")
    print(f"Best throughput: {best_throughput:.1f} tokens/sec")
    print("=" * 60)

    # Print top 5 k values
    sorted_k = sorted(throughput_by_k.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 5 k values by throughput:")
    for k, tp in sorted_k[:5]:
        marker = " (k_hat)" if k == k_hat else ""
        print(f"  k={k}: {tp:.1f} tokens/sec{marker}")

    print("\n" + "=" * 60)
    print("RECOMMENDED ENVIRONMENT VARIABLES")
    print("=" * 60)
    print("export VLLM_USE_PD_SCHEDULER=1")
    print("export VLLM_PD_ENABLE_DYNAMIC_KSTAR=0")
    print(f"export VLLM_PD_K_STAR_DYNAMIC={k_hat}")
    print("=" * 60)

    return results


if __name__ == '__main__':
    main()
