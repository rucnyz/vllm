#!/usr/bin/env python3
"""
HTTP Client for P/D Scheduler Throughput Testing

This script sends requests to a running vLLM serve instance and measures throughput.
Use this for fair comparison between fixed k and dynamic k* modes, as both servers
will be fully warmed up before testing.

Usage:
    # Step 1: Start two vLLM serve instances (on different ports):

    # Fixed k mode (port 8000):
    CUDA_VISIBLE_DEVICES=0 VLLM_USE_PD_SCHEDULER=1 VLLM_PD_ENABLE_DYNAMIC_KSTAR=0 \
        VLLM_PD_K_STAR_DYNAMIC=5 \
        vllm serve Qwen/Qwen3-8B --port 8000 --max-num-seqs 32 --gpu-memory-utilization 0.8

    # Dynamic k* mode (port 8001):
    CUDA_VISIBLE_DEVICES=1 VLLM_USE_PD_SCHEDULER=1 VLLM_PD_ENABLE_DYNAMIC_KSTAR=1 \
        vllm serve Qwen/Qwen3-8B --port 8001 --max-num-seqs 32 --gpu-memory-utilization 0.8

    # Step 2: Run throughput tests against each server:

    # Test fixed k:
    python experiments/run_serve_throughput.py \
        --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen3-8B \
        --dataset alpaca \
        --num-requests 500 \
        --num-warmup 100 \
        --test-name fixed_k

    # Test dynamic k*:
    python experiments/run_serve_throughput.py \
        --base-url http://localhost:8001/v1 \
        --model Qwen/Qwen3-8B \
        --dataset alpaca \
        --num-requests 500 \
        --num-warmup 100 \
        --test-name dynamic_kstar
"""

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass

import aiohttp
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from experiments.pd_experiment_utils import (
    load_dataset_prompts,
    DEFAULT_OUTPUT_DIR,
)


async def send_chat_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> dict:
    """Send a single chat completion request."""
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                return {"error": f"HTTP {response.status}: {error_text}"}
            result = await response.json()
            return result
    except Exception as e:
        return {"error": str(e)}


async def run_warmup(
    base_url: str,
    model: str,
    prompts: list[str],
    max_tokens: int,
    concurrency: int,
) -> None:
    """Run warmup requests to stabilize server state."""
    print(f"\nWarmup: sending {len(prompts)} requests...")

    connector = aiohttp.TCPConnector(limit=concurrency)
    timeout = aiohttp.ClientTimeout(total=300)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_request(prompt):
            async with semaphore:
                return await send_chat_request(
                    session, base_url, model, prompt, max_tokens
                )

        tasks = [bounded_request(p) for p in prompts]

        # Use tqdm for progress
        completed = 0
        with tqdm(total=len(tasks), desc="Warmup", unit="req") as pbar:
            for coro in asyncio.as_completed(tasks):
                await coro
                completed += 1
                pbar.update(1)

    print("Warmup complete.")


@dataclass
class ThroughputResult:
    """Results from throughput test."""
    elapsed_time: float
    num_completed: int
    num_errors: int
    input_tokens: list[int]
    output_tokens: list[int]

    @property
    def total_input_tokens(self) -> int:
        return sum(self.input_tokens)

    @property
    def total_output_tokens(self) -> int:
        return sum(self.output_tokens)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def requests_per_second(self) -> float:
        """Requests per second (not recommended for comparison)."""
        return self.num_completed / self.elapsed_time if self.elapsed_time > 0 else 0

    @property
    def output_tokens_per_second(self) -> float:
        """Output tokens per second - primary throughput metric."""
        return self.total_output_tokens / self.elapsed_time if self.elapsed_time > 0 else 0

    @property
    def total_tokens_per_second(self) -> float:
        """Total tokens (input + output) per second."""
        return self.total_tokens / self.elapsed_time if self.elapsed_time > 0 else 0


async def run_throughput_test(
    base_url: str,
    model: str,
    prompts: list[str],
    input_lengths: list[int],
    max_tokens: int,
    concurrency: int,
) -> ThroughputResult:
    """
    Run throughput test.

    Returns: ThroughputResult with detailed metrics
    """
    print(f"\nTesting: sending {len(prompts)} requests...")

    connector = aiohttp.TCPConnector(limit=concurrency)
    timeout = aiohttp.ClientTimeout(total=600)

    collected_input_tokens = []
    collected_output_tokens = []
    errors = 0

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_request(prompt, input_len):
            async with semaphore:
                result = await send_chat_request(
                    session, base_url, model, prompt, max_tokens
                )
                return result, input_len

        tasks = [bounded_request(p, il) for p, il in zip(prompts, input_lengths)]

        # Start timing
        start_time = time.perf_counter()

        # Gather all results with progress bar
        results = []
        with tqdm(total=len(tasks), desc="Testing", unit="req") as pbar:
            for coro in asyncio.as_completed(tasks):
                result, input_len = await coro
                results.append((result, input_len))
                pbar.update(1)

        elapsed = time.perf_counter() - start_time

    # Process results
    for result, input_len in results:
        if "error" in result:
            errors += 1
            continue

        if "choices" in result and len(result["choices"]) > 0:
            usage = result.get("usage", {})
            # Use usage info if available, otherwise use our tracked input_len
            prompt_tokens = usage.get("prompt_tokens", input_len)
            completion_tokens = usage.get("completion_tokens", 0)
            collected_input_tokens.append(prompt_tokens)
            collected_output_tokens.append(completion_tokens)

    completed = len(collected_output_tokens)

    result = ThroughputResult(
        elapsed_time=elapsed,
        num_completed=completed,
        num_errors=errors,
        input_tokens=collected_input_tokens,
        output_tokens=collected_output_tokens,
    )

    print(f"\nTest completed:")
    print(f"  Completed: {completed}/{len(prompts)} requests")
    if errors > 0:
        print(f"  Errors: {errors}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Total input tokens: {result.total_input_tokens:,}")
    print(f"  Total output tokens: {result.total_output_tokens:,}")
    print(f"  Throughput (output tokens/s): {result.output_tokens_per_second:.2f}")
    print(f"  Throughput (total tokens/s): {result.total_tokens_per_second:.2f}")
    print(f"  Throughput (requests/s): {result.requests_per_second:.2f}")
    if collected_output_tokens:
        print(f"  Output length: mean={np.mean(collected_output_tokens):.1f}, "
              f"std={np.std(collected_output_tokens):.1f}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Run throughput test against vLLM serve')

    # Server configuration
    parser.add_argument('--base-url', type=str, default='http://localhost:8000/v1',
                        help='Base URL of vLLM serve (e.g., http://localhost:8000/v1)')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (must match the served model)')

    # Dataset
    parser.add_argument('--dataset', type=str, default='alpaca',
                        choices=['alpaca', 'sharegpt', 'lmsys'])
    parser.add_argument('--sharegpt-path', type=str,
                        default='./ShareGPT_V3_unfiltered_cleaned_split.json')
    parser.add_argument('--max-input-tokens', type=int, default=32000,
                        help='Skip samples exceeding this input length')

    # Test parameters
    parser.add_argument('--num-requests', type=int, default=500,
                        help='Number of requests for throughput test')
    parser.add_argument('--num-warmup', type=int, default=100,
                        help='Number of warmup requests (more = fairer comparison)')
    parser.add_argument('--max-output-tokens', type=int, default=512,
                        help='Maximum output tokens per request')
    parser.add_argument('--concurrency', type=int, default=32,
                        help='Max concurrent requests (should match server batch size)')

    # Output
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--test-name', type=str, default='serve_test',
                        help='Name for this test (used in output filename)')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("vLLM Serve Throughput Test")
    print("="*60)
    print(f"Base URL: {args.base_url}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Num requests: {args.num_requests}")
    print(f"Num warmup: {args.num_warmup}")
    print(f"Concurrency: {args.concurrency}")
    print("="*60)

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

    # Split prompts for warmup and test
    warmup_prompts = prompts[:args.num_warmup]
    test_prompts = prompts[args.num_warmup:args.num_warmup + args.num_requests]
    test_input_lengths = input_lengths[args.num_warmup:args.num_warmup + args.num_requests]

    # Run warmup
    if args.num_warmup > 0:
        asyncio.run(run_warmup(
            base_url=args.base_url,
            model=args.model,
            prompts=warmup_prompts,
            max_tokens=args.max_output_tokens,
            concurrency=args.concurrency,
        ))

    # Run throughput test
    result = asyncio.run(run_throughput_test(
        base_url=args.base_url,
        model=args.model,
        prompts=test_prompts,
        input_lengths=test_input_lengths,
        max_tokens=args.max_output_tokens,
        concurrency=args.concurrency,
    ))

    # Save results
    results = {
        'test_type': 'serve_throughput',
        'test_name': args.test_name,
        'base_url': args.base_url,
        'model': args.model,
        'dataset': args.dataset,
        'num_requests': args.num_requests,
        'num_warmup': args.num_warmup,
        'concurrency': args.concurrency,
        'elapsed_time': result.elapsed_time,
        'num_completed': result.num_completed,
        'num_errors': result.num_errors,
        # Token counts
        'total_input_tokens': result.total_input_tokens,
        'total_output_tokens': result.total_output_tokens,
        'total_tokens': result.total_tokens,
        # Throughput metrics (primary)
        'throughput_output_tokens_per_sec': result.output_tokens_per_second,
        'throughput_total_tokens_per_sec': result.total_tokens_per_second,
        'throughput_requests_per_sec': result.requests_per_second,
        # Statistics
        'input_lengths': {
            'mean': float(np.mean(result.input_tokens)) if result.input_tokens else 0,
            'std': float(np.std(result.input_tokens)) if result.input_tokens else 0,
        },
        'output_lengths': {
            'mean': float(np.mean(result.output_tokens)) if result.output_tokens else 0,
            'std': float(np.std(result.output_tokens)) if result.output_tokens else 0,
            'min': int(min(result.output_tokens)) if result.output_tokens else 0,
            'max': int(max(result.output_tokens)) if result.output_tokens else 0,
        }
    }

    json_path = os.path.join(
        args.output_dir,
        f"{args.dataset}_{args.test_name}_results.json"
    )
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {json_path}")

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Test: {args.test_name}")
    print(f"Total time: {result.elapsed_time:.2f}s")
    print(f"Total output tokens: {result.total_output_tokens:,}")
    print(f"Throughput (output tokens/s): {result.output_tokens_per_second:.2f}")
    print(f"Throughput (total tokens/s): {result.total_tokens_per_second:.2f}")
    print(f"Throughput (requests/s): {result.requests_per_second:.2f}")
    print("="*60)

    return results


if __name__ == '__main__':
    main()