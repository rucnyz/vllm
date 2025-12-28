#!/usr/bin/env python3
"""
Online Server Profiling for P/D Competition Scheduler

Profiles a vLLM serve API backend to measure timing parameters (α_p, β_p, α_d, β_d)
and estimate termination probability p for the P/D competition scheduler.

This script is designed to work with a running vLLM server started via:
    python -m vllm.entrypoints.openai.api_server --model <model_name> ...

Usage:
    # First, start the vLLM server:
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-7B-Instruct \
        --tensor-parallel-size 1

    # Then run this profiling script:
    python experiments/profile_online_server.py \
        --api-base http://localhost:8000/v1 \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset alpaca \
        --batch-size 32
"""

import os
import sys
import time
import json
import argparse
import asyncio
import aiohttp
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

# Import shared utilities
from pd_experiment_utils import (
    DEFAULT_EMA_ALPHA,
    DEFAULT_KSTAR_UPDATE_INTERVAL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SHAREGPT_PATH,
    ProfilingResult,
    compute_analytical_kstar,
    load_dataset_prompts,
    plot_profiling_results,
)

from transformers import AutoTokenizer



@dataclass
class OnlineProfilingResult(ProfilingResult):
    """Extended profiling result with online-specific fields."""
    api_base: str = ""
    model: str = ""
    server_info: dict = field(default_factory=dict)


class VLLMAPIClient:
    """Async client for vLLM OpenAI-compatible API."""

    def __init__(self, api_base: str, model: str, api_key: str = "7355608"):
        self.api_base = api_base.rstrip('/')
        self.model = model
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> dict:
        """Generate completion for a single prompt."""
        url = f"{self.api_base}/completions"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }

        async with self.session.post(url, json=payload) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(f"API error {response.status}: {text}")
            return await response.json()

    async def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        concurrency: int = 16,
    ) -> list[dict]:
        """Generate completions for multiple prompts with concurrency control."""
        semaphore = asyncio.Semaphore(concurrency)

        async def generate_with_semaphore(prompt: str) -> dict:
            async with semaphore:
                return await self.generate(prompt, max_tokens, temperature, top_p)

        tasks = [generate_with_semaphore(p) for p in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and log errors
        valid_results = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                print(f"Request {i} failed: {r}")
            else:
                valid_results.append(r)

        return valid_results

    async def get_server_info(self) -> dict:
        """Get server model info."""
        url = f"{self.api_base}/models"
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            print(f"Could not fetch server info: {e}")
        return {}


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text using tokenizer."""
    if tokenizer:
        return len(tokenizer.encode(text))
    # Fallback: rough estimate
    return int(len(text.split()) * 1.3)


async def profile_prefill_async(
    client: VLLMAPIClient,
    prompts: list[str],
    tokenizer,
    num_samples: int = 30,
) -> tuple[list[float], list[int], float, float]:
    """
    Profile prefill times by sending single-token requests.

    Returns: (prefill_times, input_lengths, alpha_p, beta_p)
    """
    print(f"\nProfiling prefill times ({num_samples} samples)...")

    prefill_times = []
    input_lengths = []

    sample_prompts = prompts[:num_samples]

    for prompt in tqdm(sample_prompts, desc="Prefill profiling"):
        input_len = count_tokens(prompt, tokenizer)

        start = time.perf_counter()
        result = await client.generate(prompt, max_tokens=1, temperature=0)
        elapsed = time.perf_counter() - start

        prefill_times.append(elapsed)
        input_lengths.append(input_len)

    # Linear fit: prefill_time = alpha_p + beta_p * input_length
    if len(input_lengths) > 1:
        coeffs = np.polyfit(input_lengths, prefill_times, 1)
        beta_p = coeffs[0]
        alpha_p = max(0, coeffs[1])
    else:
        alpha_p, beta_p = 0.0, 0.0

    print(f"  Prefill: α_p = {alpha_p:.4f}s, β_p = {beta_p:.6f}s/token")

    return prefill_times, input_lengths, alpha_p, beta_p


async def profile_decode_async(
    client: VLLMAPIClient,
    prompts: list[str],
    tokenizer,
    max_batch_size: int = 64,
    num_repeats: int = 3,
    tokens_per_request: int = 50,
) -> tuple[list[float], list[int], float, float]:
    """
    Profile decode times by sending concurrent requests of varying batch sizes.

    Note: Since we're using an API server, we simulate batching by sending
    concurrent requests. The server's internal batching will determine
    actual batch behavior.

    Returns: (decode_times, batch_sizes, alpha_d, beta_d)
    """
    print(f"\nProfiling decode times (max_batch={max_batch_size}, repeats={num_repeats})...")

    # Test batch sizes
    test_batch_sizes = [1, 2, 4, 8, 16]
    bs = 24
    while bs <= max_batch_size:
        test_batch_sizes.append(bs)
        bs += 8 if bs < 32 else 16
    test_batch_sizes = sorted(set(b for b in test_batch_sizes if b <= max_batch_size))

    decode_times_by_batch = {}
    batch_sizes_tested = []

    for batch_size in tqdm(test_batch_sizes, desc="Decode profiling"):
        if batch_size > len(prompts):
            continue

        times_for_this_batch = []

        for _ in range(num_repeats):
            batch_prompts = prompts[:batch_size]

            # Send concurrent requests
            start = time.perf_counter()
            results = await client.generate_batch(
                batch_prompts,
                max_tokens=tokens_per_request,
                temperature=0,
                concurrency=batch_size,  # Full concurrency
            )
            elapsed = time.perf_counter() - start

            # Calculate total output tokens
            total_output_tokens = 0
            for r in results:
                if 'choices' in r and len(r['choices']) > 0:
                    text = r['choices'][0].get('text', '')
                    total_output_tokens += count_tokens(text, tokenizer)

            if total_output_tokens > 0:
                # Time per decode step = total_time / (total_tokens / batch_size)
                time_per_step = elapsed / (total_output_tokens / batch_size)
                times_for_this_batch.append(time_per_step)

        if times_for_this_batch:
            median_time = np.median(times_for_this_batch)
            decode_times_by_batch[batch_size] = median_time
            batch_sizes_tested.append(batch_size)

    decode_times = [decode_times_by_batch[bs] for bs in batch_sizes_tested]

    # Linear fit: decode_time = alpha_d + beta_d * batch_size
    if len(batch_sizes_tested) > 1:
        coeffs = np.polyfit(batch_sizes_tested, decode_times, 1)
        beta_d = coeffs[0]
        alpha_d = max(0, coeffs[1])
    else:
        alpha_d, beta_d = 0.0, 0.0

    print(f"  Decode: α_d = {alpha_d:.4f}s, β_d = {beta_d:.6f}s/request")

    return decode_times, batch_sizes_tested, alpha_d, beta_d


async def estimate_p_async(
    client: VLLMAPIClient,
    prompts: list[str],
    tokenizer,
    num_samples: int = 100,
    max_output_tokens: int = 256,
    concurrency: int = 16,
) -> tuple[list[int], float]:
    """
    Estimate termination probability p from output length distribution.

    Returns: (output_lengths, p_estimated)
    """
    print(f"\nEstimating p from {num_samples} samples...")

    sample_prompts = prompts[:num_samples]
    output_lengths = []
    truncated_count = 0

    # Process in batches for progress reporting
    batch_size = concurrency
    for i in tqdm(range(0, len(sample_prompts), batch_size), desc="Estimating p"):
        batch = sample_prompts[i:i + batch_size]
        results = await client.generate_batch(
            batch,
            max_tokens=max_output_tokens,
            temperature=0.7,
            top_p=0.9,
            concurrency=concurrency,
        )

        for r in results:
            if 'choices' in r and len(r['choices']) > 0:
                choice = r['choices'][0]
                text = choice.get('text', '')
                output_len = count_tokens(text, tokenizer)
                output_lengths.append(output_len)

                # Check if truncated
                finish_reason = choice.get('finish_reason', '')
                if finish_reason == 'length':
                    truncated_count += 1

    if len(output_lengths) > 0:
        mean_output_len = np.mean(output_lengths)
        p_estimated = 1.0 / mean_output_len if mean_output_len > 0 else 0.01

        print(f"  Output lengths: mean={mean_output_len:.1f}, std={np.std(output_lengths):.1f}")
        print(f"  Estimated p = 1/{mean_output_len:.1f} = {p_estimated:.4f}")

        if truncated_count > 0:
            truncated_pct = 100.0 * truncated_count / len(output_lengths)
            print(f"  WARNING: {truncated_count}/{len(output_lengths)} "
                  f"({truncated_pct:.1f}%) outputs truncated")
    else:
        p_estimated = 0.01

    return output_lengths, p_estimated


async def profile_server_async(
    api_base: str,
    model: str,
    prompts: list[str],
    tokenizer,
    num_prefill_samples: int = 30,
    num_decode_repeats: int = 3,
    num_p_samples: int = 100,
    max_output_tokens: int = 256,
    max_batch_size: int = 64,
    api_key: str = "7355608",
) -> OnlineProfilingResult:
    """
    Profile a vLLM server to measure timing parameters.

    Args:
        api_base: Base URL of the vLLM API server (e.g., http://localhost:8000/v1)
        model: Model name to use for requests
        prompts: List of prompts for profiling
        tokenizer: Tokenizer for counting tokens
        num_prefill_samples: Number of samples for prefill profiling
        num_decode_repeats: Repetitions per batch size for decode profiling
        num_p_samples: Number of samples for p estimation
        max_output_tokens: Max tokens for p estimation
        max_batch_size: Maximum batch size to test
        api_key: API key for vLLM server authentication

    Returns:
        OnlineProfilingResult with timing parameters
    """
    print("\n" + "=" * 60)
    print("Profiling vLLM Server")
    print("=" * 60)
    print(f"API Base: {api_base}")
    print(f"Model: {model}")
    print("=" * 60)

    result = OnlineProfilingResult()
    result.api_base = api_base
    result.model = model

    async with VLLMAPIClient(api_base, model, api_key) as client:
        # Get server info
        result.server_info = await client.get_server_info()

        # Profile prefill
        prefill_times, input_lengths, alpha_p, beta_p = await profile_prefill_async(
            client, prompts, tokenizer, num_prefill_samples
        )
        result.prefill_times = prefill_times
        result.input_lengths = input_lengths
        result.alpha_p = alpha_p
        result.beta_p = beta_p

        # Profile decode
        decode_times, batch_sizes, alpha_d, beta_d = await profile_decode_async(
            client, prompts, tokenizer, max_batch_size, num_decode_repeats
        )
        result.decode_times = decode_times
        result.batch_sizes = batch_sizes
        result.alpha_d = alpha_d
        result.beta_d = beta_d

        # Estimate p
        output_lengths, p_estimated = await estimate_p_async(
            client, prompts, tokenizer, num_p_samples, max_output_tokens
        )
        result.output_lengths = output_lengths
        result.p_estimated = p_estimated

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Profile a vLLM serve API backend for P/D scheduler parameters'
    )

    # API configuration
    parser.add_argument('--api-base', type=str, default='http://localhost:8000/v1',
                        help='Base URL of the vLLM API server')
    parser.add_argument('--api-key', type=str, default='7355608',
                        help='API key for authentication')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name to use for requests')

    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='alpaca',
                        choices=['alpaca', 'sharegpt', 'lmsys'],
                        help='Dataset for prompts')
    parser.add_argument('--sharegpt-path', type=str, default=DEFAULT_SHAREGPT_PATH,
                        help='Path to ShareGPT JSON file')
    parser.add_argument('--max-input-tokens', type=int, default=32000,
                        help='Skip samples exceeding this input length')

    # Profiling configuration
    parser.add_argument('--num-prefill-samples', type=int, default=30,
                        help='Number of samples for prefill profiling')
    parser.add_argument('--num-decode-repeats', type=int, default=3,
                        help='Repetitions per batch size for decode profiling')
    parser.add_argument('--num-p-samples', type=int, default=100,
                        help='Number of samples for p estimation')
    parser.add_argument('--max-output-tokens', type=int, default=256,
                        help='Maximum output tokens for p estimation')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Maximum batch size to test (also used for k* calculation)')

    # Output configuration
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory for saving results')

    # Chat template
    parser.add_argument('--enable-thinking', action='store_true', default=False,
                        help='Enable thinking mode for chat template')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    print(f"\nLoading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load dataset
    max_samples = args.num_prefill_samples + args.num_p_samples + args.batch_size * 2 + 100
    prompts, input_lengths = load_dataset_prompts(
        args.dataset, tokenizer, max_samples,
        sharegpt_path=args.sharegpt_path,
        max_input_tokens=args.max_input_tokens
    )

    # Apply chat template if needed
    print(f"Applying chat template (enable_thinking={args.enable_thinking})...")
    formatted_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        formatted_prompts.append(text)
    prompts = formatted_prompts

    # Run profiling
    result = asyncio.run(profile_server_async(
        api_base=args.api_base,
        model=args.model,
        prompts=prompts,
        tokenizer=tokenizer,
        num_prefill_samples=args.num_prefill_samples,
        num_decode_repeats=args.num_decode_repeats,
        num_p_samples=args.num_p_samples,
        max_output_tokens=args.max_output_tokens,
        max_batch_size=args.batch_size,
        api_key=args.api_key,
    ))

    # Compute k* using profiled parameters
    k_star = compute_analytical_kstar(
        args.batch_size,
        result.p_estimated,
        result.alpha_p,
        result.alpha_d,
        result.beta_d
    )

    # Print summary
    print("\n" + "=" * 60)
    print("PROFILING RESULTS SUMMARY")
    print("=" * 60)
    print(f"API Base: {args.api_base}")
    print(f"Model: {args.model}")
    print(f"\nTiming Parameters:")
    print(f"  α_p (prefill fixed) = {result.alpha_p:.6f} s")
    print(f"  β_p (prefill per token) = {result.beta_p:.8f} s/token")
    print(f"  α_d (decode fixed) = {result.alpha_d:.6f} s")
    print(f"  β_d (decode per request) = {result.beta_d:.8f} s/request")
    print(f"\nTermination Probability:")
    print(f"  p (estimated) = {result.p_estimated:.4f}")
    print(f"  Mean output length = {1.0/result.p_estimated:.1f} tokens")
    print(f"\nOptimal Switching Threshold:")
    print(f"  k* = {k_star} (for N={args.batch_size})")
    print("=" * 60)

    # Save results
    output_data = {
        'api_base': args.api_base,
        'model': args.model,
        'dataset': args.dataset,
        'batch_size_N': args.batch_size,
        'profiling_config': {
            'num_prefill_samples': args.num_prefill_samples,
            'num_decode_repeats': args.num_decode_repeats,
            'num_p_samples': args.num_p_samples,
            'max_output_tokens': args.max_output_tokens,
        },
        'timing_parameters': {
            'alpha_p': result.alpha_p,
            'beta_p': result.beta_p,
            'alpha_d': result.alpha_d,
            'beta_d': result.beta_d,
        },
        'termination_probability': {
            'p_estimated': result.p_estimated,
            'mean_output_length': 1.0 / result.p_estimated if result.p_estimated > 0 else 0,
            'output_length_std': float(np.std(result.output_lengths)) if result.output_lengths else 0,
        },
        'optimal_k_star': k_star,
        'server_info': result.server_info,
    }

    json_path = os.path.join(args.output_dir, f"{args.dataset}_online_profiling_results.json")
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved results to {json_path}")

    # Plot profiling results
    plot_profiling_results(result, args.output_dir, f"{args.dataset}_online")

    # Print environment variable export commands for P/D scheduler
    print("\n" + "=" * 60)
    print("ENVIRONMENT VARIABLES FOR P/D SCHEDULER")
    print("=" * 60)
    print("# Copy these to configure the P/D scheduler:")
    print(f"export VLLM_USE_PD_SCHEDULER=1")
    print(f"export VLLM_PD_ALPHA_P={result.alpha_p}")
    print(f"export VLLM_PD_ALPHA_D={result.alpha_d}")
    print(f"export VLLM_PD_BETA_D={result.beta_d}")
    print(f"export VLLM_PD_K_STAR={k_star}")
    print(f"export VLLM_PD_P={result.p_estimated}")
    print("=" * 60)

    return output_data


if __name__ == '__main__':
    main()
