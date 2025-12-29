#!/usr/bin/env python3
"""
Online Server Profiling for P/D Competition Scheduler

Profiles a vLLM server to measure timing parameters (α_p, β_p, α_d, β_d)
and estimate termination probability p.

Usage:
    # Start the vLLM server first:
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-7B-Instruct

    # Then run profiling:
    python experiments/profile_online_server.py \
        --api-base http://localhost:8000/v1 \
        --model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, field

import aiohttp
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from pd_experiment_utils import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SHAREGPT_PATH,
    ProfilingResult,
    compute_analytical_kstar,
    load_dataset_prompts,
    plot_profiling_results,
)


@dataclass
class OnlineProfilingResult(ProfilingResult):
    """Profiling result with server-specific fields."""
    api_base: str = ""
    model: str = ""
    server_info: dict = field(default_factory=dict)


class VLLMAPIClient:
    """Async client for vLLM OpenAI-compatible API with retry and timeout."""

    def __init__(self, api_base: str, model: str, api_key: str = "7355608",
                 timeout: int = 300, max_retries: int = 3):
        self.api_base = api_base.rstrip('/')
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers=self.headers, timeout=self.timeout)
        return self

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()

    async def generate(self, prompt: str, max_tokens: int = 256,
                       temperature: float = 0.7) -> dict:
        """Generate completion with retry logic."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        last_error = None
        for attempt in range(self.max_retries):
            try:
                async with self.session.post(
                    f"{self.api_base}/completions", json=payload
                ) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"API error {resp.status}")
                    return await resp.json()
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))  # Backoff
        raise last_error

    async def generate_batch(self, prompts: list[str], max_tokens: int = 256,
                             temperature: float = 0.7,
                             concurrency: int = 16) -> list[dict]:
        """Generate completions with controlled concurrency."""
        # Limit concurrency to avoid overwhelming server
        concurrency = min(concurrency, 32)
        semaphore = asyncio.Semaphore(concurrency)

        async def _generate(prompt):
            async with semaphore:
                return await self.generate(prompt, max_tokens, temperature)

        results = await asyncio.gather(
            *[_generate(p) for p in prompts], return_exceptions=True
        )
        return [r for r in results if not isinstance(r, Exception)]

    async def get_server_info(self) -> dict:
        """Get server model info."""
        try:
            async with self.session.get(f"{self.api_base}/models") as resp:
                return await resp.json() if resp.status == 200 else {}
        except Exception:
            return {}


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens using tokenizer, with fallback estimation."""
    return len(tokenizer.encode(text)) if tokenizer else int(len(text.split()) * 1.3)


def generate_synthetic_prompts(base_prompt: str, tokenizer,
                               target_lengths: list[int]) -> list[str]:
    """
    Generate prompts of varying lengths by repeating/truncating base text.
    This ensures sufficient length diversity for accurate β_p estimation.
    """
    base_tokens = tokenizer.encode(base_prompt)
    prompts = []

    for target in target_lengths:
        if target <= len(base_tokens):
            # Truncate
            tokens = base_tokens[:target]
        else:
            # Repeat base text to reach target length
            repeats = (target // len(base_tokens)) + 1
            tokens = (base_tokens * repeats)[:target]
        prompts.append(tokenizer.decode(tokens))

    return prompts


async def profile_prefill(client: VLLMAPIClient, prompts: list[str],
                          tokenizer, num_samples: int = 30,
                          use_synthetic: bool = True
                          ) -> tuple[list[float], list[int], float, float]:
    """
    Profile prefill times with single-token generation.
    Returns: (times, lengths, alpha_p, beta_p)

    Args:
        use_synthetic: If True, generate synthetic prompts with diverse lengths
                       to get accurate β_p estimation.
    """
    print(f"\nProfiling prefill ({num_samples} samples)...")

    # Check length diversity of real prompts
    prompt_lengths = [(p, count_tokens(p, tokenizer)) for p in prompts]
    all_lengths = [length for _, length in prompt_lengths]
    min_len, max_len = min(all_lengths), max(all_lengths)

    # If length diversity is insufficient (< 5x range), use synthetic prompts
    if use_synthetic and max_len < min_len * 5:
        print(f"  Real prompt lengths: {min_len}-{max_len} (insufficient diversity)")
        print("  Using synthetic prompts for accurate β_p estimation...")

        # Generate prompts at target lengths: 100, 200, 500, 1000, 2000, 4000
        target_lengths = [100, 200, 500, 1000, 2000, 4000]
        # Take longest prompt as base for better quality
        prompt_lengths.sort(key=lambda x: x[1], reverse=True)
        base_prompt = prompt_lengths[0][0]
        sample_prompts = generate_synthetic_prompts(base_prompt, tokenizer,
                                                    target_lengths)
    else:
        # Use real prompts with even sampling across length distribution
        prompt_lengths.sort(key=lambda x: x[1])
        step = max(1, len(prompt_lengths) // num_samples)
        sample_prompts = [prompt_lengths[i * step][0]
                          for i in range(min(num_samples, len(prompt_lengths)))]

    times, lengths = [], []
    for prompt in tqdm(sample_prompts, desc="Prefill"):
        length = count_tokens(prompt, tokenizer)
        start = time.perf_counter()
        await client.generate(prompt, max_tokens=1, temperature=0)
        times.append(time.perf_counter() - start)
        lengths.append(length)

    # Linear fit: T_p = α_p + β_p * L
    if len(lengths) > 1 and max(lengths) > min(lengths):
        beta_p, alpha_p = np.polyfit(lengths, times, 1)
        # Clamp to non-negative values
        alpha_p = max(0, alpha_p)
        beta_p = max(0, beta_p)
        if beta_p == 0:
            print("  WARNING: β_p clamped to 0 (unexpected)")
    else:
        # Fallback: use mean time and estimate beta from typical GPU speed
        alpha_p = np.mean(times) if times else 0.01
        beta_p = 0.0001  # ~10k tokens/sec typical
        print("  WARNING: Using fallback values (insufficient length diversity)")

    print(f"  α_p={alpha_p:.4f}s, β_p={beta_p:.6f}s/token")
    print(f"  Profiled lengths: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.0f}")
    return times, lengths, alpha_p, beta_p


async def profile_decode(client: VLLMAPIClient, prompts: list[str],
                         tokenizer, max_batch: int = 64, num_repeats: int = 2,
                         tokens_per_req: int = 20
                         ) -> tuple[list[float], list[int], float, float]:
    """
    Profile decode times by varying concurrent request count.
    Returns: (times, batch_sizes, alpha_d, beta_d)

    Note: tokens_per_req=20 is sufficient for timing; no need for more tokens.
    """
    # Cap max batch to avoid overwhelming server
    effective_max = min(max_batch, 64, len(prompts))
    print(f"\nProfiling decode (max_batch={effective_max}, "
          f"tokens={tokens_per_req}/req)...")

    # Fewer batch sizes for faster profiling (still covers the range well)
    batch_sizes = [1, 4, 8, 16, 32, 64]
    batch_sizes = [b for b in batch_sizes if b <= effective_max]

    results = {}
    failed_batches = []
    for bs in tqdm(batch_sizes, desc="Decode"):
        step_times = []
        for attempt in range(num_repeats):
            try:
                start = time.perf_counter()
                responses = await client.generate_batch(
                    prompts[:bs], max_tokens=tokens_per_req,
                    temperature=0, concurrency=min(bs, 32)
                )
                elapsed = time.perf_counter() - start

                # Calculate tokens generated
                total_tokens = sum(
                    count_tokens(r['choices'][0].get('text', ''), tokenizer)
                    for r in responses if r.get('choices')
                )
                # At least 80% success rate required
                if total_tokens > 0 and len(responses) >= bs * 0.8:
                    step_times.append(elapsed / (total_tokens / bs))
            except Exception:
                if attempt == num_repeats - 1:
                    failed_batches.append(bs)

        if step_times:
            results[bs] = np.median(step_times)

    if failed_batches:
        print(f"  WARNING: Failed batch sizes: {failed_batches}")

    times = [results[bs] for bs in batch_sizes if bs in results]
    sizes = [bs for bs in batch_sizes if bs in results]

    # Linear fit: T_d = α_d + β_d * k
    if len(sizes) > 1:
        beta_d, alpha_d = np.polyfit(sizes, times, 1)
        alpha_d = max(0, alpha_d)
    else:
        alpha_d, beta_d = 0.0, 0.0

    print(f"  α_d={alpha_d:.4f}s, β_d={beta_d:.6f}s/req")
    return times, sizes, alpha_d, beta_d


async def estimate_p(client: VLLMAPIClient, prompts: list[str],
                     tokenizer, num_samples: int = 100,
                     max_tokens: int = 256) -> tuple[list[int], float]:
    """
    Estimate termination probability p from output length distribution.
    Returns: (output_lengths, p)
    """
    print(f"\nEstimating p ({num_samples} samples)...")
    lengths = []
    truncated = 0
    batch_size = 16

    for i in tqdm(range(0, min(num_samples, len(prompts)), batch_size),
                  desc="Estimating p"):
        results = await client.generate_batch(
            prompts[i:i + batch_size], max_tokens=max_tokens,
            temperature=0.7, concurrency=batch_size
        )
        for r in results:
            if r.get('choices'):
                choice = r['choices'][0]
                lengths.append(count_tokens(choice.get('text', ''), tokenizer))
                if choice.get('finish_reason') == 'length':
                    truncated += 1

    if lengths:
        mean_len = np.mean(lengths)
        p = 1.0 / mean_len if mean_len > 0 else 0.01
        print(f"  Mean output: {mean_len:.1f} tokens, p={p:.4f}")
        if truncated:
            print(f"  WARNING: {truncated}/{len(lengths)} outputs truncated")
    else:
        p = 0.01

    return lengths, p


async def profile_server(api_base: str, model: str, prompts: list[str],
                         tokenizer, num_prefill: int = 30,
                         num_decode_repeats: int = 3, num_p_samples: int = 100,
                         max_output: int = 256, max_batch: int = 64,
                         api_key: str = "7355608") -> OnlineProfilingResult:
    """Run full server profiling."""
    print("\n" + "=" * 60)
    print(f"Profiling vLLM Server: {api_base}")
    print(f"Model: {model}")
    print("=" * 60)

    result = OnlineProfilingResult(api_base=api_base, model=model)

    async with VLLMAPIClient(api_base, model, api_key) as client:
        result.server_info = await client.get_server_info()

        # Profile prefill
        (result.prefill_times, result.input_lengths,
         result.alpha_p, result.beta_p) = await profile_prefill(
            client, prompts, tokenizer, num_prefill)

        # Profile decode
        (result.decode_times, result.batch_sizes,
         result.alpha_d, result.beta_d) = await profile_decode(
            client, prompts, tokenizer, max_batch, num_decode_repeats)

        # Estimate p
        result.output_lengths, result.p_estimated = await estimate_p(
            client, prompts, tokenizer, num_p_samples, max_output)

    return result


def apply_chat_template(prompts: list[str], tokenizer,
                        enable_thinking: bool = False) -> list[str]:
    """Apply chat template to prompts."""
    formatted = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking)
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        formatted.append(text)
    return formatted


def main():
    parser = argparse.ArgumentParser(
        description='Profile vLLM server for P/D scheduler parameters')

    # Required
    parser.add_argument('--model', type=str, required=True,
                        help='Model name')

    # API config
    parser.add_argument('--api-base', type=str,
                        default='http://localhost:8000/v1')
    parser.add_argument('--api-key', type=str, default='7355608')

    # Dataset
    parser.add_argument('--dataset', type=str, default='alpaca',
                        choices=['alpaca', 'sharegpt', 'lmsys'])
    parser.add_argument('--sharegpt-path', type=str,
                        default=DEFAULT_SHAREGPT_PATH)
    parser.add_argument('--max-input-tokens', type=int, default=32000)

    # Profiling config
    parser.add_argument('--num-prefill-samples', type=int, default=30)
    parser.add_argument('--num-decode-repeats', type=int, default=3)
    parser.add_argument('--num-p-samples', type=int, default=100)
    parser.add_argument('--max-output-tokens', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Max batch size (also N for k* calculation)')

    # Output
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--enable-thinking', action='store_true')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer and dataset
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    max_samples = args.num_prefill_samples + args.num_p_samples + args.batch_size * 2 + 100
    prompts, _ = load_dataset_prompts(
        args.dataset, tokenizer, max_samples,
        sharegpt_path=args.sharegpt_path,
        max_input_tokens=args.max_input_tokens)

    print(f"Applying chat template (thinking={args.enable_thinking})...")
    prompts = apply_chat_template(prompts, tokenizer, args.enable_thinking)

    # Run profiling
    result = asyncio.run(profile_server(
        api_base=args.api_base,
        model=args.model,
        prompts=prompts,
        tokenizer=tokenizer,
        num_prefill=args.num_prefill_samples,
        num_decode_repeats=args.num_decode_repeats,
        num_p_samples=args.num_p_samples,
        max_output=args.max_output_tokens,
        max_batch=args.batch_size,
        api_key=args.api_key,
    ))

    # Compute k*
    k_star = compute_analytical_kstar(
        args.batch_size, result.p_estimated,
        result.alpha_p, result.alpha_d, result.beta_d)

    # Print summary
    mean_len = 1.0 / result.p_estimated if result.p_estimated > 0 else 0
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Timing: α_p={result.alpha_p:.6f}, β_p={result.beta_p:.8f}")
    print(f"        α_d={result.alpha_d:.6f}, β_d={result.beta_d:.8f}")
    print(f"p={result.p_estimated:.4f} (mean output={mean_len:.1f})")
    print(f"k*={k_star} (N={args.batch_size})")
    print("=" * 60)

    # Save results
    output_data = {
        'api_base': args.api_base,
        'model': args.model,
        'dataset': args.dataset,
        'batch_size_N': args.batch_size,
        'timing_parameters': {
            'alpha_p': result.alpha_p, 'beta_p': result.beta_p,
            'alpha_d': result.alpha_d, 'beta_d': result.beta_d,
        },
        'termination_probability': {
            'p_estimated': result.p_estimated,
            'mean_output_length': mean_len,
        },
        'optimal_k_star': k_star,
    }

    json_path = os.path.join(
        args.output_dir, f"{args.dataset}_online_profiling_results.json")
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Plot results
    plot_profiling_results(result, args.output_dir, f"{args.dataset}_online")

    # Print environment variables
    print("\n" + "=" * 60)
    print("ENVIRONMENT VARIABLES FOR P/D SCHEDULER")
    print("=" * 60)
    print("export VLLM_USE_PD_SCHEDULER=1")
    print(f"export VLLM_PD_ALPHA_P={result.alpha_p}")
    print(f"export VLLM_PD_BETA_P={result.beta_p}")
    print(f"export VLLM_PD_ALPHA_D={result.alpha_d}")
    print(f"export VLLM_PD_BETA_D={result.beta_d}")
    print(f"export VLLM_PD_K_STAR={k_star}")
    print(f"# p={result.p_estimated:.4f} (estimated online by scheduler)")
    print("=" * 60)

    return output_data


if __name__ == '__main__':
    main()
