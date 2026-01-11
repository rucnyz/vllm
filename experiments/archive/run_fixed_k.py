#!/usr/bin/env python3
"""
Fixed k Throughput Test

Runs throughput test with a FIXED switching threshold k.
The scheduler does NOT update k* dynamically during this test.

IMPORTANT: Set VLLM_PD_ENABLE_DYNAMIC_KSTAR=0 BEFORE LLM initialization
to disable dynamic k* updates in the scheduler.

Usage:
    python experiments/run_fixed_k.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset alpaca \
        --num-requests 500 \
        --batch-size 32 \
        --k 5

    # Use offline-estimated k*:
    python experiments/run_fixed_k.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset alpaca \
        --use-offline-kstar
"""

import os
import sys
import time
import json
import argparse

# CRITICAL: Set environment variables BEFORE importing vLLM
os.environ["VLLM_USE_PD_SCHEDULER"] = "1"
os.environ["VLLM_PD_ENABLE_DYNAMIC_KSTAR"] = "0"  # Disable dynamic updates

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from pd_experiment_utils import (
    add_common_args,
    apply_chat_template,
    compute_analytical_kstar,
    is_degenerate_output,
    load_dataset_prompts,
    plot_input_length_distribution,
    plot_output_length_distribution,
    plot_profiling_results,
    profile_model,
)
import numpy as np


def run_fixed_k_test(
    llm: LLM,
    prompts: list[str],
    k: int,
    max_output_tokens: int,
    num_requests: int,
    exclude_degenerate: bool = False,
    num_warmup: int = 5,
) -> tuple[float, list[int], list[dict]]:
    """
    Run throughput test with a FIXED k value.

    Args:
        llm: The LLM instance.
        prompts: List of prompts to use.
        k: The fixed k value to test.
        max_output_tokens: Maximum output tokens per request.
        num_requests: Number of requests for the actual test.
        exclude_degenerate: Whether to exclude degenerate outputs from throughput.
        num_warmup: Number of warmup requests to ensure k value takes effect.

    Returns: (throughput, output_lengths, truncated_samples)
    """
    # Set k via environment variable and trigger state reset
    os.environ["VLLM_PD_K_STAR_DYNAMIC"] = str(k)
    os.environ["VLLM_PD_RESET_STATE"] = "1"

    sampling_params = SamplingParams(
        max_tokens=max_output_tokens,
        temperature=0.7,
        top_p=0.9
    )

    # Warmup: run a few requests to ensure the new k value is active
    if num_warmup > 0:
        warmup_prompts = prompts[:num_warmup]
        _ = llm.generate(warmup_prompts, sampling_params)

    # Actual test: use prompts after warmup
    test_prompts = prompts[num_warmup:num_warmup + num_requests]

    print(f"\n{'='*60}")
    print(f"Running FIXED k={k} throughput test")
    print(f"{'='*60}")
    print(f"VLLM_PD_ENABLE_DYNAMIC_KSTAR: {os.environ.get('VLLM_PD_ENABLE_DYNAMIC_KSTAR')}")
    print(f"Num requests: {len(test_prompts)}")

    # Run all requests in one call
    start_time = time.perf_counter()
    print(f"\nProcessing {len(test_prompts)} requests with fixed k={k}...")
    outputs = llm.generate(test_prompts, sampling_params)
    elapsed = time.perf_counter() - start_time

    # Process outputs
    completed = 0
    output_lengths = []
    truncated_samples = []
    degenerate_count = 0

    for i, output in enumerate(outputs):
        output_len = len(output.outputs[0].token_ids)
        finish_reason = output.outputs[0].finish_reason
        output_text = output.outputs[0].text

        is_degenerate = is_degenerate_output(output_text)
        if is_degenerate:
            degenerate_count += 1

        if exclude_degenerate:
            if not is_degenerate:
                completed += 1
        else:
            completed += 1

        if not is_degenerate:
            output_lengths.append(output_len)

        if finish_reason == "length":
            truncated_samples.append({
                'index': i,
                'prompt': test_prompts[i][:200] + "...",
                'output': output_text[:500] + "...",
                'num_tokens': output_len,
                'is_degenerate': is_degenerate
            })

    if degenerate_count > 0:
        print(f"  Found {degenerate_count} degenerate outputs (repetition loops)")

    throughput = completed / elapsed if elapsed > 0 else 0

    print(f"\nFixed k={k} test completed:")
    print(f"  Throughput: {throughput:.2f} req/s")
    print(f"  Total time: {elapsed:.2f}s for {completed} requests")
    print(f"  Output length: mean={np.mean(output_lengths):.1f}, "
          f"std={np.std(output_lengths):.1f}")

    return throughput, output_lengths, truncated_samples


def main():
    parser = argparse.ArgumentParser(
        description='Run fixed k throughput test')
    add_common_args(parser)

    # Fixed k specific args
    parser.add_argument('--k', type=int, default=None,
                        help='Fixed switching threshold k (required unless --use-offline-kstar)')
    parser.add_argument('--use-offline-kstar', action='store_true',
                        help='Use offline-estimated k* instead of specifying k')
    parser.add_argument('--exclude-degenerate', action='store_true',
                        help='Exclude degenerate outputs from throughput calculation')
    parser.add_argument('--num-warmup', type=int, default=5,
                        help='Number of warmup requests before throughput test')

    args = parser.parse_args()

    if args.k is None and not args.use_offline_kstar:
        parser.error("Either --k or --use-offline-kstar must be specified")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("Fixed k Throughput Test")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size N: {args.batch_size}")
    print(f"Num requests: {args.num_requests}")
    if args.k is not None:
        print(f"Fixed k: {args.k}")
    else:
        print("Fixed k: will use offline-estimated k*")
    print(f"VLLM_PD_ENABLE_DYNAMIC_KSTAR: {os.environ.get('VLLM_PD_ENABLE_DYNAMIC_KSTAR')}")
    print("="*60)

    # Load tokenizer
    print(f"\nLoading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load dataset (need extra for warmup)
    max_samples = (args.num_requests + args.num_warmup +
                   int(args.offline_p_multiplier * args.batch_size) + 100)
    prompts, input_lengths = load_dataset_prompts(
        args.dataset, tokenizer, max_samples,
        sharegpt_path=args.sharegpt_path,
        max_input_tokens=args.max_input_tokens
    )

    # Apply chat template
    prompts = apply_chat_template(prompts, tokenizer, args.enable_thinking)

    # Recompute input lengths
    input_lengths = [len(tokenizer.encode(p)) for p in prompts]

    # Plot input distribution
    plot_input_length_distribution(input_lengths, args.output_dir, args.dataset)

    # Initialize vLLM
    print(f"\nLoading model {args.model}...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_num_seqs=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Profile model (needed for offline k* estimation)
    num_p_samples = int(args.offline_p_multiplier * args.batch_size)
    profile_result = profile_model(
        llm, prompts,
        num_profile_samples=max(100, args.batch_size * 2),
        num_p_estimation_samples=num_p_samples,
        max_output_tokens=args.max_output_tokens,
        num_prefill_samples=args.num_prefill_samples,
        num_decode_repeats=args.num_decode_repeats,
        max_batch_size=args.batch_size
    )

    print("\nProfiling parameters:")
    print(f"  α_p = {profile_result.alpha_p:.6f}")
    print(f"  α_d = {profile_result.alpha_d:.6f}")
    print(f"  β_d = {profile_result.beta_d:.6f}")
    print(f"  p (offline) = {profile_result.p_estimated:.4f}")

    # Determine k value
    if args.use_offline_kstar:
        k = compute_analytical_kstar(
            N=args.batch_size,
            p=profile_result.p_estimated,
            alpha_p=profile_result.alpha_p,
            alpha_d=profile_result.alpha_d,
            beta_d=profile_result.beta_d
        )
        print(f"\nUsing offline-estimated k* = {k}")
    else:
        k = args.k
        print(f"\nUsing specified k = {k}")

    # Run fixed k test
    throughput, output_lengths, truncated_samples = run_fixed_k_test(
        llm, prompts,
        k=k,
        max_output_tokens=args.max_output_tokens,
        num_requests=args.num_requests,
        exclude_degenerate=args.exclude_degenerate,
        num_warmup=args.num_warmup,
    )

    # Save results
    results = {
        'test_type': 'fixed_k',
        'model': args.model,
        'dataset': args.dataset,
        'batch_size_N': args.batch_size,
        'num_requests': args.num_requests,
        'k': k,
        'throughput': throughput,
        'use_offline_kstar': args.use_offline_kstar,
        'profiled_params': {
            'alpha_p': profile_result.alpha_p,
            'alpha_d': profile_result.alpha_d,
            'beta_d': profile_result.beta_d,
            'p_offline': profile_result.p_estimated,
        },
        'output_lengths': {
            'mean': float(np.mean(output_lengths)) if output_lengths else 0,
            'std': float(np.std(output_lengths)) if output_lengths else 0,
            'min': int(min(output_lengths)) if output_lengths else 0,
            'max': int(max(output_lengths)) if output_lengths else 0,
        },
        'truncated_count': len(truncated_samples),
    }

    json_path = os.path.join(args.output_dir, f"{args.dataset}_fixed_k{k}_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {json_path}")

    # Save truncated samples if any
    if truncated_samples:
        trunc_path = os.path.join(args.output_dir, f"{args.dataset}_truncated_k{k}.json")
        with open(trunc_path, 'w', encoding='utf-8') as f:
            json.dump(truncated_samples, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(truncated_samples)} truncated samples to {trunc_path}")

    # Plot results
    plot_profiling_results(profile_result, args.output_dir, args.dataset)
    plot_output_length_distribution(output_lengths, args.output_dir, args.dataset,
                                    args.max_output_tokens)

    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Fixed k: {k}")
    print(f"Throughput: {throughput:.2f} req/s")
    if output_lengths:
        print(f"Output length: mean={np.mean(output_lengths):.1f}, "
              f"std={np.std(output_lengths):.1f}")
    print(f"Truncated: {len(truncated_samples)}/{args.num_requests}")
    print("="*60)

    return results


if __name__ == '__main__':
    main()
