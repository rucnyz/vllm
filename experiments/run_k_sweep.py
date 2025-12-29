#!/usr/bin/env python3
"""
k Sweep Search for k_hat

Runs throughput tests across multiple k values to find the empirically optimal k_hat.
Each test uses a FIXED k value (no dynamic updates).

IMPORTANT: Set VLLM_PD_ENABLE_DYNAMIC_KSTAR=0 BEFORE LLM initialization
to disable dynamic k* updates in the scheduler.

Usage:
    # Sweep all k values from 1 to N
    python experiments/run_k_sweep.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset alpaca \
        --num-requests 500 \
        --batch-size 32

    # Sweep specific k values
    python experiments/run_k_sweep.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset alpaca \
        --k-values 1 3 5 7 10 15 20

    # Sweep with adaptive sampling (dense near offline k*, sparse elsewhere)
    python experiments/run_k_sweep.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset alpaca \
        --adaptive-sweep
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
from tqdm import tqdm

from pd_experiment_utils import (
    add_common_args,
    apply_chat_template,
    compute_analytical_kstar,
    is_degenerate_output,
    load_dataset_prompts,
    plot_input_length_distribution,
    plot_profiling_results,
    plot_throughput_curve,
    profile_model,
)
import numpy as np


def run_single_k_test(
    llm: LLM,
    prompts: list[str],
    k: int,
    max_output_tokens: int,
    num_requests: int,
    num_warmup: int = 5,
) -> float:
    """
    Run throughput test for a single k value.

    Args:
        llm: The LLM instance.
        prompts: List of prompts to use.
        k: The fixed k value to test.
        max_output_tokens: Maximum output tokens per request.
        num_requests: Number of requests for the actual test.
        num_warmup: Number of warmup requests to ensure k value takes effect.

    Returns:
        Throughput in requests per second.
    """
    # Set new k value and trigger state reset
    os.environ["VLLM_PD_K_STAR_DYNAMIC"] = str(k)
    os.environ["VLLM_PD_RESET_STATE"] = "1"

    sampling_params = SamplingParams(
        max_tokens=max_output_tokens,
        temperature=0.7,
        top_p=0.9
    )

    # Warmup: run a few requests to ensure the new k value is active
    # These requests trigger the scheduler reset and stabilize the state
    if num_warmup > 0:
        warmup_prompts = prompts[:num_warmup]
        _ = llm.generate(warmup_prompts, sampling_params)

    # Actual test: use prompts after warmup
    test_prompts = prompts[num_warmup:num_warmup + num_requests]

    start_time = time.perf_counter()
    outputs = llm.generate(test_prompts, sampling_params)
    elapsed = time.perf_counter() - start_time

    completed = len(outputs)
    throughput = completed / elapsed if elapsed > 0 else 0

    return throughput


def generate_adaptive_k_values(N: int, offline_kstar: int) -> list[int]:
    """
    Generate k values with adaptive sampling.

    Dense sampling around offline k*, sparse elsewhere.
    """
    # Dense region: k*/2 to 2*k*
    k_star_region_min = max(1, offline_kstar // 2)
    k_star_region_max = min(N, offline_kstar * 2)

    # Sparse sampling outside k* region (step=5)
    sparse_low = list(range(1, k_star_region_min, 5))
    sparse_high = list(range(k_star_region_max + 5, N + 1, 5))

    # Dense sampling near k* (step=1 or 2)
    step = 1 if k_star_region_max - k_star_region_min <= 20 else 2
    dense = list(range(k_star_region_min, k_star_region_max + 1, step))

    # Combine and sort
    k_values = sparse_low + dense + sparse_high
    k_values.append(offline_kstar)  # Always include k*
    k_values.append(1)  # Always include 1
    k_values.append(N)  # Always include N

    return sorted(set(k_values))


def main():
    parser = argparse.ArgumentParser(
        description='Run k sweep search for k_hat')
    add_common_args(parser)

    # k sweep specific args
    parser.add_argument('--k-values', type=int, nargs='+', default=None,
                        help='Specific k values to test (e.g., --k-values 1 3 5 7)')
    parser.add_argument('--k-step', type=int, default=1,
                        help='Step size for k sweep (ignored if --k-values or --adaptive-sweep)')
    parser.add_argument('--adaptive-sweep', action='store_true',
                        help='Use adaptive sampling (dense near k*, sparse elsewhere)')
    parser.add_argument('--num-warmup', type=int, default=5,
                        help='Number of warmup requests per k value test')
    parser.add_argument('--online-kstar-throughput', type=float, default=None,
                        help='Throughput from dynamic k* test (for comparison in plot)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("k Sweep Search for k_hat")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size N: {args.batch_size}")
    print(f"Num requests per k: {args.num_requests}")
    print(f"VLLM_PD_ENABLE_DYNAMIC_KSTAR: {os.environ.get('VLLM_PD_ENABLE_DYNAMIC_KSTAR')}")
    print("="*60)

    # Load tokenizer
    print(f"\nLoading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load dataset (need extra for warmup per k test)
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

    # Profile model
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

    # Compute offline k*
    offline_kstar = compute_analytical_kstar(
        N=args.batch_size,
        p=profile_result.p_estimated,
        alpha_p=profile_result.alpha_p,
        alpha_d=profile_result.alpha_d,
        beta_d=profile_result.beta_d
    )
    print(f"\nOffline k* = {offline_kstar}")

    # Determine k values to test
    if args.k_values is not None:
        k_values = sorted(set(args.k_values))
    elif args.adaptive_sweep:
        k_values = generate_adaptive_k_values(args.batch_size, offline_kstar)
    else:
        k_values = list(range(1, args.batch_size + 1, args.k_step))

    # Validate k values
    k_values = [k for k in k_values if 1 <= k <= args.batch_size]

    print(f"\n{'='*60}")
    print("Starting k sweep")
    print(f"{'='*60}")
    print(f"k values to test: {k_values}")
    print(f"Total tests: {len(k_values)}")
    print(f"Requests per k: {args.num_requests}")
    total_requests = len(k_values) * args.num_requests
    print(f"Total requests: {total_requests}")

    # Run sweep
    throughput_by_k = {}
    best_k = 1
    best_throughput = 0

    overall_pbar = tqdm(total=total_requests, desc="Overall progress",
                        unit="req", ncols=100)

    for i, k in enumerate(k_values):
        overall_pbar.set_description(f"Testing k={k} ({i+1}/{len(k_values)})")

        throughput = run_single_k_test(
            llm, prompts, k,
            max_output_tokens=args.max_output_tokens,
            num_requests=args.num_requests,
            num_warmup=args.num_warmup,
        )
        throughput_by_k[k] = throughput

        overall_pbar.update(args.num_requests)
        tqdm.write(f"  k={k}: throughput={throughput:.2f} req/s")

        if throughput > best_throughput:
            best_throughput = throughput
            best_k = k

    overall_pbar.close()

    k_hat = best_k
    throughput_kstar = throughput_by_k.get(offline_kstar, None)

    # If k* wasn't in the sweep, test it
    if offline_kstar not in throughput_by_k:
        print(f"\nTesting offline k*={offline_kstar} specifically...")
        throughput_kstar = run_single_k_test(
            llm, prompts, offline_kstar,
            max_output_tokens=args.max_output_tokens,
            num_requests=args.num_requests,
            num_warmup=args.num_warmup,
        )
        throughput_by_k[offline_kstar] = throughput_kstar
        print(f"  k*={offline_kstar}: throughput={throughput_kstar:.2f} req/s")

    # Calculate gaps
    gap_kstar_vs_khat = 0
    if best_throughput > 0 and throughput_kstar is not None:
        gap_kstar_vs_khat = abs(best_throughput - throughput_kstar) / best_throughput * 100

    # Save results
    results = {
        'test_type': 'k_sweep',
        'model': args.model,
        'dataset': args.dataset,
        'batch_size_N': args.batch_size,
        'num_requests_per_k': args.num_requests,
        'k_values_tested': k_values,
        'offline_kstar': offline_kstar,
        'k_hat': k_hat,
        'delta_k': abs(offline_kstar - k_hat),
        'throughput_kstar': throughput_kstar,
        'throughput_khat': best_throughput,
        'throughput_gap_pct': gap_kstar_vs_khat,
        'profiled_params': {
            'alpha_p': profile_result.alpha_p,
            'alpha_d': profile_result.alpha_d,
            'beta_d': profile_result.beta_d,
            'p_offline': profile_result.p_estimated,
        },
        'throughput_by_k': {str(k): v for k, v in throughput_by_k.items()}
    }

    json_path = os.path.join(args.output_dir, f"{args.dataset}_k_sweep_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {json_path}")

    # Plot results
    plot_profiling_results(profile_result, args.output_dir, args.dataset)
    plot_throughput_curve(
        throughput_by_k,
        k_star=offline_kstar,
        k_hat=k_hat,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        online_kstar_throughput=args.online_kstar_throughput,
        kstar_is_offline=True,  # This is offline k* from formula
    )

    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Offline k*: {offline_kstar}")
    print(f"Empirical k_hat: {k_hat}")
    print(f"Δk: {abs(offline_kstar - k_hat)}")
    print(f"Throughput at k*: {throughput_kstar:.2f} req/s")
    print(f"Throughput at k_hat: {best_throughput:.2f} req/s")
    print(f"Throughput gap: {gap_kstar_vs_khat:.2f}%")
    print("="*60)

    # Print top 5 k values by throughput
    sorted_k = sorted(throughput_by_k.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 5 k values by throughput:")
    for k, tp in sorted_k[:5]:
        marker = " (k*)" if k == offline_kstar else " (k_hat)" if k == k_hat else ""
        print(f"  k={k}: {tp:.2f} req/s{marker}")

    return results


if __name__ == '__main__':
    main()
