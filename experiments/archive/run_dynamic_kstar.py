#!/usr/bin/env python3
"""
Dynamic k* Throughput Test

Runs throughput test with DYNAMIC k* using vLLM's internal online learning.
The scheduler updates p and k* periodically as requests complete.

IMPORTANT: Set VLLM_PD_ENABLE_DYNAMIC_KSTAR=1 BEFORE LLM initialization
to enable dynamic k* updates in the scheduler.

Usage:
    python experiments/run_dynamic_kstar.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset alpaca \
        --num-requests 500 \
        --batch-size 32
"""

import os
import sys
import time
import json
import argparse

# CRITICAL: Set environment variables BEFORE importing vLLM
os.environ["VLLM_USE_PD_SCHEDULER"] = "1"
os.environ["VLLM_PD_ENABLE_DYNAMIC_KSTAR"] = "1"

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from pd_experiment_utils import (
    DEFAULT_EMA_ALPHA,
    DEFAULT_KSTAR_UPDATE_INTERVAL,
    OnlinePEstimator,
    add_common_args,
    apply_chat_template,
    compute_analytical_kstar,
    load_dataset_prompts,
    plot_input_length_distribution,
    plot_online_kstar_convergence,
    plot_output_length_distribution,
    plot_profiling_results,
    profile_model,
)
import numpy as np


def run_dynamic_kstar_test(
    llm: LLM,
    prompts: list[str],
    N: int,
    alpha_p: float,
    alpha_d: float,
    beta_d: float,
    max_output_tokens: int,
    num_requests: int,
    initial_p: float,
    ema_alpha: float,
    update_interval: int,
    num_warmup: int = 5,
) -> tuple[float, OnlinePEstimator, list[int]]:
    """
    Run throughput test with DYNAMIC k*.

    Args:
        llm: The LLM instance.
        prompts: List of prompts to use.
        N: Batch size.
        alpha_p, alpha_d, beta_d: Timing parameters from profiling.
        max_output_tokens: Maximum output tokens per request.
        num_requests: Number of requests for the actual test.
        initial_p: Initial termination probability from profiling.
        ema_alpha: EMA smoothing factor for online p updates.
        update_interval: Number of requests between k* updates.
        num_warmup: Number of warmup requests before throughput test.

    Returns: (throughput, estimator, output_lengths)
    """
    sampling_params = SamplingParams(
        max_tokens=max_output_tokens,
        temperature=0.7,
        top_p=0.9
    )

    # Create estimator to track results
    estimator = OnlinePEstimator(initial_p=initial_p, ema_alpha=ema_alpha)

    # Calculate initial k*
    initial_k_star = compute_analytical_kstar(
        N, initial_p, alpha_p, alpha_d, beta_d)
    estimator.record_kstar(initial_k_star)

    # Set initial k* for scheduler via environment and trigger state reset
    os.environ["VLLM_PD_K_STAR"] = str(initial_k_star)
    os.environ["VLLM_PD_P"] = str(initial_p)
    os.environ["VLLM_PD_RESET_STATE"] = "1"

    print(f"\n{'='*60}")
    print("Running DYNAMIC k* throughput test")
    print(f"{'='*60}")
    print(f"Initial k* = {initial_k_star}")
    print(f"Initial p = {initial_p:.4f}")
    print(f"EMA alpha = {ema_alpha}")
    print(f"Update interval = {update_interval}")
    print(f"Num requests = {num_requests}")

    # Warmup: run a few requests to stabilize the scheduler state
    if num_warmup > 0:
        warmup_prompts = prompts[:num_warmup]
        _ = llm.generate(warmup_prompts, sampling_params)

    # Actual test: use prompts after warmup
    test_prompts = prompts[num_warmup:num_warmup + num_requests]

    # Run all requests in one call
    start_time = time.perf_counter()
    print(f"\nProcessing {len(test_prompts)} requests...")
    outputs = llm.generate(test_prompts, sampling_params)
    elapsed = time.perf_counter() - start_time
    # Collect output lengths and track k* evolution
    all_output_lengths = []
    for output in outputs:
        output_len = len(output.outputs[0].token_ids)
        all_output_lengths.append(output_len)
        estimator.update(output_len)
        # Compute and record k* after each p update
        current_p = estimator.get_p()
        current_kstar = compute_analytical_kstar(N, current_p, alpha_p, alpha_d, beta_d)
        estimator.record_kstar(current_kstar)

    # Calculate final statistics
    throughput = len(outputs) / elapsed if elapsed > 0 else 0
    final_p = estimator.get_p()
    final_k_star = compute_analytical_kstar(N, final_p, alpha_p, alpha_d, beta_d)

    print(f"\nDynamic k* test completed:")
    print(f"  Throughput: {throughput:.2f} req/s")
    print(f"  Total time: {elapsed:.2f}s for {len(outputs)} requests")
    print(f"  Final p (from outputs): {final_p:.4f}")
    print(f"  Final k* (computed): {final_k_star}")

    return throughput, estimator, all_output_lengths


def main():
    parser = argparse.ArgumentParser(
        description='Run dynamic k* throughput test')
    add_common_args(parser)

    # Dynamic k* specific args
    parser.add_argument('--ema-alpha', type=float, default=DEFAULT_EMA_ALPHA,
                        help='EMA smoothing factor for online p updates')
    parser.add_argument('--kstar-update-interval', type=int,
                        default=DEFAULT_KSTAR_UPDATE_INTERVAL,
                        help='Number of requests between k* updates')
    parser.add_argument('--num-warmup', type=int, default=5,
                        help='Number of warmup requests before throughput test')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set scheduler parameters via environment
    os.environ["VLLM_PD_EMA_ALPHA"] = str(args.ema_alpha)
    os.environ["VLLM_PD_UPDATE_INTERVAL"] = str(args.kstar_update_interval)

    print("="*60)
    print("Dynamic k* Throughput Test")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size N: {args.batch_size}")
    print(f"Num requests: {args.num_requests}")
    print(f"EMA alpha: {args.ema_alpha}")
    print(f"Update interval: {args.kstar_update_interval}")
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

    # Set profiled parameters
    os.environ["VLLM_PD_ALPHA_P"] = str(profile_result.alpha_p)
    os.environ["VLLM_PD_ALPHA_D"] = str(profile_result.alpha_d)
    os.environ["VLLM_PD_BETA_D"] = str(profile_result.beta_d)

    print("\nProfiling parameters:")
    print(f"  α_p = {profile_result.alpha_p:.6f}")
    print(f"  α_d = {profile_result.alpha_d:.6f}")
    print(f"  β_d = {profile_result.beta_d:.6f}")
    print(f"  p (offline) = {profile_result.p_estimated:.4f}")

    # Run dynamic k* test
    throughput, estimator, output_lengths = run_dynamic_kstar_test(
        llm, prompts,
        N=args.batch_size,
        alpha_p=profile_result.alpha_p,
        alpha_d=profile_result.alpha_d,
        beta_d=profile_result.beta_d,
        max_output_tokens=args.max_output_tokens,
        num_requests=args.num_requests,
        initial_p=profile_result.p_estimated,
        ema_alpha=args.ema_alpha,
        update_interval=args.kstar_update_interval,
        num_warmup=args.num_warmup,
    )

    # Get statistics
    stats = estimator.get_statistics()

    # Save results
    results = {
        'test_type': 'dynamic_kstar',
        'model': args.model,
        'dataset': args.dataset,
        'batch_size_N': args.batch_size,
        'num_requests': args.num_requests,
        'throughput': throughput,
        'ema_alpha': args.ema_alpha,
        'update_interval': args.kstar_update_interval,
        'profiled_params': {
            'alpha_p': profile_result.alpha_p,
            'alpha_d': profile_result.alpha_d,
            'beta_d': profile_result.beta_d,
            'p_offline': profile_result.p_estimated,
        },
        'online_stats': stats,
        'output_lengths': {
            'mean': float(np.mean(output_lengths)),
            'std': float(np.std(output_lengths)),
            'min': int(min(output_lengths)),
            'max': int(max(output_lengths)),
        }
    }

    json_path = os.path.join(args.output_dir, f"{args.dataset}_dynamic_kstar_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {json_path}")

    # Plot results
    plot_profiling_results(profile_result, args.output_dir, args.dataset)
    plot_online_kstar_convergence(estimator, args.output_dir, args.dataset)
    plot_output_length_distribution(output_lengths, args.output_dir, args.dataset,
                                    args.max_output_tokens)

    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Throughput: {throughput:.2f} req/s")
    print(f"Final p: {stats['p_final']:.4f}")
    print(f"p range: [{stats['p_min']:.4f}, {stats['p_max']:.4f}]")
    if 'k_star_final' in stats:
        print(f"Final k*: {stats['k_star_final']}")
        print(f"k* range: [{stats['k_star_min']}, {stats['k_star_max']}]")
        print(f"k* mode: {stats['k_star_mode']}")
    print("="*60)

    return results


if __name__ == '__main__':
    main()
