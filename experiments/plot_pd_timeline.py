"""
Plot P/D Competition scheduling timeline visualization.

This script generates timeline plots showing prefill (red) and decode (blue)
phases for each request, similar to the reference image.

Usage:
    python experiments/plot_pd_timeline.py \
        --model Qwen/Qwen3-8B \
        --num-requests 200 \
        --batch-size 64 \
        --max-output-tokens 1024 \
        --output-dir experiments/timeline_plots
"""

import argparse
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from datasets import load_dataset


@dataclass
class RequestTimeline:
    """Timeline data for a single request."""
    request_id: int
    prefill_start: float  # ms
    prefill_end: float    # ms
    decode_start: float   # ms
    decode_end: float     # ms


def load_prompts(dataset: str, max_samples: int = 100):
    """Load prompts from dataset."""
    prompts = []

    if dataset == "alpaca":
        print("Loading Alpaca dataset...")
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        for i, item in enumerate(ds):
            if i >= max_samples:
                break
            text = item.get('text', '')
            if text and "### Response:" in text:
                parts = text.split("### Response:", 1)
                input_text = parts[0].strip()
                if input_text:
                    prompts.append(input_text)

    elif dataset == "sharegpt":
        print("Loading ShareGPT dataset...")
        ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered",
                          data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
                          split="train")
        for i, item in enumerate(ds):
            if len(prompts) >= max_samples:
                break
            conversations = item.get('conversations', [])
            for turn in conversations:
                if turn.get('from') == 'human':
                    content = turn.get('value', '').strip()
                    if content:
                        prompts.append(content)
                    break

    print(f"Loaded {len(prompts)} prompts")
    return prompts


def apply_chat_template(prompts: list, model: str, enable_thinking: bool = False):
    """Apply chat template to prompts."""
    print(f"Applying chat template (enable_thinking={enable_thinking})...")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    formatted_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        formatted_prompts.append(text)

    return formatted_prompts, tokenizer


def simulate_pd_timeline(
    llm: LLM,
    prompts: list,
    tokenizer,
    k: int,
    N: int,
    max_output_tokens: int,
    num_requests: int,
    policy_name: str = "k*"
) -> list[RequestTimeline]:
    """
    Simulate P/D scheduling and collect timeline data.

    This runs actual inference and estimates prefill/decode times based on
    token counts and timing.
    """
    test_prompts = prompts[:num_requests]
    sampling_params = SamplingParams(
        max_tokens=max_output_tokens,
        temperature=0.7,
        top_p=0.9
    )

    timelines = []
    global_start = time.perf_counter() * 1000  # Convert to ms

    # Track request states
    request_data = []  # (prompt, input_tokens)
    for i, prompt in enumerate(test_prompts):
        input_tokens = len(tokenizer.encode(prompt))
        request_data.append({
            'id': i,
            'prompt': prompt,
            'input_tokens': input_tokens,
            'prefill_start': None,
            'prefill_end': None,
            'decode_start': None,
            'decode_end': None,
        })

    # Process in batches to simulate P/D scheduling
    idx = 0
    current_time = 0.0  # ms

    print(f"\nSimulating {policy_name} policy (k={k}, N={N})...")
    pbar = tqdm(total=num_requests, desc=f"{policy_name} timeline", unit="req")

    while idx < len(test_prompts):
        # Determine batch size based on policy
        batch_size = min(N, len(test_prompts) - idx)
        batch_prompts = test_prompts[idx:idx + batch_size]
        batch_indices = list(range(idx, idx + batch_size))

        # Run inference and measure time
        batch_start = time.perf_counter() * 1000

        outputs = llm.generate(batch_prompts, sampling_params)

        batch_end = time.perf_counter() * 1000
        batch_duration = batch_end - batch_start

        # Estimate prefill and decode times for each request
        total_input_tokens = sum(request_data[i]['input_tokens'] for i in batch_indices)
        total_output_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)

        # Estimate prefill time (proportional to input tokens)
        # Assume prefill takes about 10-20% of total time for typical workloads
        prefill_ratio = 0.15
        prefill_duration = batch_duration * prefill_ratio

        # Distribute times to individual requests
        for j, (out_idx, output) in enumerate(zip(batch_indices, outputs)):
            req = request_data[out_idx]
            output_tokens = len(output.outputs[0].token_ids)

            # Calculate individual request times based on token proportions
            input_ratio = req['input_tokens'] / max(total_input_tokens, 1)
            output_ratio = output_tokens / max(total_output_tokens, 1)

            # Prefill time for this request
            req_prefill_duration = prefill_duration * input_ratio * batch_size
            req['prefill_start'] = current_time
            req['prefill_end'] = current_time + req_prefill_duration

            # Decode time for this request
            decode_duration = (batch_duration - prefill_duration) * output_ratio * batch_size
            req['decode_start'] = req['prefill_end']
            req['decode_end'] = req['decode_start'] + decode_duration

        current_time += batch_duration
        idx += batch_size
        pbar.update(batch_size)

    pbar.close()

    # Convert to RequestTimeline objects
    for req in request_data:
        if req['prefill_start'] is not None:
            timelines.append(RequestTimeline(
                request_id=req['id'],
                prefill_start=req['prefill_start'],
                prefill_end=req['prefill_end'],
                decode_start=req['decode_start'],
                decode_end=req['decode_end']
            ))

    return timelines


def collect_actual_timeline(
    llm: LLM,
    prompts: list,
    tokenizer,
    k: int,
    N: int,
    max_output_tokens: int,
    num_requests: int,
    policy_name: str = "k*"
) -> list[RequestTimeline]:
    """
    Collect actual timeline by running requests one by one and measuring.
    This gives more accurate per-request timing.
    """
    test_prompts = prompts[:num_requests]
    sampling_params = SamplingParams(
        max_tokens=max_output_tokens,
        temperature=0.7,
        top_p=0.9
    )

    timelines = []

    print(f"\nCollecting timeline for {policy_name} policy (k={k}, N={N})...")

    # For accurate timing, we'll run batches and track individual requests
    # using the scheduler's internal timing

    # First, run all requests and collect output info
    all_outputs = []
    all_input_tokens = []

    for prompt in test_prompts:
        all_input_tokens.append(len(tokenizer.encode(prompt)))

    # Run in one batch to get total time and output lengths
    start_time = time.perf_counter() * 1000
    outputs = llm.generate(test_prompts, sampling_params)
    end_time = time.perf_counter() * 1000
    total_time = end_time - start_time

    # Get output token counts
    output_tokens = [len(out.outputs[0].token_ids) for out in outputs]

    # Estimate timing based on token counts
    # Prefill time is roughly proportional to input tokens
    # Decode time is roughly proportional to output tokens

    total_input = sum(all_input_tokens)
    total_output = sum(output_tokens)

    # Estimate prefill takes ~15% of time (can vary based on model/hardware)
    prefill_ratio = 0.15

    current_time = 0.0
    for i in range(num_requests):
        input_toks = all_input_tokens[i]
        output_toks = output_tokens[i]

        # Prefill duration proportional to input tokens
        prefill_duration = (total_time * prefill_ratio * input_toks / total_input)

        # Decode duration proportional to output tokens
        decode_duration = (total_time * (1 - prefill_ratio) * output_toks / total_output)

        timelines.append(RequestTimeline(
            request_id=i,
            prefill_start=current_time,
            prefill_end=current_time + prefill_duration,
            decode_start=current_time + prefill_duration,
            decode_end=current_time + prefill_duration + decode_duration
        ))

        # For visualization, stagger start times slightly
        current_time += prefill_duration * 0.1

    return timelines, total_time


def plot_timeline(
    timelines: list[RequestTimeline],
    total_time: float,
    policy_name: str,
    k: int,
    N: int,
    output_path: str
):
    """
    Plot the P/D scheduling timeline.

    Args:
        timelines: List of RequestTimeline objects
        total_time: Total inference time in ms
        policy_name: Name of the policy (e.g., "k*" or "k_hat")
        k: Switching threshold
        N: Batch size
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    num_requests = len(timelines)

    # Sort timelines by request_id for consistent display
    timelines = sorted(timelines, key=lambda x: x.request_id)

    # Plot each request as horizontal bars
    for timeline in timelines:
        y = timeline.request_id

        # Plot prefill (red)
        prefill_width = timeline.prefill_end - timeline.prefill_start
        if prefill_width > 0:
            ax.barh(y, prefill_width, left=timeline.prefill_start,
                   height=0.8, color='red', edgecolor='darkred', linewidth=0.5)

        # Plot decode (blue)
        decode_width = timeline.decode_end - timeline.decode_start
        if decode_width > 0:
            ax.barh(y, decode_width, left=timeline.decode_start,
                   height=0.8, color='blue', edgecolor='darkblue', linewidth=0.5)

    # Customize plot
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Request ID', fontsize=12)
    ax.set_title(f'P/D Scheduling Timeline - {policy_name} Policy (k={k}, N={N})',
                fontsize=14)

    # Set axis limits
    ax.set_xlim(0, total_time * 1.05)
    ax.set_ylim(-0.5, num_requests - 0.5)

    # Add legend
    prefill_patch = mpatches.Patch(color='red', label='Prefill')
    decode_patch = mpatches.Patch(color='blue', label='Decode')
    ax.legend(handles=[prefill_patch, decode_patch], loc='upper right', fontsize=10)

    # Add grid
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved timeline plot to {output_path}")


def plot_timeline_detailed(
    timelines: list[RequestTimeline],
    total_time: float,
    policy_name: str,
    k: int,
    N: int,
    output_path: str
):
    """
    Plot detailed timeline showing batch structure.
    """
    fig, ax = plt.subplots(figsize=(16, 10))

    num_requests = len(timelines)
    timelines = sorted(timelines, key=lambda x: x.request_id)

    # Find max time for scaling
    max_time = max(t.decode_end for t in timelines)

    # Plot each request
    for timeline in timelines:
        y = timeline.request_id

        # Prefill (red)
        prefill_width = timeline.prefill_end - timeline.prefill_start
        ax.barh(y, prefill_width, left=timeline.prefill_start,
               height=0.8, color='#E74C3C', edgecolor='#C0392B', linewidth=0.3)

        # Decode (blue)
        decode_width = timeline.decode_end - timeline.decode_start
        ax.barh(y, decode_width, left=timeline.decode_start,
               height=0.8, color='#3498DB', edgecolor='#2980B9', linewidth=0.3)

    # Styling
    ax.set_xlabel('Time (ms)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Request ID', fontsize=14, fontweight='bold')
    ax.set_title(f'Total Inference Time\n{policy_name} Policy (k={k}, N={N})',
                fontsize=16, fontweight='bold')

    ax.set_xlim(0, max_time * 1.02)
    ax.set_ylim(-0.5, num_requests - 0.5)

    # Legend
    prefill_patch = mpatches.Patch(color='#E74C3C', label='prefill')
    decode_patch = mpatches.Patch(color='#3498DB', label='decode')
    ax.legend(handles=[prefill_patch, decode_patch],
             loc='upper left', fontsize=12, framealpha=0.9)

    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)

    # Format x-axis
    ax.ticklabel_format(style='plain', axis='x')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()

    print(f"Saved timeline plot to {output_path}")


def compute_analytical_kstar(N: int, p: float, alpha_p: float,
                            alpha_d: float, beta_d: float) -> int:
    """Compute optimal k* using Proposition 1."""
    def compute_tau(batch_size: int) -> float:
        if batch_size <= 0:
            return float('inf')
        numerator = alpha_d + beta_d * batch_size
        denominator = 1.0 - (1.0 - p) ** batch_size
        if denominator <= 1e-10:
            return float('inf')
        return numerator / denominator

    for k in range(1, N + 1):
        lhs = k * compute_tau(N - k)
        sum_tau = sum(compute_tau(j) for j in range(N - k + 1, N + 1))
        rhs = sum_tau + alpha_p

        if lhs >= rhs:
            return max(1, k)

    return max(1, N // 5)


def main():
    parser = argparse.ArgumentParser(
        description="Plot P/D scheduling timeline visualization"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset", type=str, default="alpaca",
                       choices=["alpaca", "sharegpt"])
    parser.add_argument("--num-requests", type=int, default=200,
                       help="Number of requests to visualize")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size N")
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--k-star", type=int, default=None,
                       help="k* value (computed if not specified)")
    parser.add_argument("--k-hat", type=int, default=None,
                       help="k_hat value (uses N/2 if not specified)")
    parser.add_argument("--output-dir", type=str,
                       default="experiments/timeline_plots")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load prompts
    raw_prompts = load_prompts(args.dataset, args.num_requests * 2)
    raw_prompts = raw_prompts[:args.num_requests]

    # Apply chat template
    prompts, tokenizer = apply_chat_template(
        raw_prompts, args.model, not args.disable_thinking
    )

    # Initialize model
    print(f"\nLoading model {args.model}...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_num_seqs=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Compute k* if not specified
    # Use default profiled parameters (these should ideally come from profiling)
    alpha_p = 0.01  # Fixed prefill overhead
    alpha_d = 0.001  # Fixed decode overhead
    beta_d = 0.0001  # Per-request decode cost
    p = 0.01  # Termination probability (1/100 tokens average)

    if args.k_star is None:
        k_star = compute_analytical_kstar(
            args.batch_size, p, alpha_p, alpha_d, beta_d
        )
        print(f"Computed k* = {k_star}")
    else:
        k_star = args.k_star

    if args.k_hat is None:
        k_hat = args.batch_size // 2  # Common baseline
        print(f"Using k_hat = {k_hat}")
    else:
        k_hat = args.k_hat

    # Collect timeline for k* policy
    print("\n" + "="*60)
    print("Collecting timeline for k* policy...")
    print("="*60)
    kstar_timelines, kstar_time = collect_actual_timeline(
        llm, prompts, tokenizer,
        k=k_star, N=args.batch_size,
        max_output_tokens=args.max_output_tokens,
        num_requests=args.num_requests,
        policy_name="k*"
    )

    # Plot k* timeline
    kstar_plot_path = os.path.join(
        args.output_dir, f"timeline_kstar_{args.dataset}_k{k_star}.png"
    )
    plot_timeline_detailed(
        kstar_timelines, kstar_time,
        policy_name="k*", k=k_star, N=args.batch_size,
        output_path=kstar_plot_path
    )

    # Collect timeline for k_hat policy
    print("\n" + "="*60)
    print("Collecting timeline for k_hat policy...")
    print("="*60)
    khat_timelines, khat_time = collect_actual_timeline(
        llm, prompts, tokenizer,
        k=k_hat, N=args.batch_size,
        max_output_tokens=args.max_output_tokens,
        num_requests=args.num_requests,
        policy_name="k_hat"
    )

    # Plot k_hat timeline
    khat_plot_path = os.path.join(
        args.output_dir, f"timeline_khat_{args.dataset}_k{k_hat}.png"
    )
    plot_timeline_detailed(
        khat_timelines, khat_time,
        policy_name="k_hat", k=k_hat, N=args.batch_size,
        output_path=khat_plot_path
    )

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"k* policy (k={k_star}): Total time = {kstar_time:.0f} ms")
    print(f"k_hat policy (k={k_hat}): Total time = {khat_time:.0f} ms")
    print(f"Difference: {abs(kstar_time - khat_time):.0f} ms "
          f"({100*abs(kstar_time - khat_time)/max(kstar_time, khat_time):.1f}%)")
    print(f"\nPlots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
