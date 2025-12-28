"""
Memory Model Validation Experiment (Real Inference)

Validates the theoretical memory model from Section 3.3 of the P/D Competition paper
by running REAL inference with vLLM and measuring actual GPU memory usage.

This experiment validates three key theoretical predictions:
1. E[X0] - Expected initial memory at decode start
2. E[Xmax] - Expected peak memory during decode
3. P(sup Y_t > x) <= exp(-2|d|x/v) - OOM probability bound

Unlike Monte Carlo simulation, this script:
- Runs actual LLM inference with vLLM
- Measures real GPU memory via torch.cuda.memory_allocated()
- Tracks memory at each decode step
- Compares empirical measurements against theoretical predictions

Usage:
    python experiments/memory_model_validation.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset alpaca \
        --batch-size 32 \
        --num-cycles 100

For detailed documentation, see: experiments/README_memory_validation.md
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional
import json
import os
import time
import argparse
from tqdm import tqdm
from collections import Counter
import warnings

# Suppress tokenizer warnings
warnings.filterwarnings("ignore", message="Token indices sequence length")

# Configuration Constants
WORD_TO_TOKEN_RATIO = 1.3
BYTES_PER_GB = 1e9
BYTES_PER_MB = 1e6

# Try importing vLLM and torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not available")

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available")

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


@dataclass
class MemoryModelParams:
    """Parameters for the memory model."""
    N: int = 64              # Batch size
    k: int = 10              # Switching threshold
    p: float = 0.01          # Termination probability per step
    mean_input_len: float = 100.0
    var_input_len: float = 1000.0
    mean_output_len: float = 100.0

    @property
    def theta(self) -> float:
        return self.k / self.N if self.N > 0 else 0.2


@dataclass
class TheoreticalPredictions:
    """Theoretical predictions from the memory model."""
    E_X0: float = 0.0           # Expected initial memory (tokens)
    E_Xmax: float = 0.0         # Expected peak memory (tokens)
    kappa: float = 0.0          # Supremum constant κ
    d_N: float = 0.0            # Expected per-step drift
    v_N: float = 0.0            # Per-step variance
    E_O_partial: float = 0.0    # Expected partial output length


@dataclass
class MemoryMeasurement:
    """Memory measurements from one decode cycle."""
    initial_memory_bytes: float = 0.0    # Memory at decode start
    peak_memory_bytes: float = 0.0       # Peak memory during decode
    final_memory_bytes: float = 0.0      # Memory at decode end
    memory_trajectory: list = field(default_factory=list)  # Memory at each step
    num_steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    num_requests: int = 0


def get_gpu_memory_bytes() -> float:
    """Get current GPU memory allocated in bytes."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.cuda.memory_allocated()
    return 0.0


def get_gpu_memory_mb() -> float:
    """Get current GPU memory allocated in MB."""
    return get_gpu_memory_bytes() / BYTES_PER_MB


def load_prompts_from_dataset(
    dataset: str,
    max_samples: int = 5000,
    tokenizer=None,
    sharegpt_path: str = "./ShareGPT_V3_unfiltered_cleaned_split.json",
    max_input_tokens: int = 4096
) -> tuple[list[str], list[int], list[int]]:
    """Load prompts and their lengths from dataset."""

    if dataset == "alpaca":
        return load_alpaca_prompts(max_samples, tokenizer)
    elif dataset == "sharegpt":
        return load_sharegpt_prompts(sharegpt_path, max_samples, tokenizer)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def load_alpaca_prompts(
    max_samples: int,
    tokenizer=None
) -> tuple[list[str], list[int], list[int]]:
    """Load prompts from Alpaca dataset."""
    if not HF_AVAILABLE:
        raise ImportError("Please install datasets: pip install datasets")

    print("Loading tatsu-lab/alpaca from HuggingFace...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    prompts = []
    input_lengths = []
    output_lengths = []

    for i, item in enumerate(dataset):
        if i >= max_samples:
            break

        text = item.get('text', '')
        if text and "### Response:" in text:
            parts = text.split("### Response:", 1)
            input_text = parts[0].strip()
            output_text = parts[1].strip() if len(parts) > 1 else ""

            if input_text and output_text:
                prompts.append(input_text)
                if tokenizer:
                    input_len = len(tokenizer.encode(input_text))
                    output_len = len(tokenizer.encode(output_text))
                else:
                    input_len = int(len(input_text.split()) * WORD_TO_TOKEN_RATIO)
                    output_len = int(len(output_text.split()) * WORD_TO_TOKEN_RATIO)
                input_lengths.append(max(1, input_len))
                output_lengths.append(max(1, output_len))

    print(f"Loaded {len(prompts)} prompts from Alpaca")
    print(f"  Input length: mean={np.mean(input_lengths):.1f}, "
          f"std={np.std(input_lengths):.1f}")
    print(f"  Output length: mean={np.mean(output_lengths):.1f}, "
          f"std={np.std(output_lengths):.1f}")
    return prompts, input_lengths, output_lengths


def load_sharegpt_prompts(
    json_path: str,
    max_samples: int,
    tokenizer=None
) -> tuple[list[str], list[int], list[int]]:
    """Load prompts from ShareGPT dataset."""
    print(f"Loading ShareGPT from {json_path}...")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"ShareGPT file not found: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    prompts = []
    input_lengths = []
    output_lengths = []
    sample_count = 0

    for item in data:
        if sample_count >= max_samples:
            break

        conversations = item.get('conversations', [])
        if not conversations:
            continue

        i = 0
        while i < len(conversations) and sample_count < max_samples:
            turn = conversations[i]

            if turn.get('from') == 'human':
                input_text = turn.get('value', '').strip()
                output_text = ''

                if i + 1 < len(conversations):
                    next_turn = conversations[i + 1]
                    if next_turn.get('from') == 'gpt':
                        output_text = next_turn.get('value', '').strip()
                        i += 1

                if input_text and output_text:
                    prompts.append(input_text)
                    if tokenizer:
                        input_len = len(tokenizer.encode(input_text))
                        output_len = len(tokenizer.encode(output_text))
                    else:
                        input_len = int(len(input_text.split()) * WORD_TO_TOKEN_RATIO)
                        output_len = int(len(output_text.split()) * WORD_TO_TOKEN_RATIO)
                    input_lengths.append(max(1, input_len))
                    output_lengths.append(max(1, output_len))
                    sample_count += 1
            i += 1

    print(f"Loaded {len(prompts)} prompts from ShareGPT")
    print(f"  Input length: mean={np.mean(input_lengths):.1f}, "
          f"std={np.std(input_lengths):.1f}")
    print(f"  Output length: mean={np.mean(output_lengths):.1f}, "
          f"std={np.std(output_lengths):.1f}")
    return prompts, input_lengths, output_lengths


def compute_theoretical_predictions(
    params: MemoryModelParams,
    kv_cache_bytes_per_token: int = 256
) -> TheoreticalPredictions:
    """
    Compute theoretical predictions from the memory model.

    Returns predictions in BYTES (not tokens).
    """
    N = params.N
    p = params.p
    theta = params.theta
    E_L = params.mean_input_len
    c = kv_cache_bytes_per_token  # bytes per token

    theta = max(0.01, min(0.99, theta))
    p = max(1e-6, p)

    # E[O_partial] = (E[A] - 1) * T_d
    ln_term = np.log(1.0 / (1.0 - theta))
    T_d = ln_term / p
    E_A = 1.0 / theta
    E_O_partial = (E_A - 1) * T_d

    # E[X0] in tokens
    partial_output_coef = ((1.0 - theta) ** 2) / (theta * p) * ln_term
    E_X0_tokens = N * E_L + N * partial_output_coef

    # κ in tokens
    kappa_tokens = 1.0 / (p * p * E_L) if E_L > 0 else 0

    # E[Xmax] in tokens
    E_Xmax_tokens = E_X0_tokens + kappa_tokens

    # Convert to bytes
    E_X0 = E_X0_tokens * c
    E_Xmax = E_Xmax_tokens * c
    kappa = kappa_tokens * c

    # Drift and variance (in tokens, for probability bound)
    d_N = -N * p * E_L
    E_S = E_L + 1.0 / p
    Var_S = params.var_input_len + (1.0 - p) / (p * p)
    v_N = N * p * Var_S + N * p * (1 - p) * (E_S ** 2)

    return TheoreticalPredictions(
        E_X0=E_X0,
        E_Xmax=E_Xmax,
        kappa=kappa,
        d_N=d_N,
        v_N=v_N,
        E_O_partial=E_O_partial
    )


def run_single_decode_cycle_with_memory_tracking(
    llm: LLM,
    prompts: list[str],
    batch_size: int,
    max_output_tokens: int,
    prompt_idx_start: int = 0
) -> MemoryMeasurement:
    """
    Run one batch of inference and track memory throughout.

    This measures the actual GPU memory during decode phase.
    """
    measurement = MemoryMeasurement()

    # Select batch of prompts
    batch_prompts = prompts[prompt_idx_start:prompt_idx_start + batch_size]
    if len(batch_prompts) < batch_size:
        # Wrap around if needed
        batch_prompts = batch_prompts + prompts[:batch_size - len(batch_prompts)]

    measurement.num_requests = len(batch_prompts)

    # Force garbage collection before measurement
    if TORCH_AVAILABLE:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Record baseline memory
    baseline_memory = get_gpu_memory_bytes()

    # Sampling params for generation
    sampling_params = SamplingParams(
        max_tokens=max_output_tokens,
        temperature=0.7,
        top_p=0.9
    )

    # Run inference and track memory
    # Note: vLLM's generate() doesn't give us step-by-step access,
    # so we measure before and after, plus peak
    initial_memory = get_gpu_memory_bytes()
    measurement.initial_memory_bytes = initial_memory - baseline_memory

    # Run the generation
    outputs = llm.generate(batch_prompts, sampling_params)

    # Record final and peak memory
    if TORCH_AVAILABLE:
        torch.cuda.synchronize()

    final_memory = get_gpu_memory_bytes()
    peak_memory = torch.cuda.max_memory_allocated() if TORCH_AVAILABLE else final_memory

    measurement.final_memory_bytes = final_memory - baseline_memory
    measurement.peak_memory_bytes = peak_memory - baseline_memory

    # Count tokens
    total_input_tokens = 0
    total_output_tokens = 0
    for output in outputs:
        total_input_tokens += len(output.prompt_token_ids)
        total_output_tokens += len(output.outputs[0].token_ids)

    measurement.input_tokens = total_input_tokens
    measurement.output_tokens = total_output_tokens
    measurement.num_steps = total_output_tokens // batch_size if batch_size > 0 else 0

    # Reset peak memory for next measurement
    if TORCH_AVAILABLE:
        torch.cuda.reset_peak_memory_stats()

    return measurement


def run_memory_validation_experiment(
    llm: LLM,
    prompts: list[str],
    input_lengths: list[int],
    output_lengths: list[int],
    batch_size: int,
    num_cycles: int,
    max_output_tokens: int,
    kv_cache_bytes_per_token: int = 256
) -> dict:
    """
    Run multiple decode cycles and collect memory measurements.

    Returns dictionary with measurements and statistics.
    """
    print(f"\n--- Running {num_cycles} decode cycles ---")
    print(f"Batch size: {batch_size}")
    print(f"Max output tokens: {max_output_tokens}")

    measurements = []
    prompt_idx = 0

    # Reset peak memory stats
    if TORCH_AVAILABLE:
        torch.cuda.reset_peak_memory_stats()

    for cycle in tqdm(range(num_cycles), desc="Decode cycles"):
        measurement = run_single_decode_cycle_with_memory_tracking(
            llm, prompts, batch_size, max_output_tokens, prompt_idx
        )
        measurements.append(measurement)
        prompt_idx = (prompt_idx + batch_size) % len(prompts)

    # Aggregate statistics
    initial_memories = [m.initial_memory_bytes for m in measurements]
    peak_memories = [m.peak_memory_bytes for m in measurements]
    output_tokens_list = [m.output_tokens for m in measurements]
    input_tokens_list = [m.input_tokens for m in measurements]

    # Compute sup(Y) = peak - initial for each cycle
    sup_Y_values = [m.peak_memory_bytes - m.initial_memory_bytes for m in measurements]

    return {
        'measurements': measurements,
        'initial_memory': {
            'mean': np.mean(initial_memories),
            'std': np.std(initial_memories),
            'min': np.min(initial_memories),
            'max': np.max(initial_memories),
            'values': initial_memories
        },
        'peak_memory': {
            'mean': np.mean(peak_memories),
            'std': np.std(peak_memories),
            'min': np.min(peak_memories),
            'max': np.max(peak_memories),
            'values': peak_memories
        },
        'sup_Y': {
            'mean': np.mean(sup_Y_values),
            'std': np.std(sup_Y_values),
            'min': np.min(sup_Y_values),
            'max': np.max(sup_Y_values),
            'values': sup_Y_values
        },
        'output_tokens': {
            'mean': np.mean(output_tokens_list),
            'std': np.std(output_tokens_list),
        },
        'input_tokens': {
            'mean': np.mean(input_tokens_list),
            'std': np.std(input_tokens_list),
        }
    }


def validate_oom_probability_bound(
    sup_Y_values: list[float],
    d_N: float,
    v_N: float,
    kv_cache_bytes_per_token: int,
    num_thresholds: int = 20
) -> dict:
    """
    Validate the OOM probability bound: P(sup Y > x) <= exp(-2|d|x/v).

    Note: d_N and v_N are in tokens, sup_Y_values are in bytes.
    """
    sup_Y_arr = np.array(sup_Y_values)

    # Convert theoretical values to bytes
    c = kv_cache_bytes_per_token
    d_N_bytes = d_N * c
    v_N_bytes = v_N * c * c  # variance scales with c²

    x_min = np.percentile(sup_Y_arr, 50)
    x_max = np.percentile(sup_Y_arr, 99)

    if x_max <= x_min:
        x_max = x_min + 1

    thresholds = np.linspace(x_min, x_max, num_thresholds)

    observed_probs = []
    theoretical_bounds = []

    abs_d = abs(d_N_bytes)

    for x in thresholds:
        prob_observed = np.mean(sup_Y_arr > x)
        observed_probs.append(prob_observed)

        if v_N_bytes > 0 and abs_d > 0:
            exponent = -2 * abs_d * x / v_N_bytes
            bound = np.exp(max(-700, exponent))
        else:
            bound = 1.0
        theoretical_bounds.append(bound)

    return {
        'thresholds': thresholds,
        'observed_probs': np.array(observed_probs),
        'theoretical_bounds': np.array(theoretical_bounds),
        'bound_valid': np.array(observed_probs) <= np.array(theoretical_bounds) + 0.01
    }


def plot_validation_results(
    params: MemoryModelParams,
    theory: TheoreticalPredictions,
    empirical: dict,
    oom_validation: dict,
    dataset_name: str,
    output_dir: str
):
    """Generate validation plots comparing theory vs empirical measurements."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    initial_arr = np.array(empirical['initial_memory']['values']) / BYTES_PER_MB
    peak_arr = np.array(empirical['peak_memory']['values']) / BYTES_PER_MB
    sup_Y_arr = np.array(empirical['sup_Y']['values']) / BYTES_PER_MB

    theory_X0_mb = theory.E_X0 / BYTES_PER_MB
    theory_Xmax_mb = theory.E_Xmax / BYTES_PER_MB
    theory_kappa_mb = theory.kappa / BYTES_PER_MB

    # Plot 1: Initial Memory X0
    ax1 = axes[0, 0]
    ax1.hist(initial_arr, bins=30, density=True, alpha=0.7,
             label='Empirical X₀', color='steelblue')
    ax1.axvline(theory_X0_mb, color='red', linestyle='--', linewidth=2,
                label=f'Theory E[X₀] = {theory_X0_mb:.1f} MB')
    ax1.axvline(np.mean(initial_arr), color='green', linestyle='-', linewidth=2,
                label=f'Empirical Mean = {np.mean(initial_arr):.1f} MB')
    ax1.set_xlabel('Initial Memory X₀ (MB)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title(f'Initial Memory Distribution ({dataset_name})', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    error_X0 = abs(np.mean(initial_arr) - theory_X0_mb) / theory_X0_mb * 100
    ax1.text(0.95, 0.95, f'Error: {error_X0:.1f}%',
             transform=ax1.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Peak Memory Xmax
    ax2 = axes[0, 1]
    ax2.hist(peak_arr, bins=30, density=True, alpha=0.7,
             label='Empirical Xₘₐₓ', color='steelblue')
    ax2.axvline(theory_Xmax_mb, color='red', linestyle='--', linewidth=2,
                label=f'Theory E[Xₘₐₓ] = {theory_Xmax_mb:.1f} MB')
    ax2.axvline(np.mean(peak_arr), color='green', linestyle='-', linewidth=2,
                label=f'Empirical Mean = {np.mean(peak_arr):.1f} MB')
    ax2.set_xlabel('Peak Memory Xₘₐₓ (MB)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title(f'Peak Memory Distribution ({dataset_name})', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    error_Xmax = abs(np.mean(peak_arr) - theory_Xmax_mb) / theory_Xmax_mb * 100
    ax2.text(0.95, 0.95, f'Error: {error_Xmax:.1f}%',
             transform=ax2.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Memory Supremum sup(Y)
    ax3 = axes[1, 0]
    ax3.hist(sup_Y_arr, bins=30, density=True, alpha=0.7,
             label='Empirical sup(Y)', color='steelblue')
    ax3.axvline(theory_kappa_mb, color='red', linestyle='--', linewidth=2,
                label=f'Theory κ = {theory_kappa_mb:.1f} MB')
    ax3.axvline(np.mean(sup_Y_arr), color='green', linestyle='-', linewidth=2,
                label=f'Empirical Mean = {np.mean(sup_Y_arr):.1f} MB')
    ax3.set_xlabel('Memory Supremum sup(Y) (MB)', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.set_title(f'Memory Supremum Distribution ({dataset_name})', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    if theory_kappa_mb > 0:
        error_kappa = abs(np.mean(sup_Y_arr) - theory_kappa_mb) / theory_kappa_mb * 100
        ax3.text(0.95, 0.95, f'Error vs κ: {error_kappa:.1f}%',
                 transform=ax3.transAxes, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: OOM Probability Bound
    ax4 = axes[1, 1]
    thresholds_mb = oom_validation['thresholds'] / BYTES_PER_MB

    # Filter out zero probabilities for log scale
    obs_probs = oom_validation['observed_probs']
    theo_bounds = oom_validation['theoretical_bounds']
    mask = obs_probs > 0

    if np.any(mask):
        ax4.semilogy(thresholds_mb[mask], obs_probs[mask],
                     'bo-', linewidth=2, markersize=6,
                     label='Empirical P(sup Y > x)')
    ax4.semilogy(thresholds_mb, theo_bounds,
                 'r--', linewidth=2, label='Bound: exp(-2|d|x/v)')

    ax4.set_xlabel('Threshold x (MB)', fontsize=12)
    ax4.set_ylabel('Probability (log scale)', fontsize=12)
    ax4.set_title(f'OOM Probability Bound Validation ({dataset_name})', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    valid_pct = np.mean(oom_validation['bound_valid']) * 100
    color = 'lightgreen' if valid_pct > 90 else 'lightsalmon'
    ax4.text(0.95, 0.05, f'Bound valid: {valid_pct:.0f}%',
             transform=ax4.transAxes, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{dataset_name}_memory_validation.pdf")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved validation plot to {plot_path}")


def run_full_validation(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    dataset: str = "alpaca",
    batch_size: int = 32,
    switching_threshold: Optional[int] = None,
    num_cycles: int = 100,
    max_output_tokens: int = 1024,
    output_dir: str = "./memory_validation_results",
    sharegpt_path: str = "./ShareGPT_V3_unfiltered_cleaned_split.json",
    tensor_parallel_size: int = 1,
    kv_cache_bytes_per_token: int = 256,
    max_samples: int = 5000
):
    """Run the full memory model validation experiment."""

    if not VLLM_AVAILABLE:
        print("Error: vLLM is required. Please install it first.")
        return None

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Memory Model Validation (Real Inference)")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset}")
    print(f"Batch size N: {batch_size}")
    print(f"Num cycles: {num_cycles}")
    print(f"Max output tokens: {max_output_tokens}")

    # Load tokenizer
    tokenizer = None
    if TRANSFORMERS_AVAILABLE:
        print(f"\nLoading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load dataset
    print("\n--- Loading Dataset ---")
    prompts, input_lengths, output_lengths = load_prompts_from_dataset(
        dataset, max_samples, tokenizer, sharegpt_path
    )

    if len(prompts) == 0:
        print("Error: No prompts loaded")
        return None

    # Compute statistics
    mean_input = np.mean(input_lengths)
    var_input = np.var(input_lengths)
    mean_output = np.mean(output_lengths)

    # Estimate p from output lengths
    p_estimated = 1.0 / mean_output if mean_output > 0 else 0.01

    # Set switching threshold
    k = switching_threshold if switching_threshold else max(1, batch_size // 5)

    print(f"\nModel parameters:")
    print(f"  N = {batch_size}")
    print(f"  k = {k} (θ = {k/batch_size:.2f})")
    print(f"  p = {p_estimated:.4f} (from mean output length)")
    print(f"  E[L] = {mean_input:.1f} tokens")
    print(f"  E[O] = {mean_output:.1f} tokens")

    # Create model parameters
    params = MemoryModelParams(
        N=batch_size,
        k=k,
        p=p_estimated,
        mean_input_len=mean_input,
        var_input_len=var_input,
        mean_output_len=mean_output
    )

    # Compute theoretical predictions
    print("\n--- Theoretical Predictions ---")
    theory = compute_theoretical_predictions(params, kv_cache_bytes_per_token)

    print(f"E[X₀] (initial memory): {theory.E_X0/BYTES_PER_MB:.1f} MB")
    print(f"κ (supremum constant): {theory.kappa/BYTES_PER_MB:.1f} MB")
    print(f"E[Xₘₐₓ] (peak memory): {theory.E_Xmax/BYTES_PER_MB:.1f} MB")

    # Initialize vLLM
    print("\n--- Initializing vLLM ---")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        max_num_seqs=batch_size,
        gpu_memory_utilization=0.6,
    )

    # Run validation
    empirical = run_memory_validation_experiment(
        llm, prompts, input_lengths, output_lengths,
        batch_size, num_cycles, max_output_tokens,
        kv_cache_bytes_per_token
    )

    # Print results
    print("\n--- Empirical Results ---")
    print(f"X₀ (initial memory):")
    print(f"  Theory E[X₀]: {theory.E_X0/BYTES_PER_MB:.1f} MB")
    print(f"  Empirical mean: {empirical['initial_memory']['mean']/BYTES_PER_MB:.1f} MB")
    error_X0 = abs(empirical['initial_memory']['mean'] - theory.E_X0) / theory.E_X0 * 100
    print(f"  Error: {error_X0:.1f}%")

    print(f"\nXₘₐₓ (peak memory):")
    print(f"  Theory E[Xₘₐₓ]: {theory.E_Xmax/BYTES_PER_MB:.1f} MB")
    print(f"  Empirical mean: {empirical['peak_memory']['mean']/BYTES_PER_MB:.1f} MB")
    error_Xmax = abs(empirical['peak_memory']['mean'] - theory.E_Xmax) / theory.E_Xmax * 100
    print(f"  Error: {error_Xmax:.1f}%")

    print(f"\nsup(Y) (memory supremum):")
    print(f"  Theory κ: {theory.kappa/BYTES_PER_MB:.1f} MB")
    print(f"  Empirical E[sup Y]: {empirical['sup_Y']['mean']/BYTES_PER_MB:.1f} MB")
    if theory.kappa > 0:
        error_kappa = abs(empirical['sup_Y']['mean'] - theory.kappa) / theory.kappa * 100
        print(f"  Error vs κ: {error_kappa:.1f}%")

    # Validate OOM probability bound
    print("\n--- OOM Probability Bound Validation ---")
    oom_validation = validate_oom_probability_bound(
        empirical['sup_Y']['values'],
        theory.d_N, theory.v_N,
        kv_cache_bytes_per_token
    )

    bound_valid_pct = np.mean(oom_validation['bound_valid']) * 100
    print(f"Bound P(sup Y > x) ≤ exp(-2|d|x/v) valid at {bound_valid_pct:.0f}% of thresholds")

    if bound_valid_pct >= 90:
        print("✓ OOM probability bound is VALID")
    else:
        print("✗ OOM probability bound may be VIOLATED")

    # Generate plots
    print("\n--- Generating Plots ---")
    plot_validation_results(
        params, theory, empirical, oom_validation,
        dataset, output_dir
    )

    # Save results
    results = {
        'model': model_name,
        'dataset': dataset,
        'params': {
            'N': params.N,
            'k': params.k,
            'theta': params.theta,
            'p': params.p,
            'mean_input_len': params.mean_input_len,
            'mean_output_len': params.mean_output_len,
        },
        'theoretical': {
            'E_X0_bytes': theory.E_X0,
            'E_Xmax_bytes': theory.E_Xmax,
            'kappa_bytes': theory.kappa,
        },
        'empirical': {
            'X0_mean_bytes': empirical['initial_memory']['mean'],
            'X0_std_bytes': empirical['initial_memory']['std'],
            'Xmax_mean_bytes': empirical['peak_memory']['mean'],
            'Xmax_std_bytes': empirical['peak_memory']['std'],
            'sup_Y_mean_bytes': empirical['sup_Y']['mean'],
            'sup_Y_std_bytes': empirical['sup_Y']['std'],
        },
        'errors': {
            'X0_error_pct': error_X0,
            'Xmax_error_pct': error_Xmax,
            'kappa_error_pct': error_kappa if theory.kappa > 0 else 0,
        },
        'oom_bound': {
            'valid_pct': float(bound_valid_pct),
        },
        'num_cycles': num_cycles,
    }

    results_path = os.path.join(output_dir, f"{dataset}_memory_validation.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Batch size N = {batch_size}, k = {k}, θ = {params.theta:.2f}")
    print(f"\n{'Metric':<20} {'Theory (MB)':>15} {'Empirical (MB)':>15} {'Error':>10}")
    print("-" * 60)
    print(f"{'E[X₀]':<20} {theory.E_X0/BYTES_PER_MB:>15.1f} "
          f"{empirical['initial_memory']['mean']/BYTES_PER_MB:>15.1f} "
          f"{error_X0:>9.1f}%")
    print(f"{'E[Xₘₐₓ]':<20} {theory.E_Xmax/BYTES_PER_MB:>15.1f} "
          f"{empirical['peak_memory']['mean']/BYTES_PER_MB:>15.1f} "
          f"{error_Xmax:>9.1f}%")
    print(f"{'E[sup Y] vs κ':<20} {theory.kappa/BYTES_PER_MB:>15.1f} "
          f"{empirical['sup_Y']['mean']/BYTES_PER_MB:>15.1f} "
          f"{error_kappa if theory.kappa > 0 else 0:>9.1f}%")
    print(f"\nOOM Probability Bound: "
          f"{'VALID' if bound_valid_pct >= 90 else 'VIOLATED'} "
          f"({bound_valid_pct:.0f}%)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate memory model with real LLM inference"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model name or path")
    parser.add_argument("--dataset", type=str, default="alpaca",
                        choices=["alpaca", "sharegpt"],
                        help="Dataset to use")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size N")
    parser.add_argument("--switching-threshold", type=int, default=None,
                        help="Switching threshold k (default: N//5)")
    parser.add_argument("--num-cycles", type=int, default=100,
                        help="Number of decode cycles to run")
    parser.add_argument("--max-output-tokens", type=int, default=1024,
                        help="Max output tokens per request")
    parser.add_argument("--output-dir", type=str,
                        default="./memory_validation_results",
                        help="Output directory")
    parser.add_argument("--sharegpt-path", type=str,
                        default="./ShareGPT_V3_unfiltered_cleaned_split.json",
                        help="Path to ShareGPT JSON file")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor parallel size")
    parser.add_argument("--kv-cache-bytes", type=int, default=256,
                        help="KV cache bytes per token")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Max samples to load from dataset")
    parser.add_argument("--run-both", action="store_true",
                        help="Run on both Alpaca and ShareGPT")

    args = parser.parse_args()

    if args.run_both:
        print("\n" + "=" * 70)
        print("RUNNING VALIDATION ON BOTH DATASETS")
        print("=" * 70)

        # Run on Alpaca
        print("\n>>> ALPACA DATASET <<<")
        alpaca_results = run_full_validation(
            model_name=args.model,
            dataset="alpaca",
            batch_size=args.batch_size,
            switching_threshold=args.switching_threshold,
            num_cycles=args.num_cycles,
            max_output_tokens=args.max_output_tokens,
            output_dir=args.output_dir,
            tensor_parallel_size=args.tensor_parallel_size,
            kv_cache_bytes_per_token=args.kv_cache_bytes,
            max_samples=args.max_samples
        )

        # Run on ShareGPT
        print("\n>>> SHAREGPT DATASET <<<")
        try:
            sharegpt_results = run_full_validation(
                model_name=args.model,
                dataset="sharegpt",
                batch_size=args.batch_size,
                switching_threshold=args.switching_threshold,
                num_cycles=args.num_cycles,
                max_output_tokens=args.max_output_tokens,
                output_dir=args.output_dir,
                sharegpt_path=args.sharegpt_path,
                tensor_parallel_size=args.tensor_parallel_size,
                kv_cache_bytes_per_token=args.kv_cache_bytes,
                max_samples=args.max_samples
            )
        except FileNotFoundError as e:
            print(f"Skipping ShareGPT: {e}")
            sharegpt_results = None

        # Comparison
        if alpaca_results and sharegpt_results:
            print("\n" + "=" * 70)
            print("COMPARISON: ALPACA vs SHAREGPT")
            print("=" * 70)
            print(f"\n{'Metric':<25} {'Alpaca':>15} {'ShareGPT':>15}")
            print("-" * 55)
            print(f"{'E[X₀] error %':<25} "
                  f"{alpaca_results['errors']['X0_error_pct']:>14.1f}% "
                  f"{sharegpt_results['errors']['X0_error_pct']:>14.1f}%")
            print(f"{'E[Xₘₐₓ] error %':<25} "
                  f"{alpaca_results['errors']['Xmax_error_pct']:>14.1f}% "
                  f"{sharegpt_results['errors']['Xmax_error_pct']:>14.1f}%")
            print(f"{'κ error %':<25} "
                  f"{alpaca_results['errors']['kappa_error_pct']:>14.1f}% "
                  f"{sharegpt_results['errors']['kappa_error_pct']:>14.1f}%")
            print(f"{'OOM bound valid %':<25} "
                  f"{alpaca_results['oom_bound']['valid_pct']:>14.0f}% "
                  f"{sharegpt_results['oom_bound']['valid_pct']:>14.0f}%")

    else:
        run_full_validation(
            model_name=args.model,
            dataset=args.dataset,
            batch_size=args.batch_size,
            switching_threshold=args.switching_threshold,
            num_cycles=args.num_cycles,
            max_output_tokens=args.max_output_tokens,
            output_dir=args.output_dir,
            sharegpt_path=args.sharegpt_path,
            tensor_parallel_size=args.tensor_parallel_size,
            kv_cache_bytes_per_token=args.kv_cache_bytes,
            max_samples=args.max_samples
        )


if __name__ == "__main__":
    main()
