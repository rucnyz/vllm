"""
Shared utilities for P/D Competition Scheduler experiments.

This module contains common functions used by:
- run_dynamic_kstar.py: Dynamic k* throughput test
- run_fixed_k.py: Fixed k throughput test
- run_k_sweep.py: k sweep search for k_hat
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter
import json
import os
import time
import warnings
from tqdm import tqdm

# Suppress tokenizer length warnings globally
warnings.filterwarnings(
    "ignore",
    message="Token indices sequence length is longer than"
)

# HuggingFace datasets
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: 'datasets' library not installed. Run: pip install datasets")

# Transformers (for tokenizer)
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available for tokenizer")

# ============================================================
# Configuration Constants
# ============================================================

DEFAULT_EMA_ALPHA = 0.3
DEFAULT_INITIAL_P = 0.01
DEFAULT_OUTPUT_LEN_ESTIMATE = 100.0
WORD_TO_TOKEN_RATIO = 1.3

DEFAULT_PREFILL_SAMPLES = 64
DEFAULT_DECODE_REPEATS = 5
DEFAULT_P_ESTIMATION_SAMPLES = 64

DEFAULT_INFERENCE_BATCH_SIZE = 16
DEFAULT_KSTAR_UPDATE_INTERVAL = 10

DEFAULT_OUTPUT_DIR = "./experiment_results"
DEFAULT_SHAREGPT_PATH = "./ShareGPT_V3_unfiltered_cleaned_split.json"


@dataclass
class PDModelParams:
    """Parameters for the Prefill-Decode competition model."""
    N: int = 64
    alpha_p: float = 0.0
    alpha_d: float = 0.0
    beta_d: float = 0.0
    beta_p: float = 0.0
    avg_input_len: float = 0.0
    avg_output_len: float = 0.0


@dataclass
class ProfilingResult:
    """Results from profiling the model."""
    alpha_p: float = 0.0
    beta_p: float = 0.0
    alpha_d: float = 0.0
    beta_d: float = 0.0
    p_estimated: float = 0.0
    prefill_times: list = field(default_factory=list)
    decode_times: list = field(default_factory=list)
    input_lengths: list = field(default_factory=list)
    batch_sizes: list = field(default_factory=list)
    output_lengths: list = field(default_factory=list)
    var_input_len: float = 0.0


class OnlinePEstimator:
    """Online estimator for p using Exponential Moving Average (EMA)."""

    def __init__(self, initial_p: float = DEFAULT_INITIAL_P,
                 ema_alpha: float = DEFAULT_EMA_ALPHA):
        self.ema_alpha = ema_alpha
        self.p = initial_p
        self.mean_length = 1.0 / initial_p if initial_p > 0 else 100.0
        self.p_history: list[float] = [initial_p]
        self.mean_length_history: list[float] = [self.mean_length]
        self.k_star_history: list[int] = []
        self.num_samples = 0

    def update(self, output_length: int):
        self.num_samples += 1
        self.mean_length = (self.ema_alpha * output_length +
                           (1 - self.ema_alpha) * self.mean_length)
        self.p = 1.0 / self.mean_length if self.mean_length > 0 else 0.01
        self.p_history.append(self.p)
        self.mean_length_history.append(self.mean_length)

    def update_batch(self, output_lengths: list[int]):
        for length in output_lengths:
            self.update(length)

    def record_kstar(self, k_star: int):
        self.k_star_history.append(k_star)

    def get_p(self) -> float:
        return self.p

    def get_statistics(self) -> dict:
        p_array = np.array(self.p_history)
        k_array = np.array(self.k_star_history) if self.k_star_history else None

        stats = {
            'p_final': self.p,
            'p_mean': float(np.mean(p_array)),
            'p_std': float(np.std(p_array)),
            'p_min': float(np.min(p_array)),
            'p_max': float(np.max(p_array)),
            'mean_length_final': self.mean_length,
            'num_samples': self.num_samples,
        }

        if k_array is not None and len(k_array) > 0:
            stats.update({
                'k_star_final': int(k_array[-1]),
                'k_star_mean': float(np.mean(k_array)),
                'k_star_std': float(np.std(k_array)),
                'k_star_min': int(np.min(k_array)),
                'k_star_max': int(np.max(k_array)),
                'k_star_mode': int(np.bincount(k_array).argmax()),
            })

        return stats


def is_degenerate_output(text: str, threshold: float = 0.5,
                         min_length: int = 100) -> bool:
    """Detect degenerate outputs with repetitive characters."""
    if len(text) < min_length:
        return False

    char_counts = Counter(text)
    if not char_counts:
        return False

    most_common_char, most_common_count = char_counts.most_common(1)[0]
    ratio = most_common_count / len(text)

    if ratio >= threshold:
        return True

    for pattern_len in range(2, 6):
        if len(text) < pattern_len * 10:
            continue
        pattern = text[:pattern_len]
        expected_repeats = len(text) // pattern_len
        actual_text = pattern * expected_repeats
        if text[:len(actual_text)] == actual_text:
            return True

    return False


def compute_tau(batch_size: int, p: float,
                alpha_d: float, beta_d: float) -> float:
    """Compute expected time per completion with given batch size."""
    if batch_size <= 0:
        return float('inf')
    numerator = alpha_d + beta_d * batch_size
    denominator = 1.0 - (1.0 - p) ** batch_size
    if denominator <= 1e-10:
        return float('inf')
    return numerator / denominator


def compute_analytical_kstar(N: int, p: float, alpha_p: float,
                             alpha_d: float, beta_d: float) -> int:
    """Compute optimal k* using Proposition 1."""
    for k in range(1, N + 1):
        lhs = k * compute_tau(N - k, p, alpha_d, beta_d)
        sum_tau = sum(compute_tau(j, p, alpha_d, beta_d)
                      for j in range(N - k + 1, N + 1))
        rhs = sum_tau + alpha_p

        if lhs >= rhs:
            return max(1, k)

    return max(1, N // 5)


# ============================================================
# Dataset Loading Functions
# ============================================================

def parse_alpaca_text(text: str) -> tuple[str, str]:
    """Parse Alpaca text format to extract input and output."""
    if "### Response:" in text:
        parts = text.split("### Response:", 1)
        input_text = parts[0].strip()
        output_text = parts[1].strip() if len(parts) > 1 else ""
    else:
        input_text = text
        output_text = ""
    return input_text, output_text


def load_alpaca_prompts(max_samples: int = 1000, tokenizer=None
                        ) -> tuple[list[str], list[int], list[int]]:
    """Load prompts from Alpaca dataset."""
    if not HF_AVAILABLE:
        raise ImportError("Please install datasets: pip install datasets")

    print("Loading tatsu-lab/alpaca from HuggingFace...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    prompts = []
    input_lengths = []
    output_lengths = []

    use_tokenizer = tokenizer is not None

    for i, item in enumerate(dataset):
        if i >= max_samples:
            break

        text = item.get('text', '')
        if text:
            input_text, output_text = parse_alpaca_text(text)

            if input_text and output_text:
                prompts.append(input_text)

                if use_tokenizer:
                    input_len = len(tokenizer.encode(input_text))
                    output_len = len(tokenizer.encode(output_text))
                else:
                    input_len = int(len(input_text.split()) * WORD_TO_TOKEN_RATIO)
                    output_len = int(len(output_text.split()) * WORD_TO_TOKEN_RATIO)

                input_lengths.append(input_len)
                output_lengths.append(output_len)

    print(f"Loaded {len(prompts)} prompts")
    print(f"Average input length: {np.mean(input_lengths):.1f} tokens")
    print(f"Average output length: {np.mean(output_lengths):.1f} tokens")

    return prompts, input_lengths, output_lengths


def load_sharegpt_prompts(json_path: str = DEFAULT_SHAREGPT_PATH,
                          max_samples: int = 1000, tokenizer=None
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

    use_tokenizer = tokenizer is not None
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

                    if use_tokenizer:
                        input_len = len(tokenizer.encode(input_text))
                        output_len = len(tokenizer.encode(output_text))
                    else:
                        input_len = int(len(input_text.split()) * WORD_TO_TOKEN_RATIO)
                        output_len = int(len(output_text.split()) * WORD_TO_TOKEN_RATIO)

                    input_lengths.append(input_len)
                    output_lengths.append(output_len)
                    sample_count += 1

            i += 1

    print(f"Loaded {len(prompts)} prompts from ShareGPT")
    if len(prompts) > 0:
        print(f"Average input length: {np.mean(input_lengths):.1f} tokens")
        print(f"Average output length: {np.mean(output_lengths):.1f} tokens")

    return prompts, input_lengths, output_lengths


def load_lmsys_prompts(max_samples: int = 1000, tokenizer=None
                       ) -> tuple[list[str], list[int], list[int]]:
    """Load prompts from LMSYS-Chat-1M dataset."""
    if not HF_AVAILABLE:
        raise ImportError("Please install datasets: pip install datasets")

    print("Loading lmsys/lmsys-chat-1m from HuggingFace...")
    dataset = load_dataset("lmsys/lmsys-chat-1m", split="train")

    prompts = []
    input_lengths = []
    output_lengths = []

    use_tokenizer = tokenizer is not None
    sample_count = 0

    for item in dataset:
        if sample_count >= max_samples:
            break

        conversation = item.get('conversation', [])
        if not conversation:
            continue

        input_text = ''
        output_text = ''

        for turn in conversation:
            role = turn.get('role', '')
            content = turn.get('content', '').strip()

            if role == 'user' and not input_text:
                input_text = content
            elif role == 'assistant' and input_text and not output_text:
                output_text = content
                break

        if input_text and output_text:
            prompts.append(input_text)

            if use_tokenizer:
                input_len = len(tokenizer.encode(input_text))
                output_len = len(tokenizer.encode(output_text))
            else:
                input_len = int(len(input_text.split()) * WORD_TO_TOKEN_RATIO)
                output_len = int(len(output_text.split()) * WORD_TO_TOKEN_RATIO)

            input_lengths.append(input_len)
            output_lengths.append(output_len)
            sample_count += 1

    print(f"Loaded {len(prompts)} prompts from LMSYS-Chat-1M")
    if len(prompts) > 0:
        print(f"Average input length: {np.mean(input_lengths):.1f} tokens")
        print(f"Average output length: {np.mean(output_lengths):.1f} tokens")

    return prompts, input_lengths, output_lengths


def load_dataset_prompts(dataset: str, tokenizer, max_samples: int,
                         sharegpt_path: str = DEFAULT_SHAREGPT_PATH,
                         max_input_tokens: int = 32000,
                         processbench_split: str = "gsm8k"
                         ) -> tuple[list[str], list[int]]:
    """Load prompts from specified dataset."""
    if dataset == "sharegpt":
        prompts, input_lengths, _ = load_sharegpt_prompts(
            json_path=sharegpt_path,
            max_samples=max_samples,
            tokenizer=tokenizer
        )
    elif dataset == "lmsys":
        prompts, input_lengths, _ = load_lmsys_prompts(
            max_samples=max_samples,
            tokenizer=tokenizer
        )
    else:  # alpaca (default)
        prompts, input_lengths, _ = load_alpaca_prompts(
            max_samples=max_samples,
            tokenizer=tokenizer
        )

    return prompts, input_lengths


def apply_chat_template(prompts: list[str], tokenizer,
                        enable_thinking: bool = True) -> list[str]:
    """Apply chat template to prompts."""
    print(f"Applying chat template (enable_thinking={enable_thinking})...")
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

    print(f"Applied chat template to {len(formatted_prompts)} prompts")
    return formatted_prompts


# ============================================================
# Profiling Functions
# ============================================================

def profile_model(llm, prompts: list[str],
                  num_profile_samples: int = 50,
                  num_p_estimation_samples: int = 100,
                  max_output_tokens: int = 256,
                  num_prefill_samples: int = 30,
                  num_decode_repeats: int = 3,
                  max_batch_size: int = 64) -> ProfilingResult:
    """Profile the model to measure timing parameters."""
    from vllm import SamplingParams

    print("\n" + "="*60)
    print("Profiling model parameters...")
    print("="*60)

    result = ProfilingResult()
    sample_prompts = prompts[:num_profile_samples]

    # Profile prefill
    print(f"\nProfiling prefill times ({num_prefill_samples} samples)...")
    prefill_times = []
    input_lens = []

    for prompt in tqdm(sample_prompts[:num_prefill_samples], desc="Prefill profiling"):
        sampling_params = SamplingParams(max_tokens=1, temperature=0)

        start = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params)
        elapsed = time.perf_counter() - start

        input_len = len(outputs[0].prompt_token_ids)
        prefill_times.append(elapsed)
        input_lens.append(input_len)

    if len(input_lens) > 1:
        coeffs = np.polyfit(input_lens, prefill_times, 1)
        result.beta_p = coeffs[0]
        result.alpha_p = max(0, coeffs[1])

    result.prefill_times = prefill_times
    result.input_lengths = input_lens

    print(f"  Prefill: α_p = {result.alpha_p:.4f}s, β_p = {result.beta_p:.6f}s/token")

    # Profile decode
    print(f"\nProfiling decode times (repeats={num_decode_repeats})...")
    decode_times_by_batch = {}
    batch_sizes_tested = []

    test_batch_sizes = [1, 2, 4, 8, 16]
    bs = 24
    while bs <= max_batch_size:
        test_batch_sizes.append(bs)
        bs += 8 if bs < 32 else 16

    test_batch_sizes = sorted(set(b for b in test_batch_sizes if b <= max_batch_size))

    for batch_size in tqdm(test_batch_sizes, desc="Decode profiling"):
        if batch_size > len(sample_prompts):
            continue

        times_for_this_batch = []

        for _ in range(num_decode_repeats):
            batch_prompts = sample_prompts[:batch_size]
            sampling_params = SamplingParams(max_tokens=50, temperature=0)

            start = time.perf_counter()
            outputs = llm.generate(batch_prompts, sampling_params)
            elapsed = time.perf_counter() - start

            total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            if total_output_tokens > 0:
                time_per_step = elapsed / (total_output_tokens / batch_size)
                times_for_this_batch.append(time_per_step)

        if times_for_this_batch:
            median_time = np.median(times_for_this_batch)
            decode_times_by_batch[batch_size] = median_time
            batch_sizes_tested.append(batch_size)

    decode_times = [decode_times_by_batch[bs] for bs in batch_sizes_tested]

    if len(batch_sizes_tested) > 1:
        coeffs = np.polyfit(batch_sizes_tested, decode_times, 1)
        result.beta_d = coeffs[0]
        result.alpha_d = max(0, coeffs[1])

    result.decode_times = decode_times
    result.batch_sizes = batch_sizes_tested

    print(f"  Decode: α_d = {result.alpha_d:.4f}s, β_d = {result.beta_d:.6f}s/request")

    # Estimate p
    print("\nEstimating p from model outputs...")
    p_estimation_prompts = prompts[:num_p_estimation_samples]
    output_lengths = []
    truncated_count = 0

    sampling_params = SamplingParams(
        max_tokens=max_output_tokens,
        temperature=0.7,
        top_p=0.9
    )

    batch_size = 16
    for i in tqdm(range(0, len(p_estimation_prompts), batch_size),
                  desc="Estimating p"):
        batch = p_estimation_prompts[i:i + batch_size]
        outputs = llm.generate(batch, sampling_params)

        for output in outputs:
            output_len = len(output.outputs[0].token_ids)
            output_lengths.append(output_len)
            if output.outputs[0].finish_reason == "length":
                truncated_count += 1

    result.output_lengths = output_lengths

    if len(output_lengths) > 0:
        mean_output_len = np.mean(output_lengths)
        result.p_estimated = 1.0 / mean_output_len if mean_output_len > 0 else 0.01

        print(f"  Output lengths: mean={mean_output_len:.1f}, "
              f"std={np.std(output_lengths):.1f}")
        print(f"  Estimated p = 1/{mean_output_len:.1f} = {result.p_estimated:.4f}")

        if truncated_count > 0:
            truncated_pct = 100.0 * truncated_count / len(output_lengths)
            print(f"  WARNING: {truncated_count}/{len(output_lengths)} "
                  f"({truncated_pct:.1f}%) outputs truncated")
    else:
        result.p_estimated = 0.01

    return result


# ============================================================
# Plotting Functions
# ============================================================

def plot_throughput_curve(throughput_by_k: dict, k_star: int, k_hat: int,
                          output_dir: str, dataset_name: str = "",
                          online_kstar_throughput: float = None,
                          kstar_is_offline: bool = True):
    """
    Plot throughput vs k curve.

    Args:
        throughput_by_k: Dict mapping k values to throughput.
        k_star: The theoretical optimal k* (from formula).
        k_hat: The empirical best k (highest throughput).
        output_dir: Directory to save the plot.
        dataset_name: Name of the dataset for labeling.
        online_kstar_throughput: If provided, draw horizontal line for dynamic k* throughput.
        kstar_is_offline: If True, label k* as "offline"; if False, label as "online".
    """
    k_values = sorted(throughput_by_k.keys())
    throughputs = [throughput_by_k[k] for k in k_values]

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, throughputs, 'b-o', markersize=4, label='Throughput')

    # Mark k* (label depends on whether it's offline or online)
    kstar_label = "offline" if kstar_is_offline else "online"
    if k_star in throughput_by_k:
        plt.axvline(x=k_star, color='r', linestyle='--', linewidth=2,
                    label=f'k* ({kstar_label}) = {k_star}')
        plt.scatter([k_star], [throughput_by_k[k_star]], color='r', s=100, zorder=5)

    # Mark k_hat
    if k_hat in throughput_by_k:
        plt.axvline(x=k_hat, color='g', linestyle='--', linewidth=2,
                    label=f'k_hat = {k_hat}')
        plt.scatter([k_hat], [throughput_by_k[k_hat]], color='g', s=100, zorder=5)

    # Dynamic k* throughput line
    if online_kstar_throughput is not None:
        plt.axhline(y=online_kstar_throughput, color='purple', linestyle='-.',
                    linewidth=2, label=f'Dynamic k* = {online_kstar_throughput:.2f}')

    plt.xlabel('Switching Threshold k')
    plt.ylabel('Throughput (requests/second)')
    title = f'Throughput vs Switching Threshold k'
    if dataset_name:
        title += f' ({dataset_name})'
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    prefix = f"{dataset_name}_" if dataset_name else ""
    plt.savefig(os.path.join(output_dir, f'{prefix}throughput_vs_k.pdf'),
                bbox_inches='tight')
    plt.close()
    print(f"Saved throughput curve to {output_dir}/{prefix}throughput_vs_k.pdf")


def plot_profiling_results(profile_result: ProfilingResult, output_dir: str,
                           dataset_name: str = ""):
    """Plot profiling results (prefill and decode time fitting)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Prefill plot
    ax1 = axes[0]
    ax1.scatter(profile_result.input_lengths, profile_result.prefill_times,
                alpha=0.6, label='Measured')
    if profile_result.input_lengths:
        x_fit = np.linspace(min(profile_result.input_lengths),
                            max(profile_result.input_lengths), 100)
        y_fit = profile_result.alpha_p + profile_result.beta_p * x_fit
        ax1.plot(x_fit, y_fit, 'r-', linewidth=2,
                 label=f'Fit: {profile_result.alpha_p:.4f} + {profile_result.beta_p:.6f}×L')
    ax1.set_xlabel('Input Length (tokens)')
    ax1.set_ylabel('Prefill Time (seconds)')
    ax1.set_title('Prefill Time vs Input Length')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Decode plot
    ax2 = axes[1]
    ax2.scatter(profile_result.batch_sizes, profile_result.decode_times,
                alpha=0.6, label='Measured')
    if profile_result.batch_sizes:
        x_fit = np.linspace(min(profile_result.batch_sizes),
                            max(profile_result.batch_sizes), 100)
        y_fit = profile_result.alpha_d + profile_result.beta_d * x_fit
        ax2.plot(x_fit, y_fit, 'r-', linewidth=2,
                 label=f'Fit: {profile_result.alpha_d:.4f} + {profile_result.beta_d:.6f}×j')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Time per Decode Step (seconds)')
    ax2.set_title('Decode Time vs Batch Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    prefix = f"{dataset_name}_" if dataset_name else ""
    plt.savefig(os.path.join(output_dir, f'{prefix}profiling_results.pdf'),
                bbox_inches='tight')
    plt.close()


def plot_output_length_distribution(output_lengths: list, output_dir: str,
                                    dataset_name: str = "",
                                    max_output_tokens: int = None):
    """Plot output length distribution histogram."""
    if not output_lengths:
        return

    plt.figure(figsize=(10, 6))
    plt.hist(output_lengths, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=np.mean(output_lengths), color='r', linestyle='--',
                label=f'Mean = {np.mean(output_lengths):.1f}')
    plt.axvline(x=np.median(output_lengths), color='g', linestyle='--',
                label=f'Median = {np.median(output_lengths):.1f}')

    if max_output_tokens:
        plt.axvline(x=max_output_tokens, color='orange', linestyle=':',
                    label=f'Max = {max_output_tokens}')

    plt.xlabel('Output Length (tokens)')
    plt.ylabel('Frequency')
    title = 'Output Length Distribution'
    if dataset_name:
        title += f' ({dataset_name})'
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    prefix = f"{dataset_name}_" if dataset_name else ""
    plt.savefig(os.path.join(output_dir, f'{prefix}output_length_distribution.pdf'),
                bbox_inches='tight')
    plt.close()


def plot_online_kstar_convergence(estimator: OnlinePEstimator, output_dir: str,
                                  dataset_name: str = ""):
    """Plot online k* convergence over time."""
    if not estimator.k_star_history:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # k* over time
    ax1 = axes[0]
    ax1.plot(estimator.k_star_history, 'b-', linewidth=1)
    ax1.set_ylabel('k*')
    ax1.set_title('Online k* Convergence')
    ax1.grid(True, alpha=0.3)

    # p over time
    ax2 = axes[1]
    ax2.plot(estimator.p_history, 'r-', linewidth=0.5, alpha=0.7)
    ax2.set_xlabel('Update Step')
    ax2.set_ylabel('p')
    ax2.set_title('Online p Estimation')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    prefix = f"{dataset_name}_" if dataset_name else ""
    plt.savefig(os.path.join(output_dir, f'{prefix}online_kstar_convergence.pdf'),
                bbox_inches='tight')
    plt.close()


def plot_input_length_distribution(input_lengths: list, output_dir: str,
                                   dataset_name: str = ""):
    """Plot input length distribution histogram."""
    if not input_lengths:
        return

    plt.figure(figsize=(10, 6))
    plt.hist(input_lengths, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=np.mean(input_lengths), color='r', linestyle='--',
                label=f'Mean = {np.mean(input_lengths):.1f}')
    plt.axvline(x=np.median(input_lengths), color='g', linestyle='--',
                label=f'Median = {np.median(input_lengths):.1f}')

    plt.xlabel('Input Length (tokens)')
    plt.ylabel('Frequency')
    title = 'Input Length Distribution'
    if dataset_name:
        title += f' ({dataset_name})'
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    prefix = f"{dataset_name}_" if dataset_name else ""
    plt.savefig(os.path.join(output_dir, f'{prefix}input_length_distribution.pdf'),
                bbox_inches='tight')
    plt.close()


# ============================================================
# Common Argument Parser
# ============================================================

def add_common_args(parser):
    """Add common arguments to argument parser."""
    # Model configuration
    parser.add_argument('--model', type=str, required=True,
                        help='HuggingFace model name or local path')
    parser.add_argument('--tensor-parallel-size', type=int, default=1,
                        help='Number of GPUs for tensor parallelism')

    # Dataset selection
    parser.add_argument('--dataset', type=str, default='alpaca',
                        choices=['alpaca', 'sharegpt', 'lmsys'],
                        help='Dataset choice')
    parser.add_argument('--sharegpt-path', type=str, default=DEFAULT_SHAREGPT_PATH,
                        help='Path to ShareGPT JSON file')
    parser.add_argument('--max-input-tokens', type=int, default=32000,
                        help='Skip samples exceeding this input length')

    # Experiment parameters
    parser.add_argument('--num-requests', type=int, default=500,
                        help='Number of requests for throughput testing')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Scheduler batch size N')
    parser.add_argument('--max-output-tokens', type=int, default=4096,
                        help='Maximum output tokens per request')

    # Profiling configuration
    parser.add_argument('--num-prefill-samples', type=int, default=30,
                        help='Number of samples for prefill profiling')
    parser.add_argument('--num-decode-repeats', type=int, default=3,
                        help='Repetitions per batch size for decode profiling')
    parser.add_argument('--offline-p-multiplier', type=float, default=2.0,
                        help='Offline samples = multiplier × batch-size')

    # Other
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory for saving results')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                        help='GPU memory utilization for vLLM')
    parser.add_argument('--enable-thinking', action='store_true', default=True,
                        help='Enable thinking mode for Qwen3 models')
    parser.add_argument('--disable-thinking', dest='enable_thinking',
                        action='store_false',
                        help='Disable thinking mode')

    return parser
