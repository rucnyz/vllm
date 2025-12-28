"""
Experiment: Comparing analytical k* vs empirical k_hat using a REAL model.

This script validates the P/D Competition Scheduler's analytical k* formula
by comparing it with the empirically optimal k_hat on real LLM inference.

Workflow:
    1. Load prompts from supported datasets
    2. Profile the model to measure timing parameters (alpha_p, alpha_d,
       beta_p, beta_d)
    3. Estimate termination probability p (offline or online with EMA)
    4. Compute analytical k* using Proposition 1
    5. Test different k values to find empirical optimal k_hat
    6. Compare throughput gap between k* and k_hat

Supported datasets:
    - alpaca: Instruction-following, short inputs, medium outputs
    - sharegpt: Multi-turn conversation, heavy-tail output distribution
    - longbench: Long-context QA (v2), very long inputs, short outputs
    - longbench_v1: Long-context QA (v1)
    - lmsys: Real user conversations from LMSYS-Chat-1M
    - processbench: Math problem solving (gsm8k, math, olympiadbench, omni-math)

For detailed usage, see: experiments/README_kstar_experiment.md
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter
import json
import os
import time
import argparse
import warnings
from tqdm import tqdm

# Suppress tokenizer length warnings globally (for LongBench long contexts)
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

# vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available")

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

# Default values for p estimation
DEFAULT_EMA_ALPHA = 0.3
DEFAULT_INITIAL_P = 0.01
DEFAULT_OUTPUT_LEN_ESTIMATE = 100.0

# Token estimation
WORD_TO_TOKEN_RATIO = 1.3

# Profiling defaults
DEFAULT_PREFILL_SAMPLES = 64
DEFAULT_DECODE_REPEATS = 5
DEFAULT_P_ESTIMATION_SAMPLES = 64

# Batch processing
DEFAULT_INFERENCE_BATCH_SIZE = 16
DEFAULT_KSTAR_UPDATE_INTERVAL = 10

# Output paths
DEFAULT_OUTPUT_DIR = "./experiment_results"
DEFAULT_SHAREGPT_PATH = "./ShareGPT_V3_unfiltered_cleaned_split.json"


@dataclass
class PDModelParams:
    """Parameters for the Prefill-Decode competition model."""
    N: int = 64              # Batch size
    alpha_p: float = 0.0     # Fixed prefill overhead (seconds)
    alpha_d: float = 0.0     # Fixed decode overhead per step (seconds)
    beta_d: float = 0.0      # Per-request decode cost (seconds)
    beta_p: float = 0.0      # Per-token prefill cost (seconds)
    avg_input_len: float = 0.0   # Average input length
    avg_output_len: float = 0.0  # Average output length


@dataclass
class ProfilingResult:
    """Results from profiling the model."""
    alpha_p: float = 0.0     # Fixed prefill overhead
    beta_p: float = 0.0      # Per-token prefill cost
    alpha_d: float = 0.0     # Fixed decode overhead
    beta_d: float = 0.0      # Per-request decode cost
    p_estimated: float = 0.0 # Estimated termination probability from model outputs
    prefill_times: list = field(default_factory=list)
    decode_times: list = field(default_factory=list)
    input_lengths: list = field(default_factory=list)
    batch_sizes: list = field(default_factory=list)
    output_lengths: list = field(default_factory=list)  # Actual output lengths
    var_input_len: float = 0.0  # Variance of input length


@dataclass
class MemoryConstraintConfig:
    """
    Configuration for memory-constrained batch sizing.

    Based on Section 3.3 of the P/D Competition paper.

    Memory model:
        - Total GPU memory: M (bytes)
        - Model weights: m (bytes)
        - Per-token KV cache: c (bytes)
        - Available budget: C = (M - m) / c (tokens)

    The optimal batch size N* ensures peak memory stays within budget
    with probability at least (1 - epsilon).
    """
    total_memory_gb: float = 80.0    # Total GPU memory M (GB)
    model_memory_gb: float = 16.0    # Model weights m (GB)
    kv_cache_bytes_per_token: int = 256  # Per-token KV cache size c (bytes)
    epsilon: float = 0.05            # OOM probability tolerance


class OnlinePEstimator:
    """
    Online estimator for p using Exponential Moving Average (EMA).

    Updates p incrementally as new output lengths are observed.
    """

    def __init__(self, initial_p: float = DEFAULT_INITIAL_P,
                 ema_alpha: float = DEFAULT_EMA_ALPHA):
        """
        Args:
            initial_p: Initial estimate of p
            ema_alpha: EMA smoothing factor (0 < alpha <= 1)
                       Higher alpha = more weight on recent observations
        """
        self.ema_alpha = ema_alpha
        self.p = initial_p
        self.mean_length = 1.0 / initial_p if initial_p > 0 else 100.0

        # History tracking for analysis
        self.p_history: list[float] = [initial_p]
        self.mean_length_history: list[float] = [self.mean_length]
        self.k_star_history: list[int] = []
        self.num_samples = 0

    def update(self, output_length: int):
        """Update p estimate with a new observed output length."""
        self.num_samples += 1

        # Update mean length using EMA
        self.mean_length = (self.ema_alpha * output_length +
                           (1 - self.ema_alpha) * self.mean_length)

        # Update p = 1 / mean_length
        self.p = 1.0 / self.mean_length if self.mean_length > 0 else 0.01

        # Record history
        self.p_history.append(self.p)
        self.mean_length_history.append(self.mean_length)

    def update_batch(self, output_lengths: list[int]):
        """Update p estimate with a batch of observed output lengths."""
        for length in output_lengths:
            self.update(length)

    def record_kstar(self, k_star: int):
        """Record k* value for history tracking."""
        self.k_star_history.append(k_star)

    def get_p(self) -> float:
        """Get current p estimate."""
        return self.p

    def get_statistics(self) -> dict:
        """Get statistics about p estimation."""
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


class OutputLengthDistribution:
    """Represents an output length distribution from a dataset."""

    def __init__(self, lengths: list[int], name: str):
        self.name = name
        self.lengths = np.array(lengths)
        self.min_len = max(1, int(self.lengths.min()))
        self.max_len = int(self.lengths.max())

        # Compute PMF
        counter = Counter(lengths)
        total = len(lengths)
        self.pmf = np.zeros(self.max_len + 1)
        for l, count in counter.items():
            if l <= self.max_len:
                self.pmf[l] = count / total

        # Compute survival function S(l) = P(L > l)
        self.survival = np.zeros(self.max_len + 2)
        self.survival[0] = 1.0
        for l in range(1, self.max_len + 2):
            if l <= self.max_len:
                self.survival[l] = self.survival[l-1] - self.pmf[l-1]
            else:
                self.survival[l] = 0.0
        self.survival = np.maximum(self.survival, 0)

        # Compute hazard rate
        self.hazard = np.zeros(self.max_len + 1)
        for l in range(1, self.max_len + 1):
            if self.survival[l-1] > 1e-10:
                self.hazard[l] = self.pmf[l] / self.survival[l-1]

        # Compute mean and p (geometric approximation)
        self.mean_length = float(np.mean(self.lengths))
        self.p_geometric = 1.0 / self.mean_length if self.mean_length > 0 else 0.01

    def sample(self, n: int) -> np.ndarray:
        """Sample n output lengths from the distribution."""
        return np.random.choice(self.lengths, size=n, replace=True)


def is_degenerate_output(text: str, threshold: float = 0.5,
                         min_length: int = 100) -> bool:
    """
    Detect degenerate outputs with repetitive characters (repetition loops).

    A degenerate output is characterized by a single character or short pattern
    being repeated excessively. This often happens when the model gets stuck
    in a repetition loop.

    Args:
        text: The output text to check
        threshold: Ratio of most common character to total length that indicates
                   degeneracy (default 0.5 = 50% of output is one character)
        min_length: Minimum length to apply this check (short outputs are not
                    considered degenerate)

    Returns:
        True if the output appears to be degenerate, False otherwise

    Examples:
        - "ччччччччч..." (single char repeated) -> True
        - "abcabcabc..." (short pattern repeated) -> True
        - "Normal text with varied characters" -> False
    """
    if len(text) < min_length:
        return False

    # Count character frequencies
    char_counts = Counter(text)
    if not char_counts:
        return False

    # Check if any single character dominates the output
    most_common_char, most_common_count = char_counts.most_common(1)[0]
    ratio = most_common_count / len(text)

    if ratio >= threshold:
        return True

    # Also check for short repeating patterns (2-5 chars)
    for pattern_len in range(2, 6):
        if len(text) < pattern_len * 10:
            continue
        # Sample the beginning of the text to find a potential pattern
        pattern = text[:pattern_len]
        # Count how many times this pattern appears consecutively
        expected_repeats = len(text) // pattern_len
        actual_text = pattern * expected_repeats
        # If the text is mostly this pattern repeated, it's degenerate
        if text[:len(actual_text)] == actual_text:
            return True

    return False


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


def load_sharegpt_prompts(
    json_path: str = "./ShareGPT_V3_unfiltered_cleaned_split.json",
    max_samples: int = 1000,
    tokenizer=None
) -> tuple[list[str], list[int], list[int]]:
    """
    Load prompts from ShareGPT dataset.

    ShareGPT format:
    [
        {
            "id": "...",
            "conversations": [
                {"from": "human", "value": "..."},
                {"from": "gpt", "value": "..."},
                ...
            ]
        },
        ...
    ]

    For each conversation, we treat each "human" message as input
    and the following "gpt" message as output.

    Args:
        json_path: Path to the ShareGPT JSON file
        max_samples: Maximum number of samples to load
        tokenizer: Optional tokenizer for accurate token counting

    Returns: (prompts, input_lengths, output_lengths)
    """
    print(f"Loading ShareGPT from {json_path}...")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"ShareGPT file not found: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    prompts = []
    input_lengths = []
    output_lengths = []

    use_tokenizer = tokenizer is not None
    if use_tokenizer:
        print("  Using tokenizer for accurate token counting")
    else:
        print("  Using word count approximation (words * WORD_TO_TOKEN_RATIO)")

    sample_count = 0
    for item in data:
        if sample_count >= max_samples:
            break

        conversations = item.get('conversations', [])
        if not conversations:
            continue

        # Process each human-gpt pair in the conversation
        i = 0
        while i < len(conversations) and sample_count < max_samples:
            turn = conversations[i]

            # Find a human message
            if turn.get('from') == 'human':
                input_text = turn.get('value', '').strip()

                # Look for the next gpt response
                output_text = ''
                if i + 1 < len(conversations):
                    next_turn = conversations[i + 1]
                    if next_turn.get('from') == 'gpt':
                        output_text = next_turn.get('value', '').strip()
                        i += 1  # Skip the gpt turn

                # Only add if both input and output are non-empty
                if input_text and output_text:
                    prompts.append(input_text)

                    # Calculate token counts
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
        print(f"Output length range: [{min(output_lengths)}, {max(output_lengths)}]")
    else:
        print("Warning: No valid prompts found in ShareGPT dataset")

    return prompts, input_lengths, output_lengths


def load_alpaca_prompts(
    max_samples: int = 1000,
    tokenizer=None
) -> tuple[list[str], list[int], list[int]]:
    """
    Load prompts from Alpaca dataset.

    Args:
        max_samples: Maximum number of samples to load
        tokenizer: Optional tokenizer to use for accurate token counting.
                   If None, uses word count * WORD_TO_TOKEN_RATIO approximation.

    Returns: (prompts, input_lengths, output_lengths)
    """
    if not HF_AVAILABLE:
        raise ImportError("Please install datasets: pip install datasets")

    print("Loading tatsu-lab/alpaca from HuggingFace...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    prompts = []
    input_lengths = []
    output_lengths = []

    use_tokenizer = tokenizer is not None
    if use_tokenizer:
        print("  Using tokenizer for accurate token counting")
    else:
        print("  Using word count approximation (words * WORD_TO_TOKEN_RATIO)")

    for i, item in enumerate(dataset):
        if i >= max_samples:
            break

        text = item.get('text', '')
        if text:
            input_text, output_text = parse_alpaca_text(text)

            if input_text and output_text:
                prompts.append(input_text)

                # Calculate token counts
                if use_tokenizer:
                    input_len = len(tokenizer.encode(input_text))
                    output_len = len(tokenizer.encode(output_text))
                else:
                    # Approximate token count (words * WORD_TO_TOKEN_RATIO)
                    input_len = int(len(input_text.split()) * WORD_TO_TOKEN_RATIO)
                    output_len = int(len(output_text.split()) * WORD_TO_TOKEN_RATIO)

                input_lengths.append(input_len)
                output_lengths.append(output_len)

    print(f"Loaded {len(prompts)} prompts")
    print(f"Average input length: {np.mean(input_lengths):.1f} tokens")
    print(f"Average output length: {np.mean(output_lengths):.1f} tokens")

    return prompts, input_lengths, output_lengths


def load_longbench_prompts(
    max_samples: int = 1000,
    tokenizer=None,
    max_input_tokens: int = 32000
) -> tuple[list[str], list[int], list[int]]:
    """
    Load prompts from LongBench-v2 dataset.

    LongBench-v2 format:
    {
        "_id": "Unique identifier",
        "domain": "Primary domain category",
        "sub_domain": "Sub-domain category",
        "difficulty": "easy or hard",
        "length": "short, medium, or long",
        "question": "The question/query",
        "choice_A/B/C/D": "Multiple choice options",
        "answer": "Groundtruth (A/B/C/D)",
        "context": "Long context (documents, books, code, etc.)"
    }

    The input is constructed as: context + question + choices
    The output is the answer (A/B/C/D), so output length is very short.

    Args:
        max_samples: Maximum number of samples to load
        tokenizer: Optional tokenizer for accurate token counting
        max_input_tokens: Maximum input length in tokens (skip longer samples)

    Returns: (prompts, input_lengths, output_lengths)
    """
    if not HF_AVAILABLE:
        raise ImportError("Please install datasets: pip install datasets")

    print("Loading THUDM/LongBench-v2 from HuggingFace...")
    dataset = load_dataset("THUDM/LongBench-v2", split="train")

    # Sort by length category: short -> medium -> long
    # This prioritizes shorter samples which are more likely to fit in context
    length_priority = {"short": 0, "medium": 1, "long": 2}
    sorted_indices = sorted(
        range(len(dataset)),
        key=lambda i: length_priority.get(dataset[i].get('length', 'long'), 2)
    )
    print(f"  Dataset has {len(dataset)} samples, sorted by length (short first)")

    prompts = []
    input_lengths = []
    output_lengths = []

    use_tokenizer = tokenizer is not None
    if use_tokenizer:
        print("  Using tokenizer for accurate token counting")
        print(f"  Filtering samples with input > {max_input_tokens} tokens")
    else:
        print("  Using word count approximation (words * WORD_TO_TOKEN_RATIO)")

    # Pre-filter threshold: use conservative estimate (~2 chars per token)
    # to skip extremely long texts before tokenization
    max_chars_estimate = max_input_tokens * 2

    skipped_count = 0
    sample_count = 0

    for idx in sorted_indices:
        item = dataset[idx]
        if sample_count >= max_samples:
            break

        context = item.get('context', '')
        question = item.get('question', '')
        answer = item.get('answer', '')

        # Build choices string
        choices = []
        for opt in ['A', 'B', 'C', 'D']:
            choice_val = item.get(f'choice_{opt}', '')
            if choice_val:
                choices.append(f"{opt}. {choice_val}")
        choices_str = "\n".join(choices)

        # Construct the full prompt
        if context and question:
            input_text = f"{context}\n\nQuestion: {question}\n\n{choices_str}"
            # Output is just the answer letter (A/B/C/D)
            output_text = answer if answer else "A"

            # Quick pre-filter by character count to avoid tokenizer warnings
            if len(input_text) > max_chars_estimate:
                skipped_count += 1
                continue

            # Calculate token counts
            if use_tokenizer:
                input_len = len(tokenizer.encode(input_text))
                output_len = len(tokenizer.encode(output_text))

                # Skip if too long
                if input_len > max_input_tokens:
                    skipped_count += 1
                    continue
            else:
                input_len = int(len(input_text.split()) * WORD_TO_TOKEN_RATIO)
                output_len = int(len(output_text.split()) * WORD_TO_TOKEN_RATIO)

                # Skip if too long (approximate)
                if input_len > max_input_tokens:
                    skipped_count += 1
                    continue

            prompts.append(input_text)
            input_lengths.append(input_len)
            output_lengths.append(output_len)
            sample_count += 1

    print(f"Loaded {len(prompts)} prompts from LongBench-v2")
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} samples exceeding {max_input_tokens} tokens")
    if len(prompts) > 0:
        print(f"Average input length: {np.mean(input_lengths):.1f} tokens")
        print(f"Average output length: {np.mean(output_lengths):.1f} tokens")
        print(f"Input length range: [{min(input_lengths)}, {max(input_lengths)}]")
    else:
        print("Warning: No valid prompts found in LongBench-v2 dataset")

    return prompts, input_lengths, output_lengths


def load_longbench_v1_prompts(
    max_samples: int = 1000,
    tokenizer=None,
    max_input_tokens: int = 131072
) -> tuple[list[str], list[int], list[int]]:
    """
    Load prompts from LongBench v1 dataset (zai-org community version).

    Uses zai-org/LongBench which is a working copy of the original THUDM/LongBench.

    LongBench v1 has more moderate-length samples (2k-32k tokens typically)
    compared to v2 which has mostly 100k+ token samples.

    Data format:
    {
        "input": "The input/command for the task",
        "context": "The long context required for the task",
        "answers": "A List of all true answers",
        "length": "Total length (characters for Chinese, words for English)",
        "dataset": "The name of the dataset",
        "language": "The language of this piece of data",
        ...
    }

    We extract input + context as the full prompt.

    Args:
        max_samples: Maximum number of samples to load
        tokenizer: Optional tokenizer for accurate token counting
        max_input_tokens: Maximum input length in tokens (skip longer samples)

    Returns: (prompts, input_lengths, output_lengths)
    """
    if not HF_AVAILABLE:
        raise ImportError("Please install datasets: pip install datasets")

    print("Loading zai-org/LongBench from HuggingFace (v1)...")

    # All available LongBench v1 configs (English only to avoid tokenizer issues)
    configs_to_load = [
        "narrativeqa",
        "qasper",
        "multifieldqa_en",
        "hotpotqa",
        "2wikimqa",
        "musique",
        "gov_report",
        "qmsum",
        "multi_news",
        "trec",
        "triviaqa",
        "samsum",
        "passage_count",
        "passage_retrieval_en",
        "lcc",
        "repobench-p",
    ]

    prompts = []
    input_lengths = []
    output_lengths = []

    use_tokenizer = tokenizer is not None
    if use_tokenizer:
        print("  Using tokenizer for accurate token counting")
        print(f"  Filtering samples with input > {max_input_tokens} tokens")
    else:
        print("  Using word count approximation (words * WORD_TO_TOKEN_RATIO)")

    # Pre-filter threshold (characters)
    max_chars_estimate = max_input_tokens * 4

    skipped_count = 0
    sample_count = 0

    for config in configs_to_load:
        if sample_count >= max_samples:
            break

        try:
            print(f"  Loading config: {config}...")
            dataset = load_dataset("zai-org/LongBench", config, split="test")
            print(f"    Loaded {len(dataset)} samples")
        except Exception as e:
            print(f"    Warning: Could not load config '{config}': {e}")
            continue

        for item in dataset:
            if sample_count >= max_samples:
                break

            context = item.get('context', '')
            input_text = item.get('input', '')

            # Construct the full prompt: context + input
            if context or input_text:
                if context and input_text:
                    full_input = f"{context}\n\n{input_text}"
                elif context:
                    full_input = context
                else:
                    full_input = input_text

                # Quick pre-filter by character count
                if len(full_input) > max_chars_estimate:
                    skipped_count += 1
                    continue

                # Calculate token counts
                if use_tokenizer:
                    input_len = len(tokenizer.encode(full_input))

                    # Skip if too long
                    if input_len > max_input_tokens:
                        skipped_count += 1
                        continue
                else:
                    input_len = int(len(full_input.split()) * WORD_TO_TOKEN_RATIO)

                    if input_len > max_input_tokens:
                        skipped_count += 1
                        continue

                # Output length is estimated (we don't have ground truth for generation)
                # Use a reasonable estimate based on the dataset type
                output_len = 50  # Default estimate

                prompts.append(full_input)
                input_lengths.append(input_len)
                output_lengths.append(output_len)
                sample_count += 1

    print(f"Loaded {len(prompts)} prompts from LongBench v1")
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} samples exceeding {max_input_tokens} tokens")
    if len(prompts) > 0:
        print(f"Average input length: {np.mean(input_lengths):.1f} tokens")
        print(f"Input length range: [{min(input_lengths)}, {max(input_lengths)}]")
    else:
        print("Warning: No valid prompts found in LongBench v1 dataset")

    return prompts, input_lengths, output_lengths


def load_lmsys_prompts(
    max_samples: int = 1000,
    tokenizer=None
) -> tuple[list[str], list[int], list[int]]:
    """
    Load prompts from LMSYS-Chat-1M dataset.

    LMSYS-Chat-1M format:
    {
        "conversation": [
            {"content": "user message", "role": "user"},
            {"content": "assistant response", "role": "assistant"},
            ...
        ],
        ...
    }

    We use the first "user" message as input and the first "assistant"
    response as output.

    Args:
        max_samples: Maximum number of samples to load
        tokenizer: Optional tokenizer for accurate token counting

    Returns: (prompts, input_lengths, output_lengths)
    """
    if not HF_AVAILABLE:
        raise ImportError("Please install datasets: pip install datasets")

    print("Loading lmsys/lmsys-chat-1m from HuggingFace...")
    dataset = load_dataset("lmsys/lmsys-chat-1m", split="train")

    prompts = []
    input_lengths = []
    output_lengths = []

    use_tokenizer = tokenizer is not None
    if use_tokenizer:
        print("  Using tokenizer for accurate token counting")
    else:
        print("  Using word count approximation (words * WORD_TO_TOKEN_RATIO)")

    sample_count = 0
    for item in dataset:
        if sample_count >= max_samples:
            break

        conversation = item.get('conversation', [])
        if not conversation:
            continue

        # Find first user message and first assistant response
        input_text = ''
        output_text = ''

        for turn in conversation:
            role = turn.get('role', '')
            content = turn.get('content', '').strip()

            if role == 'user' and not input_text:
                input_text = content
            elif role == 'assistant' and input_text and not output_text:
                output_text = content
                break  # Got both, stop looking

        # Only add if both input and output are non-empty
        if input_text and output_text:
            prompts.append(input_text)

            # Calculate token counts
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
        print(f"Output length range: [{min(output_lengths)}, {max(output_lengths)}]")
    else:
        print("Warning: No valid prompts found in LMSYS-Chat-1M dataset")

    return prompts, input_lengths, output_lengths


def load_processbench_prompts(
    max_samples: int = 1000,
    tokenizer=None,
    split: str = "gsm8k"
) -> tuple[list[str], list[int], list[int]]:
    """
    Load prompts from Qwen/ProcessBench dataset.

    ProcessBench format:
    {
        "id": "gsm8k-0",
        "generator": "Qwen2-7B-Instruct",
        "problem": "Sue lives in a fun neighborhood...",
        "steps": [...],
        "final_answer_correct": false,
        "label": 1
    }

    We use the "problem" field as input prompt.

    Available splits: gsm8k, math, olympiadbench, omni-math

    Args:
        max_samples: Maximum number of samples to load
        tokenizer: Optional tokenizer for accurate token counting
        split: Dataset split to use (default: gsm8k)

    Returns: (prompts, input_lengths, output_lengths)
    """
    if not HF_AVAILABLE:
        raise ImportError("Please install datasets: pip install datasets")

    print(f"Loading Qwen/ProcessBench from HuggingFace (split: {split})...")
    try:
        dataset = load_dataset("Qwen/ProcessBench", split=split)
        print(f"  Dataset loaded with {len(dataset)} samples")
    except Exception as e:
        print(f"  Error loading ProcessBench: {e}")
        return [], [], []

    prompts = []
    input_lengths = []
    output_lengths = []

    use_tokenizer = tokenizer is not None
    if use_tokenizer:
        print("  Using tokenizer for accurate token counting")
    else:
        print("  Using word count approximation (words * WORD_TO_TOKEN_RATIO)")

    sample_count = 0

    for item in dataset:
        if sample_count >= max_samples:
            break

        problem = item.get('problem', '')

        if problem:
            prompts.append(problem)

            # Calculate token counts
            if use_tokenizer:
                input_len = len(tokenizer.encode(problem))
            else:
                input_len = int(len(problem.split()) * WORD_TO_TOKEN_RATIO)

            # Estimate output length (math problems typically have moderate outputs)
            output_len = 100  # Default estimate for math problem solutions

            input_lengths.append(input_len)
            output_lengths.append(output_len)
            sample_count += 1

    print(f"Loaded {len(prompts)} prompts from ProcessBench")
    if len(prompts) > 0:
        print(f"Average input length: {np.mean(input_lengths):.1f} tokens")
        print(f"Input length range: [{min(input_lengths)}, {max(input_lengths)}]")
    else:
        print("Warning: No valid prompts found in ProcessBench dataset")

    return prompts, input_lengths, output_lengths


def profile_model(llm: LLM, prompts: list[str],
                  num_profile_samples: int = 50,
                  num_p_estimation_samples: int = 100,
                  max_output_tokens: int = 256,
                  num_prefill_samples: int = 30,
                  num_decode_repeats: int = 3,
                  max_batch_size: int = 64) -> ProfilingResult:
    """
    Profile the model to measure α_p, β_p, α_d, β_d, and estimate p.

    Runs inference on varying batch sizes and input lengths to fit the linear model.
    Also runs additional samples to estimate p from actual model outputs.

    Args:
        llm: The vLLM model
        prompts: List of prompts to use for profiling
        num_profile_samples: Number of samples for timing profiling
        num_p_estimation_samples: Number of samples to estimate p
        max_output_tokens: Max tokens for p estimation runs
        num_prefill_samples: Number of samples for prefill profiling
        num_decode_repeats: Number of repetitions per batch size for decode profiling
        max_batch_size: Maximum batch size to test for decode profiling
    """
    print("\n" + "="*60)
    print("Profiling model parameters...")
    print("="*60)

    result = ProfilingResult()

    # Sample prompts of varying lengths
    sample_prompts = prompts[:num_profile_samples]

    # Profile prefill: measure time vs input length
    print(f"\nProfiling prefill times ({num_prefill_samples} samples)...")
    prefill_times = []
    input_lens = []

    for prompt in tqdm(sample_prompts[:num_prefill_samples], desc="Prefill profiling"):
        sampling_params = SamplingParams(max_tokens=1, temperature=0)

        start = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params)
        elapsed = time.perf_counter() - start

        # Get actual input length from tokenizer
        input_len = len(outputs[0].prompt_token_ids)
        prefill_times.append(elapsed)
        input_lens.append(input_len)

    # Fit linear model: T_p = α_p + β_p * L
    if len(input_lens) > 1:
        coeffs = np.polyfit(input_lens, prefill_times, 1)
        result.beta_p = coeffs[0]
        result.alpha_p = max(0, coeffs[1])

    result.prefill_times = prefill_times
    result.input_lengths = input_lens

    print(f"  Prefill: α_p = {result.alpha_p:.4f}s, β_p = {result.beta_p:.6f}s/token")

    # Profile decode: measure time vs batch size
    print(f"\nProfiling decode times (repeats={num_decode_repeats})...")
    decode_times_by_batch = {}
    batch_sizes_tested = []

    # Test more batch sizes, scaled by max_batch_size
    # Generate batch sizes: 1, 2, 4, 8, 16, 24, 32, 48, 64, ...
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

        # Run multiple times to reduce variance
        for _ in range(num_decode_repeats):
            batch_prompts = sample_prompts[:batch_size]
            sampling_params = SamplingParams(max_tokens=50, temperature=0)

            start = time.perf_counter()
            outputs = llm.generate(batch_prompts, sampling_params)
            elapsed = time.perf_counter() - start

            # Calculate average decode time per token
            total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            if total_output_tokens > 0:
                time_per_step = elapsed / (total_output_tokens / batch_size)
                times_for_this_batch.append(time_per_step)

        if times_for_this_batch:
            # Use median to be robust to outliers
            median_time = np.median(times_for_this_batch)
            decode_times_by_batch[batch_size] = median_time
            batch_sizes_tested.append(batch_size)

    # Convert to lists for fitting
    decode_times = [decode_times_by_batch[bs] for bs in batch_sizes_tested]

    # Fit linear model: t(j) = α_d + β_d * j
    if len(batch_sizes_tested) > 1:
        coeffs = np.polyfit(batch_sizes_tested, decode_times, 1)
        result.beta_d = coeffs[0]
        result.alpha_d = max(0, coeffs[1])

    result.decode_times = decode_times
    result.batch_sizes = batch_sizes_tested

    print(f"  Decode: α_d = {result.alpha_d:.4f}s, β_d = {result.beta_d:.6f}s/request")
    print(f"  Tested batch sizes: {batch_sizes_tested}")

    # ===== Estimate p from actual model outputs =====
    print("\nEstimating p from model outputs...")
    p_estimation_prompts = prompts[:num_p_estimation_samples]
    output_lengths = []
    truncated_count = 0  # Count outputs that hit max_tokens limit

    # Run model to get actual output lengths
    sampling_params = SamplingParams(
        max_tokens=max_output_tokens,
        temperature=0.7,
        top_p=0.9
    )

    # Process in batches
    batch_size = 16
    for i in tqdm(range(0, len(p_estimation_prompts), batch_size),
                  desc="Estimating p"):
        batch = p_estimation_prompts[i:i + batch_size]
        outputs = llm.generate(batch, sampling_params)

        for output in outputs:
            output_len = len(output.outputs[0].token_ids)
            output_lengths.append(output_len)
            # Check if output was truncated (hit max_tokens limit)
            finish_reason = output.outputs[0].finish_reason
            if finish_reason == "length":
                truncated_count += 1

    result.output_lengths = output_lengths

    # Estimate p = 1 / mean_output_length
    if len(output_lengths) > 0:
        mean_output_len = np.mean(output_lengths)
        result.p_estimated = 1.0 / mean_output_len if mean_output_len > 0 else 0.01

        print(f"  Output lengths: mean={mean_output_len:.1f}, "
              f"std={np.std(output_lengths):.1f}, "
              f"min={min(output_lengths)}, max={max(output_lengths)}")
        print(f"  Estimated p = 1/{mean_output_len:.1f} = {result.p_estimated:.4f}")

        # Report truncation statistics
        if truncated_count > 0:
            truncated_pct = 100.0 * truncated_count / len(output_lengths)
            print(f"  WARNING: {truncated_count}/{len(output_lengths)} "
                  f"({truncated_pct:.1f}%) outputs truncated at "
                  f"max_tokens={max_output_tokens}")
            print("  Consider increasing --max-output-tokens for accurate p estimation")
        else:
            print("  All outputs completed naturally (no truncation)")
    else:
        result.p_estimated = 0.01
        print("  Warning: No outputs collected, using default p=0.01")

    return result


def compute_tau(batch_size: int, p: float,
                alpha_d: float, beta_d: float) -> float:
    """
    Compute expected time per completion with given batch size.

    Args:
        batch_size: Number of requests in the batch
        p: Termination probability per token
        alpha_d: Fixed decode overhead per step (seconds)
        beta_d: Per-request decode cost (seconds)

    Returns:
        Expected time per completion (seconds)
    """
    if batch_size <= 0:
        return float('inf')
    numerator = alpha_d + beta_d * batch_size
    denominator = 1.0 - (1.0 - p) ** batch_size
    if denominator <= 1e-10:
        return float('inf')
    return numerator / denominator


def compute_analytical_kstar(N: int, p: float, alpha_p: float,
                             alpha_d: float, beta_d: float) -> int:
    """
    Compute optimal k* using Proposition 1.

    k* is the smallest integer k satisfying:
        k * τ(N-k) - Σ_{j=N-k+1}^{N} τ(j) >= α_p
    """
    for k in range(1, N + 1):
        lhs = k * compute_tau(N - k, p, alpha_d, beta_d)
        sum_tau = sum(compute_tau(j, p, alpha_d, beta_d)
                      for j in range(N - k + 1, N + 1))
        rhs = sum_tau + alpha_p

        if lhs >= rhs:
            return max(1, k)

    return max(1, N // 5)


def compute_optimal_batch_size_N(
    memory_config: MemoryConstraintConfig,
    p: float,
    mean_input_len: float,
    var_input_len: float = 0.0,
    theta: Optional[float] = None,
    min_batch_size: int = 1,
    max_batch_size: int = 1024
) -> dict:
    """
    Compute optimal batch size N* under memory constraints.

    Based on Section 3.3 (Memory-Constrained Batch Sizing) of the paper.

    Memory dynamics during decode:
        - Memory at decode start: X_0 = N*E[L] + N*(1-θ)²/θp * ln(1/(1-θ))
        - Peak memory supremum: κ = 1/(p²*E[L])
        - Probabilistic bound for OOM prob ≤ ε:
          N* = floor((C - κ*ln(1/ε)) / (E[L] + (1-θ)²/θp * ln(1/(1-θ))))

    Args:
        memory_config: Memory constraint configuration
        p: Termination probability per decode step
        mean_input_len: Expected input length E[L]
        var_input_len: Variance of input length Var[L] (for kappa calculation)
        theta: Normalized switching threshold k/N (if None, uses default 0.2)
        min_batch_size: Minimum allowed batch size
        max_batch_size: Maximum allowed batch size

    Returns:
        Dictionary containing:
            - N_star: Optimal batch size (probabilistic bound)
            - N_expected: Batch size using expected peak bound
            - N_static: Batch size ignoring decode dynamics
            - C: Available memory budget in tokens
            - kappa: Supremum constant
            - analysis: Detailed analysis of memory components
    """
    # Default theta if not provided (typical value from experiments)
    if theta is None:
        theta = 0.2

    # Ensure theta is in valid range (0, 1)
    theta = max(0.01, min(0.99, theta))

    # Ensure p is positive
    p = max(1e-6, p)

    # Compute available memory budget C (in tokens)
    # C = (M - m) / c
    M_bytes = memory_config.total_memory_gb * 1e9
    m_bytes = memory_config.model_memory_gb * 1e9
    c = memory_config.kv_cache_bytes_per_token
    C = (M_bytes - m_bytes) / c

    epsilon = memory_config.epsilon

    # Compute κ (kappa) - supremum constant (Eq. 3.8)
    # κ = 1 / (p² * E[L])
    # This is O(1) in N, depends only on output length statistics
    kappa = 1.0 / (p * p * mean_input_len) if mean_input_len > 0 else 0

    # Compute the memory coefficient per request (denominator term)
    # memory_per_request = E[L] + (1-θ)²/θp * ln(1/(1-θ))
    ln_term = np.log(1.0 / (1.0 - theta))
    partial_output_coef = ((1.0 - theta) ** 2) / (theta * p) * ln_term
    memory_per_request = mean_input_len + partial_output_coef

    # Compute three batch size bounds

    # 1. Static bound (ignores decode dynamics)
    # N_static = floor(C / (E[L] + (1-θ)²/θp * ln(1/(1-θ))))
    if memory_per_request > 0:
        N_static = int(C / memory_per_request)
    else:
        N_static = max_batch_size

    # 2. Expected peak bound (includes expected supremum)
    # N_expected = floor((C - κ) / memory_per_request)
    if memory_per_request > 0:
        N_expected = int((C - kappa) / memory_per_request)
    else:
        N_expected = max_batch_size

    # 3. Probabilistic bound (with OOM probability ≤ ε)
    # N* = floor((C - κ*ln(1/ε)) / memory_per_request)
    safety_margin = kappa * np.log(1.0 / epsilon) if epsilon > 0 else kappa
    if memory_per_request > 0:
        N_star = int((C - safety_margin) / memory_per_request)
    else:
        N_star = max_batch_size

    # Clamp to valid range
    N_star = max(min_batch_size, min(max_batch_size, N_star))
    N_expected = max(min_batch_size, min(max_batch_size, N_expected))
    N_static = max(min_batch_size, min(max_batch_size, N_static))

    # Compute expected memory usage for N_star
    expected_X0 = N_star * mean_input_len + N_star * partial_output_coef
    expected_Xmax = expected_X0 + kappa

    return {
        'N_star': N_star,
        'N_expected': N_expected,
        'N_static': N_static,
        'C': C,
        'kappa': kappa,
        'theta': theta,
        'memory_per_request': memory_per_request,
        'partial_output_coef': partial_output_coef,
        'safety_margin': safety_margin,
        'analysis': {
            'expected_initial_memory': expected_X0,
            'expected_peak_memory': expected_Xmax,
            'memory_utilization': expected_Xmax / C if C > 0 else 0,
            'p': p,
            'mean_input_len': mean_input_len,
            'epsilon': epsilon,
        }
    }


def analyze_kstar_sensitivity(N: int, alpha_p: float, alpha_d: float,
                               beta_d: float, p_min: float, p_max: float,
                               num_points: int = 100) -> dict:
    """
    Analyze how k* changes with p.

    Returns a dict with:
    - p_values: array of p values tested
    - k_star_values: corresponding k* values
    - transitions: list of (p_threshold, k_before, k_after) tuples
    """
    p_values = np.linspace(p_min, p_max, num_points)
    k_star_values = []

    for p in p_values:
        k_star = compute_analytical_kstar(N, p, alpha_p, alpha_d, beta_d)
        k_star_values.append(k_star)

    k_star_values = np.array(k_star_values)

    # Find transition points where k* changes
    transitions = []
    for i in range(1, len(k_star_values)):
        if k_star_values[i] != k_star_values[i-1]:
            transitions.append((
                p_values[i],
                k_star_values[i-1],
                k_star_values[i]
            ))

    return {
        'p_values': p_values,
        'k_star_values': k_star_values,
        'transitions': transitions,
        'k_star_min': int(k_star_values.min()),
        'k_star_max': int(k_star_values.max()),
    }


def run_throughput_test(llm: LLM, prompts: list[str],
                        k: int, N: int,
                        max_output_tokens: int = 4096,
                        num_requests: int = 200,
                        show_progress: bool = False,
                        return_output_lengths: bool = False,
                        collect_truncated: bool = False,
                        exclude_degenerate_from_throughput: bool = False
                        ) -> (float | tuple[float, list[int]] |
                             tuple[float, list[int], list[dict]]):
    """
    Run throughput test with a FIXED switching threshold k.

    Passes all requests to vLLM in one call. The vLLM scheduler handles
    internal batching (limited by max_num_seqs=N) and uses the P/D
    competition scheduling with the specified k as switching threshold.

    IMPORTANT: This disables dynamic k* updates in the scheduler to ensure
    k remains fixed throughout the test. This is essential for fair comparison
    of different k values.

    Args:
        llm: vLLM model
        prompts: List of prompts
        k: Switching threshold (fixed, set via environment variable)
        N: Batch size (for reference, actual batching done by vLLM)
        max_output_tokens: Max tokens per request
        num_requests: Number of requests to process
        show_progress: Whether to show progress bar
        return_output_lengths: Whether to return output lengths list
        collect_truncated: Whether to collect truncated outputs for inspection
        exclude_degenerate_from_throughput: If True, degenerate outputs (repetition
            loops) are excluded from throughput calculation. Default False means
            all outputs count toward throughput.

    Returns:
        If return_output_lengths is False and collect_truncated is False:
            throughput (requests/second)
        If return_output_lengths is True and collect_truncated is False:
            (throughput, output_lengths)
        If collect_truncated is True:
            (throughput, output_lengths, truncated_samples)
    """
    # Set k via environment variable so vLLM scheduler uses it
    update_kstar_via_env(k)

    # CRITICAL: Disable dynamic k* updates for fixed-k test
    os.environ["VLLM_PD_ENABLE_DYNAMIC_KSTAR"] = "0"

    test_prompts = prompts[:num_requests]
    sampling_params = SamplingParams(
        max_tokens=max_output_tokens,
        temperature=0.7,
        top_p=0.9
    )

    if show_progress:
        print(f"  Processing {len(test_prompts)} requests with fixed k={k}...")

    # Run all requests in one call - vLLM handles internal scheduling
    start_time = time.perf_counter()
    outputs = llm.generate(test_prompts, sampling_params)
    elapsed = time.perf_counter() - start_time

    # Process outputs
    completed = 0
    all_output_lengths = []
    truncated_samples = []
    degenerate_count = 0

    for i, output in enumerate(outputs):
        output_len = len(output.outputs[0].token_ids)
        finish_reason = output.outputs[0].finish_reason
        output_text = output.outputs[0].text

        # Check for degenerate outputs (repetition loops)
        is_degenerate = is_degenerate_output(output_text)
        if is_degenerate:
            degenerate_count += 1

        # Count toward completed based on exclude_degenerate_from_throughput
        if exclude_degenerate_from_throughput:
            if not is_degenerate:
                completed += 1
        else:
            completed += 1

        # Only include non-degenerate outputs in length statistics
        if (return_output_lengths or collect_truncated) and not is_degenerate:
            all_output_lengths.append(output_len)

        # Collect truncated samples for inspection (mark degenerate ones)
        if collect_truncated and finish_reason == "length":
            truncated_samples.append({
                'index': i,
                'prompt': test_prompts[i],
                'output': output_text,
                'num_tokens': output_len,
                'finish_reason': finish_reason,
                'is_degenerate': is_degenerate
            })

    # Report degenerate outputs if any were found
    if degenerate_count > 0:
        if exclude_degenerate_from_throughput:
            print(f"  Note: Found {degenerate_count} degenerate outputs "
                  f"(repetition loops, excluded from throughput)")
        else:
            print(f"  Note: Found {degenerate_count} degenerate outputs "
                  f"(repetition loops)")

    throughput = completed / elapsed if elapsed > 0 else 0

    if collect_truncated:
        return throughput, all_output_lengths, truncated_samples
    if return_output_lengths:
        return throughput, all_output_lengths
    return throughput


def run_throughput_test_with_online_learning(
    llm: LLM, prompts: list[str],
    N: int, alpha_p: float, alpha_d: float, beta_d: float,
    max_output_tokens: int = 4096,
    num_requests: int = 200,
    initial_p: float = DEFAULT_INITIAL_P,
    ema_alpha: float = DEFAULT_EMA_ALPHA,
    update_interval: int = DEFAULT_KSTAR_UPDATE_INTERVAL,
    show_progress: bool = True
) -> tuple[float, OnlinePEstimator]:
    """
    Run throughput test with online learning of p and dynamic k*.

    Updates p estimate as requests complete, and recalculates k* periodically.

    Args:
        llm: vLLM model
        prompts: List of prompts
        N: Batch size
        alpha_p, alpha_d, beta_d: Model timing parameters
        max_output_tokens: Max tokens per request
        num_requests: Number of requests to process
        initial_p: Initial p estimate
        ema_alpha: EMA smoothing factor for p updates
        update_interval: Recalculate k* every N completed requests
        show_progress: Whether to show progress bar

    Returns: (throughput, online_estimator)
    """
    test_prompts = prompts[:num_requests]
    sampling_params = SamplingParams(
        max_tokens=max_output_tokens,
        temperature=0.7,
        top_p=0.9
    )

    # Initialize online estimator
    estimator = OnlinePEstimator(initial_p=initial_p, ema_alpha=ema_alpha)

    # Calculate initial k*
    current_k_star = compute_analytical_kstar(N, initial_p, alpha_p, alpha_d, beta_d)
    estimator.record_kstar(current_k_star)

    start_time = time.perf_counter()
    completed = 0
    idx = 0
    batch_size = 16
    last_kstar_update = 0  # Track when k* was last updated

    # Create progress bar
    pbar = None
    if show_progress:
        bar_fmt = '{l_bar}{bar}| {n_fmt}/{total_fmt} ' \
                  '[{elapsed}<{remaining}, {rate_fmt}]'
        pbar = tqdm(total=len(test_prompts), desc="Online learning",
                    unit="req", ncols=100, bar_format=bar_fmt)

    while idx < len(test_prompts):
        # Process a batch
        current_batch_size = min(batch_size, len(test_prompts) - idx)
        batch_prompts = test_prompts[idx:idx + current_batch_size]

        outputs = llm.generate(batch_prompts, sampling_params)

        # Update p estimate with observed output lengths
        for output in outputs:
            output_len = len(output.outputs[0].token_ids)
            estimator.update(output_len)

        completed += len(outputs)
        idx += current_batch_size

        # Update progress bar with current k* info
        if pbar is not None:
            pbar.update(len(outputs))
            pbar.set_postfix({'k*': current_k_star, 'p': f'{estimator.get_p():.4f}'})

        # Periodically update k* (based on requests since last update)
        if completed - last_kstar_update >= update_interval:
            new_k_star = compute_analytical_kstar(
                N, estimator.get_p(), alpha_p, alpha_d, beta_d
            )
            estimator.record_kstar(new_k_star)
            current_k_star = new_k_star
            last_kstar_update = completed

    if pbar is not None:
        pbar.close()

    elapsed = time.perf_counter() - start_time
    throughput = completed / elapsed if elapsed > 0 else 0

    # Record final k*
    final_k_star = compute_analytical_kstar(
        N, estimator.get_p(), alpha_p, alpha_d, beta_d
    )
    estimator.record_kstar(final_k_star)

    return throughput, estimator


def update_kstar_via_env(k_star: int, p: float = None) -> None:
    """
    Update k* (and optionally p) via environment variables.

    This allows dynamic k* updates even in multi-process mode.
    The vLLM scheduler checks VLLM_PD_K_STAR_DYNAMIC on each schedule() call.
    """
    os.environ["VLLM_PD_K_STAR_DYNAMIC"] = str(k_star)
    if p is not None:
        os.environ["VLLM_PD_P_DYNAMIC"] = str(p)


def run_throughput_test_with_dynamic_kstar(
    llm: LLM, prompts: list[str],
    N: int, alpha_p: float, alpha_d: float, beta_d: float,
    max_output_tokens: int = 4096,
    num_requests: int = 200,
    initial_p: float = DEFAULT_INITIAL_P,
    ema_alpha: float = DEFAULT_EMA_ALPHA,
    update_interval: int = DEFAULT_KSTAR_UPDATE_INTERVAL,
    show_progress: bool = True
) -> tuple[float, OnlinePEstimator, list[int]]:
    """
    Run throughput test with DYNAMIC k* using vLLM's internal online learning.

    The vLLM scheduler handles online p estimation and k* updates internally:
    - Collects output lengths as requests complete
    - Updates p with EMA every `update_interval` completions
    - Recomputes k* when p changes

    This function simply passes all requests to vLLM in one call and measures
    throughput. The scheduler parameters (alpha_p, alpha_d, beta_d, p, ema_alpha,
    update_interval) should be set via environment variables before calling.

    IMPORTANT: This enables dynamic k* updates in the scheduler, unlike
    run_throughput_test which uses a fixed k.

    Args:
        llm: vLLM model
        prompts: List of prompts
        N: Maximum batch size (scheduler's batch capacity)
        alpha_p, alpha_d, beta_d: Model timing parameters (for offline k* calc)
        max_output_tokens: Max tokens per request
        num_requests: Number of requests to process
        initial_p: Initial p estimate (for offline k* calculation)
        ema_alpha: EMA smoothing factor (passed to scheduler via env var)
        update_interval: k* update interval (passed to scheduler via env var)
        show_progress: Whether to show progress bar

    Returns:
        (throughput, online_estimator, output_lengths)
    """
    # CRITICAL: Enable dynamic k* updates for online learning test
    os.environ["VLLM_PD_ENABLE_DYNAMIC_KSTAR"] = "1"

    test_prompts = prompts[:num_requests]
    sampling_params = SamplingParams(
        max_tokens=max_output_tokens,
        temperature=0.7,
        top_p=0.9
    )

    # Create estimator to track results (for compatibility with plotting)
    # Note: actual online learning happens inside the scheduler
    estimator = OnlinePEstimator(initial_p=initial_p, ema_alpha=ema_alpha)

    # Calculate initial k* (scheduler will update this online)
    initial_k_star = compute_analytical_kstar(
        N, initial_p, alpha_p, alpha_d, beta_d)
    estimator.record_kstar(initial_k_star)

    # Set initial k* for scheduler
    update_kstar_via_env(initial_k_star, initial_p)

    if show_progress:
        print(f"  Running with DYNAMIC k* learning (initial k*={initial_k_star})")
        print(f"  Scheduler will update p and k* every {update_interval} "
              "completions")

    # Run all requests in one call - vLLM handles scheduling internally
    start_time = time.perf_counter()

    if show_progress:
        print(f"  Processing {len(test_prompts)} requests...")

    outputs = llm.generate(test_prompts, sampling_params)

    elapsed = time.perf_counter() - start_time

    # Collect output lengths for analysis
    all_output_lengths = []
    for output in outputs:
        output_len = len(output.outputs[0].token_ids)
        all_output_lengths.append(output_len)
        estimator.update(output_len)

    # Calculate final statistics
    throughput = len(outputs) / elapsed if elapsed > 0 else 0
    final_p = estimator.get_p()
    final_k_star = compute_analytical_kstar(N, final_p, alpha_p, alpha_d, beta_d)
    estimator.record_kstar(final_k_star)

    if show_progress:
        print("\n  Dynamic k* test completed:")
        print(f"    Throughput: {throughput:.2f} req/s")
        print(f"    Total time: {elapsed:.2f}s for {len(outputs)} requests")
        print(f"    Final p (from outputs): {final_p:.4f}")
        print(f"    Final k* (computed): {final_k_star}")

    return throughput, estimator, all_output_lengths


def run_experiment_with_real_model(
    model_name: str = "Qwen/Qwen3-8B",
    num_requests: int = 1024,
    max_output_tokens: int = 4096,
    batch_size_N: int = 32,
    tensor_parallel_size: int = 1,
    ema_alpha: float = DEFAULT_EMA_ALPHA,
    offline_p_multiplier: float = 2.0,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    num_prefill_samples: int = DEFAULT_PREFILL_SAMPLES,
    num_decode_repeats: int = DEFAULT_DECODE_REPEATS,
    dataset: str = "alpaca",
    sharegpt_path: str = DEFAULT_SHAREGPT_PATH,
    max_input_tokens: int = 32000,
    processbench_split: str = "gsm8k",
    kstar_update_interval: int = 32,
    memory_config: Optional[MemoryConstraintConfig] = None,
    custom_prompts_path: Optional[str] = None,
    enable_thinking: bool = True,
    gpu_memory_utilization: float = 0.9,
    exclude_degenerate_from_throughput: bool = False,
    disable_multiprocessing: bool = False,
    scheduler_type: str = "pd",
):
    """
    Run the full experiment with a real model.

    Both offline and online p estimation are performed simultaneously:
    - Offline: batch estimation (offline_p_multiplier * kstar_update_interval samples)
    - Online: EMA update during dynamic k* inference

    Args:
        model_name: HuggingFace model name or path
        num_requests: Number of requests for throughput testing
        max_output_tokens: Maximum output tokens per request
        batch_size_N: Batch size N for the scheduler (overridden if memory_config set)
        tensor_parallel_size: Number of GPUs for tensor parallelism
        ema_alpha: EMA smoothing factor for online p estimation
        offline_p_multiplier: Offline samples = multiplier * kstar_update_interval
        output_dir: Directory to save results
        num_prefill_samples: Number of samples for prefill profiling
        num_decode_repeats: Number of repetitions per batch size for decode profiling
        dataset: Dataset to use ("alpaca", "sharegpt", "longbench", or "lmsys")
        sharegpt_path: Path to ShareGPT JSON file (only used if dataset="sharegpt")
        kstar_update_interval: Number of completed requests between k* updates
        max_input_tokens: Maximum input length in tokens (for LongBench filtering)
        memory_config: Memory constraint config for computing optimal N*
        custom_prompts_path: Path to JSON file with filtered prompts (overrides dataset)
        enable_thinking: Enable thinking mode for Qwen3 models (default: True)
        gpu_memory_utilization: GPU memory utilization for vLLM (default: 0.9)
        exclude_degenerate_from_throughput: Exclude degenerate outputs from throughput
        disable_multiprocessing: Disable vLLM multiprocessing for timeline recording
    """
    os.makedirs(output_dir, exist_ok=True)

    # Record experiment start time
    experiment_start_time = time.perf_counter()

    # Compute offline p estimation samples
    num_p_estimation_samples = int(offline_p_multiplier * kstar_update_interval)

    print("="*60)
    print(f"Running k* vs k_hat experiment with real model")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset}")
    print(f"Batch size N: {batch_size_N}")
    print(f"Num requests: {num_requests}")
    print(f"Enable thinking: {enable_thinking}")
    print("P estimation: BOTH offline and online")
    print(f"  Offline samples: {num_p_estimation_samples} "
          f"({offline_p_multiplier} x {kstar_update_interval})")
    print(f"  Online EMA alpha: {ema_alpha}")
    print(f"  k* update interval: {kstar_update_interval}")
    print("="*60)

    # Load tokenizer first for accurate token counting
    tokenizer = None
    if TRANSFORMERS_AVAILABLE:
        print(f"\nLoading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load prompts based on dataset choice or custom prompts file
    max_samples = num_requests + num_p_estimation_samples + 100

    # Check for custom prompts file first
    if custom_prompts_path:
        print(f"\nLoading custom prompts from {custom_prompts_path}...")
        with open(custom_prompts_path, 'r', encoding='utf-8') as f:
            custom_data = json.load(f)
        prompts = custom_data.get("prompts", [])
        # Compute input lengths
        input_lengths = []
        for p in prompts:
            if tokenizer:
                input_lengths.append(len(tokenizer.encode(p)))
            else:
                input_lengths.append(len(p) // 4)  # Rough estimate
        print(f"Loaded {len(prompts)} filtered prompts")
        if "metadata" in custom_data:
            meta = custom_data["metadata"]
            print(f"  Source dataset: {meta.get('dataset', 'unknown')}")
            print(f"  Max thinking tokens: {meta.get('max_thinking_tokens', 'unknown')}")
            print(f"  Filtered samples: {meta.get('num_samples', len(prompts))}")
    elif dataset == "sharegpt":
        prompts, input_lengths, _ = load_sharegpt_prompts(
            json_path=sharegpt_path,
            max_samples=max_samples,
            tokenizer=tokenizer
        )
    elif dataset == "longbench":
        prompts, input_lengths, _ = load_longbench_prompts(
            max_samples=max_samples,
            tokenizer=tokenizer,
            max_input_tokens=max_input_tokens
        )
    elif dataset == "longbench_v1":
        prompts, input_lengths, _ = load_longbench_v1_prompts(
            max_samples=max_samples,
            tokenizer=tokenizer,
            max_input_tokens=max_input_tokens
        )
    elif dataset == "lmsys":
        prompts, input_lengths, _ = load_lmsys_prompts(
            max_samples=max_samples,
            tokenizer=tokenizer
        )
    elif dataset == "processbench":
        prompts, input_lengths, _ = load_processbench_prompts(
            max_samples=max_samples,
            tokenizer=tokenizer,
            split=processbench_split
        )
    else:  # alpaca (default)
        prompts, input_lengths, _ = load_alpaca_prompts(
            max_samples=max_samples,
            tokenizer=tokenizer
        )

    # Apply chat template to prompts
    print(f"\nApplying chat template (enable_thinking={enable_thinking})...")
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
            # Fallback for models that don't support enable_thinking parameter
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        formatted_prompts.append(text)
    prompts = formatted_prompts

    # Recompute input lengths after applying chat template
    input_lengths = []
    for p in prompts:
        input_lengths.append(len(tokenizer.encode(p)))
    print(f"Applied chat template to {len(prompts)} prompts")

    # Plot and save input length distribution
    plot_input_length_distribution(input_lengths, output_dir, dataset)

    # Set scheduler type via environment variable
    if scheduler_type == "pd":
        os.environ["VLLM_USE_PD_SCHEDULER"] = "1"
        print("Using P/D Competition Scheduler")
    else:
        os.environ["VLLM_USE_PD_SCHEDULER"] = "0"
        print("Using original vLLM Scheduler")

    # Initialize vLLM
    print(f"\nLoading model {model_name}...")
    # Build LLM kwargs
    llm_kwargs = {
        "model": model_name,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": True,
        "max_num_seqs": batch_size_N,
        "gpu_memory_utilization": gpu_memory_utilization,
    }

    # Disable multiprocessing if requested (required for timeline recording with TP)
    if disable_multiprocessing:
        llm_kwargs["disable_custom_all_reduce"] = True
        # Use spawn method which works in single-process mode
        import os as _os
        _os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        print("Multiprocessing disabled for timeline recording")

    llm = LLM(**llm_kwargs)

    # Profile model parameters and estimate p offline
    # Use more samples for larger batch sizes
    num_profile_samples = max(100, batch_size_N * 2)
    profile_result = profile_model(
        llm, prompts,
        num_profile_samples=num_profile_samples,
        num_p_estimation_samples=num_p_estimation_samples,  # Always do offline p estimation
        max_output_tokens=max_output_tokens,
        num_prefill_samples=num_prefill_samples,
        num_decode_repeats=num_decode_repeats,
        max_batch_size=batch_size_N  # Test up to N for decode profiling
    )

    # Compute input length statistics
    mean_input_len = np.mean(input_lengths)
    var_input_len = np.var(input_lengths) if len(input_lengths) > 1 else 0.0

    # Set profiled parameters via environment variables for scheduler
    # The scheduler will read these on the next schedule() call
    params = profile_result
    os.environ["VLLM_PD_ALPHA_P_DYNAMIC"] = str(params.alpha_p)
    os.environ["VLLM_PD_ALPHA_D_DYNAMIC"] = str(params.alpha_d)
    os.environ["VLLM_PD_BETA_D_DYNAMIC"] = str(params.beta_d)
    os.environ["VLLM_PD_EMA_ALPHA"] = str(ema_alpha)
    os.environ["VLLM_PD_UPDATE_INTERVAL"] = str(kstar_update_interval)
    print("\nProfiling parameters set for scheduler:")
    print(f"  α_p={params.alpha_p:.6f}, α_d={params.alpha_d:.6f}, "
          f"β_d={params.beta_d:.6f}")

    # ===== Run BOTH offline and online p estimation simultaneously =====
    online_estimator = None
    n_star_result = None  # Will be set if memory_config is provided
    dynamic_kstar_throughput = None  # Throughput from online dynamic k* inference

    # ----- Step 1: Offline p estimation (already done in profile_model) -----
    offline_p_estimated = profile_result.p_estimated
    if profile_result.output_lengths:
        offline_mean_output_len = np.mean(profile_result.output_lengths)
    else:
        offline_mean_output_len = DEFAULT_OUTPUT_LEN_ESTIMATE

    print(f"\n*** OFFLINE p estimation: {offline_p_estimated:.4f} ***")
    print(f"  (from {num_p_estimation_samples} samples)")

    # Set initial p for scheduler (will be updated online as requests complete)
    os.environ["VLLM_PD_P_DYNAMIC"] = str(offline_p_estimated)

    # ===== Memory-constrained batch sizing (N* optimization) =====
    if memory_config is not None:
        print("\n" + "="*60)
        print("Computing optimal batch size N* (memory-constrained)")
        print("="*60)

        # Compute initial theta estimate (will be refined iteratively)
        theta_initial = 0.2

        n_star_result = compute_optimal_batch_size_N(
            memory_config=memory_config,
            p=offline_p_estimated,
            mean_input_len=mean_input_len,
            var_input_len=var_input_len,
            theta=theta_initial,
            min_batch_size=1,
            max_batch_size=1024
        )

        print(f"Memory budget C: {n_star_result['C']:.0f} tokens")
        print(f"κ (supremum constant): {n_star_result['kappa']:.2f}")
        mem_per_req = n_star_result['memory_per_request']
        print(f"Memory per request: {mem_per_req:.2f} tokens")
        print(f"Safety margin: {n_star_result['safety_margin']:.2f} tokens")
        print("\nBatch size bounds:")
        print(f"  N* (probabilistic, ε={memory_config.epsilon}): "
              f"{n_star_result['N_star']}")
        print(f"  N_expected (expected peak): {n_star_result['N_expected']}")
        print(f"  N_static (no dynamics): {n_star_result['N_static']}")
        print(f"\nUsing N* = {n_star_result['N_star']} "
              f"(was: {batch_size_N})")

        # Update batch_size_N with computed N*
        batch_size_N = n_star_result['N_star']

    # Create model params with (possibly updated) batch_size_N
    params = PDModelParams(
        N=batch_size_N,
        alpha_p=profile_result.alpha_p,
        alpha_d=profile_result.alpha_d,
        beta_d=profile_result.beta_d,
        beta_p=profile_result.beta_p,
        avg_input_len=mean_input_len,
        avg_output_len=offline_mean_output_len
    )

    # Compute analytical k* using offline p
    offline_k_star = compute_analytical_kstar(
        N=params.N,
        p=offline_p_estimated,
        alpha_p=params.alpha_p,
        alpha_d=params.alpha_d,
        beta_d=params.beta_d
    )
    print(f"Offline k* = {offline_k_star}")

    # ----- Step 2: Online p estimation with dynamic k* -----
    print(f"\n*** ONLINE p estimation with EMA (alpha={ema_alpha}) ***")
    print("*** Dynamic k* mode: k* will be used as switching threshold ***")

    # Use offline-estimated p as initial value for better convergence
    if offline_p_estimated > 0:
        initial_p = offline_p_estimated
        print(f"Using offline p={initial_p:.4f} as initial value")
    else:
        initial_p = DEFAULT_INITIAL_P
        print(f"Using default initial p={initial_p:.4f}")

    # Compute initial k* and update vLLM scheduler
    initial_k_star = compute_analytical_kstar(
        N=params.N,
        p=initial_p,
        alpha_p=params.alpha_p,
        alpha_d=params.alpha_d,
        beta_d=params.beta_d
    )
    print(f"Initial k* = {initial_k_star}")

    # Set initial k* via environment variable (works in all modes)
    update_kstar_via_env(initial_k_star, initial_p)
    print(f"Initial k*={initial_k_star} set via environment variable")

    # Run dynamic k* inference (always enabled now)
    dynamic_kstar_throughput, online_estimator, online_output_lengths = \
        run_throughput_test_with_dynamic_kstar(
            llm, prompts,
            N=batch_size_N,
            alpha_p=params.alpha_p,
            alpha_d=params.alpha_d,
            beta_d=params.beta_d,
            max_output_tokens=max_output_tokens,
            num_requests=num_requests,
            initial_p=initial_p,
            ema_alpha=ema_alpha,
            update_interval=kstar_update_interval
        )

    # Get final online estimates
    online_stats = online_estimator.get_statistics()
    online_p_estimated = online_stats['p_final']
    online_mean_output_len = online_stats['mean_length_final']
    params.avg_output_len = online_mean_output_len

    online_k_star = online_stats.get('k_star_final', 1)

    print(f"\nOnline p final: {online_p_estimated:.4f}")
    print(f"p range: [{online_stats['p_min']:.4f}, {online_stats['p_max']:.4f}]")
    print(f"Online k* final: {online_k_star}")
    if 'k_star_min' in online_stats:
        print(f"k* range: [{online_stats['k_star_min']}, "
              f"{online_stats['k_star_max']}]")
        print(f"k* mode: {online_stats['k_star_mode']}")

    # ----- Summary: Compare offline vs online -----
    print("\n" + "-"*40)
    print("Offline vs Online Comparison:")
    print(f"  Offline p: {offline_p_estimated:.4f}, Online p: {online_p_estimated:.4f}")
    print(f"  Offline k*: {offline_k_star}, Online k* (final): {online_k_star}")
    print("-"*40)

    # Use online k* as the primary k* (since it's the final converged value)
    k_star = online_k_star
    p_estimated = online_p_estimated
    mean_output_len = online_mean_output_len

    print(f"\nAnalytical k* = {k_star} (using model-estimated p={p_estimated:.4f})")

    # Test different k values to find empirical k_hat
    print("\n" + "="*60)
    print("Testing different k values...")
    print("="*60)

    # Determine search range based on N
    # Use adaptive step size: smaller steps near k*, larger steps far away
    if batch_size_N <= 32:
        # Small N: test all odd values
        k_values = list(range(1, batch_size_N + 1, 2))
    else:
        # Large N: use adaptive sampling
        # Dense sampling around expected k* region (k*/2 to 2*k*)
        k_star_region_min = max(1, k_star // 2)
        k_star_region_max = min(batch_size_N, k_star * 2)

        # Sparse sampling outside k* region (step=5)
        sparse_low = list(range(1, k_star_region_min, 5))
        sparse_high = list(range(k_star_region_max + 5, batch_size_N + 1, 5))

        # Dense sampling near k* (step=2)
        dense = list(range(k_star_region_min, k_star_region_max + 1, 2))

        k_values = sparse_low + dense + sparse_high

    # Always include k* and some boundary values
    k_values.append(k_star)
    k_values.append(1)
    k_values.append(batch_size_N)
    k_values = sorted(set(k_values))

    total_k_values = len(k_values)
    requests_per_k = min(num_requests, len(prompts))
    total_requests = total_k_values * requests_per_k
    print(f"Search range: k ∈ {{{min(k_values)}, ..., {max(k_values)}}}, "
          f"{total_k_values} values to test")
    print(f"Total requests to process: {total_requests} "
          f"({total_k_values} k values × {requests_per_k} requests each)")

    throughput_by_k = {}
    best_k = 1
    best_throughput = 0

    # Create overall progress bar
    overall_pbar = tqdm(total=total_requests, desc="Overall progress",
                        unit="req", ncols=100,
                        bar_format='{desc}: {percentage:3.0f}%|{bar}| '
                                   '{n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    for i, k in enumerate(k_values):
        # Update description to show current k value
        overall_pbar.set_description(
            f"Testing k={k} ({i+1}/{total_k_values})")

        throughput = run_throughput_test(
            llm, prompts, k, batch_size_N,
            max_output_tokens=max_output_tokens,
            num_requests=requests_per_k,
            show_progress=False,  # Don't show inner progress bar
            exclude_degenerate_from_throughput=exclude_degenerate_from_throughput
        )
        throughput_by_k[k] = throughput

        # Update overall progress
        overall_pbar.update(requests_per_k)

        # Show result for this k
        tqdm.write(f"  k={k}: throughput={throughput:.2f} req/s")

        if throughput > best_throughput:
            best_throughput = throughput
            best_k = k

    overall_pbar.close()

    k_hat = best_k
    throughput_kstar = throughput_by_k.get(k_star, 0)

    # If k_star wasn't tested, test it specifically
    if k_star not in throughput_by_k:
        print(f"Testing k*={k_star} specifically...")
        throughput_kstar = run_throughput_test(
            llm, prompts, k_star, batch_size_N,
            max_output_tokens=max_output_tokens,
            num_requests=requests_per_k,
            show_progress=True,
            exclude_degenerate_from_throughput=exclude_degenerate_from_throughput
        )
        throughput_by_k[k_star] = throughput_kstar

    # Collect output lengths and truncated samples from k_hat run
    # Also record timeline data if scheduler is accessible
    print(f"\nCollecting output lengths with k_hat={k_hat}...")

    # Try to enable timeline recording
    timeline_data = None
    try:
        scheduler = llm.llm_engine.engine_core.engine_core.scheduler
        scheduler.enable_pd_timeline(True)
        print("Timeline recording enabled")
    except AttributeError:
        print("Note: Could not enable timeline recording (multi-process mode)")

    _, collected_output_lengths, truncated_samples = run_throughput_test(
        llm, prompts, k_hat, batch_size_N,
        max_output_tokens=max_output_tokens,
        num_requests=requests_per_k,
        show_progress=True,
        return_output_lengths=True,
        collect_truncated=True,
        exclude_degenerate_from_throughput=exclude_degenerate_from_throughput
    )

    # Retrieve timeline data
    try:
        scheduler = llm.llm_engine.engine_core.engine_core.scheduler
        timeline_data = scheduler.get_pd_timeline()
        scheduler.enable_pd_timeline(False)
        print(f"Collected timeline data for {len(timeline_data.get('requests', {}))} "
              f"requests")
    except AttributeError:
        pass

    # Print final P/D scheduler stats to verify it was used
    try:
        scheduler = llm.llm_engine.engine_core.engine_core.scheduler
        pd_stats = scheduler.get_pd_stats()
        print(f"\n[P/D Scheduler Final Status] phase={pd_stats['phase']}, "
              f"k*={pd_stats['k_star']}, N={pd_stats['N']}, "
              f"prefilled={pd_stats['prefilled_count']}, "
              f"completed_decode={pd_stats['completed_decode_count']}, "
              f"decoding_reqs={pd_stats['decoding_requests']}, "
              f"waiting_reqs={pd_stats['waiting_requests']}")
    except AttributeError:
        pass

    # Save truncated samples for inspection
    if truncated_samples:
        # Count degenerate samples
        num_degenerate = sum(
            1 for s in truncated_samples if s.get('is_degenerate', False)
        )
        num_normal = len(truncated_samples) - num_degenerate

        truncated_path = os.path.join(
            output_dir, f"truncated_samples_{dataset}.json"
        )
        with open(truncated_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'model': model_name,
                    'dataset': dataset,
                    'max_output_tokens': max_output_tokens,
                    'num_truncated': len(truncated_samples),
                    'num_degenerate': num_degenerate,
                    'num_normal_truncated': num_normal,
                    'total_samples': requests_per_k,
                    'truncated_pct': 100.0 * len(truncated_samples) / requests_per_k,
                    'degenerate_pct': 100.0 * num_degenerate / requests_per_k
                },
                'samples': truncated_samples
            }, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(truncated_samples)} truncated samples to {truncated_path}")
        if num_degenerate > 0:
            print(f"  ({num_degenerate} degenerate/repetition-loop outputs, "
                  f"{num_normal} normal truncated)")

    # Save timeline data for visualization
    if timeline_data and timeline_data.get('requests'):
        timeline_path = os.path.join(
            output_dir, f"pd_timeline_{dataset}.json"
        )
        with open(timeline_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'model': model_name,
                    'dataset': dataset,
                    'batch_size_N': batch_size_N,
                    'k_hat': k_hat,
                    'k_star': k_star,
                    'max_output_tokens': max_output_tokens,
                    'num_requests': len(timeline_data.get('requests', {})),
                },
                'timeline': timeline_data
            }, f, ensure_ascii=False, indent=2)
        print(f"Saved P/D timeline data to {timeline_path}")
        print(f"  Use with: python experiments/plot_pd_timeline_v2.py "
              f"--timeline-json {timeline_path}")

    throughput_gap = (abs(best_throughput - throughput_kstar) /
                      best_throughput * 100 if best_throughput > 0 else 0)

    # Results
    results = {
        'model': model_name,
        'batch_size_N': batch_size_N,
        'num_requests': num_requests,
        'k_star': k_star,
        'k_hat': k_hat,
        'delta_k': abs(k_star - k_hat),
        'throughput_kstar': throughput_kstar,
        'throughput_khat': best_throughput,
        'throughput_gap_pct': throughput_gap,
        'profiled_params': {
            'alpha_p': params.alpha_p,
            'alpha_d': params.alpha_d,
            'beta_p': params.beta_p,
            'beta_d': params.beta_d,
        },
        'offline_estimation': {
            'p_estimated': offline_p_estimated,
            'k_star': offline_k_star,
            'mean_output_len': offline_mean_output_len,
            'num_samples': num_p_estimation_samples,
        },
        'online_estimation': {
            'p_final': online_p_estimated,
            'k_star_final': online_k_star,
            'mean_output_len': online_mean_output_len,
            'ema_alpha': ema_alpha,
            'update_interval': kstar_update_interval,
        },
        'output_dist': {
            'mean_length': mean_output_len,
            'p_estimated': p_estimated,
        },
        'throughput_by_k': {str(k): v for k, v in throughput_by_k.items()}
    }

    # Add online learning statistics if applicable
    if online_estimator is not None:
        online_stats = online_estimator.get_statistics()
        results['online_learning'] = {
            'ema_alpha': ema_alpha,
            'p_final': online_stats['p_final'],
            'p_mean': online_stats['p_mean'],
            'p_std': online_stats['p_std'],
            'p_range': [online_stats['p_min'], online_stats['p_max']],
            'mean_length_final': online_stats['mean_length_final'],
            'num_samples': online_stats['num_samples'],
        }
        if 'k_star_final' in online_stats:
            k_min = online_stats['k_star_min']
            k_max = online_stats['k_star_max']
            results['online_learning'].update({
                'k_star_final': online_stats['k_star_final'],
                'k_star_mean': online_stats['k_star_mean'],
                'k_star_std': online_stats['k_star_std'],
                'k_star_range': [k_min, k_max],
                'k_star_mode': online_stats['k_star_mode'],
            })

    # Add dynamic k* throughput if applicable
    if dynamic_kstar_throughput is not None:
        results['dynamic_kstar'] = {
            'enabled': True,
            'update_interval': kstar_update_interval,
            'throughput': dynamic_kstar_throughput,
        }
        # Calculate gap between dynamic k* and k_hat
        gap_vs_khat = (abs(best_throughput - dynamic_kstar_throughput) /
                       best_throughput * 100 if best_throughput > 0 else 0)
        results['dynamic_kstar']['gap_vs_khat_pct'] = gap_vs_khat

    # Add memory-constrained N* results if applicable
    if memory_config is not None and n_star_result is not None:
        results['memory_constraint'] = {
            'enabled': True,
            'config': {
                'total_memory_gb': memory_config.total_memory_gb,
                'model_memory_gb': memory_config.model_memory_gb,
                'kv_cache_bytes_per_token': memory_config.kv_cache_bytes_per_token,
                'epsilon': memory_config.epsilon,
            },
            'N_star': n_star_result['N_star'],
            'N_expected': n_star_result['N_expected'],
            'N_static': n_star_result['N_static'],
            'C_tokens': n_star_result['C'],
            'kappa': n_star_result['kappa'],
            'memory_per_request': n_star_result['memory_per_request'],
            'analysis': n_star_result['analysis'],
        }

    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Batch size N: {batch_size_N}")

    print("\n--- Offline Estimation ---")
    print(f"Samples: {num_p_estimation_samples}")
    print(f"p: {offline_p_estimated:.4f}")
    print(f"k*: {offline_k_star}")

    if online_estimator is not None:
        online_stats = online_estimator.get_statistics()
        print("\n--- Online Learning Statistics ---")
        print(f"Final p: {online_stats['p_final']:.4f}")
        print(f"p range: [{online_stats['p_min']:.4f}, {online_stats['p_max']:.4f}]")
        print(f"p std: {online_stats['p_std']:.4f}")
        if 'k_star_final' in online_stats:
            k_min = online_stats['k_star_min']
            k_max = online_stats['k_star_max']
            print(f"k* final: {online_stats['k_star_final']}")
            print(f"k* range: [{k_min}, {k_max}]")
            print(f"k* mode: {online_stats['k_star_mode']}")
            print(f"k* std: {online_stats['k_star_std']:.2f}")
        print("-----------------------------------")

    print(f"\nAnalytical k*: {k_star}")
    print(f"Empirical k_hat: {k_hat}")
    print(f"Δk: {abs(k_star - k_hat)}")
    print(f"Throughput at k* (fixed): {throughput_kstar:.2f} req/s")
    print(f"Throughput at k_hat: {best_throughput:.2f} req/s")
    print(f"Throughput gap (k* vs k_hat): {throughput_gap:.2f}%")

    if dynamic_kstar_throughput is not None:
        gap_vs_khat = results['dynamic_kstar']['gap_vs_khat_pct']
        print("\n--- Dynamic k* Policy ---")
        print(f"Throughput (dynamic k*): {dynamic_kstar_throughput:.2f} req/s")
        print(f"Update interval: {kstar_update_interval} requests")
        print(f"Gap vs k_hat: {gap_vs_khat:.2f}%")

    if memory_config is not None and n_star_result is not None:
        print("\n--- Memory-Constrained Batch Sizing ---")
        print(f"GPU Memory: {memory_config.total_memory_gb:.1f} GB")
        print(f"Model Memory: {memory_config.model_memory_gb:.1f} GB")
        print(f"KV Cache: {memory_config.kv_cache_bytes_per_token} bytes/token")
        print(f"OOM Tolerance (ε): {memory_config.epsilon}")
        print(f"Memory Budget (C): {n_star_result['C']:.0f} tokens")
        print(f"Optimal N* = {n_star_result['N_star']} "
              f"(static: {n_star_result['N_static']}, "
              f"expected: {n_star_result['N_expected']})")
        utilization = n_star_result['analysis']['memory_utilization'] * 100
        print(f"Expected Memory Utilization: {utilization:.1f}%")

    # Analyze k* sensitivity to p
    print("\n--- k* Sensitivity Analysis ---")
    # Use a wider p range to show transitions
    p_range_min = max(0.001, p_estimated * 0.1)  # 10x smaller
    p_range_max = min(0.5, p_estimated * 10)     # 10x larger
    sensitivity = analyze_kstar_sensitivity(
        N=batch_size_N,
        alpha_p=params.alpha_p,
        alpha_d=params.alpha_d,
        beta_d=params.beta_d,
        p_min=p_range_min,
        p_max=p_range_max,
        num_points=200
    )
    print(f"Testing p range: [{p_range_min:.4f}, {p_range_max:.4f}]")
    k_min_sens = sensitivity['k_star_min']
    k_max_sens = sensitivity['k_star_max']
    print(f"k* range in this p range: [{k_min_sens}, {k_max_sens}]")

    if sensitivity['transitions']:
        print("k* transition points:")
        for p_thresh, k_before, k_after in sensitivity['transitions']:
            print(f"  p = {p_thresh:.4f}: k* changes from {k_before} -> {k_after}")
    else:
        print("No k* transitions in this p range (k* is insensitive to p changes)")

    # Check if observed p range would cause any k* changes
    if online_estimator is not None:
        online_stats = online_estimator.get_statistics()
        observed_p_min = online_stats['p_min']
        observed_p_max = online_stats['p_max']

        would_change = False
        for p_thresh, k_before, k_after in sensitivity['transitions']:
            if observed_p_min <= p_thresh <= observed_p_max:
                would_change = True
                print(f"\nNote: Observed p range "
                      f"[{observed_p_min:.4f}, {observed_p_max:.4f}] "
                      f"crosses transition at p={p_thresh:.4f}")
                break

        if not would_change:
            print(f"\nNote: Observed p range "
                  f"[{observed_p_min:.4f}, {observed_p_max:.4f}] "
                  "does not cross any k* transition points")
            # Find nearest transition
            if sensitivity['transitions']:
                nearest_dist = float('inf')
                nearest_trans = None
                for p_thresh, k_before, k_after in sensitivity['transitions']:
                    dist_min = abs(p_thresh - observed_p_min)
                    dist_max = abs(p_thresh - observed_p_max)
                    dist = min(dist_min, dist_max)
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_trans = (p_thresh, k_before, k_after)
                if nearest_trans:
                    print(f"  Nearest transition: p={nearest_trans[0]:.4f} "
                          f"(k*: {nearest_trans[1]} -> {nearest_trans[2]})")

    print("--------------------------------")

    # Add sensitivity info to results
    results['kstar_sensitivity'] = {
        'p_range_tested': [p_range_min, p_range_max],
        'kstar_range': [sensitivity['k_star_min'], sensitivity['k_star_max']],
        'transitions': [(float(p), int(k1), int(k2))
                       for p, k1, k2 in sensitivity['transitions']]
    }

    # Save results with dataset name in filename
    json_path = os.path.join(output_dir, f"{dataset}_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {json_path}")

    # Plot throughput vs k
    # Pass online k* throughput for horizontal line, and offline k* if different
    offline_kstar_for_plot = offline_k_star if offline_k_star != k_star else None
    plot_throughput_curve(
        throughput_by_k, k_star, k_hat, output_dir,
        dataset_name=dataset,
        online_kstar_throughput=dynamic_kstar_throughput,
        offline_kstar=offline_kstar_for_plot
    )

    # Plot profiling results
    plot_profiling_results(profile_result, output_dir, dataset_name=dataset)

    # Plot online k* convergence if applicable
    if online_estimator is not None:
        plot_online_kstar_convergence(online_estimator, output_dir,
                                      dataset_name=dataset)

    # Plot output length distribution
    plot_output_length_distribution(collected_output_lengths, output_dir,
                                    dataset_name=dataset,
                                    max_output_tokens=max_output_tokens)

    # Calculate and display total experiment time
    total_time = time.perf_counter() - experiment_start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED")
    print("="*60)
    if hours > 0:
        print(f"Total time: {hours}h {minutes}m {seconds:.1f}s")
    elif minutes > 0:
        print(f"Total time: {minutes}m {seconds:.1f}s")
    else:
        print(f"Total time: {seconds:.1f}s")
    print("="*60)

    # Add total time to results
    results['total_time_seconds'] = total_time

    return results


def plot_input_length_distribution(input_lengths: list, output_dir: str,
                                    dataset_name: str = ""):
    """
    Plot and save the input length distribution histogram.

    Args:
        input_lengths: List of input token lengths
        output_dir: Directory to save the plot
        dataset_name: Name of the dataset for labeling
    """
    if not input_lengths:
        print("Warning: No input lengths to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate statistics
    mean_len = np.mean(input_lengths)
    median_len = np.median(input_lengths)
    std_len = np.std(input_lengths)
    min_len = min(input_lengths)
    max_len = max(input_lengths)

    # Create histogram
    n_bins = min(50, len(set(input_lengths)))  # Adaptive bin count
    ax.hist(input_lengths, bins=n_bins, edgecolor='black', alpha=0.7, color='steelblue')

    # Add vertical lines for mean and median
    ax.axvline(x=mean_len, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_len:.1f}')
    ax.axvline(x=median_len, color='green', linestyle='-', linewidth=2,
               label=f'Median: {median_len:.1f}')

    # Labels and title
    ax.set_xlabel('Input Length (tokens)', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    title = 'Input Length Distribution'
    if dataset_name:
        title += f' ({dataset_name})'
    ax.set_title(title, fontsize=16)

    # Add statistics text box
    stats_text = (f'N = {len(input_lengths)}\n'
                  f'Mean = {mean_len:.1f}\n'
                  f'Median = {median_len:.1f}\n'
                  f'Std = {std_len:.1f}\n'
                  f'Min = {min_len}\n'
                  f'Max = {max_len}')
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    if dataset_name:
        filename = f"{dataset_name}_input_length_distribution.pdf"
    else:
        filename = "input_length_distribution.pdf"
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved input length distribution plot to {plot_path}")


def plot_output_length_distribution(output_lengths: list, output_dir: str,
                                    dataset_name: str = "",
                                    max_output_tokens: int = None):
    """
    Plot and save the output length distribution histogram.

    Args:
        output_lengths: List of output token lengths
        output_dir: Directory to save the plot
        dataset_name: Name of the dataset for labeling
        max_output_tokens: Max output tokens setting (to show truncation line)
    """
    if not output_lengths:
        print("Warning: No output lengths to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate statistics
    mean_len = np.mean(output_lengths)
    median_len = np.median(output_lengths)
    std_len = np.std(output_lengths)
    min_len = min(output_lengths)
    max_len = max(output_lengths)

    # Count truncated outputs
    truncated_count = 0
    if max_output_tokens is not None:
        truncated_count = sum(
            1 for length in output_lengths if length >= max_output_tokens
        )
        truncated_pct = 100.0 * truncated_count / len(output_lengths)

    # Create histogram
    n_bins = min(50, len(set(output_lengths)))
    ax.hist(output_lengths, bins=n_bins, edgecolor='black', alpha=0.7,
            color='coral')

    # Add vertical lines for mean and median
    ax.axvline(x=mean_len, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_len:.1f}')
    ax.axvline(x=median_len, color='green', linestyle='-', linewidth=2,
               label=f'Median: {median_len:.1f}')

    # Add max_output_tokens line if provided
    if max_output_tokens is not None:
        ax.axvline(x=max_output_tokens, color='purple', linestyle=':',
                   linewidth=2, label=f'Max tokens: {max_output_tokens}')

    # Labels and title
    ax.set_xlabel('Output Length (tokens)', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    title = 'Output Length Distribution'
    if dataset_name:
        title += f' ({dataset_name})'
    ax.set_title(title, fontsize=16)

    # Add statistics text box
    stats_text = (f'N = {len(output_lengths)}\n'
                  f'Mean = {mean_len:.1f}\n'
                  f'Median = {median_len:.1f}\n'
                  f'Std = {std_len:.1f}\n'
                  f'Min = {min_len}\n'
                  f'Max = {max_len}')
    if max_output_tokens is not None:
        stats_text += f'\nTruncated = {truncated_count} ({truncated_pct:.1f}%)'

    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Set x-axis limit to actual max output length (not max_output_tokens)
    ax.set_xlim(0, max_len * 1.05)

    plt.tight_layout()

    # Save the plot
    if dataset_name:
        filename = f"{dataset_name}_output_length_distribution.pdf"
    else:
        filename = "output_length_distribution.pdf"
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved output length distribution plot to {plot_path}")


def plot_throughput_curve(throughput_by_k: dict, k_star: int, k_hat: int,
                          output_dir: str, dataset_name: str = "",
                          online_kstar_throughput: float | None = None,
                          offline_kstar: int | None = None):
    """
    Plot throughput vs k curve.

    Args:
        throughput_by_k: Dict mapping k values to throughput
        k_star: Final k* value (from online learning if available, else offline)
        k_hat: Empirically optimal k
        output_dir: Directory to save plot
        dataset_name: Name of dataset for title
        online_kstar_throughput: Throughput achieved with dynamic online k*
            (drawn as horizontal line since k* varies during inference)
        offline_kstar: k* computed from offline p estimation (if different from k_star)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    k_values = sorted(throughput_by_k.keys())
    throughputs = [throughput_by_k[k] for k in k_values]

    ax.plot(k_values, throughputs, 'b-o', linewidth=2, markersize=8,
            label='Fixed k throughput')

    # Mark offline k_star (if provided and different from online k_star)
    if (offline_kstar is not None and offline_kstar != k_star
            and offline_kstar in throughput_by_k):
        ax.axvline(x=offline_kstar, color='purple', linestyle=':', linewidth=2,
                   label=f'Offline $k^*$ = {offline_kstar}')
        ax.scatter([offline_kstar], [throughput_by_k[offline_kstar]],
                   color='purple', s=150, zorder=5, marker='d')

    # Mark online k_star (final value)
    if k_star in throughput_by_k:
        ax.axvline(x=k_star, color='green', linestyle='--', linewidth=2,
                   label=f'Online $k^*_{{final}}$ = {k_star}')
        ax.scatter([k_star], [throughput_by_k[k_star]], color='green',
                   s=150, zorder=5, marker='*')

    # Mark k_hat
    ax.axvline(x=k_hat, color='red', linestyle='-', linewidth=2,
               label=f'$\\hat{{k}}$ = {k_hat}')
    ax.scatter([k_hat], [throughput_by_k[k_hat]], color='red',
               s=150, zorder=5, marker='s')

    # Draw horizontal line for online dynamic k* throughput
    if online_kstar_throughput is not None:
        ax.axhline(y=online_kstar_throughput, color='orange', linestyle='-.',
                   linewidth=2.5,
                   label=f'Online dynamic $k^*$ = {online_kstar_throughput:.2f} req/s')

    ax.set_xlabel('Switching Threshold k', fontsize=14)
    ax.set_ylabel('Throughput (req/s)', fontsize=14)
    title_suffix = f" ({dataset_name})" if dataset_name else ""
    ax.set_title(f'Throughput vs Switching Threshold k{title_suffix}', fontsize=16)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if dataset_name:
        filename = f"{dataset_name}_throughput_vs_k.pdf"
    else:
        filename = "throughput_vs_k_real.pdf"
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved throughput plot to {plot_path}")


def plot_online_kstar_convergence(estimator: OnlinePEstimator, output_dir: str,
                                   dataset_name: str = ""):
    """
    Plot k* convergence over time during online learning.

    Creates two subplots:
    1. p estimate convergence over samples
    2. k* values over time with histogram
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: p convergence
    ax1 = axes[0, 0]
    samples = range(len(estimator.p_history))
    ax1.plot(samples, estimator.p_history, 'b-', linewidth=1.5, alpha=0.8)
    ax1.axhline(y=estimator.p, color='red', linestyle='--', linewidth=2,
                label=f'Final p = {estimator.p:.4f}')
    ax1.set_xlabel('Number of Samples', fontsize=12)
    ax1.set_ylabel('p Estimate', fontsize=12)
    ax1.set_title('Online p Estimation Convergence', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Mean length convergence
    ax2 = axes[0, 1]
    ax2.plot(samples, estimator.mean_length_history, 'g-', linewidth=1.5, alpha=0.8)
    ax2.axhline(y=estimator.mean_length, color='red', linestyle='--', linewidth=2,
                label=f'Final mean = {estimator.mean_length:.1f}')
    ax2.set_xlabel('Number of Samples', fontsize=12)
    ax2.set_ylabel('Mean Output Length Estimate', fontsize=12)
    ax2.set_title('Online Mean Output Length Convergence', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: k* over time
    ax3 = axes[1, 0]
    if estimator.k_star_history:
        update_points = range(len(estimator.k_star_history))
        ax3.step(update_points, estimator.k_star_history, 'b-', linewidth=2,
                 where='post', label='k* over time')
        ax3.scatter(update_points, estimator.k_star_history, color='blue',
                    s=30, zorder=5)
        final_k = estimator.k_star_history[-1]
        ax3.axhline(y=final_k, color='red', linestyle='--', linewidth=2,
                    label=f'Final k* = {final_k}')
    ax3.set_xlabel('Update Point', fontsize=12)
    ax3.set_ylabel('k*', fontsize=12)
    ax3.set_title('k* Evolution During Online Learning', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: k* histogram
    ax4 = axes[1, 1]
    if estimator.k_star_history:
        k_values = np.array(estimator.k_star_history)
        unique_k = np.unique(k_values)
        counts = [np.sum(k_values == k) for k in unique_k]
        bars = ax4.bar(unique_k, counts, color='steelblue', edgecolor='black',
                       alpha=0.7)

        # Highlight the mode
        mode_k = unique_k[np.argmax(counts)]
        mode_idx = np.where(unique_k == mode_k)[0][0]
        bars[mode_idx].set_color('red')
        bars[mode_idx].set_alpha(1.0)

        ax4.set_xlabel('k* Value', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title(f'k* Distribution (Mode = {mode_k})', fontsize=14)
        ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if dataset_name:
        filename = f"{dataset_name}_online_kstar_convergence.pdf"
    else:
        filename = "online_kstar_convergence.pdf"
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved online k* convergence plot to {plot_path}")


def plot_profiling_results(profile: ProfilingResult, output_dir: str,
                            dataset_name: str = ""):
    """Plot profiling results showing linear fits."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Prefill: time vs input length
    ax1 = axes[0]
    ax1.scatter(profile.input_lengths, profile.prefill_times,
                alpha=0.7, s=50, label='Measured')

    if len(profile.input_lengths) > 1:
        x_min = min(profile.input_lengths)
        x_max = max(profile.input_lengths)
        x_fit = np.linspace(x_min, x_max, 100)
        y_fit = profile.alpha_p + profile.beta_p * x_fit
        ax1.plot(x_fit, y_fit, 'r-', linewidth=2,
                 label=f'Fit: T = {profile.alpha_p:.3f} + {profile.beta_p:.5f}×L')

    ax1.set_xlabel('Input Length (tokens)', fontsize=12)
    ax1.set_ylabel('Prefill Time (s)', fontsize=12)
    ax1.set_title('Prefill Time vs Input Length', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Decode: time vs batch size
    ax2 = axes[1]
    ax2.scatter(profile.batch_sizes, profile.decode_times,
                alpha=0.7, s=50, label='Measured')

    if len(profile.batch_sizes) > 1:
        x_fit = np.linspace(min(profile.batch_sizes), max(profile.batch_sizes), 100)
        y_fit = profile.alpha_d + profile.beta_d * x_fit
        ax2.plot(x_fit, y_fit, 'r-', linewidth=2,
                 label=f'Fit: t = {profile.alpha_d:.4f} + {profile.beta_d:.5f}×j')

    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('Decode Time per Step (s)', fontsize=12)
    ax2.set_title('Decode Time vs Batch Size', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if dataset_name:
        filename = f"{dataset_name}_profiling_results.pdf"
    else:
        filename = "profiling_results.pdf"
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved profiling plot to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run k* vs k_hat experiment with real model"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                        help="Model name or path")
    parser.add_argument("--disable-thinking", action="store_true",
                        help="Disable thinking mode for Qwen3 (default: enabled)")
    parser.add_argument("--num-requests", type=int, default=1024,
                        help="Number of requests to test")
    parser.add_argument("--max-output-tokens", type=int, default=4096,
                        help="Maximum output tokens per request")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size N")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor parallel size")
    parser.add_argument("--scheduler", type=str, default="pd",
                        choices=["pd", "default"],
                        help="Scheduler type: 'pd' (P/D competition) or "
                             "'default' (original vLLM). Default: pd")
    parser.add_argument("--ema-alpha", type=float, default=0.2,
                        help="EMA smoothing factor for online p estimation "
                             "(0 < alpha <= 1, higher = more recent weight)")
    parser.add_argument("--offline-p-multiplier", type=float, default=1.0,
                        help="Offline p estimation samples = "
                             "multiplier * kstar-update-interval. Default: 1.0")
    parser.add_argument("--output-dir", type=str, default="./experiment_results",
                        help="Output directory")
    parser.add_argument("--num-prefill-samples", type=int, default=64,
                        help="Number of samples for prefill profiling")
    parser.add_argument("--num-decode-repeats", type=int, default=5,
                        help="Repetitions per batch size for decode profiling")
    parser.add_argument("--dataset", type=str, default="alpaca",
                        choices=["alpaca", "sharegpt", "longbench", "longbench_v1",
                                 "lmsys", "processbench"],
                        help="Dataset: 'alpaca', 'sharegpt', 'longbench' (v2), "
                             "'longbench_v1', 'lmsys', or 'processbench'")
    parser.add_argument("--processbench-split", type=str, default="gsm8k",
                        choices=["gsm8k", "math", "olympiadbench", "omni-math"],
                        help="ProcessBench split to use (default: gsm8k)")
    parser.add_argument("--sharegpt-path", type=str,
                        default="./ShareGPT_V3_unfiltered_cleaned_split.json",
                        help="Path to ShareGPT JSON file "
                             "(only used if --dataset=sharegpt)")
    parser.add_argument("--max-input-tokens", type=int, default=10240,
                        help="Max input tokens (skip longer samples in LongBench)")
    parser.add_argument("--kstar-update-interval", type=int, default=32,
                        help="Number of completed requests between k* updates "
                             "during online learning. Default: 32")
    parser.add_argument("--exclude-degenerate-throughput", action="store_true",
                        help="Exclude degenerate outputs (repetition loops) from "
                             "throughput calculation. Default: include all outputs.")

    # Memory-constrained batch sizing arguments
    parser.add_argument("--optimize-batch-size", action="store_true",
                        help="Automatically compute optimal batch size N* based on "
                             "memory constraints (overrides --batch-size)")
    parser.add_argument("--gpu-memory-gb", type=float, default=80.0,
                        help="Total GPU memory in GB (default: 80 for A100)")
    parser.add_argument("--model-memory-gb", type=float, default=None,
                        help="Model weights memory in GB (auto-detected if not set)")
    parser.add_argument("--kv-cache-bytes", type=int, default=None,
                        help="KV cache bytes per token (auto-computed if not set)")
    parser.add_argument("--oom-tolerance", type=float, default=0.05,
                        help="OOM probability tolerance epsilon (default: 0.05)")
    parser.add_argument("--custom-prompts", type=str, default=None,
                        help="Path to JSON file with filtered prompts "
                             "(from filter_samples_by_thinking.py). "
                             "Overrides --dataset when provided.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="GPU memory utilization for vLLM (default: 0.9)")
    parser.add_argument("--disable-multiprocessing", action="store_true",
                        help="Disable vLLM multiprocessing mode. Required for "
                             "timeline recording with tensor parallelism.")

    args = parser.parse_args()

    if not VLLM_AVAILABLE:
        print("Error: vLLM is required. Please install it first.")
        exit(1)

    # Build memory config if optimizing batch size
    memory_config = None
    if args.optimize_batch_size:
        model_mem = args.model_memory_gb if args.model_memory_gb else 16.0
        kv_bytes = args.kv_cache_bytes if args.kv_cache_bytes else 256
        memory_config = MemoryConstraintConfig(
            total_memory_gb=args.gpu_memory_gb,
            model_memory_gb=model_mem,
            kv_cache_bytes_per_token=kv_bytes,
            epsilon=args.oom_tolerance
        )

    run_experiment_with_real_model(
        model_name=args.model,
        num_requests=args.num_requests,
        max_output_tokens=args.max_output_tokens,
        batch_size_N=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        ema_alpha=args.ema_alpha,
        offline_p_multiplier=args.offline_p_multiplier,
        output_dir=args.output_dir,
        num_prefill_samples=args.num_prefill_samples,
        num_decode_repeats=args.num_decode_repeats,
        dataset=args.dataset,
        sharegpt_path=args.sharegpt_path,
        max_input_tokens=args.max_input_tokens,
        processbench_split=args.processbench_split,
        kstar_update_interval=args.kstar_update_interval,
        memory_config=memory_config,
        custom_prompts_path=args.custom_prompts,
        enable_thinking=not args.disable_thinking,
        gpu_memory_utilization=args.gpu_memory_utilization,
        exclude_degenerate_from_throughput=args.exclude_degenerate_throughput,
        disable_multiprocessing=args.disable_multiprocessing,
        scheduler_type=args.scheduler,
    )
