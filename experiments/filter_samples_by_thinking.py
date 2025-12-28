"""
Filter dataset samples by thinking token length.

This script pre-processes datasets to filter out samples where the model's
thinking phase exceeds a specified token threshold. This ensures that
downstream experiments (e.g., kstar_vs_khat_real_model.py) work with clean
samples that can complete within reasonable token limits.

Usage:
    python experiments/filter_samples_by_thinking.py \
        --model Qwen/Qwen3-8B \
        --dataset alpaca \
        --max-thinking-tokens 4096 \
        --max-total-tokens 8192 \
        --num-samples 100 \
        --output-dir experiments/filtered_datasets

The script will:
1. Load prompts from the specified dataset
2. Apply chat template with thinking enabled
3. Generate outputs and measure thinking token length
4. Filter out samples where thinking exceeds the threshold
5. Save filtered samples to JSON for downstream use
"""

import argparse
import json
import os
import re
from tqdm import tqdm

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from datasets import load_dataset


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
        print("Loading ShareGPT dataset from Hugging Face...")
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

    elif dataset == "lmsys":
        print("Loading LMSYS dataset...")
        ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
        for i, item in enumerate(ds):
            if len(prompts) >= max_samples:
                break
            conv = item.get('conversation', [])
            if conv and conv[0].get('role') == 'user':
                prompts.append(conv[0].get('content', '').strip())

    elif dataset == "processbench":
        print("Loading ProcessBench dataset...")
        all_data = []
        for split in ["gsm8k", "math", "olympiadbench", "omnimath"]:
            ds = load_dataset("Qwen/ProcessBench", split=split)
            all_data.extend(ds)
        print(f"Loaded {len(all_data)} total ProcessBench problems")

        for item in all_data[:max_samples]:
            problem = item.get('problem', '').strip()
            if problem:
                prompt = f"Please solve the following math problem step by step:\n\n{problem}"
                prompts.append(prompt)

    print(f"Loaded {len(prompts)} prompts")
    return prompts


def apply_chat_template(prompts: list, model: str, enable_thinking: bool = True):
    """Apply chat template to prompts."""
    print(f"Applying chat template (enable_thinking={enable_thinking})...")
    tokenizer = AutoTokenizer.from_pretrained(model)

    formatted_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        formatted_prompts.append(text)

    return formatted_prompts


def extract_thinking_tokens(output_text: str, tokenizer) -> int:
    """
    Extract and count thinking tokens from model output.

    Qwen3 thinking format:
    <think>
    ... thinking content ...
    </think>
    actual response
    """
    # Match thinking block
    think_pattern = r'<think>(.*?)</think>'
    match = re.search(think_pattern, output_text, re.DOTALL)

    if match:
        thinking_content = match.group(1)
        # Count tokens in thinking block
        thinking_tokens = len(tokenizer.encode(thinking_content))
        return thinking_tokens

    # If no </think> tag found, the entire output might be thinking
    # (truncated before completing thinking phase)
    if '<think>' in output_text and '</think>' not in output_text:
        # Extract content after <think>
        think_start = output_text.find('<think>') + len('<think>')
        thinking_content = output_text[think_start:]
        thinking_tokens = len(tokenizer.encode(thinking_content))
        return thinking_tokens

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Filter dataset samples by thinking token length"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset", type=str, default="alpaca",
                        choices=["alpaca", "sharegpt", "lmsys", "processbench"])
    parser.add_argument("--max-thinking-tokens", type=int, default=4096,
                        help="Maximum allowed thinking tokens")
    parser.add_argument("--max-total-tokens", type=int, default=8192,
                        help="Maximum total output tokens for generation")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples to process")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--output-dir", type=str,
                        default="experiments/filtered_datasets")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for generation")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load prompts
    raw_prompts = load_prompts(args.dataset, args.num_samples * 2)
    raw_prompts = raw_prompts[:args.num_samples]

    # Apply chat template with thinking enabled
    formatted_prompts = apply_chat_template(raw_prompts, args.model, enable_thinking=True)

    # Initialize model
    print(f"\nLoading model {args.model}...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )

    # Load tokenizer for counting
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Generate outputs with high max_tokens to capture full thinking
    sampling_params = SamplingParams(
        max_tokens=args.max_total_tokens,
        temperature=0.7,
        top_p=0.9
    )

    print(f"\nGenerating outputs with max_tokens={args.max_total_tokens}...")
    print(f"Processing {len(formatted_prompts)} samples...")

    # Process in batches
    all_outputs = []
    for i in tqdm(range(0, len(formatted_prompts), args.batch_size), desc="Generating"):
        batch = formatted_prompts[i:i + args.batch_size]
        outputs = llm.generate(batch, sampling_params)
        all_outputs.extend(outputs)

    # Analyze and filter
    filtered_samples = []
    rejected_samples = []
    stats = {
        "total": len(all_outputs),
        "kept": 0,
        "rejected_thinking_too_long": 0,
        "rejected_truncated_in_thinking": 0,
        "thinking_tokens_distribution": []
    }

    print("\nAnalyzing outputs...")
    for i, output in enumerate(tqdm(all_outputs, desc="Filtering")):
        raw_prompt = raw_prompts[i]
        formatted_prompt = formatted_prompts[i]
        generated_text = output.outputs[0].text
        total_tokens = len(output.outputs[0].token_ids)
        finish_reason = output.outputs[0].finish_reason

        # Extract thinking tokens
        thinking_tokens = extract_thinking_tokens(generated_text, tokenizer)
        stats["thinking_tokens_distribution"].append(thinking_tokens)

        # Check if thinking completed
        thinking_completed = '</think>' in generated_text

        # Check if truncated during thinking phase
        truncated_in_thinking = (
            finish_reason == "length" and
            '<think>' in generated_text and
            '</think>' not in generated_text
        )

        sample_info = {
            "index": i,
            "prompt": raw_prompt,
            "formatted_prompt": formatted_prompt,
            "output": generated_text,
            "total_tokens": total_tokens,
            "thinking_tokens": thinking_tokens,
            "thinking_completed": thinking_completed,
            "finish_reason": finish_reason
        }

        # Filter logic
        if truncated_in_thinking:
            # Rejected: thinking was truncated (would exceed max_total_tokens)
            stats["rejected_truncated_in_thinking"] += 1
            sample_info["rejection_reason"] = "truncated_in_thinking"
            rejected_samples.append(sample_info)
        elif thinking_tokens > args.max_thinking_tokens:
            # Rejected: thinking too long
            stats["rejected_thinking_too_long"] += 1
            sample_info["rejection_reason"] = "thinking_too_long"
            rejected_samples.append(sample_info)
        else:
            # Accepted
            stats["kept"] += 1
            filtered_samples.append(sample_info)

    # Calculate statistics
    thinking_tokens_list = stats["thinking_tokens_distribution"]
    if thinking_tokens_list:
        stats["thinking_tokens_mean"] = sum(thinking_tokens_list) / len(thinking_tokens_list)
        stats["thinking_tokens_max"] = max(thinking_tokens_list)
        stats["thinking_tokens_min"] = min(thinking_tokens_list)
        sorted_tokens = sorted(thinking_tokens_list)
        stats["thinking_tokens_median"] = sorted_tokens[len(sorted_tokens) // 2]
        stats["thinking_tokens_p90"] = sorted_tokens[int(len(sorted_tokens) * 0.9)]
        stats["thinking_tokens_p95"] = sorted_tokens[int(len(sorted_tokens) * 0.95)]

    # Print summary
    print("\n" + "=" * 80)
    print("FILTERING SUMMARY")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Max thinking tokens threshold: {args.max_thinking_tokens}")
    print(f"Max total tokens for generation: {args.max_total_tokens}")
    print("-" * 40)
    print(f"Total samples processed: {stats['total']}")
    print(f"Samples kept: {stats['kept']} ({100*stats['kept']/stats['total']:.1f}%)")
    print(f"Rejected (thinking too long): {stats['rejected_thinking_too_long']}")
    print(f"Rejected (truncated in thinking): {stats['rejected_truncated_in_thinking']}")
    print("-" * 40)
    print("Thinking tokens statistics:")
    print(f"  Mean: {stats.get('thinking_tokens_mean', 0):.0f}")
    print(f"  Median: {stats.get('thinking_tokens_median', 0):.0f}")
    print(f"  Min: {stats.get('thinking_tokens_min', 0):.0f}")
    print(f"  Max: {stats.get('thinking_tokens_max', 0):.0f}")
    print(f"  P90: {stats.get('thinking_tokens_p90', 0):.0f}")
    print(f"  P95: {stats.get('thinking_tokens_p95', 0):.0f}")

    # Save filtered samples
    filtered_path = os.path.join(
        args.output_dir,
        f"filtered_{args.dataset}_think{args.max_thinking_tokens}.json"
    )
    with open(filtered_path, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "dataset": args.dataset,
                "model": args.model,
                "max_thinking_tokens": args.max_thinking_tokens,
                "max_total_tokens": args.max_total_tokens,
                "stats": {k: v for k, v in stats.items()
                         if k != "thinking_tokens_distribution"}
            },
            "samples": filtered_samples
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(filtered_samples)} filtered samples to {filtered_path}")

    # Save rejected samples for inspection
    rejected_path = os.path.join(
        args.output_dir,
        f"rejected_{args.dataset}_think{args.max_thinking_tokens}.json"
    )
    with open(rejected_path, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "dataset": args.dataset,
                "model": args.model,
                "max_thinking_tokens": args.max_thinking_tokens,
                "rejection_counts": {
                    "thinking_too_long": stats["rejected_thinking_too_long"],
                    "truncated_in_thinking": stats["rejected_truncated_in_thinking"]
                }
            },
            "samples": rejected_samples
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(rejected_samples)} rejected samples to {rejected_path}")

    # Save just the prompts for easy loading
    prompts_only_path = os.path.join(
        args.output_dir,
        f"prompts_{args.dataset}_think{args.max_thinking_tokens}.json"
    )
    with open(prompts_only_path, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "dataset": args.dataset,
                "model": args.model,
                "max_thinking_tokens": args.max_thinking_tokens,
                "num_samples": len(filtered_samples)
            },
            "prompts": [s["prompt"] for s in filtered_samples]
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved prompts only to {prompts_only_path}")

    print("\n" + "=" * 80)
    print("Done! Use the filtered prompts file with kstar_vs_khat_real_model.py:")
    print(f"  --custom-prompts {prompts_only_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
