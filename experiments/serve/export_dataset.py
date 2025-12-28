#!/usr/bin/env python3
"""
Export datasets to CSV format for use with vllm bench serve.

Usage:
    python experiments/serve/export_dataset.py \
        --dataset alpaca \
        --model Qwen/Qwen3-8B \
        --num-samples 1000 \
        --output ./experiments/serve/alpaca_prompts.csv

Then use with vllm bench serve:
    vllm bench serve \
        --dataset-path ./experiments/serve/alpaca_prompts.csv \
        --dataset-prompt-column "prompt" \
        ...
"""

import argparse
import csv
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer

from pd_experiment_utils import (
    load_alpaca_prompts,
    load_sharegpt_prompts,
    load_lmsys_prompts,
    apply_chat_template,
    DEFAULT_SHAREGPT_PATH,
)


def main():
    parser = argparse.ArgumentParser(
        description='Export dataset to CSV for vllm bench serve')

    parser.add_argument('--dataset', type=str, default='alpaca',
                        choices=['alpaca', 'sharegpt', 'lmsys'],
                        help='Dataset to export')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-8B',
                        help='Model for tokenizer (used for chat template)')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of samples to export')
    parser.add_argument('--sharegpt-path', type=str, default=DEFAULT_SHAREGPT_PATH,
                        help='Path to ShareGPT JSON file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV file path')
    parser.add_argument('--apply-chat-template', action='store_true',
                        help='Apply chat template to prompts')
    parser.add_argument('--disable-thinking', action='store_true',
                        help='Disable thinking mode for Qwen3 models')

    args = parser.parse_args()

    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == 'alpaca':
        prompts, input_lengths, output_lengths = load_alpaca_prompts(
            max_samples=args.num_samples,
            tokenizer=tokenizer
        )
    elif args.dataset == 'sharegpt':
        prompts, input_lengths, output_lengths = load_sharegpt_prompts(
            json_path=args.sharegpt_path,
            max_samples=args.num_samples,
            tokenizer=tokenizer
        )
    elif args.dataset == 'lmsys':
        prompts, input_lengths, output_lengths = load_lmsys_prompts(
            max_samples=args.num_samples,
            tokenizer=tokenizer
        )

    # Optionally apply chat template
    if args.apply_chat_template:
        print("Applying chat template...")
        prompts = apply_chat_template(
            prompts, tokenizer,
            enable_thinking=not args.disable_thinking
        )

    # Export to CSV
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    print(f"Exporting {len(prompts)} prompts to {args.output}...")
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['prompt', 'input_len', 'output_len'])
        for prompt, input_len, output_len in zip(prompts, input_lengths, output_lengths):
            writer.writerow([prompt, input_len, output_len])

    print(f"Done! Exported {len(prompts)} prompts")
    print(f"\nUsage with vllm bench serve:")
    print(f"  vllm bench serve \\")
    print(f"      --dataset-path {args.output} \\")
    print(f"      --dataset-prompt-column prompt \\")
    print(f"      ...")


if __name__ == '__main__':
    main()
