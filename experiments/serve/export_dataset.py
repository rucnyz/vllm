#!/usr/bin/env python3
"""
Export datasets to JSONL format for use with vllm bench serve.

Usage:
    python experiments/serve/export_dataset.py \
        --dataset alpaca \
        --model Qwen/Qwen3-8B \
        --num-samples 1000 \
        --output ./experiments/serve/alpaca_prompts.jsonl

Then use with vllm bench serve:
    vllm bench serve \
        --dataset-name custom \
        --dataset-path ./experiments/serve/alpaca_prompts.jsonl \
        --custom-output-len 256 \
        ...
"""

import argparse
import json
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

    # Export to JSONL (required by vllm CustomDataset)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Ensure output file ends with .jsonl
    output_path = args.output
    if not output_path.endswith('.jsonl'):
        output_path = output_path.rsplit('.', 1)[0] + '.jsonl'
        print(f"Note: Changed output extension to .jsonl (required by vllm)")

    print(f"Exporting {len(prompts)} prompts to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for prompt, input_len, output_len in zip(prompts, input_lengths, output_lengths):
            record = {"prompt": prompt, "input_len": input_len, "output_len": output_len}
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # Calculate statistics
    avg_input = sum(input_lengths) / len(input_lengths)
    avg_output = sum(output_lengths) / len(output_lengths)

    print(f"Done! Exported {len(prompts)} prompts")
    print(f"  Average input length: {avg_input:.1f} tokens")
    print(f"  Average output length: {avg_output:.1f} tokens")
    print(f"\nUsage with vllm bench serve:")
    print(f"  vllm bench serve \\")
    print(f"      --dataset-name custom \\")
    print(f"      --dataset-path {output_path} \\")
    print(f"      --custom-output-len 256 \\")
    print(f"      ...")


if __name__ == '__main__':
    main()
