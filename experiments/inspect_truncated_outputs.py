"""
Inspect truncated outputs to understand why they hit max_tokens limit.

Usage:
    python experiments/inspect_truncated_outputs.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset alpaca \
        --max-output-tokens 1024 \
        --num-samples 20
"""

import argparse

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from datasets import load_dataset
import json
import os


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
        # Load from Hugging Face
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
        # Load all splits and combine
        all_data = []
        for split in ["gsm8k", "math", "olympiadbench", "omnimath"]:
            ds = load_dataset("Qwen/ProcessBench", split=split)
            all_data.extend(ds)
        print(f"Loaded {len(all_data)} total ProcessBench problems")

        for item in all_data[:max_samples]:
            problem = item.get('problem', '').strip()
            if problem:
                # Format as a math problem prompt
                prompt = f"Please solve the following math problem step by step:\n\n{problem}"
                prompts.append(prompt)

    print(f"Loaded {len(prompts)} prompts")
    return prompts


def apply_chat_template(prompts: list, model: str, enable_thinking: bool = False):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset", type=str, default="alpaca",
                        choices=["alpaca", "sharegpt", "lmsys", "processbench"])
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--disable-thinking", action="store_true",
                        help="Disable thinking mode for Qwen3 (default: enabled)")
    args = parser.parse_args()

    # Load prompts
    raw_prompts = load_prompts(args.dataset, args.num_samples * 3)

    # Apply chat template
    raw_prompts = raw_prompts[:args.num_samples]
    prompts = apply_chat_template(raw_prompts, args.model, not args.disable_thinking)

    # Initialize model
    print(f"\nLoading model {args.model}...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )

    # Generate outputs
    sampling_params = SamplingParams(
        max_tokens=args.max_output_tokens,
        temperature=0.7,
        top_p=0.9
    )

    print(f"\nGenerating outputs with max_tokens={args.max_output_tokens}...")
    outputs = llm.generate(prompts[:args.num_samples], sampling_params)

    # Analyze outputs
    truncated = []
    completed = []

    for i, output in enumerate(outputs):
        raw_prompt = raw_prompts[i]
        generated_text = output.outputs[0].text
        num_tokens = len(output.outputs[0].token_ids)
        finish_reason = output.outputs[0].finish_reason

        result = {
            'index': i,
            'prompt': raw_prompt,
            'formatted_prompt': prompts[i],
            'output': generated_text,
            'num_tokens': num_tokens,
            'finish_reason': finish_reason
        }

        if finish_reason == "length":
            truncated.append(result)
        else:
            completed.append(result)

    # Print summary
    print("\n" + "=" * 80)
    print(f"SUMMARY: {len(truncated)} truncated, {len(completed)} completed naturally")
    print("=" * 80)

    # Show truncated outputs
    if truncated:
        print(f"\n{'='*80}")
        print("TRUNCATED OUTPUTS (hit max_tokens limit)")
        print("=" * 80)

        for item in truncated[:10]:  # Show up to 10
            print(f"\n--- Sample {item['index']} ({item['num_tokens']} tokens, {item['finish_reason']}) ---")
            print(f"PROMPT ({len(item['prompt'])} chars):")
            print("-" * 40)
            print(item['prompt'])
            print("-" * 40)
            print(f"\nFULL OUTPUT ({item['num_tokens']} tokens, {len(item['output'])} chars):")
            print("-" * 40)
            print(item['output'])
            print("-" * 40)
            print(f"\n>>> OUTPUT ENDS HERE (truncated at {args.max_output_tokens} tokens)")

    # Show a few completed outputs for comparison
    if completed:
        print(f"\n{'='*80}")
        print("NATURALLY COMPLETED OUTPUTS (for comparison)")
        print("=" * 80)

        for item in completed[:3]:  # Show up to 3
            print(f"\n--- Sample {item['index']} ({item['num_tokens']} tokens, {item['finish_reason']}) ---")
            print(f"PROMPT: {item['prompt'][:200]}...")
            print(f"OUTPUT ({item['num_tokens']} tokens): {item['output'][:300]}...")

    # Save truncated samples to JSON
    if truncated:
        save_data = []
        for item in truncated[:10]:  # Save up to 10
            save_data.append(item)

        json_path = f"experiments/truncated_samples_{args.dataset}_{args.max_output_tokens}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(save_data)} truncated samples to {json_path}")

    # Save completed samples to JSON
    if completed:
        save_data = []
        for item in completed[:10]:  # Save up to 10
            save_data.append(item)

        json_path = f"experiments/completed_samples_{args.dataset}_{args.max_output_tokens}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(save_data)} completed samples to {json_path}")

    # Statistics
    if truncated:
        avg_truncated = sum(t['num_tokens'] for t in truncated) / len(truncated)
        print(f"\n{'='*80}")
        print("STATISTICS")
        print("=" * 80)
        print(f"Truncated outputs: {len(truncated)}/{len(outputs)} ({100*len(truncated)/len(outputs):.1f}%)")
        print(f"Average truncated length: {avg_truncated:.0f} tokens (= max_tokens)")
        if completed:
            avg_completed = sum(c['num_tokens'] for c in completed) / len(completed)
            print(f"Average completed length: {avg_completed:.0f} tokens")
            print(f"\nThis suggests natural output length may be around {avg_completed:.0f} tokens")
            if avg_truncated > avg_completed * 0.8:
                print("WARNING: Many outputs want to be longer than max_tokens allows!")


if __name__ == "__main__":
    main()
