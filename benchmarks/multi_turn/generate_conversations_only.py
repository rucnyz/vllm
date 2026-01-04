# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Standalone script to generate multi-turn conversation data file.
Usage:
    python generate_conversations_only.py \
        --model Qwen/Qwen3-8B \
        --input-file generate_multi_turn.json \
        -o generated_conversations.json
"""
import argparse
import json

from bench_dataset import (
    conversations_dict_to_list,
    generate_conversations,
    parse_input_json_file,
)
from bench_utils import Color, logger
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-turn conversation data file"
    )
    parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        required=True,
        help="Input JSON config file for generation of synthetic conversations",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        required=True,
        help="Output JSON file containing generated conversations",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Path or name of the model (for tokenizer)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for random number generators (default: 0)",
    )

    args = parser.parse_args()

    # Set random seed
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Read input config file
    logger.info(f"Reading input file: {args.input_file}")
    with open(args.input_file) as f:
        input_data = json.load(f)

    if not isinstance(input_data, dict) or "filetype" not in input_data:
        raise ValueError(
            f"Input file {args.input_file} must be a config file with 'filetype' field"
        )

    if input_data["filetype"] != "generate_conversations":
        raise ValueError(
            f"Expected filetype 'generate_conversations', got '{input_data['filetype']}'"
        )

    # Parse config and generate conversations
    gen_conv_args = parse_input_json_file(input_data)
    conversations = generate_conversations(gen_conv_args, tokenizer)

    # Convert to list format and save
    output_data = conversations_dict_to_list(conversations)

    logger.info(
        f"{Color.GREEN}Writing {len(output_data)} conversations to: {args.output_file}{Color.RESET}"
    )
    with open(args.output_file, "w") as f:
        json.dump(output_data, f, indent=4)

    logger.info(f"{Color.GREEN}Done!{Color.RESET}")


if __name__ == "__main__":
    main()
