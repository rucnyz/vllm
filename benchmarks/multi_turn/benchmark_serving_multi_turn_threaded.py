# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Multi-threaded version of benchmark_serving_multi_turn.py
Uses threading instead of multiprocessing for lower overhead.
"""
import argparse
import asyncio
import json
import random
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from statistics import mean
from typing import NamedTuple

import aiohttp
import numpy as np
import pandas as pd
from bench_dataset import (
    ConversationsMap,
    ConvId,
    GenConvArgs,
    MessagesList,
    conversations_dict_to_list,
    conversations_list_to_dict,
    generate_conversations,
    parse_input_json_file,
)
from bench_utils import TEXT_SEPARATOR, Color, logger
from transformers import AutoTokenizer

NUM_TOKENS_FROM_DATASET = 0
TERM_SIGNAL = None


class RequestStats(NamedTuple):
    ttft_ms: float
    tpot_ms: float
    latency_ms: float
    start_time_ms: float
    input_num_turns: int
    input_num_tokens: int
    output_num_tokens: int
    output_num_chunks: int
    output_num_first_chunk_tokens: int
    approx_cached_percent: float
    conversation_id: str
    client_id: int


def nanosec_to_millisec(value: float) -> float:
    return value / 1000000.0


def nanosec_to_sec(value: float) -> float:
    return value / 1000000000.0


def get_token_count(tokenizer: AutoTokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def get_messages_token_count(
    tokenizer: AutoTokenizer, messages: list[dict[str, str]]
) -> int:
    token_count = 0
    for m in messages:
        token_count += get_token_count(tokenizer, m["content"])
    return token_count


async def send_request_async(
    session: aiohttp.ClientSession,
    messages: list[dict[str, str]],
    chat_url: str,
    model: str,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
    timeout_sec: int = 120,
) -> tuple[bool, float, float, float, str, int]:
    """Send a single request and return metrics."""
    payload = {
        "model": model,
        "messages": messages,
        "seed": 0,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": False},
    }

    if min_tokens is not None:
        payload["min_tokens"] = min_tokens
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    headers = {"Content-Type": "application/json"}
    timeout = aiohttp.ClientTimeout(total=timeout_sec)

    ttft: float | None = None
    chunk_delays: list[int] = []
    latency: float | None = None
    first_chunk = ""
    generated_text = ""
    valid = True

    start_time = time.perf_counter_ns()
    most_recent_timestamp = start_time

    try:
        async with session.post(
            url=chat_url, json=payload, headers=headers, timeout=timeout
        ) as response:
            if response.status == 200:
                async for chunk_bytes in response.content:
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue

                    chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                    if chunk == "[DONE]":
                        latency = time.perf_counter_ns() - start_time
                    else:
                        timestamp = time.perf_counter_ns()
                        try:
                            data = json.loads(chunk)
                            delta = data["choices"][0]["delta"]
                            if delta.get("content"):
                                if ttft is None:
                                    ttft = time.perf_counter_ns() - start_time
                                    first_chunk = delta["content"]
                                else:
                                    chunk_delays.append(timestamp - most_recent_timestamp)
                                generated_text += delta["content"]
                            most_recent_timestamp = timestamp
                        except (json.JSONDecodeError, KeyError):
                            pass
            else:
                valid = False
                logger.warning(f"Request failed with status {response.status}")
    except asyncio.TimeoutError:
        valid = False
        logger.warning("Request timed out")
    except Exception as e:
        valid = False
        logger.warning(f"Request failed: {e}")

    if latency is None:
        latency = time.perf_counter_ns() - start_time
    if ttft is None:
        ttft = latency

    tpot = mean(chunk_delays) if chunk_delays else 0.0
    num_chunks = len(chunk_delays)

    return (
        valid,
        nanosec_to_millisec(ttft),
        nanosec_to_millisec(tpot),
        nanosec_to_millisec(latency),
        generated_text,
        num_chunks,
    )


def client_worker(
    client_id: int,
    tokenizer: AutoTokenizer,
    chat_url: str,
    model: str,
    task_queue: Queue,
    result_queue: Queue,
    stop_event: threading.Event,
    max_num_requests: int | None,
    max_turns: int | None,
    limit_min_tokens: int,
    limit_max_tokens: int,
    timeout_sec: int,
    verbose: bool,
):
    """Worker function that runs in a thread."""
    logger.info(f"{Color.CYAN}Started client {client_id}{Color.RESET}")

    # Set per-thread random seed
    thread_seed = int(time.time() * 1000) + client_id
    random.seed(thread_seed)
    np.random.seed(thread_seed % (2**32))

    turns_count: Counter = Counter()
    num_successes = 0
    num_failures = 0

    async def run_client():
        nonlocal num_successes, num_failures

        async with aiohttp.ClientSession() as session:
            while not stop_event.is_set():
                if max_num_requests and (num_successes + num_failures) >= max_num_requests:
                    break

                try:
                    conv_id, messages = task_queue.get(timeout=1)
                except Empty:
                    continue

                if conv_id is TERM_SIGNAL:
                    task_queue.put((TERM_SIGNAL, TERM_SIGNAL))  # Re-add for other workers
                    break

                # Process conversation turns
                turns_count[conv_id] += 1
                current_turn = turns_count[conv_id]

                if current_turn > len(messages):
                    continue

                # Prepare messages for this turn
                turn_messages = messages[:current_turn]
                if turn_messages[-1]["role"] != "user":
                    continue

                # Determine token limits
                min_tokens = None if limit_min_tokens < 0 else limit_min_tokens
                max_tokens = None if limit_max_tokens < 0 else limit_max_tokens

                if len(messages) > current_turn and (
                    min_tokens == NUM_TOKENS_FROM_DATASET or max_tokens == NUM_TOKENS_FROM_DATASET
                ):
                    answer_tokens = get_token_count(tokenizer, messages[current_turn]["content"])
                    if min_tokens == NUM_TOKENS_FROM_DATASET:
                        min_tokens = max(1, answer_tokens)
                    if max_tokens == NUM_TOKENS_FROM_DATASET:
                        max_tokens = max(1, answer_tokens)

                # Send request
                valid, ttft_ms, tpot_ms, latency_ms, content, num_chunks = await send_request_async(
                    session, turn_messages, chat_url, model, min_tokens, max_tokens, timeout_sec
                )

                if valid:
                    num_successes += 1

                    # Calculate token counts
                    input_num_tokens = get_messages_token_count(tokenizer, turn_messages)
                    output_num_tokens = get_token_count(tokenizer, content) if content else 0
                    first_chunk_tokens = get_token_count(tokenizer, content[:50]) if content else 0

                    # Compute metrics
                    question_tokens = get_token_count(tokenizer, turn_messages[-1]["content"])
                    history_tokens = input_num_tokens - question_tokens
                    approx_cached = 100.0 * history_tokens / input_num_tokens if input_num_tokens > 0 else 0.0

                    # Adjust TPOT based on actual tokens
                    if output_num_tokens > 1 and output_num_tokens > first_chunk_tokens:
                        decode_ms = latency_ms - ttft_ms
                        decode_tokens = output_num_tokens - first_chunk_tokens
                        tpot_ms = decode_ms / decode_tokens if decode_tokens > 0 else 0.0

                    stats = RequestStats(
                        ttft_ms=ttft_ms,
                        tpot_ms=tpot_ms,
                        latency_ms=latency_ms,
                        start_time_ms=nanosec_to_millisec(time.perf_counter_ns()),
                        input_num_turns=len(turn_messages),
                        input_num_tokens=input_num_tokens,
                        output_num_tokens=output_num_tokens,
                        output_num_chunks=num_chunks,
                        output_num_first_chunk_tokens=first_chunk_tokens,
                        approx_cached_percent=approx_cached,
                        conversation_id=conv_id,
                        client_id=client_id,
                    )
                    result_queue.put(stats)

                    # Update conversation with response
                    turns_count[conv_id] += 1
                    if current_turn < len(messages):
                        messages[current_turn]["content"] = content

                    # Check if more turns available
                    max_allowed = len(messages) if max_turns is None else min(max_turns, len(messages))
                    if turns_count[conv_id] < max_allowed:
                        task_queue.put((conv_id, messages))

                    if verbose:
                        logger.info(
                            f"Client {client_id}: conv={conv_id}, turn={current_turn}, "
                            f"ttft={ttft_ms:.1f}ms, latency={latency_ms:.1f}ms"
                        )
                else:
                    num_failures += 1

        result_queue.put(TERM_SIGNAL)

    asyncio.run(run_client())
    logger.info(
        f"{Color.CYAN}Client {client_id} done: {num_successes} successes, {num_failures} failures{Color.RESET}"
    )


def run_benchmark(
    conversations: ConversationsMap,
    tokenizer: AutoTokenizer,
    url: str,
    model: str,
    num_clients: int,
    max_num_requests: int | None,
    max_turns: int | None,
    limit_min_tokens: int,
    limit_max_tokens: int,
    timeout_sec: int,
    verbose: bool,
) -> list[RequestStats]:
    """Run the multi-threaded benchmark."""
    chat_url = f"{url}/v1/chat/completions"

    task_queue: Queue = Queue()
    result_queue: Queue = Queue()
    stop_event = threading.Event()

    # Add all conversations to the task queue
    for conv_id, messages in conversations.items():
        task_queue.put((conv_id, messages))

    # Add termination signal
    task_queue.put((TERM_SIGNAL, TERM_SIGNAL))

    # Start worker threads
    start_time = time.perf_counter_ns()
    logger.info(f"{Color.GREEN}Starting {num_clients} client threads{Color.RESET}")

    threads = []
    for client_id in range(num_clients):
        t = threading.Thread(
            target=client_worker,
            args=(
                client_id,
                tokenizer,
                chat_url,
                model,
                task_queue,
                result_queue,
                stop_event,
                max_num_requests // num_clients if max_num_requests else None,
                max_turns,
                limit_min_tokens,
                limit_max_tokens,
                timeout_sec,
                verbose,
            ),
            daemon=True,
        )
        threads.append(t)
        t.start()

    # Collect results
    all_results: list[RequestStats] = []
    clients_done = 0

    while clients_done < num_clients:
        result = result_queue.get()
        if result is TERM_SIGNAL:
            clients_done += 1
            logger.info(f"{Color.CYAN}{clients_done}/{num_clients} clients done{Color.RESET}")
        else:
            all_results.append(result)
            if len(all_results) % 100 == 0:
                runtime = nanosec_to_sec(time.perf_counter_ns() - start_time)
                rps = len(all_results) / runtime
                logger.info(
                    f"{Color.CYAN}Collected {len(all_results)} results, "
                    f"{rps:.2f} req/s{Color.RESET}"
                )

    # Wait for all threads
    for t in threads:
        t.join(timeout=5)

    runtime = nanosec_to_sec(time.perf_counter_ns() - start_time)
    logger.info(
        f"{Color.GREEN}Benchmark done: {len(all_results)} requests in {runtime:.2f}s "
        f"({len(all_results)/runtime:.2f} req/s){Color.RESET}"
    )

    return all_results


def print_statistics(results: list[RequestStats]) -> None:
    """Print benchmark statistics."""
    if not results:
        logger.info("No results to display")
        return

    df = pd.DataFrame(results)

    percentiles = [0.25, 0.5, 0.75, 0.9, 0.99]
    exclude = ["start_time_ms", "conversation_id", "client_id", "output_num_first_chunk_tokens", "approx_cached_percent"]

    print(TEXT_SEPARATOR)
    print(f"{Color.YELLOW}Statistics Summary:{Color.RESET}")
    print(TEXT_SEPARATOR)

    stats_df = df.drop(columns=exclude, errors="ignore").describe(percentiles=percentiles).transpose()
    pd.set_option("display.precision", 2)
    print(stats_df)
    print(TEXT_SEPARATOR)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-threaded benchmark for multi-turn conversations"
    )
    parser.add_argument("-i", "--input-file", type=str, required=True, help="Input JSON file")
    parser.add_argument("-o", "--output-file", type=str, default=None, help="Output JSON file")
    parser.add_argument("-m", "--model", type=str, required=True, help="Model path for tokenizer")
    parser.add_argument("--served-model-name", type=str, default=None, help="Model name in API")
    parser.add_argument("-u", "--url", type=str, default="http://localhost:8000", help="Server URL")
    parser.add_argument("-p", "--num-clients", type=int, default=1, help="Number of client threads")
    parser.add_argument("-n", "--max-num-requests", type=int, default=None, help="Max requests")
    parser.add_argument("--max-turns", type=int, default=None, help="Max turns per conversation")
    parser.add_argument("--limit-min-tokens", type=int, default=0, help="Min output tokens")
    parser.add_argument("--limit-max-tokens", type=int, default=0, help="Max output tokens")
    parser.add_argument("--request-timeout-sec", type=int, default=120, help="Request timeout")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    logger.info(f"Reading input file: {args.input_file}")
    with open(args.input_file) as f:
        input_data = json.load(f)

    if isinstance(input_data, list):
        conversations = conversations_list_to_dict(input_data)
    elif isinstance(input_data, dict) and input_data.get("filetype") == "generate_conversations":
        gen_args = parse_input_json_file(input_data)
        conversations = generate_conversations(gen_args, tokenizer)
    else:
        raise ValueError("Invalid input file format")

    model_name = args.served_model_name or args.model

    results = run_benchmark(
        conversations=conversations,
        tokenizer=tokenizer,
        url=args.url,
        model=model_name,
        num_clients=args.num_clients,
        max_num_requests=args.max_num_requests,
        max_turns=args.max_turns,
        limit_min_tokens=args.limit_min_tokens,
        limit_max_tokens=args.limit_max_tokens,
        timeout_sec=args.request_timeout_sec,
        verbose=args.verbose,
    )

    print_statistics(results)

    if args.output_file:
        output_data = conversations_dict_to_list(conversations)
        with open(args.output_file, "w") as f:
            json.dump(output_data, f, indent=4)
        logger.info(f"Saved output to {args.output_file}")


if __name__ == "__main__":
    main()
