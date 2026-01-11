#!/usr/bin/env python3
"""
Simple benchmark script to generate load for testing scheduler statistics.

Usage:
    python simple_benchmark.py --url http://localhost:8124 --num-requests 500 --concurrency 128
"""

import argparse
import asyncio
import csv
import json
import time
from pathlib import Path

import aiohttp


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    api_key: str,
    model: str,
    max_tokens: int = 100,
) -> dict:
    """Send a single completion request."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }

    start_time = time.perf_counter()
    try:
        async with session.post(
            f"{url}/v1/completions",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as response:
            result = await response.json()
            end_time = time.perf_counter()

            if response.status == 200:
                output_tokens = result.get("usage", {}).get("completion_tokens", 0)
                return {
                    "success": True,
                    "latency": end_time - start_time,
                    "output_tokens": output_tokens,
                }
            else:
                return {
                    "success": False,
                    "latency": end_time - start_time,
                    "error": result.get("error", str(result)),
                }
    except Exception as e:
        end_time = time.perf_counter()
        return {
            "success": False,
            "latency": end_time - start_time,
            "error": str(e),
        }


async def worker(
    queue: asyncio.Queue,
    session: aiohttp.ClientSession,
    url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    results: list,
):
    """Worker coroutine that processes requests from the queue."""
    while True:
        try:
            prompt = await asyncio.wait_for(queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            break

        if prompt is None:  # Poison pill
            queue.task_done()
            break

        result = await send_request(session, url, prompt, api_key, model, max_tokens)
        results.append(result)
        queue.task_done()


async def run_benchmark(
    url: str,
    prompts: list[str],
    api_key: str,
    model: str,
    concurrency: int,
    max_tokens: int,
    max_time: float,
) -> list[dict]:
    """Run the benchmark with specified concurrency."""
    results = []
    queue = asyncio.Queue()

    # Add all prompts to the queue
    for prompt in prompts:
        await queue.put(prompt)

    # Add poison pills for workers
    for _ in range(concurrency):
        await queue.put(None)

    start_time = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        workers = [
            asyncio.create_task(
                worker(queue, session, url, api_key, model, max_tokens, results)
            )
            for _ in range(concurrency)
        ]

        # Wait for completion or timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*workers, return_exceptions=True),
                timeout=max_time,
            )
        except asyncio.TimeoutError:
            print(f"\nTimeout reached ({max_time}s), stopping...")
            for w in workers:
                w.cancel()

    elapsed = time.perf_counter() - start_time
    return results, elapsed


def load_prompts(csv_path: str, prompt_column: str = "prompt") -> list[str]:
    """Load prompts from CSV file."""
    prompts = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if prompt_column in row:
                prompts.append(row[prompt_column])
    return prompts


def print_results(results: list[dict], elapsed: float):
    """Print benchmark results."""
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print(f"\nTotal requests: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {elapsed:.2f}s")

    if successful:
        latencies = [r["latency"] for r in successful]
        output_tokens = [r["output_tokens"] for r in successful]

        print(f"\nThroughput: {len(successful) / elapsed:.2f} req/s")
        print(f"Total output tokens: {sum(output_tokens)}")
        print(f"Token throughput: {sum(output_tokens) / elapsed:.2f} tokens/s")

        print(f"\nLatency (s):")
        print(f"  Mean: {sum(latencies) / len(latencies):.3f}")
        print(f"  Min:  {min(latencies):.3f}")
        print(f"  Max:  {max(latencies):.3f}")
        sorted_lat = sorted(latencies)
        p50_idx = int(len(sorted_lat) * 0.5)
        p99_idx = int(len(sorted_lat) * 0.99)
        print(f"  P50:  {sorted_lat[p50_idx]:.3f}")
        print(f"  P99:  {sorted_lat[min(p99_idx, len(sorted_lat)-1)]:.3f}")

    if failed:
        print(f"\nFirst 3 errors:")
        for r in failed[:3]:
            print(f"  - {r['error'][:100]}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Simple vLLM benchmark")
    parser.add_argument("--url", default="http://localhost:8124", help="vLLM server URL")
    parser.add_argument("--api-key", default="7355608", help="API key")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Model name")
    parser.add_argument(
        "--dataset", default="./experiments/serve/alpaca_prompts.csv", help="CSV file path"
    )
    parser.add_argument("--prompt-column", default="prompt", help="Prompt column name")
    parser.add_argument("--num-requests", type=int, default=500, help="Number of requests")
    parser.add_argument("--concurrency", type=int, default=128, help="Concurrent requests")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max output tokens")
    parser.add_argument("--max-time", type=float, default=120, help="Max benchmark time (s)")

    args = parser.parse_args()

    # Load prompts
    print(f"Loading prompts from {args.dataset}...")
    prompts = load_prompts(args.dataset, args.prompt_column)
    print(f"Loaded {len(prompts)} prompts")

    # Limit to requested number
    prompts = prompts[: args.num_requests]
    print(f"Using {len(prompts)} prompts")

    print(f"\nStarting benchmark...")
    print(f"  URL: {args.url}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Max time: {args.max_time}s")

    # Run benchmark
    results, elapsed = asyncio.run(
        run_benchmark(
            url=args.url,
            prompts=prompts,
            api_key=args.api_key,
            model=args.model,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            max_time=args.max_time,
        )
    )

    print_results(results, elapsed)


if __name__ == "__main__":
    main()
