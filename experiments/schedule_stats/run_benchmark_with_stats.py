#!/usr/bin/env python3
"""
运行 genai-bench 并为每个 run 保存 schedule stats
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def read_stats_file(stats_file: str) -> list:
    """读取 stats 文件，返回条目列表

    文件格式: {"stats": [{...}, {...}, ...]}
    """
    if not os.path.exists(stats_file):
        return []
    try:
        with open(stats_file, "r") as f:
            data = json.load(f)
        # 格式是 {"stats": [...]}
        if isinstance(data, dict) and "stats" in data:
            return data["stats"]
        # 如果直接是 list
        elif isinstance(data, list):
            return data
        else:
            return []
    except (json.JSONDecodeError, Exception):
        return []


def run_single_benchmark(
    scenario: str,
    concurrency: int,
    stats_file: str,
    results_dir: str,
    api_key: str,
    api_base: str,
    model: str,
    max_time: int,
    max_requests: int,
) -> dict:
    """运行单个 benchmark 并保存 stats (无需重启服务器)"""

    # 生成运行名称
    scenario_name = scenario.replace("(", "").replace(")", "").replace(",", "_")
    run_name = f"{scenario_name}_c{concurrency}"

    print(f"\n{'=' * 60}")
    print(f"Running: {run_name}")
    print(f"  Scenario: {scenario}")
    print(f"  Concurrency: {concurrency}")
    print(f"{'=' * 60}")

    # 记录 run 前的 stats 数量
    stats_before = read_stats_file(stats_file)
    num_before = len(stats_before)
    print(f"  Stats entries before run: {num_before}")

    # 调试：检查文件是否存在
    if os.path.exists(stats_file):
        file_size = os.path.getsize(stats_file)
        print(f"  Stats file exists, size: {file_size} bytes")
    else:
        print(f"  WARNING: Stats file does not exist: {stats_file}")

    # 构建命令
    cmd = [
        "genai-bench", "benchmark",
        "--api-backend", "vllm",
        "--api-key", api_key,
        "--api-base", api_base,
        "--api-model-name", model,
        "--model-tokenizer", model,
        "--task", "text-to-text",
        "--traffic-scenario", scenario,
        "--num-concurrency", str(concurrency),
        "--max-time-per-run", str(max_time),
        "--max-requests-per-run", str(max_requests),
        "--experiment-base-dir", results_dir,
    ]

    # 运行 benchmark
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start_time

    # 读取 run 后的 stats，提取新增条目
    stats_after = read_stats_file(stats_file)
    num_after = len(stats_after)
    new_entries = stats_after[num_before:]  # 只取新增的条目

    print(f"  Stats entries after run: {num_after}")
    print(f"  New entries this run: {len(new_entries)}")

    # 调试：如果没有新条目，检查原因
    if len(new_entries) == 0:
        if os.path.exists(stats_file):
            file_size = os.path.getsize(stats_file)
            print(f"  WARNING: No new entries! File size: {file_size} bytes")
            print("  Check if server has VLLM_COLLECT_SCHEDULE_STATS=1")
        else:
            print("  WARNING: Stats file not found after run!")

    # 保存这次 run 的 stats
    stats_output_dir = Path(results_dir) / "schedule_stats"
    stats_output_dir.mkdir(parents=True, exist_ok=True)
    stats_output = stats_output_dir / f"{run_name}.json"

    run_info = {
        "run_name": run_name,
        "scenario": scenario,
        "concurrency": concurrency,
        "elapsed_seconds": elapsed,
        "return_code": result.returncode,
        "num_schedule_entries": len(new_entries),
    }

    output_data = {
        "run_info": run_info,
        "schedule_stats": new_entries,
    }

    with open(stats_output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"  Saved stats to: {stats_output}")

def main():
    parser = argparse.ArgumentParser(description="Run genai-bench with schedule stats collection")
    parser.add_argument("--stats-file", type=str,
                        default="/scratch/yuzhou/zwf/vllm/schedule_stats.json",
                        help="Path to schedule stats file")
    parser.add_argument("--results-dir", type=str,
                        default="./results/scenario_sweep",
                        help="Results directory")
    parser.add_argument("--api-key", type=str, default="7355608")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--max-time", type=int, default=180)
    parser.add_argument("--max-requests", type=int, default=1000)
    args = parser.parse_args()

    # 配置
    scenarios = ["D(1024,256)", "D(128,2048)", "D(512,512)"]
    concurrencies = [8, 32, 64, 128, 256]

    total_runs = len(scenarios) * len(concurrencies)
    print("=" * 60)
    print(f"Starting benchmark with schedule stats collection")
    print(f"Total runs: {total_runs}")
    print(f"Scenarios: {scenarios}")
    print(f"Concurrencies: {concurrencies}")
    print("=" * 60)

    all_results = []
    run_index = 0

    for scenario in scenarios:
        for concurrency in concurrencies:
            run_index += 1
            print(f"\n>>> Run {run_index}/{total_runs}")

            run_info = run_single_benchmark(
                scenario=scenario,
                concurrency=concurrency,
                stats_file=args.stats_file,
                results_dir=args.results_dir,
                api_key=args.api_key,
                api_base=args.api_base,
                model=args.model,
                max_time=args.max_time,
                max_requests=args.max_requests,
            )
            all_results.append(run_info)

            # 短暂等待
            time.sleep(2)

    # 保存汇总结果
    summary_file = Path(args.results_dir) / "schedule_stats" / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # 打印汇总
    print("\n" + "=" * 60)
    print("All runs complete!")
    print("=" * 60)
    print(f"\n{'Run Name':<25} {'Entries':<10} {'Time(s)':<10} {'Status'}")
    print("-" * 60)
    for r in all_results:
        status = "OK" if r["return_code"] == 0 else f"FAIL({r['return_code']})"
        print(f"{r['run_name']:<25} {r.get('num_schedule_entries', 'N/A'):<10} "
              f"{r['elapsed_seconds']:<10.1f} {status}")

    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
