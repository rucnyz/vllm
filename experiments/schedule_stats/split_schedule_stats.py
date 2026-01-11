#!/usr/bin/env python3
"""
将一个大的 schedule_stats.json 按 benchmark runs 分割成多个文件。

方法：
1. summary: 使用 summary.json 中的 elapsed_seconds 按时间分割（最精确）
2. auto: 根据 timestamp 的间隔自动检测 run 边界
3. even: 均匀分割
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_stats(filepath: str) -> list[dict]:
    """加载 stats 文件（支持修复不完整的 JSON）"""
    try:
        with open(filepath) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  JSON decode error: {e}")
        print("  Attempting to repair truncated JSON...")
        data = repair_json(filepath)
        if data is None:
            return []

    if isinstance(data, dict) and "stats" in data:
        return data["stats"]
    elif isinstance(data, list):
        return data
    return []


def repair_json(filepath: str) -> dict | None:
    """尝试修复不完整的 JSON 文件（如被截断的文件）"""
    print("  Reading file for repair...")
    with open(filepath) as f:
        content = f.read()

    print(f"  File size: {len(content)} bytes")

    # 找到最后一个完整的 JSON 对象 '},\n' 或 '}\n  ]'
    last_complete = content.rfind('},')
    if last_complete == -1:
        last_complete = content.rfind('}]')

    if last_complete == -1:
        print("  Could not find valid JSON structure")
        return None

    print(f"  Last complete entry at position: {last_complete}")

    # 截断到最后一个完整条目
    truncated = content[:last_complete + 1]

    # 补全 JSON 结构
    if '"stats"' in truncated:
        truncated = truncated.rstrip().rstrip(',') + '\n  ]\n}'
    else:
        truncated = truncated.rstrip().rstrip(',') + '\n]'

    try:
        data = json.loads(truncated)
        if isinstance(data, dict) and "stats" in data:
            print(f"  Repaired JSON successfully! Found {len(data['stats'])} entries")
        else:
            print(f"  Repaired JSON successfully!")
        return data
    except json.JSONDecodeError as e:
        print(f"  Repair failed: {e}")
        return None


def load_summary(filepath: str) -> list[dict]:
    """加载 summary.json"""
    with open(filepath) as f:
        return json.load(f)


def split_by_summary(stats: list[dict], summary: list[dict]) -> list[list[dict]]:
    """
    使用 summary.json 中的 elapsed_seconds 来分割 stats

    原理：每个 run 的 elapsed_seconds 对应一段时间内的 stats
    stats 中的 timestamp 是相对时间（从服务器启动开始）
    """
    if not stats or not summary:
        return []

    # 获取所有 stats 的 timestamp
    timestamps = np.array([s.get("timestamp", 0) for s in stats])

    # 计算每个 run 的累计结束时间
    cumulative_times = []
    total = 0
    for run in summary:
        total += run.get("elapsed_seconds", 0)
        cumulative_times.append(total)

    print(f"  Stats timestamp range: {timestamps[0]:.2f} - {timestamps[-1]:.2f}")
    print(f"  Cumulative run times: {[f'{t:.1f}' for t in cumulative_times]}")

    # 找到每个 run 的边界索引
    # 假设 stats 的 timestamp 和 benchmark 运行时间大致对应
    # 需要对齐：stats 可能在 benchmark 开始前就有记录

    # 方法：用相对比例分割
    total_stats_time = timestamps[-1] - timestamps[0]
    total_benchmark_time = cumulative_times[-1]

    runs = []
    start_idx = 0

    for i, run in enumerate(summary):
        run_duration = run.get("elapsed_seconds", 0)

        if i == len(summary) - 1:
            # 最后一个 run，取剩余所有
            end_idx = len(stats)
        else:
            # 按时间比例计算结束索引
            # 计算这个 run 应该占的 stats 数量
            proportion = run_duration / total_benchmark_time
            expected_count = int(len(stats) * proportion)
            end_idx = min(start_idx + expected_count, len(stats))

        runs.append(stats[start_idx:end_idx])
        start_idx = end_idx

    return runs


def detect_run_boundaries(stats: list[dict], num_runs: int = 15) -> list[int]:
    """
    检测 run 边界（基于 timestamp 间隔）

    Returns: 每个 run 的起始索引列表
    """
    if not stats:
        return []

    timestamps = [s.get("timestamp", 0) for s in stats]

    # 计算相邻 timestamp 的间隔
    gaps = []
    for i in range(1, len(timestamps)):
        gap = timestamps[i] - timestamps[i-1]
        gaps.append((i, gap))

    # 按间隔大小排序，取前 num_runs-1 个最大间隔作为边界
    gaps.sort(key=lambda x: x[1], reverse=True)

    # 取前 num_runs-1 个边界点
    boundary_indices = sorted([0] + [g[0] for g in gaps[:num_runs-1]])

    return boundary_indices


def split_by_boundaries(stats: list[dict], boundaries: list[int]) -> list[list[dict]]:
    """按边界索引分割 stats"""
    runs = []
    for i, start in enumerate(boundaries):
        if i + 1 < len(boundaries):
            end = boundaries[i + 1]
        else:
            end = len(stats)
        runs.append(stats[start:end])
    return runs


def split_evenly(stats: list[dict], num_runs: int) -> list[list[dict]]:
    """均匀分割成 num_runs 份"""
    chunk_size = len(stats) // num_runs
    runs = []
    for i in range(num_runs):
        start = i * chunk_size
        if i == num_runs - 1:
            end = len(stats)
        else:
            end = (i + 1) * chunk_size
        runs.append(stats[start:end])
    return runs


def main():
    parser = argparse.ArgumentParser(
        description="Split schedule_stats.json into multiple files per benchmark run"
    )
    parser.add_argument("stats_file", help="Path to schedule_stats.json")
    parser.add_argument("-o", "--output-dir", default="./split_stats",
                        help="Output directory")
    parser.add_argument("-n", "--num-runs", type=int, default=15,
                        help="Number of benchmark runs")
    parser.add_argument("--method", choices=["summary", "auto", "even"], default="auto",
                        help="Split method: summary (use summary.json), "
                             "auto (detect gaps), even (equal chunks)")
    parser.add_argument("--summary", type=str, default=None,
                        help="Path to summary.json (required for --method summary)")
    parser.add_argument("--scenarios", nargs="+",
                        default=["D(1024,256)", "D(128,2048)", "D(512,512)"],
                        help="Scenario names in order")
    parser.add_argument("--concurrencies", nargs="+", type=int,
                        default=[8, 32, 64, 128, 256],
                        help="Concurrency levels in order")

    args = parser.parse_args()

    # 加载 stats
    print(f"Loading stats from: {args.stats_file}")
    stats = load_stats(args.stats_file)
    print(f"Total entries: {len(stats)}")

    if not stats:
        print("Error: No stats found!")
        return

    # 生成 run 名称列表
    run_names = []
    run_infos = []
    for scenario in args.scenarios:
        for concurrency in args.concurrencies:
            scenario_name = scenario.replace("(", "").replace(")", "").replace(",", "_")
            run_name = f"{scenario_name}_c{concurrency}"
            run_names.append(run_name)
            run_infos.append({"scenario": scenario, "concurrency": concurrency})

    print(f"Expected runs: {len(run_names)}")

    # 分割
    summary_data = None
    if args.method == "summary":
        if not args.summary:
            print("Error: --summary is required for --method summary")
            return
        print(f"\nLoading summary from: {args.summary}")
        summary_data = load_summary(args.summary)
        print(f"Summary has {len(summary_data)} runs")
        print("\nSplitting by summary elapsed times...")
        runs = split_by_summary(stats, summary_data)
    elif args.method == "auto":
        print(f"\nDetecting run boundaries (looking for {args.num_runs} runs)...")
        boundaries = detect_run_boundaries(stats, args.num_runs)
        print(f"Detected boundaries at indices: {boundaries}")
        runs = split_by_boundaries(stats, boundaries)
    else:
        print(f"\nSplitting evenly into {args.num_runs} chunks...")
        runs = split_evenly(stats, args.num_runs)

    # 检查分割结果
    print(f"\nSplit results:")
    for i, run_stats in enumerate(runs):
        if i < len(run_names):
            name = run_names[i]
        else:
            name = f"run_{i}"
        print(f"  {name}: {len(run_stats)} entries")

    # 保存
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for i, run_stats in enumerate(runs):
        if i < len(run_names):
            name = run_names[i]
            info = run_infos[i]
        else:
            name = f"run_{i}"
            info = {"scenario": "unknown", "concurrency": 0}

        # 如果有原始 summary，合并信息
        if summary_data and i < len(summary_data):
            orig = summary_data[i]
            run_info = {
                "run_name": name,
                "scenario": info["scenario"],
                "concurrency": info["concurrency"],
                "elapsed_seconds": orig.get("elapsed_seconds", 0),
                "return_code": orig.get("return_code", 0),
                "num_schedule_entries": len(run_stats),
            }
        else:
            run_info = {
                "run_name": name,
                "scenario": info["scenario"],
                "concurrency": info["concurrency"],
                "num_schedule_entries": len(run_stats),
            }

        output_data = {
            "run_info": run_info,
            "schedule_stats": run_stats,
        }

        output_file = output_dir / f"{name}.json"
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        all_summaries.append(run_info)

    # 保存 summary
    summary_file = output_dir / "split_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\nSaved {len(runs)} files to: {output_dir}/")
    print(f"Summary saved to: {summary_file}")

    # 打印汇总表
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Name':<25} {'Entries':<10} {'Time(s)':<10}")
    print("-" * 45)
    for s in all_summaries:
        elapsed = s.get("elapsed_seconds", "N/A")
        if isinstance(elapsed, float):
            elapsed = f"{elapsed:.1f}"
        print(f"{s['run_name']:<25} {s['num_schedule_entries']:<10} {elapsed:<10}")


if __name__ == "__main__":
    main()
