#!/usr/bin/env python3
"""
K* 参数扫描结果可视化脚本
支持两种数据源:
  - schedule: 调度统计文件 (baseline_stats.json, fixed*_stats.json) - 来自 VLLM_COLLECT_SCHEDULE_STATS
  - bench: Benchmark 结果文件 (bench_baseline.json, bench_fixed*.json) - 来自 vllm bench serve

用法示例:
  # 绘制调度统计 (schedule stats)
  python plot_kstar_sweep.py --results-dir ./results/long_in_short_out --mode schedule

  # 绘制 benchmark 结果 (vllm bench serve 输出)
  python plot_kstar_sweep.py --results-dir ./results/long_in_short_out --mode bench

  # 绘制组合图 (schedule + bench 在同一张图, 2x6 布局, 含 preemption 统计)
  python plot_kstar_sweep.py --results-dir ./results/long_in_short_out --mode all

  # 过滤 K* 范围
  python plot_kstar_sweep.py --results-dir ./results --mode bench --k-min 4 --k-max 64

  # 绘制 dynamic K* 结果
  python plot_kstar_sweep.py --results-dir ./results --mode schedule --plot-dynamic

  # 指定输出文件名
  python plot_kstar_sweep.py --results-dir ./results --mode bench --output my_results.png
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_json(filepath: Path) -> dict | None:
    """加载 JSON 文件，支持修复损坏的文件"""
    try:
        with open(filepath, encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"  JSON decode error in {filepath.name}: {e}")
        print("  Attempting to repair truncated JSON...")
        return repair_json(filepath)


def repair_json(filepath: Path) -> dict | None:
    """尝试修复不完整的 JSON 文件（如被截断的文件）"""
    with open(filepath, encoding='utf-8') as f:
        content = f.read()

    # 尝试找到最后一个完整的 JSON 对象
    last_complete = content.rfind('},')
    if last_complete == -1:
        last_complete = content.rfind('}]')

    if last_complete == -1:
        print("  Could not find valid JSON structure")
        return None

    # 截断到最后一个完整条目
    truncated = content[:last_complete + 1]

    # 补全 JSON 结构
    if '"stats"' in truncated:
        truncated = truncated.rstrip().rstrip(',') + '\n  ]\n}'
    else:
        truncated = truncated.rstrip().rstrip(',') + '\n]'

    try:
        data = json.loads(truncated)
        print(f"  Repaired JSON successfully (truncated at char {last_complete})")
        return data
    except json.JSONDecodeError as e:
        print(f"  Repair failed: {e}")
        return None


def compute_schedule_metrics(stats: list[dict]) -> dict:
    """从原始调度记录计算性能指标 (schedule mode)"""
    if not stats:
        return {}

    total_tokens = [s["total_tokens"] for s in stats]
    prefill_tokens = [s["prefill_tokens"] for s in stats]
    decode_tokens = [s["decode_tokens"] for s in stats]
    elapsed_us = [s["elapsed_us"] for s in stats]
    timestamps = [s["timestamp"] for s in stats]

    # 计算总时间和吞吐量
    total_time = stats[-1]["timestamp"] - stats[0]["timestamp"] if len(stats) > 1 else 1
    total_prefill = sum(prefill_tokens)
    total_decode = sum(decode_tokens)

    # 计算调度间隔
    schedule_intervals_ms = []
    for i in range(1, len(timestamps)):
        interval = (timestamps[i] - timestamps[i - 1]) * 1000
        schedule_intervals_ms.append(interval)

    # 提取 k_star 轨迹（如果存在）
    k_star_trajectory = []
    k_star_times = []
    if stats and "k_star" in stats[0]:
        start_time = stats[0]["timestamp"]
        for s in stats:
            k_star_trajectory.append(s.get("k_star", 0))
            k_star_times.append(s["timestamp"] - start_time)

    # 提取 N 轨迹（如果存在）- N 是 batch size
    n_trajectory = []
    n_times = []
    if stats and "N" in stats[0]:
        start_time = stats[0]["timestamp"]
        for s in stats:
            n_trajectory.append(s.get("N", 0))
            n_times.append(s["timestamp"] - start_time)

    # 提取 preemption 统计
    num_preempted_reqs = [s.get("num_preempted_reqs", 0) for s in stats]
    preempted_tokens = [s.get("preempted_tokens", 0) for s in stats]
    total_preempted_reqs = sum(num_preempted_reqs)
    total_preempted_tokens = sum(preempted_tokens)
    schedules_with_preemption = sum(1 for n in num_preempted_reqs if n > 0)

    # 提取 preemption 轨迹（累积）
    preemption_trajectory = []
    preemption_times = []
    if stats:
        start_time = stats[0]["timestamp"]
        cumulative_preemptions = 0
        for s in stats:
            cumulative_preemptions += s.get("num_preempted_reqs", 0)
            preemption_trajectory.append(cumulative_preemptions)
            preemption_times.append(s["timestamp"] - start_time)

    return {
        "total_time_s": total_time,
        "total_tokens": total_prefill + total_decode,
        "total_prefill_tokens": total_prefill,
        "total_decode_tokens": total_decode,
        "overall_throughput_tps": (total_prefill + total_decode) / total_time if total_time > 0 else 0,
        "prefill_throughput_tps": total_prefill / total_time if total_time > 0 else 0,
        "decode_throughput_tps": total_decode / total_time if total_time > 0 else 0,
        "schedule_frequency_hz": len(stats) / total_time if total_time > 0 else 0,
        "tokens_per_schedule_mean": np.mean(total_tokens),
        "tokens_per_schedule_p50": np.percentile(total_tokens, 50),
        "tokens_per_schedule_p99": np.percentile(total_tokens, 99),
        "schedule_time_us_mean": np.mean(elapsed_us),
        "schedule_time_us_p99": np.percentile(elapsed_us, 99),
        "schedule_interval_ms_mean": np.mean(schedule_intervals_ms) if schedule_intervals_ms else 0,
        "schedule_interval_ms_p99": np.percentile(schedule_intervals_ms, 99) if schedule_intervals_ms else 0,
        "empty_schedule_ratio": sum(1 for t in total_tokens if t == 0) / len(stats) if stats else 0,
        "num_schedules": len(stats),
        "k_star_trajectory": k_star_trajectory,
        "k_star_times": k_star_times,
        # N trajectory (batch size)
        "n_trajectory": n_trajectory,
        "n_times": n_times,
        # Preemption statistics
        "total_preempted_reqs": total_preempted_reqs,
        "total_preempted_tokens": total_preempted_tokens,
        "schedules_with_preemption": schedules_with_preemption,
        "preemption_rate": schedules_with_preemption / len(stats) if stats else 0,
        "preemption_trajectory": preemption_trajectory,
        "preemption_times": preemption_times,
    }


def extract_bench_metrics(data: dict) -> dict:
    """从 vllm bench serve 输出提取指标 (bench mode)"""
    return {
        "total_token_throughput": data.get("total_token_throughput", 0),
        "output_throughput": data.get("output_throughput", 0),
        "request_throughput": data.get("request_throughput", 0),
        "mean_ttft_ms": data.get("mean_ttft_ms", 0),
        "p99_ttft_ms": data.get("p99_ttft_ms", 0),
        "mean_tpot_ms": data.get("mean_tpot_ms", 0),
        "p99_tpot_ms": data.get("p99_tpot_ms", 0),
        "mean_itl_ms": data.get("mean_itl_ms", 0),
        "p99_itl_ms": data.get("p99_itl_ms", 0),
    }


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="K* 参数扫描结果可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 绘制调度统计
  python plot_kstar_sweep.py --results-dir ./results/long_in_short_out --mode schedule

  # 绘制 benchmark 结果
  python plot_kstar_sweep.py --results-dir ./results/long_in_short_out --mode bench

  # 过滤 K* 范围并保存到指定文件
  python plot_kstar_sweep.py --results-dir ./results --mode bench --k-min 4 --k-max 64 --output result.png
"""
    )
    parser.add_argument(
        "--mode", type=str, default="all", choices=["schedule", "bench", "all"],
        help="数据源模式: schedule=调度统计, bench=benchmark结果, all=组合图(2x6布局,含preemption) (default: all)"
    )
    parser.add_argument(
        "--k-min", type=int, default=4,
        help="K* 最小值 (default: 4)"
    )
    parser.add_argument(
        "--k-max", type=int, default=None,
        help="K* 最大值 (default: None, 无限制)"
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="结果目录路径 (default: 脚本同目录下的 results)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出图片路径 (default: kstar_sweep_results.png 或 benchmark_results.png)"
    )
    parser.add_argument(
        "--plot-dynamic", action="store_true",
        help="是否绘制 dynamic K* 结果 (仅 schedule 模式, default: False)"
    )
    parser.add_argument(
        "--plot-n-trajectory", action="store_true",
        help="绘制 N (batch size) 变动轨迹图 (仅 K ratio 模式, default: False)"
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="不显示图表窗口，仅保存文件"
    )
    return parser.parse_args()


def load_schedule_data(results_dir: Path, args) -> tuple[dict, dict, dict, dict, dict]:
    """加载 schedule stats 数据，返回 (fixed_k_results, baseline, dynamic_results, kratio_results, special_results)

    special_results 包含:
      - 'ratio_auto': 动态 θ* 模式结果
      - 'dynamic': 动态 k* 模式结果 (DP algorithm)
    """
    k_star_results = {}
    baseline_metrics = None
    dynamic_results = {}
    kratio_results = {}  # K ratio 结果 (自适应 N)
    special_results = {}  # ratio_auto, dynamic 等特殊模式结果

    # 加载 fixed*_stats.json (固定 K* 模式)
    for filepath in results_dir.glob("fixed*_stats.json"):
        match = re.search(r'fixed(\d+)_stats\.json', filepath.name)
        if match:
            k_star = int(match.group(1))
            if k_star < args.k_min:
                continue
            if args.k_max is not None and k_star > args.k_max:
                continue
            data = load_json(filepath)
            if data is None or "stats" not in data:
                print(f"  Skipping K*={k_star}: invalid data")
                continue
            metrics = compute_schedule_metrics(data["stats"])
            k_star_results[k_star] = metrics
            print(f"Loaded K*={k_star}: throughput={metrics['overall_throughput_tps']:.0f} tok/s")

    # 加载 kratio_*_stats.json (K ratio 模式，自适应 N)
    for filepath in results_dir.glob("kratio_*_stats.json"):
        match = re.search(r'kratio_(\d+)_(\d+)_stats\.json', filepath.name)
        if match:
            # 0_5 -> 0.5
            ratio = float(f"{match.group(1)}.{match.group(2)}")
            data = load_json(filepath)
            if data is None or "stats" not in data:
                print(f"  Skipping K ratio={ratio}: invalid data")
                continue
            metrics = compute_schedule_metrics(data["stats"])
            kratio_results[ratio] = metrics
            print(f"Loaded K ratio={ratio}: throughput={metrics['overall_throughput_tps']:.0f} tok/s")

    # 加载 baseline (支持 baseline_stats.json 或 baseline.json)
    baseline_path = results_dir / "baseline_stats.json"
    if not baseline_path.exists():
        baseline_path = results_dir / "baseline.json"
    if baseline_path.exists():
        data = load_json(baseline_path)
        if data is not None and "stats" in data:
            baseline_metrics = compute_schedule_metrics(data["stats"])
            print(f"Loaded baseline: throughput={baseline_metrics['overall_throughput_tps']:.0f} tok/s")
        else:
            print("  Skipping baseline: invalid data")

    # 加载 dynamic K* 结果
    if args.plot_dynamic:
        for filepath in results_dir.glob("dynamic_ema*.json"):
            match = re.search(r'dynamic_ema([\d.]+)\.json', filepath.name)
            if match:
                ema_alpha = match.group(1)
                data = load_json(filepath)
                if data is None or "stats" not in data:
                    continue
                metrics = compute_schedule_metrics(data["stats"])
                dynamic_results[f"EMA α={ema_alpha}"] = metrics
                tps = metrics['overall_throughput_tps']
                print(f"Loaded Dynamic (EMA α={ema_alpha}): throughput={tps:.0f} tok/s")

        for filepath in results_dir.glob("dynamic_interval*.json"):
            match = re.search(r'dynamic_interval(\d+)\.json', filepath.name)
            if match:
                interval = match.group(1)
                data = load_json(filepath)
                if data is None or "stats" not in data:
                    continue
                metrics = compute_schedule_metrics(data["stats"])
                dynamic_results[f"Interval={interval}"] = metrics
                tps = metrics['overall_throughput_tps']
                print(f"Loaded Dynamic (Interval={interval}): throughput={tps:.0f} tok/s")

    # 加载 ratio_auto_stats.json (动态 θ* 模式)
    ratio_auto_path = results_dir / "ratio_auto_stats.json"
    if ratio_auto_path.exists():
        data = load_json(ratio_auto_path)
        if data is not None and "stats" in data:
            metrics = compute_schedule_metrics(data["stats"])
            special_results["ratio_auto"] = metrics
            print(f"Loaded ratio_auto: throughput={metrics['overall_throughput_tps']:.0f} tok/s")

    # 加载 dynamic_stats.json (动态 k* 模式, DP algorithm)
    dynamic_path = results_dir / "dynamic_stats.json"
    if dynamic_path.exists():
        data = load_json(dynamic_path)
        if data is not None and "stats" in data:
            metrics = compute_schedule_metrics(data["stats"])
            special_results["dynamic"] = metrics
            print(f"Loaded dynamic: throughput={metrics['overall_throughput_tps']:.0f} tok/s")

    return k_star_results, baseline_metrics, dynamic_results, kratio_results, special_results


def load_bench_data(results_dir: Path, args) -> tuple[dict, dict, dict, dict]:
    """加载 vllm bench serve 输出数据，返回 (fixed_k_results, baseline, kratio_results, special_results)

    special_results 包含:
      - 'ratio_auto': 动态 θ* 模式结果
      - 'dynamic': 动态 k* 模式结果 (DP algorithm)
    """
    k_star_results = {}
    baseline_result = None
    kratio_results = {}  # K ratio 结果 (自适应 N)
    special_results = {}  # ratio_auto, dynamic 等特殊模式结果

    # 加载 bench_fixed*.json (固定 K* 模式)
    for filepath in results_dir.glob("bench_fixed*.json"):
        match = re.search(r'bench_fixed(\d+)\.json', filepath.name)
        if match:
            k_star = int(match.group(1))
            if k_star < args.k_min:
                continue
            if args.k_max is not None and k_star > args.k_max:
                continue
            data = load_json(filepath)
            if data is None:
                print(f"  Skipping K*={k_star}: invalid data")
                continue
            metrics = extract_bench_metrics(data)
            k_star_results[k_star] = metrics
            tps = metrics['total_token_throughput']
            ttft = metrics['mean_ttft_ms']
            print(f"Loaded K*={k_star}: throughput={tps:.0f} tok/s, TTFT={ttft:.0f}ms")

    # 加载 bench_kratio_*.json (K ratio 模式，自适应 N)
    for filepath in results_dir.glob("bench_kratio_*.json"):
        match = re.search(r'bench_kratio_(\d+)_(\d+)\.json', filepath.name)
        if match:
            # 0_5 -> 0.5
            ratio = float(f"{match.group(1)}.{match.group(2)}")
            data = load_json(filepath)
            if data is None:
                print(f"  Skipping K ratio={ratio}: invalid data")
                continue
            metrics = extract_bench_metrics(data)
            kratio_results[ratio] = metrics
            tps = metrics['total_token_throughput']
            ttft = metrics['mean_ttft_ms']
            print(f"Loaded K ratio={ratio}: throughput={tps:.0f} tok/s, TTFT={ttft:.0f}ms")

    # 加载 baseline
    baseline_path = results_dir / "bench_baseline.json"
    if baseline_path.exists():
        data = load_json(baseline_path)
        if data is not None:
            baseline_result = extract_bench_metrics(data)
            tps = baseline_result['total_token_throughput']
            ttft = baseline_result['mean_ttft_ms']
            print(f"Loaded baseline: throughput={tps:.0f} tok/s, TTFT={ttft:.0f}ms")
        else:
            print("  Skipping baseline: invalid data")

    # 加载 bench_ratio_auto.json (动态 θ* 模式)
    ratio_auto_path = results_dir / "bench_ratio_auto.json"
    if ratio_auto_path.exists():
        data = load_json(ratio_auto_path)
        if data is not None:
            metrics = extract_bench_metrics(data)
            special_results["ratio_auto"] = metrics
            tps = metrics['total_token_throughput']
            ttft = metrics['mean_ttft_ms']
            print(f"Loaded ratio_auto: throughput={tps:.0f} tok/s, TTFT={ttft:.0f}ms")

    # 加载 bench_dynamic.json (动态 k* 模式, DP algorithm)
    dynamic_path = results_dir / "bench_dynamic.json"
    if dynamic_path.exists():
        data = load_json(dynamic_path)
        if data is not None:
            metrics = extract_bench_metrics(data)
            special_results["dynamic"] = metrics
            tps = metrics['total_token_throughput']
            ttft = metrics['mean_ttft_ms']
            print(f"Loaded dynamic: throughput={tps:.0f} tok/s, TTFT={ttft:.0f}ms")

    return k_star_results, baseline_result, kratio_results, special_results


def plot_kratio_schedule_mode(kratio_results: dict, baseline_metrics: dict, output_path: str, no_show: bool):
    """绘制 K ratio (自适应 N) 的 schedule stats 图表"""
    k_ratios = sorted(kratio_results.keys())

    # 提取各项指标
    throughputs = [kratio_results[r]["overall_throughput_tps"] for r in k_ratios]
    prefill_tps = [kratio_results[r]["prefill_throughput_tps"] for r in k_ratios]
    decode_tps = [kratio_results[r]["decode_throughput_tps"] for r in k_ratios]
    schedule_intervals = [kratio_results[r]["schedule_interval_ms_mean"] for r in k_ratios]
    tokens_per_schedule = [kratio_results[r]["tokens_per_schedule_mean"] for r in k_ratios]
    num_schedules = [kratio_results[r]["num_schedules"] for r in k_ratios]
    preempted_reqs = [kratio_results[r].get("total_preempted_reqs", 0) for r in k_ratios]
    preempted_tokens = [kratio_results[r].get("total_preempted_tokens", 0) for r in k_ratios]

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    # 1. 总吞吐量 vs K ratio
    ax = axes[0, 0]
    ax.plot(k_ratios, throughputs, 'b-o', linewidth=2, markersize=6, label='K Ratio (Adaptive N)')
    if baseline_metrics:
        ax.axhline(y=baseline_metrics['overall_throughput_tps'], color='r',
                   linestyle='--', linewidth=2,
                   label=f"Baseline ({baseline_metrics['overall_throughput_tps']:.0f})")
    ax.set_xlabel('K Ratio (k* = ratio × N)', fontsize=12)
    ax.set_ylabel('Throughput (tokens/s)', fontsize=12)
    ax.set_title('Overall Throughput vs K Ratio', fontsize=14)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    best_idx = np.argmax(throughputs)
    best_ratio = k_ratios[best_idx]
    best_tps = throughputs[best_idx]
    ax.annotate(f'Best: ratio={best_ratio}\n{best_tps:.0f} tok/s',
                xy=(best_ratio, best_tps), xytext=(30, -30), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')

    # 2. Prefill vs Decode 吞吐量 (双 Y 轴)
    ax = axes[0, 1]
    ax.plot(k_ratios, prefill_tps, 'g-o', linewidth=2, markersize=5, label='Prefill', alpha=0.8)
    if baseline_metrics:
        ax.axhline(y=baseline_metrics['prefill_throughput_tps'], color='g',
                   linestyle=':', alpha=0.5, label='Baseline Prefill')
    ax.set_xlabel('K Ratio', fontsize=12)
    ax.set_ylabel('Prefill Throughput (tokens/s)', fontsize=12, color='g')
    ax.tick_params(axis='y', labelcolor='g')

    ax2 = ax.twinx()
    ax2.plot(k_ratios, decode_tps, 'orange', linestyle='-', marker='s',
             linewidth=2, markersize=5, label='Decode', alpha=0.8)
    if baseline_metrics:
        ax2.axhline(y=baseline_metrics['decode_throughput_tps'], color='orange',
                    linestyle=':', alpha=0.5, label='Baseline Decode')
    ax2.set_ylabel('Decode Throughput (tokens/s)', fontsize=12, color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    ax.set_title('Prefill vs Decode Throughput', fontsize=14)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # 3. 调度间隔 + 每次调度 tokens 数 (双 Y 轴)
    ax = axes[1, 0]
    ax.plot(k_ratios, schedule_intervals, 'purple', linestyle='-', marker='d',
            linewidth=2, markersize=6, label='Schedule Interval')
    if baseline_metrics:
        ax.axhline(y=baseline_metrics['schedule_interval_ms_mean'], color='purple',
                   linestyle='--', linewidth=2, alpha=0.5,
                   label=f"Baseline ({baseline_metrics['schedule_interval_ms_mean']:.2f} ms)")
    ax.set_xlabel('K Ratio', fontsize=12)
    ax.set_ylabel('Schedule Interval (ms)', fontsize=12, color='purple')
    ax.tick_params(axis='y', labelcolor='purple')

    ax2 = ax.twinx()
    ax2.plot(k_ratios, tokens_per_schedule, 'teal', linestyle='-', marker='^',
             linewidth=2, markersize=6, label='Tokens per Schedule')
    if baseline_metrics:
        ax2.axhline(y=baseline_metrics['tokens_per_schedule_mean'], color='teal',
                    linestyle='--', linewidth=2, alpha=0.5,
                    label=f"Baseline ({baseline_metrics['tokens_per_schedule_mean']:.1f})")
    ax2.set_ylabel('Tokens per Schedule', fontsize=12, color='teal')
    ax2.tick_params(axis='y', labelcolor='teal')

    ax.set_title('Schedule Interval & Tokens per Schedule', fontsize=14)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # 4. 调度总次数 vs K ratio
    ax = axes[1, 1]
    ax.plot(k_ratios, num_schedules, 'brown', linestyle='-', marker='v',
            linewidth=2, markersize=6, label='K Ratio (Adaptive N)')
    if baseline_metrics:
        ax.axhline(y=baseline_metrics['num_schedules'], color='r',
                   linestyle='--', linewidth=2,
                   label=f"Baseline ({baseline_metrics['num_schedules']})")
    ax.set_xlabel('K Ratio', fontsize=12)
    ax.set_ylabel('Number of Schedules', fontsize=12)
    ax.set_title('Total Number of Schedules vs K Ratio', fontsize=14)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    min_sched_idx = np.argmin(num_schedules)
    min_sched_ratio = k_ratios[min_sched_idx]
    min_sched_val = num_schedules[min_sched_idx]
    ax.annotate(f'Min: ratio={min_sched_ratio}\n{min_sched_val}',
                xy=(min_sched_ratio, min_sched_val), xytext=(30, 20), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')

    # 5. Preemption 统计
    ax = axes[0, 2]
    has_preemption = any(p > 0 for p in preempted_reqs)
    if has_preemption:
        ax.plot(k_ratios, preempted_reqs, 'red', linestyle='-', marker='x',
                linewidth=2, markersize=6, label='Preempted Requests')
        ax.set_ylabel('Preempted Requests', fontsize=12, color='red')
        ax.tick_params(axis='y', labelcolor='red')

        ax2 = ax.twinx()
        ax2.plot(k_ratios, preempted_tokens, 'darkred', linestyle='--', marker='+',
                 linewidth=2, markersize=6, label='Preempted Tokens')
        ax2.set_ylabel('Preempted Tokens', fontsize=12, color='darkred')
        ax2.tick_params(axis='y', labelcolor='darkred')

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='best')

        if max(preempted_reqs) > 0:
            max_preempt_idx = np.argmax(preempted_reqs)
            ax.annotate(f'Max: ratio={k_ratios[max_preempt_idx]}\n{preempted_reqs[max_preempt_idx]} reqs',
                        xy=(k_ratios[max_preempt_idx], preempted_reqs[max_preempt_idx]),
                        xytext=(30, -20), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=9, color='red')
    else:
        ax.text(0.5, 0.5, 'No Preemption\nDetected', transform=ax.transAxes,
                fontsize=14, ha='center', va='center', color='green')
        ax.set_ylabel('Preempted Requests', fontsize=12)

    ax.set_xlabel('K Ratio', fontsize=12)
    ax.set_title('Preemption Statistics vs K Ratio', fontsize=14)
    ax.grid(True, alpha=0.3)

    # 6. Prefill Overhead 分析
    ax = axes[1, 2]
    actual_prefill = [kratio_results[r]["total_prefill_tokens"] for r in k_ratios]
    actual_decode = [kratio_results[r]["total_decode_tokens"] for r in k_ratios]
    overhead_pct = [(p - d) / d * 100 if d > 0 else 0 for p, d in zip(actual_prefill, actual_decode)]

    bar_width = (k_ratios[-1] - k_ratios[0]) / len(k_ratios) * 0.8 if len(k_ratios) > 1 else 0.1
    ax.bar(k_ratios, overhead_pct, width=bar_width,
           color=['red' if o > 0 else 'green' for o in overhead_pct], alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    ax.set_xlabel('K Ratio', fontsize=12)
    ax.set_ylabel('Prefill Overhead (%)', fontsize=12)
    ax.set_title('Prefill Overhead vs K Ratio\n(Prefill - Decode) / Decode', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(w_pad=3.0, h_pad=2.0)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    if not no_show:
        plt.show()

    # 打印汇总表
    print("\n" + "=" * 120)
    print("K RATIO SWEEP RESULTS SUMMARY (Schedule Stats, Adaptive N)")
    print("=" * 120)
    print(f"{'Ratio':>8} | {'Throughput':>12} | {'Prefill':>10} | {'Decode':>10} | {'Interval':>10} | {'Tok/Sched':>10} | {'#Scheds':>10} | {'#Preempt':>10} | {'Overhead':>8}")
    print(f"{'':>8} | {'(tok/s)':>12} | {'(tok/s)':>10} | {'(tok/s)':>10} | {'(ms)':>10} | {'':>10} | {'':>10} | {'(reqs)':>10} | {'(%)':>8}")
    print("-" * 120)

    if baseline_metrics:
        base_overhead = ((baseline_metrics['total_prefill_tokens'] - baseline_metrics['total_decode_tokens'])
                        / baseline_metrics['total_decode_tokens'] * 100
                        if baseline_metrics.get('total_decode_tokens', 0) > 0 else 0)
        print(f"{'BASE':>8} | {baseline_metrics['overall_throughput_tps']:>12.0f} | "
              f"{baseline_metrics['prefill_throughput_tps']:>10.0f} | "
              f"{baseline_metrics['decode_throughput_tps']:>10.0f} | "
              f"{baseline_metrics['schedule_interval_ms_mean']:>10.2f} | "
              f"{baseline_metrics['tokens_per_schedule_mean']:>10.1f} | "
              f"{baseline_metrics['num_schedules']:>10} | "
              f"{baseline_metrics.get('total_preempted_reqs', 0):>10} | "
              f"{base_overhead:>8.1f}")
        print("-" * 120)

    for r in k_ratios:
        m = kratio_results[r]
        marker = " <-- BEST TPS" if r == best_ratio else ""
        marker = " <-- MIN SCHED" if r == min_sched_ratio else marker
        overhead = (m['total_prefill_tokens'] - m['total_decode_tokens']) / m['total_decode_tokens'] * 100 if m['total_decode_tokens'] > 0 else 0
        print(f"{r:>8.2f} | {m['overall_throughput_tps']:>12.0f} | "
              f"{m['prefill_throughput_tps']:>10.0f} | "
              f"{m['decode_throughput_tps']:>10.0f} | "
              f"{m['schedule_interval_ms_mean']:>10.2f} | "
              f"{m['tokens_per_schedule_mean']:>10.1f} | "
              f"{m['num_schedules']:>10} | "
              f"{m.get('total_preempted_reqs', 0):>10} | "
              f"{overhead:>8.1f}{marker}")

    print("=" * 120)

    if baseline_metrics:
        print(f"\nBest Throughput ratio={best_ratio} vs Baseline:")
        improvement = (best_tps - baseline_metrics['overall_throughput_tps']) / baseline_metrics['overall_throughput_tps'] * 100
        print(f"  Throughput improvement: {improvement:+.1f}%")
        print(f"\nMin Schedules ratio={min_sched_ratio} vs Baseline:")
        sched_reduction = (min_sched_val - baseline_metrics['num_schedules']) / baseline_metrics['num_schedules'] * 100
        print(f"  Schedule count change: {sched_reduction:+.1f}%")


def plot_kratio_bench_mode(kratio_results: dict, baseline_result: dict, output_path: str, no_show: bool):
    """绘制 K ratio (自适应 N) 的 benchmark 结果图表"""
    k_ratios = sorted(kratio_results.keys())

    # 提取各项指标
    throughputs = [kratio_results[r]["total_token_throughput"] for r in k_ratios]
    output_tps = [kratio_results[r]["output_throughput"] for r in k_ratios]
    request_tps = [kratio_results[r]["request_throughput"] for r in k_ratios]
    mean_ttft = [kratio_results[r]["mean_ttft_ms"] for r in k_ratios]
    p99_ttft = [kratio_results[r]["p99_ttft_ms"] for r in k_ratios]
    mean_tpot = [kratio_results[r]["mean_tpot_ms"] for r in k_ratios]
    p99_tpot = [kratio_results[r]["p99_tpot_ms"] for r in k_ratios]
    mean_itl = [kratio_results[r]["mean_itl_ms"] for r in k_ratios]
    p99_itl = [kratio_results[r]["p99_itl_ms"] for r in k_ratios]

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    # 1. Total Token Throughput vs K ratio
    ax = axes[0, 0]
    ax.plot(k_ratios, throughputs, 'b-o', linewidth=2, markersize=6, label='K Ratio (Adaptive N)')
    if baseline_result:
        ax.axhline(y=baseline_result['total_token_throughput'], color='r',
                   linestyle='--', linewidth=2,
                   label=f"Baseline ({baseline_result['total_token_throughput']:.0f})")
    ax.set_xlabel('K Ratio (k* = ratio × N)', fontsize=12)
    ax.set_ylabel('Throughput (tokens/s)', fontsize=12)
    ax.set_title('Total Token Throughput vs K Ratio', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    best_idx = np.argmax(throughputs)
    best_ratio = k_ratios[best_idx]
    best_tps = throughputs[best_idx]
    ax.annotate(f'Best: ratio={best_ratio}\n{best_tps:.0f} tok/s',
                xy=(best_ratio, best_tps), xytext=(30, -30), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')

    # 2. Output Throughput & Request Throughput
    ax = axes[0, 1]
    ax.plot(k_ratios, output_tps, 'g-o', linewidth=2, markersize=5,
            label='Output (tok/s)', alpha=0.8)
    ax2 = ax.twinx()
    ax2.plot(k_ratios, request_tps, 'orange', linestyle='-', marker='s',
             linewidth=2, markersize=5, label='Request (req/s)', alpha=0.8)
    if baseline_result:
        ax.axhline(y=baseline_result['output_throughput'], color='g', linestyle=':', alpha=0.5)
        ax2.axhline(y=baseline_result['request_throughput'], color='orange', linestyle=':', alpha=0.5)
    ax.set_xlabel('K Ratio', fontsize=12)
    ax.set_ylabel('Output Throughput (tok/s)', fontsize=12, color='g')
    ax2.set_ylabel('Request Throughput (req/s)', fontsize=12, color='orange')
    ax.set_title('Output & Request Throughput vs K Ratio', fontsize=14)
    ax.legend(loc='upper left', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. TTFT vs K ratio
    ax = axes[0, 2]
    baseline_ttft_mean = baseline_result['mean_ttft_ms'] if baseline_result else None
    baseline_ttft_p99 = baseline_result['p99_ttft_ms'] if baseline_result else None
    plot_metric_with_adaptive_axis(ax, k_ratios, mean_ttft, p99_ttft,
                                   baseline_ttft_mean, baseline_ttft_p99,
                                   'purple', 'TTFT', 'ms')
    ax.set_xlabel('K Ratio', fontsize=12)
    ax.set_ylabel('Mean TTFT (ms)', fontsize=12)
    ax.set_title('Time To First Token vs K Ratio', fontsize=14)
    ax.grid(True, alpha=0.3)

    min_ttft_idx = np.argmin(mean_ttft)
    min_ttft_ratio = k_ratios[min_ttft_idx]
    min_ttft_val = mean_ttft[min_ttft_idx]
    ax.annotate(f'Min: ratio={min_ttft_ratio}\n{min_ttft_val:.0f}ms',
                xy=(min_ttft_ratio, min_ttft_val), xytext=(30, -30), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')

    # 4. TPOT vs K ratio
    ax = axes[1, 0]
    baseline_tpot_mean = baseline_result['mean_tpot_ms'] if baseline_result else None
    baseline_tpot_p99 = baseline_result['p99_tpot_ms'] if baseline_result else None
    plot_metric_with_adaptive_axis(ax, k_ratios, mean_tpot, p99_tpot,
                                   baseline_tpot_mean, baseline_tpot_p99,
                                   'teal', 'TPOT', 'ms')
    ax.set_xlabel('K Ratio', fontsize=12)
    ax.set_ylabel('Mean TPOT (ms)', fontsize=12)
    ax.set_title('Time Per Output Token vs K Ratio', fontsize=14)
    ax.grid(True, alpha=0.3)

    min_tpot_idx = np.argmin(mean_tpot)
    min_tpot_ratio = k_ratios[min_tpot_idx]
    min_tpot_val = mean_tpot[min_tpot_idx]
    ax.annotate(f'Min: ratio={min_tpot_ratio}\n{min_tpot_val:.2f}ms',
                xy=(min_tpot_ratio, min_tpot_val), xytext=(30, -30), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')

    # 5. ITL vs K ratio
    ax = axes[1, 1]
    baseline_itl_mean = baseline_result['mean_itl_ms'] if baseline_result else None
    baseline_itl_p99 = baseline_result['p99_itl_ms'] if baseline_result else None
    plot_metric_with_adaptive_axis(ax, k_ratios, mean_itl, p99_itl,
                                   baseline_itl_mean, baseline_itl_p99,
                                   'brown', 'ITL', 'ms')
    ax.set_xlabel('K Ratio', fontsize=12)
    ax.set_ylabel('Mean ITL (ms)', fontsize=12)
    ax.set_title('Inter-Token Latency vs K Ratio', fontsize=14)
    ax.grid(True, alpha=0.3)

    min_itl_idx = np.argmin(mean_itl)
    min_itl_ratio = k_ratios[min_itl_idx]
    min_itl_val = mean_itl[min_itl_idx]
    ax.annotate(f'Min: ratio={min_itl_ratio}\n{min_itl_val:.2f}ms',
                xy=(min_itl_ratio, min_itl_val), xytext=(30, -30), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')

    # 6. Throughput vs TTFT Trade-off
    ax = axes[1, 2]
    scatter = ax.scatter(mean_ttft, throughputs, c=k_ratios, cmap='viridis',
                         s=100, edgecolors='black', linewidths=0.5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('K Ratio', fontsize=10)
    if baseline_result:
        ax.scatter(baseline_result['mean_ttft_ms'],
                   baseline_result['total_token_throughput'],
                   c='red', s=150, marker='*', edgecolors='black',
                   linewidths=1, label='Baseline', zorder=5)
    ax.set_xlabel('Mean TTFT (ms)', fontsize=12)
    ax.set_ylabel('Throughput (tokens/s)', fontsize=12)
    ax.set_title('Throughput vs TTFT Trade-off', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(w_pad=3.0, h_pad=2.0)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    if not no_show:
        plt.show()

    # 打印汇总表
    print("\n" + "=" * 105)
    print("BENCHMARK RESULTS SUMMARY (K Ratio, Adaptive N)")
    print("=" * 105)
    print(f"{'Ratio':>8} | {'Throughput':>12} | {'Output TPS':>12} | {'Req TPS':>10} | "
          f"{'TTFT Mean':>10} | {'TTFT P99':>10} | {'TPOT Mean':>10} | {'ITL Mean':>10}")
    print(f"{'':>8} | {'(tok/s)':>12} | {'(tok/s)':>12} | {'(req/s)':>10} | "
          f"{'(ms)':>10} | {'(ms)':>10} | {'(ms)':>10} | {'(ms)':>10}")
    print("-" * 105)

    if baseline_result:
        print(f"{'BASE':>8} | {baseline_result['total_token_throughput']:>12.0f} | "
              f"{baseline_result['output_throughput']:>12.0f} | "
              f"{baseline_result['request_throughput']:>10.1f} | "
              f"{baseline_result['mean_ttft_ms']:>10.0f} | "
              f"{baseline_result['p99_ttft_ms']:>10.0f} | "
              f"{baseline_result['mean_tpot_ms']:>10.2f} | "
              f"{baseline_result['mean_itl_ms']:>10.2f}")
        print("-" * 105)

    for r in k_ratios:
        res = kratio_results[r]
        marker = " <-- BEST TPS" if r == best_ratio else ""
        marker = " <-- MIN TTFT" if r == min_ttft_ratio else marker
        print(f"{r:>8.2f} | {res['total_token_throughput']:>12.0f} | "
              f"{res['output_throughput']:>12.0f} | "
              f"{res['request_throughput']:>10.1f} | "
              f"{res['mean_ttft_ms']:>10.0f} | "
              f"{res['p99_ttft_ms']:>10.0f} | "
              f"{res['mean_tpot_ms']:>10.2f} | "
              f"{res['mean_itl_ms']:>10.2f}{marker}")

    print("=" * 105)

    if baseline_result:
        print(f"\nBest Throughput ratio={best_ratio} vs Baseline:")
        tps_improvement = (best_tps - baseline_result['total_token_throughput']) / baseline_result['total_token_throughput'] * 100
        print(f"  Throughput improvement: {tps_improvement:+.1f}%")

        print(f"\nMin TTFT ratio={min_ttft_ratio} vs Baseline:")
        ttft_improvement = (min_ttft_val - baseline_result['mean_ttft_ms']) / baseline_result['mean_ttft_ms'] * 100
        print(f"  TTFT improvement: {ttft_improvement:+.1f}%")


def plot_schedule_mode(k_star_results: dict, baseline_metrics: dict, dynamic_results: dict,
                       output_path: str, no_show: bool, kratio_results: dict = None,
                       special_results: dict = None):
    """绘制 schedule stats 图表

    Args:
        special_results: 包含 'ratio_auto' 和 'dynamic' 模式的结果
    """
    kratio_results = kratio_results or {}
    special_results = special_results or {}

    # 如果只有 kratio 结果，没有固定 K* 结果，使用 kratio 数据
    if not k_star_results and kratio_results:
        print("\nPlotting K ratio results (adaptive N mode)...")
        plot_kratio_schedule_mode(kratio_results, baseline_metrics, output_path, no_show)
        return

    # 如果没有固定 K* 或 kratio 结果，但有 special_results，打印汇总
    if not k_star_results and not kratio_results and special_results:
        print("\n" + "=" * 100)
        print("SPECIAL MODE RESULTS SUMMARY (Schedule Stats)")
        print("=" * 100)
        print(f"{'Mode':<15} | {'Throughput':>12} | {'Prefill':>10} | {'Decode':>10} | {'Interval':>10} | {'#Scheds':>10} | {'#Preempt':>10}")
        print(f"{'':>15} | {'(tok/s)':>12} | {'(tok/s)':>10} | {'(tok/s)':>10} | {'(ms)':>10} | {'':>10} | {'(reqs)':>10}")
        print("-" * 100)

        if baseline_metrics:
            print(f"{'baseline':<15} | {baseline_metrics['overall_throughput_tps']:>12.0f} | "
                  f"{baseline_metrics['prefill_throughput_tps']:>10.0f} | "
                  f"{baseline_metrics['decode_throughput_tps']:>10.0f} | "
                  f"{baseline_metrics['schedule_interval_ms_mean']:>10.2f} | "
                  f"{baseline_metrics['num_schedules']:>10} | "
                  f"{baseline_metrics.get('total_preempted_reqs', 0):>10}")

        for name, metrics in special_results.items():
            print(f"{name:<15} | {metrics['overall_throughput_tps']:>12.0f} | "
                  f"{metrics['prefill_throughput_tps']:>10.0f} | "
                  f"{metrics['decode_throughput_tps']:>10.0f} | "
                  f"{metrics['schedule_interval_ms_mean']:>10.2f} | "
                  f"{metrics['num_schedules']:>10} | "
                  f"{metrics.get('total_preempted_reqs', 0):>10}")
        print("=" * 100)
        return

    k_stars = sorted(k_star_results.keys())

    # 提取各项指标
    throughputs = [k_star_results[k]["overall_throughput_tps"] for k in k_stars]
    prefill_tps = [k_star_results[k]["prefill_throughput_tps"] for k in k_stars]
    decode_tps = [k_star_results[k]["decode_throughput_tps"] for k in k_stars]
    schedule_intervals = [k_star_results[k]["schedule_interval_ms_mean"] for k in k_stars]
    tokens_per_schedule = [k_star_results[k]["tokens_per_schedule_mean"] for k in k_stars]
    num_schedules = [k_star_results[k]["num_schedules"] for k in k_stars]
    # Preemption metrics
    preempted_reqs = [k_star_results[k].get("total_preempted_reqs", 0) for k in k_stars]
    preempted_tokens = [k_star_results[k].get("total_preempted_tokens", 0) for k in k_stars]

    dynamic_colors = ['magenta', 'cyan', 'lime', 'yellow', 'pink', 'brown']

    # 增加图宽度，为双Y轴标签留出空间 (2x3 布局)
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    # 1. 总吞吐量 vs K*
    ax = axes[0, 0]
    ax.plot(k_stars, throughputs, 'b-o', linewidth=2, markersize=6, label='Fixed K*')
    if baseline_metrics:
        ax.axhline(y=baseline_metrics['overall_throughput_tps'], color='r',
                   linestyle='--', linewidth=2,
                   label=f"Baseline ({baseline_metrics['overall_throughput_tps']:.0f})")
    for i, (name, metrics) in enumerate(dynamic_results.items()):
        color = dynamic_colors[i % len(dynamic_colors)]
        ax.axhline(y=metrics['overall_throughput_tps'], color=color,
                   linestyle=':', linewidth=2, alpha=0.8,
                   label=f"Dynamic {name} ({metrics['overall_throughput_tps']:.0f})")
    # 绘制 special_results (ratio_auto, dynamic)
    special_colors = {'ratio_auto': 'green', 'dynamic': 'purple'}
    special_styles = {'ratio_auto': '-.', 'dynamic': ':'}
    for name, metrics in special_results.items():
        color = special_colors.get(name, 'gray')
        style = special_styles.get(name, '-.')
        display_name = {'ratio_auto': 'Ratio Auto (θ* auto)', 'dynamic': 'Dynamic (DP)'}.get(name, name)
        ax.axhline(y=metrics['overall_throughput_tps'], color=color,
                   linestyle=style, linewidth=2.5, alpha=0.9,
                   label=f"{display_name} ({metrics['overall_throughput_tps']:.0f})")
    ax.set_xlabel('K* Value', fontsize=12)
    ax.set_ylabel('Throughput (tokens/s)', fontsize=12)
    ax.set_title('Overall Throughput vs K*', fontsize=14)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # 标注最佳点 (使用 offset points 避免超出边界)
    best_idx = np.argmax(throughputs)
    best_k = k_stars[best_idx]
    best_tps = throughputs[best_idx]
    ax.annotate(f'Best: K*={best_k}\n{best_tps:.0f} tok/s',
                xy=(best_k, best_tps), xytext=(30, -30), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')

    # 2. Prefill vs Decode 吞吐量 (双 Y 轴)
    ax = axes[0, 1]
    ax.plot(k_stars, prefill_tps, 'g-o', linewidth=2, markersize=5, label='Prefill', alpha=0.8)
    if baseline_metrics:
        ax.axhline(y=baseline_metrics['prefill_throughput_tps'], color='g',
                   linestyle=':', alpha=0.5, label='Baseline Prefill')
    # 绘制 special_results (ratio_auto, dynamic) 的 Prefill
    for name, metrics in special_results.items():
        color = special_colors.get(name, 'gray')
        style = special_styles.get(name, '-.')
        display_name = {'ratio_auto': 'R.Auto', 'dynamic': 'Dynamic'}.get(name, name)
        ax.axhline(y=metrics['prefill_throughput_tps'], color=color,
                   linestyle=style, linewidth=2, alpha=0.7,
                   label=f"{display_name} Prefill ({metrics['prefill_throughput_tps']:.0f})")
    ax.set_xlabel('K* Value', fontsize=12)
    ax.set_ylabel('Prefill Throughput (tokens/s)', fontsize=12, color='g')
    ax.tick_params(axis='y', labelcolor='g')

    ax2 = ax.twinx()
    ax2.plot(k_stars, decode_tps, 'orange', linestyle='-', marker='s',
             linewidth=2, markersize=5, label='Decode', alpha=0.8)
    if baseline_metrics:
        ax2.axhline(y=baseline_metrics['decode_throughput_tps'], color='orange',
                    linestyle=':', alpha=0.5, label='Baseline Decode')
    # 绘制 special_results (ratio_auto, dynamic) 的 Decode
    special_decode_colors = {'ratio_auto': 'darkgreen', 'dynamic': 'indigo'}
    for name, metrics in special_results.items():
        color = special_decode_colors.get(name, 'gray')
        style = special_styles.get(name, '-.')
        display_name = {'ratio_auto': 'R.Auto', 'dynamic': 'Dynamic'}.get(name, name)
        ax2.axhline(y=metrics['decode_throughput_tps'], color=color,
                    linestyle=style, linewidth=2, alpha=0.7,
                    label=f"{display_name} Decode ({metrics['decode_throughput_tps']:.0f})")
    ax2.set_ylabel('Decode Throughput (tokens/s)', fontsize=12, color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    ax.set_title('Prefill vs Decode Throughput', fontsize=14)
    # 合并两个轴的图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # 标注最佳 Prefill 和 Decode 点 (使用 offset points 避免超出边界)
    best_prefill_idx = np.argmax(prefill_tps)
    best_decode_idx = np.argmax(decode_tps)
    ax.annotate(f'Best Prefill: K*={k_stars[best_prefill_idx]}',
                xy=(k_stars[best_prefill_idx], prefill_tps[best_prefill_idx]),
                xytext=(30, -20), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=9, color='green')
    ax2.annotate(f'Best Decode: K*={k_stars[best_decode_idx]}',
                 xy=(k_stars[best_decode_idx], decode_tps[best_decode_idx]),
                 xytext=(30, 20), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='orange'),
                 fontsize=9, color='orange')

    # 3. 调度间隔 + 每次调度 tokens 数 (双 Y 轴, 合并为一个图)
    ax = axes[1, 0]
    ax.plot(k_stars, schedule_intervals, 'purple', linestyle='-', marker='d',
            linewidth=2, markersize=6, label='Schedule Interval')
    if baseline_metrics:
        ax.axhline(y=baseline_metrics['schedule_interval_ms_mean'], color='purple',
                   linestyle='--', linewidth=2, alpha=0.5,
                   label=f"Baseline Interval ({baseline_metrics['schedule_interval_ms_mean']:.2f} ms)")
    # 绘制 special_results (ratio_auto, dynamic) 的 Schedule Interval
    special_interval_colors = {'ratio_auto': 'magenta', 'dynamic': 'darkviolet'}
    for name, metrics in special_results.items():
        color = special_interval_colors.get(name, 'gray')
        style = special_styles.get(name, '-.')
        display_name = {'ratio_auto': 'R.Auto', 'dynamic': 'Dynamic'}.get(name, name)
        ax.axhline(y=metrics['schedule_interval_ms_mean'], color=color,
                   linestyle=style, linewidth=2, alpha=0.7,
                   label=f"{display_name} ({metrics['schedule_interval_ms_mean']:.2f} ms)")
    ax.set_xlabel('K* Value', fontsize=12)
    ax.set_ylabel('Schedule Interval (ms)', fontsize=12, color='purple')
    ax.tick_params(axis='y', labelcolor='purple')

    ax2 = ax.twinx()
    ax2.plot(k_stars, tokens_per_schedule, 'teal', linestyle='-', marker='^',
             linewidth=2, markersize=6, label='Tokens per Schedule')
    if baseline_metrics:
        ax2.axhline(y=baseline_metrics['tokens_per_schedule_mean'], color='teal',
                    linestyle='--', linewidth=2, alpha=0.5,
                    label=f"Baseline Tok/Sched ({baseline_metrics['tokens_per_schedule_mean']:.1f})")
    # 绘制 special_results (ratio_auto, dynamic) 的 Tokens per Schedule
    special_toksched_colors = {'ratio_auto': 'darkcyan', 'dynamic': 'cadetblue'}
    for name, metrics in special_results.items():
        color = special_toksched_colors.get(name, 'gray')
        style = special_styles.get(name, '-.')
        display_name = {'ratio_auto': 'R.Auto', 'dynamic': 'Dynamic'}.get(name, name)
        ax2.axhline(y=metrics['tokens_per_schedule_mean'], color=color,
                    linestyle=style, linewidth=2, alpha=0.7,
                    label=f"{display_name} ({metrics['tokens_per_schedule_mean']:.1f})")
    ax2.set_ylabel('Tokens per Schedule', fontsize=12, color='teal')
    ax2.tick_params(axis='y', labelcolor='teal')

    ax.set_title('Schedule Interval & Tokens per Schedule vs K*', fontsize=14)
    # 合并两个轴的图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # 标注最佳点 (最小间隔和最大 tokens/schedule, 使用 offset points 避免超出边界)
    min_interval_idx = np.argmin(schedule_intervals)
    max_tok_sched_idx = np.argmax(tokens_per_schedule)
    ax.annotate(f'Min Interval: K*={k_stars[min_interval_idx]}',
                xy=(k_stars[min_interval_idx], schedule_intervals[min_interval_idx]),
                xytext=(30, 20), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='purple'),
                fontsize=9, color='purple')
    ax2.annotate(f'Max Tok/Sched: K*={k_stars[max_tok_sched_idx]}',
                 xy=(k_stars[max_tok_sched_idx], tokens_per_schedule[max_tok_sched_idx]),
                 xytext=(30, -20), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='teal'),
                 fontsize=9, color='teal')

    # 4. 调度总次数 vs K*
    ax = axes[1, 1]
    ax.plot(k_stars, num_schedules, 'brown', linestyle='-', marker='v',
            linewidth=2, markersize=6, label='Fixed K*')
    if baseline_metrics:
        ax.axhline(y=baseline_metrics['num_schedules'], color='r',
                   linestyle='--', linewidth=2,
                   label=f"Baseline ({baseline_metrics['num_schedules']})")
    for i, (name, metrics) in enumerate(dynamic_results.items()):
        color = dynamic_colors[i % len(dynamic_colors)]
        ax.axhline(y=metrics['num_schedules'], color=color,
                   linestyle=':', linewidth=2, alpha=0.8, label=f"Dynamic {name}")
    # 绘制 special_results (ratio_auto, dynamic)
    for name, metrics in special_results.items():
        color = special_colors.get(name, 'gray')
        style = special_styles.get(name, '-.')
        display_name = {'ratio_auto': 'Ratio Auto', 'dynamic': 'Dynamic (DP)'}.get(name, name)
        ax.axhline(y=metrics['num_schedules'], color=color,
                   linestyle=style, linewidth=2.5, alpha=0.9,
                   label=f"{display_name} ({metrics['num_schedules']})")
    ax.set_xlabel('K* Value', fontsize=12)
    ax.set_ylabel('Number of Schedules', fontsize=12)
    ax.set_title('Total Number of Schedules vs K*', fontsize=14)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # 标注最少调度次数的点 (使用 offset points 避免超出边界)
    min_sched_idx = np.argmin(num_schedules)
    min_sched_k = k_stars[min_sched_idx]
    min_sched_val = num_schedules[min_sched_idx]
    ax.annotate(f'Min: K*={min_sched_k}\n{min_sched_val}',
                xy=(min_sched_k, min_sched_val), xytext=(30, 20), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')

    # 5. Preemption 统计 (双 Y 轴: 请求数 & Token 数)
    ax = axes[0, 2]
    has_preemption = any(p > 0 for p in preempted_reqs)
    has_special_preemption = any(m.get('total_preempted_reqs', 0) > 0 for m in special_results.values())
    if has_preemption or has_special_preemption:
        ax.plot(k_stars, preempted_reqs, 'red', linestyle='-', marker='x',
                linewidth=2, markersize=6, label='Preempted Requests')
        # 绘制 special_results (ratio_auto, dynamic) 的 Preemption
        for name, metrics in special_results.items():
            color = special_colors.get(name, 'gray')
            style = special_styles.get(name, '-.')
            display_name = {'ratio_auto': 'R.Auto', 'dynamic': 'Dynamic'}.get(name, name)
            preempt_reqs = metrics.get('total_preempted_reqs', 0)
            ax.axhline(y=preempt_reqs, color=color,
                       linestyle=style, linewidth=2, alpha=0.7,
                       label=f"{display_name} Reqs ({preempt_reqs})")
        ax.set_ylabel('Preempted Requests', fontsize=12, color='red')
        ax.tick_params(axis='y', labelcolor='red')

        ax2 = ax.twinx()
        ax2.plot(k_stars, preempted_tokens, 'darkred', linestyle='--', marker='+',
                 linewidth=2, markersize=6, label='Preempted Tokens')
        # 绘制 special_results 的 Preempted Tokens
        special_preempt_tok_colors = {'ratio_auto': 'darkgreen', 'dynamic': 'indigo'}
        for name, metrics in special_results.items():
            color = special_preempt_tok_colors.get(name, 'gray')
            style = special_styles.get(name, '-.')
            display_name = {'ratio_auto': 'R.Auto', 'dynamic': 'Dynamic'}.get(name, name)
            preempt_toks = metrics.get('total_preempted_tokens', 0)
            ax2.axhline(y=preempt_toks, color=color,
                        linestyle=style, linewidth=2, alpha=0.7,
                        label=f"{display_name} Toks ({preempt_toks})")
        ax2.set_ylabel('Preempted Tokens', fontsize=12, color='darkred')
        ax2.tick_params(axis='y', labelcolor='darkred')

        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='best')

        # 标注最大 preemption 点
        if max(preempted_reqs) > 0:
            max_preempt_idx = np.argmax(preempted_reqs)
            ax.annotate(f'Max: K*={k_stars[max_preempt_idx]}\n{preempted_reqs[max_preempt_idx]} reqs',
                        xy=(k_stars[max_preempt_idx], preempted_reqs[max_preempt_idx]),
                        xytext=(30, -20), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=9, color='red')
    else:
        ax.text(0.5, 0.5, 'No Preemption\nDetected', transform=ax.transAxes,
                fontsize=14, ha='center', va='center', color='green')
        ax.set_ylabel('Preempted Requests', fontsize=12)

    ax.set_xlabel('K* Value', fontsize=12)
    ax.set_title('Preemption Statistics vs K*', fontsize=14)
    ax.grid(True, alpha=0.3)

    # 6. Prefill Overhead 分析 (实际 prefill vs decode tokens)
    ax = axes[1, 2]
    actual_prefill = [k_star_results[k]["total_prefill_tokens"] for k in k_stars]
    actual_decode = [k_star_results[k]["total_decode_tokens"] for k in k_stars]
    # 计算 overhead: 如果 prefill > decode，说明有重复 prefill
    overhead_pct = [(p - d) / d * 100 if d > 0 else 0 for p, d in zip(actual_prefill, actual_decode)]

    bar_width = (k_stars[-1] - k_stars[0]) / len(k_stars) * 0.8 if len(k_stars) > 1 else 50
    ax.bar(k_stars, overhead_pct, width=bar_width,
           color=['red' if o > 0 else 'green' for o in overhead_pct], alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # 绘制 special_results (ratio_auto, dynamic) 的 Prefill Overhead
    for name, metrics in special_results.items():
        color = special_colors.get(name, 'gray')
        style = special_styles.get(name, '-.')
        display_name = {'ratio_auto': 'R.Auto', 'dynamic': 'Dynamic'}.get(name, name)
        sp_prefill = metrics.get('total_prefill_tokens', 0)
        sp_decode = metrics.get('total_decode_tokens', 0)
        sp_overhead = (sp_prefill - sp_decode) / sp_decode * 100 if sp_decode > 0 else 0
        ax.axhline(y=sp_overhead, color=color,
                   linestyle=style, linewidth=2.5, alpha=0.9,
                   label=f"{display_name} ({sp_overhead:.1f}%)")

    ax.set_xlabel('K* Value', fontsize=12)
    ax.set_ylabel('Prefill Overhead (%)', fontsize=12)
    ax.set_title('Prefill Overhead vs K*\n(Prefill - Decode) / Decode', fontsize=14)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # 标注最大 overhead
    if any(o > 0 for o in overhead_pct):
        max_overhead_idx = np.argmax(overhead_pct)
        if overhead_pct[max_overhead_idx] > 0:
            ax.annotate(f'Max: {overhead_pct[max_overhead_idx]:.1f}%\nK*={k_stars[max_overhead_idx]}',
                        xy=(k_stars[max_overhead_idx], overhead_pct[max_overhead_idx]),
                        xytext=(30, -20), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=9, color='red')

    # 增加子图间距，避免双Y轴标签重叠
    plt.tight_layout(w_pad=3.0, h_pad=2.0)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    if not no_show:
        plt.show()

    # Dynamic K* 轨迹图
    if dynamic_results:
        trajectory_data = {
            name: metrics for name, metrics in dynamic_results.items()
            if metrics.get('k_star_trajectory')
        }
        if trajectory_data:
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            for i, (name, metrics) in enumerate(trajectory_data.items()):
                color = dynamic_colors[i % len(dynamic_colors)]
                times = metrics['k_star_times']
                k_values = metrics['k_star_trajectory']
                ax2.plot(times, k_values, color=color, linewidth=1.5,
                         alpha=0.8, label=f"Dynamic {name}")
            ax2.set_xlabel('Time (s)', fontsize=12)
            ax2.set_ylabel('K* Value', fontsize=12)
            ax2.set_title('Dynamic K* Trajectory Over Time', fontsize=14)
            ax2.legend(fontsize=10, loc='best')
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            trajectory_path = output_path.replace('.png', '_trajectory.png')
            plt.savefig(trajectory_path, dpi=150, bbox_inches='tight')
            print(f"K* trajectory plot saved to: {trajectory_path}")
            if not no_show:
                plt.show()

    # 打印汇总表
    print("\n" + "=" * 120)
    print("K* SWEEP RESULTS SUMMARY (Schedule Stats)")
    print("=" * 120)
    print(f"{'K*':>6} | {'Throughput':>12} | {'Prefill':>10} | {'Decode':>10} | {'Interval':>10} | {'Tok/Sched':>10} | {'#Scheds':>10} | {'#Preempt':>10} | {'Overhead':>8}")
    print(f"{'':>6} | {'(tok/s)':>12} | {'(tok/s)':>10} | {'(tok/s)':>10} | {'(ms)':>10} | {'':>10} | {'':>10} | {'(reqs)':>10} | {'(%)':>8}")
    print("-" * 120)

    if baseline_metrics:
        base_overhead = ((baseline_metrics['total_prefill_tokens'] - baseline_metrics['total_decode_tokens'])
                        / baseline_metrics['total_decode_tokens'] * 100
                        if baseline_metrics.get('total_decode_tokens', 0) > 0 else 0)
        print(f"{'BASE':>6} | {baseline_metrics['overall_throughput_tps']:>12.0f} | "
              f"{baseline_metrics['prefill_throughput_tps']:>10.0f} | "
              f"{baseline_metrics['decode_throughput_tps']:>10.0f} | "
              f"{baseline_metrics['schedule_interval_ms_mean']:>10.2f} | "
              f"{baseline_metrics['tokens_per_schedule_mean']:>10.1f} | "
              f"{baseline_metrics['num_schedules']:>10} | "
              f"{baseline_metrics.get('total_preempted_reqs', 0):>10} | "
              f"{base_overhead:>8.1f}")

    # 打印 special_results (ratio_auto, dynamic)
    for name, m in special_results.items():
        overhead = (m['total_prefill_tokens'] - m['total_decode_tokens']) / m['total_decode_tokens'] * 100 if m['total_decode_tokens'] > 0 else 0
        display_name = {'ratio_auto': 'R_AUTO', 'dynamic': 'DYNMC'}.get(name, name[:6].upper())
        print(f"{display_name:>6} | {m['overall_throughput_tps']:>12.0f} | "
              f"{m['prefill_throughput_tps']:>10.0f} | "
              f"{m['decode_throughput_tps']:>10.0f} | "
              f"{m['schedule_interval_ms_mean']:>10.2f} | "
              f"{m['tokens_per_schedule_mean']:>10.1f} | "
              f"{m['num_schedules']:>10} | "
              f"{m.get('total_preempted_reqs', 0):>10} | "
              f"{overhead:>8.1f}")

    if baseline_metrics or special_results:
        print("-" * 120)

    for k in k_stars:
        m = k_star_results[k]
        marker = " <-- BEST TPS" if k == best_k else ""
        marker = " <-- MIN SCHED" if k == min_sched_k else marker
        overhead = (m['total_prefill_tokens'] - m['total_decode_tokens']) / m['total_decode_tokens'] * 100 if m['total_decode_tokens'] > 0 else 0
        print(f"{k:>6} | {m['overall_throughput_tps']:>12.0f} | "
              f"{m['prefill_throughput_tps']:>10.0f} | "
              f"{m['decode_throughput_tps']:>10.0f} | "
              f"{m['schedule_interval_ms_mean']:>10.2f} | "
              f"{m['tokens_per_schedule_mean']:>10.1f} | "
              f"{m['num_schedules']:>10} | "
              f"{m.get('total_preempted_reqs', 0):>10} | "
              f"{overhead:>8.1f}{marker}")

    print("=" * 120)

    if baseline_metrics:
        print(f"\nBest Throughput K*={best_k} vs Baseline:")
        improvement = (best_tps - baseline_metrics['overall_throughput_tps']) / baseline_metrics['overall_throughput_tps'] * 100
        print(f"  Throughput improvement: {improvement:+.1f}%")
        print(f"\nMin Schedules K*={min_sched_k} vs Baseline:")
        sched_reduction = (min_sched_val - baseline_metrics['num_schedules']) / baseline_metrics['num_schedules'] * 100
        print(f"  Schedule count change: {sched_reduction:+.1f}%")

    # Preemption 汇总
    total_preemptions = sum(preempted_reqs)
    if total_preemptions > 0:
        print(f"\n--- Preemption Summary ---")
        print(f"  Total preempted requests across all K* values: {total_preemptions}")
        print(f"  K* with most preemptions: {k_stars[np.argmax(preempted_reqs)]} ({max(preempted_reqs)} reqs)")
        print(f"  K* with least preemptions: {k_stars[np.argmin(preempted_reqs)]} ({min(preempted_reqs)} reqs)")


def needs_dual_axis(mean_vals: list, p99_vals: list, threshold: float = 3.0) -> bool:
    """判断是否需要双 Y 轴（当 P99 和 Mean 差距过大时）"""
    if not mean_vals or not p99_vals:
        return False
    max_mean = max(mean_vals) if mean_vals else 1
    max_p99 = max(p99_vals) if p99_vals else 1
    min_mean = min(mean_vals) if mean_vals else 0
    # 如果 P99 最大值超过 Mean 最大值的 threshold 倍，使用双轴
    if max_mean > 0 and max_p99 / max_mean > threshold:
        return True
    # 或者如果 P99 的范围会导致 Mean 的变化被压缩得看不清
    mean_range = max(mean_vals) - min(mean_vals) if mean_vals else 0
    p99_range = max(p99_vals) - min(p99_vals) if p99_vals else 0
    if mean_range > 0 and p99_range / mean_range > threshold * 2:
        return True
    return False


def plot_metric_with_adaptive_axis(ax, k_stars: list, mean_vals: list, p99_vals: list,
                                    baseline_mean: float | None, baseline_p99: float | None,
                                    color: str, metric_name: str, unit: str = "ms"):
    """
    绘制指标图，自动判断是否需要双 Y 轴
    返回: (需要双轴时的第二个轴或None, 最小mean值的索引, 最小mean值)
    """
    use_dual = needs_dual_axis(mean_vals, p99_vals)

    # 绘制 Mean 线
    ax.plot(k_stars, mean_vals, color, linestyle='-', marker='d' if 'TTFT' in metric_name else ('^' if 'TPOT' in metric_name else 'v'),
            linewidth=2, markersize=6, label=f'Mean {metric_name}')

    if use_dual:
        # P99 差距过大，使用第二个 Y 轴
        ax2 = ax.twinx()
        lighter_color = color if color != 'brown' else 'orangered'
        ax2.plot(k_stars, p99_vals, lighter_color, linestyle='--', marker='o',
                 linewidth=1.5, markersize=4, alpha=0.8, label=f'P99 {metric_name}')
        ax2.set_ylabel(f'P99 {metric_name} ({unit})', fontsize=12, color=lighter_color)
        ax2.tick_params(axis='y', labelcolor=lighter_color)

        # Baseline 线
        if baseline_mean is not None:
            ax.axhline(y=baseline_mean, color='r',
                       linestyle='-', linewidth=2, alpha=0.7,
                       label=f"Baseline Mean ({baseline_mean:.1f}{unit})")
        if baseline_p99 is not None:
            ax2.axhline(y=baseline_p99, color='r',
                        linestyle='--', linewidth=1.5, alpha=0.5,
                        label=f"Baseline P99 ({baseline_p99:.1f}{unit})")

        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='best')

        return ax2
    else:
        # 正常单轴绘制
        ax.plot(k_stars, p99_vals, color, linestyle='--', marker='d' if 'TTFT' in metric_name else ('^' if 'TPOT' in metric_name else 'v'),
                linewidth=1.5, markersize=4, alpha=0.6, label=f'P99 {metric_name}')

        if baseline_mean is not None:
            ax.axhline(y=baseline_mean, color='r',
                       linestyle='-', linewidth=2, alpha=0.7,
                       label=f"Baseline Mean ({baseline_mean:.1f}{unit})")
        if baseline_p99 is not None:
            ax.axhline(y=baseline_p99, color='r',
                       linestyle='--', linewidth=1.5, alpha=0.5,
                       label=f"Baseline P99 ({baseline_p99:.1f}{unit})")
        ax.legend(fontsize=9)

        return None


def plot_n_trajectory(kratio_results: dict, output_path: str, no_show: bool):
    """绘制 N (batch size) 变动轨迹图，展示自适应 N 如何随时间变化"""
    if not kratio_results:
        print("No K ratio results with N trajectory data!")
        return

    # 检查是否有 N 轨迹数据
    has_trajectory = any(
        kratio_results[r].get("n_trajectory") and kratio_results[r].get("n_times")
        for r in kratio_results
    )
    if not has_trajectory:
        print("No N trajectory data found in results!")
        return

    k_ratios = sorted(kratio_results.keys())
    num_ratios = len(k_ratios)

    # 根据 ratio 数量决定布局
    if num_ratios <= 3:
        nrows, ncols = 1, num_ratios
        figsize = (6 * num_ratios, 5)
    elif num_ratios <= 6:
        nrows, ncols = 2, (num_ratios + 1) // 2
        figsize = (6 * ncols, 10)
    else:
        nrows, ncols = 3, (num_ratios + 2) // 3
        figsize = (6 * ncols, 15)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    colors = plt.cm.viridis(np.linspace(0, 1, num_ratios))

    for idx, ratio in enumerate(k_ratios):
        metrics = kratio_results[ratio]
        ax = axes[idx]

        n_trajectory = metrics.get("n_trajectory", [])
        n_times = metrics.get("n_times", [])
        k_star_trajectory = metrics.get("k_star_trajectory", [])
        k_star_times = metrics.get("k_star_times", [])

        if not n_trajectory or not n_times:
            ax.text(0.5, 0.5, 'No N trajectory\ndata available',
                    transform=ax.transAxes, fontsize=12,
                    ha='center', va='center', color='gray')
            ax.set_title(f'K Ratio = {ratio}', fontsize=12, fontweight='bold')
            continue

        # 绘制 N 轨迹
        ax.plot(n_times, n_trajectory, 'b-', linewidth=1.5, alpha=0.8, label='N (batch size)')
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('N (batch size)', fontsize=11, color='blue')
        ax.tick_params(axis='y', labelcolor='blue')

        # 如果有 k* 轨迹，用双 Y 轴显示
        if k_star_trajectory and k_star_times:
            ax2 = ax.twinx()
            ax2.plot(k_star_times, k_star_trajectory, 'r-', linewidth=1.5, alpha=0.6, label='k*')
            ax2.set_ylabel('k*', fontsize=11, color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            # 合并图例
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
        else:
            ax.legend(fontsize=8, loc='upper right')

        ax.set_title(f'K Ratio = {ratio} (k* = {ratio} × N)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 标注 N 的统计信息
        n_mean = np.mean(n_trajectory)
        n_min = np.min(n_trajectory)
        n_max = np.max(n_trajectory)
        n_std = np.std(n_trajectory)
        stats_text = f'N: mean={n_mean:.0f}, range=[{n_min:.0f}, {n_max:.0f}], std={n_std:.1f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 隐藏多余的子图
    for idx in range(num_ratios, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('N (Batch Size) Trajectory Over Time - Adaptive N Mode', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    trajectory_path = output_path.replace('.png', '_n_trajectory.png')
    plt.savefig(trajectory_path, dpi=150, bbox_inches='tight')
    print(f"\nN trajectory plot saved to: {trajectory_path}")

    if not no_show:
        plt.show()

    # 打印 N 变动统计
    print("\n" + "=" * 80)
    print("N (BATCH SIZE) TRAJECTORY SUMMARY")
    print("=" * 80)
    print(f"{'Ratio':>8} | {'N Mean':>10} | {'N Min':>10} | {'N Max':>10} | {'N Std':>10} | {'% Change':>10}")
    print("-" * 80)

    for ratio in k_ratios:
        metrics = kratio_results[ratio]
        n_trajectory = metrics.get("n_trajectory", [])
        if n_trajectory:
            n_mean = np.mean(n_trajectory)
            n_min = np.min(n_trajectory)
            n_max = np.max(n_trajectory)
            n_std = np.std(n_trajectory)
            pct_change = (n_max - n_min) / n_mean * 100 if n_mean > 0 else 0
            print(f"{ratio:>8.2f} | {n_mean:>10.0f} | {n_min:>10.0f} | {n_max:>10.0f} | {n_std:>10.1f} | {pct_change:>9.1f}%")
        else:
            print(f"{ratio:>8.2f} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10}")

    print("=" * 80)


def plot_kratio_all_mode(kratio_results_sched: dict, baseline_metrics: dict,
                          kratio_results_bench: dict, baseline_result: dict,
                          output_path: str, no_show: bool):
    """绘制 K ratio (自适应 N) 的组合图表 (schedule + bench)"""
    k_ratios_sched = sorted(kratio_results_sched.keys()) if kratio_results_sched else []
    k_ratios_bench = sorted(kratio_results_bench.keys()) if kratio_results_bench else []

    has_schedule = bool(k_ratios_sched)
    has_bench = bool(k_ratios_bench)

    if not has_schedule and not has_bench:
        print("No K ratio data available for plotting!")
        return

    # 创建 4x3 的子图布局
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))

    # ==================== Row 1: Schedule plots ====================
    if has_schedule:
        throughputs_sched = [kratio_results_sched[r]["overall_throughput_tps"] for r in k_ratios_sched]
        prefill_tps = [kratio_results_sched[r]["prefill_throughput_tps"] for r in k_ratios_sched]
        decode_tps = [kratio_results_sched[r]["decode_throughput_tps"] for r in k_ratios_sched]
        schedule_intervals = [kratio_results_sched[r]["schedule_interval_ms_mean"] for r in k_ratios_sched]
        tokens_per_schedule = [kratio_results_sched[r]["tokens_per_schedule_mean"] for r in k_ratios_sched]
        num_schedules = [kratio_results_sched[r]["num_schedules"] for r in k_ratios_sched]
        preempted_reqs = [kratio_results_sched[r].get("total_preempted_reqs", 0) for r in k_ratios_sched]
        preempted_tokens = [kratio_results_sched[r].get("total_preempted_tokens", 0) for r in k_ratios_sched]

        # 1. 总吞吐量 vs K ratio (Schedule)
        ax = axes[0, 0]
        ax.plot(k_ratios_sched, throughputs_sched, 'b-o', linewidth=2, markersize=6, label='K Ratio')
        if baseline_metrics:
            ax.axhline(y=baseline_metrics['overall_throughput_tps'], color='r',
                       linestyle='--', linewidth=2,
                       label=f"Baseline ({baseline_metrics['overall_throughput_tps']:.0f})")
        ax.set_xlabel('K Ratio (k* = ratio × N)', fontsize=11)
        ax.set_ylabel('Throughput (tokens/s)', fontsize=11)
        ax.set_title('[Schedule] Overall Throughput', fontsize=12, fontweight='bold')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

        best_idx_sched = np.argmax(throughputs_sched)
        best_ratio_sched = k_ratios_sched[best_idx_sched]
        best_tps_sched = throughputs_sched[best_idx_sched]
        ax.annotate(f'Best: ratio={best_ratio_sched}\n{best_tps_sched:.0f} tok/s',
                    xy=(best_ratio_sched, best_tps_sched), xytext=(30, -30), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='green'),
                    fontsize=9, color='green')

        # 2. Prefill vs Decode 吞吐量
        ax = axes[0, 1]
        ax.plot(k_ratios_sched, prefill_tps, 'g-o', linewidth=2, markersize=5, label='Prefill', alpha=0.8)
        ax.set_xlabel('K Ratio', fontsize=11)
        ax.set_ylabel('Prefill Throughput (tok/s)', fontsize=11, color='g')
        ax.tick_params(axis='y', labelcolor='g')
        ax2 = ax.twinx()
        ax2.plot(k_ratios_sched, decode_tps, 'orange', linestyle='-', marker='s',
                 linewidth=2, markersize=5, label='Decode', alpha=0.8)
        ax2.set_ylabel('Decode Throughput (tok/s)', fontsize=11, color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax.set_title('[Schedule] Prefill vs Decode', fontsize=12, fontweight='bold')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

        # 3. 调度间隔 + 每次调度 tokens 数
        ax = axes[0, 2]
        ax.plot(k_ratios_sched, schedule_intervals, 'purple', linestyle='-', marker='d',
                linewidth=2, markersize=6, label='Schedule Interval')
        ax.set_xlabel('K Ratio', fontsize=11)
        ax.set_ylabel('Schedule Interval (ms)', fontsize=11, color='purple')
        ax.tick_params(axis='y', labelcolor='purple')
        ax2 = ax.twinx()
        ax2.plot(k_ratios_sched, tokens_per_schedule, 'teal', linestyle='-', marker='^',
                 linewidth=2, markersize=6, label='Tokens per Schedule')
        ax2.set_ylabel('Tokens per Schedule', fontsize=11, color='teal')
        ax2.tick_params(axis='y', labelcolor='teal')
        ax.set_title('[Schedule] Interval & Tok/Sched', fontsize=12, fontweight='bold')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

        # 4. 调度总次数
        ax = axes[1, 0]
        ax.plot(k_ratios_sched, num_schedules, 'brown', linestyle='-', marker='v',
                linewidth=2, markersize=6, label='K Ratio')
        if baseline_metrics:
            ax.axhline(y=baseline_metrics['num_schedules'], color='r',
                       linestyle='--', linewidth=2,
                       label=f"Baseline ({baseline_metrics['num_schedules']})")
        ax.set_xlabel('K Ratio', fontsize=11)
        ax.set_ylabel('Number of Schedules', fontsize=11)
        ax.set_title('[Schedule] Total Schedules', fontsize=12, fontweight='bold')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

        # 5. Preemption 统计
        ax = axes[1, 1]
        has_preemption = any(p > 0 for p in preempted_reqs)
        has_baseline_preemption = baseline_metrics and baseline_metrics.get('total_preempted_reqs', 0) > 0
        if has_preemption or has_baseline_preemption:
            # K Ratio 数据: 红色实线
            ax.plot(k_ratios_sched, preempted_reqs, 'red', linestyle='-', marker='x',
                    linewidth=2, markersize=6, label='Preempted Requests')
            ax.set_ylabel('Preempted Requests', fontsize=11, color='red')
            ax.tick_params(axis='y', labelcolor='red')
            # Baseline: 橙色虚线 (区分于红色)
            if baseline_metrics:
                ax.axhline(y=baseline_metrics.get('total_preempted_reqs', 0), color='orange',
                           linestyle='--', linewidth=2,
                           label=f"Baseline Reqs ({baseline_metrics.get('total_preempted_reqs', 0)})")
            ax2 = ax.twinx()
            # K Ratio 数据: 蓝色实线 (区分于红色)
            ax2.plot(k_ratios_sched, preempted_tokens, 'blue', linestyle='-', marker='+',
                     linewidth=2, markersize=6, label='Preempted Tokens')
            ax2.set_ylabel('Preempted Tokens', fontsize=11, color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
            # Baseline: 青色虚线 (区分于蓝色)
            if baseline_metrics:
                ax2.axhline(y=baseline_metrics.get('total_preempted_tokens', 0), color='cyan',
                            linestyle='--', linewidth=2,
                            label=f"Baseline Tokens ({baseline_metrics.get('total_preempted_tokens', 0)})")
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='best')
        else:
            ax.text(0.5, 0.5, 'No Preemption\nDetected', transform=ax.transAxes,
                    fontsize=12, ha='center', va='center', color='green')
        ax.set_xlabel('K Ratio', fontsize=11)
        ax.set_title('[Schedule] Preemption Stats', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 6. Prefill Overhead
        ax = axes[1, 2]
        actual_prefill = [kratio_results_sched[r]["total_prefill_tokens"] for r in k_ratios_sched]
        actual_decode = [kratio_results_sched[r]["total_decode_tokens"] for r in k_ratios_sched]
        overhead_pct = [(p - d) / d * 100 if d > 0 else 0 for p, d in zip(actual_prefill, actual_decode)]
        bar_width = (k_ratios_sched[-1] - k_ratios_sched[0]) / len(k_ratios_sched) * 0.8 if len(k_ratios_sched) > 1 else 0.1
        ax.bar(k_ratios_sched, overhead_pct, width=bar_width,
               color=['red' if o > 0 else 'green' for o in overhead_pct], alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('K Ratio', fontsize=11)
        ax.set_ylabel('Prefill Overhead (%)', fontsize=11)
        ax.set_title('[Schedule] Prefill Overhead', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    else:
        for r in range(2):
            for c in range(3):
                axes[r, c].set_visible(False)

    # ==================== Row 2-3: Bench plots ====================
    if has_bench:
        throughputs_bench = [kratio_results_bench[r]["total_token_throughput"] for r in k_ratios_bench]
        output_tps = [kratio_results_bench[r]["output_throughput"] for r in k_ratios_bench]
        request_tps = [kratio_results_bench[r]["request_throughput"] for r in k_ratios_bench]
        mean_ttft = [kratio_results_bench[r]["mean_ttft_ms"] for r in k_ratios_bench]
        p99_ttft = [kratio_results_bench[r]["p99_ttft_ms"] for r in k_ratios_bench]
        mean_tpot = [kratio_results_bench[r]["mean_tpot_ms"] for r in k_ratios_bench]
        p99_tpot = [kratio_results_bench[r]["p99_tpot_ms"] for r in k_ratios_bench]
        mean_itl = [kratio_results_bench[r]["mean_itl_ms"] for r in k_ratios_bench]
        p99_itl = [kratio_results_bench[r]["p99_itl_ms"] for r in k_ratios_bench]

        # 1. Total Token Throughput (Bench)
        ax = axes[2, 0]
        ax.plot(k_ratios_bench, throughputs_bench, 'b-o', linewidth=2, markersize=6, label='K Ratio')
        if baseline_result:
            ax.axhline(y=baseline_result['total_token_throughput'], color='r',
                       linestyle='--', linewidth=2,
                       label=f"Baseline ({baseline_result['total_token_throughput']:.0f})")
        ax.set_xlabel('K Ratio', fontsize=11)
        ax.set_ylabel('Throughput (tokens/s)', fontsize=11)
        ax.set_title('[Bench] Total Throughput', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

        best_idx_bench = np.argmax(throughputs_bench)
        best_ratio_bench = k_ratios_bench[best_idx_bench]
        best_tps_bench = throughputs_bench[best_idx_bench]
        ax.annotate(f'Best: ratio={best_ratio_bench}\n{best_tps_bench:.0f} tok/s',
                    xy=(best_ratio_bench, best_tps_bench), xytext=(30, -30), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='green'),
                    fontsize=9, color='green')

        # 2. Output & Request Throughput
        ax = axes[2, 1]
        ax.plot(k_ratios_bench, output_tps, 'g-o', linewidth=2, markersize=5, label='Output (tok/s)', alpha=0.8)
        ax2 = ax.twinx()
        ax2.plot(k_ratios_bench, request_tps, 'orange', linestyle='-', marker='s',
                 linewidth=2, markersize=5, label='Request (req/s)', alpha=0.8)
        ax.set_xlabel('K Ratio', fontsize=11)
        ax.set_ylabel('Output Throughput (tok/s)', fontsize=11, color='g')
        ax2.set_ylabel('Request Throughput (req/s)', fontsize=11, color='orange')
        ax.set_title('[Bench] Output & Request TPS', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax2.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        # 3. TTFT
        ax = axes[2, 2]
        baseline_ttft_mean = baseline_result['mean_ttft_ms'] if baseline_result else None
        baseline_ttft_p99 = baseline_result['p99_ttft_ms'] if baseline_result else None
        plot_metric_with_adaptive_axis(ax, k_ratios_bench, mean_ttft, p99_ttft,
                                       baseline_ttft_mean, baseline_ttft_p99, 'purple', 'TTFT', 'ms')
        ax.set_xlabel('K Ratio', fontsize=11)
        ax.set_ylabel('Mean TTFT (ms)', fontsize=11)
        ax.set_title('[Bench] TTFT', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 4. TPOT
        ax = axes[3, 0]
        baseline_tpot_mean = baseline_result['mean_tpot_ms'] if baseline_result else None
        baseline_tpot_p99 = baseline_result['p99_tpot_ms'] if baseline_result else None
        plot_metric_with_adaptive_axis(ax, k_ratios_bench, mean_tpot, p99_tpot,
                                       baseline_tpot_mean, baseline_tpot_p99, 'teal', 'TPOT', 'ms')
        ax.set_xlabel('K Ratio', fontsize=11)
        ax.set_ylabel('Mean TPOT (ms)', fontsize=11)
        ax.set_title('[Bench] TPOT', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 5. ITL
        ax = axes[3, 1]
        baseline_itl_mean = baseline_result['mean_itl_ms'] if baseline_result else None
        baseline_itl_p99 = baseline_result['p99_itl_ms'] if baseline_result else None
        plot_metric_with_adaptive_axis(ax, k_ratios_bench, mean_itl, p99_itl,
                                       baseline_itl_mean, baseline_itl_p99, 'brown', 'ITL', 'ms')
        ax.set_xlabel('K Ratio', fontsize=11)
        ax.set_ylabel('Mean ITL (ms)', fontsize=11)
        ax.set_title('[Bench] ITL', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 6. Throughput vs TTFT Trade-off
        ax = axes[3, 2]
        scatter = ax.scatter(mean_ttft, throughputs_bench, c=k_ratios_bench, cmap='viridis',
                             s=100, edgecolors='black', linewidths=0.5)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('K Ratio', fontsize=10)
        if baseline_result:
            ax.scatter(baseline_result['mean_ttft_ms'],
                       baseline_result['total_token_throughput'],
                       c='red', s=150, marker='*', edgecolors='black',
                       linewidths=1, label='Baseline', zorder=5)
        ax.set_xlabel('Mean TTFT (ms)', fontsize=11)
        ax.set_ylabel('Throughput (tokens/s)', fontsize=11)
        ax.set_title('[Bench] TPS vs TTFT Trade-off', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    else:
        for r in range(2, 4):
            for c in range(3):
                axes[r, c].set_visible(False)

    plt.tight_layout(w_pad=2.5, h_pad=2.0)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nCombined K ratio plot saved to: {output_path}")

    if not no_show:
        plt.show()

    # 打印汇总表
    if has_schedule:
        print("\n" + "=" * 95)
        print("K RATIO SWEEP RESULTS SUMMARY (Schedule Stats, Adaptive N)")
        print("=" * 95)
        print(f"{'Ratio':>8} | {'Throughput':>12} | {'Prefill':>10} | {'Decode':>10} | {'Interval':>10} | {'Tok/Sched':>10} | {'#Scheds':>10}")
        print("-" * 95)
        if baseline_metrics:
            print(f"{'BASE':>8} | {baseline_metrics['overall_throughput_tps']:>12.0f} | "
                  f"{baseline_metrics['prefill_throughput_tps']:>10.0f} | "
                  f"{baseline_metrics['decode_throughput_tps']:>10.0f} | "
                  f"{baseline_metrics['schedule_interval_ms_mean']:>10.2f} | "
                  f"{baseline_metrics['tokens_per_schedule_mean']:>10.1f} | "
                  f"{baseline_metrics['num_schedules']:>10}")
            print("-" * 95)
        for r in k_ratios_sched:
            m = kratio_results_sched[r]
            print(f"{r:>8.2f} | {m['overall_throughput_tps']:>12.0f} | "
                  f"{m['prefill_throughput_tps']:>10.0f} | "
                  f"{m['decode_throughput_tps']:>10.0f} | "
                  f"{m['schedule_interval_ms_mean']:>10.2f} | "
                  f"{m['tokens_per_schedule_mean']:>10.1f} | "
                  f"{m['num_schedules']:>10}")
        print("=" * 95)

    if has_bench:
        print("\n" + "=" * 105)
        print("BENCHMARK RESULTS SUMMARY (K Ratio, Adaptive N)")
        print("=" * 105)
        print(f"{'Ratio':>8} | {'Throughput':>12} | {'Output TPS':>12} | {'Req TPS':>10} | "
              f"{'TTFT Mean':>10} | {'TTFT P99':>10} | {'TPOT Mean':>10} | {'ITL Mean':>10}")
        print("-" * 105)
        if baseline_result:
            print(f"{'BASE':>8} | {baseline_result['total_token_throughput']:>12.0f} | "
                  f"{baseline_result['output_throughput']:>12.0f} | "
                  f"{baseline_result['request_throughput']:>10.1f} | "
                  f"{baseline_result['mean_ttft_ms']:>10.0f} | "
                  f"{baseline_result['p99_ttft_ms']:>10.0f} | "
                  f"{baseline_result['mean_tpot_ms']:>10.2f} | "
                  f"{baseline_result['mean_itl_ms']:>10.2f}")
            print("-" * 105)
        for r in k_ratios_bench:
            res = kratio_results_bench[r]
            print(f"{r:>8.2f} | {res['total_token_throughput']:>12.0f} | "
                  f"{res['output_throughput']:>12.0f} | "
                  f"{res['request_throughput']:>10.1f} | "
                  f"{res['mean_ttft_ms']:>10.0f} | "
                  f"{res['p99_ttft_ms']:>10.0f} | "
                  f"{res['mean_tpot_ms']:>10.2f} | "
                  f"{res['mean_itl_ms']:>10.2f}")
        print("=" * 105)


def plot_all_mode(schedule_data: tuple, bench_data: tuple, output_path: str, no_show: bool):
    """绘制 schedule 和 bench 的组合图表 (all mode)，所有子图大小一致"""
    # schedule_data: (fixed_k_results, baseline, dynamic_results, kratio_results, special_results)
    # bench_data: (fixed_k_results, baseline, kratio_results, special_results)
    k_star_results_sched, baseline_metrics, dynamic_results, kratio_results_sched, special_results_sched = schedule_data
    k_star_results_bench, baseline_result, kratio_results_bench, special_results_bench = bench_data

    # 检查数据是否足够 (包括 kratio 和 special 结果)
    has_schedule = bool(k_star_results_sched) or bool(kratio_results_sched) or bool(special_results_sched)
    has_bench = bool(k_star_results_bench) or bool(kratio_results_bench) or bool(special_results_bench)

    # 如果只有 kratio 结果，使用 kratio 数据
    use_kratio_sched = not k_star_results_sched and kratio_results_sched
    use_kratio_bench = not k_star_results_bench and kratio_results_bench

    # 如果都是 kratio 模式，调用专门的 kratio 绘图函数
    if use_kratio_sched and use_kratio_bench:
        print("\nPlotting K ratio results (adaptive N mode) for combined view...")
        plot_kratio_all_mode(kratio_results_sched, baseline_metrics, kratio_results_bench,
                             baseline_result, output_path, no_show)
        return
    elif use_kratio_sched:
        print("\nSchedule data uses K ratio mode, Bench uses fixed K* mode...")
    elif use_kratio_bench:
        print("\nSchedule uses fixed K* mode, Bench data uses K ratio mode...")

    if not has_schedule and not has_bench:
        print("No data available for plotting!")
        return

    # 创建 4x3 的子图布局，所有子图大小一致 (增加 preemption 图)
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))

    dynamic_colors = ['magenta', 'cyan', 'lime', 'yellow', 'pink', 'brown']

    # ==================== Row 1: Schedule plots (4) + Bench plot (1) ====================

    if has_schedule:
        k_stars_sched = sorted(k_star_results_sched.keys())
        throughputs_sched = [k_star_results_sched[k]["overall_throughput_tps"] for k in k_stars_sched]
        prefill_tps = [k_star_results_sched[k]["prefill_throughput_tps"] for k in k_stars_sched]
        decode_tps = [k_star_results_sched[k]["decode_throughput_tps"] for k in k_stars_sched]
        schedule_intervals = [k_star_results_sched[k]["schedule_interval_ms_mean"] for k in k_stars_sched]
        tokens_per_schedule = [k_star_results_sched[k]["tokens_per_schedule_mean"] for k in k_stars_sched]
        num_schedules = [k_star_results_sched[k]["num_schedules"] for k in k_stars_sched]
        # Preemption metrics
        preempted_reqs = [k_star_results_sched[k].get("total_preempted_reqs", 0) for k in k_stars_sched]
        preempted_tokens = [k_star_results_sched[k].get("total_preempted_tokens", 0) for k in k_stars_sched]

        # 1. 总吞吐量 vs K* (Schedule)
        ax = axes[0, 0]
        ax.plot(k_stars_sched, throughputs_sched, 'b-o', linewidth=2, markersize=6, label='Fixed K*')
        if baseline_metrics:
            ax.axhline(y=baseline_metrics['overall_throughput_tps'], color='r',
                       linestyle='--', linewidth=2,
                       label=f"Baseline ({baseline_metrics['overall_throughput_tps']:.0f})")
        for i, (name, metrics) in enumerate(dynamic_results.items()):
            color = dynamic_colors[i % len(dynamic_colors)]
            ax.axhline(y=metrics['overall_throughput_tps'], color=color,
                       linestyle=':', linewidth=2, alpha=0.8,
                       label=f"Dynamic {name} ({metrics['overall_throughput_tps']:.0f})")
        # 绘制 special_results (ratio_auto, dynamic)
        special_colors = {'ratio_auto': 'green', 'dynamic': 'purple'}
        special_styles = {'ratio_auto': '-.', 'dynamic': ':'}
        for name, metrics in special_results_sched.items():
            color = special_colors.get(name, 'gray')
            style = special_styles.get(name, '-.')
            display_name = {'ratio_auto': 'Ratio Auto', 'dynamic': 'Dynamic (DP)'}.get(name, name)
            ax.axhline(y=metrics['overall_throughput_tps'], color=color,
                       linestyle=style, linewidth=2.5, alpha=0.9,
                       label=f"{display_name} ({metrics['overall_throughput_tps']:.0f})")
        ax.set_xlabel('K* Value', fontsize=11)
        ax.set_ylabel('Throughput (tokens/s)', fontsize=11)
        ax.set_title('[Schedule] Overall Throughput', fontsize=12, fontweight='bold')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

        best_idx_sched = np.argmax(throughputs_sched)
        best_k_sched = k_stars_sched[best_idx_sched]
        best_tps_sched = throughputs_sched[best_idx_sched]
        ax.annotate(f'Best: K*={best_k_sched}\n{best_tps_sched:.0f} tok/s',
                    xy=(best_k_sched, best_tps_sched), xytext=(30, -30), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='green'),
                    fontsize=9, color='green')

        # 2. Prefill vs Decode 吞吐量 (双 Y 轴)
        ax = axes[0, 1]
        ax.plot(k_stars_sched, prefill_tps, 'g-o', linewidth=2, markersize=5, label='Prefill', alpha=0.8)
        if baseline_metrics:
            ax.axhline(y=baseline_metrics['prefill_throughput_tps'], color='g',
                       linestyle=':', alpha=0.5, label='Baseline Prefill')
        # 绘制 special_results (ratio_auto, dynamic) 的 Prefill
        for name, metrics in special_results_sched.items():
            color = special_colors.get(name, 'gray')
            style = special_styles.get(name, '-.')
            display_name = {'ratio_auto': 'R.Auto', 'dynamic': 'Dynamic'}.get(name, name)
            ax.axhline(y=metrics['prefill_throughput_tps'], color=color,
                       linestyle=style, linewidth=2, alpha=0.7,
                       label=f"{display_name} P ({metrics['prefill_throughput_tps']:.0f})")
        ax.set_xlabel('K* Value', fontsize=11)
        ax.set_ylabel('Prefill Throughput (tok/s)', fontsize=11, color='g')
        ax.tick_params(axis='y', labelcolor='g')

        ax2 = ax.twinx()
        ax2.plot(k_stars_sched, decode_tps, 'orange', linestyle='-', marker='s',
                 linewidth=2, markersize=5, label='Decode', alpha=0.8)
        if baseline_metrics:
            ax2.axhline(y=baseline_metrics['decode_throughput_tps'], color='orange',
                        linestyle=':', alpha=0.5, label='Baseline Decode')
        # 绘制 special_results (ratio_auto, dynamic) 的 Decode
        special_decode_colors = {'ratio_auto': 'darkgreen', 'dynamic': 'indigo'}
        for name, metrics in special_results_sched.items():
            color = special_decode_colors.get(name, 'gray')
            style = special_styles.get(name, '-.')
            display_name = {'ratio_auto': 'R.Auto', 'dynamic': 'Dynamic'}.get(name, name)
            ax2.axhline(y=metrics['decode_throughput_tps'], color=color,
                        linestyle=style, linewidth=2, alpha=0.7,
                        label=f"{display_name} D ({metrics['decode_throughput_tps']:.0f})")
        ax2.set_ylabel('Decode Throughput (tok/s)', fontsize=11, color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        ax.set_title('[Schedule] Prefill vs Decode', fontsize=12, fontweight='bold')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc='best')
        ax.grid(True, alpha=0.3)

        # 3. 调度间隔 + 每次调度 tokens 数 (双 Y 轴)
        ax = axes[0, 2]
        ax.plot(k_stars_sched, schedule_intervals, 'purple', linestyle='-', marker='d',
                linewidth=2, markersize=6, label='Schedule Interval')
        if baseline_metrics:
            ax.axhline(y=baseline_metrics['schedule_interval_ms_mean'], color='purple',
                       linestyle='--', linewidth=2, alpha=0.5,
                       label=f"Baseline ({baseline_metrics['schedule_interval_ms_mean']:.2f} ms)")
        # 绘制 special_results 的 Schedule Interval
        special_interval_colors = {'ratio_auto': 'magenta', 'dynamic': 'darkviolet'}
        for name, metrics in special_results_sched.items():
            color = special_interval_colors.get(name, 'gray')
            style = special_styles.get(name, '-.')
            display_name = {'ratio_auto': 'R.Auto', 'dynamic': 'Dynamic'}.get(name, name)
            ax.axhline(y=metrics['schedule_interval_ms_mean'], color=color,
                       linestyle=style, linewidth=2, alpha=0.7,
                       label=f"{display_name} ({metrics['schedule_interval_ms_mean']:.2f}ms)")
        ax.set_xlabel('K* Value', fontsize=11)
        ax.set_ylabel('Schedule Interval (ms)', fontsize=11, color='purple')
        ax.tick_params(axis='y', labelcolor='purple')

        ax2 = ax.twinx()
        ax2.plot(k_stars_sched, tokens_per_schedule, 'teal', linestyle='-', marker='^',
                 linewidth=2, markersize=6, label='Tokens per Schedule')
        if baseline_metrics:
            ax2.axhline(y=baseline_metrics['tokens_per_schedule_mean'], color='teal',
                        linestyle='--', linewidth=2, alpha=0.5,
                        label=f"Baseline ({baseline_metrics['tokens_per_schedule_mean']:.1f})")
        # 绘制 special_results 的 Tokens per Schedule
        special_toksched_colors = {'ratio_auto': 'darkcyan', 'dynamic': 'cadetblue'}
        for name, metrics in special_results_sched.items():
            color = special_toksched_colors.get(name, 'gray')
            style = special_styles.get(name, '-.')
            display_name = {'ratio_auto': 'R.Auto', 'dynamic': 'Dynamic'}.get(name, name)
            ax2.axhline(y=metrics['tokens_per_schedule_mean'], color=color,
                        linestyle=style, linewidth=2, alpha=0.7,
                        label=f"{display_name} ({metrics['tokens_per_schedule_mean']:.1f})")
        ax2.set_ylabel('Tokens per Schedule', fontsize=11, color='teal')
        ax2.tick_params(axis='y', labelcolor='teal')

        ax.set_title('[Schedule] Interval & Tok/Sched', fontsize=12, fontweight='bold')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc='best')
        ax.grid(True, alpha=0.3)

        # 4. 调度总次数 vs K*
        ax = axes[1, 0]
        ax.plot(k_stars_sched, num_schedules, 'brown', linestyle='-', marker='v',
                linewidth=2, markersize=6, label='Fixed K*')
        if baseline_metrics:
            ax.axhline(y=baseline_metrics['num_schedules'], color='r',
                       linestyle='--', linewidth=2,
                       label=f"Baseline ({baseline_metrics['num_schedules']})")
        for i, (name, metrics) in enumerate(dynamic_results.items()):
            color = dynamic_colors[i % len(dynamic_colors)]
            ax.axhline(y=metrics['num_schedules'], color=color,
                       linestyle=':', linewidth=2, alpha=0.8, label=f"Dynamic {name}")
        # 绘制 special_results (ratio_auto, dynamic)
        for name, metrics in special_results_sched.items():
            color = special_colors.get(name, 'gray')
            style = special_styles.get(name, '-.')
            display_name = {'ratio_auto': 'R.Auto', 'dynamic': 'Dynamic'}.get(name, name)
            ax.axhline(y=metrics['num_schedules'], color=color,
                       linestyle=style, linewidth=2.5, alpha=0.9,
                       label=f"{display_name} ({metrics['num_schedules']})")
        ax.set_xlabel('K* Value', fontsize=11)
        ax.set_ylabel('Number of Schedules', fontsize=11)
        ax.set_title('[Schedule] Total Schedules', fontsize=12, fontweight='bold')
        ax.legend(fontsize=6, loc='best')
        ax.grid(True, alpha=0.3)

        min_sched_idx = np.argmin(num_schedules)
        min_sched_k = k_stars_sched[min_sched_idx]
        min_sched_val = num_schedules[min_sched_idx]
        ax.annotate(f'Min: K*={min_sched_k}\n{min_sched_val}',
                    xy=(min_sched_k, min_sched_val), xytext=(30, 20), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='green'),
                    fontsize=9, color='green')

        # 5. Preemption 统计 (双 Y 轴: 请求数 & Token 数)
        ax = axes[1, 1]
        has_preemption = any(p > 0 for p in preempted_reqs)
        has_baseline_preemption = baseline_metrics and baseline_metrics.get('total_preempted_reqs', 0) > 0
        has_special_preemption = any(m.get('total_preempted_reqs', 0) > 0 for m in special_results_sched.values())
        if has_preemption or has_baseline_preemption or has_special_preemption:
            # K* 数据: 红色实线
            ax.plot(k_stars_sched, preempted_reqs, 'red', linestyle='-', marker='x',
                    linewidth=2, markersize=6, label='Preempted Requests')
            # 绘制 special_results 的 Preemption Requests
            for name, metrics in special_results_sched.items():
                color = special_colors.get(name, 'gray')
                style = special_styles.get(name, '-.')
                display_name = {'ratio_auto': 'R.Auto', 'dynamic': 'Dynamic'}.get(name, name)
                preempt_reqs = metrics.get('total_preempted_reqs', 0)
                ax.axhline(y=preempt_reqs, color=color,
                           linestyle=style, linewidth=2, alpha=0.7,
                           label=f"{display_name} Reqs ({preempt_reqs})")
            ax.set_ylabel('Preempted Requests', fontsize=11, color='red')
            ax.tick_params(axis='y', labelcolor='red')
            # Baseline: 橙色虚线 (区分于红色)
            if baseline_metrics:
                ax.axhline(y=baseline_metrics.get('total_preempted_reqs', 0), color='orange',
                           linestyle='--', linewidth=2,
                           label=f"Baseline Reqs ({baseline_metrics.get('total_preempted_reqs', 0)})")

            ax2 = ax.twinx()
            # K* 数据: 蓝色实线 (区分于红色)
            ax2.plot(k_stars_sched, preempted_tokens, 'blue', linestyle='-', marker='+',
                     linewidth=2, markersize=6, label='Preempted Tokens')
            # 绘制 special_results 的 Preempted Tokens
            special_preempt_tok_colors = {'ratio_auto': 'darkgreen', 'dynamic': 'indigo'}
            for name, metrics in special_results_sched.items():
                color = special_preempt_tok_colors.get(name, 'gray')
                style = special_styles.get(name, '-.')
                display_name = {'ratio_auto': 'R.Auto', 'dynamic': 'Dynamic'}.get(name, name)
                preempt_toks = metrics.get('total_preempted_tokens', 0)
                ax2.axhline(y=preempt_toks, color=color,
                            linestyle=style, linewidth=2, alpha=0.7,
                            label=f"{display_name} Toks ({preempt_toks})")
            ax2.set_ylabel('Preempted Tokens', fontsize=11, color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
            # Baseline: 青色虚线 (区分于蓝色)
            if baseline_metrics:
                ax2.axhline(y=baseline_metrics.get('total_preempted_tokens', 0), color='cyan',
                            linestyle='--', linewidth=2,
                            label=f"Baseline Tokens ({baseline_metrics.get('total_preempted_tokens', 0)})")

            # 合并图例
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc='best')

            # 标注最大 preemption 点
            if max(preempted_reqs) > 0:
                max_preempt_idx = np.argmax(preempted_reqs)
                ax.annotate(f'Max: K*={k_stars_sched[max_preempt_idx]}\n{preempted_reqs[max_preempt_idx]} reqs',
                            xy=(k_stars_sched[max_preempt_idx], preempted_reqs[max_preempt_idx]),
                            xytext=(30, -20), textcoords='offset points',
                            arrowprops=dict(arrowstyle='->', color='red'),
                            fontsize=8, color='red')
        else:
            ax.text(0.5, 0.5, 'No Preemption\nDetected', transform=ax.transAxes,
                    fontsize=12, ha='center', va='center', color='green')
            ax.set_ylabel('Preempted Requests', fontsize=11)

        ax.set_xlabel('K* Value', fontsize=11)
        ax.set_title('[Schedule] Preemption Stats', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 6. Prefill Overhead 分析 (实际 prefill vs decode tokens)
        ax = axes[1, 2]
        actual_prefill = [k_star_results_sched[k]["total_prefill_tokens"] for k in k_stars_sched]
        actual_decode = [k_star_results_sched[k]["total_decode_tokens"] for k in k_stars_sched]
        # 计算 overhead: 如果 prefill > decode，说明有重复 prefill
        overhead_pct = [(p - d) / d * 100 if d > 0 else 0 for p, d in zip(actual_prefill, actual_decode)]

        bar_width = (k_stars_sched[-1] - k_stars_sched[0]) / len(k_stars_sched) * 0.8 if len(k_stars_sched) > 1 else 50
        ax.bar(k_stars_sched, overhead_pct, width=bar_width,
               color=['red' if o > 0 else 'green' for o in overhead_pct], alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

        # 绘制 special_results 的 Prefill Overhead
        for name, metrics in special_results_sched.items():
            color = special_colors.get(name, 'gray')
            style = special_styles.get(name, '-.')
            display_name = {'ratio_auto': 'R.Auto', 'dynamic': 'Dynamic'}.get(name, name)
            sp_prefill = metrics.get('total_prefill_tokens', 0)
            sp_decode = metrics.get('total_decode_tokens', 0)
            sp_overhead = (sp_prefill - sp_decode) / sp_decode * 100 if sp_decode > 0 else 0
            ax.axhline(y=sp_overhead, color=color,
                       linestyle=style, linewidth=2.5, alpha=0.9,
                       label=f"{display_name} ({sp_overhead:.1f}%)")

        ax.set_xlabel('K* Value', fontsize=11)
        ax.set_ylabel('Prefill Overhead (%)', fontsize=11)
        ax.set_title('[Schedule] Prefill Overhead\n(Prefill - Decode) / Decode', fontsize=12, fontweight='bold')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

        # 标注最大 overhead
        if any(o > 0 for o in overhead_pct):
            max_overhead_idx = np.argmax(overhead_pct)
            if overhead_pct[max_overhead_idx] > 0:
                ax.annotate(f'Max: {overhead_pct[max_overhead_idx]:.1f}%\nK*={k_stars_sched[max_overhead_idx]}',
                            xy=(k_stars_sched[max_overhead_idx], overhead_pct[max_overhead_idx]),
                            xytext=(30, -20), textcoords='offset points',
                            arrowprops=dict(arrowstyle='->', color='red'),
                            fontsize=8, color='red')
    else:
        # 如果没有 schedule 数据，隐藏前 2 行子图
        for r in range(2):
            for c in range(3):
                axes[r, c].set_visible(False)

    # ==================== Row 2-3: Bench plots (6) ====================
    if has_bench:
        k_stars_bench = sorted(k_star_results_bench.keys())
        throughputs_bench = [k_star_results_bench[k]["total_token_throughput"] for k in k_stars_bench]
        output_tps = [k_star_results_bench[k]["output_throughput"] for k in k_stars_bench]
        request_tps = [k_star_results_bench[k]["request_throughput"] for k in k_stars_bench]
        mean_ttft = [k_star_results_bench[k]["mean_ttft_ms"] for k in k_stars_bench]
        p99_ttft = [k_star_results_bench[k]["p99_ttft_ms"] for k in k_stars_bench]
        mean_tpot = [k_star_results_bench[k]["mean_tpot_ms"] for k in k_stars_bench]
        p99_tpot = [k_star_results_bench[k]["p99_tpot_ms"] for k in k_stars_bench]
        mean_itl = [k_star_results_bench[k]["mean_itl_ms"] for k in k_stars_bench]
        p99_itl = [k_star_results_bench[k]["p99_itl_ms"] for k in k_stars_bench]

        # 1. Total Token Throughput (Bench)
        ax = axes[2, 0]
        ax.plot(k_stars_bench, throughputs_bench, 'b-o', linewidth=2, markersize=6, label='Fixed K*')
        if baseline_result:
            ax.axhline(y=baseline_result['total_token_throughput'], color='r',
                       linestyle='--', linewidth=2,
                       label=f"Baseline ({baseline_result['total_token_throughput']:.0f})")
        # 绘制 special_results (ratio_auto, dynamic)
        special_colors = {'ratio_auto': 'green', 'dynamic': 'purple'}
        special_styles = {'ratio_auto': '-.', 'dynamic': ':'}
        for name, metrics in special_results_bench.items():
            color = special_colors.get(name, 'gray')
            style = special_styles.get(name, '-.')
            display_name = {'ratio_auto': 'Ratio Auto', 'dynamic': 'Dynamic (DP)'}.get(name, name)
            ax.axhline(y=metrics['total_token_throughput'], color=color,
                       linestyle=style, linewidth=2.5, alpha=0.9,
                       label=f"{display_name} ({metrics['total_token_throughput']:.0f})")
        ax.set_xlabel('K* Value', fontsize=11)
        ax.set_ylabel('Throughput (tokens/s)', fontsize=11)
        ax.set_title('[Bench] Total Throughput', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

        best_idx_bench = np.argmax(throughputs_bench)
        best_k_bench = k_stars_bench[best_idx_bench]
        best_tps_bench = throughputs_bench[best_idx_bench]
        ax.annotate(f'Best: K*={best_k_bench}\n{best_tps_bench:.0f} tok/s',
                    xy=(best_k_bench, best_tps_bench), xytext=(30, -30), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='green'),
                    fontsize=9, color='green')

        # 2. Output & Request Throughput
        ax = axes[2, 1]
        ax.plot(k_stars_bench, output_tps, 'g-o', linewidth=2, markersize=5,
                label='Output (tok/s)', alpha=0.8)
        ax2 = ax.twinx()
        ax2.plot(k_stars_bench, request_tps, 'orange', linestyle='-', marker='s',
                 linewidth=2, markersize=5, label='Request (req/s)', alpha=0.8)
        if baseline_result:
            ax.axhline(y=baseline_result['output_throughput'], color='g', linestyle=':', alpha=0.5,
                       label=f"Baseline Out ({baseline_result['output_throughput']:.0f})")
            ax2.axhline(y=baseline_result['request_throughput'], color='orange', linestyle=':', alpha=0.5,
                        label=f"Baseline Req ({baseline_result['request_throughput']:.1f})")
        # 绘制 special_results 的 Output & Request Throughput
        special_output_colors = {'ratio_auto': 'darkgreen', 'dynamic': 'teal'}
        special_request_colors = {'ratio_auto': 'coral', 'dynamic': 'darkorange'}
        for name, metrics in special_results_bench.items():
            out_color = special_output_colors.get(name, 'gray')
            req_color = special_request_colors.get(name, 'gray')
            style = special_styles.get(name, '-.')
            display_name = {'ratio_auto': 'R.Auto', 'dynamic': 'Dynamic'}.get(name, name)
            ax.axhline(y=metrics['output_throughput'], color=out_color,
                       linestyle=style, linewidth=2, alpha=0.7,
                       label=f"{display_name} Out ({metrics['output_throughput']:.0f})")
            ax2.axhline(y=metrics['request_throughput'], color=req_color,
                        linestyle=style, linewidth=2, alpha=0.7,
                        label=f"{display_name} Req ({metrics['request_throughput']:.1f})")
        ax.set_xlabel('K* Value', fontsize=11)
        ax.set_ylabel('Output Throughput (tok/s)', fontsize=11, color='g')
        ax2.set_ylabel('Request Throughput (req/s)', fontsize=11, color='orange')
        ax.set_title('[Bench] Output & Request TPS', fontsize=12, fontweight='bold')
        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc='best')
        ax.grid(True, alpha=0.3)

        # 3. TTFT
        ax = axes[2, 2]
        baseline_ttft_mean = baseline_result['mean_ttft_ms'] if baseline_result else None
        baseline_ttft_p99 = baseline_result['p99_ttft_ms'] if baseline_result else None
        plot_metric_with_adaptive_axis(ax, k_stars_bench, mean_ttft, p99_ttft,
                                       baseline_ttft_mean, baseline_ttft_p99,
                                       'purple', 'TTFT', 'ms')
        # 绘制 special_results 的 TTFT
        for name, metrics in special_results_bench.items():
            color = special_colors.get(name, 'gray')
            style = special_styles.get(name, '-.')
            display_name = {'ratio_auto': 'R.Auto', 'dynamic': 'Dynamic'}.get(name, name)
            ax.axhline(y=metrics['mean_ttft_ms'], color=color,
                       linestyle=style, linewidth=2, alpha=0.7,
                       label=f"{display_name} ({metrics['mean_ttft_ms']:.0f}ms)")
        ax.set_xlabel('K* Value', fontsize=11)
        ax.set_ylabel('Mean TTFT (ms)', fontsize=11)
        ax.set_title('[Bench] TTFT', fontsize=12, fontweight='bold')
        ax.legend(fontsize=6, loc='best')
        ax.grid(True, alpha=0.3)

        min_ttft_idx = np.argmin(mean_ttft)
        min_ttft_k = k_stars_bench[min_ttft_idx]
        min_ttft_val = mean_ttft[min_ttft_idx]
        ax.annotate(f'Min: K*={min_ttft_k}\n{min_ttft_val:.0f}ms',
                    xy=(min_ttft_k, min_ttft_val), xytext=(30, -30), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='green'),
                    fontsize=9, color='green')

        # 4. TPOT
        ax = axes[3, 0]
        baseline_tpot_mean = baseline_result['mean_tpot_ms'] if baseline_result else None
        baseline_tpot_p99 = baseline_result['p99_tpot_ms'] if baseline_result else None
        plot_metric_with_adaptive_axis(ax, k_stars_bench, mean_tpot, p99_tpot,
                                       baseline_tpot_mean, baseline_tpot_p99,
                                       'teal', 'TPOT', 'ms')
        # 绘制 special_results 的 TPOT
        for name, metrics in special_results_bench.items():
            color = special_colors.get(name, 'gray')
            style = special_styles.get(name, '-.')
            display_name = {'ratio_auto': 'R.Auto', 'dynamic': 'Dynamic'}.get(name, name)
            ax.axhline(y=metrics['mean_tpot_ms'], color=color,
                       linestyle=style, linewidth=2, alpha=0.7,
                       label=f"{display_name} ({metrics['mean_tpot_ms']:.2f}ms)")
        ax.set_xlabel('K* Value', fontsize=11)
        ax.set_ylabel('Mean TPOT (ms)', fontsize=11)
        ax.set_title('[Bench] TPOT', fontsize=12, fontweight='bold')
        ax.legend(fontsize=6, loc='best')
        ax.grid(True, alpha=0.3)

        min_tpot_idx = np.argmin(mean_tpot)
        min_tpot_k = k_stars_bench[min_tpot_idx]
        min_tpot_val = mean_tpot[min_tpot_idx]
        ax.annotate(f'Min: K*={min_tpot_k}\n{min_tpot_val:.2f}ms',
                    xy=(min_tpot_k, min_tpot_val), xytext=(30, -30), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='green'),
                    fontsize=9, color='green')

        # 5. ITL
        ax = axes[3, 1]
        baseline_itl_mean = baseline_result['mean_itl_ms'] if baseline_result else None
        baseline_itl_p99 = baseline_result['p99_itl_ms'] if baseline_result else None
        plot_metric_with_adaptive_axis(ax, k_stars_bench, mean_itl, p99_itl,
                                       baseline_itl_mean, baseline_itl_p99,
                                       'brown', 'ITL', 'ms')
        # 绘制 special_results 的 ITL
        for name, metrics in special_results_bench.items():
            color = special_colors.get(name, 'gray')
            style = special_styles.get(name, '-.')
            display_name = {'ratio_auto': 'R.Auto', 'dynamic': 'Dynamic'}.get(name, name)
            ax.axhline(y=metrics['mean_itl_ms'], color=color,
                       linestyle=style, linewidth=2, alpha=0.7,
                       label=f"{display_name} ({metrics['mean_itl_ms']:.2f}ms)")
        ax.set_xlabel('K* Value', fontsize=11)
        ax.set_ylabel('Mean ITL (ms)', fontsize=11)
        ax.set_title('[Bench] ITL', fontsize=12, fontweight='bold')
        ax.legend(fontsize=6, loc='best')
        ax.grid(True, alpha=0.3)

        min_itl_idx = np.argmin(mean_itl)
        min_itl_k = k_stars_bench[min_itl_idx]
        min_itl_val = mean_itl[min_itl_idx]
        ax.annotate(f'Min: K*={min_itl_k}\n{min_itl_val:.2f}ms',
                    xy=(min_itl_k, min_itl_val), xytext=(30, -30), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='green'),
                    fontsize=9, color='green')

        # 6. Throughput vs TTFT Trade-off
        ax = axes[3, 2]
        scatter = ax.scatter(mean_ttft, throughputs_bench, c=k_stars_bench, cmap='viridis',
                             s=100, edgecolors='black', linewidths=0.5)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('K* Value', fontsize=10)
        if baseline_result:
            ax.scatter(baseline_result['mean_ttft_ms'],
                       baseline_result['total_token_throughput'],
                       c='red', s=150, marker='*', edgecolors='black',
                       linewidths=1, label='Baseline', zorder=5)
        # 绘制 special_results (ratio_auto, dynamic) 在 trade-off 图上
        special_markers = {'ratio_auto': 's', 'dynamic': 'D'}
        for name, metrics in special_results_bench.items():
            color = special_colors.get(name, 'gray')
            marker = special_markers.get(name, 'o')
            display_name = {'ratio_auto': 'Ratio Auto', 'dynamic': 'Dynamic (DP)'}.get(name, name)
            ax.scatter(metrics['mean_ttft_ms'], metrics['total_token_throughput'],
                       c=color, s=150, marker=marker, edgecolors='black',
                       linewidths=1, label=display_name, zorder=5)
        ax.set_xlabel('Mean TTFT (ms)', fontsize=11)
        ax.set_ylabel('Throughput (tokens/s)', fontsize=11)
        ax.set_title('[Bench] TPS vs TTFT Trade-off', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    else:
        # 如果没有 bench 数据，隐藏第 2-3 行所有子图
        for r in range(2, 4):
            for c in range(3):
                axes[r, c].set_visible(False)

    # 调整布局
    plt.tight_layout(w_pad=2.5, h_pad=2.0)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nCombined plot saved to: {output_path}")

    if not no_show:
        plt.show()

    # 打印汇总表
    if has_schedule:
        k_stars_sched = sorted(k_star_results_sched.keys())
        print("\n" + "=" * 95)
        print("K* SWEEP RESULTS SUMMARY (Schedule Stats)")
        print("=" * 95)
        print(f"{'K*':>6} | {'Throughput':>12} | {'Prefill':>10} | {'Decode':>10} | {'Interval':>10} | {'Tok/Sched':>10} | {'#Scheds':>10}")
        print(f"{'':>6} | {'(tok/s)':>12} | {'(tok/s)':>10} | {'(tok/s)':>10} | {'(ms)':>10} | {'':>10} | {'':>10}")
        print("-" * 95)

        if baseline_metrics:
            print(f"{'BASE':>6} | {baseline_metrics['overall_throughput_tps']:>12.0f} | "
                  f"{baseline_metrics['prefill_throughput_tps']:>10.0f} | "
                  f"{baseline_metrics['decode_throughput_tps']:>10.0f} | "
                  f"{baseline_metrics['schedule_interval_ms_mean']:>10.2f} | "
                  f"{baseline_metrics['tokens_per_schedule_mean']:>10.1f} | "
                  f"{baseline_metrics['num_schedules']:>10}")
            print("-" * 95)

        best_k_sched = k_stars_sched[np.argmax([k_star_results_sched[k]["overall_throughput_tps"] for k in k_stars_sched])]
        min_sched_k = k_stars_sched[np.argmin([k_star_results_sched[k]["num_schedules"] for k in k_stars_sched])]
        for k in k_stars_sched:
            m = k_star_results_sched[k]
            marker = " <-- BEST TPS" if k == best_k_sched else ""
            marker = " <-- MIN SCHED" if k == min_sched_k else marker
            print(f"{k:>6} | {m['overall_throughput_tps']:>12.0f} | "
                  f"{m['prefill_throughput_tps']:>10.0f} | "
                  f"{m['decode_throughput_tps']:>10.0f} | "
                  f"{m['schedule_interval_ms_mean']:>10.2f} | "
                  f"{m['tokens_per_schedule_mean']:>10.1f} | "
                  f"{m['num_schedules']:>10}{marker}")
        print("=" * 95)

    if has_bench:
        k_stars_bench = sorted(k_star_results_bench.keys())
        print("\n" + "=" * 100)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 100)
        print(f"{'K*':>6} | {'Throughput':>12} | {'Output TPS':>12} | {'Req TPS':>10} | "
              f"{'TTFT Mean':>10} | {'TTFT P99':>10} | {'TPOT Mean':>10} | {'ITL Mean':>10}")
        print(f"{'':>6} | {'(tok/s)':>12} | {'(tok/s)':>12} | {'(req/s)':>10} | "
              f"{'(ms)':>10} | {'(ms)':>10} | {'(ms)':>10} | {'(ms)':>10}")
        print("-" * 100)

        if baseline_result:
            print(f"{'BASE':>6} | {baseline_result['total_token_throughput']:>12.0f} | "
                  f"{baseline_result['output_throughput']:>12.0f} | "
                  f"{baseline_result['request_throughput']:>10.1f} | "
                  f"{baseline_result['mean_ttft_ms']:>10.0f} | "
                  f"{baseline_result['p99_ttft_ms']:>10.0f} | "
                  f"{baseline_result['mean_tpot_ms']:>10.2f} | "
                  f"{baseline_result['mean_itl_ms']:>10.2f}")
            print("-" * 100)

        best_k_bench = k_stars_bench[np.argmax([k_star_results_bench[k]["total_token_throughput"] for k in k_stars_bench])]
        min_ttft_k = k_stars_bench[np.argmin([k_star_results_bench[k]["mean_ttft_ms"] for k in k_stars_bench])]
        for k in k_stars_bench:
            r = k_star_results_bench[k]
            marker = " <-- BEST TPS" if k == best_k_bench else ""
            marker = " <-- MIN TTFT" if k == min_ttft_k else marker
            print(f"{k:>6} | {r['total_token_throughput']:>12.0f} | "
                  f"{r['output_throughput']:>12.0f} | "
                  f"{r['request_throughput']:>10.1f} | "
                  f"{r['mean_ttft_ms']:>10.0f} | "
                  f"{r['p99_ttft_ms']:>10.0f} | "
                  f"{r['mean_tpot_ms']:>10.2f} | "
                  f"{r['mean_itl_ms']:>10.2f}{marker}")
        print("=" * 100)


def plot_bench_mode(k_star_results: dict, baseline_result: dict, output_path: str, no_show: bool,
                    kratio_results: dict = None, special_results: dict = None):
    """绘制 benchmark 结果图表

    Args:
        special_results: 包含 'ratio_auto' 和 'dynamic' 模式的结果
    """
    kratio_results = kratio_results or {}
    special_results = special_results or {}

    # 如果只有 kratio 结果，没有固定 K* 结果，使用 kratio 数据
    if not k_star_results and kratio_results:
        print("\nPlotting K ratio results (adaptive N mode)...")
        plot_kratio_bench_mode(kratio_results, baseline_result, output_path, no_show)
        return

    # 如果没有固定 K* 或 kratio 结果，但有 special_results，打印汇总
    if not k_star_results and not kratio_results and special_results:
        print("\n" + "=" * 105)
        print("SPECIAL MODE RESULTS SUMMARY (Benchmark)")
        print("=" * 105)
        print(f"{'Mode':<12} | {'Throughput':>12} | {'Output TPS':>12} | {'Req TPS':>10} | "
              f"{'TTFT Mean':>10} | {'TTFT P99':>10} | {'TPOT Mean':>10} | {'ITL Mean':>10}")
        print(f"{'':>12} | {'(tok/s)':>12} | {'(tok/s)':>12} | {'(req/s)':>10} | "
              f"{'(ms)':>10} | {'(ms)':>10} | {'(ms)':>10} | {'(ms)':>10}")
        print("-" * 105)

        if baseline_result:
            print(f"{'baseline':<12} | {baseline_result['total_token_throughput']:>12.0f} | "
                  f"{baseline_result['output_throughput']:>12.0f} | "
                  f"{baseline_result['request_throughput']:>10.1f} | "
                  f"{baseline_result['mean_ttft_ms']:>10.0f} | "
                  f"{baseline_result['p99_ttft_ms']:>10.0f} | "
                  f"{baseline_result['mean_tpot_ms']:>10.2f} | "
                  f"{baseline_result['mean_itl_ms']:>10.2f}")

        for name, r in special_results.items():
            print(f"{name:<12} | {r['total_token_throughput']:>12.0f} | "
                  f"{r['output_throughput']:>12.0f} | "
                  f"{r['request_throughput']:>10.1f} | "
                  f"{r['mean_ttft_ms']:>10.0f} | "
                  f"{r['p99_ttft_ms']:>10.0f} | "
                  f"{r['mean_tpot_ms']:>10.2f} | "
                  f"{r['mean_itl_ms']:>10.2f}")
        print("=" * 105)
        return

    k_stars = sorted(k_star_results.keys())

    # 提取各项指标
    throughputs = [k_star_results[k]["total_token_throughput"] for k in k_stars]
    output_tps = [k_star_results[k]["output_throughput"] for k in k_stars]
    request_tps = [k_star_results[k]["request_throughput"] for k in k_stars]
    mean_ttft = [k_star_results[k]["mean_ttft_ms"] for k in k_stars]
    p99_ttft = [k_star_results[k]["p99_ttft_ms"] for k in k_stars]
    mean_tpot = [k_star_results[k]["mean_tpot_ms"] for k in k_stars]
    p99_tpot = [k_star_results[k]["p99_tpot_ms"] for k in k_stars]
    mean_itl = [k_star_results[k]["mean_itl_ms"] for k in k_stars]
    p99_itl = [k_star_results[k]["p99_itl_ms"] for k in k_stars]

    # 增加图宽度，为双Y轴标签留出空间
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    # 1. Total Token Throughput vs K*
    ax = axes[0, 0]
    ax.plot(k_stars, throughputs, 'b-o', linewidth=2, markersize=6, label='Fixed K*')
    if baseline_result:
        ax.axhline(y=baseline_result['total_token_throughput'], color='r',
                   linestyle='--', linewidth=2,
                   label=f"Baseline ({baseline_result['total_token_throughput']:.0f})")
    # 绘制 special_results (ratio_auto, dynamic)
    special_colors = {'ratio_auto': 'green', 'dynamic': 'purple'}
    special_styles = {'ratio_auto': '-.', 'dynamic': ':'}
    for name, metrics in special_results.items():
        color = special_colors.get(name, 'gray')
        style = special_styles.get(name, '-.')
        display_name = {'ratio_auto': 'Ratio Auto (θ* auto)', 'dynamic': 'Dynamic (DP)'}.get(name, name)
        ax.axhline(y=metrics['total_token_throughput'], color=color,
                   linestyle=style, linewidth=2.5, alpha=0.9,
                   label=f"{display_name} ({metrics['total_token_throughput']:.0f})")
    ax.set_xlabel('K* Value', fontsize=12)
    ax.set_ylabel('Throughput (tokens/s)', fontsize=12)
    ax.set_title('Total Token Throughput vs K*', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 记录最佳点 (用于后续 trade-off 图)
    best_idx = np.argmax(throughputs)
    best_k = k_stars[best_idx]
    best_tps = throughputs[best_idx]

    # 标注最佳吞吐点 (使用 offset points 避免超出边界)
    ax.annotate(f'Best: K*={best_k}\n{best_tps:.0f} tok/s',
                xy=(best_k, best_tps), xytext=(30, -30), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')

    # 2. Output Throughput & Request Throughput vs K*
    ax = axes[0, 1]
    ax.plot(k_stars, output_tps, 'g-o', linewidth=2, markersize=5,
            label='Output (tok/s)', alpha=0.8)
    ax2 = ax.twinx()
    ax2.plot(k_stars, request_tps, 'orange', linestyle='-', marker='s',
             linewidth=2, markersize=5, label='Request (req/s)', alpha=0.8)
    if baseline_result:
        ax.axhline(y=baseline_result['output_throughput'], color='g', linestyle=':', alpha=0.5,
                   label=f"Baseline Output ({baseline_result['output_throughput']:.0f})")
        ax2.axhline(y=baseline_result['request_throughput'], color='orange', linestyle=':', alpha=0.5,
                    label=f"Baseline Req ({baseline_result['request_throughput']:.1f})")
    # 绘制 special_results (ratio_auto, dynamic) 的 Output & Request Throughput
    special_output_colors = {'ratio_auto': 'darkgreen', 'dynamic': 'teal'}
    special_request_colors = {'ratio_auto': 'coral', 'dynamic': 'darkorange'}
    for name, metrics in special_results.items():
        out_color = special_output_colors.get(name, 'gray')
        req_color = special_request_colors.get(name, 'gray')
        style = special_styles.get(name, '-.')
        display_name = {'ratio_auto': 'R.Auto', 'dynamic': 'Dynamic'}.get(name, name)
        ax.axhline(y=metrics['output_throughput'], color=out_color,
                   linestyle=style, linewidth=2, alpha=0.7,
                   label=f"{display_name} Out ({metrics['output_throughput']:.0f})")
        ax2.axhline(y=metrics['request_throughput'], color=req_color,
                    linestyle=style, linewidth=2, alpha=0.7,
                    label=f"{display_name} Req ({metrics['request_throughput']:.1f})")
    ax.set_xlabel('K* Value', fontsize=12)
    ax.set_ylabel('Output Throughput (tok/s)', fontsize=12, color='g')
    ax2.set_ylabel('Request Throughput (req/s)', fontsize=12, color='orange')
    ax.set_title('Output & Request Throughput vs K*', fontsize=14)
    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    # 标注最佳 Output 和 Request 吞吐点 (使用 offset points 避免超出边界)
    best_output_idx = np.argmax(output_tps)
    best_request_idx = np.argmax(request_tps)
    ax.annotate(f'Best Output: K*={k_stars[best_output_idx]}',
                xy=(k_stars[best_output_idx], output_tps[best_output_idx]),
                xytext=(30, -20), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=9, color='green')
    ax2.annotate(f'Best Req: K*={k_stars[best_request_idx]}',
                 xy=(k_stars[best_request_idx], request_tps[best_request_idx]),
                 xytext=(30, 20), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='orange'),
                 fontsize=9, color='orange')

    # 3. TTFT vs K*
    ax = axes[0, 2]
    baseline_ttft_mean = baseline_result['mean_ttft_ms'] if baseline_result else None
    baseline_ttft_p99 = baseline_result['p99_ttft_ms'] if baseline_result else None
    plot_metric_with_adaptive_axis(ax, k_stars, mean_ttft, p99_ttft,
                                   baseline_ttft_mean, baseline_ttft_p99,
                                   'purple', 'TTFT', 'ms')
    # 绘制 special_results (ratio_auto, dynamic) 的 TTFT
    for name, metrics in special_results.items():
        color = special_colors.get(name, 'gray')
        style = special_styles.get(name, '-.')
        display_name = {'ratio_auto': 'R.Auto', 'dynamic': 'Dynamic'}.get(name, name)
        ax.axhline(y=metrics['mean_ttft_ms'], color=color,
                   linestyle=style, linewidth=2, alpha=0.7,
                   label=f"{display_name} ({metrics['mean_ttft_ms']:.0f}ms)")
    ax.set_xlabel('K* Value', fontsize=12)
    ax.set_ylabel('Mean TTFT (ms)', fontsize=12)
    ax.set_title('Time To First Token vs K*', fontsize=14)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    # 标注最小 TTFT (使用 offset points 避免超出边界)
    min_ttft_idx = np.argmin(mean_ttft)
    min_ttft_k = k_stars[min_ttft_idx]
    min_ttft_val = mean_ttft[min_ttft_idx]
    ax.annotate(f'Min: K*={min_ttft_k}\n{min_ttft_val:.0f}ms',
                xy=(min_ttft_k, min_ttft_val), xytext=(30, -30), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')

    # 4. TPOT vs K*
    ax = axes[1, 0]
    baseline_tpot_mean = baseline_result['mean_tpot_ms'] if baseline_result else None
    baseline_tpot_p99 = baseline_result['p99_tpot_ms'] if baseline_result else None
    plot_metric_with_adaptive_axis(ax, k_stars, mean_tpot, p99_tpot,
                                   baseline_tpot_mean, baseline_tpot_p99,
                                   'teal', 'TPOT', 'ms')
    # 绘制 special_results (ratio_auto, dynamic) 的 TPOT
    for name, metrics in special_results.items():
        color = special_colors.get(name, 'gray')
        style = special_styles.get(name, '-.')
        display_name = {'ratio_auto': 'R.Auto', 'dynamic': 'Dynamic'}.get(name, name)
        ax.axhline(y=metrics['mean_tpot_ms'], color=color,
                   linestyle=style, linewidth=2, alpha=0.7,
                   label=f"{display_name} ({metrics['mean_tpot_ms']:.2f}ms)")
    ax.set_xlabel('K* Value', fontsize=12)
    ax.set_ylabel('Mean TPOT (ms)', fontsize=12)
    ax.set_title('Time Per Output Token vs K*', fontsize=14)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    # 标注最小 TPOT (使用 offset points 避免超出边界)
    min_tpot_idx = np.argmin(mean_tpot)
    min_tpot_k = k_stars[min_tpot_idx]
    min_tpot_val = mean_tpot[min_tpot_idx]
    ax.annotate(f'Min: K*={min_tpot_k}\n{min_tpot_val:.2f}ms',
                xy=(min_tpot_k, min_tpot_val), xytext=(30, -30), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')

    # 5. ITL vs K*
    ax = axes[1, 1]
    baseline_itl_mean = baseline_result['mean_itl_ms'] if baseline_result else None
    baseline_itl_p99 = baseline_result['p99_itl_ms'] if baseline_result else None
    plot_metric_with_adaptive_axis(ax, k_stars, mean_itl, p99_itl,
                                   baseline_itl_mean, baseline_itl_p99,
                                   'brown', 'ITL', 'ms')
    # 绘制 special_results (ratio_auto, dynamic) 的 ITL
    for name, metrics in special_results.items():
        color = special_colors.get(name, 'gray')
        style = special_styles.get(name, '-.')
        display_name = {'ratio_auto': 'R.Auto', 'dynamic': 'Dynamic'}.get(name, name)
        ax.axhline(y=metrics['mean_itl_ms'], color=color,
                   linestyle=style, linewidth=2, alpha=0.7,
                   label=f"{display_name} ({metrics['mean_itl_ms']:.2f}ms)")
    ax.set_xlabel('K* Value', fontsize=12)
    ax.set_ylabel('Mean ITL (ms)', fontsize=12)
    ax.set_title('Inter-Token Latency vs K*', fontsize=14)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    # 标注最小 ITL (使用 offset points 避免超出边界)
    min_itl_idx = np.argmin(mean_itl)
    min_itl_k = k_stars[min_itl_idx]
    min_itl_val = mean_itl[min_itl_idx]
    ax.annotate(f'Min: K*={min_itl_k}\n{min_itl_val:.2f}ms',
                xy=(min_itl_k, min_itl_val), xytext=(30, -30), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')

    # 6. Throughput vs TTFT Trade-off
    ax = axes[1, 2]
    scatter = ax.scatter(mean_ttft, throughputs, c=k_stars, cmap='viridis',
                         s=100, edgecolors='black', linewidths=0.5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('K* Value', fontsize=10)
    if baseline_result:
        ax.scatter(baseline_result['mean_ttft_ms'],
                   baseline_result['total_token_throughput'],
                   c='red', s=150, marker='*', edgecolors='black',
                   linewidths=1, label='Baseline', zorder=5)
    # 绘制 special_results (ratio_auto, dynamic) 在 trade-off 图上
    special_markers = {'ratio_auto': 's', 'dynamic': 'D'}
    for name, metrics in special_results.items():
        color = special_colors.get(name, 'gray')
        marker = special_markers.get(name, 'o')
        display_name = {'ratio_auto': 'R.Auto', 'dynamic': 'Dynamic'}.get(name, name)
        ax.scatter(metrics['mean_ttft_ms'], metrics['total_token_throughput'],
                   c=color, s=150, marker=marker, edgecolors='black',
                   linewidths=1, label=display_name, zorder=5)
    ax.set_xlabel('Mean TTFT (ms)', fontsize=12)
    ax.set_ylabel('Throughput (tokens/s)', fontsize=12)
    ax.set_title('Throughput vs TTFT Trade-off', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 标注最佳吞吐点和最小延迟点 (使用相对偏移避免超出边界)
    ttft_range = max(mean_ttft) - min(mean_ttft) if len(mean_ttft) > 1 else 1000
    tps_range = max(throughputs) - min(throughputs) if len(throughputs) > 1 else 1000
    ax.annotate(f'K*={best_k}', xy=(mean_ttft[best_idx], throughputs[best_idx]),
                xytext=(mean_ttft[best_idx] + ttft_range * 0.1, throughputs[best_idx] - tps_range * 0.1),
                fontsize=9, color='blue',
                arrowprops=dict(arrowstyle='->', color='blue', alpha=0.5))
    ax.annotate(f'K*={min_ttft_k}', xy=(mean_ttft[min_ttft_idx], throughputs[min_ttft_idx]),
                xytext=(mean_ttft[min_ttft_idx] + ttft_range * 0.1, throughputs[min_ttft_idx] + tps_range * 0.1),
                fontsize=9, color='green',
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.5))

    # 增加子图间距，避免双Y轴标签重叠
    plt.tight_layout(w_pad=3.0, h_pad=2.0)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    if not no_show:
        plt.show()

    # 打印汇总表
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 100)
    print(f"{'K*':>6} | {'Throughput':>12} | {'Output TPS':>12} | {'Req TPS':>10} | "
          f"{'TTFT Mean':>10} | {'TTFT P99':>10} | {'TPOT Mean':>10} | {'ITL Mean':>10}")
    print(f"{'':>6} | {'(tok/s)':>12} | {'(tok/s)':>12} | {'(req/s)':>10} | "
          f"{'(ms)':>10} | {'(ms)':>10} | {'(ms)':>10} | {'(ms)':>10}")
    print("-" * 100)

    if baseline_result:
        print(f"{'BASE':>6} | {baseline_result['total_token_throughput']:>12.0f} | "
              f"{baseline_result['output_throughput']:>12.0f} | "
              f"{baseline_result['request_throughput']:>10.1f} | "
              f"{baseline_result['mean_ttft_ms']:>10.0f} | "
              f"{baseline_result['p99_ttft_ms']:>10.0f} | "
              f"{baseline_result['mean_tpot_ms']:>10.2f} | "
              f"{baseline_result['mean_itl_ms']:>10.2f}")

    # 打印 special_results (ratio_auto, dynamic)
    for name, r in special_results.items():
        display_name = {'ratio_auto': 'R_AUTO', 'dynamic': 'DYNMC'}.get(name, name[:6].upper())
        print(f"{display_name:>6} | {r['total_token_throughput']:>12.0f} | "
              f"{r['output_throughput']:>12.0f} | "
              f"{r['request_throughput']:>10.1f} | "
              f"{r['mean_ttft_ms']:>10.0f} | "
              f"{r['p99_ttft_ms']:>10.0f} | "
              f"{r['mean_tpot_ms']:>10.2f} | "
              f"{r['mean_itl_ms']:>10.2f}")

    if baseline_result or special_results:
        print("-" * 100)

    for k in k_stars:
        r = k_star_results[k]
        marker = " <-- BEST TPS" if k == best_k else ""
        marker = " <-- MIN TTFT" if k == min_ttft_k else marker
        print(f"{k:>6} | {r['total_token_throughput']:>12.0f} | "
              f"{r['output_throughput']:>12.0f} | "
              f"{r['request_throughput']:>10.1f} | "
              f"{r['mean_ttft_ms']:>10.0f} | "
              f"{r['p99_ttft_ms']:>10.0f} | "
              f"{r['mean_tpot_ms']:>10.2f} | "
              f"{r['mean_itl_ms']:>10.2f}{marker}")

    print("=" * 100)

    # 计算相对于 baseline 的提升
    if baseline_result:
        print(f"\nBest Throughput K*={best_k} vs Baseline:")
        tps_improvement = (best_tps - baseline_result['total_token_throughput']) / baseline_result['total_token_throughput'] * 100
        print(f"  Throughput improvement: {tps_improvement:+.1f}%")

        print(f"\nMin TTFT K*={min_ttft_k} vs Baseline:")
        ttft_improvement = (min_ttft_val - baseline_result['mean_ttft_ms']) / baseline_result['mean_ttft_ms'] * 100
        print(f"  TTFT improvement: {ttft_improvement:+.1f}%")


def main():
    args = parse_args()

    # 结果目录
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path(__file__).parent / "results"
        if not results_dir.exists():
            results_dir = Path("results")

    print(f"Results directory: {results_dir}")
    print(f"Mode: {args.mode}")
    print(f"K* range: [{args.k_min}, {args.k_max if args.k_max else 'inf'}]")

    # all 模式：将 schedule 和 bench 画在同一张图里
    if args.mode == "all":
        print(f"\n{'=' * 50}")
        print("Generating combined plot (schedule + bench)...")
        print(f"{'=' * 50}")

        # 加载数据
        schedule_data = load_schedule_data(results_dir, args)
        bench_data = load_bench_data(results_dir, args)

        # schedule_data: (fixed_k_results, baseline, dynamic_results, kratio_results, special_results)
        # bench_data: (fixed_k_results, baseline, kratio_results, special_results)
        if not schedule_data[0] and not schedule_data[3] and not schedule_data[4] and \
           not bench_data[0] and not bench_data[2] and not bench_data[3]:
            print("No data found for either schedule or bench!")
            return

        # 确定输出文件名 (保存到 results_dir)
        if args.output:
            output_path = args.output
        else:
            output_path = str(results_dir / "kstar_sweep_results.png")

        plot_all_mode(schedule_data, bench_data, output_path, args.no_show)

        # 如果有 kratio 结果且设置了 --plot-n-trajectory，绘制 N 轨迹图
        kratio_results_sched = schedule_data[3]  # kratio_results from schedule_data
        if args.plot_n_trajectory and kratio_results_sched:
            print(f"\n{'=' * 50}")
            print("Generating N trajectory plot...")
            print(f"{'=' * 50}")
            plot_n_trajectory(kratio_results_sched, output_path, args.no_show)

        return

    # 单独模式：只画 schedule 或 bench
    mode = args.mode
    print(f"\n{'=' * 50}")
    print(f"Generating {mode} plot...")
    print(f"{'=' * 50}")

    # 确定输出文件名 (保存到 results_dir)
    if args.output:
        output_path = args.output
    else:
        filename = "benchmark_results.png" if mode == "bench" else "kstar_sweep_results.png"
        output_path = str(results_dir / filename)

    if mode == "schedule":
        k_star_results, baseline_metrics, dynamic_results, kratio_results, special_results = load_schedule_data(results_dir, args)
        if not k_star_results and not kratio_results and not special_results:
            print("No fixed*_stats.json, kratio_*_stats.json, ratio_auto_stats.json, or dynamic_stats.json files found!")
            return
        plot_schedule_mode(k_star_results, baseline_metrics, dynamic_results, output_path, args.no_show, kratio_results, special_results)

        # 如果有 kratio 结果且设置了 --plot-n-trajectory，绘制 N 轨迹图
        if args.plot_n_trajectory and kratio_results:
            print(f"\n{'=' * 50}")
            print("Generating N trajectory plot...")
            print(f"{'=' * 50}")
            plot_n_trajectory(kratio_results, output_path, args.no_show)

    else:  # bench mode
        k_star_results, baseline_result, kratio_results, special_results = load_bench_data(results_dir, args)
        if not k_star_results and not kratio_results and not special_results:
            print("No bench_fixed*.json, bench_kratio_*.json, bench_ratio_auto.json, or bench_dynamic.json files found!")
            return
        plot_bench_mode(k_star_results, baseline_result, output_path, args.no_show, kratio_results, special_results)


if __name__ == "__main__":
    main()
