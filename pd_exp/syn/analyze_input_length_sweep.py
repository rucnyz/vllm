#!/usr/bin/env python3
"""
P2 实验分析脚本：验证 θ* 独立于输入长度 μ_L

功能：
  - 从 k* sweep 结果中找到最优 k* 和对应的 θ* = k*/N
  - 验证 θ* 不随输入长度变化

用法：
  python analyze_input_length_sweep.py <results_dir>
  python analyze_input_length_sweep.py pd_exp/outputs/input_length_sweep_N256_O128
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np


def load_json(filepath: Path) -> dict | None:
    """加载 JSON 文件"""
    try:
        with open(filepath, encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"  Error loading {filepath}: {e}")
        return None


def extract_k_from_filename(filename: str) -> int | None:
    """从文件名中提取 k* 值"""
    # bench_fixed128.json -> 128
    match = re.search(r'fixed(\d+)', filename)
    if match:
        return int(match.group(1))
    return None


def get_throughput(bench_result: dict) -> float:
    """从 benchmark 结果中提取吞吐量"""
    # 使用 output_throughput (tokens/s) 作为主要指标
    return bench_result.get("output_throughput", 0)


def analyze_scenario(scenario_dir: Path, num_repeats: int = 3) -> dict:
    """分析单个场景的结果，找到最优 k*"""
    results = defaultdict(list)

    # 收集所有 k* 的吞吐量
    for bench_file in scenario_dir.glob("bench_fixed*_run*.json"):
        k_star = extract_k_from_filename(bench_file.name)
        if k_star is None:
            continue

        data = load_json(bench_file)
        if data is None:
            continue

        throughput = get_throughput(data)
        results[k_star].append(throughput)

    # 如果没有 _run 后缀的文件
    if not results:
        for bench_file in scenario_dir.glob("bench_fixed*.json"):
            if "_run" in bench_file.name:
                continue
            k_star = extract_k_from_filename(bench_file.name)
            if k_star is None:
                continue

            data = load_json(bench_file)
            if data is None:
                continue

            throughput = get_throughput(data)
            results[k_star].append(throughput)

    if not results:
        return {}

    # 计算每个 k* 的平均吞吐量
    k_mean_throughput = {}
    for k, throughputs in results.items():
        k_mean_throughput[k] = np.mean(throughputs)

    # 找到最优 k*
    optimal_k = max(k_mean_throughput, key=k_mean_throughput.get)
    optimal_throughput = k_mean_throughput[optimal_k]

    return {
        "optimal_k_star": optimal_k,
        "optimal_throughput": optimal_throughput,
        "all_k_throughputs": k_mean_throughput,
        "num_samples": {k: len(v) for k, v in results.items()},
    }


def main():
    parser = argparse.ArgumentParser(
        description="P2 实验分析：验证 θ* 独立于输入长度"
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="实验结果目录"
    )
    parser.add_argument(
        "--batch-size", "-N",
        type=int,
        default=256,
        help="批大小 N (默认 256)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 JSON 文件路径"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    N = args.batch_size

    if not results_dir.exists():
        print(f"Error: Directory {results_dir} does not exist")
        return

    # 加载实验配置
    config_file = results_dir / "experiment_config.json"
    if config_file.exists():
        config = load_json(config_file)
        if config:
            N = config.get("fixed_params", {}).get("N", N)
            print(f"从配置文件读取 N = {N}")

    print(f"\n{'='*60}")
    print("P2 实验：验证 θ* = k*/N 独立于输入长度 μ_L")
    print(f"{'='*60}")
    print(f"批大小 N = {N}")
    print()

    # 分析每个输入长度场景
    results = {}
    theta_stars = []

    for scenario_dir in sorted(results_dir.glob("in*_out*")):
        if not scenario_dir.is_dir():
            continue

        # 解析场景名
        match = re.match(r'in(\d+)_out(\d+)', scenario_dir.name)
        if not match:
            continue

        input_len = int(match.group(1))
        output_len = int(match.group(2))

        print(f"分析场景: μ_L = {input_len}, E[O] = {output_len}")

        scenario_result = analyze_scenario(scenario_dir)
        if not scenario_result:
            print(f"  警告: 没有找到有效结果")
            continue

        optimal_k = scenario_result["optimal_k_star"]
        theta_star = optimal_k / N
        theta_stars.append(theta_star)

        print(f"  最优 k* = {optimal_k}")
        print(f"  θ* = k*/N = {theta_star:.4f}")
        print(f"  吞吐量 = {scenario_result['optimal_throughput']:.2f} tokens/s")

        results[input_len] = {
            "input_len": input_len,
            "output_len": output_len,
            "optimal_k_star": optimal_k,
            "theta_star": theta_star,
            "throughput": scenario_result["optimal_throughput"],
        }

    if not results:
        print("\n错误: 没有找到任何有效结果")
        return

    # 统计 θ* 的分布
    theta_mean = np.mean(theta_stars)
    theta_std = np.std(theta_stars)

    print(f"\n{'='*60}")
    print("汇总结果")
    print(f"{'='*60}")
    print(f"\n{'μ_L':<10} {'k*':<10} {'θ*':<10}")
    print("-" * 30)
    for input_len in sorted(results.keys()):
        r = results[input_len]
        print(f"{input_len:<10} {r['optimal_k_star']:<10} {r['theta_star']:.4f}")
    print("-" * 30)
    print(f"{'平均':<10} {'':<10} {theta_mean:.4f}")
    print(f"{'标准差':<10} {'':<10} {theta_std:.4f}")

    print(f"\n结论:")
    print(f"  θ* = {theta_mean:.4f} ± {theta_std:.4f}")
    print(f"  变异系数 CV = {theta_std/theta_mean*100:.1f}%")

    if theta_std / theta_mean < 0.15:  # CV < 15%
        print(f"  √ θ* 基本独立于输入长度 μ_L (P2 验证通过)")
    else:
        print(f"  ? θ* 可能依赖于输入长度 (需要进一步分析)")

    # 输出 LaTeX 格式
    print(f"\nLaTeX 格式:")
    mu_L_values = sorted(results.keys())
    theta_values = [f"{results[mu_L]['theta_star']:.2f}" for mu_L in mu_L_values]
    print(f"  $\\mu_L \\in \\{{{', '.join(map(str, mu_L_values))}\\}}$")
    print(f"  $\\hat{{\\theta}}^* = $ {', '.join(theta_values)}")
    print(f"  (std = {theta_std:.2f})")

    # 保存结果
    if args.output:
        output_data = {
            "experiment": "P2_input_length_independence",
            "N": N,
            "results": results,
            "summary": {
                "theta_star_mean": theta_mean,
                "theta_star_std": theta_std,
                "theta_star_cv": theta_std / theta_mean,
            }
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
