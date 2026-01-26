#!/usr/bin/env python3
"""
验证 Proposition: 最优切换阈值公式

命题：θ* 满足：
    θ/(1-θ) + ln(1-θ) = p * α_p / α_d

验证方法：
1. 固定 N 和 p（几何分布参数）
2. 扫描 k = 1, 2, ..., N，测量每个 k 的实际吞吐量
3. 找到实验最优 k*_exp
4. 与公式预测的 k*_theory = θ* × N 比较

用法：
    python verify_proposition.py --results-dir ../outputs/kstar_geometric_bs128_c2049_n1000

    # 绘制多个 p 值的结果
    python verify_proposition.py --results-dir ../outputs/kstar_geometric_bs128_c2049_n1000 --plot
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def solve_theta_star(C: float, tol: float = 1e-8) -> float:
    """
    求解 θ/(1-θ) + ln(1-θ) = C

    f(θ) 在 (0,1) 上单调递增，f(0+) = 0, f(1-) = +∞
    使用二分法求解。
    """
    if C <= 0:
        return 0.0

    def f(theta):
        if theta <= 0 or theta >= 1:
            return float('inf')
        return theta / (1 - theta) + math.log(1 - theta)

    # 二分法
    lo, hi = 1e-10, 1 - 1e-10
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if f(mid) < C:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def load_calibration(calibration_file: str) -> dict:
    """加载硬件校准参数"""
    with open(calibration_file) as f:
        return json.load(f)


def parse_scenario_dir(scenario_name: str) -> tuple[int, int]:
    """解析场景目录名，返回 (input_len, output_len)"""
    # 格式: in{input}_out{output}
    parts = scenario_name.replace("in", "").replace("out", " ").split("_")
    input_len = int(parts[0])
    output_len = int(parts[1])
    return input_len, output_len


def load_results(results_dir: str) -> dict:
    """
    加载实验结果

    返回格式：
    {
        (input_len, output_len): {
            "baseline": throughput,
            "kstar": {k: throughput, ...},
            "direct": throughput,
            ...
        }
    }
    """
    results = {}
    results_path = Path(results_dir)

    for scenario_dir in results_path.iterdir():
        if not scenario_dir.is_dir():
            continue
        if scenario_dir.name.startswith(".") or scenario_dir.name == "logs":
            continue

        try:
            input_len, output_len = parse_scenario_dir(scenario_dir.name)
        except (ValueError, IndexError):
            continue

        scenario_results = {
            "baseline": None,
            "kstar": {},
            "direct": None,
        }

        for bench_file in scenario_dir.glob("bench_*.json"):
            with open(bench_file) as f:
                data = json.load(f)

            throughput = data.get("request_throughput", 0)
            filename = bench_file.stem  # bench_fixed32 -> fixed32

            if "baseline" in filename:
                scenario_results["baseline"] = throughput
            elif "direct" in filename:
                scenario_results["direct"] = throughput
            elif "fixed" in filename:
                # bench_fixed32.json -> k=32
                k = int(filename.replace("bench_fixed", ""))
                scenario_results["kstar"][k] = throughput

        if scenario_results["kstar"]:
            results[(input_len, output_len)] = scenario_results

    return results


def analyze_scenario(
    scenario_key: tuple[int, int],
    scenario_data: dict,
    alpha_p: float,
    alpha_d: float,
    N: int,
) -> dict:
    """
    分析单个场景的结果

    返回：
    {
        "input_len": int,
        "output_len": int,
        "p": float,  # 几何分布参数
        "C": float,  # p * alpha_p / alpha_d
        "theta_theory": float,
        "k_theory": int,
        "k_exp": int,
        "throughput_theory": float,
        "throughput_exp": float,
        "throughput_baseline": float,
        "k_values": list,
        "throughputs": list,
    }
    """
    input_len, output_len = scenario_key

    # 几何分布参数 p = 1/E[output_len]
    p = 1.0 / output_len

    # 计算 C = p * α_p / α_d
    C = p * alpha_p / alpha_d

    # 理论最优 θ*
    theta_theory = solve_theta_star(C)
    k_theory = max(1, round(theta_theory * N))

    # 实验最优 k*
    kstar_results = scenario_data["kstar"]
    if not kstar_results:
        return None

    k_values = sorted(kstar_results.keys())
    throughputs = [kstar_results[k] for k in k_values]

    k_exp = k_values[np.argmax(throughputs)]
    throughput_exp = max(throughputs)
    throughput_theory = kstar_results.get(k_theory, 0)

    return {
        "input_len": input_len,
        "output_len": output_len,
        "p": p,
        "C": C,
        "theta_theory": theta_theory,
        "k_theory": k_theory,
        "k_exp": k_exp,
        "theta_exp": k_exp / N,
        "throughput_theory": throughput_theory,
        "throughput_exp": throughput_exp,
        "throughput_baseline": scenario_data.get("baseline", 0),
        "throughput_direct": scenario_data.get("direct", 0),
        "k_values": k_values,
        "throughputs": throughputs,
        "N": N,
    }


def plot_verification(analyses: list[dict], output_path: str):
    """
    绘制验证结果

    1. 每个 p 值的吞吐量曲线 (throughput vs k)
    2. 理论 vs 实验的 k* 对比
    3. θ* 公式验证: 实际 θ* vs 预测 θ*
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 吞吐量曲线 (左上)
    ax1 = axes[0, 0]
    for analysis in analyses:
        label = f"E[O]={analysis['output_len']} (p={analysis['p']:.3f})"
        ax1.plot(analysis["k_values"], analysis["throughputs"], 'o-', label=label, markersize=3)
        # 标记理论最优
        ax1.axvline(analysis["k_theory"], linestyle='--', alpha=0.5)
        ax1.plot(analysis["k_exp"], analysis["throughput_exp"], 's', markersize=8)

    ax1.set_xlabel("k (switching threshold)")
    ax1.set_ylabel("Throughput (req/s)")
    ax1.set_title("Throughput vs Switching Threshold k")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. k* 对比 (右上)
    ax2 = axes[0, 1]
    k_theory = [a["k_theory"] for a in analyses]
    k_exp = [a["k_exp"] for a in analyses]
    output_lens = [a["output_len"] for a in analyses]

    ax2.scatter(k_theory, k_exp, c=output_lens, cmap='viridis', s=100)
    max_k = max(max(k_theory), max(k_exp)) * 1.1
    ax2.plot([0, max_k], [0, max_k], 'k--', label='y=x (perfect match)')
    ax2.set_xlabel("k* (theory)")
    ax2.set_ylabel("k* (experiment)")
    ax2.set_title("Theory vs Experiment: Optimal k*")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label("E[output_len]")

    # 3. θ* 公式验证 (左下)
    ax3 = axes[1, 0]
    theta_theory = [a["theta_theory"] for a in analyses]
    theta_exp = [a["theta_exp"] for a in analyses]

    ax3.scatter(theta_theory, theta_exp, c=output_lens, cmap='viridis', s=100)
    ax3.plot([0, 1], [0, 1], 'k--', label='y=x')
    ax3.set_xlabel("θ* (theory from proposition)")
    ax3.set_ylabel("θ* (experiment)")
    ax3.set_title("Proposition Verification: θ* = k*/N")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 吞吐量对比 (右下)
    ax4 = axes[1, 1]
    x = np.arange(len(analyses))
    width = 0.2

    baseline = [a["throughput_baseline"] or 0 for a in analyses]
    direct = [a["throughput_direct"] or 0 for a in analyses]
    theory_tput = [a["throughput_theory"] for a in analyses]
    exp_tput = [a["throughput_exp"] for a in analyses]

    ax4.bar(x - 1.5*width, baseline, width, label='Baseline', alpha=0.8)
    ax4.bar(x - 0.5*width, direct, width, label='Direct (auto k*)', alpha=0.8)
    ax4.bar(x + 0.5*width, theory_tput, width, label='k* (theory)', alpha=0.8)
    ax4.bar(x + 1.5*width, exp_tput, width, label='k* (exp best)', alpha=0.8)

    ax4.set_xlabel("Scenario")
    ax4.set_ylabel("Throughput (req/s)")
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"E[O]={a['output_len']}" for a in analyses], rotation=45, ha='right')
    ax4.set_title("Throughput Comparison")
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    plt.close()


def print_summary(analyses: list[dict]):
    """打印验证结果摘要"""
    print("\n" + "=" * 80)
    print("Proposition Verification Summary")
    print("=" * 80)
    print(f"Formula: θ/(1-θ) + ln(1-θ) = p·α_p/α_d")
    print()

    print(f"{'E[O]':>6} {'p':>8} {'C':>8} {'θ*_th':>8} {'θ*_exp':>8} "
          f"{'k*_th':>6} {'k*_exp':>6} {'Δk':>4} {'Tput_th':>10} {'Tput_exp':>10} {'Δ%':>6}")
    print("-" * 100)

    errors = []
    for a in analyses:
        delta_k = a["k_exp"] - a["k_theory"]
        delta_pct = (a["throughput_exp"] - a["throughput_theory"]) / a["throughput_exp"] * 100 if a["throughput_exp"] > 0 else 0

        print(f"{a['output_len']:>6} {a['p']:>8.4f} {a['C']:>8.4f} {a['theta_theory']:>8.4f} {a['theta_exp']:>8.4f} "
              f"{a['k_theory']:>6} {a['k_exp']:>6} {delta_k:>+4} {a['throughput_theory']:>10.2f} {a['throughput_exp']:>10.2f} {delta_pct:>+6.1f}%")
        errors.append(abs(delta_k))

    print("-" * 100)
    print(f"Mean |Δk|: {np.mean(errors):.2f}")
    print(f"Max  |Δk|: {np.max(errors)}")

    # 计算 θ* 误差
    theta_errors = [abs(a["theta_exp"] - a["theta_theory"]) for a in analyses]
    print(f"Mean |Δθ*|: {np.mean(theta_errors):.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Verify Proposition 1")
    parser.add_argument("--results-dir", required=True, help="Directory containing experiment results")
    parser.add_argument("--calibration-file", help="Path to calibration JSON file")
    parser.add_argument("--N", type=int, default=128, help="Fixed batch size N")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--output-plot", default="proposition_verification.png", help="Output plot path")
    args = parser.parse_args()

    # 加载校准参数
    if args.calibration_file:
        calib = load_calibration(args.calibration_file)
    else:
        # 尝试从实验配置中读取
        config_file = Path(args.results_dir) / "experiment_config.json"
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
            calib = config.get("calibration_params", {})
        else:
            print("Warning: No calibration file found, using defaults")
            calib = {"alpha_p": 0.006, "alpha_d": 0.004}

    alpha_p = calib.get("alpha_p", 0.006)
    alpha_d = calib.get("alpha_d", 0.004)

    print(f"Using calibration: α_p={alpha_p}, α_d={alpha_d}")
    print(f"Fixed N = {args.N}")

    # 加载实验结果
    results = load_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}")
        return

    print(f"Found {len(results)} scenarios")

    # 分析每个场景
    analyses = []
    for scenario_key, scenario_data in sorted(results.items()):
        analysis = analyze_scenario(scenario_key, scenario_data, alpha_p, alpha_d, args.N)
        if analysis:
            analyses.append(analysis)

    if not analyses:
        print("No valid analyses")
        return

    # 打印摘要
    print_summary(analyses)

    # 绘图
    if args.plot:
        output_path = Path(args.results_dir) / args.output_plot
        plot_verification(analyses, str(output_path))


if __name__ == "__main__":
    main()
