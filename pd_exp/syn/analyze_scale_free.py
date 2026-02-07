#!/usr/bin/env python3
"""
分析 k* 实验结果，验证两个关键性质：

1. Scale-free: 对于相同的 p（几何分布参数），不同 N 下的 θ* = k*/N 应该相同
2. 单调性: θ* 应该随着 p 的减小（即 E[O] 增大）而减小

用法:
    python analyze_scale_free.py --base-dir ../outputs

    # 指定特定的实验目录
    python analyze_scale_free.py --dirs ../outputs/kstar_geometric_bs64_c4096_n4000 ../outputs/kstar_geometric_bs128_c4096_n4000
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_experiment_results(exp_dir: Path) -> dict:
    """
    加载单个实验目录的结果

    返回:
    {
        "N": int,  # batch size
        "calibration": {...},
        "scenarios": {
            output_len: {
                k: throughput,
                ...
            }
        }
    }
    """
    config_file = exp_dir / "experiment_config.json"
    if not config_file.exists():
        return None

    with open(config_file) as f:
        config = json.load(f)

    N = int(config.get("pd_max_batch_size", 0))
    if N == 0:
        return None

    result = {
        "N": N,
        "calibration": config.get("calibration_params", {}),
        "scenarios": {}
    }

    # 遍历每个场景目录
    for scenario_dir in exp_dir.iterdir():
        if not scenario_dir.is_dir():
            continue
        if not scenario_dir.name.startswith("in"):
            continue

        # 解析 output_len: in1_out32 -> 32
        try:
            output_len = int(scenario_dir.name.split("_out")[1])
        except (ValueError, IndexError):
            continue

        k_results = {}
        for bench_file in scenario_dir.glob("bench_fixed*.json"):
            # bench_fixed32.json -> k=32
            try:
                k = int(bench_file.stem.replace("bench_fixed", ""))
            except ValueError:
                continue

            with open(bench_file) as f:
                data = json.load(f)

            throughput = data.get("request_throughput", 0)
            if throughput > 0:
                k_results[k] = throughput

        if k_results:
            result["scenarios"][output_len] = k_results

    return result


def find_optimal_k(k_results: dict) -> tuple[int, float]:
    """找到最优 k* 和对应的吞吐量"""
    if not k_results:
        return 0, 0.0
    best_k = max(k_results.keys(), key=lambda k: k_results[k])
    return best_k, k_results[best_k]


def analyze_all_experiments(exp_dirs: list[Path]) -> dict:
    """
    分析所有实验，返回汇总数据

    返回:
    {
        output_len: {
            N: {
                "k_star": int,
                "theta_star": float,
                "throughput": float,
                "k_results": {k: throughput}
            }
        }
    }
    """
    all_data = defaultdict(dict)

    for exp_dir in exp_dirs:
        result = load_experiment_results(exp_dir)
        if result is None:
            continue

        N = result["N"]
        for output_len, k_results in result["scenarios"].items():
            k_star, throughput = find_optimal_k(k_results)
            theta_star = k_star / N if N > 0 else 0

            all_data[output_len][N] = {
                "k_star": k_star,
                "theta_star": theta_star,
                "throughput": throughput,
                "k_results": k_results,
            }

    return dict(all_data)


def print_analysis(all_data: dict, calibration: dict):
    """打印分析结果"""
    alpha_p = calibration.get("alpha_p", 0)
    alpha_d = calibration.get("alpha_d", 0)

    print("\n" + "=" * 100)
    print("Scale-Free and Monotonicity Analysis")
    print("=" * 100)
    print(f"Calibration: α_p={alpha_p:.6f}, α_d={alpha_d:.6f}")
    print()

    # 按 output_len 排序
    sorted_output_lens = sorted(all_data.keys())

    # 获取所有 N 值
    all_N = set()
    for output_len in sorted_output_lens:
        all_N.update(all_data[output_len].keys())
    sorted_N = sorted(all_N)

    # 打印表头
    header = f"{'E[O]':>6} {'p':>8}"
    for N in sorted_N:
        header += f" | N={N}: k*  θ*"
    header += " | θ* std | monotonic?"
    print(header)
    print("-" * len(header))

    # 用于验证单调性
    prev_theta_mean = None
    monotonic_violations = []

    for output_len in sorted_output_lens:
        p = 1.0 / output_len
        row = f"{output_len:>6} {p:>8.4f}"

        thetas = []
        for N in sorted_N:
            if N in all_data[output_len]:
                data = all_data[output_len][N]
                k_star = data["k_star"]
                theta = data["theta_star"]
                thetas.append(theta)
                row += f" |      {k_star:>3}  {theta:.3f}"
            else:
                row += " |       -    -"

        # 计算 θ* 的标准差（验证 scale-free）
        if len(thetas) >= 2:
            theta_std = np.std(thetas)
            theta_mean = np.mean(thetas)
            row += f" |  {theta_std:.4f}"

            # 验证单调性
            if prev_theta_mean is not None:
                if theta_mean > prev_theta_mean + 0.01:  # 允许小误差
                    row += " | ✗ (↑)"
                    monotonic_violations.append((output_len, theta_mean, prev_theta_mean))
                else:
                    row += " | ✓"
            else:
                row += " | -"

            prev_theta_mean = theta_mean
        else:
            row += " |    N/A | N/A"

        print(row)

    print("-" * len(header))

    # 总结
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    # Scale-free 验证
    all_stds = []
    for output_len in sorted_output_lens:
        thetas = [all_data[output_len][N]["theta_star"]
                  for N in sorted_N if N in all_data[output_len]]
        if len(thetas) >= 2:
            all_stds.append(np.std(thetas))

    if all_stds:
        print(f"Scale-free property:")
        print(f"  Mean θ* std across N: {np.mean(all_stds):.4f}")
        print(f"  Max  θ* std across N: {np.max(all_stds):.4f}")
        if np.mean(all_stds) < 0.05:
            print(f"  ✓ Scale-free holds (std < 0.05)")
        else:
            print(f"  ✗ Scale-free violated (std >= 0.05)")

    # 单调性验证
    print(f"\nMonotonicity property (θ* decreases as p decreases):")
    if monotonic_violations:
        print(f"  ✗ {len(monotonic_violations)} violations found:")
        for output_len, theta, prev_theta in monotonic_violations:
            print(f"    E[O]={output_len}: θ*={theta:.3f} > prev θ*={prev_theta:.3f}")
    else:
        print(f"  ✓ Monotonicity holds")


def plot_analysis(all_data: dict, output_path: str, calibration: dict):
    """绘制分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 获取所有 N 和 output_len
    sorted_output_lens = sorted(all_data.keys())
    all_N = set()
    for output_len in sorted_output_lens:
        all_N.update(all_data[output_len].keys())
    sorted_N = sorted(all_N)

    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_N)))

    # 1. θ* vs p for different N (左上)
    ax1 = axes[0, 0]
    for i, N in enumerate(sorted_N):
        p_vals = []
        theta_vals = []
        for output_len in sorted_output_lens:
            if N in all_data[output_len]:
                p_vals.append(1.0 / output_len)
                theta_vals.append(all_data[output_len][N]["theta_star"])
        if p_vals:
            ax1.plot(p_vals, theta_vals, 'o-', color=colors[i], label=f'N={N}', markersize=8)

    ax1.set_xlabel("p = 1/E[O]")
    ax1.set_ylabel("θ* = k*/N")
    ax1.set_title("θ* vs p for different N (Scale-free test)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. θ* vs E[O] (右上) - 更直观
    ax2 = axes[0, 1]
    for i, N in enumerate(sorted_N):
        output_lens = []
        theta_vals = []
        for output_len in sorted_output_lens:
            if N in all_data[output_len]:
                output_lens.append(output_len)
                theta_vals.append(all_data[output_len][N]["theta_star"])
        if output_lens:
            ax2.plot(output_lens, theta_vals, 'o-', color=colors[i], label=f'N={N}', markersize=8)

    ax2.set_xlabel("E[O] (expected output length)")
    ax2.set_ylabel("θ* = k*/N")
    ax2.set_title("θ* vs E[O] (Monotonicity test)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Throughput curves for one scenario (左下)
    ax3 = axes[1, 0]
    # 选择一个中间的 output_len
    mid_output_len = sorted_output_lens[len(sorted_output_lens) // 2]
    for i, N in enumerate(sorted_N):
        if N in all_data[mid_output_len]:
            k_results = all_data[mid_output_len][N]["k_results"]
            k_vals = sorted(k_results.keys())
            throughputs = [k_results[k] for k in k_vals]
            # 归一化 k 到 [0, 1] 范围
            theta_vals = [k / N for k in k_vals]
            ax3.plot(theta_vals, throughputs, 'o-', color=colors[i], label=f'N={N}', markersize=6)
            # 标记最优点
            k_star = all_data[mid_output_len][N]["k_star"]
            theta_star = k_star / N
            max_tput = all_data[mid_output_len][N]["throughput"]
            ax3.plot(theta_star, max_tput, 's', color=colors[i], markersize=12)

    ax3.set_xlabel("θ = k/N")
    ax3.set_ylabel("Throughput (req/s)")
    ax3.set_title(f"Throughput vs θ at E[O]={mid_output_len} (p={1/mid_output_len:.4f})")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. θ* consistency across N (右下)
    ax4 = axes[1, 1]
    theta_by_output = {}
    for output_len in sorted_output_lens:
        thetas = [all_data[output_len][N]["theta_star"]
                  for N in sorted_N if N in all_data[output_len]]
        if len(thetas) >= 2:
            theta_by_output[output_len] = {
                "mean": np.mean(thetas),
                "std": np.std(thetas),
                "min": np.min(thetas),
                "max": np.max(thetas),
            }

    if theta_by_output:
        output_lens = list(theta_by_output.keys())
        means = [theta_by_output[o]["mean"] for o in output_lens]
        stds = [theta_by_output[o]["std"] for o in output_lens]

        ax4.errorbar(output_lens, means, yerr=stds, fmt='o-', capsize=5, markersize=8)
        ax4.fill_between(output_lens,
                         [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                         alpha=0.2)

        ax4.set_xlabel("E[O]")
        ax4.set_ylabel("θ* (mean ± std across N)")
        ax4.set_title("θ* consistency across different N")
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze scale-free and monotonicity properties")
    parser.add_argument("--base-dir", default="../outputs", help="Base directory containing experiment results")
    parser.add_argument("--dirs", nargs="+", help="Specific experiment directories to analyze")
    parser.add_argument("--plot", action="store_true", default=True, help="Generate plots")
    parser.add_argument("--output-plot", default="scale_free_analysis.png", help="Output plot path")
    args = parser.parse_args()

    # 找到所有 geometric 实验目录
    if args.dirs:
        exp_dirs = [Path(d) for d in args.dirs]
    else:
        base_path = Path(args.base_dir)
        exp_dirs = sorted(base_path.glob("kstar_geometric_bs*"))

    print(f"Found {len(exp_dirs)} experiment directories:")
    for d in exp_dirs:
        print(f"  {d.name}")

    # 加载校准参数（从第一个实验目录）
    calibration = {}
    for exp_dir in exp_dirs:
        result = load_experiment_results(exp_dir)
        if result and result.get("calibration"):
            calibration = result["calibration"]
            break

    # 分析所有实验
    all_data = analyze_all_experiments(exp_dirs)

    if not all_data:
        print("No valid data found!")
        return

    # 打印分析结果
    print_analysis(all_data, calibration)

    # 绘图
    if args.plot:
        output_plot = Path(args.base_dir) / args.output_plot
        plot_analysis(all_data, str(output_plot), calibration)


if __name__ == "__main__":
    main()
