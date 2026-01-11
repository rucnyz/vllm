#!/usr/bin/env python3
"""
绘制 Grid Search 实验结果对比图

1. 固定 TB，变化 BS 时的对比
2. 固定 BS，变化 TB 时的对比
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import sys

# 设置中文字体支持
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def load_all_results(base_dir: Path) -> Dict:
    """加载所有实验结果"""
    results = {}

    for tb_dir in base_dir.glob("tb*"):
        tb = int(tb_dir.name[2:])

        for bs_dir in tb_dir.glob("bs*"):
            bs = int(bs_dir.name[2:])

            for scenario_dir in bs_dir.iterdir():
                if not scenario_dir.is_dir():
                    continue
                scenario = scenario_dir.name

                # 初始化结构
                if scenario not in results:
                    results[scenario] = {}
                if tb not in results[scenario]:
                    results[scenario][tb] = {}
                if bs not in results[scenario][tb]:
                    results[scenario][tb][bs] = {}

                # 加载 baseline 和 pd 结果
                for scheduler in ["baseline", "pd"]:
                    bench_file = scenario_dir / f"bench_{scheduler}.json"
                    if bench_file.exists():
                        with open(bench_file) as f:
                            data = json.load(f)
                        results[scenario][tb][bs][scheduler] = {
                            "throughput": data.get("request_throughput", 0),
                            "output_throughput": data.get("output_throughput", 0),
                            "mean_itl_ms": data.get("mean_itl_ms", 0),
                            "p99_itl_ms": data.get("p99_itl_ms", 0),
                            "mean_ttft_ms": data.get("mean_ttft_ms", 0),
                            "p99_ttft_ms": data.get("p99_ttft_ms", 0),
                        }

    return results


def plot_fixed_tb_vary_bs(results: Dict, output_dir: Path):
    """固定 TB，变化 BS 时的对比图"""

    metrics = [
        ("throughput", "Request Throughput (req/s)", True),
        ("output_throughput", "Output Throughput (tok/s)", True),
        ("mean_itl_ms", "Mean ITL (ms)", False),
        ("p99_itl_ms", "P99 ITL (ms)", False),
        ("mean_ttft_ms", "Mean TTFT (ms)", False),
        ("p99_ttft_ms", "P99 TTFT (ms)", False),
    ]

    scenarios = sorted(results.keys())
    tb_values = sorted(set(tb for scenario in results.values() for tb in scenario.keys()))

    for scenario in scenarios:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, (metric, ylabel, higher_better) in enumerate(metrics):
            ax = axes[idx]

            for tb in tb_values:
                if tb not in results[scenario]:
                    continue

                bs_values = sorted(results[scenario][tb].keys())

                baseline_vals = []
                pd_vals = []
                valid_bs = []

                for bs in bs_values:
                    data = results[scenario][tb][bs]
                    if "baseline" in data and "pd" in data:
                        baseline_vals.append(data["baseline"].get(metric, 0))
                        pd_vals.append(data["pd"].get(metric, 0))
                        valid_bs.append(bs)

                if valid_bs:
                    ax.plot(valid_bs, baseline_vals, 'o-', label=f'Baseline TB={tb}', alpha=0.7)
                    ax.plot(valid_bs, pd_vals, 's--', label=f'PD TB={tb}', alpha=0.7)

            ax.set_xlabel('Max Batch Size (BS)')
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(sorted(set(bs for tb in results[scenario].values() for bs in tb.keys())))

        plt.suptitle(f'Fixed TB, Vary BS - Scenario: {scenario}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        output_path = output_dir / f"fixed_tb_vary_bs_{scenario}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")


def plot_fixed_bs_vary_tb(results: Dict, output_dir: Path):
    """固定 BS，变化 TB 时的对比图"""

    metrics = [
        ("throughput", "Request Throughput (req/s)", True),
        ("output_throughput", "Output Throughput (tok/s)", True),
        ("mean_itl_ms", "Mean ITL (ms)", False),
        ("p99_itl_ms", "P99 ITL (ms)", False),
        ("mean_ttft_ms", "Mean TTFT (ms)", False),
        ("p99_ttft_ms", "P99 TTFT (ms)", False),
    ]

    scenarios = sorted(results.keys())

    # 获取所有 BS 值
    all_bs = set()
    for scenario in results.values():
        for tb in scenario.values():
            all_bs.update(tb.keys())
    bs_values = sorted(all_bs)

    for scenario in scenarios:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, (metric, ylabel, higher_better) in enumerate(metrics):
            ax = axes[idx]

            for bs in bs_values:
                # 收集所有 TB 值对应的数据
                tb_list = []
                baseline_vals = []
                pd_vals = []

                for tb in sorted(results[scenario].keys()):
                    if bs in results[scenario][tb]:
                        data = results[scenario][tb][bs]
                        if "baseline" in data and "pd" in data:
                            tb_list.append(tb)
                            baseline_vals.append(data["baseline"].get(metric, 0))
                            pd_vals.append(data["pd"].get(metric, 0))

                if tb_list:
                    ax.plot(tb_list, baseline_vals, 'o-', label=f'Baseline BS={bs}', alpha=0.7)
                    ax.plot(tb_list, pd_vals, 's--', label=f'PD BS={bs}', alpha=0.7)

            ax.set_xlabel('Token Budget (TB)')
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'Fixed BS, Vary TB - Scenario: {scenario}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        output_path = output_dir / f"fixed_bs_vary_tb_{scenario}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")


def plot_single_tb_comparison(results: Dict, output_dir: Path):
    """每个 TB 值单独一张图，对比 Baseline 和 PD"""

    metrics = [
        ("throughput", "Throughput (req/s)", True),
        ("mean_itl_ms", "Mean ITL (ms)", False),
        ("p99_itl_ms", "P99 ITL (ms)", False),
    ]

    scenarios = sorted(results.keys())
    tb_values = sorted(set(tb for scenario in results.values() for tb in scenario.keys()))

    for scenario in scenarios:
        for tb in tb_values:
            if tb not in results[scenario]:
                continue

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            bs_values = sorted(results[scenario][tb].keys())

            for idx, (metric, ylabel, higher_better) in enumerate(metrics):
                ax = axes[idx]

                baseline_vals = []
                pd_vals = []
                valid_bs = []

                for bs in bs_values:
                    data = results[scenario][tb][bs]
                    if "baseline" in data and "pd" in data:
                        baseline_vals.append(data["baseline"].get(metric, 0))
                        pd_vals.append(data["pd"].get(metric, 0))
                        valid_bs.append(bs)

                if valid_bs:
                    x = np.arange(len(valid_bs))
                    width = 0.35

                    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='#2ecc71')
                    bars2 = ax.bar(x + width/2, pd_vals, width, label='PD', color='#3498db')

                    ax.set_xlabel('Max Batch Size (BS)')
                    ax.set_ylabel(ylabel)
                    ax.set_title(ylabel)
                    ax.set_xticks(x)
                    ax.set_xticklabels(valid_bs)
                    ax.legend()
                    ax.grid(True, alpha=0.3, axis='y')

                    # 添加数值标注
                    for bar, val in zip(bars1, baseline_vals):
                        ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                   ha='center', va='bottom', fontsize=7, rotation=45)
                    for bar, val in zip(bars2, pd_vals):
                        ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                   ha='center', va='bottom', fontsize=7, rotation=45)

            plt.suptitle(f'{scenario} | TB={tb} | Baseline vs PD', fontsize=12, fontweight='bold')
            plt.tight_layout()
            output_path = output_dir / f"tb{tb}_{scenario}_comparison.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {output_path}")


def plot_single_bs_comparison(results: Dict, output_dir: Path):
    """每个 BS 值单独一张图，对比 Baseline 和 PD"""

    metrics = [
        ("throughput", "Throughput (req/s)", True),
        ("mean_itl_ms", "Mean ITL (ms)", False),
        ("p99_itl_ms", "P99 ITL (ms)", False),
    ]

    scenarios = sorted(results.keys())

    # 获取所有 BS 值
    all_bs = set()
    for scenario in results.values():
        for tb in scenario.values():
            all_bs.update(tb.keys())
    bs_values = sorted(all_bs)

    for scenario in scenarios:
        for bs in bs_values:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            # 收集所有 TB 值对应的数据
            tb_list = []
            baseline_data = {m[0]: [] for m in metrics}
            pd_data = {m[0]: [] for m in metrics}

            for tb in sorted(results[scenario].keys()):
                if bs in results[scenario][tb]:
                    data = results[scenario][tb][bs]
                    if "baseline" in data and "pd" in data:
                        tb_list.append(tb)
                        for m, _, _ in metrics:
                            baseline_data[m].append(data["baseline"].get(m, 0))
                            pd_data[m].append(data["pd"].get(m, 0))

            if not tb_list:
                plt.close()
                continue

            for idx, (metric, ylabel, higher_better) in enumerate(metrics):
                ax = axes[idx]

                x = np.arange(len(tb_list))
                width = 0.35

                bars1 = ax.bar(x - width/2, baseline_data[metric], width, label='Baseline', color='#2ecc71')
                bars2 = ax.bar(x + width/2, pd_data[metric], width, label='PD', color='#3498db')

                ax.set_xlabel('Token Budget (TB)')
                ax.set_ylabel(ylabel)
                ax.set_title(ylabel)
                ax.set_xticks(x)
                ax.set_xticklabels(tb_list)
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')

                # 添加数值标注
                for bar, val in zip(bars1, baseline_data[metric]):
                    ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                               ha='center', va='bottom', fontsize=7, rotation=45)
                for bar, val in zip(bars2, pd_data[metric]):
                    ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                               ha='center', va='bottom', fontsize=7, rotation=45)

            plt.suptitle(f'{scenario} | BS={bs} | Baseline vs PD', fontsize=12, fontweight='bold')
            plt.tight_layout()
            output_path = output_dir / f"bs{bs}_{scenario}_comparison.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {output_path}")


def plot_improvement_heatmap(results: Dict, output_dir: Path):
    """绘制 PD 相对 Baseline 的改进热力图"""

    metrics = [
        ("throughput", "Throughput Improvement (%)", True),
        ("mean_itl_ms", "Mean ITL Improvement (%)", False),
        ("p99_itl_ms", "P99 ITL Improvement (%)", False),
    ]

    scenarios = sorted(results.keys())

    for scenario in scenarios:
        tb_values = sorted(results[scenario].keys())
        all_bs = set()
        for tb in results[scenario].values():
            all_bs.update(tb.keys())
        bs_values = sorted(all_bs)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, (metric, title, higher_better) in enumerate(metrics):
            ax = axes[idx]

            # 创建改进矩阵
            improvement_matrix = np.full((len(bs_values), len(tb_values)), np.nan)

            for i, bs in enumerate(bs_values):
                for j, tb in enumerate(tb_values):
                    if tb in results[scenario] and bs in results[scenario][tb]:
                        data = results[scenario][tb][bs]
                        if "baseline" in data and "pd" in data:
                            b_val = data["baseline"].get(metric, 0)
                            p_val = data["pd"].get(metric, 0)
                            if b_val > 0:
                                if higher_better:
                                    improvement = (p_val - b_val) / b_val * 100
                                else:
                                    improvement = (b_val - p_val) / b_val * 100
                                improvement_matrix[i, j] = improvement

            # 绘制热力图
            im = ax.imshow(improvement_matrix, cmap='RdYlGn', aspect='auto',
                          vmin=-20, vmax=20)

            ax.set_xticks(range(len(tb_values)))
            ax.set_xticklabels(tb_values)
            ax.set_yticks(range(len(bs_values)))
            ax.set_yticklabels(bs_values)
            ax.set_xlabel('Token Budget (TB)')
            ax.set_ylabel('Max Batch Size (BS)')
            ax.set_title(title)

            # 添加数值标注
            for i in range(len(bs_values)):
                for j in range(len(tb_values)):
                    val = improvement_matrix[i, j]
                    if not np.isnan(val):
                        color = 'white' if abs(val) > 10 else 'black'
                        ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                               color=color, fontsize=8)

            plt.colorbar(im, ax=ax, label='Improvement (%)')

        plt.suptitle(f'PD vs Baseline Improvement - {scenario}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        output_path = output_dir / f"improvement_heatmap_{scenario}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")


def main():
    if len(sys.argv) < 2:
        base_dir = Path("grid_search_20260110_014442")
    else:
        base_dir = Path(sys.argv[1])

    if not base_dir.exists():
        print(f"Error: Directory {base_dir} not found")
        sys.exit(1)

    output_dir = base_dir / "comparison_plots"
    output_dir.mkdir(exist_ok=True)

    print(f"Loading results from {base_dir}...")
    results = load_all_results(base_dir)

    print(f"\nFound {len(results)} scenarios")
    for scenario, data in results.items():
        tb_count = len(data)
        bs_count = len(set(bs for tb in data.values() for bs in tb.keys()))
        print(f"  {scenario}: {tb_count} TB values × {bs_count} BS values")

    print("\nGenerating plots...")

    # 1. 固定 TB，变化 BS
    print("\n1. Fixed TB, Vary BS plots...")
    plot_fixed_tb_vary_bs(results, output_dir)

    # 2. 固定 BS，变化 TB
    print("\n2. Fixed BS, Vary TB plots...")
    plot_fixed_bs_vary_tb(results, output_dir)

    # 3. 每个 TB 值单独对比图
    print("\n3. Single TB comparison plots...")
    plot_single_tb_comparison(results, output_dir)

    # 4. 每个 BS 值单独对比图
    print("\n4. Single BS comparison plots...")
    plot_single_bs_comparison(results, output_dir)

    # 5. 改进热力图
    print("\n5. Improvement heatmaps...")
    plot_improvement_heatmap(results, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
