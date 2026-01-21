#!/usr/bin/env python3
"""
分析 TB × BS 网格搜索实验结果

用法:
    python analyze_grid_search.py <experiment_dir>

输出:
    - grid_summary.json: 完整数据汇总
    - heatmap_{scenario}_{metric}.png: 热力图
    - optimal_comparison.png: 最优配置对比图
    - analysis_report.txt: 文本分析报告
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlib 未安装，跳过绑图")


def load_bench_result(filepath: Path) -> Optional[Dict]:
    """加载 benchmark 结果文件"""
    if not filepath.exists():
        return None
    try:
        with open(filepath) as f:
            return json.load(f)
    except:
        return None


def extract_metrics(bench_result: Dict) -> Dict[str, float]:
    """从 benchmark 结果提取关键指标"""
    return {
        "throughput": bench_result.get("request_throughput", 0),
        "output_throughput": bench_result.get("output_throughput", 0),
        "mean_ttft_ms": bench_result.get("mean_ttft_ms", 0),
        "median_ttft_ms": bench_result.get("median_ttft_ms", 0),
        "p99_ttft_ms": bench_result.get("p99_ttft_ms", 0),
        "mean_tpot_ms": bench_result.get("mean_tpot_ms", 0),
        "median_tpot_ms": bench_result.get("median_tpot_ms", 0),
        "p99_tpot_ms": bench_result.get("p99_tpot_ms", 0),
        "mean_itl_ms": bench_result.get("mean_itl_ms", 0),
        "median_itl_ms": bench_result.get("median_itl_ms", 0),
        "p99_itl_ms": bench_result.get("p99_itl_ms", 0),
        "mean_e2e_latency_ms": bench_result.get("mean_e2e_latency_ms", 0),
        "median_e2e_latency_ms": bench_result.get("median_e2e_latency_ms", 0),
        "p99_e2e_latency_ms": bench_result.get("p99_e2e_latency_ms", 0),
    }


def collect_grid_results(exp_dir: Path) -> Dict:
    """收集网格搜索结果

    Returns:
        {
            "tb_values": [...],
            "bs_values": [...],
            "scenarios": [...],
            "results": {
                scenario: {
                    (tb, bs): {"baseline": metrics, "pd": metrics}
                }
            }
        }
    """
    tb_values = set()
    bs_values = set()
    scenarios = set()
    results = {}

    # 遍历目录结构: tb{TB}/bs{BS}/{scenario}/
    for tb_dir in sorted(exp_dir.iterdir()):
        if not tb_dir.is_dir() or not tb_dir.name.startswith("tb"):
            continue

        tb = int(tb_dir.name[2:])
        tb_values.add(tb)

        for bs_dir in sorted(tb_dir.iterdir()):
            if not bs_dir.is_dir() or not bs_dir.name.startswith("bs"):
                continue

            bs = int(bs_dir.name[2:])
            bs_values.add(bs)

            # 文件名映射: scheduler_name -> file_suffix
            scheduler_file_map = {
                "baseline": "baseline",
                "pd_ratio": "pd_ratio",          # PD with fixed θ*
                "pd_ratio_auto": "pd_ratio_auto",  # PD with dynamic θ*
                "pd_direct": "pd_direct",        # PD with dynamic k* (DP algorithm)
                "pd": "pd",                        # Legacy: for backward compatibility
                "v0": "default_v0",                # bench_default_v0.json
                "v0_chunked": "default_v0_chunked",  # bench_default_v0_chunked.json
            }

            # 检查是否有直接在 bs_dir 下的 bench_*.json 文件（真实数据集格式）
            # 或者在子目录中（合成数据集格式）
            direct_bench_files = list(bs_dir.glob("bench_*.json"))

            if direct_bench_files:
                # 真实数据集格式: tb{TB}/bs{BS}/bench_*.json
                # 使用 "default" 作为默认 scenario 名称
                scenario = "default"
                scenarios.add(scenario)

                if scenario not in results:
                    results[scenario] = {}

                key = (tb, bs)
                if key not in results[scenario]:
                    results[scenario][key] = {}

                for scheduler, file_suffix in scheduler_file_map.items():
                    bench_file = bs_dir / f"bench_{file_suffix}.json"
                    bench_result = load_bench_result(bench_file)
                    if bench_result:
                        results[scenario][key][scheduler] = extract_metrics(bench_result)
            else:
                # 合成数据集格式: tb{TB}/bs{BS}/{scenario}/bench_*.json
                for scenario_dir in sorted(bs_dir.iterdir()):
                    if not scenario_dir.is_dir():
                        continue

                    scenario = scenario_dir.name
                    # 跳过 logs 目录和其他非 scenario 目录
                    if scenario in ("logs", "__pycache__"):
                        continue
                    scenarios.add(scenario)

                    if scenario not in results:
                        results[scenario] = {}

                    key = (tb, bs)
                    if key not in results[scenario]:
                        results[scenario][key] = {}

                    for scheduler, file_suffix in scheduler_file_map.items():
                        bench_file = scenario_dir / f"bench_{file_suffix}.json"
                        bench_result = load_bench_result(bench_file)
                        if bench_result:
                            results[scenario][key][scheduler] = extract_metrics(bench_result)

    return {
        "tb_values": sorted(tb_values),
        "bs_values": sorted(bs_values),
        "scenarios": sorted(scenarios),
        "results": results
    }


def find_optimal_configs(data: Dict) -> Dict:
    """找到每个 scenario 和 scheduler 的最优配置"""
    optimal = {}

    # 所有支持的调度器
    all_schedulers = ["baseline", "pd_ratio", "pd_ratio_auto", "pd_direct", "pd", "v0", "v0_chunked"]

    for scenario in data["scenarios"]:
        optimal[scenario] = {}
        results = data["results"].get(scenario, {})

        for scheduler in all_schedulers:
            best_key = None
            best_throughput = 0

            for (tb, bs), sched_results in results.items():
                if scheduler in sched_results:
                    tp = sched_results[scheduler].get("throughput", 0)
                    if tp > best_throughput:
                        best_throughput = tp
                        best_key = (tb, bs)

            if best_key:
                optimal[scenario][scheduler] = {
                    "tb": best_key[0],
                    "bs": best_key[1],
                    "metrics": results[best_key][scheduler]
                }

    return optimal


def compute_improvement_grid(data: Dict, scenario: str, metric: str, higher_better: bool = True) -> Tuple[np.ndarray, List, List]:
    """计算 PD 相对 baseline 的改进网格

    Returns:
        (improvement_matrix, tb_values, bs_values)
    """
    tb_values = data["tb_values"]
    bs_values = data["bs_values"]
    results = data["results"].get(scenario, {})

    matrix = np.full((len(tb_values), len(bs_values)), np.nan)

    for i, tb in enumerate(tb_values):
        for j, bs in enumerate(bs_values):
            key = (tb, bs)
            if key in results:
                baseline = results[key].get("baseline", {}).get(metric, 0)
                pd = results[key].get("pd", {}).get(metric, 0)

                if baseline > 0:
                    if higher_better:
                        improvement = (pd - baseline) / baseline * 100
                    else:
                        improvement = (baseline - pd) / baseline * 100
                    matrix[i, j] = improvement

    return matrix, tb_values, bs_values


def plot_heatmaps(data: Dict, output_dir: Path):
    """绘制热力图"""
    if not HAS_MATPLOTLIB:
        return

    metrics_to_plot = [
        ("throughput", "Throughput Improvement (%)", True),
        ("mean_itl_ms", "Mean ITL Improvement (%)", False),
        ("mean_ttft_ms", "Mean TTFT Improvement (%)", False),
        ("output_throughput", "Output Throughput Improvement (%)", True),
    ]

    for scenario in data["scenarios"]:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for idx, (metric, title, higher_better) in enumerate(metrics_to_plot):
            ax = axes[idx]
            matrix, tb_vals, bs_vals = compute_improvement_grid(data, scenario, metric, higher_better)

            # 设置颜色范围 (绿色=改进, 红色=退步)
            vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 10)
            vmin = -vmax

            cmap = plt.cm.RdYlGn  # 红黄绿
            im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

            # 设置刻度
            ax.set_xticks(range(len(bs_vals)))
            ax.set_xticklabels([str(b) for b in bs_vals], rotation=45)
            ax.set_yticks(range(len(tb_vals)))
            ax.set_yticklabels([str(t) for t in tb_vals])

            ax.set_xlabel('max_num_seqs (BS)')
            ax.set_ylabel('max_num_batched_tokens (TB)')
            ax.set_title(title)

            # 添加数值标注
            for i in range(len(tb_vals)):
                for j in range(len(bs_vals)):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        color = 'white' if abs(val) > vmax * 0.5 else 'black'
                        ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                                color=color, fontsize=8)

            plt.colorbar(im, ax=ax, shrink=0.8)

        plt.suptitle(f'{scenario}: PD vs Baseline Improvement', fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = output_dir / f"heatmap_{scenario}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")


def plot_optimal_comparison(optimal: Dict, output_dir: Path):
    """绘制最优配置对比图 - 每个指标单独一个子图，支持 v0, baseline, pd 三个调度器"""
    if not HAS_MATPLOTLIB:
        return

    scenarios = list(optimal.keys())
    n_scenarios = len(scenarios)

    # 定义要绘制的指标，每个指标单独一个子图
    metrics_to_plot = [
        ("throughput", "Request Throughput (req/s)", True),
        ("output_throughput", "Output Throughput (tok/s)", True),
        ("mean_itl_ms", "Mean ITL (ms)", False),
        ("p99_itl_ms", "P99 ITL (ms)", False),
        ("mean_ttft_ms", "Mean TTFT (ms)", False),
        ("p99_ttft_ms", "P99 TTFT (ms)", False),
    ]

    n_metrics = len(metrics_to_plot)

    # 每个 scenario 一行，每个指标一列
    fig, axes = plt.subplots(n_scenarios, n_metrics, figsize=(4 * n_metrics, 4 * n_scenarios))
    if n_scenarios == 1:
        axes = axes.reshape(1, -1)

    # 调度器配置: (key, color, label)
    schedulers = [
        ("v0", '#e74c3c', 'V0'),
        ("v0_chunked", '#e67e22', 'V0+Chunk'),
        ("baseline", '#2ecc71', 'Baseline'),
        ("pd_ratio", '#3498db', 'PD (ratio)'),    # Ratio mode (k* = θ* × N)
        ("pd_ratio_auto", '#9b59b6', 'PD (ratio auto)'),  # Ratio mode with auto θ*
        ("pd_direct", '#1abc9c', 'PD (direct)'), # Direct mode (auto k*)
        ("pd", '#5dade2', 'PD'),                   # Legacy
    ]

    for i, scenario in enumerate(scenarios):
        # 获取各调度器的最优配置
        sched_opts = {s[0]: optimal[scenario].get(s[0], {}) for s in schedulers}

        # 过滤掉没有数据的调度器
        available_scheds = [(key, color, label) for key, color, label in schedulers if sched_opts.get(key)]

        if len(available_scheds) < 2:
            continue

        for j, (metric, ylabel, higher_better) in enumerate(metrics_to_plot):
            ax = axes[i, j]

            # 收集各调度器的值
            values = []
            colors = []
            labels = []
            for key, color, label in available_scheds:
                opt = sched_opts[key]
                val = opt["metrics"].get(metric, 0)
                values.append(val)
                colors.append(color)
                labels.append(f'{label}\nTB={opt["tb"]}\nBS={opt["bs"]}')

            x = np.arange(len(values))
            width = 0.6 if len(values) <= 3 else 0.4
            bars = ax.bar(x, values, width=width, color=colors)

            # 计算 PD 相对 baseline 的改进
            baseline_val = sched_opts.get("baseline", {}).get("metrics", {}).get(metric, 0)
            pd_val = sched_opts.get("pd", {}).get("metrics", {}).get(metric, 0)
            if baseline_val > 0 and pd_val > 0:
                if higher_better:
                    improvement = (pd_val - baseline_val) / baseline_val * 100
                else:
                    improvement = (baseline_val - pd_val) / baseline_val * 100
                imp_str = f"PD vs Base: {improvement:+.1f}%"
            else:
                imp_str = ""
                improvement = 0

            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_ylabel(ylabel.split('(')[1].replace(')', '') if '(' in ylabel else '')

            # 第一行显示指标名称作为标题
            if i == 0:
                ax.set_title(ylabel, fontsize=10)

            # 第一列显示 scenario 名称
            if j == 0:
                ax.set_ylabel(f'{scenario}\n{ylabel.split("(")[1].replace(")", "")}' if '(' in ylabel else scenario, fontsize=9)

            # 在柱子上方标注数值
            for bar, val in zip(bars, values):
                ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=7)

            # 标注改进百分比
            if imp_str:
                ax.annotate(imp_str, xy=(0.5, 0.95), xycoords='axes fraction',
                           ha='center', va='top', fontsize=8, fontweight='bold',
                           color='green' if improvement > 0 else 'red' if improvement < 0 else 'gray')

            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(bottom=0)

    plt.suptitle('Optimal Configuration Comparison: V0 / Baseline / PD variants', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / "optimal_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_report(data: Dict, optimal: Dict) -> str:
    """生成文本分析报告，支持 v0, v0_chunked, baseline, pd_ratio, pd_ratio_auto, pd_direct 等调度器"""
    lines = []
    lines.append("=" * 80)
    lines.append("TB × BS 网格搜索分析报告")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"TB 值: {data['tb_values']}")
    lines.append(f"BS 值: {data['bs_values']}")
    lines.append(f"Scenarios: {data['scenarios']}")
    lines.append("")

    # 所有可能的调度器 (按显示顺序排列)
    all_schedulers = [
        ("v0", "V0"),
        ("v0_chunked", "V0+Chunk"),
        ("baseline", "Baseline"),
        ("pd_ratio", "PD (ratio)"),
        ("pd_ratio_auto", "PD (ratio auto)"),
        ("pd_direct", "PD (direct)"),
        ("pd", "PD (legacy)"),
    ]

    for scenario in data["scenarios"]:
        lines.append("=" * 80)
        lines.append(f"Scenario: {scenario}")
        lines.append("=" * 80)

        # 收集可用的调度器
        available = []
        for key, display_name in all_schedulers:
            opt = optimal[scenario].get(key, {})
            if opt:
                available.append((key, display_name, opt))

        if len(available) >= 2:
            lines.append("")
            lines.append("最优配置:")
            for key, display_name, opt in available:
                lines.append(f"  {display_name:>15}: TB={opt['tb']}, BS={opt['bs']}")
            lines.append("")

            # 获取特定调度器的最优配置 (用于改进计算)
            baseline_opt = optimal[scenario].get("baseline", {})
            v0_opt = optimal[scenario].get("v0", {})
            # 获取 PD 变体用于对比
            pd_direct_opt = optimal[scenario].get("pd_direct", {})
            pd_ratio_opt = optimal[scenario].get("pd_ratio", {})

            # 对比关键指标
            header = f"{'Metric':<25}"
            for key, display_name, opt in available:
                header += f" {display_name:<15}"
            # 添加两列对比
            if baseline_opt and pd_direct_opt:
                header += f" {'PD(dir) vs Base':<15}"
            if baseline_opt and pd_ratio_opt:
                header += f" {'PD(rat) vs Base':<15}"
            lines.append(header)
            extra_cols = (1 if baseline_opt and pd_direct_opt else 0) + (1 if baseline_opt and pd_ratio_opt else 0)
            lines.append("-" * (25 + 15 * len(available) + 15 * extra_cols))

            metrics_compare = [
                ("throughput", "Throughput (req/s)", True),
                ("output_throughput", "Output (tok/s)", True),
                ("mean_itl_ms", "Mean ITL (ms)", False),
                ("mean_ttft_ms", "Mean TTFT (ms)", False),
                ("p99_itl_ms", "P99 ITL (ms)", False),
                ("p99_ttft_ms", "P99 TTFT (ms)", False),
                ("mean_e2e_latency_ms", "Mean E2E (ms)", False),
            ]

            for metric, metric_name, higher_better in metrics_compare:
                row = f"{metric_name:<25}"
                for key, display_name, opt in available:
                    val = opt["metrics"].get(metric, 0)
                    row += f" {val:<15.2f}"

                # 计算 PD(DP) vs Baseline 改进
                if baseline_opt and pd_direct_opt:
                    b_val = baseline_opt["metrics"].get(metric, 0)
                    p_val = pd_direct_opt["metrics"].get(metric, 0)
                    if b_val > 0:
                        if higher_better:
                            improvement = (p_val - b_val) / b_val * 100
                        else:
                            improvement = (b_val - p_val) / b_val * 100
                        imp_str = f"{improvement:+.2f}%"
                    else:
                        imp_str = "N/A"
                    row += f" {imp_str:<15}"

                # 计算 PD(θ*) vs Baseline 改进
                if baseline_opt and pd_ratio_opt:
                    b_val = baseline_opt["metrics"].get(metric, 0)
                    p_val = pd_ratio_opt["metrics"].get(metric, 0)
                    if b_val > 0:
                        if higher_better:
                            improvement = (p_val - b_val) / b_val * 100
                        else:
                            improvement = (b_val - p_val) / b_val * 100
                        imp_str = f"{improvement:+.2f}%"
                    else:
                        imp_str = "N/A"
                    row += f" {imp_str:<15}"

                lines.append(row)

            # 判断胜负 (基于吞吐量)
            throughputs = {display_name: opt["metrics"].get("throughput", 0) for key, display_name, opt in available}
            winner = max(throughputs, key=throughputs.get)
            best_tp = throughputs[winner]

            lines.append("")
            lines.append(f"结论: {winner} 在吞吐量上胜出 ({best_tp:.2f} req/s)")

            # 如果有 v0，显示 baseline 相对 v0 的改进
            if v0_opt and baseline_opt:
                v0_tp = v0_opt["metrics"].get("throughput", 0)
                b_tp = baseline_opt["metrics"].get("throughput", 0)
                if v0_tp > 0:
                    imp = (b_tp - v0_tp) / v0_tp * 100
                    lines.append(f"  Baseline vs V0: {imp:+.2f}%")

            # 如果有 v0，显示 PD 相对 v0 的改进
            if v0_opt and pd_direct_opt:
                v0_tp = v0_opt["metrics"].get("throughput", 0)
                p_tp = pd_direct_opt["metrics"].get("throughput", 0)
                if v0_tp > 0:
                    imp = (p_tp - v0_tp) / v0_tp * 100
                    lines.append(f"  PD(direct) vs V0: {imp:+.2f}%")
            if v0_opt and pd_ratio_opt:
                v0_tp = v0_opt["metrics"].get("throughput", 0)
                p_tp = pd_ratio_opt["metrics"].get("throughput", 0)
                if v0_tp > 0:
                    imp = (p_tp - v0_tp) / v0_tp * 100
                    lines.append(f"  PD(ratio) vs V0: {imp:+.2f}%")

        lines.append("")

    # 总结
    lines.append("=" * 80)
    lines.append("总体结论")
    lines.append("=" * 80)

    # 统计各调度器在各 scenario 下的胜出情况
    # 调度器分组: baseline, PD variants (pd_ratio, pd_direct, pd_ratio_auto, pd)
    scheduler_groups = {
        "baseline": ["baseline"],
        "pd_ratio": ["pd_ratio"],        # Ratio mode (k* = θ* × N)
        "pd_direct": ["pd_direct"],      # Direct mode (auto k*)
        "pd_ratio_auto": ["pd_ratio_auto"],
        "v0": ["v0"],
        "v0_chunked": ["v0_chunked"],
        "pd_legacy": ["pd"],
    }

    # 收集每个 scenario 的胜者
    scenario_winners = {}
    for scenario in data["scenarios"]:
        available = {}
        for key, display_name in all_schedulers:
            opt = optimal[scenario].get(key, {})
            if opt:
                available[key] = opt["metrics"].get("throughput", 0)

        if available:
            winner = max(available, key=available.get)
            scenario_winners[scenario] = winner

    # 统计胜出次数
    wins = {}
    for key, _ in all_schedulers:
        wins[key] = sum(1 for w in scenario_winners.values() if w == key)

    total = len(data['scenarios'])

    lines.append("")
    lines.append("各 Scenario 胜者 (基于最高吞吐量):")
    for scenario, winner in scenario_winners.items():
        display_name = dict(all_schedulers).get(winner, winner)
        opt = optimal[scenario].get(winner, {})
        tp = opt.get("metrics", {}).get("throughput", 0)
        lines.append(f"  {scenario}: {display_name} ({tp:.2f} req/s)")

    lines.append("")
    lines.append("胜出统计:")
    for key, display_name in all_schedulers:
        if wins.get(key, 0) > 0:
            lines.append(f"  {display_name}: {wins[key]}/{total} scenarios")

    # 汇总: Baseline vs 所有 PD 变体
    pd_variants = ["pd_ratio", "pd_ratio_auto", "pd_direct", "pd"]
    baseline_wins = wins.get("baseline", 0)
    pd_total_wins = sum(wins.get(v, 0) for v in pd_variants)

    lines.append("")
    if baseline_wins > pd_total_wins:
        lines.append(f"总体: BASELINE 领先 ({baseline_wins} vs {pd_total_wins})")
    elif pd_total_wins > baseline_wins:
        lines.append(f"总体: PD 系列领先 ({pd_total_wins} vs {baseline_wins})")
    else:
        lines.append(f"总体: BASELINE 与 PD 系列持平 ({baseline_wins} vs {pd_total_wins})")

    lines.append("")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("用法: python analyze_grid_search.py <experiment_dir>")
        sys.exit(1)

    exp_dir = Path(sys.argv[1])
    if not exp_dir.exists():
        print(f"目录不存在: {exp_dir}")
        sys.exit(1)

    print(f"分析实验目录: {exp_dir}")
    print("")

    # 收集数据
    data = collect_grid_results(exp_dir)

    if not data["results"]:
        print("未找到任何实验结果")
        sys.exit(1)

    print(f"找到 {len(data['tb_values'])} 个 TB 值: {data['tb_values']}")
    print(f"找到 {len(data['bs_values'])} 个 BS 值: {data['bs_values']}")
    print(f"找到 {len(data['scenarios'])} 个 scenarios: {data['scenarios']}")
    print("")

    # 找最优配置
    optimal = find_optimal_configs(data)

    # 生成报告
    report = generate_report(data, optimal)
    print(report)

    # 保存报告
    report_path = exp_dir / "analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n报告已保存: {report_path}")

    # 绘图
    plot_heatmaps(data, exp_dir)
    plot_optimal_comparison(optimal, exp_dir)

    # 保存汇总 JSON
    summary = {
        "tb_values": data["tb_values"],
        "bs_values": data["bs_values"],
        "scenarios": data["scenarios"],
        "optimal": optimal,
        "all_results": {
            scenario: {
                f"tb{tb}_bs{bs}": sched_results
                for (tb, bs), sched_results in results.items()
            }
            for scenario, results in data["results"].items()
        }
    }

    summary_path = exp_dir / "grid_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"汇总数据已保存: {summary_path}")


if __name__ == "__main__":
    main()
