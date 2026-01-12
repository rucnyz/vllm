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

            for scenario_dir in sorted(bs_dir.iterdir()):
                if not scenario_dir.is_dir():
                    continue

                scenario = scenario_dir.name
                scenarios.add(scenario)

                if scenario not in results:
                    results[scenario] = {}

                key = (tb, bs)
                if key not in results[scenario]:
                    results[scenario][key] = {}

                # 加载 baseline 和 pd 结果
                for scheduler in ["baseline", "pd"]:
                    bench_file = scenario_dir / f"bench_{scheduler}.json"
                    bench_result = load_bench_result(bench_file)
                    if bench_result:
                        results[scenario][key][scheduler] = extract_metrics(bench_result)

    return {
        "tb_values": sorted(tb_values),
        "bs_values": sorted(bs_values),
        "scenarios": sorted(scenarios),
        "results": results
    }


def find_optimal_configs(data: Dict, verbose: bool = True) -> Dict:
    """找到每个 scenario 和 scheduler 的最优配置

    Args:
        data: 网格搜索数据
        verbose: 是否打印详细的选择过程
    """
    optimal = {}

    for scenario in data["scenarios"]:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Scenario: {scenario} - 最优配置选择过程")
            print(f"{'='*60}")

        optimal[scenario] = {}
        results = data["results"].get(scenario, {})

        for scheduler in ["baseline", "pd"]:
            if verbose:
                print(f"\n  [{scheduler.upper()}] 调度器配置排名 (按吞吐量降序):")
                print(f"  {'排名':<6} {'TB':<8} {'BS':<8} {'Throughput':<15} {'Output Tput':<15} {'Mean ITL':<12}")
                print(f"  {'-'*65}")

            # 收集所有配置的吞吐量
            config_throughputs = []
            for (tb, bs), sched_results in results.items():
                if scheduler in sched_results:
                    tp = sched_results[scheduler].get("throughput", 0)
                    output_tp = sched_results[scheduler].get("output_throughput", 0)
                    mean_itl = sched_results[scheduler].get("mean_itl_ms", 0)
                    config_throughputs.append({
                        "tb": tb,
                        "bs": bs,
                        "throughput": tp,
                        "output_throughput": output_tp,
                        "mean_itl_ms": mean_itl,
                        "metrics": sched_results[scheduler]
                    })

            # 按吞吐量排序
            config_throughputs.sort(key=lambda x: x["throughput"], reverse=True)

            if verbose:
                for rank, cfg in enumerate(config_throughputs, 1):
                    marker = " ★" if rank == 1 else ""
                    print(f"  {rank:<6} {cfg['tb']:<8} {cfg['bs']:<8} {cfg['throughput']:<15.2f} {cfg['output_throughput']:<15.2f} {cfg['mean_itl_ms']:<12.2f}{marker}")

            if config_throughputs:
                best = config_throughputs[0]
                optimal[scenario][scheduler] = {
                    "tb": best["tb"],
                    "bs": best["bs"],
                    "metrics": best["metrics"],
                    "rank_info": config_throughputs  # 保存排名信息
                }

                if verbose and len(config_throughputs) > 1:
                    second = config_throughputs[1]
                    gap = (best["throughput"] - second["throughput"]) / second["throughput"] * 100 if second["throughput"] > 0 else 0
                    print(f"\n  → 最优: TB={best['tb']}, BS={best['bs']}, Throughput={best['throughput']:.2f}")
                    print(f"  → 与第二名差距: +{gap:.2f}%")

        # 对比 baseline 和 pd 的最优配置
        if verbose and "baseline" in optimal[scenario] and "pd" in optimal[scenario]:
            b_opt = optimal[scenario]["baseline"]
            p_opt = optimal[scenario]["pd"]
            b_tp = b_opt["metrics"]["throughput"]
            p_tp = p_opt["metrics"]["throughput"]

            print(f"\n  {'─'*60}")
            print(f"  最优配置对比:")
            print(f"    Baseline: TB={b_opt['tb']}, BS={b_opt['bs']} → {b_tp:.2f} req/s")
            print(f"    PD:       TB={p_opt['tb']}, BS={p_opt['bs']} → {p_tp:.2f} req/s")

            if b_tp > 0:
                improvement = (p_tp - b_tp) / b_tp * 100
                winner = "PD" if improvement > 0 else "Baseline"
                print(f"    结论: {winner} 胜出 ({improvement:+.2f}%)")

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

        plt.show()
        output_path = output_dir / f"heatmap_{scenario}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")


def plot_optimal_comparison(optimal: Dict, output_dir: Path):
    """绘制最优配置对比图 - 每个指标单独一个子图"""
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

    width = 0.35

    for i, scenario in enumerate(scenarios):
        baseline_opt = optimal[scenario].get("baseline", {})
        pd_opt = optimal[scenario].get("pd", {})

        if not baseline_opt or not pd_opt:
            continue

        for j, (metric, ylabel, higher_better) in enumerate(metrics_to_plot):
            ax = axes[i, j]

            b_val = baseline_opt["metrics"].get(metric, 0)
            p_val = pd_opt["metrics"].get(metric, 0)

            x = np.arange(2)
            colors = ['#2ecc71', '#3498db']
            bars = ax.bar(x, [b_val, p_val], width=0.6, color=colors)

            # 计算改进百分比
            if b_val > 0:
                if higher_better:
                    improvement = (p_val - b_val) / b_val * 100
                else:
                    improvement = (b_val - p_val) / b_val * 100
                imp_str = f"{improvement:+.1f}%"
            else:
                imp_str = "N/A"

            ax.set_xticks(x)
            ax.set_xticklabels([f'Baseline\nTB={baseline_opt["tb"]}\nBS={baseline_opt["bs"]}',
                               f'PD\nTB={pd_opt["tb"]}\nBS={pd_opt["bs"]}'], fontsize=8)
            ax.set_ylabel(ylabel.split('(')[1].replace(')', '') if '(' in ylabel else '')

            # 第一行显示指标名称作为标题
            if i == 0:
                ax.set_title(ylabel, fontsize=10)

            # 第一列显示 scenario 名称
            if j == 0:
                ax.set_ylabel(f'{scenario}\n{ylabel.split("(")[1].replace(")", "")}' if '(' in ylabel else scenario, fontsize=9)

            # 在柱子上方标注数值和改进
            for bar, val in zip(bars, [b_val, p_val]):
                ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=8)

            # 标注改进百分比
            ax.annotate(imp_str, xy=(0.5, 0.95), xycoords='axes fraction',
                       ha='center', va='top', fontsize=9, fontweight='bold',
                       color='green' if improvement > 0 else 'red' if improvement < 0 else 'gray')

            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(bottom=0)

    plt.suptitle('Optimal Configuration Comparison: Baseline vs PD Scheduler', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    output_path = output_dir / "optimal_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_report(data: Dict, optimal: Dict) -> str:
    """生成文本分析报告"""
    lines = []
    lines.append("=" * 80)
    lines.append("TB × BS 网格搜索分析报告")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"TB 值: {data['tb_values']}")
    lines.append(f"BS 值: {data['bs_values']}")
    lines.append(f"Scenarios: {data['scenarios']}")
    lines.append("")

    for scenario in data["scenarios"]:
        lines.append("=" * 80)
        lines.append(f"Scenario: {scenario}")
        lines.append("=" * 80)

        baseline_opt = optimal[scenario].get("baseline", {})
        pd_opt = optimal[scenario].get("pd", {})

        if baseline_opt and pd_opt:
            lines.append("")
            lines.append("最优配置:")
            lines.append(f"  Baseline: TB={baseline_opt['tb']}, BS={baseline_opt['bs']}")
            lines.append(f"  PD:       TB={pd_opt['tb']}, BS={pd_opt['bs']}")
            lines.append("")

            # 对比关键指标
            lines.append(f"{'Metric':<25} {'Baseline':<15} {'PD':<15} {'Improvement':<15}")
            lines.append("-" * 70)

            metrics_compare = [
                ("throughput", "Throughput (req/s)", True),
                ("output_throughput", "Output (tok/s)", True),
                ("mean_itl_ms", "Mean ITL (ms)", False),
                ("mean_ttft_ms", "Mean TTFT (ms)", False),
                ("p99_itl_ms", "P99 ITL (ms)", False),
                ("p99_ttft_ms", "P99 TTFT (ms)", False),
                ("mean_e2e_latency_ms", "Mean E2E (ms)", False),
            ]

            for metric, name, higher_better in metrics_compare:
                b_val = baseline_opt["metrics"].get(metric, 0)
                p_val = pd_opt["metrics"].get(metric, 0)

                if b_val > 0:
                    if higher_better:
                        improvement = (p_val - b_val) / b_val * 100
                    else:
                        improvement = (b_val - p_val) / b_val * 100
                    imp_str = f"{improvement:+.2f}%"
                else:
                    imp_str = "N/A"

                lines.append(f"{name:<25} {b_val:<15.2f} {p_val:<15.2f} {imp_str:<15}")

            # 判断胜负
            b_tp = baseline_opt["metrics"].get("throughput", 0)
            p_tp = pd_opt["metrics"].get("throughput", 0)
            winner = "PD Scheduler" if p_tp > b_tp else "Baseline"
            improvement = (p_tp - b_tp) / b_tp * 100 if b_tp > 0 else 0

            lines.append("")
            lines.append(f"结论: {winner} 在吞吐量上胜出 ({improvement:+.2f}%)")

        lines.append("")

    # 总结
    lines.append("=" * 80)
    lines.append("总体结论")
    lines.append("=" * 80)

    pd_wins = 0
    baseline_wins = 0
    for scenario in data["scenarios"]:
        baseline_opt = optimal[scenario].get("baseline", {})
        pd_opt = optimal[scenario].get("pd", {})
        if baseline_opt and pd_opt:
            b_tp = baseline_opt["metrics"].get("throughput", 0)
            p_tp = pd_opt["metrics"].get("throughput", 0)
            if p_tp > b_tp:
                pd_wins += 1
            else:
                baseline_wins += 1

    lines.append(f"PD Scheduler 胜出: {pd_wins}/{len(data['scenarios'])} scenarios")
    lines.append(f"Baseline 胜出: {baseline_wins}/{len(data['scenarios'])} scenarios")
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
