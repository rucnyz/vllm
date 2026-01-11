#!/usr/bin/env python3
"""
分析 Request 数量扩展实验结果

验证假设: 随着 request 数量增加，PD scheduler 的优势更明显

用法:
    python analyze_request_scaling.py <experiment_dir>

输出:
    - scaling_summary.json: 汇总数据
    - throughput_scaling.png: 吞吐量随 request 数量变化
    - latency_scaling.png: 延迟随 request 数量变化
    - improvement_scaling.png: PD 相对 baseline 的改进百分比
    - analysis_report.txt: 文本报告
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

try:
    import matplotlib.pyplot as plt
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
        "mean_itl_ms": bench_result.get("mean_itl_ms", 0),
        "median_itl_ms": bench_result.get("median_itl_ms", 0),
        "p99_itl_ms": bench_result.get("p99_itl_ms", 0),
        "mean_e2e_latency_ms": bench_result.get("mean_e2e_latency_ms", 0),
        "duration": bench_result.get("duration", 0),  # 运行时间(秒)
    }


def collect_results(exp_dir: Path) -> Dict:
    """收集所有实验结果

    Returns:
        {
            "request_counts": [500, 1000, ...],
            "configs": {
                "pd_optimal": {500: metrics, 1000: metrics, ...},
                "baseline_same": {...},
                "baseline_default": {...}
            }
        }
    """
    request_counts = set()
    configs = {"pd_optimal": {}, "baseline_same": {}, "baseline_default": {}}

    for req_dir in sorted(exp_dir.iterdir()):
        if not req_dir.is_dir() or not req_dir.name.startswith("requests_"):
            continue

        num_requests = int(req_dir.name.split("_")[1])
        request_counts.add(num_requests)

        for config_name in configs.keys():
            bench_file = req_dir / f"bench_{config_name}.json"
            bench_result = load_bench_result(bench_file)
            if bench_result:
                configs[config_name][num_requests] = extract_metrics(bench_result)

    return {
        "request_counts": sorted(request_counts),
        "configs": configs
    }


def compute_improvements(data: Dict) -> Dict:
    """计算 PD 相对于 baseline 的改进

    Returns:
        {
            "vs_same": {request_count: {metric: improvement%}},
            "vs_default": {request_count: {metric: improvement%}}
        }
    """
    improvements = {"vs_same": {}, "vs_default": {}}

    metrics_higher_better = ["throughput", "output_throughput"]
    metrics_lower_better = ["mean_ttft_ms", "p99_ttft_ms", "mean_itl_ms", "p99_itl_ms", "mean_e2e_latency_ms"]

    for req_count in data["request_counts"]:
        pd = data["configs"]["pd_optimal"].get(req_count, {})
        baseline_same = data["configs"]["baseline_same"].get(req_count, {})
        baseline_default = data["configs"]["baseline_default"].get(req_count, {})

        improvements["vs_same"][req_count] = {}
        improvements["vs_default"][req_count] = {}

        for metric in metrics_higher_better:
            pd_val = pd.get(metric, 0)
            same_val = baseline_same.get(metric, 0)
            default_val = baseline_default.get(metric, 0)

            if same_val > 0:
                improvements["vs_same"][req_count][metric] = (pd_val - same_val) / same_val * 100
            if default_val > 0:
                improvements["vs_default"][req_count][metric] = (pd_val - default_val) / default_val * 100

        for metric in metrics_lower_better:
            pd_val = pd.get(metric, 0)
            same_val = baseline_same.get(metric, 0)
            default_val = baseline_default.get(metric, 0)

            if same_val > 0:
                improvements["vs_same"][req_count][metric] = (same_val - pd_val) / same_val * 100
            if default_val > 0:
                improvements["vs_default"][req_count][metric] = (default_val - pd_val) / default_val * 100

    return improvements


def plot_throughput_scaling(data: Dict, output_dir: Path):
    """绘制吞吐量随 request 数量变化的图"""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    request_counts = data["request_counts"]

    for idx, (metric, ylabel) in enumerate([
        ("throughput", "Request Throughput (req/s)"),
        ("output_throughput", "Output Throughput (tok/s)")
    ]):
        ax = axes[idx]

        for config_name, label, color, marker in [
            ("pd_optimal", "PD Scheduler (optimal TB/BS)", "#3498db", "o"),
            ("baseline_same", "Baseline (same TB/BS)", "#2ecc71", "s"),
            ("baseline_default", "Baseline (default TB/BS)", "#e74c3c", "^"),
        ]:
            values = [data["configs"][config_name].get(rc, {}).get(metric, 0)
                     for rc in request_counts]
            ax.plot(request_counts, values, marker=marker, label=label,
                   color=color, linewidth=2, markersize=8)

        ax.set_xlabel('Number of Requests')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

    plt.suptitle('Throughput vs Number of Requests', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / "throughput_scaling.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_latency_scaling(data: Dict, output_dir: Path):
    """绘制延迟随 request 数量变化的图"""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    request_counts = data["request_counts"]

    metrics = [
        ("mean_itl_ms", "Mean ITL (ms)"),
        ("p99_itl_ms", "P99 ITL (ms)"),
        ("mean_ttft_ms", "Mean TTFT (ms)"),
        ("p99_ttft_ms", "P99 TTFT (ms)"),
    ]

    for idx, (metric, ylabel) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        for config_name, label, color, marker in [
            ("pd_optimal", "PD Scheduler", "#3498db", "o"),
            ("baseline_same", "Baseline (same)", "#2ecc71", "s"),
            ("baseline_default", "Baseline (default)", "#e74c3c", "^"),
        ]:
            values = [data["configs"][config_name].get(rc, {}).get(metric, 0)
                     for rc in request_counts]
            ax.plot(request_counts, values, marker=marker, label=label,
                   color=color, linewidth=2, markersize=8)

        ax.set_xlabel('Number of Requests')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

    plt.suptitle('Latency vs Number of Requests (lower is better)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / "latency_scaling.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_improvement_scaling(data: Dict, improvements: Dict, output_dir: Path):
    """绘制 PD 相对 baseline 的改进百分比随 request 数量变化"""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    request_counts = data["request_counts"]

    metrics = [
        ("throughput", "Throughput Improvement (%)", True),
        ("output_throughput", "Output Throughput Improvement (%)", True),
        ("mean_itl_ms", "Mean ITL Improvement (%)", False),
        ("mean_ttft_ms", "Mean TTFT Improvement (%)", False),
    ]

    for idx, (metric, ylabel, higher_better) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        # vs same TB/BS
        vs_same = [improvements["vs_same"].get(rc, {}).get(metric, 0) for rc in request_counts]
        ax.plot(request_counts, vs_same, marker="o", label="PD vs Baseline (same TB/BS)",
               color="#3498db", linewidth=2, markersize=8)

        # vs default TB/BS
        vs_default = [improvements["vs_default"].get(rc, {}).get(metric, 0) for rc in request_counts]
        ax.plot(request_counts, vs_default, marker="s", label="PD vs Baseline (default TB/BS)",
               color="#e74c3c", linewidth=2, markersize=8)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Number of Requests')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

        # 填充正负区域
        ax.fill_between(request_counts, 0, vs_same, alpha=0.1, color="#3498db")
        ax.fill_between(request_counts, 0, vs_default, alpha=0.1, color="#e74c3c")

    plt.suptitle('PD Scheduler Improvement vs Baseline (positive = PD better)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / "improvement_scaling.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_duration_scaling(data: Dict, output_dir: Path):
    """绘制运行时间随 request 数量变化的图"""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    request_counts = data["request_counts"]

    # 左图: 绝对运行时间
    ax = axes[0]
    for config_name, label, color, marker in [
        ("pd_optimal", "PD Scheduler (optimal TB/BS)", "#3498db", "o"),
        ("baseline_same", "Baseline (same TB/BS)", "#2ecc71", "s"),
        ("baseline_default", "Baseline (default TB/BS)", "#e74c3c", "^"),
    ]:
        durations = [data["configs"][config_name].get(rc, {}).get("duration", 0)
                    for rc in request_counts]
        ax.plot(request_counts, durations, marker=marker, label=label,
               color=color, linewidth=2, markersize=8)

    ax.set_xlabel('Number of Requests')
    ax.set_ylabel('Duration (seconds)')
    ax.set_title('Total Runtime vs Number of Requests')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # 右图: PD 达到 baseline 相同吞吐量所需的 request 数量 / 时间
    ax = axes[1]

    # 计算: 对于每个 baseline_default 的吞吐量，PD 需要多少 request 数量才能达到
    baseline_throughputs = []
    baseline_durations = []
    pd_equiv_requests = []
    pd_equiv_durations = []

    for rc in request_counts:
        baseline = data["configs"]["baseline_default"].get(rc, {})
        if not baseline:
            continue
        baseline_tp = baseline.get("throughput", 0)
        baseline_dur = baseline.get("duration", 0)

        if baseline_tp <= 0 or baseline_dur <= 0:
            continue

        baseline_throughputs.append(baseline_tp)
        baseline_durations.append(baseline_dur)

        # 估算 PD 处理多少 request 能达到相同的处理量
        # baseline 在 duration 秒内处理了 rc 个 request
        # PD 的吞吐量是多少?
        pd = data["configs"]["pd_optimal"].get(rc, {})
        pd_tp = pd.get("throughput", 0)

        if pd_tp > 0:
            # baseline 的总处理量 = baseline_dur * baseline_tp (约等于 rc)
            # PD 达到相同处理量需要时间 = rc / pd_tp
            pd_time_for_same_requests = rc / pd_tp
            pd_equiv_requests.append(rc)
            pd_equiv_durations.append(pd_time_for_same_requests)

    # 绘制时间对比
    if baseline_durations and pd_equiv_durations:
        x = request_counts[:len(baseline_durations)]
        ax.bar(np.arange(len(x)) - 0.2, baseline_durations, 0.4,
               label='Baseline (default)', color='#e74c3c', alpha=0.8)
        ax.bar(np.arange(len(x)) + 0.2, pd_equiv_durations, 0.4,
               label='PD Scheduler', color='#3498db', alpha=0.8)

        ax.set_xticks(np.arange(len(x)))
        ax.set_xticklabels([str(r) for r in x], rotation=45)
        ax.set_xlabel('Number of Requests')
        ax.set_ylabel('Time to Complete (seconds)')
        ax.set_title('Time to Complete Same Number of Requests')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 添加时间差标注
        for i, (base_dur, pd_dur) in enumerate(zip(baseline_durations, pd_equiv_durations)):
            diff_pct = (base_dur - pd_dur) / base_dur * 100 if base_dur > 0 else 0
            if diff_pct > 0:
                ax.annotate(f'{diff_pct:.1f}% faster', xy=(i, max(base_dur, pd_dur)),
                           ha='center', va='bottom', fontsize=8, color='green')
            elif diff_pct < 0:
                ax.annotate(f'{-diff_pct:.1f}% slower', xy=(i, max(base_dur, pd_dur)),
                           ha='center', va='bottom', fontsize=8, color='red')

    plt.suptitle('Runtime Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / "duration_scaling.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_report(data: Dict, improvements: Dict) -> str:
    """生成文本分析报告"""
    lines = []
    lines.append("=" * 80)
    lines.append("Request 数量扩展实验分析报告")
    lines.append("=" * 80)
    lines.append("")
    lines.append("假设验证: 随着 request 数量增加，PD scheduler 的稳态时间占比更大，优势更明显")
    lines.append("")
    lines.append(f"Request 数量: {data['request_counts']}")
    lines.append("")

    # 吞吐量表格
    lines.append("-" * 80)
    lines.append("吞吐量 (Request Throughput, req/s)")
    lines.append("-" * 80)
    header = f"{'Requests':<10} {'PD Optimal':<15} {'Baseline Same':<15} {'Baseline Def':<15} {'Imp vs Same':<12} {'Imp vs Def':<12}"
    lines.append(header)
    lines.append("-" * 80)

    for rc in data["request_counts"]:
        pd = data["configs"]["pd_optimal"].get(rc, {}).get("throughput", 0)
        same = data["configs"]["baseline_same"].get(rc, {}).get("throughput", 0)
        default = data["configs"]["baseline_default"].get(rc, {}).get("throughput", 0)
        imp_same = improvements["vs_same"].get(rc, {}).get("throughput", 0)
        imp_def = improvements["vs_default"].get(rc, {}).get("throughput", 0)

        lines.append(f"{rc:<10} {pd:<15.2f} {same:<15.2f} {default:<15.2f} {imp_same:+.2f}%{'':<6} {imp_def:+.2f}%")

    lines.append("")

    # ITL 表格
    lines.append("-" * 80)
    lines.append("平均 ITL (Mean ITL, ms) - 越低越好")
    lines.append("-" * 80)
    header = f"{'Requests':<10} {'PD Optimal':<15} {'Baseline Same':<15} {'Baseline Def':<15} {'Imp vs Same':<12} {'Imp vs Def':<12}"
    lines.append(header)
    lines.append("-" * 80)

    for rc in data["request_counts"]:
        pd = data["configs"]["pd_optimal"].get(rc, {}).get("mean_itl_ms", 0)
        same = data["configs"]["baseline_same"].get(rc, {}).get("mean_itl_ms", 0)
        default = data["configs"]["baseline_default"].get(rc, {}).get("mean_itl_ms", 0)
        imp_same = improvements["vs_same"].get(rc, {}).get("mean_itl_ms", 0)
        imp_def = improvements["vs_default"].get(rc, {}).get("mean_itl_ms", 0)

        lines.append(f"{rc:<10} {pd:<15.2f} {same:<15.2f} {default:<15.2f} {imp_same:+.2f}%{'':<6} {imp_def:+.2f}%")

    lines.append("")

    # 运行时间表格
    lines.append("-" * 80)
    lines.append("运行时间 (Duration, seconds) - 越低越好")
    lines.append("-" * 80)
    header = f"{'Requests':<10} {'PD Optimal':<15} {'Baseline Same':<15} {'Baseline Def':<15} {'Time Saved':<15} {'Time Saved %':<12}"
    lines.append(header)
    lines.append("-" * 80)

    for rc in data["request_counts"]:
        pd = data["configs"]["pd_optimal"].get(rc, {}).get("duration", 0)
        same = data["configs"]["baseline_same"].get(rc, {}).get("duration", 0)
        default = data["configs"]["baseline_default"].get(rc, {}).get("duration", 0)

        # 计算节省的时间 (相对于 baseline_default)
        time_saved = default - pd if default > 0 and pd > 0 else 0
        time_saved_pct = (time_saved / default * 100) if default > 0 else 0

        lines.append(f"{rc:<10} {pd:<15.2f} {same:<15.2f} {default:<15.2f} {time_saved:+.2f}s{'':<8} {time_saved_pct:+.2f}%")

    lines.append("")

    # 趋势分析
    lines.append("=" * 80)
    lines.append("趋势分析")
    lines.append("=" * 80)

    # 检查吞吐量改进是否随 request 数量增加而增加
    throughput_imps = [improvements["vs_same"].get(rc, {}).get("throughput", 0)
                       for rc in data["request_counts"]]

    if len(throughput_imps) >= 2:
        first_half_avg = np.mean(throughput_imps[:len(throughput_imps)//2])
        second_half_avg = np.mean(throughput_imps[len(throughput_imps)//2:])

        if second_half_avg > first_half_avg:
            lines.append("")
            lines.append(f"✓ 假设验证成功: 随着 request 数量增加，PD 的吞吐量优势确实增加")
            lines.append(f"  - 前半段平均改进: {first_half_avg:+.2f}%")
            lines.append(f"  - 后半段平均改进: {second_half_avg:+.2f}%")
            lines.append(f"  - 趋势: 改进提升了 {second_half_avg - first_half_avg:.2f}%")
        else:
            lines.append("")
            lines.append(f"✗ 假设未验证: PD 的优势没有随 request 数量明显增加")
            lines.append(f"  - 前半段平均改进: {first_half_avg:+.2f}%")
            lines.append(f"  - 后半段平均改进: {second_half_avg:+.2f}%")

    # 运行时间分析
    lines.append("")
    lines.append("-" * 80)
    lines.append("运行时间分析: PD 相对于 Baseline (default) 节省的时间")
    lines.append("-" * 80)

    total_pd_duration = 0
    total_baseline_duration = 0
    time_savings = []

    for rc in data["request_counts"]:
        pd_dur = data["configs"]["pd_optimal"].get(rc, {}).get("duration", 0)
        baseline_dur = data["configs"]["baseline_default"].get(rc, {}).get("duration", 0)

        if pd_dur > 0 and baseline_dur > 0:
            total_pd_duration += pd_dur
            total_baseline_duration += baseline_dur
            time_saved_pct = (baseline_dur - pd_dur) / baseline_dur * 100
            time_savings.append(time_saved_pct)
            lines.append(f"  {rc} requests: 节省 {baseline_dur - pd_dur:.1f}s ({time_saved_pct:+.1f}%)")

    if time_savings:
        avg_time_saved_pct = np.mean(time_savings)
        total_time_saved = total_baseline_duration - total_pd_duration
        lines.append("")
        lines.append(f"  平均时间节省: {avg_time_saved_pct:+.1f}%")
        lines.append(f"  总时间: PD={total_pd_duration:.1f}s vs Baseline={total_baseline_duration:.1f}s")
        lines.append(f"  累计节省: {total_time_saved:.1f}s")

        if avg_time_saved_pct > 0:
            lines.append("")
            lines.append(f"  ✓ PD scheduler 平均比 baseline 快 {avg_time_saved_pct:.1f}%")
        elif avg_time_saved_pct < 0:
            lines.append("")
            lines.append(f"  ✗ PD scheduler 平均比 baseline 慢 {-avg_time_saved_pct:.1f}%")

    lines.append("")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("用法: python analyze_request_scaling.py <experiment_dir>")
        sys.exit(1)

    exp_dir = Path(sys.argv[1])
    if not exp_dir.exists():
        print(f"目录不存在: {exp_dir}")
        sys.exit(1)

    print(f"分析实验目录: {exp_dir}")
    print("")

    # 收集数据
    data = collect_results(exp_dir)

    if not data["request_counts"]:
        print("未找到任何实验结果")
        sys.exit(1)

    print(f"找到 {len(data['request_counts'])} 个 request 数量: {data['request_counts']}")
    print("")

    # 计算改进
    improvements = compute_improvements(data)

    # 生成报告
    report = generate_report(data, improvements)
    print(report)

    # 保存报告
    report_path = exp_dir / "analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n报告已保存: {report_path}")

    # 绘图
    plot_throughput_scaling(data, exp_dir)
    plot_latency_scaling(data, exp_dir)
    plot_improvement_scaling(data, improvements, exp_dir)
    plot_duration_scaling(data, exp_dir)

    # 保存汇总 JSON
    summary = {
        "request_counts": data["request_counts"],
        "configs": data["configs"],
        "improvements": improvements
    }

    summary_path = exp_dir / "scaling_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"汇总数据已保存: {summary_path}")


if __name__ == "__main__":
    main()
