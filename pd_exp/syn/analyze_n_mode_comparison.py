#!/usr/bin/env python3
"""
分析 N 模式对比实验结果

对比 reactive (启发式) 和 paper (论文公式) 两种 N 计算方式:
- 吞吐量对比
- 延迟对比 (TTFT, TPOT, E2E)
- N 值稳定性
- 不同 workload 下的表现

用法:
    python analyze_n_mode_comparison.py <results_dir>
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_results(results_dir: str) -> pd.DataFrame:
    """加载所有实验结果"""
    results = []
    results_path = Path(results_dir)

    for scenario_dir in results_path.iterdir():
        if not scenario_dir.is_dir() or scenario_dir.name.startswith('.'):
            continue

        scenario_name = scenario_dir.name

        for bench_file in scenario_dir.glob("bench_*.json"):
            try:
                with open(bench_file) as f:
                    data = json.load(f)

                n_mode = bench_file.stem.replace("bench_", "")

                # 解析 N 模式
                if n_mode == "baseline":
                    mode_type = "baseline"
                    oom_tolerance = None
                elif n_mode == "pd_reactive":
                    mode_type = "reactive"
                    oom_tolerance = None
                elif n_mode.startswith("pd_paper_eps"):
                    mode_type = "paper"
                    eps_str = n_mode.replace("pd_paper_eps", "")
                    # eps001 -> 0.001, eps01 -> 0.01, eps05 -> 0.05, eps10 -> 0.10
                    if len(eps_str) == 3:
                        oom_tolerance = float(f"0.{eps_str}")
                    elif len(eps_str) == 2:
                        oom_tolerance = float(f"0.{eps_str}")
                    else:
                        oom_tolerance = float(eps_str) / 100
                else:
                    mode_type = n_mode
                    oom_tolerance = None

                # 加载调度统计
                stats_file = scenario_dir / f"{n_mode}_stats.json"
                n_values = []
                preemptions = 0
                if stats_file.exists():
                    try:
                        with open(stats_file) as f:
                            stats = json.load(f)
                            for s in stats:
                                if isinstance(s, dict) and "N" in s:
                                    n_values.append(s["N"])
                                if isinstance(s, dict) and "preempted" in s:
                                    preemptions += s.get("preempted", 0)
                    except Exception:
                        pass

                results.append({
                    "scenario": scenario_name,
                    "n_mode": n_mode,
                    "mode_type": mode_type,
                    "oom_tolerance": oom_tolerance,
                    "throughput_req_s": data.get("request_throughput", 0),
                    "throughput_tok_s": data.get("output_throughput", 0),
                    "ttft_mean": data.get("ttft_mean", 0),
                    "ttft_p50": data.get("ttft_p50", 0),
                    "ttft_p90": data.get("ttft_p90", 0),
                    "ttft_p99": data.get("ttft_p99", 0),
                    "tpot_mean": data.get("tpot_mean", 0),
                    "tpot_p50": data.get("tpot_p50", 0),
                    "tpot_p90": data.get("tpot_p90", 0),
                    "tpot_p99": data.get("tpot_p99", 0),
                    "e2e_mean": data.get("e2e_latency_mean", 0),
                    "e2e_p50": data.get("e2e_latency_p50", 0),
                    "e2e_p90": data.get("e2e_latency_p90", 0),
                    "e2e_p99": data.get("e2e_latency_p99", 0),
                    "n_mean": np.mean(n_values) if n_values else None,
                    "n_std": np.std(n_values) if n_values else None,
                    "n_min": np.min(n_values) if n_values else None,
                    "n_max": np.max(n_values) if n_values else None,
                    "preemptions": preemptions,
                })

            except Exception as e:
                print(f"Error loading {bench_file}: {e}")

    return pd.DataFrame(results)


def plot_throughput_comparison(df: pd.DataFrame, output_dir: str):
    """吞吐量对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    scenarios = df["scenario"].unique()
    x = np.arange(len(scenarios))

    # 获取不同配置
    modes = ["baseline", "pd_reactive"]
    paper_eps = sorted(df[df["mode_type"] == "paper"]["oom_tolerance"].dropna().unique())
    for eps in paper_eps:
        modes.append(f"pd_paper_eps{str(eps).replace('0.', '').replace('.', '')}")

    width = 0.8 / len(modes)
    colors = plt.cm.tab10(np.linspace(0, 1, len(modes)))

    # Request throughput
    ax = axes[0]
    for i, mode in enumerate(modes):
        values = []
        for scenario in scenarios:
            row = df[(df["scenario"] == scenario) & (df["n_mode"] == mode)]
            values.append(row["throughput_req_s"].values[0] if len(row) > 0 else 0)

        offset = (i - len(modes)/2 + 0.5) * width
        label = mode.replace("pd_paper_eps", "paper(ε=0.").replace("pd_", "")
        if "paper" in label:
            label = label + ")"
        ax.bar(x + offset, values, width, label=label, color=colors[i])

    ax.set_xlabel("Workload")
    ax.set_ylabel("Throughput (req/s)")
    ax.set_title("Request Throughput")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha="right")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Token throughput
    ax = axes[1]
    for i, mode in enumerate(modes):
        values = []
        for scenario in scenarios:
            row = df[(df["scenario"] == scenario) & (df["n_mode"] == mode)]
            values.append(row["throughput_tok_s"].values[0] if len(row) > 0 else 0)

        offset = (i - len(modes)/2 + 0.5) * width
        label = mode.replace("pd_paper_eps", "paper(ε=0.").replace("pd_", "")
        if "paper" in label:
            label = label + ")"
        ax.bar(x + offset, values, width, label=label, color=colors[i])

    ax.set_xlabel("Workload")
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title("Token Throughput")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha="right")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/throughput_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/throughput_comparison.png")


def plot_latency_comparison(df: pd.DataFrame, output_dir: str):
    """延迟对比图"""
    scenarios = df["scenario"].unique()
    n_scenarios = len(scenarios)

    fig, axes = plt.subplots(2, (n_scenarios + 1) // 2, figsize=(5 * ((n_scenarios + 1) // 2), 10))
    axes = axes.flatten()

    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        sc_data = df[df["scenario"] == scenario].sort_values("n_mode")

        configs = []
        ttft_p99 = []
        tpot_p99 = []
        e2e_p99 = []

        for _, row in sc_data.iterrows():
            label = row["n_mode"].replace("pd_paper_eps", "paper\nε=0.").replace("pd_", "")
            if "paper" in label and not label.endswith(")"):
                label = label
            configs.append(label)
            ttft_p99.append(row["ttft_p99"] * 1000)
            tpot_p99.append(row["tpot_p99"] * 1000)
            e2e_p99.append(row["e2e_p99"] * 1000)

        x = np.arange(len(configs))
        width = 0.25

        ax.bar(x - width, ttft_p99, width, label="TTFT P99", color="steelblue")
        ax.bar(x, tpot_p99, width, label="TPOT P99", color="darkorange")
        ax.bar(x + width, e2e_p99, width, label="E2E P99", color="forestgreen")

        ax.set_xlabel("Configuration")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"{scenario}")
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    # 隐藏多余的子图
    for idx in range(len(scenarios), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/latency_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/latency_comparison.png")


def plot_epsilon_sensitivity(df: pd.DataFrame, output_dir: str):
    """ε 敏感性分析 (仅 paper 模式)"""
    paper_data = df[df["mode_type"] == "paper"].copy()

    if len(paper_data) == 0:
        print("No paper mode data for epsilon sensitivity analysis")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    scenarios = paper_data["scenario"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios)))

    # Throughput vs epsilon
    ax = axes[0]
    for i, scenario in enumerate(scenarios):
        sc_data = paper_data[paper_data["scenario"] == scenario].sort_values("oom_tolerance")
        if len(sc_data) > 0:
            ax.plot(sc_data["oom_tolerance"], sc_data["throughput_req_s"],
                    marker="o", label=scenario, color=colors[i])
    ax.set_xlabel("OOM Tolerance (ε)")
    ax.set_ylabel("Throughput (req/s)")
    ax.set_title("Throughput vs ε")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # E2E Latency vs epsilon
    ax = axes[1]
    for i, scenario in enumerate(scenarios):
        sc_data = paper_data[paper_data["scenario"] == scenario].sort_values("oom_tolerance")
        if len(sc_data) > 0:
            ax.plot(sc_data["oom_tolerance"], sc_data["e2e_p99"] * 1000,
                    marker="o", label=scenario, color=colors[i])
    ax.set_xlabel("OOM Tolerance (ε)")
    ax.set_ylabel("E2E Latency P99 (ms)")
    ax.set_title("Latency vs ε")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # N mean vs epsilon
    ax = axes[2]
    for i, scenario in enumerate(scenarios):
        sc_data = paper_data[paper_data["scenario"] == scenario].sort_values("oom_tolerance")
        if len(sc_data) > 0 and sc_data["n_mean"].notna().any():
            ax.plot(sc_data["oom_tolerance"], sc_data["n_mean"],
                    marker="o", label=scenario, color=colors[i])
    ax.set_xlabel("OOM Tolerance (ε)")
    ax.set_ylabel("Batch Size N (mean)")
    ax.set_title("Batch Size vs ε")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/epsilon_sensitivity.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/epsilon_sensitivity.png")


def plot_reactive_vs_paper(df: pd.DataFrame, output_dir: str):
    """reactive vs paper 直接对比"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    scenarios = df["scenario"].unique()

    # 获取 reactive 和 paper (ε=0.01) 的数据
    reactive = df[df["n_mode"] == "pd_reactive"]
    paper_01 = df[df["n_mode"] == "pd_paper_eps01"]

    if len(reactive) == 0 or len(paper_01) == 0:
        print("Missing data for reactive vs paper comparison")
        return

    # 1. Throughput comparison
    ax = axes[0, 0]
    x = np.arange(len(scenarios))
    width = 0.35

    reactive_tp = [reactive[reactive["scenario"] == s]["throughput_req_s"].values[0]
                   if len(reactive[reactive["scenario"] == s]) > 0 else 0
                   for s in scenarios]
    paper_tp = [paper_01[paper_01["scenario"] == s]["throughput_req_s"].values[0]
                if len(paper_01[paper_01["scenario"] == s]) > 0 else 0
                for s in scenarios]

    ax.bar(x - width/2, reactive_tp, width, label="reactive", color="steelblue")
    ax.bar(x + width/2, paper_tp, width, label="paper (ε=0.01)", color="darkorange")
    ax.set_xlabel("Workload")
    ax.set_ylabel("Throughput (req/s)")
    ax.set_title("Throughput: reactive vs paper")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 2. Latency comparison
    ax = axes[0, 1]
    reactive_lat = [reactive[reactive["scenario"] == s]["e2e_p99"].values[0] * 1000
                    if len(reactive[reactive["scenario"] == s]) > 0 else 0
                    for s in scenarios]
    paper_lat = [paper_01[paper_01["scenario"] == s]["e2e_p99"].values[0] * 1000
                 if len(paper_01[paper_01["scenario"] == s]) > 0 else 0
                 for s in scenarios]

    ax.bar(x - width/2, reactive_lat, width, label="reactive", color="steelblue")
    ax.bar(x + width/2, paper_lat, width, label="paper (ε=0.01)", color="darkorange")
    ax.set_xlabel("Workload")
    ax.set_ylabel("E2E Latency P99 (ms)")
    ax.set_title("Latency P99: reactive vs paper")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 3. Throughput difference (%)
    ax = axes[1, 0]
    diff_pct = [(p - r) / r * 100 if r > 0 else 0
                for r, p in zip(reactive_tp, paper_tp)]
    colors = ["forestgreen" if d >= 0 else "crimson" for d in diff_pct]
    ax.bar(x, diff_pct, color=colors)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Workload")
    ax.set_ylabel("Throughput Change (%)")
    ax.set_title("paper vs reactive: Throughput Δ%")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    # 4. Latency difference (%)
    ax = axes[1, 1]
    lat_diff_pct = [(p - r) / r * 100 if r > 0 else 0
                    for r, p in zip(reactive_lat, paper_lat)]
    # 延迟降低是好的，所以颜色反转
    colors = ["forestgreen" if d <= 0 else "crimson" for d in lat_diff_pct]
    ax.bar(x, lat_diff_pct, color=colors)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Workload")
    ax.set_ylabel("Latency Change (%)")
    ax.set_title("paper vs reactive: Latency P99 Δ%")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/reactive_vs_paper.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/reactive_vs_paper.png")


def generate_summary_table(df: pd.DataFrame, output_dir: str):
    """生成汇总表格"""
    summary_rows = []
    scenarios = df["scenario"].unique()

    for scenario in scenarios:
        sc_data = df[df["scenario"] == scenario]

        # 以 reactive 为基准
        reactive = sc_data[sc_data["n_mode"] == "pd_reactive"]
        if len(reactive) > 0:
            baseline_tp = reactive["throughput_req_s"].values[0]
            baseline_lat = reactive["e2e_p99"].values[0]
        else:
            baseline_tp = 1
            baseline_lat = 1

        for _, row in sc_data.iterrows():
            tp_change = ((row["throughput_req_s"] - baseline_tp) / baseline_tp * 100
                         if baseline_tp > 0 and row["n_mode"] != "pd_reactive" else 0)
            lat_change = ((row["e2e_p99"] - baseline_lat) / baseline_lat * 100
                          if baseline_lat > 0 and row["n_mode"] != "pd_reactive" else 0)

            summary_rows.append({
                "scenario": scenario,
                "n_mode": row["n_mode"],
                "mode_type": row["mode_type"],
                "oom_tolerance": row["oom_tolerance"],
                "throughput_req_s": row["throughput_req_s"],
                "throughput_change_%": tp_change,
                "e2e_p99_ms": row["e2e_p99"] * 1000,
                "latency_change_%": lat_change,
                "n_mean": row["n_mean"],
                "preemptions": row["preemptions"],
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{output_dir}/summary.csv", index=False)
    print(f"Saved: {output_dir}/summary.csv")

    # 打印表格
    print("\n" + "=" * 130)
    print("N 模式对比实验结果 (以 pd_reactive 为基准)")
    print("=" * 130)
    print(f"{'Scenario':<12} {'N Mode':<20} {'Throughput':>12} {'Δ%':>8} "
          f"{'E2E P99':>10} {'Δ%':>8} {'N mean':>8} {'Preempt':>8}")
    print("-" * 130)

    for _, row in summary_df.iterrows():
        tp_delta = f"{row['throughput_change_%']:+.1f}%" if row["n_mode"] != "pd_reactive" else "-"
        lat_delta = f"{row['latency_change_%']:+.1f}%" if row["n_mode"] != "pd_reactive" else "-"
        n_mean = f"{row['n_mean']:.0f}" if pd.notna(row["n_mean"]) else "N/A"

        print(f"{row['scenario']:<12} {row['n_mode']:<20} {row['throughput_req_s']:>12.2f} "
              f"{tp_delta:>8} {row['e2e_p99_ms']:>10.1f} {lat_delta:>8} "
              f"{n_mean:>8} {row['preemptions']:>8}")

    print("=" * 130)

    # 打印关键发现
    print("\n关键发现:")
    paper_01 = summary_df[summary_df["n_mode"] == "pd_paper_eps01"]
    if len(paper_01) > 0:
        avg_tp_change = paper_01["throughput_change_%"].mean()
        avg_lat_change = paper_01["latency_change_%"].mean()
        print(f"  paper (ε=0.01) vs reactive:")
        print(f"    平均吞吐量变化: {avg_tp_change:+.1f}%")
        print(f"    平均延迟变化: {avg_lat_change:+.1f}%")


def main():
    parser = argparse.ArgumentParser(description="分析 N 模式对比实验结果")
    parser.add_argument("results_dir", help="实验结果目录")
    parser.add_argument("--output-dir", "-o", help="输出目录", default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir

    print(f"Loading results from: {args.results_dir}")
    df = load_results(args.results_dir)

    if len(df) == 0:
        print("No results found!")
        return

    print(f"Loaded {len(df)} experiment results")
    print(f"Scenarios: {df['scenario'].unique().tolist()}")
    print(f"N modes: {df['n_mode'].unique().tolist()}")

    # 生成图表
    plot_throughput_comparison(df, output_dir)
    plot_latency_comparison(df, output_dir)
    plot_epsilon_sensitivity(df, output_dir)
    plot_reactive_vs_paper(df, output_dir)

    # 生成汇总
    generate_summary_table(df, output_dir)


if __name__ == "__main__":
    main()
