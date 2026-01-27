#!/usr/bin/env python3
"""
生成单独的 benchmark 指标图表
- Total Throughput
- TPOT (Time Per Output Token)
- ITL (Inter-Token Latency)

用法:
  python plot_individual_metrics.py --results-dir pd_exp/outputs/kstar_bs1024_c2048_n4000/in128_out1024
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 全局字体和样式设置
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# Make text/lines more readable in paper PDFs.
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['lines.markersize'] = 10
# Embed TrueType fonts for better PDF compatibility.
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def load_json(filepath: Path) -> dict | None:
    """加载 JSON 文件"""
    try:
        with open(filepath, encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"  Error loading {filepath.name}: {e}")
        return None


def extract_bench_metrics(data: dict) -> dict:
    """从 vllm bench serve 输出中提取指标"""
    return {
        'total_token_throughput': data.get('total_token_throughput', 0),
        'output_throughput': data.get('output_throughput', 0),
        'request_throughput': data.get('request_throughput', 0),
        'mean_ttft_ms': data.get('mean_ttft_ms', 0),
        'median_ttft_ms': data.get('median_ttft_ms', 0),
        'p99_ttft_ms': data.get('p99_ttft_ms', 0),
        'mean_tpot_ms': data.get('mean_tpot_ms', 0),
        'median_tpot_ms': data.get('median_tpot_ms', 0),
        'p99_tpot_ms': data.get('p99_tpot_ms', 0),
        'mean_itl_ms': data.get('mean_itl_ms', 0),
        'median_itl_ms': data.get('median_itl_ms', 0),
        'p99_itl_ms': data.get('p99_itl_ms', 0),
    }


def load_bench_data(results_dir: Path) -> tuple[dict, dict | None, dict | None]:
    """加载 benchmark 数据"""
    k_star_results = {}
    baseline_result = None
    ifr_result = None

    # 加载 bench_fixed*.json (固定 K* 模式)
    for filepath in results_dir.glob("bench_fixed*.json"):
        match = re.search(r'bench_fixed(\d+)\.json', filepath.name)
        if match:
            k_star = int(match.group(1))
            data = load_json(filepath)
            if data is not None:
                k_star_results[k_star] = extract_bench_metrics(data)

    # 加载 baseline
    baseline_path = results_dir / "bench_baseline.json"
    if baseline_path.exists():
        data = load_json(baseline_path)
        if data is not None:
            baseline_result = extract_bench_metrics(data)
            print(f"Loaded v1 (baseline): throughput={baseline_result['total_token_throughput']:.0f} tok/s")

    # 加载 IFR
    ifr_path = results_dir / "bench_ifr.json"
    if ifr_path.exists():
        data = load_json(ifr_path)
        if data is not None:
            ifr_result = extract_bench_metrics(data)
            print(f"Loaded Adaptive (IFR): throughput={ifr_result['total_token_throughput']:.0f} tok/s")

    return k_star_results, baseline_result, ifr_result


def plot_throughput(k_star_results: dict, baseline: dict | None, ifr: dict | None,
                    output_path: Path, title_suffix: str = ""):
    """绘制 Total Throughput 图"""
    fig, ax = plt.subplots(figsize=(10, 7))

    # 排序 K* 值
    k_stars = sorted(k_star_results.keys())
    throughputs = [k_star_results[k]['total_token_throughput'] for k in k_stars]

    # 绘制 K* 曲线
    ax.plot(k_stars, throughputs, 'b-o', label='Fixed K*', alpha=0.8)

    # 绘制 baseline (v1) 水平线
    if baseline:
        ax.axhline(y=baseline['total_token_throughput'], color='red', linestyle='-',
                   linewidth=4.5, alpha=0.9,
                   label=f"v1 ({baseline['total_token_throughput']:.0f} tok/s)")

    # 绘制 IFR (Adaptive) 水平线
    if ifr:
        ax.axhline(y=ifr['total_token_throughput'], color='cyan', linestyle='--',
                   linewidth=4.5, alpha=0.9,
                   label=f"Adaptive ({ifr['total_token_throughput']:.0f} tok/s)")

    ax.set_xlabel('K* Value')
    ax.set_ylabel('Total Throughput (tokens/s)')
    ax.set_title(f'Total Throughput vs K*{title_suffix}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # 标注最佳点 (动态调整位置避免超出边界)
    if throughputs:
        best_idx = np.argmax(throughputs)
        best_k = k_stars[best_idx]
        best_tps = throughputs[best_idx]
        # 如果最佳点在右侧，标注放左边；否则放右边
        if best_idx > len(k_stars) * 0.6:
            xytext = (-100, -40)
        else:
            xytext = (40, -40)
        ax.annotate(f'Best: K*={best_k}\n{best_tps:.0f} tok/s',
                    xy=(best_k, best_tps), xytext=xytext, textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='green', lw=2.5),
                    fontsize=16, color='green', fontweight='bold')

    plt.tight_layout()
    # 保存为 PDF
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {pdf_path}")


def plot_tpot(k_star_results: dict, baseline: dict | None, ifr: dict | None,
              output_path: Path, title_suffix: str = ""):
    """绘制 TPOT (Time Per Output Token) 图"""
    fig, ax = plt.subplots(figsize=(10, 7))

    k_stars = sorted(k_star_results.keys())
    mean_tpot = [k_star_results[k]['mean_tpot_ms'] for k in k_stars]
    p99_tpot = [k_star_results[k]['p99_tpot_ms'] for k in k_stars]

    # 绘制 K* 曲线
    ax.plot(k_stars, mean_tpot, 'b-o', label='Fixed K* (mean)', alpha=0.8)
    ax.plot(k_stars, p99_tpot, 'b--s', linewidth=3, markersize=8, label='Fixed K* (p99)', alpha=0.6)

    # 绘制 baseline (v1) 水平线
    if baseline:
        ax.axhline(y=baseline['mean_tpot_ms'], color='red', linestyle='-',
                   linewidth=4.5, alpha=0.9,
                   label=f"v1 mean ({baseline['mean_tpot_ms']:.2f} ms)")
        ax.axhline(y=baseline['p99_tpot_ms'], color='red', linestyle=':',
                   linewidth=3.5, alpha=0.6,
                   label=f"v1 p99 ({baseline['p99_tpot_ms']:.2f} ms)")

    # 绘制 IFR (Adaptive) 水平线
    if ifr:
        ax.axhline(y=ifr['mean_tpot_ms'], color='cyan', linestyle='--',
                   linewidth=4.5, alpha=0.9,
                   label=f"Adaptive mean ({ifr['mean_tpot_ms']:.2f} ms)")
        ax.axhline(y=ifr['p99_tpot_ms'], color='cyan', linestyle=':',
                   linewidth=3.5, alpha=0.6,
                   label=f"Adaptive p99 ({ifr['p99_tpot_ms']:.2f} ms)")

    ax.set_xlabel('K* Value')
    ax.set_ylabel('TPOT (ms)')
    ax.set_title(f'Time Per Output Token vs K*{title_suffix}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    # 保存为 PDF
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {pdf_path}")


def plot_itl(k_star_results: dict, baseline: dict | None, ifr: dict | None,
             output_path: Path, title_suffix: str = ""):
    """绘制 ITL (Inter-Token Latency) 图"""
    fig, ax = plt.subplots(figsize=(10, 7))

    k_stars = sorted(k_star_results.keys())
    mean_itl = [k_star_results[k]['mean_itl_ms'] for k in k_stars]
    p99_itl = [k_star_results[k]['p99_itl_ms'] for k in k_stars]

    # 绘制 K* 曲线
    ax.plot(k_stars, mean_itl, 'b-o', label='Fixed K* (mean)', alpha=0.8)
    ax.plot(k_stars, p99_itl, 'b--s', linewidth=3, markersize=8, label='Fixed K* (p99)', alpha=0.6)

    # 绘制 baseline (v1) 水平线
    if baseline:
        ax.axhline(y=baseline['mean_itl_ms'], color='red', linestyle='-',
                   linewidth=4.5, alpha=0.9,
                   label=f"v1 mean ({baseline['mean_itl_ms']:.2f} ms)")
        ax.axhline(y=baseline['p99_itl_ms'], color='red', linestyle=':',
                   linewidth=3.5, alpha=0.6,
                   label=f"v1 p99 ({baseline['p99_itl_ms']:.2f} ms)")

    # 绘制 IFR (Adaptive) 水平线
    if ifr:
        ax.axhline(y=ifr['mean_itl_ms'], color='cyan', linestyle='--',
                   linewidth=4.5, alpha=0.9,
                   label=f"Adaptive mean ({ifr['mean_itl_ms']:.2f} ms)")
        ax.axhline(y=ifr['p99_itl_ms'], color='cyan', linestyle=':',
                   linewidth=3.5, alpha=0.6,
                   label=f"Adaptive p99 ({ifr['p99_itl_ms']:.2f} ms)")

    ax.set_xlabel('K* Value')
    ax.set_ylabel('ITL (ms)')
    ax.set_title(f'Inter-Token Latency vs K*{title_suffix}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    # 保存为 PDF
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate individual metric plots")
    parser.add_argument("--results-dir", type=str, required=True, help="Results directory")
    parser.add_argument("--title-suffix", type=str, default="", help="Title suffix (e.g., scenario name)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Directory {results_dir} does not exist")
        return

    print(f"Loading data from: {results_dir}")
    k_star_results, baseline, ifr = load_bench_data(results_dir)

    if not k_star_results:
        print("No K* results found!")
        return

    title_suffix = args.title_suffix if args.title_suffix else ""
    if title_suffix and not title_suffix.startswith(" "):
        title_suffix = f" ({title_suffix})"

    # 生成三张图
    plot_throughput(k_star_results, baseline, ifr,
                    results_dir / "bench_throughput.png", title_suffix)
    plot_tpot(k_star_results, baseline, ifr,
              results_dir / "bench_tpot.png", title_suffix)
    plot_itl(k_star_results, baseline, ifr,
             results_dir / "bench_itl.png", title_suffix)

    print("\nDone!")


if __name__ == "__main__":
    main()
