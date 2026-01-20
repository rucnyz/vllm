#!/usr/bin/env python3
"""
绘制调度器统计的甘特图风格可视化

横轴: 时间 (秒)
纵轴: 运行中的请求数量
颜色: phase 0 (prefill) = 红色, phase 1/2 (decode) = 蓝色
竖线: phase 切换时刻
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_stats(stats_file: Path) -> dict:
    """加载统计文件"""
    with open(stats_file) as f:
        return json.load(f)


def find_phase_transitions(stats: list) -> list:
    """找到 phase 切换的时刻"""
    transitions = []
    prev_phase = None
    for s in stats:
        phase = s.get("phase", -1)
        if prev_phase is not None and phase != prev_phase:
            transitions.append({
                "timestamp": s["timestamp"],
                "from_phase": prev_phase,
                "to_phase": phase
            })
        prev_phase = phase
    return transitions


def find_phase1_k_regions(stats: list) -> list:
    """找到所有 phase 1 区间及其 k 值

    返回列表，每个元素包含:
    - start_time: 区间开始时间
    - end_time: 区间结束时间
    - k: 该区间的 k 值
    """
    regions = []
    in_phase1 = False
    current_k = None
    region_start = None

    for i, s in enumerate(stats):
        phase = s.get("phase", -1)
        k = s.get("k_star") or s.get("k")  # 支持 k_star 和 k 两种字段名

        if phase == 1:
            if not in_phase1:
                # 刚进入 phase 1
                in_phase1 = True
                region_start = s["timestamp"]
                current_k = k
            elif k != current_k:
                # k 值发生变化，结束当前区间，开始新区间
                if region_start is not None and current_k is not None:
                    regions.append({
                        "start_time": region_start,
                        "end_time": s["timestamp"],
                        "k": current_k
                    })
                region_start = s["timestamp"]
                current_k = k
        else:
            if in_phase1:
                # 离开 phase 1，结束当前区间
                if region_start is not None and current_k is not None:
                    regions.append({
                        "start_time": region_start,
                        "end_time": s["timestamp"],
                        "k": current_k
                    })
                in_phase1 = False
                current_k = None
                region_start = None

    # 处理最后一个区间（如果还在 phase 1）
    if in_phase1 and region_start is not None and current_k is not None:
        regions.append({
            "start_time": region_start,
            "end_time": stats[-1]["timestamp"],
            "k": current_k
        })

    return regions


def plot_gantt(stats_file: Path, output_file: Path = None, title: str = None,
               show_k: bool = False):
    """绘制甘特图

    Args:
        stats_file: 统计文件路径
        output_file: 输出图片路径
        title: 图表标题
        show_k: 是否在 phase 1 区间显示 k 值
    """
    data = load_stats(stats_file)
    stats = data.get("stats", [])

    if not stats:
        print(f"No stats found in {stats_file}")
        return

    # 提取数据
    timestamps = [s["timestamp"] for s in stats]
    num_running = [s.get("num_running_reqs", 0) for s in stats]
    phases = [s.get("phase", -1) for s in stats]

    # 找到 phase 切换点
    transitions = find_phase_transitions(stats)

    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 6))

    # 按 phase 分组绘制散点
    # Phase 0, 2 = Prefill (红色)
    # Phase 1 = Decode (蓝色)
    prefill_mask = np.array([p == 0 or p == 2 for p in phases])
    decode_mask = np.array([p == 1 for p in phases])

    timestamps = np.array(timestamps)
    num_running = np.array(num_running)

    # 绘制散点
    if prefill_mask.any():
        ax.scatter(timestamps[prefill_mask], num_running[prefill_mask],
                   c='red', s=10, alpha=0.7, label='Phase 0/2 (Prefill)', zorder=3)
    if decode_mask.any():
        ax.scatter(timestamps[decode_mask], num_running[decode_mask],
                   c='blue', s=10, alpha=0.7, label='Phase 1 (Decode)', zorder=3)

    # 绘制连线 (淡色背景)
    ax.plot(timestamps, num_running, 'gray', alpha=0.3, linewidth=0.5, zorder=1)

    # 绘制 phase 切换竖线
    for trans in transitions:
        t = trans["timestamp"]
        from_p = trans["from_phase"]
        to_p = trans["to_phase"]
        color = 'green' if to_p > from_p else 'orange'  # 进入 decode 用绿色，返回 prefill 用橙色
        ax.axvline(x=t, color=color, linestyle='--', alpha=0.6, linewidth=1.5, zorder=2)

    # 添加图例说明 phase 切换线
    if transitions:
        # 添加虚拟线条用于图例
        ax.axvline(x=np.nan, color='green', linestyle='--', alpha=0.6,
                   linewidth=1.5, label='Phase transition (→ Decode)')
        ax.axvline(x=np.nan, color='orange', linestyle='--', alpha=0.6,
                   linewidth=1.5, label='Phase transition (→ Prefill)')

    # 在 phase 1 区间显示 k 值
    if show_k:
        k_regions = find_phase1_k_regions(stats)
        if k_regions:
            y_max = num_running.max() if len(num_running) > 0 else 100
            for region in k_regions:
                # 在区间中间位置标注 k 值
                mid_time = (region["start_time"] + region["end_time"]) / 2
                k_val = region["k"]
                # 在图表上方位置显示 k 值
                ax.annotate(f'k={k_val}',
                            xy=(mid_time, y_max * 0.85),
                            fontsize=8, color='blue', fontweight='bold',
                            ha='center', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.2',
                                      facecolor='lightyellow', alpha=0.8,
                                      edgecolor='blue', linewidth=0.5))

    # 设置标签
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Number of Running Requests', fontsize=12)

    # 标题
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Schedule Timeline: {stats_file.name}', fontsize=14, fontweight='bold')

    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # 添加统计信息
    total_time = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
    num_transitions = len(transitions)
    max_running = num_running.max() if len(num_running) > 0 else 0

    info_text = f"Total time: {total_time:.2f}s | Phase transitions: {num_transitions} | Max running: {max_running}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # 保存或显示
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
    else:
        # 默认保存到同目录
        default_output = stats_file.parent / f"{stats_file.stem}_gantt.png"
        plt.savefig(default_output, dpi=150, bbox_inches='tight')
        print(f"Saved: {default_output}")

    plt.close()


def find_cycles(stats: list) -> list:
    """找到所有完整的 cycle (decode开始 -> decode -> prefill -> prefill结束)

    Cycle 定义:
    - 开始: Decode phase (phase 1) 开始 (prefill -> decode 切换)
    - 结束: 下一次 Prefill 结束 (prefill -> decode 切换)

    即: prefill->decode(开始) -> decode -> decode->prefill -> prefill -> prefill->decode(结束)
    """
    cycles = []
    transitions = []

    # 收集所有 phase 变化点
    for i, s in enumerate(stats):
        if i > 0 and s['phase'] != stats[i-1]['phase']:
            transitions.append({
                'idx': i,
                'timestamp': s['timestamp'],
                'from_phase': stats[i-1]['phase'],
                'to_phase': s['phase']
            })

    # 找到完整 cycle: prefill(0/2)->decode(1) -> decode(1)->prefill(0/2) -> prefill(0/2)->decode(1)
    # t1: prefill -> decode (cycle 开始，进入 decode)
    # t2: decode -> prefill (decode 结束，进入 prefill)
    # t3: prefill -> decode (cycle 结束，prefill 结束)
    i = 0
    while i < len(transitions) - 1:
        t1 = transitions[i]
        # 找 prefill -> decode 的转换 (decode 开始)
        if t1['from_phase'] in [0, 2] and t1['to_phase'] == 1:
            # 找下一个 decode -> prefill
            if i + 1 < len(transitions):
                t2 = transitions[i + 1]
                if t2['from_phase'] == 1 and t2['to_phase'] in [0, 2]:
                    # 找下一个 prefill -> decode (cycle 结束)
                    if i + 2 < len(transitions):
                        t3 = transitions[i + 2]
                        if t3['from_phase'] in [0, 2] and t3['to_phase'] == 1:
                            cycles.append({
                                'start_idx': t1['idx'] - 5,   # 往前包含一些 prefill
                                'end_idx': t3['idx'] + 5,     # 往后包含一些 decode
                                'start_time': stats[max(0, t1['idx'] - 5)]['timestamp'],
                                'end_time': stats[min(len(stats)-1, t3['idx'] + 5)]['timestamp'],
                                'transitions': [t1, t2, t3]
                            })
                            i += 2
                            continue
        i += 1

    return cycles


def plot_cycle(stats_file: Path, cycle_num: int = 0, output_file: Path = None,
               show_k: bool = False):
    """绘制单个 cycle 的放大图

    Args:
        stats_file: 统计文件路径
        cycle_num: cycle 编号
        output_file: 输出图片路径
        show_k: 是否在 phase 1 区间显示 k 值
    """
    data = load_stats(stats_file)
    stats = data.get("stats", [])

    if not stats:
        print(f"No stats found in {stats_file}")
        return

    cycles = find_cycles(stats)

    if not cycles:
        print("No complete cycles found in data")
        return

    print(f"Found {len(cycles)} complete cycles")
    for i, c in enumerate(cycles):
        print(f"  Cycle {i}: idx {c['start_idx']}-{c['end_idx']}, "
              f"time {c['start_time']:.3f}s - {c['end_time']:.3f}s")

    if cycle_num >= len(cycles):
        print(f"Cycle {cycle_num} not found, using cycle 0")
        cycle_num = 0

    cycle = cycles[cycle_num]
    start_idx = max(0, cycle['start_idx'])
    end_idx = min(len(stats), cycle['end_idx'])

    # 提取 cycle 数据
    cycle_stats = stats[start_idx:end_idx]

    timestamps = np.array([s['timestamp'] for s in cycle_stats])
    num_running = np.array([s.get('num_running_reqs', 0) for s in cycle_stats])
    phases = [s.get('phase', -1) for s in cycle_stats]

    # 按 phase 分组
    prefill_mask = np.array([p == 0 or p == 2 for p in phases])
    decode_mask = np.array([p == 1 for p in phases])

    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 6))

    # 绘制散点
    if prefill_mask.any():
        ax.scatter(timestamps[prefill_mask], num_running[prefill_mask],
                   c='red', s=30, alpha=0.8, label='Phase 0/2 (Prefill)', zorder=3)
    if decode_mask.any():
        ax.scatter(timestamps[decode_mask], num_running[decode_mask],
                   c='blue', s=30, alpha=0.8, label='Phase 1 (Decode)', zorder=3)

    # 绘制连线
    ax.plot(timestamps, num_running, 'gray', alpha=0.4, linewidth=1, zorder=1)

    # 绘制 phase 切换竖线
    for trans in cycle['transitions']:
        t = trans['timestamp']
        from_p = trans['from_phase']
        to_p = trans['to_phase']

        if to_p in [0, 2]:  # 进入 prefill
            color = 'red'
        else:  # 进入 decode
            color = 'blue'

        ax.axvline(x=t, color=color, linestyle='--', alpha=0.7, linewidth=2, zorder=2)
        # 在竖线上方标注 phase 切换和时刻
        y_pos = num_running.max() * 0.95
        ax.annotate(f'{from_p}→{to_p}\nt={t:.3f}s',
                    xy=(t, y_pos),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=9, color=color,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # 在 phase 1 区间显示 k 值
    if show_k:
        k_regions = find_phase1_k_regions(cycle_stats)
        if k_regions:
            y_max = num_running.max() if len(num_running) > 0 else 100
            for region in k_regions:
                # 在区间中间位置标注 k 值
                mid_time = (region["start_time"] + region["end_time"]) / 2
                k_val = region["k"]
                # 在图表上方位置显示 k 值
                ax.annotate(f'k={k_val}',
                            xy=(mid_time, y_max * 0.85),
                            fontsize=9, color='blue', fontweight='bold',
                            ha='center', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.2',
                                      facecolor='lightyellow', alpha=0.8,
                                      edgecolor='blue', linewidth=0.5))

    # 设置标签
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Number of Running Requests', fontsize=12)
    ax.set_title(f'Cycle {cycle_num}: Decode → Prefill → (next Decode)\n'
                 f'Time: {cycle["start_time"]:.3f}s - {cycle["end_time"]:.3f}s',
                 fontsize=12, fontweight='bold')

    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # 添加 phase 区域背景色
    prev_phase = phases[0]
    region_start = timestamps[0]
    for i, (t, p) in enumerate(zip(timestamps, phases)):
        if p != prev_phase or i == len(timestamps) - 1:
            # 结束上一个区域
            region_end = t if i < len(timestamps) - 1 else timestamps[-1]
            color = 'red' if prev_phase in [0, 2] else 'blue'
            ax.axvspan(region_start, region_end, alpha=0.1, color=color, zorder=0)
            region_start = t
            prev_phase = p

    plt.tight_layout()

    # 保存
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
    else:
        default_output = stats_file.parent / f"{stats_file.stem}_cycle{cycle_num}.png"
        plt.savefig(default_output, dpi=150, bbox_inches='tight')
        print(f"Saved: {default_output}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot schedule Gantt chart')
    parser.add_argument('stats_file', type=str, help='Path to stats JSON file')
    parser.add_argument('--output', '-o', type=str, help='Output image file')
    parser.add_argument('--title', '-t', type=str, help='Chart title')
    parser.add_argument('--cycle', '-c', type=int, default=None,
                        help='Plot specific cycle number (zoom in on one P-D cycle)')
    parser.add_argument('--list-cycles', '-l', action='store_true',
                        help='List all available cycles')
    parser.add_argument('--show-k', '-k', action='store_true', default=False,
                        help='Show k values during phase 1 (decode) regions')

    args = parser.parse_args()

    stats_file = Path(args.stats_file)
    if not stats_file.exists():
        print(f"Error: {stats_file} not found")
        return

    # 列出所有 cycle
    if args.list_cycles:
        data = load_stats(stats_file)
        stats = data.get("stats", [])
        cycles = find_cycles(stats)
        print(f"Found {len(cycles)} complete cycles:")
        for i, c in enumerate(cycles):
            duration = c['end_time'] - c['start_time']
            print(f"  Cycle {i}: time {c['start_time']:.3f}s - {c['end_time']:.3f}s "
                  f"(duration: {duration:.3f}s)")
        return

    output_file = Path(args.output) if args.output else None

    if args.cycle is not None:
        plot_cycle(stats_file, args.cycle, output_file, show_k=args.show_k)
    else:
        plot_gantt(stats_file, output_file, args.title, show_k=args.show_k)


if __name__ == "__main__":
    main()
