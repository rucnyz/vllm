#!/usr/bin/env python3
"""
生成 PD Scheduler 实验总结报告

读取各实验目录的数据，生成跨模型、跨数据集的汇总表格。

用法:
    python generate_summary.py [MODEL1] [MODEL2] ...
    python generate_summary.py  # 自动检测所有模型
"""

import argparse
import sys
from collections import defaultdict
from datetime import datetime
from glob import glob
from pathlib import Path

# 添加模块路径
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / "real"))
sys.path.insert(0, str(SCRIPT_DIR / "multiturn"))

# 导入分析函数
from analyze_grid_search import collect_grid_results as collect_grid_results_real
from analyze_grid_search import find_optimal_configs as find_optimal_configs_real
from analyze_results import collect_grid_results as collect_grid_results_multiturn
from analyze_results import find_optimal_configs as find_optimal_configs_multiturn

OUTPUT_DIR = SCRIPT_DIR / "outputs"
EVAL_DIR = OUTPUT_DIR / "evaluation"

# 支持的模型
MODEL_MAP = {
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Qwen3-30B-A3B": "Qwen/Qwen3-30B-A3B",
    "gemma-3-1b-it": "google/gemma-3-1b-it",
    "gpt-oss-120b": "openai/gpt-oss-120b",
}

# 实验目录模式
EXP_PATTERNS = {
    "sharegpt": ("grid_search_sharegpt_prompts_{model}_*", "real"),
    "numina_math": ("grid_search_numina_math_prompts_{model}_*", "real"),
    "longbench": ("grid_search_longbench_prefill_{model}_*", "real"),
    "wildchat": ("multiturn_wildchat_multiturn_{model}_*", "multiturn"),
}

# 默认调度器列表，可通过 --schedulers 参数覆盖
DEFAULT_SCHEDULERS = ["baseline", "pd_ratio", "pd_ifr"]
# 实际使用的调度器列表 (在 main 中设置)
SCHEDULERS = DEFAULT_SCHEDULERS


def find_result_dir(exp: str, model: str) -> Path | None:
    """查找实验结果目录"""
    pattern, _ = EXP_PATTERNS[exp]
    pattern = pattern.format(model=model)
    matches = sorted(glob(str(OUTPUT_DIR / pattern)))
    return Path(matches[-1]) if matches else None


def collect_experiment_data(exp: str, model: str) -> dict | None:
    """收集单个实验的数据，返回每个 scheduler 的最优配置（包括 tb, bs, throughput）和完整网格数据"""
    result_dir = find_result_dir(exp, model)
    if not result_dir:
        return None

    _, exp_type = EXP_PATTERNS[exp]

    try:
        if exp_type == "real":
            data = collect_grid_results_real(result_dir)
            optimal = find_optimal_configs_real(data)
        else:
            data = collect_grid_results_multiturn(result_dir)
            optimal = find_optimal_configs_multiturn(data)

        # 提取每个 scheduler 的最优配置（包括 tb, bs, throughput）和完整网格数据
        results = {}
        for sched in SCHEDULERS:
            if sched in optimal:
                # 收集该 scheduler 在所有配置下的 throughput
                all_throughputs = []
                grid_data = {}  # {(tb, bs): throughput}
                for (tb, bs), sched_results in data["results"].items():
                    if sched in sched_results:
                        tp = sched_results[sched].get("throughput", 0)
                        if tp > 0:
                            all_throughputs.append(tp)
                            grid_data[(tb, bs)] = tp

                results[sched] = {
                    "throughput": optimal[sched]["metrics"].get("throughput", 0),
                    "tb": optimal[sched].get("tb", 0),
                    "bs": optimal[sched].get("bs", 0),
                    "all_throughputs": all_throughputs,
                    "grid_data": grid_data,
                    "tb_values": data["tb_values"],
                    "bs_values": data["bs_values"],
                }

        return results if results else None

    except Exception as e:
        print(f"  警告: {exp}/{model} 分析失败: {e}")
        return None


def compute_sensitivity(sched_data: dict) -> dict:
    """计算参数敏感度指标

    Returns:
        {
            "cv": 变异系数 (std/mean),
            "range_ratio": 最大/最小吞吐量比值,
            "tb_sensitivity": TB 敏感度 (固定 BS 时的平均变异系数),
            "bs_sensitivity": BS 敏感度 (固定 TB 时的平均变异系数),
        }
    """
    import statistics

    all_tp = sched_data.get("all_throughputs", [])
    grid_data = sched_data.get("grid_data", {})
    tb_values = sched_data.get("tb_values", [])
    bs_values = sched_data.get("bs_values", [])

    result = {
        "cv": 0.0,
        "range_ratio": 0.0,
        "tb_sensitivity": 0.0,
        "bs_sensitivity": 0.0,
    }

    if len(all_tp) < 2:
        return result

    # 整体变异系数
    mean_tp = statistics.mean(all_tp)
    std_tp = statistics.stdev(all_tp)
    result["cv"] = std_tp / mean_tp if mean_tp > 0 else 0.0

    # 最大/最小比值
    result["range_ratio"] = max(all_tp) / min(all_tp) if min(all_tp) > 0 else 0.0

    # TB 敏感度: 固定 BS，变化 TB 时的平均变异系数
    tb_cvs = []
    for bs in bs_values:
        tp_at_bs = [grid_data.get((tb, bs), 0) for tb in tb_values]
        tp_at_bs = [t for t in tp_at_bs if t > 0]
        if len(tp_at_bs) >= 2:
            mean_t = statistics.mean(tp_at_bs)
            std_t = statistics.stdev(tp_at_bs)
            if mean_t > 0:
                tb_cvs.append(std_t / mean_t)
    result["tb_sensitivity"] = statistics.mean(tb_cvs) if tb_cvs else 0.0

    # BS 敏感度: 固定 TB，变化 BS 时的平均变异系数
    bs_cvs = []
    for tb in tb_values:
        tp_at_tb = [grid_data.get((tb, bs), 0) for bs in bs_values]
        tp_at_tb = [t for t in tp_at_tb if t > 0]
        if len(tp_at_tb) >= 2:
            mean_t = statistics.mean(tp_at_tb)
            std_t = statistics.stdev(tp_at_tb)
            if mean_t > 0:
                bs_cvs.append(std_t / mean_t)
    result["bs_sensitivity"] = statistics.mean(bs_cvs) if bs_cvs else 0.0

    return result


def detect_models() -> list[str]:
    """自动检测已有结果的模型"""
    models = []
    for model_short in MODEL_MAP:
        for exp in EXP_PATTERNS:
            if find_result_dir(exp, model_short):
                models.append(model_short)
                break
    return models


def generate_markdown_report(
    data: dict,
    models: list[str],
    experiments: list[str],
) -> str:
    """生成 Markdown 报告"""
    lines = []
    lines.append("# PD Scheduler 实验总结")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"**模型**: {', '.join(models)}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for exp in experiments:
        if exp not in data or not data[exp]:
            continue

        lines.append(f"## {exp}")
        lines.append("")

        # 表头 - 吞吐量对比
        header = "| Strategy |"
        separator = "|----------|"
        for model in models:
            header += f" {model} |"
            separator += "------------|"
        lines.append(header)
        lines.append(separator)

        # 获取 baseline throughput
        baseline_tp = {}
        for m in models:
            sched_data = data[exp].get(m, {}).get("baseline", {})
            baseline_tp[m] = sched_data.get("throughput", 0) if isinstance(sched_data, dict) else 0

        # 每行一个 scheduler
        for sched in SCHEDULERS:
            row = f"| {sched} |"
            for model in models:
                sched_data = data[exp].get(model, {}).get(sched, {})
                if isinstance(sched_data, dict):
                    tp = sched_data.get("throughput", 0)
                else:
                    tp = 0
                if tp > 0:
                    if sched == "baseline":
                        row += f" {tp:.2f} |"
                    else:
                        base = baseline_tp.get(model, 0)
                        if base > 0:
                            improvement = (tp / base - 1) * 100
                            row += f" {tp:.2f} ({improvement:+.1f}%) |"
                        else:
                            row += f" {tp:.2f} |"
                else:
                    row += " - |"
            lines.append(row)

        lines.append("")

        # 最优配置表格 (TB, BS)
        lines.append("### 最优配置 (TB, BS)")
        lines.append("")
        header = "| Strategy |"
        separator = "|----------|"
        for model in models:
            header += f" {model} |"
            separator += "------------|"
        lines.append(header)
        lines.append(separator)

        for sched in SCHEDULERS:
            row = f"| {sched} |"
            for model in models:
                sched_data = data[exp].get(model, {}).get(sched, {})
                if isinstance(sched_data, dict) and sched_data.get("throughput", 0) > 0:
                    tb = sched_data.get("tb", "-")
                    bs = sched_data.get("bs", "-")
                    row += f" TB={tb}, BS={bs} |"
                else:
                    row += " - |"
            lines.append(row)

        lines.append("")

        # 参数敏感度分析表格
        lines.append("### 参数敏感度分析")
        lines.append("")
        lines.append("- **CV (变异系数)**: 吞吐量在所有配置下的标准差/均值，越大表示越敏感")
        lines.append("- **Range**: 最大吞吐量/最小吞吐量，越大表示配置影响越大")
        lines.append("- **TB_Sens**: 固定 BS 时，TB 变化对吞吐量的影响 (平均变异系数)")
        lines.append("- **BS_Sens**: 固定 TB 时，BS 变化对吞吐量的影响 (平均变异系数)")
        lines.append("")

        header = "| Strategy | Metric |"
        separator = "|----------|--------|"
        for model in models:
            header += f" {model} |"
            separator += "--------|"
        lines.append(header)
        lines.append(separator)

        metrics = [("CV", "cv", True), ("Range", "range_ratio", False), ("TB_Sens", "tb_sensitivity", True), ("BS_Sens", "bs_sensitivity", True)]
        for sched in SCHEDULERS:
            for metric_name, metric_key, is_percent in metrics:
                row = f"| {sched} | {metric_name} |"
                for model in models:
                    sched_data = data[exp].get(model, {}).get(sched, {})
                    if isinstance(sched_data, dict) and sched_data.get("throughput", 0) > 0:
                        sens = compute_sensitivity(sched_data)
                        val = sens.get(metric_key, 0)
                        if is_percent:
                            row += f" {val:.1%} |"
                        else:
                            row += f" {val:.2f}x |"
                    else:
                        row += " - |"
                lines.append(row)

        lines.append("")

    return "\n".join(lines)


def generate_console_report(
    data: dict,
    models: list[str],
    experiments: list[str],
) -> str:
    """生成控制台报告"""
    lines = []
    lines.append("=" * 80)
    lines.append("PD Scheduler 实验总结")
    lines.append("=" * 80)
    lines.append("")

    for exp in experiments:
        if exp not in data or not data[exp]:
            continue

        lines.append(f"## {exp}")
        lines.append("")

        # 计算列宽
        col_width = max(20, max((len(m) for m in models), default=10) + 8)

        # 表头 - 吞吐量
        header = f"{'Strategy':<12}"
        for model in models:
            header += f" {model:<{col_width}}"
        lines.append(header)
        lines.append("-" * len(header))

        # 获取 baseline throughput
        baseline_tp = {}
        for m in models:
            sched_data = data[exp].get(m, {}).get("baseline", {})
            baseline_tp[m] = sched_data.get("throughput", 0) if isinstance(sched_data, dict) else 0

        # 每行一个 scheduler - 吞吐量
        for sched in SCHEDULERS:
            row = f"{sched:<12}"
            for model in models:
                sched_data = data[exp].get(model, {}).get(sched, {})
                if isinstance(sched_data, dict):
                    tp = sched_data.get("throughput", 0)
                else:
                    tp = 0
                if tp > 0:
                    if sched == "baseline":
                        cell = f"{tp:.2f}"
                    else:
                        base = baseline_tp.get(model, 0)
                        if base > 0:
                            improvement = (tp / base - 1) * 100
                            cell = f"{tp:.2f} ({improvement:+.1f}%)"
                        else:
                            cell = f"{tp:.2f}"
                    row += f" {cell:<{col_width}}"
                else:
                    row += f" {'-':<{col_width}}"
            lines.append(row)

        lines.append("")

        # 最优配置表格
        lines.append("  最优配置 (TB, BS):")
        lines.append("")
        header = f"{'Strategy':<12}"
        for model in models:
            header += f" {model:<{col_width}}"
        lines.append(header)
        lines.append("-" * len(header))

        for sched in SCHEDULERS:
            row = f"{sched:<12}"
            for model in models:
                sched_data = data[exp].get(model, {}).get(sched, {})
                if isinstance(sched_data, dict) and sched_data.get("throughput", 0) > 0:
                    tb = sched_data.get("tb", "-")
                    bs = sched_data.get("bs", "-")
                    cell = f"TB={tb}, BS={bs}"
                    row += f" {cell:<{col_width}}"
                else:
                    row += f" {'-':<{col_width}}"
            lines.append(row)

        lines.append("")

        # 参数敏感度分析
        lines.append("  参数敏感度分析:")
        lines.append("  (CV=变异系数, Range=最大/最小比, TB_Sens=TB敏感度, BS_Sens=BS敏感度)")
        lines.append("")

        sens_col_width = max(50, max((len(m) for m in models), default=10) + 35)
        header = f"{'Strategy':<12}"
        for model in models:
            header += f" {model:<{sens_col_width}}"
        lines.append(header)
        lines.append("-" * len(header))

        for sched in SCHEDULERS:
            row = f"{sched:<12}"
            for model in models:
                sched_data = data[exp].get(model, {}).get(sched, {})
                if isinstance(sched_data, dict) and sched_data.get("throughput", 0) > 0:
                    sens = compute_sensitivity(sched_data)
                    cell = f"CV={sens['cv']:.1%} Range={sens['range_ratio']:.2f}x TB={sens['tb_sensitivity']:.1%} BS={sens['bs_sensitivity']:.1%}"
                    row += f" {cell:<{sens_col_width}}"
                else:
                    row += f" {'-':<{sens_col_width}}"
            lines.append(row)

        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="生成 PD Scheduler 实验总结")
    parser.add_argument(
        "models",
        nargs="*",
        help="模型短名列表 (如 Qwen3-8B)，不指定则自动检测",
    )
    parser.add_argument(
        "--experiments",
        default="sharegpt,numina_math,longbench,wildchat",
        help="实验列表，逗号分隔",
    )
    parser.add_argument(
        "--output", "-o",
        help="输出文件路径",
    )
    parser.add_argument(
        "--schedulers",
        default=",".join(DEFAULT_SCHEDULERS),
        help="调度器列表，逗号分隔 (默认: baseline,pd_ratio,pd_ifr)。"
             "支持版本后缀，如: baseline,pd_ratio,pd_ifr_1",
    )
    args = parser.parse_args()

    # 设置全局 SCHEDULERS
    global SCHEDULERS
    SCHEDULERS = [s.strip() for s in args.schedulers.split(",")]

    # 确定模型列表
    models = args.models if args.models else detect_models()
    if not models:
        print("错误: 未找到任何实验结果")
        return 1

    experiments = args.experiments.split(",")

    print(f"模型: {models}")
    print(f"实验: {experiments}")
    print(f"调度器: {SCHEDULERS}")
    print("")

    # 收集数据: data[exp][model][scheduler] = throughput
    data = defaultdict(dict)

    for exp in experiments:
        print(f"收集 {exp} 数据...")
        for model in models:
            result = collect_experiment_data(exp, model)
            if result:
                data[exp][model] = result
                print(f"  {model}: {list(result.keys())}")
            else:
                print(f"  {model}: 未找到")

    print("")

    # 生成报告
    console_report = generate_console_report(data, models, experiments)
    print(console_report)

    md_report = generate_markdown_report(data, models, experiments)

    # 保存
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_file = Path(args.output)
    else:
        model_str = "_".join(models)
        exp_str = "_".join(experiments)
        output_file = EVAL_DIR / f"report_{model_str}_{exp_str}.md"

    with open(output_file, "w") as f:
        f.write(md_report)

    print(f"报告已保存: {output_file}")
    return 0


if __name__ == "__main__":
    exit(main())
