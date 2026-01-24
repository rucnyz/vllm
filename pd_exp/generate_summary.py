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

SCHEDULERS = ["baseline", "pd_ratio", "pd_direct"]
# 别名映射：将新命名映射到标准命名
SCHEDULER_ALIASES = {
    "pd_kratio": "pd_ratio",
    "pd_dynamic": "pd_direct",
}


def find_result_dir(exp: str, model: str) -> Path | None:
    """查找实验结果目录"""
    pattern, _ = EXP_PATTERNS[exp]
    pattern = pattern.format(model=model)
    matches = sorted(glob(str(OUTPUT_DIR / pattern)))
    return Path(matches[-1]) if matches else None


def collect_experiment_data(exp: str, model: str) -> dict | None:
    """收集单个实验的数据，返回每个 scheduler 的最优 throughput"""
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

        # 提取每个 scheduler 的最优 throughput
        results = {}
        for sched in SCHEDULERS:
            if sched in optimal:
                results[sched] = optimal[sched]["metrics"].get("throughput", 0)

        # 检查别名
        for alias, canonical in SCHEDULER_ALIASES.items():
            if alias in optimal and canonical not in results:
                results[canonical] = optimal[alias]["metrics"].get("throughput", 0)

        return results if results else None

    except Exception as e:
        print(f"  警告: {exp}/{model} 分析失败: {e}")
        return None


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

        # 表头
        header = "| Strategy |"
        separator = "|----------|"
        for model in models:
            header += f" {model} |"
            separator += "------------|"
        lines.append(header)
        lines.append(separator)

        # 获取 baseline
        baseline_tp = {m: data[exp].get(m, {}).get("baseline", 0) for m in models}

        # 每行一个 scheduler
        for sched in SCHEDULERS:
            row = f"| {sched} |"
            for model in models:
                tp = data[exp].get(model, {}).get(sched, 0)
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
        col_width = max(16, max((len(m) for m in models), default=10) + 4)

        # 表头
        header = f"{'Strategy':<12}"
        for model in models:
            header += f" {model:<{col_width}}"
        lines.append(header)
        lines.append("-" * len(header))

        # 获取 baseline
        baseline_tp = {m: data[exp].get(m, {}).get("baseline", 0) for m in models}

        # 每行一个 scheduler
        for sched in SCHEDULERS:
            row = f"{sched:<12}"
            for model in models:
                tp = data[exp].get(model, {}).get(sched, 0)
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
    args = parser.parse_args()

    # 确定模型列表
    models = args.models if args.models else detect_models()
    if not models:
        print("错误: 未找到任何实验结果")
        return 1

    experiments = args.experiments.split(",")

    print(f"模型: {models}")
    print(f"实验: {experiments}")
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
