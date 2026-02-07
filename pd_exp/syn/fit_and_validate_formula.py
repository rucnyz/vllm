#!/usr/bin/env python3
"""
用 throughput 实验数据拟合公式参数，并验证公式正确性。

思路：
1. 从 k-sweep throughput 数据拟合 α_d, β_d, α_p, β_p
2. 用拟合参数预测其他 scenario 的 throughput
3. 对比预测 vs 实际，验证公式

公式：
    Throughput(k) = k / (E[T_d(k)] + E[T_p(k)])

    E[T_d(k)] = (α_d / p) · ln(N / (N - k)) + (β_d / p) · k
    E[T_p(k)] = α_p + β_p · k · E[I]

    其中 p = 1 / E[O]
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt


def load_kstar_sweep_data(results_dir: Path) -> dict:
    """加载 k-sweep 实验数据"""
    data = {}

    # 查找所有 in*_out* 子目录
    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        match = re.match(r'in(\d+)_out(\d+)', subdir.name)
        if not match:
            continue

        in_len = int(match.group(1))
        out_len = int(match.group(2))
        scenario_key = (in_len, out_len)

        # 加载该 scenario 下的所有实验结果
        k_values = []
        throughputs = []
        actual_Ns = []

        for json_file in sorted(subdir.glob('bench_*.json')):
            # 解析 k 值 - 只处理 fixed* 文件
            # bench_fixed512.json -> k=512
            k_match = re.search(r'bench_fixed(\d+)\.json', json_file.name)
            if not k_match:
                continue

            mode = 'fixed'
            k = int(k_match.group(1))

            # 读取数据
            with open(json_file) as f:
                bench_data = json.load(f)

            throughput = bench_data.get('output_throughput', 0)
            if throughput > 0:
                k_values.append(k)
                throughputs.append(throughput)

                # 尝试读取对应的 stats 文件获取实际 N
                stats_file = subdir / f'{mode}{k}_stats.json'
                if stats_file.exists():
                    try:
                        with open(stats_file) as f:
                            stats = json.load(f)
                    except json.JSONDecodeError:
                        actual_Ns.append(None)
                        continue
                    # 获取实际的 N（从 schedule stats）
                    if 'batches' in stats:
                        Ns = [b.get('num_decodes', 0) for b in stats['batches'] if b.get('num_decodes', 0) > 0]
                        if Ns:
                            actual_Ns.append(np.mean(Ns))
                        else:
                            actual_Ns.append(None)
                    else:
                        actual_Ns.append(None)
                else:
                    actual_Ns.append(None)

        if k_values:
            # 按 k 排序
            sorted_indices = np.argsort(k_values)
            data[scenario_key] = {
                'in_len': in_len,
                'out_len': out_len,
                'k_values': np.array(k_values)[sorted_indices],
                'throughputs': np.array(throughputs)[sorted_indices],
                'actual_Ns': [actual_Ns[i] for i in sorted_indices] if actual_Ns else None
            }

    return data


def throughput_formula(k, N, E_I, E_O, alpha_d, beta_d, alpha_p, beta_p):
    """
    计算公式预测的 throughput

    Throughput(k) = k / (E[T_d(k)] + E[T_p(k)])

    E[T_d(k)] = (α_d / p) · ln(N / (N - k)) + (β_d / p) · k
    E[T_p(k)] = α_p + β_p · k · E[I]
    """
    p = 1.0 / E_O

    # 避免 log(0) 或 log(负数)
    k_safe = np.minimum(k, N - 1)

    # E[T_d(k)]
    E_Td = (alpha_d / p) * np.log(N / (N - k_safe)) + (beta_d / p) * k_safe

    # E[T_p(k)]
    E_Tp = alpha_p + beta_p * k_safe * E_I

    # Throughput
    throughput = k_safe / (E_Td + E_Tp)

    return throughput


def fit_parameters(data: dict, N: int, scenarios_to_fit: list = None):
    """
    用多个 scenario 的数据联合拟合参数

    Args:
        data: k-sweep 数据
        N: batch size (假设固定)
        scenarios_to_fit: 用于拟合的 scenario 列表，格式 [(in_len, out_len), ...]

    Returns:
        拟合的参数 (alpha_d, beta_d, alpha_p, beta_p)
    """
    if scenarios_to_fit is None:
        scenarios_to_fit = list(data.keys())

    # 收集所有数据点
    all_k = []
    all_throughput = []
    all_E_I = []
    all_E_O = []

    for scenario_key in scenarios_to_fit:
        if scenario_key not in data:
            continue

        scenario = data[scenario_key]
        in_len = scenario['in_len']
        out_len = scenario['out_len']

        # E[I] = in_len (假设输入长度固定)
        # E[O] = out_len (geometric 分布的期望)
        E_I = in_len
        E_O = out_len

        for k, throughput in zip(scenario['k_values'], scenario['throughputs']):
            if k < N:  # 只用有效的 k 值
                all_k.append(k)
                all_throughput.append(throughput)
                all_E_I.append(E_I)
                all_E_O.append(E_O)

    all_k = np.array(all_k)
    all_throughput = np.array(all_throughput)
    all_E_I = np.array(all_E_I)
    all_E_O = np.array(all_E_O)

    print(f"拟合数据点数: {len(all_k)}")
    print(f"k 范围: {all_k.min()} - {all_k.max()}")
    print(f"throughput 范围: {all_throughput.min():.0f} - {all_throughput.max():.0f}")

    # 定义损失函数
    def loss(params):
        alpha_d, beta_d, alpha_p, beta_p = params

        # 确保参数为正
        if alpha_d <= 0 or beta_d <= 0 or alpha_p <= 0 or beta_p <= 0:
            return 1e10

        predicted = throughput_formula(all_k, N, all_E_I, all_E_O,
                                        alpha_d, beta_d, alpha_p, beta_p)

        # 相对误差的平方和
        relative_errors = (predicted - all_throughput) / all_throughput
        return np.sum(relative_errors ** 2)

    # 初始猜测（基于之前的参数量级）
    x0 = [0.005, 0.0001, 0.005, 1e-5]

    # 优化
    result = minimize(loss, x0, method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})

    alpha_d, beta_d, alpha_p, beta_p = result.x

    print(f"\n拟合结果:")
    print(f"  α_d = {alpha_d:.6e}")
    print(f"  β_d = {beta_d:.6e}")
    print(f"  α_p = {alpha_p:.6e}")
    print(f"  β_p = {beta_p:.6e}")
    print(f"  Loss = {result.fun:.6f}")

    # 计算 R²
    predicted = throughput_formula(all_k, N, all_E_I, all_E_O,
                                    alpha_d, beta_d, alpha_p, beta_p)
    ss_res = np.sum((all_throughput - predicted) ** 2)
    ss_tot = np.sum((all_throughput - np.mean(all_throughput)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    print(f"  R² = {r_squared:.4f}")

    return alpha_d, beta_d, alpha_p, beta_p, r_squared


def validate_formula(data: dict, N: int, params: tuple, scenarios_to_validate: list = None):
    """
    用拟合的参数验证公式在其他 scenario 上的预测能力
    """
    alpha_d, beta_d, alpha_p, beta_p = params

    if scenarios_to_validate is None:
        scenarios_to_validate = list(data.keys())

    results = []

    for scenario_key in scenarios_to_validate:
        if scenario_key not in data:
            continue

        scenario = data[scenario_key]
        in_len = scenario['in_len']
        out_len = scenario['out_len']
        E_I = in_len
        E_O = out_len

        k_values = scenario['k_values']
        actual_throughputs = scenario['throughputs']

        # 预测
        predicted_throughputs = throughput_formula(k_values, N, E_I, E_O,
                                                    alpha_d, beta_d, alpha_p, beta_p)

        # 计算误差
        relative_errors = (predicted_throughputs - actual_throughputs) / actual_throughputs
        mae = np.mean(np.abs(relative_errors))

        # 找最优 k
        actual_best_k = k_values[np.argmax(actual_throughputs)]
        predicted_best_k_idx = np.argmax(predicted_throughputs)
        predicted_best_k = k_values[predicted_best_k_idx]

        # 计算公式预测的理论最优 k*
        k_fine = np.linspace(1, N-1, 1000)
        throughput_fine = throughput_formula(k_fine, N, E_I, E_O,
                                              alpha_d, beta_d, alpha_p, beta_p)
        formula_best_k = k_fine[np.argmax(throughput_fine)]

        results.append({
            'scenario': f'in{in_len}_out{out_len}',
            'actual_best_k': actual_best_k,
            'predicted_best_k': predicted_best_k,
            'formula_best_k': formula_best_k,
            'mae': mae,
            'k_values': k_values,
            'actual_throughputs': actual_throughputs,
            'predicted_throughputs': predicted_throughputs,
        })

        print(f"\n{scenario_key}:")
        print(f"  实际最优 k: {actual_best_k}")
        print(f"  公式预测最优 k*: {formula_best_k:.0f}")
        print(f"  平均相对误差: {mae*100:.1f}%")

    return results


def plot_validation(results: list, output_path: str, N: int):
    """绘制验证结果"""
    n_scenarios = len(results)
    fig, axes = plt.subplots(2, (n_scenarios + 1) // 2, figsize=(5 * ((n_scenarios + 1) // 2), 8))
    axes = axes.flatten()

    for i, result in enumerate(results):
        ax = axes[i]

        k_values = result['k_values']
        actual = result['actual_throughputs']
        predicted = result['predicted_throughputs']

        ax.plot(k_values, actual, 'o-', label='Actual', markersize=4)
        ax.plot(k_values, predicted, 's--', label='Formula', markersize=4, alpha=0.7)

        ax.axvline(result['actual_best_k'], color='blue', linestyle=':', alpha=0.5,
                   label=f'Actual best k={result["actual_best_k"]}')
        ax.axvline(result['formula_best_k'], color='orange', linestyle=':', alpha=0.5,
                   label=f'Formula k*={result["formula_best_k"]:.0f}')

        ax.set_xlabel('k')
        ax.set_ylabel('Throughput (tok/s)')
        ax.set_title(f'{result["scenario"]}\nMAE={result["mae"]*100:.1f}%')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for i in range(len(results), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f'Formula Validation (N={N})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存到: {output_path}")
    plt.close()


def calculate_optimal_kstar(N: int, E_I: int, E_O: int, params: tuple):
    """用拟合参数计算理论最优 k*"""
    alpha_d, beta_d, alpha_p, beta_p = params

    k_values = np.linspace(1, N-1, 10000)
    throughputs = throughput_formula(k_values, N, E_I, E_O,
                                      alpha_d, beta_d, alpha_p, beta_p)

    best_idx = np.argmax(throughputs)
    return k_values[best_idx], throughputs[best_idx]


def main():
    parser = argparse.ArgumentParser(description='拟合并验证 PD 调度公式')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='k-sweep 实验结果目录')
    parser.add_argument('--N', type=int, default=1024,
                        help='Batch size N')
    parser.add_argument('--fit-scenarios', type=str, default=None,
                        help='用于拟合的 scenarios，格式: "64,128;128,256" (in,out pairs)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出图表路径')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"错误: 目录不存在 {results_dir}")
        return

    # 加载数据
    print(f"加载数据: {results_dir}")
    data = load_kstar_sweep_data(results_dir)

    if not data:
        print("错误: 没有找到有效数据")
        return

    print(f"找到 {len(data)} 个 scenarios:")
    for key in sorted(data.keys()):
        scenario = data[key]
        print(f"  {key}: {len(scenario['k_values'])} 个 k 值, "
              f"throughput {scenario['throughputs'].min():.0f} - {scenario['throughputs'].max():.0f}")

    # 解析要拟合的 scenarios
    if args.fit_scenarios:
        fit_scenarios = []
        for pair in args.fit_scenarios.split(';'):
            in_len, out_len = map(int, pair.split(','))
            fit_scenarios.append((in_len, out_len))
    else:
        # 默认用所有 scenarios
        fit_scenarios = list(data.keys())

    print(f"\n用于拟合的 scenarios: {fit_scenarios}")

    # 拟合参数
    print("\n" + "="*60)
    print("拟合参数")
    print("="*60)

    alpha_d, beta_d, alpha_p, beta_p, r_squared = fit_parameters(data, args.N, fit_scenarios)
    params = (alpha_d, beta_d, alpha_p, beta_p)

    # 验证
    print("\n" + "="*60)
    print("验证结果")
    print("="*60)

    validation_results = validate_formula(data, args.N, params)

    # 绘图
    output_path = args.output or str(results_dir / 'formula_validation.png')
    plot_validation(validation_results, output_path, args.N)

    # 总结
    print("\n" + "="*60)
    print("总结")
    print("="*60)

    print(f"\n拟合参数 (N={args.N}):")
    print(f"  α_d = {alpha_d:.6e} s")
    print(f"  β_d = {beta_d:.6e} s/token")
    print(f"  α_p = {alpha_p:.6e} s")
    print(f"  β_p = {beta_p:.6e} s/token")
    print(f"  R² = {r_squared:.4f}")

    print(f"\n各 scenario 的最优 k* 对比:")
    print(f"{'Scenario':<20} {'实际最优k':>12} {'公式k*':>12} {'误差':>10}")
    print("-" * 56)
    for r in validation_results:
        k_error = abs(r['formula_best_k'] - r['actual_best_k']) / r['actual_best_k'] * 100
        print(f"{r['scenario']:<20} {r['actual_best_k']:>12} {r['formula_best_k']:>12.0f} {k_error:>9.1f}%")


if __name__ == '__main__':
    main()
