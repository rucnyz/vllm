#!/usr/bin/env python3
"""
Calculate theoretical optimal k* for PD scheduler based on the corrected formula.

Formula:
    E[T_d(k)] = (α_d/p) · ln(1/(1-θ)) + (β_d/p) · k,  where θ = k/N, p = 1/E[O]
    E[T_p(k)] = α_p + β_p · k · μ_L
    Throughput(k) = k / (E[T_d(k)] + E[T_p(k)])
"""

import argparse
import math
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class HardwareParams:
    """Hardware calibration parameters."""
    alpha_p: float = 0.006528784356021418
    beta_p: float = 6.498792400220424e-06
    alpha_d: float = 0.004303444935141221
    beta_d: float = 0.00023557651251992446


@dataclass
class Scenario:
    """Experiment scenario parameters."""
    input_len: int
    output_len: int  # E[O] for geometric distribution
    batch_size: int  # N

    @property
    def p(self) -> float:
        """Geometric distribution parameter p = 1/E[O]."""
        return 1.0 / self.output_len

    @property
    def mu_L(self) -> float:
        """Mean output length."""
        return float(self.output_len)


def calculate_decode_time(k: int, scenario: Scenario, hw: HardwareParams) -> float:
    """Calculate expected decode time E[T_d(k)]."""
    N = scenario.batch_size
    p = scenario.p
    theta = k / N

    if theta >= 1.0:
        return float('inf')

    # E[T_d(k)] = (α_d/p) · ln(1/(1-θ)) + (β_d/p) · k
    log_term = (hw.alpha_d / p) * math.log(1.0 / (1.0 - theta))
    linear_term = (hw.beta_d / p) * k

    return log_term + linear_term


def calculate_prefill_time(k: int, scenario: Scenario, hw: HardwareParams) -> float:
    """Calculate expected prefill time E[T_p(k)]."""
    # E[T_p(k)] = α_p + β_p · k · μ_L
    return hw.alpha_p + hw.beta_p * k * scenario.mu_L


def calculate_throughput(k: int, scenario: Scenario, hw: HardwareParams) -> float:
    """Calculate throughput for given k."""
    if k <= 0:
        return 0.0

    t_d = calculate_decode_time(k, scenario, hw)
    t_p = calculate_prefill_time(k, scenario, hw)

    if t_d == float('inf') or t_d + t_p <= 0:
        return 0.0

    return k / (t_d + t_p)


def find_optimal_k(scenario: Scenario, hw: HardwareParams) -> Tuple[int, float]:
    """Find the optimal k* that maximizes throughput."""
    best_k = 1
    best_throughput = 0.0

    # Search from 1 to N-1 (k=N would make theta=1, causing log to diverge)
    for k in range(1, scenario.batch_size):
        throughput = calculate_throughput(k, scenario, hw)
        if throughput > best_throughput:
            best_throughput = throughput
            best_k = k

    return best_k, best_throughput


def analyze_scenario(scenario: Scenario, hw: HardwareParams, k_values: List[int] = None):
    """Analyze a single scenario and print results."""
    print(f"\n{'='*60}")
    print(f"Scenario: input={scenario.input_len}, output={scenario.output_len}, N={scenario.batch_size}")
    print(f"  p = 1/{scenario.output_len} = {scenario.p:.6f}")
    print(f"  β_d/p = {hw.beta_d/scenario.p:.4f}")
    print(f"  α_d/p = {hw.alpha_d/scenario.p:.4f}")
    print(f"{'='*60}")

    # Find optimal k*
    optimal_k, optimal_throughput = find_optimal_k(scenario, hw)
    print(f"\n>>> Optimal k* = {optimal_k} (k*/N = {optimal_k/scenario.batch_size:.1%})")
    print(f">>> Max throughput = {optimal_throughput:.4f} req/s")

    # Print throughput for various k values
    if k_values is None:
        # Default k values to show
        k_values = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 64, 80, 100, 120]

    k_values = [k for k in k_values if k < scenario.batch_size]

    print(f"\n{'k':>6} | {'θ=k/N':>8} | {'E[T_d]':>10} | {'E[T_p]':>10} | {'Total':>10} | {'Tput':>10} | {'vs opt':>8}")
    print("-" * 75)

    for k in k_values:
        t_d = calculate_decode_time(k, scenario, hw)
        t_p = calculate_prefill_time(k, scenario, hw)
        total = t_d + t_p
        throughput = calculate_throughput(k, scenario, hw)
        vs_optimal = (throughput / optimal_throughput - 1) * 100

        marker = " <-- optimal" if k == optimal_k else ""
        print(f"{k:>6} | {k/scenario.batch_size:>8.3f} | {t_d:>10.4f} | {t_p:>10.4f} | {total:>10.4f} | {throughput:>10.4f} | {vs_optimal:>+7.2f}%{marker}")


def main():
    parser = argparse.ArgumentParser(description="Calculate theoretical optimal k* for PD scheduler")
    parser.add_argument("--batch-size", "-N", type=int, default=128, help="Batch size N")
    parser.add_argument("--scenarios", type=str, nargs="+", default=["1,8", "1,16", "1,32", "1,64", "1,128", "1,256", "1,512", "1,768"],
                        help="Scenarios as 'input,output' pairs")
    parser.add_argument("--alpha-p", type=float, default=0.006528784356021418)
    parser.add_argument("--beta-p", type=float, default=6.498792400220424e-06)
    parser.add_argument("--alpha-d", type=float, default=0.004303444935141221)
    parser.add_argument("--beta-d", type=float, default=0.00023557651251992446)

    args = parser.parse_args()

    hw = HardwareParams(
        alpha_p=args.alpha_p,
        beta_p=args.beta_p,
        alpha_d=args.alpha_d,
        beta_d=args.beta_d,
    )

    print("Hardware Parameters:")
    print(f"  α_p = {hw.alpha_p:.6e}")
    print(f"  β_p = {hw.beta_p:.6e}")
    print(f"  α_d = {hw.alpha_d:.6e}")
    print(f"  β_d = {hw.beta_d:.6e}")

    # Parse scenarios
    scenarios = []
    for s in args.scenarios:
        parts = s.replace(" ", ",").split(",")
        input_len, output_len = int(parts[0]), int(parts[1])
        scenarios.append(Scenario(input_len, output_len, args.batch_size))

    # Analyze each scenario
    for scenario in scenarios:
        analyze_scenario(scenario, hw)

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Scenario':>20} | {'p':>10} | {'k*':>6} | {'k*/N':>8} | {'Tput':>10}")
    print("-" * 60)

    for scenario in scenarios:
        optimal_k, optimal_throughput = find_optimal_k(scenario, hw)
        scenario_name = f"in{scenario.input_len}_out{scenario.output_len}"
        print(f"{scenario_name:>20} | {scenario.p:>10.6f} | {optimal_k:>6} | {optimal_k/scenario.batch_size:>8.1%} | {optimal_throughput:>10.4f}")


if __name__ == "__main__":
    main()
