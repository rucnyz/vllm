#!/usr/bin/env python3
"""
Analyze and compare scheduler statistics between baseline and PD scheduler.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_stats(filepath: str) -> list[dict]:
    """Load stats from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    return data["stats"]


def analyze_stats(stats: list[dict], name: str) -> dict:
    """Compute summary statistics."""
    total_tokens = [s["total_tokens"] for s in stats]
    prefill_tokens = [s["prefill_tokens"] for s in stats]
    decode_tokens = [s["decode_tokens"] for s in stats]
    num_waiting = [s["num_waiting_reqs"] for s in stats]
    num_running = [s["num_running_reqs"] for s in stats]
    num_scheduled = [s["num_scheduled_reqs"] for s in stats]
    timestamps = [s["timestamp"] for s in stats]

    # Find warmup end (when num_running first reaches > 10 after initial phase)
    warmup_end_idx = 0
    for i, s in enumerate(stats):
        if s["num_running_reqs"] > 10 and s["num_waiting_reqs"] > 0:
            warmup_end_idx = i
            break

    # Calculate throughput (tokens per second)
    if len(timestamps) > 1:
        duration = timestamps[-1] - timestamps[0]
        total_tokens_sum = sum(total_tokens)
        throughput = total_tokens_sum / duration if duration > 0 else 0
    else:
        throughput = 0

    return {
        "name": name,
        "num_steps": len(stats),
        "total_duration": timestamps[-1] if timestamps else 0,
        "warmup_end_idx": warmup_end_idx,
        "total_tokens_sum": sum(total_tokens),
        "prefill_tokens_sum": sum(prefill_tokens),
        "decode_tokens_sum": sum(decode_tokens),
        "throughput_tokens_per_sec": throughput,
        "avg_tokens_per_step": np.mean(total_tokens),
        "avg_waiting": np.mean(num_waiting),
        "avg_running": np.mean(num_running),
        "avg_scheduled": np.mean(num_scheduled),
        "max_waiting": max(num_waiting),
        "max_running": max(num_running),
    }


def find_coldstart_end(stats: list[dict]) -> int:
    """Find where coldstart ends (system reaches steady state)."""
    for i, s in enumerate(stats):
        # Steady state: running > 50 or (running > 10 and waiting > 0)
        if s["num_running_reqs"] > 50:
            return i
        if s["num_running_reqs"] > 10 and i > 100:
            return i
    return len(stats) // 10  # Default to 10% of data


def plot_comparison(baseline_stats: list[dict], fixed_stats: list[dict],
                    output_dir: str, scenario: str):
    """Create comparison plots."""

    # Extract data
    def extract_data(stats):
        return {
            "step": list(range(len(stats))),
            "timestamp": [s["timestamp"] for s in stats],
            "total_tokens": [s["total_tokens"] for s in stats],
            "prefill_tokens": [s["prefill_tokens"] for s in stats],
            "decode_tokens": [s["decode_tokens"] for s in stats],
            "num_waiting": [s["num_waiting_reqs"] for s in stats],
            "num_running": [s["num_running_reqs"] for s in stats],
            "num_scheduled": [s["num_scheduled_reqs"] for s in stats],
            "phase": [s.get("phase", -1) for s in stats],
        }

    baseline = extract_data(baseline_stats)
    fixed = extract_data(fixed_stats)

    # Find coldstart end for both
    baseline_coldstart_end = find_coldstart_end(baseline_stats)
    fixed_coldstart_end = find_coldstart_end(fixed_stats)

    print(f"\n=== Coldstart Analysis ===")
    print(f"Baseline coldstart ends at step {baseline_coldstart_end} (t={baseline_stats[baseline_coldstart_end]['timestamp']:.3f}s)")
    print(f"Fixed64 coldstart ends at step {fixed_coldstart_end} (t={fixed_stats[fixed_coldstart_end]['timestamp']:.3f}s)")

    # Create figure with multiple subplots
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle(f"Scheduler Comparison: {scenario}\nBaseline vs Fixed K*=64", fontsize=14)

    # Limit to first N steps for clearer visualization
    max_steps = min(len(baseline["step"]), len(fixed["step"]), 5000)

    # 1. Total tokens per step
    ax = axes[0, 0]
    ax.plot(baseline["step"][:max_steps], baseline["total_tokens"][:max_steps],
            label="Baseline", alpha=0.7, linewidth=0.5)
    ax.plot(fixed["step"][:max_steps], fixed["total_tokens"][:max_steps],
            label="Fixed64", alpha=0.7, linewidth=0.5)
    ax.axvline(x=baseline_coldstart_end, color='blue', linestyle='--', alpha=0.5, label='Baseline coldstart end')
    ax.axvline(x=fixed_coldstart_end, color='orange', linestyle='--', alpha=0.5, label='Fixed64 coldstart end')
    ax.set_xlabel("Schedule Step")
    ax.set_ylabel("Total Tokens")
    ax.set_title("Total Tokens per Schedule Step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Prefill vs Decode tokens (stacked area)
    ax = axes[0, 1]
    # Use rolling average for smoother visualization
    window = 50
    baseline_prefill_smooth = np.convolve(baseline["prefill_tokens"][:max_steps],
                                          np.ones(window)/window, mode='valid')
    baseline_decode_smooth = np.convolve(baseline["decode_tokens"][:max_steps],
                                         np.ones(window)/window, mode='valid')
    fixed_prefill_smooth = np.convolve(fixed["prefill_tokens"][:max_steps],
                                       np.ones(window)/window, mode='valid')
    fixed_decode_smooth = np.convolve(fixed["decode_tokens"][:max_steps],
                                      np.ones(window)/window, mode='valid')

    ax.plot(baseline_prefill_smooth, label="Baseline Prefill", alpha=0.7)
    ax.plot(baseline_decode_smooth, label="Baseline Decode", alpha=0.7)
    ax.plot(fixed_prefill_smooth, label="Fixed64 Prefill", alpha=0.7, linestyle='--')
    ax.plot(fixed_decode_smooth, label="Fixed64 Decode", alpha=0.7, linestyle='--')
    ax.set_xlabel("Schedule Step")
    ax.set_ylabel(f"Tokens (Rolling Avg, window={window})")
    ax.set_title("Prefill vs Decode Tokens")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Number of waiting requests
    ax = axes[1, 0]
    ax.plot(baseline["step"][:max_steps], baseline["num_waiting"][:max_steps],
            label="Baseline", alpha=0.7, linewidth=0.5)
    ax.plot(fixed["step"][:max_steps], fixed["num_waiting"][:max_steps],
            label="Fixed64", alpha=0.7, linewidth=0.5)
    ax.axvline(x=baseline_coldstart_end, color='blue', linestyle='--', alpha=0.5)
    ax.axvline(x=fixed_coldstart_end, color='orange', linestyle='--', alpha=0.5)
    ax.set_xlabel("Schedule Step")
    ax.set_ylabel("Waiting Requests")
    ax.set_title("Waiting Queue Size")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Number of running requests
    ax = axes[1, 1]
    ax.plot(baseline["step"][:max_steps], baseline["num_running"][:max_steps],
            label="Baseline", alpha=0.7, linewidth=0.5)
    ax.plot(fixed["step"][:max_steps], fixed["num_running"][:max_steps],
            label="Fixed64", alpha=0.7, linewidth=0.5)
    ax.axvline(x=baseline_coldstart_end, color='blue', linestyle='--', alpha=0.5)
    ax.axvline(x=fixed_coldstart_end, color='orange', linestyle='--', alpha=0.5)
    ax.set_xlabel("Schedule Step")
    ax.set_ylabel("Running Requests")
    ax.set_title("Running Requests (Batch Size)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Number of scheduled requests per step
    ax = axes[2, 0]
    ax.plot(baseline["step"][:max_steps], baseline["num_scheduled"][:max_steps],
            label="Baseline", alpha=0.7, linewidth=0.5)
    ax.plot(fixed["step"][:max_steps], fixed["num_scheduled"][:max_steps],
            label="Fixed64", alpha=0.7, linewidth=0.5)
    ax.set_xlabel("Schedule Step")
    ax.set_ylabel("Scheduled Requests")
    ax.set_title("Scheduled Requests per Step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Phase distribution (PD scheduler only)
    ax = axes[2, 1]
    if any(p >= 0 for p in fixed["phase"]):
        phase_colors = {0: 'green', 1: 'blue', 2: 'red'}
        phase_names = {0: 'Phase 0 (Initial Prefill)', 1: 'Phase 1 (Decode)', 2: 'Phase 2 (Refill)'}
        for phase_val in [0, 1, 2]:
            phase_steps = [i for i, p in enumerate(fixed["phase"][:max_steps]) if p == phase_val]
            if phase_steps:
                ax.scatter(phase_steps, [fixed["total_tokens"][i] for i in phase_steps],
                          c=phase_colors[phase_val], label=phase_names[phase_val],
                          alpha=0.3, s=1)
        ax.set_xlabel("Schedule Step")
        ax.set_ylabel("Total Tokens")
        ax.set_title("Fixed64: Phase Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 7. Cumulative tokens over time
    ax = axes[3, 0]
    baseline_cumsum = np.cumsum(baseline["total_tokens"])
    fixed_cumsum = np.cumsum(fixed["total_tokens"])
    ax.plot(baseline["timestamp"][:len(baseline_cumsum)], baseline_cumsum,
            label="Baseline", alpha=0.7)
    ax.plot(fixed["timestamp"][:len(fixed_cumsum)], fixed_cumsum,
            label="Fixed64", alpha=0.7)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Cumulative Tokens")
    ax.set_title("Cumulative Tokens over Time (Throughput)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 8. Coldstart zoom (first 500 steps)
    ax = axes[3, 1]
    coldstart_steps = 500
    ax.plot(baseline["step"][:coldstart_steps], baseline["num_running"][:coldstart_steps],
            label="Baseline Running", alpha=0.7)
    ax.plot(fixed["step"][:coldstart_steps], fixed["num_running"][:coldstart_steps],
            label="Fixed64 Running", alpha=0.7)
    ax.plot(baseline["step"][:coldstart_steps], baseline["num_waiting"][:coldstart_steps],
            label="Baseline Waiting", alpha=0.7, linestyle='--')
    ax.plot(fixed["step"][:coldstart_steps], fixed["num_waiting"][:coldstart_steps],
            label="Fixed64 Waiting", alpha=0.7, linestyle='--')
    ax.set_xlabel("Schedule Step")
    ax.set_ylabel("Requests")
    ax.set_title(f"Coldstart Phase (First {coldstart_steps} steps)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / f"scheduler_comparison_{scenario}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {output_path}")
    plt.close()

    return baseline_coldstart_end, fixed_coldstart_end


def analyze_efficiency_gap(baseline_stats: list[dict], fixed_stats: list[dict],
                           baseline_coldstart_end: int, fixed_coldstart_end: int):
    """Analyze where fixed64 is less efficient than baseline."""

    print("\n" + "="*60)
    print("EFFICIENCY GAP ANALYSIS")
    print("="*60)

    # 1. Coldstart duration comparison
    baseline_coldstart_time = baseline_stats[baseline_coldstart_end]["timestamp"]
    fixed_coldstart_time = fixed_stats[fixed_coldstart_end]["timestamp"]

    print(f"\n1. COLDSTART DURATION:")
    print(f"   Baseline: {baseline_coldstart_time:.3f}s ({baseline_coldstart_end} steps)")
    print(f"   Fixed64:  {fixed_coldstart_time:.3f}s ({fixed_coldstart_end} steps)")
    print(f"   Difference: Fixed64 takes {fixed_coldstart_time - baseline_coldstart_time:.3f}s longer")
    print(f"   Fixed64 coldstart is {fixed_coldstart_time/baseline_coldstart_time:.2f}x slower")

    # 2. Tokens processed during coldstart
    baseline_coldstart_tokens = sum(s["total_tokens"] for s in baseline_stats[:baseline_coldstart_end])
    fixed_coldstart_tokens = sum(s["total_tokens"] for s in fixed_stats[:fixed_coldstart_end])

    print(f"\n2. TOKENS DURING COLDSTART:")
    print(f"   Baseline: {baseline_coldstart_tokens} tokens")
    print(f"   Fixed64:  {fixed_coldstart_tokens} tokens")

    # 3. Waiting queue analysis
    print(f"\n3. WAITING QUEUE ANALYSIS:")
    baseline_waiting = [s["num_waiting_reqs"] for s in baseline_stats]
    fixed_waiting = [s["num_waiting_reqs"] for s in fixed_stats]

    print(f"   Baseline - Max waiting: {max(baseline_waiting)}, Avg: {np.mean(baseline_waiting):.2f}")
    print(f"   Fixed64  - Max waiting: {max(fixed_waiting)}, Avg: {np.mean(fixed_waiting):.2f}")

    # Find periods where fixed64 has high waiting but baseline doesn't
    high_waiting_steps = []
    for i in range(min(len(baseline_stats), len(fixed_stats))):
        if fixed_stats[i]["num_waiting_reqs"] > 10 and baseline_stats[i]["num_waiting_reqs"] < 5:
            if fixed_stats[i]["num_running_reqs"] < 10:  # And low running
                high_waiting_steps.append(i)

    if high_waiting_steps:
        print(f"\n   Steps where Fixed64 has high waiting but low running:")
        print(f"   {high_waiting_steps[:20]}..." if len(high_waiting_steps) > 20 else f"   {high_waiting_steps}")

    # 4. Phase analysis for fixed64
    print(f"\n4. PHASE DISTRIBUTION (Fixed64):")
    phase_counts = {0: 0, 1: 0, 2: 0}
    phase_tokens = {0: 0, 1: 0, 2: 0}
    phase_waiting_sum = {0: 0, 1: 0, 2: 0}

    for s in fixed_stats:
        phase = s.get("phase", -1)
        if phase in phase_counts:
            phase_counts[phase] += 1
            phase_tokens[phase] += s["total_tokens"]
            phase_waiting_sum[phase] += s["num_waiting_reqs"]

    total_steps = sum(phase_counts.values())
    for phase in [0, 1, 2]:
        if phase_counts[phase] > 0:
            pct = phase_counts[phase] / total_steps * 100
            avg_tokens = phase_tokens[phase] / phase_counts[phase]
            avg_waiting = phase_waiting_sum[phase] / phase_counts[phase]
            phase_name = {0: "Initial Prefill", 1: "Decode", 2: "Refill Prefill"}[phase]
            print(f"   Phase {phase} ({phase_name}): {phase_counts[phase]} steps ({pct:.1f}%), "
                  f"avg {avg_tokens:.1f} tokens/step, avg waiting {avg_waiting:.1f}")

    # 5. Steady state comparison (after coldstart)
    print(f"\n5. STEADY STATE COMPARISON (after coldstart):")
    baseline_steady = baseline_stats[baseline_coldstart_end:]
    fixed_steady = fixed_stats[fixed_coldstart_end:]

    if baseline_steady and fixed_steady:
        baseline_steady_tokens = sum(s["total_tokens"] for s in baseline_steady)
        fixed_steady_tokens = sum(s["total_tokens"] for s in fixed_steady)
        baseline_steady_time = baseline_steady[-1]["timestamp"] - baseline_steady[0]["timestamp"]
        fixed_steady_time = fixed_steady[-1]["timestamp"] - fixed_steady[0]["timestamp"]

        baseline_throughput = baseline_steady_tokens / baseline_steady_time if baseline_steady_time > 0 else 0
        fixed_throughput = fixed_steady_tokens / fixed_steady_time if fixed_steady_time > 0 else 0

        print(f"   Baseline: {baseline_steady_tokens} tokens in {baseline_steady_time:.2f}s = {baseline_throughput:.0f} tokens/s")
        print(f"   Fixed64:  {fixed_steady_tokens} tokens in {fixed_steady_time:.2f}s = {fixed_throughput:.0f} tokens/s")
        if baseline_throughput > 0:
            print(f"   Fixed64 steady-state throughput is {fixed_throughput/baseline_throughput:.2%} of Baseline")

    # 6. Identify specific inefficiency patterns
    print(f"\n6. INEFFICIENCY PATTERNS:")

    # Pattern: Phase 1 with high waiting (requests starving)
    phase1_high_waiting = [i for i, s in enumerate(fixed_stats)
                          if s.get("phase") == 1 and s["num_waiting_reqs"] > 20 and s["num_running_reqs"] < 50]
    if phase1_high_waiting:
        print(f"   - Phase 1 with high waiting (>20) but low running (<50): {len(phase1_high_waiting)} steps")
        print(f"     This indicates requests are waiting but not being prefilled due to Phase 1 restriction")

    # Pattern: Long periods of single decode
    single_decode_runs = []
    current_run = 0
    for s in fixed_stats:
        if s["num_running_reqs"] == 1 and s["decode_tokens"] == 1 and s["prefill_tokens"] == 0:
            current_run += 1
        else:
            if current_run > 50:
                single_decode_runs.append(current_run)
            current_run = 0
    if single_decode_runs:
        print(f"   - Long single-decode runs (>50 steps): {len(single_decode_runs)} occurrences")
        print(f"     Max length: {max(single_decode_runs)} steps")
        print(f"     This is the coldstart problem - only 1 request being decoded")


def main():
    # File paths
    base_dir = Path("/scratch/yuzhou/zwf/vllm/experiments/serve/online_khat/bs256_c512_n4000/in128_out1024")
    baseline_file = base_dir / "baseline.json"
    fixed_file = base_dir / "fixed64.json"

    print("Loading data...")
    baseline_stats = load_stats(baseline_file)
    fixed_stats = load_stats(fixed_file)

    print(f"Baseline: {len(baseline_stats)} schedule steps")
    print(f"Fixed64:  {len(fixed_stats)} schedule steps")

    # Compute summary stats
    baseline_summary = analyze_stats(baseline_stats, "Baseline")
    fixed_summary = analyze_stats(fixed_stats, "Fixed64")

    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    for summary in [baseline_summary, fixed_summary]:
        print(f"\n{summary['name']}:")
        print(f"  Total steps: {summary['num_steps']}")
        print(f"  Total duration: {summary['total_duration']:.2f}s")
        print(f"  Total tokens: {summary['total_tokens_sum']}")
        print(f"  Throughput: {summary['throughput_tokens_per_sec']:.0f} tokens/s")
        print(f"  Avg tokens/step: {summary['avg_tokens_per_step']:.2f}")
        print(f"  Avg running: {summary['avg_running']:.2f}")
        print(f"  Avg waiting: {summary['avg_waiting']:.2f}")
        print(f"  Max running: {summary['max_running']}")
        print(f"  Max waiting: {summary['max_waiting']}")

    # Plot comparison
    baseline_coldstart_end, fixed_coldstart_end = plot_comparison(
        baseline_stats, fixed_stats, str(base_dir), "in128_out1024"
    )

    # Detailed efficiency analysis
    analyze_efficiency_gap(baseline_stats, fixed_stats, baseline_coldstart_end, fixed_coldstart_end)


if __name__ == "__main__":
    main()
