#!/usr/bin/env python3
"""Plot benchmark results from results.json"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open("results.json", "r") as f:
    data = json.load(f)

results = data["results"]

# Separate results by type
pure_decode = [r for r in results if r["num_prefill"] == 0]
pure_prefill = [r for r in results if r["num_decode"] == 0]
mixed = [r for r in results if r["num_decode"] > 0 and r["num_prefill"] > 0]

# Group mixed by prefill_chunk_size
mixed_2048 = [r for r in mixed if r["prefill_chunk_size"] == 2048]
mixed_4096 = [r for r in mixed if r["prefill_chunk_size"] == 4096]

# Sort by decode_context_len
pure_decode = sorted(pure_decode, key=lambda x: x["decode_context_len"])
mixed_2048 = sorted(mixed_2048, key=lambda x: x["decode_context_len"])
mixed_4096 = sorted(mixed_4096, key=lambda x: x["decode_context_len"])

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"vLLM Benchmark Results - {data['config']['model']}", fontsize=14)

# Plot 1: Execution Time vs Decode Context Length
ax1 = axes[0, 0]
ctx_lens = [r["decode_context_len"] for r in pure_decode]
decode_times = [r["mean_time_ms"] for r in pure_decode]
mixed_2048_times = [r["mean_time_ms"] for r in mixed_2048]
mixed_4096_times = [r["mean_time_ms"] for r in mixed_4096]

ax1.plot(ctx_lens, decode_times, 'o-', label='1D (pure decode)', color='blue')
ax1.plot(ctx_lens, mixed_2048_times, 's-', label='1D + 1P(2048)', color='orange')
ax1.plot(ctx_lens, mixed_4096_times, '^-', label='1D + 1P(4096)', color='green')

# Add horizontal lines for pure prefill
prefill_2048_time = [r["mean_time_ms"] for r in pure_prefill if r["prefill_chunk_size"] == 2048][0]
prefill_4096_time = [r["mean_time_ms"] for r in pure_prefill if r["prefill_chunk_size"] == 4096][0]
ax1.axhline(y=prefill_2048_time, color='orange', linestyle='--', alpha=0.5, label=f'1P(2048) baseline: {prefill_2048_time:.1f}ms')
ax1.axhline(y=prefill_4096_time, color='green', linestyle='--', alpha=0.5, label=f'1P(4096) baseline: {prefill_4096_time:.1f}ms')

ax1.set_xlabel('Decode Context Length')
ax1.set_ylabel('Execution Time (ms)')
ax1.set_title('Execution Time vs Decode Context Length')
ax1.legend(loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log', base=2)

# Plot 2: Overhead of adding decode to prefill
ax2 = axes[0, 1]
overhead_2048 = [r["mean_time_ms"] - prefill_2048_time for r in mixed_2048]
overhead_4096 = [r["mean_time_ms"] - prefill_4096_time for r in mixed_4096]
overhead_pct_2048 = [(r["mean_time_ms"] - prefill_2048_time) / prefill_2048_time * 100 for r in mixed_2048]
overhead_pct_4096 = [(r["mean_time_ms"] - prefill_4096_time) / prefill_4096_time * 100 for r in mixed_4096]

ax2.plot(ctx_lens, overhead_2048, 's-', label='1D + 1P(2048) overhead', color='orange')
ax2.plot(ctx_lens, overhead_4096, '^-', label='1D + 1P(4096) overhead', color='green')
ax2.set_xlabel('Decode Context Length')
ax2.set_ylabel('Overhead (ms)')
ax2.set_title('Overhead of Adding 1 Decode to Prefill')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log', base=2)

# Add percentage labels
ax2_pct = ax2.twinx()
ax2_pct.plot(ctx_lens, overhead_pct_2048, 's:', alpha=0.3, color='orange')
ax2_pct.plot(ctx_lens, overhead_pct_4096, '^:', alpha=0.3, color='green')
ax2_pct.set_ylabel('Overhead (%)', alpha=0.5)

# Plot 3: Throughput comparison
ax3 = axes[1, 0]
decode_throughput = [r["throughput_tokens_per_sec"] for r in pure_decode]
mixed_2048_throughput = [r["throughput_tokens_per_sec"] for r in mixed_2048]
mixed_4096_throughput = [r["throughput_tokens_per_sec"] for r in mixed_4096]

ax3.plot(ctx_lens, decode_throughput, 'o-', label='1D (pure decode)', color='blue')
ax3.set_xlabel('Decode Context Length')
ax3.set_ylabel('Throughput (tok/s) - Decode', color='blue')
ax3.tick_params(axis='y', labelcolor='blue')
ax3.set_title('Throughput vs Decode Context Length')
ax3.set_xscale('log', base=2)
ax3.grid(True, alpha=0.3)

ax3_right = ax3.twinx()
ax3_right.plot(ctx_lens, mixed_2048_throughput, 's-', label='1D + 1P(2048)', color='orange')
ax3_right.plot(ctx_lens, mixed_4096_throughput, '^-', label='1D + 1P(4096)', color='green')
ax3_right.set_ylabel('Throughput (tok/s) - Mixed', color='orange')
ax3_right.tick_params(axis='y', labelcolor='orange')

# Add legends
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_right.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

# Plot 4: Pure decode time breakdown
ax4 = axes[1, 1]
x = np.arange(len(ctx_lens))
width = 0.25

bars1 = ax4.bar(x - width, decode_times, width, label='Pure Decode', color='blue', alpha=0.7)
bars2 = ax4.bar(x, mixed_2048_times, width, label='1D + 1P(2048)', color='orange', alpha=0.7)
bars3 = ax4.bar(x + width, mixed_4096_times, width, label='1D + 1P(4096)', color='green', alpha=0.7)

ax4.set_xlabel('Decode Context Length')
ax4.set_ylabel('Execution Time (ms)')
ax4.set_title('Execution Time Comparison (Bar Chart)')
ax4.set_xticks(x)
ax4.set_xticklabels([str(c) for c in ctx_lens])
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('benchmark_results.png', dpi=150, bbox_inches='tight')
plt.savefig('benchmark_results.pdf', bbox_inches='tight')
print("Saved plots to benchmark_results.png and benchmark_results.pdf")

# Also create a focused plot on mixed throughput
fig2, ax = plt.subplots(figsize=(10, 6))

ax.plot(ctx_lens, mixed_2048_throughput, 's-', label='1D + 1P(2048)', color='orange', linewidth=2, markersize=8)
ax.plot(ctx_lens, mixed_4096_throughput, '^-', label='1D + 1P(4096)', color='green', linewidth=2, markersize=8)

# Add pure prefill baselines
prefill_2048_tp = [r["throughput_tokens_per_sec"] for r in pure_prefill if r["prefill_chunk_size"] == 2048][0]
prefill_4096_tp = [r["throughput_tokens_per_sec"] for r in pure_prefill if r["prefill_chunk_size"] == 4096][0]
ax.axhline(y=prefill_2048_tp, color='orange', linestyle='--', alpha=0.5, label=f'1P(2048) baseline: {prefill_2048_tp:.0f} tok/s')
ax.axhline(y=prefill_4096_tp, color='green', linestyle='--', alpha=0.5, label=f'1P(4096) baseline: {prefill_4096_tp:.0f} tok/s')

ax.set_xlabel('Decode Context Length', fontsize=12)
ax.set_ylabel('Throughput (tokens/sec)', fontsize=12)
ax.set_title(f'Mixed Batch Throughput vs Decode Context Length\n{data["config"]["model"]}', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xscale('log', base=2)

# Add data labels
for i, (ctx, tp2048, tp4096) in enumerate(zip(ctx_lens, mixed_2048_throughput, mixed_4096_throughput)):
    ax.annotate(f'{tp2048:.0f}', (ctx, tp2048), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
    ax.annotate(f'{tp4096:.0f}', (ctx, tp4096), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('mixed_throughput.png', dpi=150, bbox_inches='tight')
print("Saved mixed_throughput.png")

plt.show()
