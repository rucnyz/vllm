#!/usr/bin/env python3
"""Plot benchmark results from results2.json - supports multiple decode counts"""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Load data
with open("results2.json", "r") as f:
    data = json.load(f)

results = data["results"]
config = data["config"]

# Separate results by type
pure_decode = [r for r in results if r["num_prefill"] == 0]
pure_prefill = [r for r in results if r["num_decode"] == 0]
mixed = [r for r in results if r["num_decode"] > 0 and r["num_prefill"] > 0]

# Get unique values
decode_counts = sorted(set(r["num_decode"] for r in pure_decode))
decode_ctx_lens = sorted(set(r["decode_context_len"] for r in pure_decode))
prefill_sizes = sorted(set(r["prefill_chunk_size"] for r in pure_prefill))

print(f"Decode counts: {decode_counts}")
print(f"Decode context lengths: {decode_ctx_lens}")
print(f"Prefill sizes: {prefill_sizes}")

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f"vLLM Benchmark Results - {config['model']}", fontsize=14)

# ============================================
# Plot 1: Pure Decode Time vs num_decode (different context lengths)
# ============================================
ax1 = axes[0, 0]
for ctx_len in decode_ctx_lens:
    subset = sorted([r for r in pure_decode if r["decode_context_len"] == ctx_len],
                    key=lambda x: x["num_decode"])
    x = [r["num_decode"] for r in subset]
    y = [r["mean_time_ms"] for r in subset]
    ax1.plot(x, y, 'o-', label=f'ctx={ctx_len}')

ax1.set_xlabel('Number of Decode Requests')
ax1.set_ylabel('Execution Time (ms)')
ax1.set_title('Pure Decode: Time vs Batch Size')
ax1.legend()
ax1.grid(True, alpha=0.3)

# ============================================
# Plot 2: Pure Decode Throughput vs num_decode
# ============================================
ax2 = axes[0, 1]
for ctx_len in decode_ctx_lens:
    subset = sorted([r for r in pure_decode if r["decode_context_len"] == ctx_len],
                    key=lambda x: x["num_decode"])
    x = [r["num_decode"] for r in subset]
    y = [r["throughput_tokens_per_sec"] for r in subset]
    ax2.plot(x, y, 'o-', label=f'ctx={ctx_len}')

ax2.set_xlabel('Number of Decode Requests')
ax2.set_ylabel('Throughput (tok/s)')
ax2.set_title('Pure Decode: Throughput vs Batch Size')
ax2.legend()
ax2.grid(True, alpha=0.3)

# ============================================
# Plot 3: Pure Prefill Time and Throughput
# ============================================
ax3 = axes[0, 2]
prefill_sorted = sorted(pure_prefill, key=lambda x: x["prefill_chunk_size"])
x = [r["prefill_chunk_size"] for r in prefill_sorted]
y_time = [r["mean_time_ms"] for r in prefill_sorted]
y_tp = [r["throughput_tokens_per_sec"] for r in prefill_sorted]

ax3.bar(x, y_time, width=50, alpha=0.7, label='Time (ms)')
ax3.set_xlabel('Prefill Chunk Size')
ax3.set_ylabel('Execution Time (ms)', color='blue')
ax3.tick_params(axis='y', labelcolor='blue')
ax3.set_title('Pure Prefill Performance')

ax3_right = ax3.twinx()
ax3_right.plot(x, y_tp, 'ro-', label='Throughput')
ax3_right.set_ylabel('Throughput (tok/s)', color='red')
ax3_right.tick_params(axis='y', labelcolor='red')

# ============================================
# Plot 4: Mixed - Overhead vs num_decode (for each prefill size)
# ============================================
ax4 = axes[1, 0]

# Get prefill baselines
prefill_baseline = {r["prefill_chunk_size"]: r["mean_time_ms"] for r in pure_prefill}

# Group mixed by prefill_chunk_size and decode_context_len
for prefill_size in prefill_sizes:
    for ctx_len in decode_ctx_lens:
        subset = sorted([r for r in mixed
                        if r["prefill_chunk_size"] == prefill_size
                        and r["decode_context_len"] == ctx_len],
                       key=lambda x: x["num_decode"])
        if not subset:
            continue
        x = [r["num_decode"] for r in subset]
        overhead = [r["mean_time_ms"] - prefill_baseline[prefill_size] for r in subset]
        ax4.plot(x, overhead, 'o-', label=f'P({prefill_size})+ctx={ctx_len}')

ax4.set_xlabel('Number of Decode Requests')
ax4.set_ylabel('Overhead (ms)')
ax4.set_title('Overhead of Adding Decode to Prefill')
ax4.legend(fontsize=7, ncol=2)
ax4.grid(True, alpha=0.3)

# ============================================
# Plot 5: Mixed Throughput vs num_decode (fixed prefill=1024, varying ctx)
# ============================================
ax5 = axes[1, 1]
prefill_size = 1024  # Focus on largest prefill

for ctx_len in decode_ctx_lens:
    subset = sorted([r for r in mixed
                    if r["prefill_chunk_size"] == prefill_size
                    and r["decode_context_len"] == ctx_len],
                   key=lambda x: x["num_decode"])
    if not subset:
        continue
    x = [r["num_decode"] for r in subset]
    y = [r["throughput_tokens_per_sec"] for r in subset]
    ax5.plot(x, y, 'o-', label=f'ctx={ctx_len}')

# Add pure prefill baseline
baseline_tp = [r["throughput_tokens_per_sec"] for r in pure_prefill
               if r["prefill_chunk_size"] == prefill_size][0]
ax5.axhline(y=baseline_tp, color='gray', linestyle='--', alpha=0.7,
            label=f'Pure P({prefill_size}): {baseline_tp:.0f}')

ax5.set_xlabel('Number of Decode Requests')
ax5.set_ylabel('Throughput (tok/s)')
ax5.set_title(f'Mixed Throughput: xD + 1P({prefill_size})')
ax5.legend()
ax5.grid(True, alpha=0.3)

# ============================================
# Plot 6: Heatmap - Overhead % for fixed ctx_len=1024
# ============================================
ax6 = axes[1, 2]

ctx_len = 1024
overhead_matrix = []
for prefill_size in prefill_sizes:
    row = []
    for num_decode in decode_counts:
        match = [r for r in mixed
                if r["prefill_chunk_size"] == prefill_size
                and r["decode_context_len"] == ctx_len
                and r["num_decode"] == num_decode]
        if match:
            overhead_pct = (match[0]["mean_time_ms"] - prefill_baseline[prefill_size]) / prefill_baseline[prefill_size] * 100
            row.append(overhead_pct)
        else:
            row.append(np.nan)
    overhead_matrix.append(row)

overhead_matrix = np.array(overhead_matrix)
im = ax6.imshow(overhead_matrix, cmap='YlOrRd', aspect='auto')
ax6.set_xticks(range(len(decode_counts)))
ax6.set_xticklabels(decode_counts)
ax6.set_yticks(range(len(prefill_sizes)))
ax6.set_yticklabels(prefill_sizes)
ax6.set_xlabel('Number of Decode Requests')
ax6.set_ylabel('Prefill Chunk Size')
ax6.set_title(f'Overhead % (ctx={ctx_len})')

# Add text annotations
for i in range(len(prefill_sizes)):
    for j in range(len(decode_counts)):
        if not np.isnan(overhead_matrix[i, j]):
            text = ax6.text(j, i, f'{overhead_matrix[i, j]:.1f}%',
                           ha="center", va="center", color="black", fontsize=9)

plt.colorbar(im, ax=ax6, label='Overhead %')

plt.tight_layout()
plt.savefig('benchmark_results2.png', dpi=150, bbox_inches='tight')
print("Saved benchmark_results2.png")

# ============================================
# Additional focused plot: Time breakdown
# ============================================
n_prefill = len(prefill_sizes)
n_cols = min(4, n_prefill)
n_rows = (n_prefill + n_cols - 1) // n_cols
fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
fig2.suptitle(f"Time Breakdown by Prefill Size - {config['model']}", fontsize=14)

# Flatten axes for easy iteration
if n_prefill == 1:
    axes2_flat = [axes2]
else:
    axes2_flat = axes2.flatten() if hasattr(axes2, 'flatten') else [axes2]

for idx, prefill_size in enumerate(prefill_sizes):
    if idx >= len(axes2_flat):
        break
    ax = axes2_flat[idx]

    # Pure prefill baseline
    baseline = prefill_baseline[prefill_size]

    x = np.arange(len(decode_counts))
    width = 0.8 / max(len(decode_ctx_lens), 1)

    for i, ctx_len in enumerate(decode_ctx_lens):
        subset = sorted([r for r in mixed
                        if r["prefill_chunk_size"] == prefill_size
                        and r["decode_context_len"] == ctx_len],
                       key=lambda x: x["num_decode"])
        times = [r["mean_time_ms"] for r in subset]
        if len(times) != len(x):
            continue
        offset = (i - len(decode_ctx_lens)/2 + 0.5) * width
        bars = ax.bar(x + offset, times, width, label=f'ctx={ctx_len}')

    ax.axhline(y=baseline, color='red', linestyle='--', alpha=0.7,
               label=f'Pure prefill: {baseline:.1f}ms')
    ax.set_xlabel('Number of Decode Requests')
    ax.set_ylabel('Execution Time (ms)')
    ax.set_title(f'Prefill Size = {prefill_size}')
    ax.set_xticks(x[::max(1, len(x)//8)])  # Show fewer ticks if many
    ax.set_xticklabels([str(decode_counts[i]) for i in range(0, len(decode_counts), max(1, len(decode_counts)//8))])
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

# Hide unused subplots
for idx in range(n_prefill, len(axes2_flat)):
    axes2_flat[idx].set_visible(False)

plt.tight_layout()
plt.savefig('time_breakdown.png', dpi=150, bbox_inches='tight')
print("Saved time_breakdown.png")

# ============================================
# Linearity Analysis: Time vs (prefill_size + num_decode)
# ============================================
fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
fig3.suptitle(f"Linearity Analysis: Time vs Total Tokens - {config['model']}", fontsize=14)

# Plot 1: Time vs total_tokens for all mixed results
ax = axes3[0]

# Group by decode_context_len
for ctx_len in decode_ctx_lens:
    subset = [r for r in mixed if r["decode_context_len"] == ctx_len]
    x = [r["prefill_chunk_size"] + r["num_decode"] for r in subset]
    y = [r["mean_time_ms"] for r in subset]
    ax.scatter(x, y, alpha=0.7, label=f'ctx={ctx_len}', s=50)

# Add pure prefill points
x_prefill = [r["prefill_chunk_size"] for r in pure_prefill]
y_prefill = [r["mean_time_ms"] for r in pure_prefill]
ax.scatter(x_prefill, y_prefill, marker='s', s=100, color='red', label='Pure Prefill', zorder=5)

# Linear fit for all data
all_x = [r["prefill_chunk_size"] + r["num_decode"] for r in mixed] + x_prefill
all_y = [r["mean_time_ms"] for r in mixed] + y_prefill
coeffs = np.polyfit(all_x, all_y, 1)
fit_x = np.linspace(min(all_x), max(all_x), 100)
fit_y = np.polyval(coeffs, fit_x)
ax.plot(fit_x, fit_y, 'k--', alpha=0.7, label=f'Linear fit: y={coeffs[0]:.4f}x+{coeffs[1]:.2f}')

# Calculate R²
y_pred = np.polyval(coeffs, all_x)
ss_res = np.sum((np.array(all_y) - y_pred) ** 2)
ss_tot = np.sum((np.array(all_y) - np.mean(all_y)) ** 2)
r2 = 1 - ss_res / ss_tot
ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

ax.set_xlabel('Total Tokens (prefill_size + num_decode)')
ax.set_ylabel('Execution Time (ms)')
ax.set_title('All Mixed Results')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2: Residuals from linear fit
ax2 = axes3[1]
residuals = np.array(all_y) - y_pred
colors = ['blue'] * len(mixed) + ['red'] * len(pure_prefill)
ax2.scatter(all_x, residuals, c=colors, alpha=0.7)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax2.set_xlabel('Total Tokens (prefill_size + num_decode)')
ax2.set_ylabel('Residual (ms)')
ax2.set_title('Residuals from Linear Fit')
ax2.grid(True, alpha=0.3)

# Plot 3: Separate analysis - is it additive?
# Model: time = a * prefill_size + b * num_decode + c
ax3 = axes3[2]

# Multiple linear regression
from numpy.linalg import lstsq

# Build design matrix [prefill_size, num_decode, 1]
X = []
Y = []
for r in mixed:
    X.append([r["prefill_chunk_size"], r["num_decode"], r["decode_context_len"], 1])
    Y.append(r["mean_time_ms"])

X = np.array(X)
Y = np.array(Y)

# Solve least squares
coeffs_multi, residuals_multi, rank, s = lstsq(X, Y, rcond=None)
a_prefill, a_decode, a_ctx, c = coeffs_multi

# Predict
Y_pred = X @ coeffs_multi
ss_res = np.sum((Y - Y_pred) ** 2)
ss_tot = np.sum((Y - np.mean(Y)) ** 2)
r2_multi = 1 - ss_res / ss_tot

# Plot predicted vs actual
ax3.scatter(Y, Y_pred, alpha=0.7)
ax3.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--', label='Perfect fit')
ax3.set_xlabel('Actual Time (ms)')
ax3.set_ylabel('Predicted Time (ms)')
ax3.set_title('Multi-Linear Regression\ntime = a·prefill + b·decode + c·ctx + d')
ax3.text(0.05, 0.95,
         f'a_prefill = {a_prefill:.5f} ms/token\n'
         f'a_decode = {a_decode:.4f} ms/req\n'
         f'a_ctx = {a_ctx:.5f} ms/token\n'
         f'intercept = {c:.2f} ms\n'
         f'R² = {r2_multi:.4f}',
         transform=ax3.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linearity_analysis.png', dpi=150, bbox_inches='tight')
print("Saved linearity_analysis.png")

# ============================================
# Additional: Time vs prefill_size (fixed num_decode) line plot
# ============================================
fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
fig4.suptitle("Time vs Prefill Size (Fixed Decode Count)", fontsize=14)

# Left: Time vs prefill_size for different num_decode
ax = axes4[0]
for num_decode in decode_counts:
    for ctx_len in decode_ctx_lens:
        subset = sorted([r for r in mixed
                        if r["num_decode"] == num_decode
                        and r["decode_context_len"] == ctx_len],
                       key=lambda x: x["prefill_chunk_size"])
        if len(subset) < 2:
            continue
        x = [r["prefill_chunk_size"] for r in subset]
        y = [r["mean_time_ms"] for r in subset]
        ax.plot(x, y, 'o-', label=f'{num_decode}D, ctx={ctx_len}')

# Add pure prefill
x_p = [r["prefill_chunk_size"] for r in sorted(pure_prefill, key=lambda x: x["prefill_chunk_size"])]
y_p = [r["mean_time_ms"] for r in sorted(pure_prefill, key=lambda x: x["prefill_chunk_size"])]
ax.plot(x_p, y_p, 's--', color='black', linewidth=2, markersize=10, label='Pure Prefill')

ax.set_xlabel('Prefill Size')
ax.set_ylabel('Execution Time (ms)')
ax.set_title('Time vs Prefill Size')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# Right: Time vs num_decode for different prefill_size
ax = axes4[1]
for prefill_size in prefill_sizes:
    for ctx_len in decode_ctx_lens:
        subset = sorted([r for r in mixed
                        if r["prefill_chunk_size"] == prefill_size
                        and r["decode_context_len"] == ctx_len],
                       key=lambda x: x["num_decode"])
        if len(subset) < 2:
            continue
        x = [r["num_decode"] for r in subset]
        y = [r["mean_time_ms"] for r in subset]
        ax.plot(x, y, 'o-', label=f'P({prefill_size}), ctx={ctx_len}')

ax.set_xlabel('Number of Decode Requests')
ax.set_ylabel('Execution Time (ms)')
ax.set_title('Time vs Decode Count')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('time_vs_components.png', dpi=150, bbox_inches='tight')
print("Saved time_vs_components.png")

print("\n" + "="*60)
print("LINEAR REGRESSION SUMMARY")
print("="*60)
print(f"\nSimple Linear (time ~ total_tokens):")
print(f"  time = {coeffs[0]:.5f} * total_tokens + {coeffs[1]:.2f}")
print(f"  R² = {r2:.4f}")

print(f"\nMultiple Linear (time ~ prefill + decode + ctx):")
print(f"  time = {a_prefill:.5f} * prefill_size + {a_decode:.4f} * num_decode + {a_ctx:.5f} * ctx_len + {c:.2f}")
print(f"  R² = {r2_multi:.4f}")

print(f"\nInterpretation:")
print(f"  - Each prefill token adds ~{a_prefill*1000:.3f} µs")
print(f"  - Each decode request adds ~{a_decode:.3f} ms")
print(f"  - Each context token adds ~{a_ctx*1000:.3f} µs")
print("="*60)

plt.show()
