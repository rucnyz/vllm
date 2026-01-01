#!/usr/bin/env python3
"""Plot only the linearity analysis from benchmark results"""

import json
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import lstsq

# Load data
with open("results2.json", "r") as f:
    data = json.load(f)

results = data["results"]
config = data["config"]

# Separate results by type
pure_prefill = [r for r in results if r["num_decode"] == 0]
mixed = [r for r in results if r["num_decode"] > 0 and r["num_prefill"] > 0]

# Get unique values
decode_ctx_lens = sorted(set(r["decode_context_len"] for r in mixed))

# ============================================
# Linearity Analysis: Time vs (prefill_size + num_decode)
# ============================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f"Linearity Analysis: Time vs Total Tokens - {config['model']}", fontsize=14)

# Plot 1: Time vs total_tokens for all mixed results
ax = axes[0]

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
ax2 = axes[1]
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
ax3 = axes[2]

# Build design matrix [prefill_size, num_decode, decode_context_len, 1]
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