"""Plot prefill iteration time vs input length to verify linearity."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
data_file = Path(__file__).parent / "execution_time_results_qwen3_8b.json"
with open(data_file) as f:
    d = json.load(f)

model = d['config']['model']
results = d['results']

# Pure prefill (decode_percentage=0)
prefill = sorted(
    [r for r in results if r['decode_percentage'] == 0],
    key=lambda x: x['total_tokens']
)

L = np.array([r['total_tokens'] for r in prefill])
T = np.array([r['execution_time_ms'] for r in prefill])
T_std = np.array([r['execution_time_std'] for r in prefill])

# Linear fit: T = alpha + beta * L
coeffs = np.polyfit(L, T, 1)
beta, alpha = coeffs
T_fit = alpha + beta * L

# R^2
ss_res = np.sum((T - T_fit) ** 2)
ss_tot = np.sum((T - np.mean(T)) ** 2)
r2 = 1 - ss_res / ss_tot

# Also fit quadratic for comparison
coeffs2 = np.polyfit(L, T, 2)
T_fit2 = np.polyval(coeffs2, L)
ss_res2 = np.sum((T - T_fit2) ** 2)
r2_quad = 1 - ss_res2 / ss_tot

# Plot
fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

# Data points with error bars
ax.errorbar(L, T, yerr=T_std, fmt='o', color='#2196F3', markersize=7,
            capsize=4, linewidth=1.5, label='Measured', zorder=3)

# Linear fit
L_dense = np.linspace(L.min() * 0.9, L.max() * 1.05, 200)
T_fit_dense = alpha + beta * L_dense
ax.plot(L_dense, T_fit_dense, '--', color='#F44336', linewidth=2,
        label=f'Linear fit ($R^2={r2:.4f}$)\n'
              f'$T_p = {alpha:.2f} + {beta:.4f} \\cdot L$ ms')

# Residual annotation
max_residual = np.max(np.abs(T - T_fit))
mean_residual = np.mean(np.abs(T - T_fit))

ax.set_xlabel('Input length $L$ (tokens)', fontsize=12)
ax.set_ylabel('Prefill iteration time (ms)', fontsize=12)
ax.set_title(f'{model} (H200) — Prefill time vs. input length', fontsize=13)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)

# Add text box with fit quality
textstr = (f'Mean residual: {mean_residual:.2f} ms\n'
           f'Max residual: {max_residual:.2f} ms\n'
           f'Quadratic $R^2$: {r2_quad:.4f}')
ax.text(0.98, 0.25, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save
out_dir = Path(__file__).parent / "plots"
out_dir.mkdir(exist_ok=True)
for fmt in ['pdf', 'png']:
    plt.savefig(out_dir / f'prefill_linearity.{fmt}', dpi=200, bbox_inches='tight')
    print(f'Saved: {out_dir / f"prefill_linearity.{fmt}"}')

plt.close()

# Print summary
print(f'\n=== Prefill Linearity Summary ===')
print(f'Model: {model}')
print(f'Linear fit: T_p = {alpha:.2f} + {beta:.4f} * L  (ms)')
print(f'  alpha (fixed overhead) = {alpha:.2f} ms')
print(f'  beta (per-token cost)  = {beta:.4f} ms/token = {beta*1000:.2f} us/token')
print(f'  R^2 = {r2:.6f}')
print(f'  Mean absolute residual = {mean_residual:.2f} ms ({mean_residual/np.mean(T)*100:.1f}% of mean)')
print(f'  Max absolute residual  = {max_residual:.2f} ms')
print(f'Quadratic fit R^2 = {r2_quad:.6f} (improvement: {(r2_quad-r2)*1e4:.1f}e-4)')
