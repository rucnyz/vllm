"""Analyze prefill linearity for both Qwen3-4B and Llama-3.2-1B, generate combined plot + tables."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

def load_data(path):
    with open(path) as f:
        data = json.load(f)
    results = data["results"]
    L = np.array([r["prefill_chunk_size"] for r in results], dtype=float)
    T = np.array([r["mean_time_ms"] for r in results], dtype=float)
    tps = np.array([r["throughput_tokens_per_sec"] for r in results], dtype=float)
    return L, T, tps

def fit_linear(x, y):
    A = np.column_stack([np.ones_like(x), x])
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    y_pred = A @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return coeffs, 1 - ss_res / ss_tot, y_pred

def fit_quadratic(x, y):
    A = np.column_stack([np.ones_like(x), x, x**2])
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    y_pred = A @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return coeffs, 1 - ss_res / ss_tot, y_pred

# Load both datasets
L_q, T_q, tps_q = load_data(ROOT / "prefill_linearity_long.json")
L_l, T_l, tps_l = load_data(ROOT / "prefill_linearity_128k.json")

datasets = [
    ("Qwen3-4B (RTX PRO 6000)", L_q, T_q, tps_q),
    ("Llama-3.2-1B (RTX PRO 6000)", L_l, T_l, tps_l),
]

# --- Print tables ---
for name, L, T, tps in datasets:
    print(f"\n{'='*80}")
    print(f"  {name}")
    print(f"{'='*80}")

    print(f"\n{'L':>8} {'Time (ms)':>12} {'Δslope':>12} {'Tok/s':>12}")
    print("-" * 48)
    for i in range(len(L)):
        slope = ""
        if i > 0:
            slope = f"{(T[i]-T[i-1])/(L[i]-L[i-1]):.5f}"
        print(f"{int(L[i]):>8} {T[i]:>12.2f} {slope:>12} {tps[i]:>12,.0f}")

    print(f"\n  R² Comparison:")
    print(f"  {'Range':>8} {'Linear R²':>12} {'Quadratic R²':>14} {'Δ':>10}")
    print(f"  {'-'*48}")
    thresholds = [4096, 8192, 16384, 32768, 65536, 131072]
    for t in thresholds:
        mask = L <= t
        if mask.sum() < 3:
            continue
        _, r2_lin, _ = fit_linear(L[mask], T[mask])
        _, r2_quad, _ = fit_quadratic(L[mask], T[mask])
        label = f"≤{t//1024}K" if t >= 1024 else f"≤{t}"
        print(f"  {label:>8} {r2_lin:>12.6f} {r2_quad:>14.6f} {r2_quad-r2_lin:>10.6f}")

    # Full range coefficients
    lin_c, r2_lin, _ = fit_linear(L, T)
    quad_c, r2_quad, _ = fit_quadratic(L, T)
    print(f"\n  Linear:    T = {lin_c[0]:.3f} + {lin_c[1]:.6f}·L  (R²={r2_lin:.6f})")
    print(f"  Quadratic: T = {quad_c[0]:.3f} + {quad_c[1]:.6f}·L + {quad_c[2]:.2e}·L²  (R²={r2_quad:.6f})")

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

colors = [('#2196F3', '#1565C0'), ('#F44336', '#C62828')]
markers = ['o', 's']

for idx, (name, L, T, tps) in enumerate(datasets):
    ax = axes[idx]
    lin_c, r2_lin, T_lin = fit_linear(L, T)
    quad_c, r2_quad, T_quad = fit_quadratic(L, T)

    L_dense = np.linspace(0, L.max() * 1.05, 500)
    T_lin_dense = lin_c[0] + lin_c[1] * L_dense
    T_quad_dense = quad_c[0] + quad_c[1] * L_dense + quad_c[2] * L_dense**2

    ax.scatter(L / 1000, T, color='black', zorder=5, s=50, label='Measured')
    ax.plot(L_dense / 1000, T_lin_dense, '--', color=colors[idx][0], linewidth=2,
            label=f'Linear (R²={r2_lin:.4f})')
    ax.plot(L_dense / 1000, T_quad_dense, '-', color=colors[idx][1], linewidth=2,
            label=f'Quadratic (R²={r2_quad:.4f})')

    # Shade typical max_num_batched_tokens region
    ax.axvspan(0, 8.192, alpha=0.08, color='green')
    ax.text(4.0, ax.get_ylim()[0] + 0.05 * (T.max() - T.min()),
            'Typical serving\ntoken budget\n(2K–8K)', fontsize=7, color='green',
            ha='center', va='bottom', alpha=0.8)

    ax.set_xlabel('Prefill Length L (K tokens)', fontsize=11)
    ax.set_ylabel('Iteration Time (ms)', fontsize=11)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = Path(__file__).parent / "prefill_linearity_combined.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {out_path}")
plt.close()
