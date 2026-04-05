"""Generate table image for c=768 and c=1024 results, matching rebuttal format."""
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

data = {
    "c=768": {
        "Prefill-heavy\n(1024:128)": {
            "THETA+": (51809, 2333, 110.8),
            "1P+3D": (22868, 30157, 14.5),
            "2P+2D": (43827, 13986, 23.5),
            "3P+1D": None,
        },
        "Balanced\n(512:512)": {
            "THETA+": (27723, 1232, 47.9),
            "1P+3D": None,
            "2P+2D": None,
            "3P+1D": None,
        },
        "Decode-heavy\n(128:1024)": {
            "THETA+": (21242, 551, 34.8),
            "1P+3D": (18131, 4479, 37.2),
            "2P+2D": (13238, 3416, 54.0),
            "3P+1D": (5981, 15389, 116.5),
        },
    },
    "c=1024": {
        "Prefill-heavy\n(1024:128)": {
            "THETA+": (52426, 3876, 141.1),
            "1P+3D": (23013, 37333, 14.5),
            "2P+2D": (43971, 17961, 23.2),
            "3P+1D": None,
        },
        "Balanced\n(512:512)": {
            "THETA+": (30704, 2062, 47.9),
            "1P+3D": (27127, 16984, 30.4),
            "2P+2D": None,
            "3P+1D": None,
        },
        "Decode-heavy\n(128:1024)": {
            "THETA+": (None, None, None),
            "1P+3D": (19476, None, None),
            "2P+2D": (14068, None, None),
            "3P+1D": None,
        },
    },
}

metrics = ["Thpt", "TTFT", "TPOT"]
schedulers = ["THETA+", "1P+3D", "2P+2D", "3P+1D"]
workloads = ["Prefill-heavy\n(1024:128)", "Balanced\n(512:512)", "Decode-heavy\n(128:1024)"]
concs = ["c=768", "c=1024"]

# Build column headers: two-level (concurrency group + scheduler)
# Row 0: concurrency group spanning 4 cols each
# Row 1: scheduler names
header_row0 = ["", ""] + ["c=768", "", "", "", "c=1024", "", "", ""]
header_row1 = ["", ""] + schedulers + schedulers

# Build data rows, all units in ms for TTFT and TPOT
rows = []
for wl in workloads:
    for mi, metric in enumerate(metrics):
        row = [wl if mi == 0 else "", metric]
        for conc in concs:
            for s in schedulers:
                val = data[conc][wl].get(s)
                if val is None:
                    row.append("−")
                else:
                    tp, ttft, tpot = val
                    if metric == "Thpt":
                        row.append(f"{tp:,}" if tp else "−")
                    elif metric == "TTFT":
                        row.append(f"{ttft:,.0f}" if ttft is not None else "−")
                    elif metric == "TPOT":
                        row.append(f"{tpot:.1f}" if tpot is not None else "−")
        rows.append(row)

n_cols = len(header_row1)
n_rows = len(rows)

fig_height = 0.45 * (n_rows + 2) + 0.8  # tight fit
fig, ax = plt.subplots(figsize=(14, fig_height))
ax.axis('off')

# Create table with header_row1 as column labels
table = ax.table(
    cellText=rows,
    colLabels=header_row1,
    loc='center',
    cellLoc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)

# Style header row (row 0 = colLabels)
for j in range(n_cols):
    cell = table[0, j]
    cell.set_facecolor('#34495e')
    cell.set_text_props(color='white', fontweight='bold', fontsize=9)

# Add concurrency group header as merged text above table
# Get table bbox to position text
table_bbox = table.get_celld()[0, 0].get_bbox()

# Place c=768 and c=1024 labels just above the header
# cols 2-5 = c=768, cols 6-9 = c=1024
for j in range(n_cols):
    cell = table[0, j]
    if 2 <= j <= 5:
        cell.set_facecolor('#2c3e50')
    elif 6 <= j <= 9:
        cell.set_facecolor('#1a3c5e')

# Style data cells
for i in range(n_rows):
    metric = rows[i][1]
    wl_idx = i // 3
    bg = ['#eef2fa', '#fdf5e6', '#eefaee'][wl_idx]

    for j in range(n_cols):
        cell = table[i + 1, j]
        if j < 2:
            cell.set_facecolor('#f5f5f5')
            cell.set_text_props(fontweight='bold', fontsize=9)
        else:
            cell.set_facecolor(bg)

    # Bold best per concurrency group
    for conc_offset in [0, 4]:
        vals = []
        for j in range(4):
            col_idx = 2 + conc_offset + j
            txt = rows[i][col_idx].replace(",", "").replace("−", "")
            if txt:
                try:
                    vals.append((col_idx, float(txt)))
                except ValueError:
                    pass

        if not vals:
            continue

        if metric == "Thpt":
            best_col = max(vals, key=lambda x: x[1])
        else:  # TTFT, TPOT: lower is better
            best_col = min(vals, key=lambda x: x[1])

        table[i + 1, best_col[0]].set_text_props(fontweight='bold', color='#1a5276')

# Add vertical separator between concurrency groups
for i in range(n_rows + 1):
    table[i, 5].set_edgecolor('#2c3e50')
    for side in ['right']:
        table[i, 5].visible_edges = 'open'
    # Just make the border thicker via linewidth
    table[i, 6].set_edgecolor('#2c3e50')

# Concurrency group labels above header
ax.text(0.395, 1.0, 'c = 768', fontsize=11, fontweight='bold',
        ha='center', va='bottom', transform=ax.transAxes, color='#2c3e50')
ax.text(0.735, 1.0, 'c = 1024', fontsize=11, fontweight='bold',
        ha='center', va='bottom', transform=ax.transAxes, color='#1a3c5e')

# Caption below table
ax.text(0.5, -0.02,
        'THETA+ (DP=4) vs Disaggregation on 4×RTX PRO 6000 (Qwen3-8B, 2000 prompts).\n'
        'Throughput in tok/s; TTFT and TPOT in ms. "−" denotes OOM/timeout. Best per metric in bold.',
        fontsize=9, ha='center', va='top', transform=ax.transAxes, color='#555555',
        style='italic')

plt.subplots_adjust(top=0.92, bottom=0.08)
out_path = "/scr/rucnyz/projects/vllm/pd_exp/paper/extended_concurrency_table.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved to: {out_path}")
plt.close()
