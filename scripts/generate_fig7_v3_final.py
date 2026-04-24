"""
Fig 7: Per-layer H_pre Cohen's d across 3 models (MMLU)
v3: v2 기반 수정:
  1. Best layer 텍스트/화살표: 전 모델 검정색, 겹침 해소
  2. 32-layer 모델(Llama/Mistral): 마커 밀도 절반 (markevery=2)
출력: paper/figures/fig7/fig7_v3.png + paper/figures/final/fig7.png
"""
import json
import numpy as np
import matplotlib
from _paths import POT_DIR
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MMLU_DIR = POT_DIR / "experiments" / "31_MMLU_Domain_Extension"
FIG_DIR = PROJECT_ROOT / "paper" / "figures" / "fig7"
FINAL_DIR = PROJECT_ROOT / "paper" / "figures" / "final"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

DPI = 200
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'font.family': 'serif',
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
})

def load_json(p):
    with open(p) as f:
        return json.load(f)

configs = [
    ('Qwen (28L)', 'EXP_20260219_053638_mmlu_qwen/final_results.json', '#d62728', 28),
    ('Llama (32L)', 'EXP_20260219_171237_mmlu_llama/final_results.json', '#1f77b4', 32),
    ('Mistral (32L)', 'EXP_20260220_000610_mmlu_mistral/final_results.json', '#2ca02c', 32),
]

fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True, dpi=DPI)
fig.suptitle('Layerwise H_pre Discriminability and Sign (MMLU, per-layer pipeline)',
             fontsize=11)

all_ds = []
for name, fname, color, n_layers in configs:
    d = load_json(MMLU_DIR / fname)
    la = d['layer_analysis']
    ds = [-la[l]['unnormed_d'] for l in range(n_layers)]
    all_ds.append(ds)

y_min = min(min(ds) for ds in all_ds)
y_max = max(max(ds) for ds in all_ds)
y_pad = (y_max - y_min) * 0.1

for ax, (name, fname, color, n_layers), ds in zip(axes, configs, all_ds):
    layers = list(range(n_layers))
    ax.fill_between(layers, y_min - y_pad, 0, alpha=0.08, color='red', zorder=0)

    # 32-layer 모델은 마커 밀도 절반
    mevery = 2 if n_layers == 32 else 1
    ax.plot(layers, ds, '-o', color=color, linewidth=1.2, markersize=3.5,
            markevery=mevery, zorder=3)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.4, linewidth=0.5)
    ax.set_title(name, fontweight='bold')
    ax.set_xlabel('Layer')
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.grid(alpha=0.1)

    best_l = int(np.argmin(ds))
    best_val = ds[best_l]

    ax.axvline(x=best_l, color='black', linestyle='--', alpha=0.4, linewidth=0.6)
    ax.annotate(
        f'Best L{best_l}',
        xy=(best_l, best_val),
        xytext=(best_l + 4, best_val + 0.22),
        fontsize=8, fontstyle='italic', color='black',
        arrowprops=dict(arrowstyle='->', color='black', lw=0.8),
        zorder=5,
    )

axes[0].set_ylabel("Cohen's d $H_{\\mathrm{pre}}$")

legend_elements = [
    plt.Line2D([0], [0], color='gray', linewidth=0.8, marker='o', markersize=3,
               label="$H_{\\mathrm{pre}}$ Cohen's d (correct - incorrect)"),
    plt.Line2D([0], [0], color='black', linestyle='--', linewidth=0.6, alpha=0.4,
               label="Best layer (most negative)"),
    mpatches.Patch(facecolor='red', alpha=0.08,
               label="Lower $H_{\\mathrm{pre}}$ = correct (negative d)"),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3,
           bbox_to_anchor=(0.5, -0.02), frameon=False)

fig.tight_layout(rect=[0, 0.05, 1, 0.93])
fig.savefig(FIG_DIR / 'fig7_v3.png', dpi=DPI, bbox_inches='tight')
fig.savefig(FINAL_DIR / 'fig7.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("Saved fig7_v3 + final/fig7")
