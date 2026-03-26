"""
논문 Figure 1~6 생성 스크립트 v3
=================================
v2 대비 변경사항 (GPT 16차 4세션 만장일치 기반):
  - [FIX] Figure 1(a): 제목 수정 "Decoded entropy before vs. after RMSNorm"
  - [FIX] Figure 2(a): 제목 수정 "Scalar Signals Relative to Output Entropy"
  - [FIX] Figure 2(b): h_norm + output max-prob 추가 (cherry-picking 방지)
  - [FIX] Figure 3(b)(c): alpha tick 직접 표시
  - [FIX] Figure 3(d): outlier-only 라벨
  - [FIX] Figure 6(b): delta annotation 위치를 gray→green bracket 위로 이동
  - [FIX] Figure 6(b): 제목 "Classifier Choice Changes AUROC"로 중립화
  - [FIX] Figure 6(a): 범례를 데이터 비는 곳으로 이동

원본: scripts/generate_figures.py (보존)
출력: paper/figures_v2/fig{N}/ 디렉토리별 분리
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BASE_DIR = PROJECT_ROOT / "PoT_Experiment_Entropy_Attention_Extraction_Experiment"
FIG_BASE = PROJECT_ROOT / "paper" / "figures"

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_json(path):
    with open(path) as f:
        return json.load(f)


def ensure_dir(fig_num):
    d = FIG_BASE / f"fig{fig_num}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ============================================================
# Figure 1: Conceptual Overview
# v2 FIX: Pre-norm/Post-norm 라벨 위치 조정, 박스 간격 확대
# ============================================================
def generate_figure1():
    print("Generating Figure 1 v2: Conceptual Overview...")
    fig_dir = ensure_dir(1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5),
                             gridspec_kw={'width_ratios': [1.2, 0.8, 1.0]})

    # Panel A: H_pre / H_post computation path
    ax = axes[0]
    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-0.5, 6.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(a) Decoded Entropy Before vs. After RMSNorm', fontweight='bold')

    # -- Top row (Pre-norm path) --
    boxes_top = [
        (2.5, 4.5, '$h_\\ell$', '#E8F4FD'),
        (5.5, 4.5, '$W h_\\ell$', '#FFF3E0'),
        (8.5, 4.5, '$H_{\\mathrm{pre}}$', '#FFCDD2'),
    ]
    for x, y, txt, color in boxes_top:
        ax.add_patch(plt.Rectangle((x-0.9, y-0.4), 1.8, 0.8,
                     facecolor=color, edgecolor='black', linewidth=1.2, zorder=2))
        ax.text(x, y, txt, ha='center', va='center', fontsize=11, zorder=3)

    ax.annotate('', xy=(4.6, 4.5), xytext=(3.4, 4.5),
                arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.text(4.0, 4.9, 'lm_head', ha='center', fontsize=8, style='italic')
    ax.annotate('', xy=(7.6, 4.5), xytext=(6.4, 4.5),
                arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.text(7.0, 4.9, 'softmax $\\rightarrow$ H', ha='center', fontsize=8, style='italic')

    # Pre-norm label (왼쪽 여백에 배치, 겹침 해소)
    ax.text(-0.3, 4.5, 'Pre-norm\npath', ha='left', va='center', fontsize=8,
            color='#D32F2F', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFEBEE', alpha=0.8))

    # -- Bottom row (Post-norm path) --
    boxes_bot = [
        (2.5, 2.0, '$h_\\ell$', '#E8F4FD'),
        (5.5, 2.0, '$W \\cdot N_\\epsilon(h_\\ell)$', '#E8F5E9'),
        (8.5, 2.0, '$H_{\\mathrm{post}}$', '#C8E6C9'),
    ]
    for x, y, txt, color in boxes_bot:
        ax.add_patch(plt.Rectangle((x-0.9, y-0.4), 1.8, 0.8,
                     facecolor=color, edgecolor='black', linewidth=1.2, zorder=2))
        ax.text(x, y, txt, ha='center', va='center', fontsize=11, zorder=3)

    ax.annotate('', xy=(4.6, 2.0), xytext=(3.4, 2.0),
                arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.text(4.0, 2.4, 'RMSNorm$\\rightarrow$lm_head', ha='center', fontsize=8, style='italic')
    ax.annotate('', xy=(7.6, 2.0), xytext=(6.4, 2.0),
                arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.text(7.0, 2.4, 'softmax $\\rightarrow$ H', ha='center', fontsize=8, style='italic')

    # Post-norm label
    ax.text(-0.3, 2.0, 'Post-norm\npath', ha='left', va='center', fontsize=8,
            color='#388E3C', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8F5E9', alpha=0.8))

    # Panel B: h = r*u decomposition
    ax = axes[1]
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title('(b) Scale-Direction Decomposition', fontweight='bold')

    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=0.8)

    angle = np.pi/5
    r = 1.2
    # h vector
    ax.annotate('', xy=(r*np.cos(angle), r*np.sin(angle)), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='#1565C0'))
    ax.text(r*np.cos(angle)+0.08, r*np.sin(angle)+0.12,
            '$h_\\ell = r_\\ell \\cdot u_\\ell$', fontsize=11, color='#1565C0')

    # u vector (위치 조정: 아래쪽으로)
    ax.annotate('', xy=(np.cos(angle), np.sin(angle)), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#E65100'))
    ax.text(np.cos(angle)+0.1, np.sin(angle)-0.15,
            '$u_\\ell$', fontsize=11, color='#E65100')

    # r annotation (겹침 방지: 벡터 바깥쪽으로)
    mid_x = r*np.cos(angle)*0.35
    mid_y = r*np.sin(angle)*0.35
    ax.text(mid_x - 0.25, mid_y + 0.2,
            '$r_\\ell = \\|h_\\ell\\|$', fontsize=9, color='#D32F2F')

    ax.text(0, -1.3, '$z_{\\mathrm{pre}} = W(r_\\ell u_\\ell) = r_\\ell W u_\\ell$\n'
            '$r_\\ell$ acts as inverse temperature',
            ha='center', fontsize=8, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', alpha=0.8))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Panel C: Token positions
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('(c) Token Positions', fontweight='bold')

    positions = [
        (2, 3.5, 'Step 0\n(prompt-last)', '#BBDEFB', 'Before generation'),
        (5, 3.5, 'Step 1\n(first-gen)', '#C8E6C9', 'First generated token'),
        (8, 3.5, 'Full Avg\n(all gen)', '#FFF9C4', 'Mean over generation'),
    ]
    for x, y, label, color, desc in positions:
        ax.add_patch(plt.Rectangle((x-1.2, y-0.6), 2.4, 1.2,
                     facecolor=color, edgecolor='black', linewidth=1, zorder=2))
        ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold', zorder=3)
        ax.text(x, y-1.0, desc, ha='center', va='center', fontsize=7, style='italic')

    ax.annotate('', xy=(3.8, 3.5), xytext=(3.2, 3.5),
                arrowprops=dict(arrowstyle='->', lw=1.2))
    ax.annotate('', xy=(6.8, 3.5), xytext=(6.2, 3.5),
                arrowprops=dict(arrowstyle='->', lw=1.2))

    ax.text(5, 1.2, 'Generation sequence $\\rightarrow$', ha='center', fontsize=9,
            bbox=dict(boxstyle='rarrow,pad=0.3', facecolor='#E0E0E0'))

    plt.tight_layout()
    path = fig_dir / "fig1_v3.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Figure 2: Baseline Comparison
# v2 FIX: 마커 겹침 완화, 범례 위치 조정
# ============================================================
def generate_figure2():
    print("Generating Figure 2 v2: Baseline Comparison...")
    fig_dir = ensure_dir(2)

    summary = load_json(BASE_DIR / "experiments" / "32_Phase0_CalTest_Baselines" / "summary_table.json")

    methods = [
        ('output_entropy', 'Output entropy'),
        ('output_logit_max', 'Output max-prob'),
        ('output_margin', 'Output margin'),
        ('length_only', 'Length-only'),
        ('h_norm_best_layer', 'h_norm'),
        ('logit_std_best_layer', 'logit_std'),
        ('normed_entropy_best_layer', 'H_post'),
        ('unnormed_entropy_best_layer', 'H_pre'),
    ]

    conditions = [
        ('qwen_hard', 'Qwen Hard'),
        ('mmlu_qwen', 'Qwen MMLU'),
        ('mmlu_llama', 'Llama MMLU'),
        ('mmlu_mistral', 'Mistral MMLU'),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                    gridspec_kw={'width_ratios': [1.2, 0.8]})

    # Panel A: Dot plot
    y_positions = np.arange(len(methods))
    markers = ['o', 's', '^', 'D']
    cond_colors = ['#1976D2', '#FF8F00', '#2E7D32', '#C62828']

    for ci, (cond_key, cond_label) in enumerate(conditions):
        cond_data = summary['conditions'].get(cond_key, {}).get('baselines', {})
        output_ent = cond_data.get('output_entropy', {}).get('test_auroc', 0.5)

        aurocs = []
        for mk, ml in methods:
            auroc = cond_data.get(mk, {}).get('test_auroc', 0.5)
            aurocs.append(auroc - output_ent)

        x_offset = (ci - 1.5) * 0.22  # v2: offset 확대 0.15 -> 0.22
        ax1.scatter(aurocs, y_positions + x_offset, s=60, alpha=0.85,  # v2: 크기 40->60
                   label=cond_label, zorder=3,
                   marker=markers[ci], color=cond_colors[ci],
                   edgecolors='black', linewidth=0.3)

    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.3, linewidth=0.8)
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels([m[1] for m in methods])
    ax1.set_xlabel('$\\Delta$AUROC relative to output entropy')
    ax1.set_title('(a) Scalar Signals Relative to Output Entropy', fontweight='bold')
    ax1.legend(loc='lower right', fontsize=7, framealpha=0.9)
    ax1.grid(axis='x', alpha=0.2)
    ax1.invert_yaxis()

    # Panel B: Compute vs AUROC scatter (하드코딩 값 유지 + 출처 주석)
    # 출처: 32_Phase0_CalTest_Baselines summary_table.json (qwen_hard)
    # + 38_Phase4b_Fair_SC (agreement AUROC)
    # v3: h_norm + output max-prob 추가 (cherry-picking 방지)
    compute_data = {
        'Output ent.': (1, 0.6316),
        'Output max-p': (1, 0.8041),
        'H_post': (1, 0.6613),
        'H_pre': (1, 0.7672),
        'h_norm': (1, 0.8256),
        'logit_std': (1, 0.8086),
        'Fair SC': (5, 0.7905),
    }

    markers_map = {'Output ent.': 'o', 'Output max-p': 'h', 'H_post': 's', 'H_pre': '^',
                   'h_norm': 'v', 'logit_std': 'D', 'Fair SC': 'P'}
    colors_map = {'Output ent.': '#9E9E9E', 'Output max-p': '#757575', 'H_post': '#4CAF50',
                  'H_pre': '#2196F3', 'h_norm': '#FF9800', 'logit_std': '#FF5722', 'Fair SC': '#9C27B0'}

    for method, (passes, auroc) in compute_data.items():
        ax2.scatter(passes, auroc, s=140, marker=markers_map[method],
                   color=colors_map[method], label=method, zorder=3,
                   edgecolors='black', linewidth=0.5)
        # v2: 레이블 위치를 method별로 조정하여 겹침 방지
        y_off = 8 if method != 'H_pre' else -15
        ax2.annotate(f'{auroc:.3f}', (passes, auroc),
                    textcoords="offset points", xytext=(10, y_off), fontsize=8)

    ax2.set_xlabel('Forward Passes')
    ax2.set_ylabel('Test AUROC')
    ax2.set_title('(b) Compute vs. Discrimination (Qwen Hard)', fontweight='bold')
    ax2.set_xticks([1, 2, 3, 4, 5])
    ax2.set_xlim(0.5, 5.5)
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    path = fig_dir / "fig2_v3.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Figure 3: Scale Intervention (4-panel)
# v2 FIX: Panel (a) y축 주석, Panel (b) 범례 위치
# ============================================================
def generate_figure3():
    print("Generating Figure 3 v2: Scale Intervention...")
    fig_dir = ensure_dir(3)

    qwen_cm = load_json(BASE_DIR / "experiments" / "39_Phase3_Cross_Model_Scale_Intervention" / "qwen" / "intervention_analysis.json")
    llama_cm = load_json(BASE_DIR / "experiments" / "39_Phase3_Cross_Model_Scale_Intervention" / "llama" / "intervention_analysis.json")
    mistral_cm = load_json(BASE_DIR / "experiments" / "39_Phase3_Cross_Model_Scale_Intervention" / "mistral" / "intervention_analysis.json")
    qwen_orig = load_json(BASE_DIR / "experiments" / "36_Phase3_Scale_Intervention" / "intervention_analysis.json")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: Unit-norm collapse bars
    ax = axes[0, 0]
    models = ['Qwen\n(MMLU 300)', 'Llama\n(MMLU 300)', 'Mistral\n(MMLU 300)']
    unit_means = [qwen_cm['h_pre_unit_mean_all_layers'],
                  llama_cm['h_pre_unit_mean_all_layers'],
                  mistral_cm['h_pre_unit_mean_all_layers']]

    bars = ax.bar(models, unit_means, color=['#2196F3', '#FF9800', '#4CAF50'],
                  edgecolor='black', linewidth=0.8)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Max entropy (1.0)')
    ax.set_ylim(0.9995, 1.00005)
    ax.set_ylabel('Mean H_pre after unit-norm')
    ax.set_title('(a) Unit-Norm Collapse: H_pre $\\rightarrow$ 1.0', fontweight='bold')
    for bar, val in zip(bars, unit_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.00015,
                f'{val:.6f}', ha='center', va='top', fontsize=8, fontweight='bold')
    ax.legend(fontsize=8)
    # v2: y축 스케일 주석
    ax.text(0.98, 0.02, 'Note: y-axis starts at 0.9995\n(all values > 0.9999)',
            transform=ax.transAxes, fontsize=6, ha='right', va='bottom',
            style='italic', color='gray')

    # Panel B: Qwen alpha-sweep
    ax = axes[0, 1]
    alphas = qwen_orig['alphas']

    l4_pre = [qwen_orig['alpha_summary'][str(a)]['L4_h_pre'] for a in alphas]
    l4_post = [qwen_orig['alpha_summary'][str(a)]['L4_h_post'] for a in alphas]
    l16_pre = [qwen_orig['alpha_summary'][str(a)]['L16_h_pre'] for a in alphas]
    l16_post = [qwen_orig['alpha_summary'][str(a)]['L16_h_post'] for a in alphas]

    ax.plot(alphas, l16_pre, 'o-', color='#D32F2F', label='L16 H_pre', linewidth=2, markersize=5)
    ax.plot(alphas, l16_post, 's--', color='#388E3C', label='L16 H_post', linewidth=2, markersize=5)
    ax.plot(alphas, l4_pre, '^-', color='#E57373', label='L4 H_pre', linewidth=1.5, markersize=4)
    ax.plot(alphas, l4_post, 'v--', color='#81C784', label='L4 H_post', linewidth=1.5, markersize=4)

    ax.set_xlabel('Alpha (scale factor)')
    ax.set_ylabel('Normalized Entropy')
    ax.set_title('(b) Qwen Alpha-Sweep (Math Hard 500)', fontweight='bold')
    ax.legend(fontsize=7, loc='right')
    ax.set_xscale('log')
    ax.set_xticks([0.25, 0.5, 1.0, 2.0, 4.0])
    ax.set_xticklabels(['0.25', '0.5', '1', '2', '4'])
    ax.grid(alpha=0.2)

    # Panel C: Llama/Mistral alpha-sweep
    ax = axes[1, 0]
    llama_alphas = llama_cm['alphas']

    llama_l24_pre = [llama_cm['alpha_summary'][str(a)].get('L24_h_pre', np.nan) for a in llama_alphas]
    llama_l24_post = [llama_cm['alpha_summary'][str(a)].get('L24_h_post', np.nan) for a in llama_alphas]
    mistral_l24_pre = [mistral_cm['alpha_summary'][str(a)].get('L24_h_pre', np.nan) for a in llama_alphas]
    mistral_l24_post = [mistral_cm['alpha_summary'][str(a)].get('L24_h_post', np.nan) for a in llama_alphas]

    ax.plot(llama_alphas, llama_l24_pre, 'o-', color='#E65100', label='Llama L24 H_pre', linewidth=2)
    ax.plot(llama_alphas, llama_l24_post, 's--', color='#FF9800', label='Llama L24 H_post', linewidth=2)
    ax.plot(llama_alphas, mistral_l24_pre, '^-', color='#1B5E20', label='Mistral L24 H_pre', linewidth=2)
    ax.plot(llama_alphas, mistral_l24_post, 'v--', color='#4CAF50', label='Mistral L24 H_post', linewidth=2)

    ax.set_xlabel('Alpha (scale factor)')
    ax.set_ylabel('Normalized Entropy')
    ax.set_title('(c) Cross-Model Alpha-Sweep (MMLU 300, L24)', fontweight='bold')
    ax.legend(fontsize=7)
    ax.set_xscale('log')
    ax.set_xticks([0.25, 0.5, 1.0, 2.0, 4.0])
    ax.set_xticklabels(['0.25', '0.5', '1', '2', '4'])
    ax.grid(alpha=0.2)

    # Panel D: H_post variation vs h_norm scatter
    ax = axes[1, 1]

    for model_data, model_name, color, marker in [
        (qwen_cm, 'Qwen', '#2196F3', 'o'),
        (llama_cm, 'Llama', '#FF9800', 's'),
        (mistral_cm, 'Mistral', '#4CAF50', '^'),
    ]:
        h_norms = []
        h_post_vars = []
        layers_plot = []
        for l_str, data in model_data['unit_norm_summary'].items():
            l = int(l_str)
            h_norms.append(data['h_norm_mean'])
            var = model_data['h_post_max_variation'].get(l_str, 0)
            h_post_vars.append(var)
            layers_plot.append(l)

        ax.scatter(h_norms, h_post_vars, s=25, alpha=0.7, color=color,
                  marker=marker, label=model_name, zorder=3)

        # v3: outlier-only labels (variation > 0.05 only)
        for i, (hn, hv, ll) in enumerate(zip(h_norms, h_post_vars, layers_plot)):
            if hv > 0.05:
                ax.annotate(f'L{ll}', (hn, hv), textcoords="offset points",
                           xytext=(5, 5), fontsize=7, color=color, fontweight='bold')

    ax.axhline(y=0.003, color='gray', linestyle=':', alpha=0.5, label='Practical invariance (0.003)')
    ax.set_xlabel('Mean $\\|h\\|$ at layer')
    ax.set_ylabel('Max H_post alpha-variation')
    ax.set_title('(d) H_post Variation vs Hidden-State Norm', fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.2)

    plt.tight_layout()
    path = fig_dir / "fig3_v3.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Figure 4: Token Position
# v2 FIX: JSON 키 경로 수정 (치명적 버그)
# 원래: data.get(f"{metric}_{pos}", {}).get('test_auroc', 0.5)
# 수정: data['positions'][pos]['baselines'][metric]['test_auroc']
# ============================================================
def generate_figure4():
    print("Generating Figure 4 v2: Token Position (KEY FIX)...")
    fig_dir = ensure_dir(4)

    hard_data = load_json(BASE_DIR / "experiments" / "33_Phase1_Token_Position" / "phase1_position_analysis.json")
    mmlu_data = load_json(BASE_DIR / "experiments" / "33_Phase1_Token_Position" / "phase1_mmlu_position_analysis.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    positions = ['step0_prompt_last', 'step1_first_gen', 'full_gen_avg']
    pos_labels = ['Step 0\n(prompt-last)', 'Step 1\n(first-gen)', 'Full Avg']
    x = np.arange(len(positions))
    width = 0.18

    metrics = ['unnormed_entropy', 'normed_entropy', 'logit_std', 'h_norm']
    labels = ['H_pre', 'H_post', 'logit_std', 'h_norm']
    colors = ['#2196F3', '#4CAF50', '#FF5722', '#FF9800']

    for ax, data, title in [(ax1, hard_data, '(a) Qwen Hard (500 samples)'),
                             (ax2, mmlu_data, '(b) Qwen MMLU (1000 samples)')]:
        for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
            aurocs = []
            for pos in positions:
                # v2 FIX: 올바른 중첩 경로로 접근
                try:
                    val = data['positions'][pos]['baselines'][metric]['test_auroc']
                except KeyError:
                    val = 0.5
                    print(f"  WARNING: Missing key positions/{pos}/baselines/{metric}")
                aurocs.append(val)
            bars = ax.bar(x + (i - 1.5) * width, aurocs, width,
                         label=label, color=color, alpha=0.85,
                         edgecolor='black', linewidth=0.3)

            # v2: bar 위에 값 표시
            for bar, val in zip(bars, aurocs):
                if val > 0.52:  # chance level 이상만 표시
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=6, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels(pos_labels)
        ax.set_ylabel('Test AUROC')
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.2)
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)
        ax.set_ylim(0.45, 0.85)

    plt.tight_layout()
    path = fig_dir / "fig4_v3.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Figure 5: Layerwise Profile + Saturation
# v2 FIX: Llama/Mistral 겹침 -> 약간의 y-offset 시각화
# ============================================================
def generate_figure5():
    print("Generating Figure 5 v2: Layerwise Profile & Saturation...")
    fig_dir = ensure_dir(5)

    qwen_cm = load_json(BASE_DIR / "experiments" / "39_Phase3_Cross_Model_Scale_Intervention" / "qwen" / "intervention_analysis.json")
    llama_cm = load_json(BASE_DIR / "experiments" / "39_Phase3_Cross_Model_Scale_Intervention" / "llama" / "intervention_analysis.json")
    mistral_cm = load_json(BASE_DIR / "experiments" / "39_Phase3_Cross_Model_Scale_Intervention" / "mistral" / "intervention_analysis.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: H_pre layer profile
    for model_data, model_name, color, n_layers, lw, ls in [
        (qwen_cm, 'Qwen (28L)', '#2196F3', 28, 2.5, '-'),
        (llama_cm, 'Llama (32L)', '#FF9800', 32, 2.0, '-'),
        (mistral_cm, 'Mistral (32L)', '#4CAF50', 32, 2.0, '--'),  # v2: Mistral을 dashed로
    ]:
        layers = list(range(n_layers))
        h_pre_orig = [model_data['unit_norm_summary'][str(l)]['h_pre_orig_mean'] for l in layers]
        norm_layers = [l / (n_layers - 1) for l in layers]
        ax1.plot(norm_layers, h_pre_orig, ls, color=color, label=model_name,
                linewidth=lw, alpha=0.85, marker='o', markersize=2)

    ax1.axhline(y=0.99, color='red', linestyle=':', alpha=0.4, label='Ceiling (0.99)')
    ax1.set_xlabel('Relative Layer Position (0=first, 1=last)')
    ax1.set_ylabel('Mean H_pre (original)')
    ax1.set_title('(a) H_pre Layer Profiles Across Models', fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.2)
    ax1.set_ylim(-0.05, 1.05)

    # Panel B: Original H_pre vs alpha-sweep range
    for model_data, model_name, color, marker, n_layers in [
        (qwen_cm, 'Qwen', '#2196F3', 'o', 28),
        (llama_cm, 'Llama', '#FF9800', 's', 32),
        (mistral_cm, 'Mistral', '#4CAF50', '^', 32),
    ]:
        h_pre_origs = []
        sweep_ranges = []

        for l in range(n_layers):
            lk = str(l)
            orig = model_data['unit_norm_summary'][lk]['h_pre_orig_mean']

            pre_vals = []
            for alpha in model_data['alphas']:
                ak = str(alpha)
                val = model_data['alpha_summary'].get(ak, {}).get(f'L{l}_h_pre')
                if val is not None:
                    pre_vals.append(val)

            sweep_range = max(pre_vals) - min(pre_vals) if len(pre_vals) >= 2 else 0
            h_pre_origs.append(orig)
            sweep_ranges.append(sweep_range)

        ax2.scatter(h_pre_origs, sweep_ranges, s=30, alpha=0.7, color=color,
                   marker=marker, label=model_name, zorder=3)

    ax2.set_xlabel('Original H_pre (mean)')
    ax2.set_ylabel('Observable Alpha-Sweep Range')
    ax2.set_title('(b) Saturation Limits Observable Sensitivity', fontweight='bold')
    ax2.legend(fontsize=8, loc='upper right')  # v3: upper right (데이터 비는 영역)
    ax2.grid(alpha=0.2)

    # v3: annotation - 화살표로 ceiling 영역(x~1.0, y~0.0) 가리키되,
    # 텍스트는 데이터 없는 중앙 영역에 배치
    ax2.annotate('Ceiling region:\nhigh H_pre,\nlow observable range',
                xy=(0.85, 0.05),  # 화살표 끝: ceiling 영역
                xytext=(0.45, 0.55),  # 텍스트: 데이터 없는 중앙
                fontsize=7, style='italic', ha='center',
                bbox=dict(boxstyle='round', facecolor='#FFCDD2', alpha=0.5),
                arrowprops=dict(arrowstyle='->', lw=0.8, color='gray'))

    plt.tight_layout()
    path = fig_dir / "fig5_v3.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Figure 6: Lens Dependence
# v2 FIX: Panel (b) JSON 키 이름 수정 (치명적 버그)
#   el_original_knn_auroc -> EL_original_knn_k3
#   el_matched_lr_auroc   -> EL_matched_LR
#   hpre_profile_lr_auroc -> H_pre_profile_LR
# v2 FIX: Panel (a) 범례 위치 -> upper left
# ============================================================
def generate_figure6():
    print("Generating Figure 6 v2: Lens Dependence (KEY FIX)...")
    fig_dir = ensure_dir(6)

    tl_data = load_json(BASE_DIR / "experiments" / "35_Phase2b_Tuned_Lens" / "eval_faithfulness.json")
    el_summary = load_json(BASE_DIR / "experiments" / "34_Phase2_Entropy_Lens_Baseline" / "el_baseline_summary.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: TL vs LL faithfulness
    if 'per_layer' in tl_data:
        layers_tl = sorted([int(k) for k in tl_data['per_layer'].keys()])
        kl_tl = [tl_data['per_layer'][str(l)].get('tl_kl', np.nan) for l in layers_tl]
        kl_ll = [tl_data['per_layer'][str(l)].get('ll_kl', np.nan) for l in layers_tl]
        top1_tl = [tl_data['per_layer'][str(l)].get('tl_top1', np.nan) for l in layers_tl]
        top1_ll = [tl_data['per_layer'][str(l)].get('ll_top1', np.nan) for l in layers_tl]
    else:
        layers_tl = [0, 4, 8, 12, 16, 20, 24, 27]
        kl_tl = [6.856, 6.568, 6.590, 6.299, 6.112, 5.578, 3.765, 7.562]
        kl_ll = [10.757, 11.318, 12.563, 13.506, 15.499, 18.903, 21.001, 0.000]
        top1_tl = [0.131, 0.195, 0.226, 0.237, 0.241, 0.305, 0.467, 0.312]
        top1_ll = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    ax1_twin = ax1.twinx()
    l1, = ax1.plot(layers_tl, kl_tl, 'o-', color='#2196F3', label='TL KL', linewidth=2, markersize=4)
    l2, = ax1.plot(layers_tl, kl_ll, 's-', color='#F44336', label='LL KL', linewidth=2, markersize=4)
    l3, = ax1_twin.plot(layers_tl, [t*100 if t <= 1 else t for t in top1_tl],
                        '^--', color='#1565C0', label='TL Top-1%', linewidth=1.5, markersize=4, alpha=0.7)
    l4, = ax1_twin.plot(layers_tl, [t*100 if t <= 1 else t for t in top1_ll],
                        'v--', color='#C62828', label='LL Top-1%', linewidth=1.5, markersize=4, alpha=0.7)

    ax1.set_xlabel('Layer')
    ax1.set_ylabel('KL Divergence')
    ax1_twin.set_ylabel('Top-1 Agreement (%)')
    ax1.set_title('(a) Tuned Lens vs Logit Lens Faithfulness (Qwen)', fontweight='bold')

    # v2: 범례를 upper left로 이동 (LL KL 급상승과 겹치지 않도록)
    ax1.legend([l1, l2, l3, l4], ['TL KL', 'LL KL', 'TL Top-1%', 'LL Top-1%'],
              fontsize=7, loc='upper left')
    ax1.grid(alpha=0.2)

    # Panel B: EL classifier confound
    # v2 FIX: JSON 실제 키 이름 사용
    conditions_el = ['mmlu_qwen', 'mmlu_llama', 'mmlu_mistral', 'qwen_hard']
    cond_labels = ['Qwen\nMMLU', 'Llama\nMMLU', 'Mistral\nMMLU', 'Qwen\nHard']

    el_orig = []
    el_matched = []
    hpre_profile = []

    for cond in conditions_el:
        cond_data = el_summary.get(cond, {})
        # v2 FIX: 올바른 키 이름
        el_orig.append(cond_data.get('EL_original_knn_k3', 0.5))
        el_matched.append(cond_data.get('EL_matched_LR', 0.5))
        hpre_profile.append(cond_data.get('H_pre_profile_LR', 0.5))

    x = np.arange(len(conditions_el))
    width = 0.25

    ax2.bar(x - width, el_orig, width, label='EL-original (k-NN)',
            color='#BDBDBD', edgecolor='black', linewidth=0.5)
    ax2.bar(x, el_matched, width, label='H_post profile (LR)',
            color='#4CAF50', edgecolor='black', linewidth=0.5)
    ax2.bar(x + width, hpre_profile, width, label='H_pre profile (LR)',
            color='#2196F3', edgecolor='black', linewidth=0.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(cond_labels)
    ax2.set_ylabel('Test AUROC')
    ax2.set_title('(b) Classifier Choice Changes AUROC', fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.2)
    ax2.set_ylim(0.4, 0.9)

    # v3: classifier effect annotation (gray→green, EL_matched - EL_original)
    # 위치를 green bar 위에 배치 (기존: max bar 위에 → 어느 차이인지 불명확)
    for i in range(len(conditions_el)):
        diff = el_matched[i] - el_orig[i]
        ax2.annotate(f'+{diff:.2f}', xy=(x[i], el_matched[i] + 0.01),
                    ha='center', fontsize=7, color='#D32F2F', fontweight='bold')
    # feature effect annotation (green→blue) 는 별도로
    for i in range(len(conditions_el)):
        diff2 = hpre_profile[i] - el_matched[i]
        if diff2 > 0.02:
            ax2.annotate(f'+{diff2:.2f}', xy=(x[i] + width, hpre_profile[i] + 0.01),
                        ha='center', fontsize=6, color='#1565C0')

    plt.tight_layout()
    path = fig_dir / "fig6_v3.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Generating Figures 1-6 v2 (key fixes + visual improvements)")
    print("=" * 60)
    print(f"Output: {FIG_BASE}/fig{{N}}/")
    print()

    generate_figure1()
    generate_figure2()
    generate_figure3()
    generate_figure4()
    generate_figure5()
    generate_figure6()

    print(f"\nAll v2 figures saved to: {FIG_BASE}/fig{{N}}/fig{{N}}_v3.png")
    print("Original v1 preserved: paper/figures/fig{N}/fig{N}_v1.png")
    print("Done.")
