"""
Figure 1 v4 — Manual axes positioning, no tight_layout conflicts.
figsize=(14,6) → 51% scaling → all fonts ~2x to compensate.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FIG_BASE = PROJECT_ROOT / "paper" / "figures"


def generate():
    fig_dir = FIG_BASE / "fig1"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 6))

    # Manual axes: [left, bottom, width, height] in figure fraction
    ax_a = fig.add_axes([0.02, 0.08, 0.42, 0.85])   # Panel A: left 42%
    ax_b = fig.add_axes([0.46, 0.08, 0.24, 0.85])    # Panel B: middle 24%
    ax_c = fig.add_axes([0.72, 0.08, 0.27, 0.85])    # Panel C: right 27%

    fs_title = 20
    fs_box = 20
    fs_lbl = 15
    fs_path = 15
    y_off = 0.9

    # ========== Panel A ==========
    ax = ax_a
    ax.set_xlim(-1, 11.5)
    ax.set_ylim(-0.5, 8.5)
    ax.axis('off')
    ax.text(5.0, 8.2, '(a) Decoded Entropy\nBefore vs. After RMSNorm',
            ha='center', fontsize=fs_title, fontweight='bold')

    bw, bh = 2.2, 1.1

    # Top row y=6
    yt = 6.0
    for x, txt, c in [(2.5, '$h_\\ell$', '#E8F4FD'),
                       (5.5, '$Wh_\\ell$', '#FFF3E0'),
                       (8.5, '$H_{\\mathrm{pre}}$', '#FFCDD2')]:
        ax.add_patch(plt.Rectangle((x-bw/2, yt-bh/2), bw, bh,
                     fc=c, ec='black', lw=1.5, zorder=2))
        ax.text(x, yt, txt, ha='center', va='center', fontsize=fs_box, zorder=3)

    ax.annotate('', xy=(5.5-bw/2-0.05, yt), xytext=(2.5+bw/2+0.05, yt),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(4.0, yt+y_off, 'lm_head', ha='center', fontsize=fs_lbl, style='italic')
    ax.annotate('', xy=(8.5-bw/2-0.05, yt), xytext=(5.5+bw/2+0.05, yt),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(7.0, yt+y_off, 'softmax $\\to$ $H$', ha='center', fontsize=fs_lbl, style='italic')

    ax.text(-0.5, yt, 'Pre-norm\npath', ha='left', va='center', fontsize=fs_path,
            color='#D32F2F', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='#FFEBEE', alpha=0.8))

    # Bottom row y=2
    yb = 2.0
    for x, txt, c in [(2.5, '$h_\\ell$', '#E8F4FD'),
                       (5.5, '$W{\\cdot}N_\\epsilon(h_\\ell)$', '#E8F5E9'),
                       (8.5, '$H_{\\mathrm{post}}$', '#C8E6C9')]:
        ax.add_patch(plt.Rectangle((x-bw/2, yb-bh/2), bw, bh,
                     fc=c, ec='black', lw=1.5, zorder=2))
        ax.text(x, yb, txt, ha='center', va='center', fontsize=fs_box, zorder=3)

    ax.annotate('', xy=(5.5-bw/2-0.05, yb), xytext=(2.5+bw/2+0.05, yb),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(4.0, yb+y_off, 'norm $\\to$ $W$', ha='center', fontsize=fs_lbl, style='italic')
    ax.annotate('', xy=(8.5-bw/2-0.05, yb), xytext=(5.5+bw/2+0.05, yb),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(7.0, yb+y_off, 'softmax $\\to$ $H$', ha='center', fontsize=fs_lbl, style='italic')

    ax.text(-0.5, yb, 'Post-norm\npath', ha='left', va='center', fontsize=fs_path,
            color='#388E3C', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='#E8F5E9', alpha=0.8))

    # ========== Panel B ==========
    ax = ax_b
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.9, 1.6)
    ax.set_aspect('equal')
    ax.text(0, 1.75, '(b) Scale-Direction\nDecomposition',
            ha='center', va='bottom', fontsize=fs_title-2, fontweight='bold')

    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, lw=1)

    ang = np.pi/5; r = 1.2
    ax.annotate('', xy=(r*np.cos(ang), r*np.sin(ang)), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=3, color='#1565C0'))
    ax.text(r*np.cos(ang)+0.05, r*np.sin(ang)+0.14,
            '$h_\\ell = r_\\ell \\cdot u_\\ell$', fontsize=16, color='#1565C0')

    ax.annotate('', xy=(np.cos(ang), np.sin(ang)), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=2, color='#E65100'))
    ax.text(np.cos(ang)+0.1, np.sin(ang)-0.2,
            '$u_\\ell$', fontsize=16, color='#E65100')

    ax.text(r*np.cos(ang)*0.35-0.3, r*np.sin(ang)*0.35+0.2,
            '$r_\\ell = \\|h_\\ell\\|$', fontsize=13, color='#D32F2F')

    ax.text(0, -1.5,
            '$z_{\\mathrm{pre}} = r_\\ell W u_\\ell$\n$r_\\ell$ acts as inverse temperature',
            ha='center', fontsize=13, style='italic',
            bbox=dict(boxstyle='round,pad=0.4', fc='#FFF3E0', alpha=0.9))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

    # ========== Panel C ==========
    ax = ax_c
    ax.set_xlim(-1, 11)
    ax.set_ylim(-0.5, 8.5)
    ax.axis('off')
    ax.text(5, 8.2, '(c) Token Positions', ha='center',
            fontsize=fs_title, fontweight='bold')

    bw_c, bh_c = 4.5, 1.8
    items = [
        (5, 6.0, 'Step 0 (prompt-last)', '#BBDEFB', 'Last input token'),
        (5, 3.5, 'Step 1 (first-gen)', '#C8E6C9', 'First generated token'),
        (5, 1.0, 'Full Avg (all gen)', '#FFF9C4', 'Mean over generation'),
    ]
    for x, y, label, color, desc in items:
        ax.add_patch(plt.Rectangle((x-bw_c/2, y-bh_c/2), bw_c, bh_c,
                     fc=color, ec='black', lw=1.2, zorder=2))
        ax.text(x, y+0.2, label, ha='center', va='center',
                fontsize=16, fontweight='bold', zorder=3)
        ax.text(x, y-0.5, desc, ha='center', va='center',
                fontsize=12, style='italic', color='#555', zorder=3)

    for y1, y2 in [(6.0, 3.5), (3.5, 1.0)]:
        ax.annotate('', xy=(5, y2+bh_c/2+0.08), xytext=(5, y1-bh_c/2-0.08),
                    arrowprops=dict(arrowstyle='->', lw=2, color='#666'))

    # arrows already convey direction — no overlapping label needed

    path = fig_dir / "fig1_v4.png"
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(f"Saved: {path}")
    return path


if __name__ == "__main__":
    generate()
