"""
Fig 4 v5: 3-model token-position bar chart (IEEE column width, 7pt+ fonts)
==========================================================================
Replaces v4 which incorrectly showed Qwen-only 2-panel.
Uses aligned 3-model data from phase2_gpu_tokenpos_v2_aligned.
"""

import json
import numpy as np
import matplotlib
from _paths import POT_DIR
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).parent.parent
TP_BASE = POT_DIR / "experiments" / "44_GPT25_Experiments" / "phase2_gpu_tokenpos_v2_aligned"

MODEL_CONFIGS = {
    "qwen": {"n_layers": 28, "label": "Qwen"},
    "llama": {"n_layers": 32, "label": "Llama"},
    "mistral": {"n_layers": 32, "label": "Mistral"},
}

IEEE_COL = 3.5  # inches, single column


def load_checkpoint(model_key):
    path = TP_BASE / f"{model_key}_mmlu_aligned" / "data" / "checkpoint.json"
    if not path.exists():
        print(f"ERROR: {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def compute_position_auroc(data, n_layers):
    """Compute held-out AUROC for each position x metric."""
    n = len(data)
    y = np.array([1 if s.get("is_correct", False) else 0 for s in data])

    positions = ["step0_prompt_last", "step1_first_gen", "full_gen_avg"]
    metrics = ["unnormed_entropy", "normed_entropy", "logit_std"]

    np.random.seed(42)
    n_cal = int(0.7 * n)
    perm = np.random.permutation(n)
    cal_idx, test_idx = perm[:n_cal], perm[n_cal:]

    results = {}
    for pos in positions:
        results[pos] = {}
        for metric in metrics:
            feats = np.zeros((n, n_layers))
            for i, s in enumerate(data):
                pd = s.get("position_data", {})
                pos_data = pd.get(pos, {})
                for l in range(n_layers):
                    feats[i, l] = pos_data.get(str(l), {}).get(metric, 0)

            best_layer, best_sign, best_cal = -1, 1, 0.5
            for l in range(n_layers):
                for sign in [1, -1]:
                    try:
                        auc = roc_auc_score(y[cal_idx], sign * feats[cal_idx, l])
                        if auc > best_cal:
                            best_cal = auc
                            best_layer = l
                            best_sign = sign
                    except:
                        pass

            test_auroc = roc_auc_score(y[test_idx], best_sign * feats[test_idx, best_layer])
            results[pos][metric] = {
                "best_layer": int(best_layer),
                "test_auroc": round(float(test_auroc), 4),
            }

    return results


def main():
    # Compute AUROC for each model
    all_results = {}
    for model_key, cfg in MODEL_CONFIGS.items():
        print(f"Loading {model_key}...")
        data = load_checkpoint(model_key)
        if data is None:
            continue
        print(f"  {len(data)} samples, computing AUROC...")
        auroc = compute_position_auroc(data, cfg["n_layers"])
        all_results[model_key] = auroc

        # Print for verification
        for pos in ["step0_prompt_last", "step1_first_gen", "full_gen_avg"]:
            for metric in ["unnormed_entropy", "normed_entropy", "logit_std"]:
                r = auroc[pos][metric]
                print(f"  {pos} {metric}: L{r['best_layer']} AUROC={r['test_auroc']:.4f}")

    # Verify against Table 8 values
    print("\n=== Verification against Table 8 ===")
    table8 = {
        "qwen": {"unnormed_entropy": [0.6750, 0.6036, 0.6790], "normed_entropy": [0.6035, 0.6509, 0.6999], "logit_std": [0.6458, 0.5828, 0.6674]},
        "llama": {"unnormed_entropy": [0.5688, 0.5694, 0.5802], "normed_entropy": [0.5964, 0.6354, 0.5851], "logit_std": [0.5939, 0.5571, 0.6522]},
        "mistral": {"unnormed_entropy": [0.5517, 0.6204, 0.6757], "normed_entropy": [0.5874, 0.6514, 0.5723], "logit_std": [0.6321, 0.5645, 0.6734]},
    }
    positions = ["step0_prompt_last", "step1_first_gen", "full_gen_avg"]
    for model in ["qwen", "llama", "mistral"]:
        if model not in all_results:
            continue
        for metric in ["unnormed_entropy", "normed_entropy", "logit_std"]:
            for i, pos in enumerate(positions):
                computed = all_results[model][pos][metric]["test_auroc"]
                expected = table8[model][metric][i]
                diff = abs(computed - expected)
                status = "OK" if diff < 0.001 else f"DIFF={diff:.4f}"
                print(f"  {model} {metric} {pos}: computed={computed:.4f} expected={expected:.4f} {status}")

    # Generate figure
    print("\nGenerating Figure 4 v5 (3-model, IEEE format)...")

    positions = ["step0_prompt_last", "step1_first_gen", "full_gen_avg"]
    pos_labels = ["Step 0", "Step 1", "Full Avg"]
    pos_colors = ["#2E86C1", "#E67E22", "#27AE60"]
    metrics = ["unnormed_entropy", "normed_entropy", "logit_std"]
    metric_labels = [r"$H_{pre}$", r"$H_{post}$", "logit_std"]
    models = ["qwen", "llama", "mistral"]
    model_labels = ["Qwen", "Llama", "Mistral"]

    plt.rcParams.update({
        'font.size': 7,
        'axes.titlesize': 8,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
    })

    fig, axes = plt.subplots(1, 3, figsize=(IEEE_COL, 2.0), sharey=True)

    x = np.arange(len(metrics))
    width = 0.25

    for ax_idx, (model, model_label) in enumerate(zip(models, model_labels)):
        ax = axes[ax_idx]
        if model not in all_results:
            ax.set_title(f"{model_label} (no data)")
            continue

        r = all_results[model]

        for j, (pos, pos_label, color) in enumerate(zip(positions, pos_labels, pos_colors)):
            vals = [r[pos][m]["test_auroc"] for m in metrics]
            bars = ax.bar(x + (j - 1) * width, vals, width,
                         label=pos_label, color=color, alpha=0.85,
                         edgecolor='white', linewidth=0.3)
            for bar, val in zip(bars, vals):
                if val > 0.52:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=5, rotation=45)

        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=6)
        ax.set_title(model_label, fontweight='bold')
        ax.set_ylim(0.45, 0.78)
        ax.grid(True, axis='y', alpha=0.2)
        if ax_idx == 0:
            ax.set_ylabel('Held-out AUROC')

    # Shared legend at bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=6,
               bbox_to_anchor=(0.5, -0.12), frameon=True, edgecolor='lightgray')

    plt.tight_layout()

    # Save
    fig_dir = BASE / "paper" / "figures" / "fig4"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "fig4_v5.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    # Deploy to ieee_access
    import shutil
    deploy_dir = BASE / "paper" / "latex" / "ieee_access" / "figures"
    backup_path = deploy_dir / "fig4_v4_backup.png"
    current_path = deploy_dir / "fig4.png"
    if current_path.exists():
        shutil.copy2(current_path, backup_path)
        print(f"Backup: {backup_path}")
    shutil.copy2(out_path, current_path)
    print(f"Deployed: {current_path}")


if __name__ == "__main__":
    main()
