"""
Per-layer AUROC curves with 95% bootstrap CI bands.
Addresses GPT 28th W3: "critical layer shift claim is not statistically established"

Outputs:
  - Figure: per-layer AUROC curves for H_pre, H_post, logit_std, h_norm (3 models)
  - JSON: full per-layer AUROC + CI data
  - Analysis: layers statistically indistinguishable from best
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from _paths import POT_DIR

BASE = Path(__file__).resolve().parent.parent
EXP31 = POT_DIR / "experiments" / "31_MMLU_Domain_Extension"
OUT_DIR = BASE / "experiments" / "48_PerLayer_AUROC_Curves"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "qwen": {
        "path": EXP31 / "EXP_20260219_053638_mmlu_qwen" / "data" / "sample_results.json",
        "n_layers": 28,
        "label": "Qwen2.5-7B"
    },
    "llama": {
        "path": EXP31 / "EXP_20260219_171237_mmlu_llama" / "data" / "sample_results.json",
        "n_layers": 32,
        "label": "Llama-3-8B"
    },
    "mistral": {
        "path": EXP31 / "EXP_20260220_000610_mmlu_mistral" / "data" / "sample_results.json",
        "n_layers": 32,
        "label": "Mistral-7B"
    }
}

METRICS = {
    "H_pre": {"key": "unnormed_entropy", "default_sign": -1},
    "H_post": {"key": "normed_entropy", "default_sign": -1},
    "logit_std": {"key": "logit_std", "default_sign": +1},
    "h_norm": {"key": "h_norm", "default_sign": +1},
}

N_BOOTSTRAP = 1000
SEED = 42


def load_data(path):
    with open(path, "r") as f:
        samples = json.load(f)
    return samples


def compute_auroc_both_signs(y_true, scores):
    """Compute AUROC with best sign (max of pos and neg)."""
    try:
        auc_pos = roc_auc_score(y_true, scores)
        auc_neg = roc_auc_score(y_true, -np.array(scores))
        if auc_neg > auc_pos:
            return auc_neg, -1
        return auc_pos, +1
    except ValueError:
        return 0.5, +1


def bootstrap_auroc(y_true, scores, sign, n_boot=N_BOOTSTRAP, seed=SEED):
    """Bootstrap 95% CI for AUROC with fixed sign."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    y = np.array(y_true)
    s = np.array(scores) * sign
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y[idx], s[idx]))
        except ValueError:
            continue
    aucs = np.array(aucs)
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5), np.std(aucs)


def analyze_model(model_name, model_cfg):
    print(f"\n{'='*60}")
    print(f"Processing {model_cfg['label']} ({model_name})")
    print(f"{'='*60}")

    samples = load_data(model_cfg["path"])
    n_layers = model_cfg["n_layers"]
    n_samples = len(samples)

    y_true = [1 if s["is_correct"] else 0 for s in samples]
    acc = sum(y_true) / len(y_true)
    print(f"  Samples: {n_samples}, Accuracy: {acc:.1%}")

    results = {}
    for metric_name, metric_cfg in METRICS.items():
        key = metric_cfg["key"]
        print(f"\n  {metric_name} ({key}):")

        layer_results = []
        for layer in range(n_layers):
            scores = [s["layer_data"][str(layer)][key] for s in samples]
            auroc, best_sign = compute_auroc_both_signs(y_true, scores)
            ci_lo, ci_hi, std = bootstrap_auroc(y_true, scores, best_sign)

            layer_results.append({
                "layer": layer,
                "auroc": round(auroc, 4),
                "sign": int(best_sign),
                "ci_lo": round(ci_lo, 4),
                "ci_hi": round(ci_hi, 4),
                "std": round(std, 4),
            })

        # Find best layer and layers indistinguishable from best
        best_idx = max(range(n_layers), key=lambda i: layer_results[i]["auroc"])
        best_auroc = layer_results[best_idx]["auroc"]
        best_ci_lo = layer_results[best_idx]["ci_lo"]

        # A layer is "indistinguishable from best" if its CI overlaps with best's CI
        indistinguishable = []
        for lr in layer_results:
            if lr["ci_hi"] >= best_ci_lo:
                indistinguishable.append(lr["layer"])

        best_layer = layer_results[best_idx]["layer"]
        print(f"    Best: L{best_layer} AUROC={best_auroc:.4f} "
              f"[{layer_results[best_idx]['ci_lo']:.4f}, {layer_results[best_idx]['ci_hi']:.4f}]")
        print(f"    Indistinguishable layers: {len(indistinguishable)}/{n_layers} "
              f"(range L{min(indistinguishable)}-L{max(indistinguishable)})")

        results[metric_name] = {
            "per_layer": layer_results,
            "best_layer": best_layer,
            "best_auroc": best_auroc,
            "indistinguishable_from_best": indistinguishable,
            "n_indistinguishable": len(indistinguishable),
        }

    return {
        "model": model_name,
        "label": model_cfg["label"],
        "n_samples": n_samples,
        "accuracy": round(acc, 4),
        "metrics": results,
    }


def compute_shift_analysis(all_results):
    """Compute H_pre vs H_post shift with statistical context."""
    print(f"\n{'='*60}")
    print("CRITICAL LAYER SHIFT ANALYSIS")
    print(f"{'='*60}")

    shifts = []
    for model_name, res in all_results.items():
        h_pre = res["metrics"]["H_pre"]
        h_post = res["metrics"]["H_post"]

        pre_best = h_pre["best_layer"]
        post_best = h_post["best_layer"]
        shift = abs(pre_best - post_best)

        # Check if H_pre best is within H_post indistinguishable set and vice versa
        pre_in_post_set = pre_best in h_post["indistinguishable_from_best"]
        post_in_pre_set = post_best in h_pre["indistinguishable_from_best"]

        # H_pre AUROC at H_post's best layer
        pre_at_post_best = h_pre["per_layer"][post_best]["auroc"]
        # H_post AUROC at H_pre's best layer
        post_at_pre_best = h_post["per_layer"][pre_best]["auroc"]

        print(f"\n  {res['label']}:")
        print(f"    H_pre best: L{pre_best} (AUROC={h_pre['best_auroc']:.4f}, "
              f"indisting={h_pre['n_indistinguishable']}/{len(h_pre['per_layer'])} layers)")
        print(f"    H_post best: L{post_best} (AUROC={h_post['best_auroc']:.4f}, "
              f"indisting={h_post['n_indistinguishable']}/{len(h_post['per_layer'])} layers)")
        print(f"    Shift: {shift} layers")
        print(f"    H_pre best (L{pre_best}) in H_post indisting set? {pre_in_post_set}")
        print(f"    H_post best (L{post_best}) in H_pre indisting set? {post_in_pre_set}")
        print(f"    H_pre at H_post's best (L{post_best}): {pre_at_post_best:.4f} "
              f"(vs best {h_pre['best_auroc']:.4f}, drop={h_pre['best_auroc']-pre_at_post_best:.4f})")
        print(f"    H_post at H_pre's best (L{pre_best}): {post_at_pre_best:.4f} "
              f"(vs best {h_post['best_auroc']:.4f}, drop={h_post['best_auroc']-post_at_pre_best:.4f})")

        shifts.append({
            "model": model_name,
            "label": res["label"],
            "h_pre_best": pre_best,
            "h_post_best": post_best,
            "shift": shift,
            "h_pre_indisting": h_pre["n_indistinguishable"],
            "h_post_indisting": h_post["n_indistinguishable"],
            "pre_in_post_set": pre_in_post_set,
            "post_in_pre_set": post_in_pre_set,
            "h_pre_at_post_best": pre_at_post_best,
            "h_post_at_pre_best": post_at_pre_best,
        })

    return shifts


def generate_figures(all_results):
    """Generate per-layer AUROC curve figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping figures")
        return

    colors = {
        "H_pre": "#d62728",
        "H_post": "#2ca02c",
        "logit_std": "#1f77b4",
        "h_norm": "#ff7f0e",
    }

    # Figure 1: 3-panel (one per model), 4 metrics each
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (model_name, res) in enumerate(all_results.items()):
        ax = axes[i]
        n_layers = len(res["metrics"]["H_pre"]["per_layer"])

        for metric_name in ["H_pre", "H_post", "logit_std", "h_norm"]:
            data = res["metrics"][metric_name]["per_layer"]
            layers = [d["layer"] for d in data]
            aurocs = [d["auroc"] for d in data]
            ci_los = [d["ci_lo"] for d in data]
            ci_his = [d["ci_hi"] for d in data]

            ax.plot(layers, aurocs, color=colors[metric_name], label=metric_name, linewidth=1.5)
            ax.fill_between(layers, ci_los, ci_his, color=colors[metric_name], alpha=0.15)

            # Mark best layer
            best_idx = max(range(len(aurocs)), key=lambda j: aurocs[j])
            ax.plot(layers[best_idx], aurocs[best_idx], "o", color=colors[metric_name],
                    markersize=6, zorder=5)

        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.set_xlabel("Layer", fontsize=11)
        if i == 0:
            ax.set_ylabel("AUROC", fontsize=11)
        ax.set_title(f"{res['label']} (acc={res['accuracy']:.1%})", fontsize=12)
        ax.set_ylim(0.45, 0.75)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Per-Layer Correctness AUROC with 95% Bootstrap CI\n"
                 "(MMLU, generation-average, full-sample)", fontsize=13, y=1.02)
    plt.tight_layout()
    fig_path = OUT_DIR / "perlayer_auroc_curves.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {fig_path}")

    # Figure 2: H_pre vs H_post comparison (shift visualization)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (model_name, res) in enumerate(all_results.items()):
        ax = axes[i]
        n_layers = len(res["metrics"]["H_pre"]["per_layer"])

        for metric_name in ["H_pre", "H_post"]:
            data = res["metrics"][metric_name]["per_layer"]
            layers = [d["layer"] for d in data]
            aurocs = [d["auroc"] for d in data]
            ci_los = [d["ci_lo"] for d in data]
            ci_his = [d["ci_hi"] for d in data]

            ax.plot(layers, aurocs, color=colors[metric_name], label=metric_name,
                    linewidth=2)
            ax.fill_between(layers, ci_los, ci_his, color=colors[metric_name], alpha=0.2)

            best_idx = max(range(len(aurocs)), key=lambda j: aurocs[j])
            ax.plot(layers[best_idx], aurocs[best_idx], "o", color=colors[metric_name],
                    markersize=8, zorder=5)
            ax.annotate(f"L{layers[best_idx]}", (layers[best_idx], aurocs[best_idx]),
                        textcoords="offset points", xytext=(5, 8), fontsize=9,
                        color=colors[metric_name], fontweight="bold")

        # Shade indistinguishable region for each
        for metric_name in ["H_pre", "H_post"]:
            indist = res["metrics"][metric_name]["indistinguishable_from_best"]
            if indist:
                ax.axvspan(min(indist) - 0.3, max(indist) + 0.3,
                           color=colors[metric_name], alpha=0.05)

        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.set_xlabel("Layer", fontsize=11)
        if i == 0:
            ax.set_ylabel("AUROC", fontsize=11)

        shift = abs(res["metrics"]["H_pre"]["best_layer"] - res["metrics"]["H_post"]["best_layer"])
        ax.set_title(f"{res['label']} (shift={shift} layers)", fontsize=12)
        ax.set_ylim(0.45, 0.72)
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.suptitle("H_pre vs H_post: Critical Layer Shift with 95% CI\n"
                 "(MMLU, generation-average, full-sample)", fontsize=13, y=1.02)
    plt.tight_layout()
    fig_path2 = OUT_DIR / "perlayer_shift_comparison.png"
    plt.savefig(fig_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {fig_path2}")


def main():
    print("Per-Layer AUROC Analysis with Bootstrap CI")
    print(f"Bootstrap resamples: {N_BOOTSTRAP}")

    all_results = {}
    for model_name, model_cfg in MODELS.items():
        all_results[model_name] = analyze_model(model_name, model_cfg)

    shifts = compute_shift_analysis(all_results)

    # Save results
    output = {
        "config": {
            "n_bootstrap": N_BOOTSTRAP,
            "seed": SEED,
            "data_source": "exp31 MMLU generation-average",
            "ci_method": "percentile bootstrap",
        },
        "models": all_results,
        "shift_analysis": shifts,
    }

    json_path = OUT_DIR / "perlayer_auroc_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {json_path}")

    generate_figures(all_results)

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY: Layers Indistinguishable from Best")
    print(f"{'='*60}")
    for model_name, res in all_results.items():
        print(f"\n{res['label']}:")
        for metric_name in ["H_pre", "H_post", "logit_std", "h_norm"]:
            m = res["metrics"][metric_name]
            n_total = len(m["per_layer"])
            print(f"  {metric_name:10s}: best L{m['best_layer']:2d} "
                  f"({m['best_auroc']:.4f}), "
                  f"indisting={m['n_indistinguishable']:2d}/{n_total} layers "
                  f"(range L{min(m['indistinguishable_from_best'])}-"
                  f"L{max(m['indistinguishable_from_best'])})")


if __name__ == "__main__":
    main()
