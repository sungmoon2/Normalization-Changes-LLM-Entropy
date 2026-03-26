"""
Repeated Split Robustness Analysis
-----------------------------------
For each condition (Qwen Hard, Qwen MMLU, Llama MMLU, Mistral MMLU):
  - Load per-sample per-layer scalar data
  - Run 20 independent 70/30 stratified splits (seeds 0-19)
  - For each split: select best layer/sign on cal set, evaluate on test set
  - Report: layer/sign selection frequency, AUROC mean/std/min/max

Metrics: H_pre (unnormed_entropy), H_post (normed_entropy), logit_std, h_norm

No GPU needed. CPU only.
"""

import json
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
import os
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(BASE, "PoT_Experiment_Entropy_Attention_Extraction_Experiment", "experiments")

CONDITIONS = {
    "qwen_hard": {
        "path": os.path.join(EXP, "23_Normed_Difficulty_Analysis", "EXP_20260213_113717_normed_hard", "data", "sample_results.json"),
        "n_layers": 28,
    },
    "qwen_mmlu": {
        "path": os.path.join(EXP, "31_MMLU_Domain_Extension", "EXP_20260219_053638_mmlu_qwen", "data", "sample_results.json"),
        "n_layers": 28,
    },
    "llama_mmlu": {
        "path": os.path.join(EXP, "31_MMLU_Domain_Extension", "EXP_20260219_171237_mmlu_llama", "data", "sample_results.json"),
        "n_layers": 32,
    },
    "mistral_mmlu": {
        "path": os.path.join(EXP, "31_MMLU_Domain_Extension", "EXP_20260220_000610_mmlu_mistral", "data", "sample_results.json"),
        "n_layers": 32,
    },
}

METRICS = ["unnormed_entropy", "normed_entropy", "logit_std", "h_norm"]
METRIC_NAMES = {"unnormed_entropy": "H_pre", "normed_entropy": "H_post", "logit_std": "logit_std", "h_norm": "h_norm"}
N_SPLITS = 20
SEEDS = list(range(N_SPLITS))


def load_data(path, n_layers):
    with open(path) as f:
        samples = json.load(f)

    # Filter NaN samples
    valid = []
    for s in samples:
        if s.get("layer_data") and all(str(l) in s["layer_data"] for l in range(n_layers)):
            has_nan = False
            for l in range(n_layers):
                for m in METRICS:
                    v = s["layer_data"][str(l)].get(m)
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        has_nan = True
                        break
                if has_nan:
                    break
            if not has_nan:
                valid.append(s)

    n = len(valid)
    labels = np.array([1 if s["is_correct"] else 0 for s in valid])

    # Build feature matrix: (n_samples, n_layers, n_metrics)
    data = {}
    for m in METRICS:
        arr = np.zeros((n, n_layers))
        for i, s in enumerate(valid):
            for l in range(n_layers):
                arr[i, l] = s["layer_data"][str(l)][m]
        data[m] = arr

    return data, labels, n


def evaluate_split(data, labels, cal_idx, test_idx, n_layers):
    results = {}
    for metric in METRICS:
        arr = data[metric]
        cal_labels = labels[cal_idx]
        test_labels = labels[test_idx]

        best_layer = -1
        best_sign = 1
        best_cal_auroc = -1

        for l in range(n_layers):
            cal_scores = arr[cal_idx, l]
            for sign in [+1, -1]:
                signed = sign * cal_scores
                try:
                    auroc = roc_auc_score(cal_labels, signed)
                except ValueError:
                    auroc = 0.5
                if auroc > best_cal_auroc:
                    best_cal_auroc = auroc
                    best_layer = l
                    best_sign = sign

        # Evaluate on test
        test_scores = best_sign * arr[test_idx, best_layer]
        try:
            test_auroc = roc_auc_score(test_labels, test_scores)
        except ValueError:
            test_auroc = 0.5

        results[metric] = {
            "best_layer": best_layer,
            "best_sign": best_sign,
            "cal_auroc": float(best_cal_auroc),
            "test_auroc": float(test_auroc),
        }

    return results


def run_condition(name, config):
    print(f"\n{'='*60}")
    print(f"Condition: {name}")
    print(f"{'='*60}")

    data, labels, n = load_data(config["path"], config["n_layers"])
    n_layers = config["n_layers"]
    acc = labels.mean()
    print(f"  Samples: {n}, Accuracy: {acc:.3f}, Layers: {n_layers}")

    all_results = {m: [] for m in METRICS}

    for seed in SEEDS:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
        cal_idx, test_idx = next(sss.split(np.zeros(n), labels))

        res = evaluate_split(data, labels, cal_idx, test_idx, n_layers)
        for m in METRICS:
            all_results[m].append(res[m])

    # Summarize
    summary = {}
    for m in METRICS:
        layers = [r["best_layer"] for r in all_results[m]]
        signs = [r["best_sign"] for r in all_results[m]]
        aurocs = [r["test_auroc"] for r in all_results[m]]

        # Layer frequency
        from collections import Counter
        layer_freq = Counter(layers)
        most_common_layer = layer_freq.most_common(1)[0]

        # Sign consistency
        sign_freq = Counter(signs)
        dominant_sign = sign_freq.most_common(1)[0]

        summary[m] = {
            "metric_name": METRIC_NAMES[m],
            "auroc_mean": float(np.mean(aurocs)),
            "auroc_std": float(np.std(aurocs)),
            "auroc_min": float(np.min(aurocs)),
            "auroc_max": float(np.max(aurocs)),
            "most_common_layer": most_common_layer[0],
            "most_common_layer_freq": f"{most_common_layer[1]}/{N_SPLITS}",
            "layer_distribution": dict(layer_freq.most_common()),
            "dominant_sign": dominant_sign[0],
            "sign_consistency": f"{dominant_sign[1]}/{N_SPLITS}",
            "all_splits": all_results[m],
        }

        print(f"\n  {METRIC_NAMES[m]}:")
        print(f"    AUROC: {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f} [{np.min(aurocs):.4f}, {np.max(aurocs):.4f}]")
        print(f"    Most common layer: L{most_common_layer[0]} ({most_common_layer[1]}/{N_SPLITS})")
        print(f"    Layer dist: {dict(layer_freq.most_common(5))}")
        print(f"    Sign consistency: {dominant_sign[0]:+d} ({dominant_sign[1]}/{N_SPLITS})")

    return {
        "condition": name,
        "n_samples": n,
        "accuracy": float(acc),
        "n_layers": n_layers,
        "n_splits": N_SPLITS,
        "seeds": SEEDS,
        "metrics": summary,
    }


def main():
    start = time.time()
    results = {}

    for name, config in CONDITIONS.items():
        if not os.path.exists(config["path"]):
            print(f"SKIP {name}: file not found at {config['path']}")
            continue
        results[name] = run_condition(name, config)

    # Save
    out_dir = os.path.join(EXP, "41_Repeated_Split_Robustness")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "repeated_split_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Results saved to: {out_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY TABLE (for paper)")
    print(f"{'='*60}")
    print(f"{'Condition':<16} {'Metric':<10} {'AUROC mean':>10} {'std':>6} {'range':>14} {'Modal L':>8} {'L freq':>8} {'Sign':>8}")
    print("-" * 82)
    for cond_name, cond_data in results.items():
        for m in METRICS:
            s = cond_data["metrics"][m]
            print(f"{cond_name:<16} {s['metric_name']:<10} {s['auroc_mean']:>10.4f} {s['auroc_std']:>6.4f} [{s['auroc_min']:.4f},{s['auroc_max']:.4f}] L{s['most_common_layer']:>2} {s['most_common_layer_freq']:>8} {s['dominant_sign']:>+3d} {s['sign_consistency']:>5}")


if __name__ == "__main__":
    main()
