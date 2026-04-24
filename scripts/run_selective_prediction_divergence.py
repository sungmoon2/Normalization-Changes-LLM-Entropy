"""
P1: Selective Prediction Divergence — H_pre vs H_post best layer
Shows that normalization choice changes which items get rejected.
CPU only. Uses exp31 MMLU generation-average data.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
import time
from _paths import POT_DIR

BASE = Path(__file__).resolve().parent.parent
EXP31 = POT_DIR / "experiments" / "31_MMLU_Domain_Extension"

MODELS = {
    "qwen": {"dir": "EXP_20260219_053638_mmlu_qwen", "n_layers": 28},
    "llama": {"dir": "EXP_20260219_171237_mmlu_llama", "n_layers": 32},
    "mistral": {"dir": "EXP_20260220_000610_mmlu_mistral", "n_layers": 32},
}


def load_data(model_key):
    cfg = MODELS[model_key]
    path = EXP31 / cfg["dir"] / "data" / "sample_results.json"
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    n = len(data)
    L = cfg["n_layers"]

    H_pre = np.zeros((n, L))
    H_post = np.zeros((n, L))
    labels = np.zeros(n, dtype=int)

    for i, sample in enumerate(data):
        labels[i] = 1 if sample["is_correct"] else 0
        ld = sample["layer_data"]
        for j in range(L):
            layer = ld[str(j)]
            H_pre[i, j] = layer["unnormed_entropy"]
            H_post[i, j] = layer["normed_entropy"]

    return H_pre, H_post, labels, n, L


def find_best_layer_sign(entropy_matrix, labels, n, cal_ratio=0.7):
    """Find best layer and sign on calibration set."""
    np.random.seed(42)
    idx = np.random.permutation(n)
    cal_n = int(n * cal_ratio)
    cal_idx = idx[:cal_n]
    test_idx = idx[cal_n:]

    best_layer = 0
    best_sign = 1
    best_auc = 0

    for layer in range(entropy_matrix.shape[1]):
        for sign in [1, -1]:
            scores = sign * entropy_matrix[cal_idx, layer]
            auc = roc_auc_score(labels[cal_idx], scores)
            if auc > best_auc:
                best_auc = auc
                best_layer = layer
                best_sign = sign

    # Evaluate on test set
    test_scores = best_sign * entropy_matrix[test_idx, best_layer]
    test_auc = roc_auc_score(labels[test_idx], test_scores)

    return best_layer, best_sign, best_auc, test_auc, cal_idx, test_idx


def selective_prediction(scores, labels, coverage_pcts):
    """Compute accuracy at each coverage level by rejecting least confident."""
    n = len(scores)
    # Higher score = more confident (keep)
    order = np.argsort(-scores)  # descending confidence

    results = {}
    for cov in coverage_pcts:
        k = int(n * cov / 100)
        if k == 0:
            k = 1
        kept_idx = order[:k]
        acc = np.mean(labels[kept_idx])
        results[cov] = {"n_kept": k, "accuracy": round(float(acc), 4)}
    return results, order


def run_model(model_key):
    print(f"\n{'='*60}")
    print(f"  P1: {model_key}")
    print(f"{'='*60}")

    H_pre, H_post, labels, n, L = load_data(model_key)

    # Find best layer/sign for each
    pre_layer, pre_sign, pre_cal, pre_test, cal_idx, test_idx = \
        find_best_layer_sign(H_pre, labels, n)
    post_layer, post_sign, post_cal, post_test, _, _ = \
        find_best_layer_sign(H_post, labels, n)

    print(f"  H_pre:  best layer=L{pre_layer}, sign={pre_sign:+d}, cal={pre_cal:.4f}, test={pre_test:.4f}")
    print(f"  H_post: best layer=L{post_layer}, sign={post_sign:+d}, cal={post_cal:.4f}, test={post_test:.4f}")

    # Use FULL dataset for selective prediction (not just test set)
    # so the divergence numbers are on the complete 1000 items
    pre_scores = pre_sign * H_pre[:, pre_layer]
    post_scores = post_sign * H_post[:, post_layer]

    coverages = [100, 90, 80, 70, 60, 50, 40, 30, 20]

    pre_results, pre_order = selective_prediction(pre_scores, labels, coverages)
    post_results, post_order = selective_prediction(post_scores, labels, coverages)

    # Compute divergence at each coverage
    print(f"\n  {'Coverage':>10} {'H_pre acc':>10} {'H_post acc':>10} {'Delta':>8} {'Jaccard':>8} {'Disagree':>10}")
    print(f"  {'-'*58}")

    divergence = {}
    for cov in coverages:
        k = int(n * cov / 100)
        if k == 0:
            k = 1

        pre_kept = set(pre_order[:k].tolist())
        post_kept = set(post_order[:k].tolist())

        intersection = len(pre_kept & post_kept)
        union = len(pre_kept | post_kept)
        jaccard = intersection / union if union > 0 else 1.0

        # Items kept by one but not the other
        pre_only = pre_kept - post_kept
        post_only = post_kept - pre_kept
        disagree = len(pre_only) + len(post_only)

        pre_acc = pre_results[cov]["accuracy"]
        post_acc = post_results[cov]["accuracy"]
        delta = pre_acc - post_acc

        divergence[cov] = {
            "n_kept": k,
            "pre_accuracy": pre_acc,
            "post_accuracy": post_acc,
            "delta": round(delta, 4),
            "jaccard": round(jaccard, 4),
            "disagree_count": disagree,
            "disagree_pct": round(disagree / k * 100, 1) if k > 0 else 0,
            "pre_only_count": len(pre_only),
            "post_only_count": len(post_only),
        }

        print(f"  {cov:>8}% {pre_acc:>10.4f} {post_acc:>10.4f} {delta:>+8.4f} {jaccard:>8.4f} {disagree:>6} ({disagree/k*100:.0f}%)")

    # At coverage 80%, show accuracy of the DISAGREED items
    cov_detail = 80
    k = int(n * cov_detail / 100)
    pre_kept_set = set(pre_order[:k].tolist())
    post_kept_set = set(post_order[:k].tolist())
    pre_only = list(pre_kept_set - post_kept_set)
    post_only = list(post_kept_set - pre_kept_set)

    if len(pre_only) > 0:
        pre_only_acc = np.mean(labels[pre_only])
    else:
        pre_only_acc = float('nan')
    if len(post_only) > 0:
        post_only_acc = np.mean(labels[post_only])
    else:
        post_only_acc = float('nan')

    print(f"\n  At {cov_detail}% coverage:")
    print(f"    Items kept by H_pre only: {len(pre_only)}, their accuracy: {pre_only_acc:.4f}")
    print(f"    Items kept by H_post only: {len(post_only)}, their accuracy: {post_only_acc:.4f}")

    return {
        "model": model_key,
        "n_samples": n,
        "base_accuracy": round(float(np.mean(labels)), 4),
        "H_pre_best": {"layer": int(pre_layer), "sign": int(pre_sign), "cal_auroc": round(pre_cal, 4), "test_auroc": round(pre_test, 4)},
        "H_post_best": {"layer": int(post_layer), "sign": int(post_sign), "cal_auroc": round(post_cal, 4), "test_auroc": round(post_test, 4)},
        "layer_shift": abs(int(pre_layer) - int(post_layer)),
        "divergence": divergence,
        "detail_at_80pct": {
            "pre_only_count": len(pre_only),
            "pre_only_accuracy": round(float(pre_only_acc), 4) if not np.isnan(pre_only_acc) else None,
            "post_only_count": len(post_only),
            "post_only_accuracy": round(float(post_only_acc), 4) if not np.isnan(post_only_acc) else None,
        }
    }


def main():
    t0 = time.time()
    results = {}

    for model_key in ["qwen", "llama", "mistral"]:
        results[model_key] = run_model(model_key)

    # Save
    out_dir = BASE / "experiments" / "51_Selective_Prediction_Divergence"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "divergence_results.json"
    with open(out_file, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.1f}s")
    print(f"Saved: {out_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
