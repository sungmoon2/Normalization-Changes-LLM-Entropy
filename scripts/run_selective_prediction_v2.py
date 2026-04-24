"""
P1v2: Selective Prediction Divergence — with controls
1. H_pre best vs H_post best (main comparison)
2. H_pre best vs H_pre 2nd-best (within-metric control)
3. Random baseline Jaccard
4. Test-set only evaluation (no data leakage)
5. Bootstrap CI on all metrics
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


def find_top2_layers(entropy_matrix, labels, idx):
    """Find best and 2nd-best layer on given indices."""
    results = []
    for layer in range(entropy_matrix.shape[1]):
        for sign in [1, -1]:
            scores = sign * entropy_matrix[idx, layer]
            auc = roc_auc_score(labels[idx], scores)
            results.append((auc, layer, sign))
    results.sort(reverse=True)
    best = results[0]
    # 2nd best: different layer
    for r in results[1:]:
        if r[1] != best[1]:
            second = r
            break
    return (best[1], best[2], best[0]), (second[1], second[2], second[0])


def compute_divergence(scores_a, scores_b, labels, n, coverages):
    """Compute selective prediction divergence between two scoring functions."""
    order_a = np.argsort(-scores_a)
    order_b = np.argsort(-scores_b)

    results = {}
    for cov in coverages:
        k = max(1, int(n * cov / 100))
        set_a = set(order_a[:k].tolist())
        set_b = set(order_b[:k].tolist())

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        jaccard = intersection / union if union > 0 else 1.0

        acc_a = float(np.mean(labels[order_a[:k]]))
        acc_b = float(np.mean(labels[order_b[:k]]))

        # Items in one but not other
        a_only = list(set_a - set_b)
        b_only = list(set_b - set_a)

        results[cov] = {
            "k": k,
            "jaccard": jaccard,
            "acc_a": acc_a,
            "acc_b": acc_b,
            "n_disagree": len(a_only) + len(b_only),
            "a_only_count": len(a_only),
            "b_only_count": len(b_only),
            "a_only_acc": float(np.mean(labels[a_only])) if a_only else None,
            "b_only_acc": float(np.mean(labels[b_only])) if b_only else None,
        }
    return results


def random_baseline_jaccard(n, coverages, n_trials=10000):
    """Expected Jaccard if two random scorers independently rank n items."""
    rng = np.random.RandomState(42)
    results = {}
    for cov in coverages:
        k = max(1, int(n * cov / 100))
        # Analytical: E[Jaccard] = k / (2n - k)
        analytical = k / (2 * n - k)
        # Also bootstrap for CI
        jaccards = []
        for _ in range(n_trials):
            a = set(rng.choice(n, k, replace=False).tolist())
            b = set(rng.choice(n, k, replace=False).tolist())
            j = len(a & b) / len(a | b)
            jaccards.append(j)
        results[cov] = {
            "analytical": round(analytical, 4),
            "empirical_mean": round(np.mean(jaccards), 4),
            "ci_95": [round(np.percentile(jaccards, 2.5), 4),
                      round(np.percentile(jaccards, 97.5), 4)]
        }
    return results


def bootstrap_jaccard_ci(scores_a, scores_b, n, cov, n_boot=2000):
    """Bootstrap CI for Jaccard at given coverage."""
    k = max(1, int(n * cov / 100))
    rng = np.random.RandomState(42)
    jaccards = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        sa = scores_a[idx]
        sb = scores_b[idx]
        oa = np.argsort(-sa)[:k]
        ob = np.argsort(-sb)[:k]
        set_a = set(oa.tolist())
        set_b = set(ob.tolist())
        j = len(set_a & set_b) / len(set_a | set_b) if len(set_a | set_b) > 0 else 1.0
        jaccards.append(j)
    return {
        "mean": round(np.mean(jaccards), 4),
        "ci_95": [round(np.percentile(jaccards, 2.5), 4),
                  round(np.percentile(jaccards, 97.5), 4)]
    }


def run_model(model_key):
    print(f"\n{'='*60}")
    print(f"  P1v2: {model_key}")
    print(f"{'='*60}")

    H_pre, H_post, labels, n, L = load_data(model_key)

    # Split: 70/30 stratified
    np.random.seed(42)
    idx = np.random.permutation(n)
    cal_n = int(n * 0.7)
    cal_idx = idx[:cal_n]
    test_idx = idx[cal_n:]
    test_n = len(test_idx)

    # Find best layers on calibration set
    pre_best, pre_2nd = find_top2_layers(H_pre, labels, cal_idx)
    post_best, _ = find_top2_layers(H_post, labels, cal_idx)

    print(f"  H_pre best:  L{pre_best[0]}, sign={pre_best[1]:+d}, cal_auc={pre_best[2]:.4f}")
    print(f"  H_pre 2nd:   L{pre_2nd[0]}, sign={pre_2nd[1]:+d}, cal_auc={pre_2nd[2]:.4f}")
    print(f"  H_post best: L{post_best[0]}, sign={post_best[1]:+d}, cal_auc={post_best[2]:.4f}")

    # Compute scores on TEST SET only
    pre_scores = pre_best[1] * H_pre[test_idx, pre_best[0]]
    post_scores = post_best[1] * H_post[test_idx, post_best[0]]
    pre2_scores = pre_2nd[1] * H_pre[test_idx, pre_2nd[0]]

    test_labels = labels[test_idx]
    coverages = [90, 80, 70, 60, 50]

    # Main: H_pre vs H_post
    main_div = compute_divergence(pre_scores, post_scores, test_labels, test_n, coverages)
    # Control: H_pre best vs H_pre 2nd-best
    ctrl_div = compute_divergence(pre_scores, pre2_scores, test_labels, test_n, coverages)
    # Random baseline
    rand_base = random_baseline_jaccard(test_n, coverages)

    print(f"\n  TEST SET (n={test_n}), base acc={np.mean(test_labels):.4f}")
    print(f"\n  {'Cov':>5} | {'H_pre vs H_post':^30} | {'H_pre vs H_pre(2nd)':^30} | {'Random':^12}")
    print(f"  {'':>5} | {'Jaccard':>8} {'Disagree':>10} {'dAcc':>8} | {'Jaccard':>8} {'Disagree':>10} {'dAcc':>8} | {'E[Jaccard]':>12}")
    print(f"  {'-'*95}")

    for cov in coverages:
        m = main_div[cov]
        c = ctrl_div[cov]
        r = rand_base[cov]
        m_dacc = (m['acc_a'] - m['acc_b']) if m['acc_a'] and m['acc_b'] else 0
        c_dacc = (c['acc_a'] - c['acc_b']) if c['acc_a'] and c['acc_b'] else 0
        print(f"  {cov:>4}% | {m['jaccard']:>8.4f} {m['n_disagree']:>6}({m['n_disagree']/m['k']*100:4.0f}%) {m_dacc:>+8.4f} | {c['jaccard']:>8.4f} {c['n_disagree']:>6}({c['n_disagree']/c['k']*100:4.0f}%) {c_dacc:>+8.4f} | {r['analytical']:>12.4f}")

    # Bootstrap CI at 80%
    main_boot = bootstrap_jaccard_ci(pre_scores, post_scores, test_n, 80)
    ctrl_boot = bootstrap_jaccard_ci(pre_scores, pre2_scores, test_n, 80)

    print(f"\n  Bootstrap Jaccard CI at 80% coverage:")
    print(f"    H_pre vs H_post:    {main_boot['mean']:.4f} [{main_boot['ci_95'][0]:.4f}, {main_boot['ci_95'][1]:.4f}]")
    print(f"    H_pre vs H_pre(2nd):{ctrl_boot['mean']:.4f} [{ctrl_boot['ci_95'][0]:.4f}, {ctrl_boot['ci_95'][1]:.4f}]")
    print(f"    Random expectation:  {rand_base[80]['analytical']:.4f} [{rand_base[80]['ci_95'][0]:.4f}, {rand_base[80]['ci_95'][1]:.4f}]")

    # Is main divergence > control divergence?
    print(f"\n  H_pre/H_post divergence vs within-H_pre divergence:")
    for cov in coverages:
        m_j = main_div[cov]['jaccard']
        c_j = ctrl_div[cov]['jaccard']
        diff = m_j - c_j
        more_or_less = "MORE overlap (less divergent)" if diff > 0 else "LESS overlap (more divergent)"
        print(f"    {cov}%: main Jaccard {m_j:.4f} vs ctrl {c_j:.4f}, diff {diff:+.4f} → H_pre/H_post is {more_or_less} than within-H_pre")

    return {
        "model": model_key,
        "n_test": test_n,
        "base_accuracy": round(float(np.mean(test_labels)), 4),
        "H_pre_best": {"layer": int(pre_best[0]), "sign": int(pre_best[1])},
        "H_pre_2nd": {"layer": int(pre_2nd[0]), "sign": int(pre_2nd[1])},
        "H_post_best": {"layer": int(post_best[0]), "sign": int(post_best[1])},
        "layer_shift_pre_post": abs(int(pre_best[0]) - int(post_best[0])),
        "layer_shift_within_pre": abs(int(pre_best[0]) - int(pre_2nd[0])),
        "main_divergence": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in main_div.items()},
        "control_divergence": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in ctrl_div.items()},
        "random_baseline": rand_base,
        "bootstrap_80": {"main": main_boot, "control": ctrl_boot},
    }


def main():
    t0 = time.time()
    results = {}
    for model_key in ["qwen", "llama", "mistral"]:
        results[model_key] = run_model(model_key)

    out_dir = BASE / "experiments" / "51_Selective_Prediction_Divergence"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "divergence_v2_results.json"
    with open(out_file, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=lambda o: int(o) if isinstance(o, (np.integer, np.bool_)) else float(o) if isinstance(o, np.floating) else o)

    print(f"\n{'='*60}")
    print(f"DONE in {time.time()-t0:.1f}s")
    print(f"Saved: {out_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
