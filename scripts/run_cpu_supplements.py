"""
CPU Supplement Experiments: O2 (k-NN nested tuning) + S1 (sampled paired test)
No GPU required. Uses exp31 MMLU generation-average data.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
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

def load_profiles(model_key):
    """Load per-layer features from exp31 sample_results.json."""
    cfg = MODELS[model_key]
    path = EXP31 / cfg["dir"] / "data" / "sample_results.json"
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    n = len(data)
    L = cfg["n_layers"]

    H_pre = np.zeros((n, L))
    H_post = np.zeros((n, L))
    h_norm = np.zeros((n, L))
    logit_std = np.zeros((n, L))
    labels = np.zeros(n, dtype=int)

    for i, sample in enumerate(data):
        labels[i] = 1 if sample["is_correct"] else 0
        ld = sample["layer_data"]
        for j in range(L):
            layer = ld[str(j)]
            H_pre[i, j] = layer["unnormed_entropy"]
            H_post[i, j] = layer["normed_entropy"]
            h_norm[i, j] = layer["h_norm"]
            logit_std[i, j] = layer["logit_std"]

    return {
        "H_pre": H_pre, "H_post": H_post,
        "h_norm": h_norm, "logit_std": logit_std,
        "labels": labels, "n": n, "L": L
    }


def run_knn_sweep(model_key):
    """O2: k-NN nested tuning — sweep k and distance metric."""
    print(f"\n[O2] k-NN sweep for {model_key}...")
    data = load_profiles(model_key)
    labels = data["labels"]

    profiles = {
        "H_pre": data["H_pre"],
        "H_post": data["H_post"],
        "h_norm": data["h_norm"],
        "logit_std": data["logit_std"],
    }

    k_values = [1, 3, 5, 7, 11, 15, 21]
    metrics = ["euclidean", "manhattan"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    for prof_name, X in profiles.items():
        results[prof_name] = {}
        for metric in metrics:
            results[prof_name][metric] = {}
            for k in k_values:
                fold_aucs = []
                for train_idx, test_idx in skf.split(X, labels):
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X[train_idx])
                    X_test = scaler.transform(X[test_idx])

                    clf = KNeighborsClassifier(n_neighbors=k, metric=metric)
                    clf.fit(X_train, labels[train_idx])
                    probs = clf.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(labels[test_idx], probs)
                    fold_aucs.append(auc)

                mean_auc = np.mean(fold_aucs)
                std_auc = np.std(fold_aucs)
                results[prof_name][metric][k] = {
                    "mean": round(mean_auc, 4),
                    "std": round(std_auc, 4),
                    "folds": [round(x, 4) for x in fold_aucs]
                }

    # Find best k for each profile
    summary = {}
    for prof_name in profiles:
        best_k = None
        best_auc = 0
        best_metric = None
        for metric in metrics:
            for k in k_values:
                auc = results[prof_name][metric][k]["mean"]
                if auc > best_auc:
                    best_auc = auc
                    best_k = k
                    best_metric = metric
        summary[prof_name] = {
            "best_k": best_k,
            "best_metric": best_metric,
            "best_auroc": best_auc,
            "k3_euclidean": results[prof_name]["euclidean"][3]["mean"],
            "lr_auroc": None  # filled below
        }

    # Compare with LR on same profiles
    for prof_name, X in profiles.items():
        fold_aucs = []
        for train_idx, test_idx in skf.split(X, labels):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            clf = LogisticRegression(max_iter=1000, solver='lbfgs')
            clf.fit(X_train, labels[train_idx])
            probs = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(labels[test_idx], probs)
            fold_aucs.append(auc)
        summary[prof_name]["lr_auroc"] = round(np.mean(fold_aucs), 4)

    # Print summary
    print(f"  {'Profile':<12} {'k=3 euc':>10} {'Best k-NN':>10} {'Best k,metric':>15} {'LR':>10} {'LR-bestKNN':>12}")
    for prof_name in profiles:
        s = summary[prof_name]
        delta = s["lr_auroc"] - s["best_auroc"]
        best_label = "k={},{}".format(s["best_k"], s["best_metric"][:3])
        print(f"  {prof_name:<12} {s['k3_euclidean']:>10.4f} {s['best_auroc']:>10.4f} {best_label:>15} {s['lr_auroc']:>10.4f} {delta:>+12.4f}")

    return {"sweep": results, "summary": summary}


def run_sampled_paired_test(model_key):
    """S1: Sampled Table 9 paired fold-wise test (scale vs H_pre)."""
    print(f"\n[S1] Sampled paired test for {model_key}...")
    data = load_profiles(model_key)
    labels = data["labels"]
    L = data["L"]

    # Build profiles
    profiles = {
        "H_pre": data["H_pre"],
        "H_post": data["H_post"],
        "h_norm": data["h_norm"],
        "logit_std": data["logit_std"],
        "scale_combined": np.hstack([data["logit_std"], data["h_norm"]]),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Get per-fold AUROCs for all profiles
    fold_aucs = {name: [] for name in profiles}
    for train_idx, test_idx in skf.split(profiles["H_pre"], labels):
        for name, X in profiles.items():
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            clf = LogisticRegression(max_iter=1000, solver='lbfgs')
            clf.fit(X_train, labels[train_idx])
            probs = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(labels[test_idx], probs)
            fold_aucs[name].append(auc)

    # Paired comparisons
    comparisons = [
        ("h_norm", "H_pre"),
        ("logit_std", "H_pre"),
        ("scale_combined", "H_pre"),
        ("scale_combined", "H_post"),
    ]

    paired = {}
    for a, b in comparisons:
        diffs = [fold_aucs[a][i] - fold_aucs[b][i] for i in range(5)]
        mean_diff = np.mean(diffs)
        # Bootstrap CI on paired differences
        rng = np.random.RandomState(42)
        boot_means = []
        for _ in range(10000):
            idx = rng.choice(5, 5, replace=True)
            boot_means.append(np.mean([diffs[i] for i in idx]))
        ci_lo = np.percentile(boot_means, 2.5)
        ci_hi = np.percentile(boot_means, 97.5)
        excludes_zero = (ci_lo > 0) or (ci_hi < 0)

        key = f"{a}_vs_{b}"
        paired[key] = {
            "mean_diff": round(mean_diff, 4),
            "ci": [round(ci_lo, 4), round(ci_hi, 4)],
            "ci_excludes_zero": excludes_zero,
            "fold_diffs": [round(d, 4) for d in diffs]
        }
        sig = "YES" if excludes_zero else "no"
        print(f"  {key:<25} diff={mean_diff:+.4f} CI=[{ci_lo:.4f}, {ci_hi:.4f}] sig={sig}")

    # Also report absolute fold AUROCs
    fold_summary = {}
    for name in profiles:
        fold_summary[name] = {
            "mean": round(np.mean(fold_aucs[name]), 4),
            "std": round(np.std(fold_aucs[name]), 4),
            "folds": [round(x, 4) for x in fold_aucs[name]]
        }

    return {"paired": paired, "fold_aucs": fold_summary}


def main():
    t0 = time.time()
    all_results = {}

    for model_key in ["qwen", "llama", "mistral"]:
        print(f"\n{'='*60}")
        print(f"  Model: {model_key}")
        print(f"{'='*60}")

        all_results[model_key] = {}

        # O2: k-NN sweep
        knn_results = run_knn_sweep(model_key)
        all_results[model_key]["knn_sweep"] = knn_results

        # S1: Sampled paired test
        paired_results = run_sampled_paired_test(model_key)
        all_results[model_key]["sampled_paired"] = paired_results

    # Save results
    out_dir = BASE / "experiments" / "50_CPU_Supplements"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "cpu_supplements_results.json"

    def convert(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(out_file, "w", encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=convert)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.1f}s")
    print(f"Saved: {out_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
