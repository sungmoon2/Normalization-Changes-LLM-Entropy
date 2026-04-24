"""
Entropy-Lens Re-evaluation: H_pre vs H_post vs Scale Profiles

Purpose: Show that Entropy-Lens-style conclusions change depending on
         whether H_pre or H_post profiles are used, and that H_pre's
         advantage is absorbed by scale covariates.

Entropy-Lens (Ali et al., 2025) uses layerwise entropy profiles to:
  1. Classify decision strategies (expansion vs pruning)
  2. Discriminate correctness via profile-based classifiers (k-NN)

We replicate this setup with:
  - H_pre profiles (unnormed_entropy per layer)
  - H_post profiles (normed_entropy per layer)
  - Scale profiles (logit_std + h_norm per layer)
  - Classifiers: k-NN (original) + LR (matched)

Key outputs:
  - Strategy label flip rate (H_pre vs H_post)
  - Profile-based AUROC comparison
  - Scale covariate absorption test
  - Critical layer shift analysis

Data: exp31 MMLU (3 models x 1000 samples, generation-average)
GPU: Not required (CPU only, existing data)
"""

import json
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from scipy import stats
from _paths import POT_DIR

SEED = 42
np.random.seed(SEED)

BASE = POT_DIR / "experiments"
EXP31 = BASE / "31_MMLU_Domain_Extension"
OUTPUT = BASE / "45_Entropy_Lens_Reeval"
OUTPUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "qwen": {
        "path": EXP31 / "EXP_20260219_053638_mmlu_qwen/data/sample_results.json",
        "n_layers": 28,
    },
    "llama": {
        "path": EXP31 / "EXP_20260219_171237_mmlu_llama/data/sample_results.json",
        "n_layers": 32,
    },
    "mistral": {
        "path": EXP31 / "EXP_20260220_000610_mmlu_mistral/data/sample_results.json",
        "n_layers": 32,
    },
}


def load_profiles(path, n_layers):
    """Load per-sample, per-layer profiles from exp31 data."""
    data = json.load(open(path))
    n = len(data)

    h_pre = np.zeros((n, n_layers))
    h_post = np.zeros((n, n_layers))
    logit_std = np.zeros((n, n_layers))
    h_norm = np.zeros((n, n_layers))
    y = np.zeros(n, dtype=int)

    for i, s in enumerate(data):
        y[i] = 1 if s["is_correct"] else 0
        ld = s["layer_data"]
        for l in range(n_layers):
            layer = ld[str(l)]
            h_pre[i, l] = layer["unnormed_entropy"]
            h_post[i, l] = layer["normed_entropy"]
            logit_std[i, l] = layer["logit_std"]
            h_norm[i, l] = layer["h_norm"]

    return h_pre, h_post, logit_std, h_norm, y


def classify_strategy(profile):
    """
    Entropy-Lens strategy classification.
    Expansion: entropy trend increases across layers (positive slope).
    Pruning: entropy trend decreases (negative slope).
    Based on linear regression slope of entropy vs layer index.
    """
    n_samples, n_layers = profile.shape
    x = np.arange(n_layers)
    labels = []
    slopes = []
    for i in range(n_samples):
        slope, _, _, _, _ = stats.linregress(x, profile[i])
        labels.append("expansion" if slope > 0 else "pruning")
        slopes.append(slope)
    return np.array(labels), np.array(slopes)


def cv_auroc(X, y, clf_type="knn", k=3, n_splits=5):
    """Stratified 5-fold CV AUROC for profile-based classification."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    aurocs = []

    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        if clf_type == "knn":
            clf = KNeighborsClassifier(n_neighbors=k)
        else:
            clf = LogisticRegression(max_iter=1000, random_state=SEED)

        clf.fit(X_train, y[train_idx])

        if hasattr(clf, "predict_proba"):
            prob = clf.predict_proba(X_test)[:, 1]
        else:
            prob = clf.predict(X_test)

        try:
            aurocs.append(roc_auc_score(y[test_idx], prob))
        except ValueError:
            aurocs.append(0.5)

    return np.mean(aurocs), np.std(aurocs)


def run_model(model_name, config):
    """Run full Entropy-Lens re-evaluation for one model."""
    print(f"\n{'='*60}")
    print(f"  {model_name.upper()} (n_layers={config['n_layers']})")
    print(f"{'='*60}")

    h_pre, h_post, logit_std, h_norm, y = load_profiles(
        config["path"], config["n_layers"]
    )
    n = len(y)
    n_correct = y.sum()
    print(f"  Samples: {n}, Correct: {n_correct} ({n_correct/n*100:.1f}%)")

    results = {
        "model": model_name,
        "n_samples": n,
        "n_correct": int(n_correct),
        "accuracy": round(float(n_correct / n), 4),
        "n_layers": config["n_layers"],
    }

    # ============================================================
    # 1. Strategy classification: expansion vs pruning
    # ============================================================
    print("\n--- 1. Strategy Classification ---")

    labels_pre, slopes_pre = classify_strategy(h_pre)
    labels_post, slopes_post = classify_strategy(h_post)

    n_expansion_pre = (labels_pre == "expansion").sum()
    n_expansion_post = (labels_post == "expansion").sum()

    # Flip rate: samples that change strategy label
    flips = (labels_pre != labels_post).sum()
    flip_rate = flips / n

    print(f"  H_pre:  expansion={n_expansion_pre}, pruning={n-n_expansion_pre}")
    print(f"  H_post: expansion={n_expansion_post}, pruning={n-n_expansion_post}")
    print(f"  Label flips: {flips}/{n} ({flip_rate*100:.1f}%)")

    # Flip breakdown by correctness
    correct_mask = y == 1
    flips_correct = ((labels_pre != labels_post) & correct_mask).sum()
    flips_incorrect = ((labels_pre != labels_post) & ~correct_mask).sum()

    results["strategy"] = {
        "h_pre_expansion": int(n_expansion_pre),
        "h_pre_pruning": int(n - n_expansion_pre),
        "h_post_expansion": int(n_expansion_post),
        "h_post_pruning": int(n - n_expansion_post),
        "label_flips": int(flips),
        "flip_rate": round(float(flip_rate), 4),
        "flips_correct": int(flips_correct),
        "flips_incorrect": int(flips_incorrect),
        "slope_correlation": round(float(np.corrcoef(slopes_pre, slopes_post)[0, 1]), 4),
    }

    # ============================================================
    # 2. Profile-based correctness discrimination (AUROC)
    # ============================================================
    print("\n--- 2. Profile-based Correctness AUROC (5-fold CV) ---")

    profile_configs = {
        "H_pre (k-NN k=3)": (h_pre, "knn"),
        "H_pre (LR)": (h_pre, "lr"),
        "H_post (k-NN k=3)": (h_post, "knn"),
        "H_post (LR)": (h_post, "lr"),
        "logit_std (k-NN k=3)": (logit_std, "knn"),
        "logit_std (LR)": (logit_std, "lr"),
        "h_norm (k-NN k=3)": (h_norm, "knn"),
        "h_norm (LR)": (h_norm, "lr"),
        "scale (logit_std+h_norm, LR)": (np.hstack([logit_std, h_norm]), "lr"),
    }

    auroc_results = {}
    for name, (X, clf_type) in profile_configs.items():
        mean_auc, std_auc = cv_auroc(X, y, clf_type=clf_type)
        print(f"  {name:35s}: {mean_auc:.4f} +/- {std_auc:.4f}")
        auroc_results[name] = {
            "mean": round(float(mean_auc), 4),
            "std": round(float(std_auc), 4),
        }

    results["profile_auroc"] = auroc_results

    # ============================================================
    # 3. Scale covariate absorption test
    # ============================================================
    print("\n--- 3. Scale Covariate Absorption (LR, 5-fold CV) ---")

    absorption_configs = {
        "H_pre only": h_pre,
        "H_pre + scale": np.hstack([h_pre, logit_std, h_norm]),
        "H_post only": h_post,
        "H_post + scale": np.hstack([h_post, logit_std, h_norm]),
        "scale only": np.hstack([logit_std, h_norm]),
    }

    absorption_results = {}
    for name, X in absorption_configs.items():
        mean_auc, std_auc = cv_auroc(X, y, clf_type="lr")
        print(f"  {name:25s}: {mean_auc:.4f} +/- {std_auc:.4f}")
        absorption_results[name] = {
            "mean": round(float(mean_auc), 4),
            "std": round(float(std_auc), 4),
        }

    # Key metric: does adding scale to H_pre change anything?
    hpre_only = absorption_results["H_pre only"]["mean"]
    hpre_scale = absorption_results["H_pre + scale"]["mean"]
    scale_only = absorption_results["scale only"]["mean"]
    hpost_only = absorption_results["H_post only"]["mean"]

    print(f"\n  H_pre advantage over scale: {hpre_only - scale_only:+.4f}")
    print(f"  H_pre + scale vs scale only: {hpre_scale - scale_only:+.4f}")
    print(f"  H_post advantage over scale: {hpost_only - scale_only:+.4f}")

    results["absorption"] = absorption_results
    results["absorption_summary"] = {
        "hpre_vs_scale": round(float(hpre_only - scale_only), 4),
        "hpre_plus_scale_vs_scale": round(float(hpre_scale - scale_only), 4),
        "hpost_vs_scale": round(float(hpost_only - scale_only), 4),
    }

    # ============================================================
    # 4. Critical layer analysis
    # ============================================================
    print("\n--- 4. Per-Layer Single-Feature AUROC ---")

    n_layers = config["n_layers"]

    # 70/30 held-out split (same protocol as main paper)
    np.random.seed(SEED)
    perm = np.random.permutation(n)
    cal_idx, test_idx = perm[: int(0.7 * n)], perm[int(0.7 * n) :]

    best_layers = {}
    for metric_name, X in [("H_pre", h_pre), ("H_post", h_post), ("logit_std", logit_std), ("h_norm", h_norm)]:
        best_l, best_s, best_cal = -1, 1, 0.5
        for l in range(n_layers):
            for sign in [1, -1]:
                auc = roc_auc_score(y[cal_idx], sign * X[cal_idx, l])
                if auc > best_cal:
                    best_cal, best_l, best_s = auc, l, sign
        test_auc = roc_auc_score(y[test_idx], best_s * X[test_idx, best_l])
        print(f"  {metric_name:10s}: best=L{best_l} sign={best_s:+d}, cal={best_cal:.4f}, test={test_auc:.4f}")
        best_layers[metric_name] = {
            "best_layer": int(best_l),
            "sign": int(best_s),
            "cal_auroc": round(float(best_cal), 4),
            "test_auroc": round(float(test_auc), 4),
        }

    # Critical layer shift
    hpre_layer = best_layers["H_pre"]["best_layer"]
    hpost_layer = best_layers["H_post"]["best_layer"]
    layer_shift = abs(hpre_layer - hpost_layer)
    print(f"\n  Critical layer shift: H_pre=L{hpre_layer} vs H_post=L{hpost_layer} (shift={layer_shift})")

    results["critical_layer"] = best_layers
    results["critical_layer_shift"] = int(layer_shift)

    # ============================================================
    # 5. Profile correlation analysis
    # ============================================================
    print("\n--- 5. Profile Correlation ---")

    # Mean profile correlation across samples
    corrs = [np.corrcoef(h_pre[i], h_post[i])[0, 1] for i in range(n)]
    mean_corr = np.nanmean(corrs)
    print(f"  Mean H_pre-H_post profile correlation: {mean_corr:.4f}")

    corrs_scale = [np.corrcoef(h_pre[i], logit_std[i])[0, 1] for i in range(n)]
    mean_corr_scale = np.nanmean(corrs_scale)
    print(f"  Mean H_pre-logit_std profile correlation: {mean_corr_scale:.4f}")

    results["profile_correlation"] = {
        "hpre_hpost_mean": round(float(mean_corr), 4),
        "hpre_logit_std_mean": round(float(mean_corr_scale), 4),
    }

    return results


def main():
    print("=" * 60)
    print("ENTROPY-LENS RE-EVALUATION")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)

    all_results = {}
    for model_name, config in MODELS.items():
        all_results[model_name] = run_model(model_name, config)

    # ============================================================
    # Cross-model summary
    # ============================================================
    print("\n" + "=" * 60)
    print("CROSS-MODEL SUMMARY")
    print("=" * 60)

    print("\n--- Strategy Flip Rates ---")
    for m in MODELS:
        r = all_results[m]["strategy"]
        print(f"  {m:8s}: {r['flip_rate']*100:.1f}% ({r['label_flips']}/{all_results[m]['n_samples']})")

    print("\n--- Profile AUROC (LR, 5-fold CV) ---")
    print(f"  {'Model':8s} | {'H_pre':>8s} | {'H_post':>8s} | {'scale':>8s} | {'H_pre-scale':>12s} | {'H_post-scale':>12s}")
    for m in MODELS:
        r = all_results[m]["profile_auroc"]
        s = all_results[m]["absorption_summary"]
        print(f"  {m:8s} | {r['H_pre (LR)']['mean']:8.4f} | {r['H_post (LR)']['mean']:8.4f} | {r['scale (logit_std+h_norm, LR)']['mean']:8.4f} | {s['hpre_vs_scale']:+12.4f} | {s['hpost_vs_scale']:+12.4f}")

    print("\n--- Critical Layer Shifts ---")
    for m in MODELS:
        cl = all_results[m]["critical_layer"]
        shift = all_results[m]["critical_layer_shift"]
        print(f"  {m:8s}: H_pre=L{cl['H_pre']['best_layer']} vs H_post=L{cl['H_post']['best_layer']} (shift={shift})")

    print("\n--- Key Finding: Does H_pre advantage survive scale control? ---")
    for m in MODELS:
        ab = all_results[m]["absorption"]
        hpre = ab["H_pre only"]["mean"]
        hpre_s = ab["H_pre + scale"]["mean"]
        scale = ab["scale only"]["mean"]
        hpost = ab["H_post only"]["mean"]
        hpost_s = ab["H_post + scale"]["mean"]
        delta_hpre = hpre_s - scale
        delta_hpost = hpost_s - scale
        print(f"  {m:8s}: scale={scale:.4f}, +H_pre={delta_hpre:+.4f}, +H_post={delta_hpost:+.4f}")

    # Save results
    all_results["_metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "Entropy-Lens Re-evaluation",
        "data_source": "exp31 MMLU (generation-average)",
        "protocol": "5-fold stratified CV (seed=42)",
    }

    out_path = OUTPUT / "entropy_lens_reeval_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")
    print("DONE.")


if __name__ == "__main__":
    main()
