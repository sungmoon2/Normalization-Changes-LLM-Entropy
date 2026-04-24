# -*- coding: utf-8 -*-
"""
Entropy-Lens Exact Reproduction: Analysis
==========================================
Analyzes the Llama-3.2-3B-Instruct results using:
  - k-NN k=3, 10-fold CV (matching Entropy-Lens protocol)
  - H_pre vs H_post vs scale profile comparison
  - Strategy classification (expansion/pruning)
  - Critical layer shift analysis
  - Scale covariate absorption test
  - Comparison with our existing 3-model results

This is the "exact reproduction" defense for Section 7.3.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy import stats
from datetime import datetime
from _paths import POT_DIR

SEED = 42
np.random.seed(SEED)

BASE = POT_DIR / "experiments"
EXP47 = BASE / "47_Entropy_Lens_Exact_Reproduction"
OUTPUT = EXP47 / "analysis"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Also load our existing 3-model data for comparison
EXP31 = BASE / "31_MMLU_Domain_Extension"
EXISTING_MODELS = {
    "qwen": {"path": EXP31 / "EXP_20260219_053638_mmlu_qwen/data/sample_results.json", "n_layers": 28},
    "llama8b": {"path": EXP31 / "EXP_20260219_171237_mmlu_llama/data/sample_results.json", "n_layers": 32},
    "mistral": {"path": EXP31 / "EXP_20260220_000610_mmlu_mistral/data/sample_results.json", "n_layers": 32},
}


def load_profiles(path, n_layers):
    """Load per-sample, per-layer profiles."""
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
            lk = str(l)
            if lk in ld:
                layer = ld[lk]
                h_pre[i, l] = layer["unnormed_entropy"]
                h_post[i, l] = layer["normed_entropy"]
                logit_std[i, l] = layer["logit_std"]
                h_norm[i, l] = layer["h_norm"]

    return h_pre, h_post, logit_std, h_norm, y


def cv_auroc(X, y, clf_type="knn", k=3, n_splits=10):
    """Stratified k-fold CV AUROC."""
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
            prob = clf.predict(X_test).astype(float)

        try:
            aurocs.append(roc_auc_score(y[test_idx], prob))
        except ValueError:
            aurocs.append(0.5)

    return np.mean(aurocs), np.std(aurocs), aurocs


def single_layer_auroc(X_col, y, n_splits=10):
    """Single-layer AUROC via 10-fold CV (simple threshold, no classifier)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    aurocs_pos = []
    aurocs_neg = []

    for train_idx, test_idx in skf.split(X_col, y):
        try:
            aurocs_pos.append(roc_auc_score(y[test_idx], X_col[test_idx]))
            aurocs_neg.append(roc_auc_score(y[test_idx], -X_col[test_idx]))
        except ValueError:
            aurocs_pos.append(0.5)
            aurocs_neg.append(0.5)

    mean_pos = np.mean(aurocs_pos)
    mean_neg = np.mean(aurocs_neg)

    if mean_pos >= mean_neg:
        return mean_pos, np.std(aurocs_pos), +1, aurocs_pos
    else:
        return mean_neg, np.std(aurocs_neg), -1, aurocs_neg


def classify_strategy(profile):
    """Expansion vs pruning classification based on entropy profile slope."""
    n_samples, n_layers = profile.shape
    x = np.arange(n_layers)
    labels = []
    slopes = []
    for i in range(n_samples):
        slope, _, _, _, _ = stats.linregress(x, profile[i])
        labels.append("expansion" if slope > 0 else "pruning")
        slopes.append(slope)
    return np.array(labels), np.array(slopes)


def analyze_model(name, path, n_layers, n_cv_splits=10):
    """Full analysis for one model."""
    print(f"\n{'='*70}")
    print(f"  {name.upper()} (layers={n_layers}, {n_cv_splits}-fold CV)")
    print(f"{'='*70}")

    h_pre, h_post, logit_std_arr, h_norm_arr, y = load_profiles(path, n_layers)
    n = len(y)
    n_correct = y.sum()
    acc = n_correct / n * 100
    print(f"  Samples: {n}, Correct: {n_correct} ({acc:.1f}%)")

    results = {
        "model": name,
        "n_samples": n,
        "n_correct": int(n_correct),
        "accuracy": round(acc, 2),
        "n_layers": n_layers,
        "n_cv_splits": n_cv_splits,
    }

    # ================================================================
    # 1. Strategy classification
    # ================================================================
    print("\n--- 1. Strategy Classification ---")
    labels_pre, slopes_pre = classify_strategy(h_pre)
    labels_post, slopes_post = classify_strategy(h_post)

    flips = (labels_pre != labels_post).sum()
    flip_rate = flips / n

    print(f"  H_pre:  expansion={sum(labels_pre == 'expansion')}, pruning={sum(labels_pre == 'pruning')}")
    print(f"  H_post: expansion={sum(labels_post == 'expansion')}, pruning={sum(labels_post == 'pruning')}")
    print(f"  Label flips: {flips}/{n} ({flip_rate*100:.1f}%)")

    results["strategy"] = {
        "h_pre_expansion": int(sum(labels_pre == "expansion")),
        "h_post_expansion": int(sum(labels_post == "expansion")),
        "label_flips": int(flips),
        "flip_rate": round(float(flip_rate), 4),
        "slope_correlation": round(float(np.corrcoef(slopes_pre, slopes_post)[0, 1]), 4),
    }

    # ================================================================
    # 2. Profile-based correctness AUROC
    # ================================================================
    print(f"\n--- 2. Profile-based Correctness AUROC ({n_cv_splits}-fold CV) ---")

    profile_tests = {
        "H_pre (k-NN k=3)": (h_pre, "knn"),
        "H_pre (LR)": (h_pre, "lr"),
        "H_post (k-NN k=3)": (h_post, "knn"),
        "H_post (LR)": (h_post, "lr"),
        "scale (LR)": (np.hstack([logit_std_arr, h_norm_arr]), "lr"),
    }

    profile_results = {}
    for pname, (X, clf_type) in profile_tests.items():
        mean_auc, std_auc, fold_aucs = cv_auroc(X, y, clf_type=clf_type, n_splits=n_cv_splits)
        print(f"  {pname:25s}: {mean_auc:.4f} +/- {std_auc:.4f}")
        profile_results[pname] = {
            "mean": round(mean_auc, 4),
            "std": round(std_auc, 4),
            "fold_aurocs": [round(a, 4) for a in fold_aucs],
        }

    results["profile_auroc"] = profile_results

    # ================================================================
    # 3. Scale covariate absorption
    # ================================================================
    print(f"\n--- 3. Scale Covariate Absorption ---")

    scale_only = np.hstack([logit_std_arr, h_norm_arr])
    scale_plus_hpre = np.hstack([logit_std_arr, h_norm_arr, h_pre])
    scale_plus_hpost = np.hstack([logit_std_arr, h_norm_arr, h_post])

    scale_auc, scale_std, _ = cv_auroc(scale_only, y, clf_type="lr", n_splits=n_cv_splits)
    sph_auc, sph_std, _ = cv_auroc(scale_plus_hpre, y, clf_type="lr", n_splits=n_cv_splits)
    spho_auc, spho_std, _ = cv_auroc(scale_plus_hpost, y, clf_type="lr", n_splits=n_cv_splits)

    print(f"  scale only:     {scale_auc:.4f} +/- {scale_std:.4f}")
    print(f"  scale + H_pre:  {sph_auc:.4f} +/- {sph_std:.4f}  (delta={sph_auc-scale_auc:+.4f})")
    print(f"  scale + H_post: {spho_auc:.4f} +/- {spho_std:.4f}  (delta={spho_auc-scale_auc:+.4f})")

    results["scale_absorption"] = {
        "scale_only": {"mean": round(scale_auc, 4), "std": round(scale_std, 4)},
        "scale_plus_hpre": {"mean": round(sph_auc, 4), "std": round(sph_std, 4), "delta": round(sph_auc - scale_auc, 4)},
        "scale_plus_hpost": {"mean": round(spho_auc, 4), "std": round(spho_std, 4), "delta": round(spho_auc - scale_auc, 4)},
    }

    # ================================================================
    # 4. Critical layer shift (single-layer best)
    # ================================================================
    print(f"\n--- 4. Critical Layer Shift ---")

    best_layers = {}
    for metric_name, arr in [("H_pre", h_pre), ("H_post", h_post), ("logit_std", logit_std_arr), ("h_norm", h_norm_arr)]:
        best_auc = 0
        best_l = 0
        best_sign = +1
        layer_results = []

        for l in range(n_layers):
            auc, std, sign, _ = single_layer_auroc(arr[:, l], y, n_splits=n_cv_splits)
            layer_results.append({"layer": l, "auroc": round(auc, 4), "sign": sign})
            if auc > best_auc:
                best_auc = auc
                best_l = l
                best_sign = sign

        print(f"  {metric_name:10s}: best L{best_l} (sign={'+' if best_sign > 0 else '-'}, AUROC={best_auc:.4f})")
        best_layers[metric_name] = {
            "best_layer": best_l,
            "best_sign": best_sign,
            "best_auroc": round(best_auc, 4),
            "per_layer": layer_results,
        }

    hpre_best = best_layers["H_pre"]["best_layer"]
    hpost_best = best_layers["H_post"]["best_layer"]
    shift = abs(hpre_best - hpost_best)
    print(f"  Critical layer shift: |L{hpre_best} - L{hpost_best}| = {shift} layers")

    results["critical_layer"] = best_layers
    results["critical_layer_shift"] = shift

    return results


def main():
    print("=" * 70)
    print("  ENTROPY-LENS EXACT REPRODUCTION ANALYSIS")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    all_results = {}

    # ================================================================
    # Entropy-Lens original models (exact reproduction)
    # ================================================================
    exp_dirs = sorted(EXP47.glob("EXP_*"))
    if not exp_dirs:
        print("ERROR: No experiment directory found in exp47. Run the experiment first.")
        return

    for exp_dir in exp_dirs:
        data_path = exp_dir / "data" / "sample_results.json"
        if not data_path.exists():
            continue
        summary_path = exp_dir / "summary.json"
        if summary_path.exists():
            summary = json.load(open(summary_path))
            n_layers = summary.get("num_layers", 32)
            model_short = summary.get("model_short", exp_dir.name.split("_")[-1])
            protocol = summary.get("protocol", "")
        else:
            n_layers = 32
            model_short = exp_dir.name.split("_")[-1]
            protocol = ""

        # Only use v3 (final protocol-matched version)
        if "v3" not in model_short:
            print(f"  Skipping {exp_dir.name} (not v3)")
            continue

        tag = f"{model_short}_EL"
        print(f"\n{tag} data: {data_path}")
        all_results[tag] = analyze_model(tag, data_path, n_layers, n_cv_splits=10)

    # ================================================================
    # Existing 3 models (for comparison, using same 10-fold CV)
    # ================================================================
    for model_name, config in EXISTING_MODELS.items():
        if config["path"].exists():
            all_results[model_name] = analyze_model(
                model_name, config["path"], config["n_layers"], n_cv_splits=10
            )

    # ================================================================
    # Cross-model comparison table
    # ================================================================
    print(f"\n{'='*70}")
    print("  CROSS-MODEL COMPARISON (10-fold CV)")
    print(f"{'='*70}")

    header = f"{'Model':15s} {'Acc':>6s} | {'H_pre kNN':>10s} {'H_pre LR':>10s} {'H_post kNN':>10s} {'H_post LR':>10s} {'Scale LR':>10s} | {'Shift':>5s} {'Flip%':>6s}"
    print(header)
    print("-" * len(header))

    for name, r in all_results.items():
        pa = r["profile_auroc"]
        shift = r.get("critical_layer_shift", "?")
        flip = r["strategy"]["flip_rate"] * 100

        print(f"{name:15s} {r['accuracy']:5.1f}% | "
              f"{pa.get('H_pre (k-NN k=3)', {}).get('mean', 0):10.4f} "
              f"{pa.get('H_pre (LR)', {}).get('mean', 0):10.4f} "
              f"{pa.get('H_post (k-NN k=3)', {}).get('mean', 0):10.4f} "
              f"{pa.get('H_post (LR)', {}).get('mean', 0):10.4f} "
              f"{pa.get('scale (LR)', {}).get('mean', 0):10.4f} | "
              f"{shift:>5} {flip:5.1f}%")

    # ================================================================
    # Save
    # ================================================================
    output_path = OUTPUT / "exact_reproduction_results.json"
    json.dump(all_results, open(output_path, "w"), indent=2)
    print(f"\nResults saved to: {output_path}")

    # Summary for paper
    print(f"\n{'='*70}")
    print("  KEY FINDINGS FOR PAPER")
    print(f"{'='*70}")

    for name, r in all_results.items():
        if "_EL" not in name:
            continue
        pa = r["profile_auroc"]
        sa = r["scale_absorption"]
        cl = r["critical_layer"]

        print(f"\n  {name} (Entropy-Lens protocol: 1-shot, 32 tokens, k-NN k=3, 10-fold CV)")
        print(f"  Accuracy: {r['accuracy']}%")
        print(f"  Profile AUROC:")
        print(f"    H_pre  k-NN: {pa['H_pre (k-NN k=3)']['mean']:.4f} +/- {pa['H_pre (k-NN k=3)']['std']:.4f}")
        print(f"    H_post k-NN: {pa['H_post (k-NN k=3)']['mean']:.4f} +/- {pa['H_post (k-NN k=3)']['std']:.4f}")
        print(f"    scale  LR:   {pa['scale (LR)']['mean']:.4f} +/- {pa['scale (LR)']['std']:.4f}")
        print(f"  Scale absorption: +H_pre={sa['scale_plus_hpre']['delta']:+.4f}, +H_post={sa['scale_plus_hpost']['delta']:+.4f}")
        print(f"  Critical layer shift: {r['critical_layer_shift']} layers (H_pre=L{cl['H_pre']['best_layer']}, H_post=L{cl['H_post']['best_layer']})")
        print(f"  Strategy flip rate: {r['strategy']['flip_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
