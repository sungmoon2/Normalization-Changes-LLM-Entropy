"""
Incremental Utility Test: H_pre의 logit_std 통제 후 잔여 정보 검증
================================================================
GPT 13차+14차 양쪽 만장일치 마지막 필수 실험.

목적:
  "H_pre는 logit_std의 그림자에 불과한가?"
  → logit_std를 이미 알고 있을 때, H_pre를 추가하면 AUROC가 올라가는가?

설계 (GPT 14차 확장+표준 수렴):
  - 조건 1: Qwen Hard 500 (H_pre가 가장 강한 best-case)
  - 조건 2: Qwen MMLU 1000 (일반화 확인)
  - 5개 feature 조합 × LogisticRegression
  - 70/30 cal/test (기존 동일 split, seed=42)
  - delta-AUROC + paired bootstrap CI (1000 resamples)

GPU 불필요. CPU only. 예상 소요: 1-2분.
"""

import json
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

SEED = 42
PROJECT_ROOT = Path(__file__).parent.parent
BASE_DIR = PROJECT_ROOT / "PoT_Experiment_Entropy_Attention_Extraction_Experiment"
OUTPUT_DIR = BASE_DIR / "experiments" / "40_Incremental_Utility_Test"

# Data paths
DATA_PATHS = {
    "qwen_hard": {
        "path": BASE_DIR / "experiments" / "23_Normed_Difficulty_Analysis" / "EXP_20260213_113717_normed_hard" / "data" / "sample_results.json",
        "label": "Qwen Hard (L4-5)",
        "num_layers": 28,
    },
    "qwen_mmlu": {
        "path": BASE_DIR / "experiments" / "31_MMLU_Domain_Extension" / "EXP_20260219_053638_mmlu_qwen" / "data" / "sample_results.json",
        "label": "Qwen MMLU",
        "num_layers": 28,
    },
}

# Feature combinations (GPT 14차 설계)
FEATURE_SETS = {
    "logit_std_only": ["logit_std"],
    "H_pre_only": ["unnormed_entropy"],
    "logit_std+H_pre": ["logit_std", "unnormed_entropy"],
    "full_scale": ["logit_std", "h_norm", "length"],
    "full_scale+H_pre": ["logit_std", "h_norm", "length", "unnormed_entropy"],
}


def load_per_sample_data(config):
    """Load per-sample features from sample_results.json."""
    path = config["path"]
    num_layers = config["num_layers"]

    print(f"  Loading: {path.name} from {path.parent.parent.name}")
    with open(path) as f:
        samples = json.load(f)

    print(f"  Total samples: {len(samples)}")

    # Extract per-sample features at each layer
    records = []
    for s in samples:
        if "layer_data" not in s:
            continue

        is_correct = int(s.get("is_correct", False))
        num_tokens = s.get("num_tokens", 0)

        # For each layer, extract features
        layer_features = {}
        for l in range(num_layers):
            lk = str(l)
            if lk not in s["layer_data"]:
                continue
            ld = s["layer_data"][lk]
            layer_features[l] = {
                "unnormed_entropy": ld.get("unnormed_entropy", np.nan),
                "normed_entropy": ld.get("normed_entropy", np.nan),
                "h_norm": ld.get("h_norm", np.nan),
                "logit_std": ld.get("logit_std", np.nan),
                "logit_max": ld.get("logit_max", np.nan),
                "logit_margin": ld.get("logit_margin", np.nan),
                "wh_norm": ld.get("wh_norm", np.nan),
                "wh_rms": ld.get("wh_rms", np.nan),
            }

        records.append({
            "is_correct": is_correct,
            "num_tokens": num_tokens,
            "layer_features": layer_features,
        })

    print(f"  Valid records: {len(records)}")
    return records


def extract_best_layer_features(records, feature_name, num_layers, y, cal_idx):
    """Select best layer on cal set, return feature values for all samples."""
    # Try each layer on cal set
    best_layer = 0
    best_auroc = 0.0
    best_sign = 1

    for l in range(num_layers):
        vals = np.array([r["layer_features"].get(l, {}).get(feature_name, np.nan) for r in records])
        cal_vals = vals[cal_idx]
        cal_y = y[cal_idx]

        mask = ~np.isnan(cal_vals)
        if mask.sum() < 10 or len(np.unique(cal_y[mask])) < 2:
            continue

        try:
            auroc_pos = roc_auc_score(cal_y[mask], cal_vals[mask])
            auroc_neg = roc_auc_score(cal_y[mask], -cal_vals[mask])

            if max(auroc_pos, auroc_neg) > best_auroc:
                best_auroc = max(auroc_pos, auroc_neg)
                best_layer = l
                best_sign = 1 if auroc_pos >= auroc_neg else -1
        except ValueError:
            continue

    # Extract feature at best layer with sign calibration
    vals = np.array([r["layer_features"].get(best_layer, {}).get(feature_name, np.nan) for r in records])
    vals = vals * best_sign

    return vals, best_layer, best_sign, best_auroc


def build_feature_matrix(records, feature_names, num_layers, y, cal_idx):
    """Build feature matrix with best-layer selection per feature."""
    X = np.zeros((len(records), len(feature_names)))
    layer_info = {}

    for i, feat_name in enumerate(feature_names):
        if feat_name == "length":
            # Length is not layer-dependent
            vals = np.array([r["num_tokens"] for r in records], dtype=float)
            X[:, i] = vals
            layer_info[feat_name] = {"layer": "N/A", "sign": "+", "cal_auroc": "N/A"}
        else:
            vals, best_l, best_s, cal_auc = extract_best_layer_features(
                records, feat_name, num_layers, y, cal_idx
            )
            X[:, i] = vals
            layer_info[feat_name] = {
                "layer": best_l,
                "sign": "+" if best_s == 1 else "-",
                "cal_auroc": round(cal_auc, 4),
            }

    return X, layer_info


def paired_bootstrap_delta(y_true, scores_a, scores_b, n_bootstrap=1000, seed=42):
    """Compute paired bootstrap CI for AUROC difference."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    deltas = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            auc_a = roc_auc_score(y_b, scores_a[idx])
            auc_b = roc_auc_score(y_b, scores_b[idx])
            deltas.append(auc_b - auc_a)
        except ValueError:
            continue

    if len(deltas) < 100:
        return 0.0, -1.0, 1.0, 1.0

    deltas = np.array(deltas)
    mean_delta = float(np.mean(deltas))
    ci_lo = float(np.percentile(deltas, 2.5))
    ci_hi = float(np.percentile(deltas, 97.5))
    # p-value: proportion of deltas <= 0
    p_value = float(np.mean(deltas <= 0))

    return mean_delta, ci_lo, ci_hi, p_value


def run_condition(condition_key, config):
    """Run incremental utility test for one condition."""
    print(f"\n{'='*70}")
    print(f"Condition: {config['label']}")
    print(f"{'='*70}")

    records = load_per_sample_data(config)
    n = len(records)
    y = np.array([r["is_correct"] for r in records])
    num_layers = config["num_layers"]

    print(f"  Accuracy: {y.sum()}/{n} ({y.mean()*100:.1f}%)")

    # 70/30 stratified split (same as Phase 0)
    np.random.seed(SEED)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
    cal_idx, test_idx = next(sss.split(np.zeros(n), y))
    print(f"  Cal: {len(cal_idx)}, Test: {len(test_idx)}")

    results = {}

    # Run each feature set
    for fs_name, feat_names in FEATURE_SETS.items():
        print(f"\n  --- {fs_name}: {feat_names} ---")

        X, layer_info = build_feature_matrix(records, feat_names, num_layers, y, cal_idx)

        # Handle NaN
        nan_mask = np.any(np.isnan(X), axis=1)
        if nan_mask.sum() > 0:
            print(f"  Warning: {nan_mask.sum()} samples with NaN, filling with median")
            for col in range(X.shape[1]):
                col_vals = X[:, col]
                median_val = np.nanmedian(col_vals)
                col_vals[np.isnan(col_vals)] = median_val
                X[:, col] = col_vals

        # Scale features
        scaler = StandardScaler()
        X_cal = scaler.fit_transform(X[cal_idx])
        X_test = scaler.transform(X[test_idx])
        y_cal = y[cal_idx]
        y_test = y[test_idx]

        # Fit LogisticRegression
        lr = LogisticRegression(max_iter=1000, random_state=SEED, solver='lbfgs')
        lr.fit(X_cal, y_cal)

        # Predict
        proba_test = lr.predict_proba(X_test)[:, 1]

        # AUROC
        try:
            auroc = float(roc_auc_score(y_test, proba_test))
        except ValueError:
            auroc = 0.5

        results[fs_name] = {
            "features": feat_names,
            "layer_info": layer_info,
            "test_auroc": round(auroc, 6),
            "n_cal": len(cal_idx),
            "n_test": len(test_idx),
            "lr_coefs": {feat_names[i]: round(float(lr.coef_[0][i]), 6) for i in range(len(feat_names))},
        }

        print(f"  Test AUROC: {auroc:.4f}")
        for fn, info in layer_info.items():
            if info["layer"] != "N/A":
                print(f"    {fn}: L{info['layer']} (sign={info['sign']}, cal_auroc={info['cal_auroc']})")

    # Compute deltas with paired bootstrap
    print(f"\n  --- Delta Analysis ---")

    # Key comparisons
    comparisons = [
        ("logit_std+H_pre", "logit_std_only", "H_pre adds to logit_std?"),
        ("full_scale+H_pre", "full_scale", "H_pre adds to full scale baseline?"),
        ("H_pre_only", "logit_std_only", "H_pre vs logit_std standalone"),
    ]

    delta_results = {}
    for aug_key, base_key, question in comparisons:
        print(f"\n  {question}")
        print(f"    {base_key}: {results[base_key]['test_auroc']:.4f}")
        print(f"    {aug_key}: {results[aug_key]['test_auroc']:.4f}")

        raw_delta = results[aug_key]['test_auroc'] - results[base_key]['test_auroc']
        print(f"    Raw delta: {raw_delta:+.4f}")

        # Paired bootstrap
        # Need to recompute predictions for bootstrap
        X_base, _ = build_feature_matrix(records, FEATURE_SETS[base_key], num_layers, y, cal_idx)
        X_aug, _ = build_feature_matrix(records, FEATURE_SETS[aug_key], num_layers, y, cal_idx)

        # Handle NaN
        for X_tmp in [X_base, X_aug]:
            for col in range(X_tmp.shape[1]):
                col_vals = X_tmp[:, col]
                median_val = np.nanmedian(col_vals)
                col_vals[np.isnan(col_vals)] = median_val
                X_tmp[:, col] = col_vals

        scaler_b = StandardScaler().fit(X_base[cal_idx])
        scaler_a = StandardScaler().fit(X_aug[cal_idx])

        lr_b = LogisticRegression(max_iter=1000, random_state=SEED).fit(
            scaler_b.transform(X_base[cal_idx]), y[cal_idx])
        lr_a = LogisticRegression(max_iter=1000, random_state=SEED).fit(
            scaler_a.transform(X_aug[cal_idx]), y[cal_idx])

        scores_b = lr_b.predict_proba(scaler_b.transform(X_base[test_idx]))[:, 1]
        scores_a = lr_a.predict_proba(scaler_a.transform(X_aug[test_idx]))[:, 1]

        mean_d, ci_lo, ci_hi, p_val = paired_bootstrap_delta(y[test_idx], scores_b, scores_a)

        print(f"    Bootstrap delta: {mean_d:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        print(f"    p-value (delta<=0): {p_val:.4f}")

        sig = "YES" if p_val < 0.05 else "NO"
        print(f"    Significant (p<0.05): {sig}")

        delta_results[f"{aug_key}_vs_{base_key}"] = {
            "question": question,
            "base_auroc": results[base_key]['test_auroc'],
            "aug_auroc": results[aug_key]['test_auroc'],
            "raw_delta": round(raw_delta, 6),
            "bootstrap_mean_delta": round(mean_d, 6),
            "bootstrap_ci_95": [round(ci_lo, 6), round(ci_hi, 6)],
            "p_value": round(p_val, 6),
            "significant": sig,
        }

    return {
        "condition": config["label"],
        "n_samples": n,
        "accuracy": round(float(y.mean()), 4),
        "n_cal": len(cal_idx),
        "n_test": len(test_idx),
        "feature_set_results": results,
        "delta_comparisons": delta_results,
    }


def main():
    np.random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Incremental Utility Test")
    print("GPT 13+14차 만장일치 마지막 필수 실험")
    print("Q: H_pre가 logit_std 통제 후에도 정보를 남기는가?")
    print("=" * 70)

    all_results = {}
    for cond_key, config in DATA_PATHS.items():
        if not config["path"].exists():
            print(f"\n  SKIP: {config['path']} does not exist")
            continue
        result = run_condition(cond_key, config)
        all_results[cond_key] = result

    # Save results
    results_path = OUTPUT_DIR / "incremental_utility_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {results_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for cond_key, result in all_results.items():
        print(f"\n{result['condition']} (n={result['n_samples']}, acc={result['accuracy']*100:.1f}%)")
        print(f"  Feature Set AUROC:")
        for fs_name, fs_result in result['feature_set_results'].items():
            print(f"    {fs_name:25s}: {fs_result['test_auroc']:.4f}")

        print(f"\n  Delta Comparisons:")
        for comp_key, comp in result['delta_comparisons'].items():
            sig_mark = "*" if comp['significant'] == "YES" else ""
            print(f"    {comp['question']}")
            print(f"      delta={comp['raw_delta']:+.4f} "
                  f"[{comp['bootstrap_ci_95'][0]:+.4f}, {comp['bootstrap_ci_95'][1]:+.4f}] "
                  f"p={comp['p_value']:.4f}{sig_mark}")

    # Interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")

    for cond_key, result in all_results.items():
        key_delta = result['delta_comparisons'].get('logit_std+H_pre_vs_logit_std_only', {})
        full_delta = result['delta_comparisons'].get('full_scale+H_pre_vs_full_scale', {})

        print(f"\n{result['condition']}:")
        if key_delta:
            d = key_delta['raw_delta']
            sig = key_delta['significant']
            if abs(d) < 0.005 and sig == "NO":
                print(f"  -> H_pre adds LITTLE beyond logit_std (delta={d:+.4f}, not significant)")
                print(f"  -> Supports: 'H_pre is largely a compressed scale proxy'")
            elif d > 0 and sig == "YES":
                print(f"  -> H_pre adds SIGNIFICANT information beyond logit_std (delta={d:+.4f})")
                print(f"  -> Supports: 'H_pre contains residual non-scale information'")
            else:
                print(f"  -> H_pre effect is AMBIGUOUS (delta={d:+.4f}, sig={sig})")

    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print("Done.")


if __name__ == "__main__":
    main()
