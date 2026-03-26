"""
Phase 0b: Norm-Binned Observational Control + Multi-Layer Profile Baseline
==========================================================================
GPT 12차 수렴 권고 기반.

목적:
  1. Norm-bin stratification: h_norm 분위수 bin 안에서 AUROC 재계산
     → "scale(h_norm)을 통제하면 H_pre 신호가 얼마나 남는가?"
  2. Logit_std-bin stratification: 같은 분석을 logit_std로
  3. Length-bin stratification: 같은 분석을 num_tokens로
  4. Multi-layer profile baseline: L개 레이어의 metric을 profile로 → AUROC
  5. Residual analysis: H_pre에서 logit_std 효과를 residualize한 뒤 AUROC

원칙: 기존 Phase 0의 70/30 split 재사용 (동일 test set).
"""

import json
import os
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict

SEED = 42
N_BINS = 5  # quintile bins
BASE_DIR = Path(__file__).parent.parent / "PoT_Experiment_Entropy_Attention_Extraction_Experiment"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
OUTPUT_DIR = EXPERIMENTS_DIR / "32_Phase0_CalTest_Baselines"

# Conditions with full metrics (8 fields)
CONDITIONS = {
    "mmlu_qwen": {
        "path": "31_MMLU_Domain_Extension/EXP_20260219_053638_mmlu_qwen/data/sample_results.json",
        "num_layers": 28,
    },
    "mmlu_llama": {
        "path": "31_MMLU_Domain_Extension/EXP_20260219_171237_mmlu_llama/data/sample_results.json",
        "num_layers": 32,
    },
    "mmlu_mistral": {
        "path": "31_MMLU_Domain_Extension/EXP_20260220_000610_mmlu_mistral/data/sample_results.json",
        "num_layers": 32,
    },
    "qwen_hard": {
        "path": "23_Normed_Difficulty_Analysis/EXP_20260213_113717_normed_hard/data/sample_results.json",
        "num_layers": 28,
    },
}

METRICS = ["unnormed_entropy", "normed_entropy", "h_norm", "wh_norm",
           "wh_rms", "logit_std", "logit_max", "logit_margin"]


def safe_auroc(y, scores):
    mask = ~np.isnan(scores)
    y_m, s_m = y[mask], scores[mask]
    if len(np.unique(y_m)) < 2 or len(y_m) < 10:
        return 0.5
    try:
        return float(roc_auc_score(y_m, s_m))
    except ValueError:
        return 0.5


def load_data(config):
    filepath = EXPERIMENTS_DIR / config["path"]
    with open(filepath, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if isinstance(samples, dict) and "results" in samples:
        samples = samples["results"]

    num_layers = config["num_layers"]
    labels, num_tokens_list = [], []
    metric_arrays = {m: [] for m in METRICS}
    valid = []

    for sample in samples:
        if "layer_data" not in sample or "is_correct" not in sample:
            continue
        ld = sample["layer_data"]
        layer_keys = [str(l) for l in range(num_layers)]
        if not all(k in ld for k in layer_keys):
            continue

        ok = True
        row = {m: [] for m in METRICS}
        for lk in layer_keys:
            d = ld[lk]
            for m in METRICS:
                if m not in d:
                    ok = False
                    break
                v = d[m]
                if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                    v = np.nan
                row[m].append(v)
            if not ok:
                break
        if not ok:
            continue

        labels.append(1 if sample["is_correct"] else 0)
        for m in METRICS:
            metric_arrays[m].append(row[m])
        num_tokens_list.append(sample.get("num_tokens", None))
        valid.append(True)

    labels = np.array(labels)
    for m in METRICS:
        metric_arrays[m] = np.array(metric_arrays[m], dtype=np.float64)
    num_tokens = None
    if all(nt is not None for nt in num_tokens_list):
        num_tokens = np.array(num_tokens_list, dtype=np.float64)

    return labels, metric_arrays, num_tokens, num_layers


def load_split(condition_name):
    """Load the Phase 0 split indices."""
    split_file = OUTPUT_DIR / f"{condition_name}_baselines.json"
    with open(split_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    cal_idx = np.array(data["split"]["cal_indices"])
    test_idx = np.array(data["split"]["test_indices"])
    return cal_idx, test_idx


def norm_binned_auroc(y, target_scores, control_scores, n_bins=N_BINS):
    """
    Compute AUROC of target_scores within bins of control_scores.
    Returns: overall within-bin AUROC (weighted average), per-bin AUROCs.
    """
    mask = ~(np.isnan(target_scores) | np.isnan(control_scores))
    y_m = y[mask]
    t_m = target_scores[mask]
    c_m = control_scores[mask]

    if len(y_m) < 20:
        return 0.5, []

    # Create quantile bins
    try:
        bin_edges = np.percentile(c_m, np.linspace(0, 100, n_bins + 1))
        bin_edges[0] -= 1e-10
        bin_edges[-1] += 1e-10
    except Exception:
        return 0.5, []

    bin_aurocs = []
    bin_weights = []

    for i in range(n_bins):
        in_bin = (c_m >= bin_edges[i]) & (c_m < bin_edges[i + 1])
        y_bin = y_m[in_bin]
        t_bin = t_m[in_bin]

        if len(np.unique(y_bin)) < 2 or len(y_bin) < 5:
            bin_aurocs.append({
                "bin": i,
                "n": int(in_bin.sum()),
                "n_correct": int(y_bin.sum()) if len(y_bin) > 0 else 0,
                "auroc": None,
                "control_range": [float(bin_edges[i]), float(bin_edges[i + 1])],
            })
            continue

        auroc = safe_auroc(y_bin, t_bin)
        bin_aurocs.append({
            "bin": i,
            "n": int(in_bin.sum()),
            "n_correct": int(y_bin.sum()),
            "auroc": round(auroc, 6),
            "control_range": [round(float(bin_edges[i]), 6), round(float(bin_edges[i + 1]), 6)],
        })
        bin_weights.append((in_bin.sum(), auroc))

    # Weighted average AUROC
    if len(bin_weights) == 0:
        return 0.5, bin_aurocs

    total_w = sum(w for w, _ in bin_weights)
    weighted_auroc = sum(w * a for w, a in bin_weights) / total_w if total_w > 0 else 0.5

    return round(weighted_auroc, 6), bin_aurocs


def residual_auroc(y, target, confounder):
    """
    Residualize target against confounder via OLS, then compute AUROC.
    """
    mask = ~(np.isnan(target) | np.isnan(confounder))
    y_m = y[mask]
    t_m = target[mask]
    c_m = confounder[mask]

    if len(y_m) < 20 or len(np.unique(y_m)) < 2:
        return 0.5, 0.0, 0.0

    # OLS: target = a * confounder + b + residual
    slope, intercept = np.polyfit(c_m, t_m, 1)
    residual = t_m - (slope * c_m + intercept)

    orig_auroc = safe_auroc(y_m, -t_m)  # negative because lower entropy = correct
    resid_auroc = safe_auroc(y_m, -residual)

    return round(resid_auroc, 6), round(orig_auroc, 6), round(float(slope), 6)


def profile_auroc(y, X, method="logistic"):
    """
    Multi-layer profile AUROC using LogisticRegression with 5-fold CV.
    X: (n_samples, n_layers) matrix.
    """
    mask = ~np.any(np.isnan(X), axis=1)
    y_m = y[mask]
    X_m = X[mask]

    if len(np.unique(y_m)) < 2 or len(y_m) < 20:
        return 0.5

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_m)

    clf = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0, solver="lbfgs")
    try:
        proba = cross_val_predict(clf, X_scaled, y_m, cv=5, method="predict_proba")[:, 1]
        return round(safe_auroc(y_m, proba), 6)
    except Exception:
        return 0.5


def analyze_condition(name, config):
    print(f"\n{'='*60}")
    print(f"Condition: {name}")
    print(f"{'='*60}")

    labels, metrics, num_tokens, num_layers = load_data(config)
    cal_idx, test_idx = load_split(name)

    y_test = labels[test_idx]
    print(f"  Test set: {len(test_idx)} samples, {y_test.sum()} correct ({y_test.mean()*100:.1f}%)")

    # Load Phase 0 results for best layers
    phase0_file = OUTPUT_DIR / f"{name}_baselines.json"
    with open(phase0_file, "r") as f:
        phase0 = json.load(f)

    results = {
        "condition": name,
        "timestamp": datetime.now().isoformat(),
        "test_size": len(test_idx),
    }

    # ================================================================
    # 1. Norm-binned control: H_pre within bins of scale proxies
    # ================================================================
    print(f"\n  --- Norm-Binned Control (H_pre within bins of control variable) ---")

    h_pre_layer = phase0["baselines"]["unnormed_entropy_best_layer"]["cal_best_layer"]
    h_pre_sign = phase0["baselines"]["unnormed_entropy_best_layer"]["cal_best_sign"]
    h_pre_test = h_pre_sign * metrics["unnormed_entropy"][test_idx, h_pre_layer]
    h_pre_orig_auroc = safe_auroc(y_test, h_pre_test)

    controls = {
        "h_norm": ("h_norm", h_pre_layer),
        "logit_std": ("logit_std", h_pre_layer),
        "wh_norm": ("wh_norm", h_pre_layer),
    }
    if num_tokens is not None:
        controls["length"] = None  # special handling

    binned_results = {}
    for ctrl_name, ctrl_cfg in controls.items():
        if ctrl_cfg is None:
            ctrl_scores = num_tokens[test_idx]
        else:
            ctrl_metric, ctrl_layer = ctrl_cfg
            ctrl_scores = metrics[ctrl_metric][test_idx, ctrl_layer]

        weighted_auroc, bins = norm_binned_auroc(
            y_test, h_pre_test, ctrl_scores
        )

        binned_results[ctrl_name] = {
            "weighted_auroc_within_bins": weighted_auroc,
            "original_auroc": round(h_pre_orig_auroc, 6),
            "retention_pct": round(
                ((weighted_auroc - 0.5) / (h_pre_orig_auroc - 0.5) * 100)
                if h_pre_orig_auroc != 0.5 else 0, 1
            ),
            "bins": bins,
        }

        n_valid_bins = sum(1 for b in bins if b["auroc"] is not None)
        print(f"  H_pre | control={ctrl_name:12s} | orig={h_pre_orig_auroc:.4f} "
              f"| within-bin={weighted_auroc:.4f} | retention={binned_results[ctrl_name]['retention_pct']:.1f}% "
              f"| valid_bins={n_valid_bins}/{N_BINS}")

    results["norm_binned_control"] = binned_results

    # ================================================================
    # 2. Residual analysis: H_pre residualized against logit_std
    # ================================================================
    print(f"\n  --- Residual Analysis (H_pre residualized against confounders) ---")

    h_pre_raw = metrics["unnormed_entropy"][test_idx, h_pre_layer]
    residual_results = {}

    for confounder_name in ["logit_std", "h_norm", "wh_norm"]:
        confounder = metrics[confounder_name][test_idx, h_pre_layer]
        resid_auroc, orig_auroc, slope = residual_auroc(y_test, h_pre_raw, confounder)

        retention = round(
            ((resid_auroc - 0.5) / (orig_auroc - 0.5) * 100)
            if orig_auroc != 0.5 else 0, 1
        )

        residual_results[confounder_name] = {
            "original_auroc": orig_auroc,
            "residual_auroc": resid_auroc,
            "retention_pct": retention,
            "slope": slope,
        }

        print(f"  H_pre | confound={confounder_name:12s} | orig={orig_auroc:.4f} "
              f"| resid={resid_auroc:.4f} | retention={retention:.1f}% | slope={slope:.4f}")

    if num_tokens is not None:
        confounder = num_tokens[test_idx]
        resid_auroc, orig_auroc, slope = residual_auroc(y_test, h_pre_raw, confounder)
        retention = round(
            ((resid_auroc - 0.5) / (orig_auroc - 0.5) * 100)
            if orig_auroc != 0.5 else 0, 1
        )
        residual_results["length"] = {
            "original_auroc": orig_auroc,
            "residual_auroc": resid_auroc,
            "retention_pct": retention,
            "slope": slope,
        }
        print(f"  H_pre | confound={'length':12s} | orig={orig_auroc:.4f} "
              f"| resid={resid_auroc:.4f} | retention={retention:.1f}% | slope={slope:.4f}")

    results["residual_analysis"] = residual_results

    # ================================================================
    # 3. Multi-layer profile baselines (5-fold CV on test set)
    # ================================================================
    print(f"\n  --- Multi-Layer Profile Baselines (5-fold CV LogisticRegression) ---")

    profile_results = {}
    for metric_name in METRICS:
        X_test = metrics[metric_name][test_idx]
        auroc = profile_auroc(y_test, X_test)
        profile_results[metric_name] = auroc
        print(f"  {metric_name:25s} profile | AUROC={auroc:.4f}")

    # Combined profile: all metrics at best H_pre layer
    combined = np.column_stack([
        metrics[m][test_idx, h_pre_layer] for m in METRICS
    ])
    combined_auroc = profile_auroc(y_test, combined)
    profile_results["all_metrics_best_layer"] = combined_auroc
    print(f"  {'all_metrics@best_layer':25s} profile | AUROC={combined_auroc:.4f}")

    results["profile_baselines"] = profile_results

    # ================================================================
    # 4. Summary
    # ================================================================
    print(f"\n  {'─'*60}")
    print(f"  SCALE MEDIATION SUMMARY")
    print(f"  {'─'*60}")
    print(f"  H_pre original AUROC:           {h_pre_orig_auroc:.4f}")
    for ctrl_name, ctrl_data in binned_results.items():
        print(f"  After {ctrl_name:12s} bin-control: {ctrl_data['weighted_auroc_within_bins']:.4f} "
              f"(retention {ctrl_data['retention_pct']:.1f}%)")
    for conf_name, conf_data in residual_results.items():
        print(f"  After {conf_name:12s} residualize: {conf_data['residual_auroc']:.4f} "
              f"(retention {conf_data['retention_pct']:.1f}%)")
    print(f"  {'─'*60}")

    return results


def main():
    print("=" * 70)
    print("Phase 0b: Norm-Binned Control + Profile Baselines")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    all_results = {}

    for name, config in CONDITIONS.items():
        try:
            result = analyze_condition(name, config)
            all_results[name] = result
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    outpath = OUTPUT_DIR / "phase0b_norm_binned_results.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {outpath}")

    # Cross-condition summary
    print(f"\n{'='*80}")
    print("CROSS-CONDITION: H_pre Retention After Scale Control")
    print(f"{'='*80}")
    print(f"{'Condition':20s} {'Orig':>8s} {'h_norm':>10s} {'logit_std':>10s} {'wh_norm':>10s} {'length':>10s}")
    print("-" * 70)
    for name, result in all_results.items():
        orig = result["norm_binned_control"]["h_norm"]["original_auroc"]
        h_ret = result["norm_binned_control"]["h_norm"]["retention_pct"]
        l_ret = result["norm_binned_control"]["logit_std"]["retention_pct"]
        w_ret = result["norm_binned_control"]["wh_norm"]["retention_pct"]
        len_ret = result["norm_binned_control"].get("length", {}).get("retention_pct", "N/A")
        print(f"{name:20s} {orig:8.4f} {h_ret:9.1f}% {l_ret:9.1f}% {w_ret:9.1f}% {str(len_ret):>9s}%")

    print(f"\nCompleted: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
