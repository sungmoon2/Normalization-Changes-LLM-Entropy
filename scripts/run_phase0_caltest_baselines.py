"""
Phase 0: Cal/Test 70/30 Split + Scalar Baselines
=================================================
GPT 12차 4개 응답 전면 수렴 기반 실행.

목적:
  1. 70/30 stratified locked split 생성 (seed=42)
  2. Cal에서 best layer + sign 선택
  3. Test에서 AUROC + AURC 평가 (모든 baseline 동일 split)
  4. Paired bootstrap significance test

Baselines (scalar, single-layer):
  - H_pre (unnormed_entropy)
  - H_post (normed_entropy)
  - h_norm
  - wh_norm
  - logit_std
  - logit_max
  - logit_margin
  - wh_rms
  - output_entropy (last layer unnormed)
  - output_maxprob (last layer logit_max)
  - output_margin (last layer logit_margin)
  - length_only (num_tokens)

원칙:
  - 할루시네이션/추정 없이 실측값만 사용
  - scalar baseline은 logistic regression 불필요, raw AUROC (GPT 12 수렴)
  - same split across all methods (paired comparison)
  - sign calibration on cal split (GPT 12 수렴: Llama d 부호 반전 대응)
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

# ============================================================
# Constants
# ============================================================
SEED = 42
CAL_RATIO = 0.7
TEST_RATIO = 0.3
N_BOOTSTRAP = 1000
ALPHA = 0.05

BASE_DIR = Path(__file__).parent.parent / "PoT_Experiment_Entropy_Attention_Extraction_Experiment"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
OUTPUT_DIR = EXPERIMENTS_DIR / "32_Phase0_CalTest_Baselines"

# Experiment paths
EXPERIMENT_CONFIGS = {
    "mmlu_qwen": {
        "path": "31_MMLU_Domain_Extension/EXP_20260219_053638_mmlu_qwen/data/sample_results.json",
        "model": "Qwen2.5-7B-Instruct",
        "num_layers": 28,
        "data_format": "layer_data",
        "has_full_metrics": True,
        "dataset": "MMLU",
    },
    "mmlu_llama": {
        "path": "31_MMLU_Domain_Extension/EXP_20260219_171237_mmlu_llama/data/sample_results.json",
        "model": "Llama-3-8B-Instruct",
        "num_layers": 32,
        "data_format": "layer_data",
        "has_full_metrics": True,
        "dataset": "MMLU",
    },
    "mmlu_mistral": {
        "path": "31_MMLU_Domain_Extension/EXP_20260220_000610_mmlu_mistral/data/sample_results.json",
        "model": "Mistral-7B-Instruct-v0.3",
        "num_layers": 32,
        "data_format": "layer_data",
        "has_full_metrics": True,
        "dataset": "MMLU",
    },
    "qwen_hard": {
        "path": "23_Normed_Difficulty_Analysis/EXP_20260213_113717_normed_hard/data/sample_results.json",
        "model": "Qwen2.5-7B-Instruct",
        "num_layers": 28,
        "data_format": "layer_data",
        "has_full_metrics": True,
        "dataset": "Math Hard (L4-5)",
    },
    "qwen_easy": {
        "path": "23_Normed_Difficulty_Analysis/EXP_20260213_013643_normed_easy/data/sample_results.json",
        "model": "Qwen2.5-7B-Instruct",
        "num_layers": 28,
        "data_format": "layer_data",
        "has_full_metrics": False,  # only 4 fields
        "dataset": "Math Easy (L1-2)",
    },
}

# Metrics to evaluate
FULL_METRICS = [
    "unnormed_entropy", "normed_entropy", "h_norm", "wh_norm",
    "wh_rms", "logit_std", "logit_max", "logit_margin"
]
REDUCED_METRICS = ["unnormed_entropy", "normed_entropy", "h_norm", "wh_norm"]


# ============================================================
# Data Loading
# ============================================================
def load_experiment_data(config):
    """Load experiment data and extract per-layer metrics."""
    filepath = EXPERIMENTS_DIR / config["path"]
    if not filepath.exists():
        print(f"  [ERROR] File not found: {filepath}")
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if isinstance(samples, dict) and "results" in samples:
        samples = samples["results"]

    num_layers = config["num_layers"]
    metrics = FULL_METRICS if config["has_full_metrics"] else REDUCED_METRICS

    labels = []
    num_tokens_list = []
    metric_arrays = {m: [] for m in metrics}
    valid_indices = []

    for i, sample in enumerate(samples):
        if "layer_data" not in sample:
            continue
        if "is_correct" not in sample:
            continue

        layer_data = sample["layer_data"]

        # Verify all layers exist
        layer_keys = [str(l) for l in range(num_layers)]
        if not all(k in layer_data for k in layer_keys):
            # Try with actual keys
            actual_layers = sorted(layer_data.keys(), key=lambda x: int(x))
            if len(actual_layers) != num_layers:
                continue
            layer_keys = actual_layers

        # Check for valid data (no NaN/Inf in critical metrics)
        valid = True
        row = {m: [] for m in metrics}
        for lk in layer_keys:
            ld = layer_data[lk]
            for m in metrics:
                if m not in ld:
                    valid = False
                    break
                val = ld[m]
                if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                    val = np.nan  # Mark for later handling
                row[m].append(val)
            if not valid:
                break

        if not valid:
            continue

        # Check if any metric has all NaN
        skip = False
        for m in metrics:
            if all(np.isnan(v) for v in row[m] if isinstance(v, float)):
                skip = True
                break
        if skip:
            continue

        labels.append(1 if sample["is_correct"] else 0)
        valid_indices.append(i)

        for m in metrics:
            metric_arrays[m].append(row[m])

        # num_tokens
        nt = sample.get("num_tokens", None)
        num_tokens_list.append(nt)

    labels = np.array(labels)
    for m in metrics:
        metric_arrays[m] = np.array(metric_arrays[m], dtype=np.float64)

    num_tokens_arr = None
    if all(nt is not None for nt in num_tokens_list) and len(num_tokens_list) > 0:
        num_tokens_arr = np.array(num_tokens_list, dtype=np.float64)

    print(f"  Loaded: {len(labels)} samples, {labels.sum()} correct ({labels.mean()*100:.1f}%), "
          f"{num_layers} layers, {len(metrics)} metrics")

    return {
        "labels": labels,
        "metrics": metric_arrays,
        "num_tokens": num_tokens_arr,
        "num_layers": num_layers,
        "metric_names": metrics,
        "n_samples": len(labels),
        "accuracy": float(labels.mean()),
        "valid_indices": valid_indices,
    }


# ============================================================
# Evaluation Functions
# ============================================================
def safe_auroc(y_true, scores):
    """Compute AUROC with safety checks."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    # Handle NaN in scores
    mask = ~np.isnan(scores)
    if mask.sum() < 10 or len(np.unique(y_true[mask])) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y_true[mask], scores[mask]))
    except ValueError:
        return 0.5


def compute_aurc(y_true, scores):
    """
    Compute Area Under Risk-Coverage curve.
    Lower AURC = better selective prediction.
    scores: higher = more confident (correct prediction).
    """
    n = len(y_true)
    if n == 0:
        return 1.0

    # Sort by confidence (descending)
    order = np.argsort(-scores)
    y_sorted = y_true[order]

    # Compute risk at each coverage level
    coverages = []
    risks = []
    for k in range(1, n + 1):
        cov = k / n
        risk = 1.0 - y_sorted[:k].mean()  # error rate
        coverages.append(cov)
        risks.append(risk)

    coverages = np.array(coverages)
    risks = np.array(risks)

    # AURC = integral of risk over coverage
    aurc = float(np.trapz(risks, coverages))
    return aurc


def compute_e_aurc(y_true, scores):
    """
    Compute Excess AURC (E-AURC).
    E-AURC = AURC - AURC_optimal
    AURC_optimal = area under oracle risk-coverage curve.
    """
    n = len(y_true)
    error_rate = 1.0 - y_true.mean()
    n_errors = int((1 - y_true).sum())

    # Oracle: all errors at the end
    coverages_opt = []
    risks_opt = []
    for k in range(1, n + 1):
        cov = k / n
        # Oracle selects correct samples first
        n_correct_available = int(y_true.sum())
        n_selected_correct = min(k, n_correct_available)
        n_selected_incorrect = k - n_selected_correct
        risk = n_selected_incorrect / k if k > 0 else 0
        coverages_opt.append(cov)
        risks_opt.append(risk)

    aurc_opt = float(np.trapz(risks_opt, coverages_opt))
    aurc = compute_aurc(y_true, scores)
    e_aurc = aurc - aurc_opt
    return e_aurc


def compute_risk_coverage_curve(y_true, scores, n_points=20):
    """Compute risk-coverage curve at fixed coverage levels."""
    n = len(y_true)
    order = np.argsort(-scores)
    y_sorted = y_true[order]

    curve = []
    for i in range(1, n_points + 1):
        cov = i / n_points
        k = max(1, int(n * cov))
        acc = float(y_sorted[:k].mean())
        curve.append({
            "coverage": round(cov, 2),
            "accuracy": round(acc, 6),
            "risk": round(1.0 - acc, 6),
            "n_selected": k,
        })
    return curve


def select_best_layer_and_sign(y_cal, X_cal):
    """
    On calibration data, select best layer and sign.
    Try both directions (positive and negative correlation with correctness).
    Returns (best_layer, sign), where sign is +1 or -1 to multiply scores.
    """
    n_layers = X_cal.shape[1]
    best_auroc = -1
    best_layer = 0
    best_sign = 1

    for l in range(n_layers):
        col = X_cal[:, l]
        mask = ~np.isnan(col)
        if mask.sum() < 10 or len(np.unique(y_cal[mask])) < 2:
            continue

        # Try positive direction (higher value = correct)
        auroc_pos = safe_auroc(y_cal, col)
        # Try negative direction (lower value = correct)
        auroc_neg = safe_auroc(y_cal, -col)

        if auroc_pos >= auroc_neg and auroc_pos > best_auroc:
            best_auroc = auroc_pos
            best_layer = l
            best_sign = 1
        elif auroc_neg > auroc_pos and auroc_neg > best_auroc:
            best_auroc = auroc_neg
            best_layer = l
            best_sign = -1

    return best_layer, best_sign, best_auroc


def evaluate_baseline(y_test, scores_test):
    """Compute all evaluation metrics for a baseline on test set."""
    mask = ~np.isnan(scores_test)
    y_m = y_test[mask]
    s_m = scores_test[mask]

    if len(np.unique(y_m)) < 2 or len(y_m) < 10:
        return {
            "auroc": 0.5,
            "aurc": 1.0,
            "e_aurc": 0.5,
            "n_valid": int(mask.sum()),
            "risk_coverage": [],
        }

    auroc = safe_auroc(y_m, s_m)
    aurc = compute_aurc(y_m, s_m)
    e_aurc = compute_e_aurc(y_m, s_m)
    rc_curve = compute_risk_coverage_curve(y_m, s_m)

    return {
        "auroc": round(auroc, 6),
        "aurc": round(aurc, 6),
        "e_aurc": round(e_aurc, 6),
        "n_valid": int(mask.sum()),
        "risk_coverage": rc_curve,
    }


def paired_bootstrap_test(y_test, scores_a, scores_b, n_bootstrap=N_BOOTSTRAP):
    """
    Paired bootstrap test for AUROC difference.
    H0: AUROC_A = AUROC_B
    Returns p-value and CI for difference.
    """
    rng = np.random.RandomState(SEED)
    mask = ~(np.isnan(scores_a) | np.isnan(scores_b))
    y = y_test[mask]
    sa = scores_a[mask]
    sb = scores_b[mask]

    if len(np.unique(y)) < 2 or len(y) < 20:
        return {"p_value": 1.0, "delta_ci_lower": 0.0, "delta_ci_upper": 0.0, "observed_delta": 0.0}

    observed_a = safe_auroc(y, sa)
    observed_b = safe_auroc(y, sb)
    observed_delta = observed_a - observed_b

    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y), size=len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        d_a = safe_auroc(y[idx], sa[idx])
        d_b = safe_auroc(y[idx], sb[idx])
        deltas.append(d_a - d_b)

    if len(deltas) < 100:
        return {"p_value": 1.0, "delta_ci_lower": 0.0, "delta_ci_upper": 0.0, "observed_delta": round(observed_delta, 6)}

    deltas = np.array(deltas)
    # Two-sided p-value: proportion of bootstrap deltas on opposite side of zero
    if observed_delta >= 0:
        p_value = float(np.mean(deltas <= 0)) * 2
    else:
        p_value = float(np.mean(deltas >= 0)) * 2
    p_value = min(p_value, 1.0)

    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))

    return {
        "p_value": round(p_value, 6),
        "delta_ci_lower": round(ci_lower, 6),
        "delta_ci_upper": round(ci_upper, 6),
        "observed_delta": round(observed_delta, 6),
    }


# ============================================================
# Main Analysis
# ============================================================
def analyze_condition(name, config):
    """Run full Phase 0 analysis for one experimental condition."""
    print(f"\n{'='*60}")
    print(f"Condition: {name}")
    print(f"  Model: {config['model']}, Dataset: {config['dataset']}")
    print(f"{'='*60}")

    data = load_experiment_data(config)
    if data is None:
        return None

    labels = data["labels"]
    metrics = data["metrics"]
    num_layers = data["num_layers"]
    metric_names = data["metric_names"]
    num_tokens = data["num_tokens"]
    n = data["n_samples"]

    # --------------------------------------------------------
    # 1. Create 70/30 stratified split
    # --------------------------------------------------------
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_RATIO, random_state=SEED)
    cal_idx, test_idx = next(splitter.split(np.zeros(n), labels))

    y_cal = labels[cal_idx]
    y_test = labels[test_idx]

    print(f"\n  Split: cal={len(cal_idx)} ({y_cal.mean()*100:.1f}% correct), "
          f"test={len(test_idx)} ({y_test.mean()*100:.1f}% correct)")

    # --------------------------------------------------------
    # 2. Evaluate each layer-wise metric
    # --------------------------------------------------------
    results = {
        "condition": name,
        "model": config["model"],
        "dataset": config["dataset"],
        "n_samples": n,
        "accuracy": round(data["accuracy"], 6),
        "split": {
            "cal_size": len(cal_idx),
            "test_size": len(test_idx),
            "cal_accuracy": round(float(y_cal.mean()), 6),
            "test_accuracy": round(float(y_test.mean()), 6),
            "cal_indices": cal_idx.tolist(),
            "test_indices": test_idx.tolist(),
            "seed": SEED,
            "ratio": f"{int(CAL_RATIO*100)}/{int(TEST_RATIO*100)}",
        },
        "baselines": {},
        "paired_tests": {},
        "timestamp": datetime.now().isoformat(),
    }

    # Layer-wise baselines
    all_test_scores = {}

    for metric_name in metric_names:
        X = metrics[metric_name]
        X_cal = X[cal_idx]
        X_test = X[test_idx]

        # Select best layer + sign on cal
        best_layer, best_sign, cal_auroc = select_best_layer_and_sign(y_cal, X_cal)

        # Evaluate on test
        test_scores = best_sign * X_test[:, best_layer]
        eval_result = evaluate_baseline(y_test, test_scores)

        # Also compute all-layer AUROCs on test for reporting
        per_layer_test = {}
        for l in range(num_layers):
            col = X_test[:, l]
            a_pos = safe_auroc(y_test, col)
            a_neg = safe_auroc(y_test, -col)
            per_layer_test[str(l)] = {
                "auroc_pos": round(a_pos, 6),
                "auroc_neg": round(a_neg, 6),
                "best": round(max(a_pos, a_neg), 6),
            }

        baseline_key = f"{metric_name}_best_layer"
        results["baselines"][baseline_key] = {
            "metric": metric_name,
            "type": "layer_wise_scalar",
            "cal_best_layer": int(best_layer),
            "cal_best_sign": int(best_sign),
            "cal_auroc": round(cal_auroc, 6),
            "test_auroc": eval_result["auroc"],
            "test_aurc": eval_result["aurc"],
            "test_e_aurc": eval_result["e_aurc"],
            "test_n_valid": eval_result["n_valid"],
            "risk_coverage": eval_result["risk_coverage"],
            "per_layer_test_auroc": per_layer_test,
        }

        all_test_scores[baseline_key] = test_scores
        sign_str = "+" if best_sign == 1 else "-"
        print(f"  {metric_name:25s} | Layer {best_layer:2d} ({sign_str}) | "
              f"Cal AUROC={cal_auroc:.4f} | Test AUROC={eval_result['auroc']:.4f} | "
              f"Test AURC={eval_result['aurc']:.4f} | E-AURC={eval_result['e_aurc']:.4f}")

    # --------------------------------------------------------
    # 3. Output-level baselines (last layer only, no layer selection)
    # --------------------------------------------------------
    last_layer = num_layers - 1

    output_baselines = {}
    if "unnormed_entropy" in metric_names:
        output_baselines["output_entropy"] = {
            "source": "unnormed_entropy",
            "layer": last_layer,
            "sign": -1,  # lower entropy = more confident
        }
    if "logit_max" in metric_names:
        output_baselines["output_logit_max"] = {
            "source": "logit_max",
            "layer": last_layer,
            "sign": 1,  # higher max logit = more confident
        }
    if "logit_margin" in metric_names:
        output_baselines["output_margin"] = {
            "source": "logit_margin",
            "layer": last_layer,
            "sign": 1,  # larger margin = more confident
        }

    for out_name, out_cfg in output_baselines.items():
        X = metrics[out_cfg["source"]]
        X_test_col = X[test_idx, out_cfg["layer"]]
        test_scores = out_cfg["sign"] * X_test_col
        eval_result = evaluate_baseline(y_test, test_scores)

        results["baselines"][out_name] = {
            "metric": out_cfg["source"],
            "type": "output_level",
            "layer": out_cfg["layer"],
            "sign": out_cfg["sign"],
            "test_auroc": eval_result["auroc"],
            "test_aurc": eval_result["aurc"],
            "test_e_aurc": eval_result["e_aurc"],
            "test_n_valid": eval_result["n_valid"],
            "risk_coverage": eval_result["risk_coverage"],
        }

        all_test_scores[out_name] = test_scores
        print(f"  {out_name:25s} | Layer {out_cfg['layer']:2d} (fixed) | "
              f"Test AUROC={eval_result['auroc']:.4f} | "
              f"Test AURC={eval_result['aurc']:.4f} | E-AURC={eval_result['e_aurc']:.4f}")

    # --------------------------------------------------------
    # 4. Length-only baseline
    # --------------------------------------------------------
    if num_tokens is not None:
        nt_cal = num_tokens[cal_idx]
        nt_test = num_tokens[test_idx]

        # Determine sign on cal
        auroc_pos = safe_auroc(y_cal, nt_cal)
        auroc_neg = safe_auroc(y_cal, -nt_cal)
        if auroc_pos >= auroc_neg:
            length_sign = 1
            cal_auroc_len = auroc_pos
        else:
            length_sign = -1
            cal_auroc_len = auroc_neg

        test_scores_len = length_sign * nt_test
        eval_result = evaluate_baseline(y_test, test_scores_len)

        results["baselines"]["length_only"] = {
            "metric": "num_tokens",
            "type": "confound_control",
            "sign": int(length_sign),
            "cal_auroc": round(cal_auroc_len, 6),
            "test_auroc": eval_result["auroc"],
            "test_aurc": eval_result["aurc"],
            "test_e_aurc": eval_result["e_aurc"],
            "test_n_valid": eval_result["n_valid"],
            "risk_coverage": eval_result["risk_coverage"],
        }

        all_test_scores["length_only"] = test_scores_len
        sign_str = "+" if length_sign == 1 else "-"
        print(f"  {'length_only':25s} | num_tokens ({sign_str}) | "
              f"Cal AUROC={cal_auroc_len:.4f} | Test AUROC={eval_result['auroc']:.4f} | "
              f"Test AURC={eval_result['aurc']:.4f} | E-AURC={eval_result['e_aurc']:.4f}")

    # --------------------------------------------------------
    # 5. Paired bootstrap tests (vs H_pre best layer)
    # --------------------------------------------------------
    reference_key = "unnormed_entropy_best_layer"
    if reference_key in all_test_scores:
        ref_scores = all_test_scores[reference_key]
        print(f"\n  Paired bootstrap tests (reference: {reference_key}):")
        for bname, bscores in all_test_scores.items():
            if bname == reference_key:
                continue
            pbt = paired_bootstrap_test(y_test, ref_scores, bscores)
            results["paired_tests"][f"{reference_key}_vs_{bname}"] = pbt
            sig_marker = "*" if pbt["p_value"] < ALPHA else " "
            print(f"    vs {bname:30s} | delta={pbt['observed_delta']:+.4f} "
                  f"[{pbt['delta_ci_lower']:+.4f}, {pbt['delta_ci_upper']:+.4f}] "
                  f"p={pbt['p_value']:.4f} {sig_marker}")

    # --------------------------------------------------------
    # 6. Summary table
    # --------------------------------------------------------
    print(f"\n  {'─'*70}")
    print(f"  SUMMARY TABLE (sorted by Test AUROC)")
    print(f"  {'─'*70}")
    print(f"  {'Baseline':35s} {'AUROC':>8s} {'AURC':>8s} {'E-AURC':>8s} {'Layer':>6s} {'Sign':>5s}")
    print(f"  {'─'*70}")

    sorted_baselines = sorted(
        results["baselines"].items(),
        key=lambda x: x[1]["test_auroc"],
        reverse=True
    )
    for bname, bdata in sorted_baselines:
        layer_str = str(bdata.get("cal_best_layer", bdata.get("layer", "-")))
        sign_str = "+" if bdata.get("cal_best_sign", bdata.get("sign", 1)) == 1 else "-"
        print(f"  {bname:35s} {bdata['test_auroc']:8.4f} {bdata['test_aurc']:8.4f} "
              f"{bdata['test_e_aurc']:8.4f} {layer_str:>6s} {sign_str:>5s}")
    print(f"  {'─'*70}")

    return results


def main():
    print("=" * 70)
    print("Phase 0: Cal/Test 70/30 Split + Scalar Baselines")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Seed: {SEED}, Split: {int(CAL_RATIO*100)}/{int(TEST_RATIO*100)}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = {}

    for name, config in EXPERIMENT_CONFIGS.items():
        result = analyze_condition(name, config)
        if result is not None:
            all_results[name] = result

            # Save individual result
            outpath = OUTPUT_DIR / f"{name}_baselines.json"
            with open(outpath, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n  Saved: {outpath}")

    # ========================================================
    # Cross-condition summary
    # ========================================================
    print("\n\n" + "=" * 90)
    print("CROSS-CONDITION SUMMARY")
    print("=" * 90)

    # Build summary table
    summary = {
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "split_ratio": f"{int(CAL_RATIO*100)}/{int(TEST_RATIO*100)}",
        "conditions": {},
    }

    # Print header
    conditions = list(all_results.keys())
    header = f"{'Baseline':35s}"
    for c in conditions:
        header += f" {c:>15s}"
    print(header)
    print("-" * (35 + 16 * len(conditions)))

    # Collect all baseline names
    all_baseline_names = set()
    for r in all_results.values():
        all_baseline_names.update(r["baselines"].keys())
    all_baseline_names = sorted(all_baseline_names)

    for bname in all_baseline_names:
        row = f"{bname:35s}"
        for c in conditions:
            if bname in all_results[c]["baselines"]:
                auroc = all_results[c]["baselines"][bname]["test_auroc"]
                row += f" {auroc:15.4f}"
            else:
                row += f" {'N/A':>15s}"
        print(row)

    # Save summary
    for c in conditions:
        cond_summary = {
            "model": all_results[c]["model"],
            "dataset": all_results[c]["dataset"],
            "n_samples": all_results[c]["n_samples"],
            "accuracy": all_results[c]["accuracy"],
            "baselines": {}
        }
        for bname, bdata in all_results[c]["baselines"].items():
            cond_summary["baselines"][bname] = {
                "test_auroc": bdata["test_auroc"],
                "test_aurc": bdata["test_aurc"],
                "test_e_aurc": bdata["test_e_aurc"],
                "cal_best_layer": bdata.get("cal_best_layer"),
                "cal_best_sign": bdata.get("cal_best_sign"),
            }
        summary["conditions"][c] = cond_summary

    summary_path = OUTPUT_DIR / "summary_table.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved: {summary_path}")

    print(f"\nPhase 0 completed: {datetime.now().isoformat()}")
    print(f"Results directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
