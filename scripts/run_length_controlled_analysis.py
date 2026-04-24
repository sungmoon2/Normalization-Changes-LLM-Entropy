# -*- coding: utf-8 -*-
"""
Length-Controlled Analysis: 길이 confound 통제
================================================
GPT 8차 자문 공격 포인트 E 방어:
"정답/오답을 맞춘 게 아니라 길이를 맞춘 거 아니냐?"

방법:
1. 길이↔정답 상관 (point-biserial r)
2. 레이어별 길이↔entropy 상관 (Pearson r)
3. OLS 잔차화: entropy에서 길이 성분 제거
4. 원본 AUROC vs 잔차 AUROC 비교
5. 잔차 AUROC에 대한 Bootstrap 95% CI

CPU only. 기존 sample_results.json의 num_tokens 필드 활용.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.metrics import roc_auc_score
import warnings
from _paths import POT_DIR
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
EXP_ROOT = POT_DIR / "experiments"

EXPERIMENTS = {
    "qwen_easy": {
        "path": EXP_ROOT / "23_Normed_Difficulty_Analysis" / "EXP_20260213_013643_normed_easy" / "data" / "sample_results.json",
        "num_layers": 28, "model": "Qwen2.5-7B",
    },
    "qwen_hard": {
        "path": EXP_ROOT / "23_Normed_Difficulty_Analysis" / "EXP_20260213_113717_normed_hard" / "data" / "sample_results.json",
        "num_layers": 28, "model": "Qwen2.5-7B",
    },
    "qwen_arc": {
        "path": EXP_ROOT / "23_Normed_Difficulty_Analysis" / "EXP_20260214_182847_normed_arc" / "data" / "sample_results.json",
        "num_layers": 28, "model": "Qwen2.5-7B",
    },
    "llama_easy": {
        "path": EXP_ROOT / "26_Llama_Generalization" / "EXP_20260215_012714_llama_easy" / "data" / "sample_results.json",
        "num_layers": 32, "model": "Llama-3-8B",
    },
    "llama_hard": {
        "path": EXP_ROOT / "26_Llama_Generalization" / "EXP_20260215_101331_llama_hard" / "data" / "sample_results.json",
        "num_layers": 32, "model": "Llama-3-8B",
    },
    "qwen_arc_challenge": {
        "path": EXP_ROOT / "29_ARC_Challenge_Choice_Entropy" / "EXP_20260216_124125_arc_challenge" / "data" / "sample_results.json",
        "num_layers": 28, "model": "Qwen2.5-7B",
    },
}

SEED = 42
METRICS = ['unnormed_entropy', 'normed_entropy']
N_BOOTSTRAP = 1000


def load_data_with_length(path, num_layers):
    """sample_results.json에서 layer_data + num_tokens 추출."""
    with open(path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    valid = [s for s in samples
             if 'error' not in s and 'layer_data' in s and s.get('num_tokens') is not None]

    labels = np.array([1 if s['is_correct'] else 0 for s in valid])
    lengths = np.array([s['num_tokens'] for s in valid], dtype=float)

    data = {}
    for metric in METRICS:
        mat = np.zeros((len(valid), num_layers))
        for i, s in enumerate(valid):
            for l in range(num_layers):
                k = str(l)
                if k in s['layer_data'] and metric in s['layer_data'][k]:
                    val = s['layer_data'][k][metric]
                    if isinstance(val, (int, float)) and not (np.isinf(val) or np.isnan(val)):
                        mat[i, l] = val
        data[metric] = mat

    return data, labels, lengths


def compute_cohens_d(group1, group2):
    """Cohen's d (group1 - group2)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt((var1 + var2) / 2)
    if pooled_std < 1e-12:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def analyze_length_correctness(labels, lengths):
    """길이↔정답 기본 분석."""
    correct_lengths = lengths[labels == 1]
    incorrect_lengths = lengths[labels == 0]

    corr, p_value = stats.pointbiserialr(labels, lengths)
    d = compute_cohens_d(incorrect_lengths, correct_lengths)
    t_stat, t_p = stats.ttest_ind(correct_lengths, incorrect_lengths)

    return {
        "correlation": round(float(corr), 6),
        "correlation_p": float(p_value),
        "cohens_d_length": round(float(d), 4),
        "t_stat": round(float(t_stat), 4),
        "t_p": float(t_p),
        "correct_mean_length": round(float(np.mean(correct_lengths)), 1),
        "incorrect_mean_length": round(float(np.mean(incorrect_lengths)), 1),
        "correct_std_length": round(float(np.std(correct_lengths)), 1),
        "incorrect_std_length": round(float(np.std(incorrect_lengths)), 1),
        "overall_mean_length": round(float(np.mean(lengths)), 1),
        "overall_std_length": round(float(np.std(lengths)), 1),
    }


def layer_length_correlations(data_metric, lengths, num_layers):
    """레이어별 길이↔entropy 상관."""
    results = []
    for l in range(num_layers):
        x = data_metric[:, l]
        if np.std(x) > 1e-12 and np.std(lengths) > 1e-12:
            r, p = stats.pearsonr(lengths, x)
        else:
            r, p = 0.0, 1.0
        results.append({
            "layer": l,
            "corr_length_entropy": round(float(r), 6),
            "p_value": float(p),
        })
    return results


def residualize_entropy(data_metric, lengths, num_layers):
    """OLS: entropy_l = a + b*length + residual. 잔차 반환."""
    residuals = np.zeros_like(data_metric)
    regression_stats = []
    for l in range(num_layers):
        y = data_metric[:, l]
        if np.std(lengths) > 1e-12 and np.std(y) > 1e-12:
            slope, intercept = np.polyfit(lengths, y, 1)
            predicted = slope * lengths + intercept
            residuals[:, l] = y - predicted
            ss_res = np.sum((y - predicted) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        else:
            residuals[:, l] = y
            slope, intercept, r_squared = 0.0, float(np.mean(y)), 0.0

        regression_stats.append({
            "layer": l,
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": round(float(r_squared), 6),
        })
    return residuals, regression_stats


def compare_original_vs_residual(data_metric, residuals, labels, num_layers):
    """레이어별 원본 vs 잔차 AUROC·Cohen's d 비교."""
    results = []
    for l in range(num_layers):
        orig = data_metric[:, l]
        resid = residuals[:, l]

        # AUROC (higher entropy → incorrect → score = -entropy for label=1=correct)
        try:
            orig_auroc = roc_auc_score(labels, -orig)
        except ValueError:
            orig_auroc = 0.5
        try:
            resid_auroc = roc_auc_score(labels, -resid)
        except ValueError:
            resid_auroc = 0.5

        # Cohen's d (incorrect - correct)
        correct_orig = orig[labels == 1]
        incorrect_orig = orig[labels == 0]
        orig_d = compute_cohens_d(incorrect_orig, correct_orig)

        correct_resid = resid[labels == 1]
        incorrect_resid = resid[labels == 0]
        resid_d = compute_cohens_d(incorrect_resid, correct_resid)

        auroc_retention = (resid_auroc / orig_auroc * 100) if orig_auroc > 0.5 else None
        d_retention = (abs(resid_d) / abs(orig_d) * 100) if abs(orig_d) > 0.01 else None

        results.append({
            "layer": l,
            "original_auroc": round(float(orig_auroc), 6),
            "residual_auroc": round(float(resid_auroc), 6),
            "auroc_drop": round(float(orig_auroc - resid_auroc), 6),
            "auroc_retention_pct": round(float(auroc_retention), 1) if auroc_retention is not None else None,
            "original_d": round(float(orig_d), 4),
            "residual_d": round(float(resid_d), 4),
            "d_drop": round(float(abs(orig_d) - abs(resid_d)), 4),
            "d_retention_pct": round(float(d_retention), 1) if d_retention is not None else None,
        })
    return results


def bootstrap_residual_auroc(residuals, labels, layer, n_bootstrap=N_BOOTSTRAP):
    """최적 레이어 잔차 AUROC의 Bootstrap 95% CI."""
    np.random.seed(SEED)
    n = len(labels)
    aurocs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        y = labels[idx]
        x = residuals[idx, layer]
        if len(np.unique(y)) < 2:
            continue
        try:
            aurocs.append(roc_auc_score(y, -x))
        except ValueError:
            continue

    if len(aurocs) < 100:
        return {"error": f"insufficient valid bootstraps: {len(aurocs)}"}

    aurocs = np.array(aurocs)
    return {
        "mean": round(float(np.mean(aurocs)), 6),
        "ci_lower": round(float(np.percentile(aurocs, 2.5)), 6),
        "ci_upper": round(float(np.percentile(aurocs, 97.5)), 6),
        "above_chance": bool(np.percentile(aurocs, 2.5) > 0.5),
        "n_valid_bootstraps": len(aurocs),
    }


def main():
    start = datetime.now()
    print(f"\n{'#'*100}")
    print(f"  Length-Controlled Analysis: 길이 confound 통제")
    print(f"  {start.isoformat()}")
    print(f"{'#'*100}\n")

    all_results = {"timestamp": start.isoformat(), "seed": SEED,
                   "n_bootstrap": N_BOOTSTRAP, "conditions": {}}

    for cond, info in EXPERIMENTS.items():
        if not info["path"].exists():
            print(f"[SKIP] {cond}: file not found at {info['path']}")
            continue

        data, labels, lengths = load_data_with_length(info["path"], info["num_layers"])
        n_layers = info["num_layers"]
        n = len(labels)
        n_correct = int(labels.sum())
        n_incorrect = n - n_correct

        print(f"\n{'='*100}")
        print(f"  {cond.upper()} | {info['model']} | {n} samples | "
              f"{n_correct} correct / {n_incorrect} incorrect")
        print(f"{'='*100}")

        cond_result = {
            "n_samples": n,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "model": info["model"],
            "num_layers": n_layers,
        }

        # 1. Length-correctness basic
        lc = analyze_length_correctness(labels, lengths)
        cond_result["length_correctness"] = lc
        print(f"\n  Length ↔ Correctness:")
        print(f"    Point-biserial r = {lc['correlation']:.4f} (p = {lc['correlation_p']:.2e})")
        print(f"    Cohen's d (length): {lc['cohens_d_length']:.4f}")
        print(f"    Correct  mean length: {lc['correct_mean_length']:.1f} ± {lc['correct_std_length']:.1f}")
        print(f"    Incorrect mean length: {lc['incorrect_mean_length']:.1f} ± {lc['incorrect_std_length']:.1f}")

        for metric in METRICS:
            if metric not in data:
                continue

            print(f"\n  --- {metric} ---")
            metric_result = {}

            # 2. Layer-entropy correlations
            le_corrs = layer_length_correlations(data[metric], lengths, n_layers)
            metric_result["length_entropy_correlations"] = le_corrs

            max_corr_entry = max(le_corrs, key=lambda x: abs(x['corr_length_entropy']))
            print(f"    Max |r(length, entropy)|: L{max_corr_entry['layer']} "
                  f"r={max_corr_entry['corr_length_entropy']:.4f}")

            # 3. Residualize
            residuals, reg_stats = residualize_entropy(data[metric], lengths, n_layers)
            metric_result["regression_stats"] = reg_stats

            max_r2 = max(reg_stats, key=lambda x: x['r_squared'])
            print(f"    Max R² (length→entropy): L{max_r2['layer']} R²={max_r2['r_squared']:.4f}")

            # 4. Compare original vs residual
            comparison = compare_original_vs_residual(data[metric], residuals, labels, n_layers)
            metric_result["original_vs_residual"] = comparison

            # Find best original AUROC layer
            best_orig = max(comparison, key=lambda x: x['original_auroc'])
            best_layer = best_orig['layer']
            metric_result["best_original_layer"] = best_layer

            print(f"\n    {'Layer':>6} | {'Orig AUROC':>11} | {'Resid AUROC':>12} | "
                  f"{'Drop':>7} | {'Retain%':>8} | {'Orig d':>8} | {'Resid d':>8} | {'d Retain%':>10}")
            print(f"    {'-'*95}")

            for c in comparison:
                flag = " ***" if c['layer'] == best_layer else ""
                ret_str = f"{c['auroc_retention_pct']:>7.1f}%" if c['auroc_retention_pct'] is not None else "    N/A"
                d_ret_str = f"{c['d_retention_pct']:>9.1f}%" if c['d_retention_pct'] is not None else "      N/A"
                print(f"    {c['layer']:>6} | {c['original_auroc']:>11.4f} | {c['residual_auroc']:>12.4f} | "
                      f"{c['auroc_drop']:>+7.4f} | {ret_str} | {c['original_d']:>+8.4f} | "
                      f"{c['residual_d']:>+8.4f} | {d_ret_str}{flag}")

            # 5. Bootstrap CI for residual at best layer
            boot = bootstrap_residual_auroc(residuals, labels, best_layer)
            metric_result["bootstrap_residual_best_layer"] = boot

            if "error" not in boot:
                above_str = "YES" if boot['above_chance'] else "NO"
                print(f"\n    Bootstrap CI (L{best_layer}): "
                      f"mean={boot['mean']:.4f} [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")
                print(f"    Above chance (0.5)? {above_str}")
            else:
                print(f"\n    Bootstrap error: {boot['error']}")

            cond_result[metric] = metric_result

        all_results["conditions"][cond] = cond_result

    # ========================================================================
    # Summary
    # ========================================================================
    print(f"\n\n{'#'*100}")
    print(f"  SUMMARY: Length-Controlled Analysis")
    print(f"{'#'*100}")
    print(f"\n{'Condition':>15} | {'Metric':>20} | {'Best L':>6} | "
          f"{'Orig AUROC':>11} | {'Resid AUROC':>12} | {'Retain%':>8} | "
          f"{'Boot CI':>20} | {'Above 0.5?':>10}")
    print("-" * 120)

    for cond, cond_data in all_results["conditions"].items():
        for metric in METRICS:
            if metric not in cond_data:
                continue
            mdata = cond_data[metric]
            bl = mdata["best_original_layer"]
            comp = [c for c in mdata["original_vs_residual"] if c["layer"] == bl][0]
            boot = mdata["bootstrap_residual_best_layer"]

            ret_str = f"{comp['auroc_retention_pct']:.1f}%" if comp['auroc_retention_pct'] is not None else "N/A"
            if "error" not in boot:
                ci_str = f"[{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]"
                above_str = "YES" if boot['above_chance'] else "NO"
            else:
                ci_str = "ERROR"
                above_str = "?"

            print(f"{cond:>15} | {metric:>20} | L{bl:>4} | "
                  f"{comp['original_auroc']:>11.4f} | {comp['residual_auroc']:>12.4f} | "
                  f"{ret_str:>8} | {ci_str:>20} | {above_str:>10}")

    # Save
    out_dir = EXP_ROOT / "28_GPT6_Supplements"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "length_controlled_analysis_results.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    elapsed = datetime.now() - start
    print(f"\n[DONE] {elapsed.total_seconds():.1f}초 → {out_path}")


if __name__ == "__main__":
    main()
