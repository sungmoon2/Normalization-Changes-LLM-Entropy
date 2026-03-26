# -*- coding: utf-8 -*-
"""
MMLU 3-Model Post-Processing Pipeline
=======================================
Qwen/Llama/Mistral MMLU 실험 완료 후 전체 후처리 실행.
기존 후처리 7종을 MMLU 경로에 맞게 통합.

분석 항목:
  1. AUROC / ECE / AUPRC (레이어별)
  2. Statistical Validation (Bonferroni / BH-FDR / Bootstrap CI / 5-fold CV / Permutation)
  3. Length-Controlled Analysis (길이 confound 통제)
  4. Nested CV (편향 없는 레이어 선택 평가)
  5. Mediation Bootstrap (||h|| scale 매개 분석)
  6. Risk-Coverage Curve (선택적 예측)
  7. Output-Proxy Baseline (마지막 레이어 vs best 레이어)
  8. Decomposition (normed+||h|| 결합 vs unnormed 단독)

입력: 31_MMLU_Domain_Extension/EXP_*/data/sample_results.json
출력: 31_MMLU_Domain_Extension/mmlu_postprocessing_results.json
CPU only.

실행: python scripts/run_mmlu_postprocessing.py
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 설정
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
EXP_ROOT = PROJECT_ROOT / "PoT_Experiment_Entropy_Attention_Extraction_Experiment" / "experiments"
MMLU_DIR = EXP_ROOT / "31_MMLU_Domain_Extension"

EXPERIMENTS = {
    "mmlu_qwen": {
        "path": MMLU_DIR / "EXP_20260219_053638_mmlu_qwen" / "data" / "sample_results.json",
        "num_layers": 28,
        "last_layer": 27,
        "model": "Qwen2.5-7B-Instruct",
    },
    "mmlu_llama": {
        "path": MMLU_DIR / "EXP_20260219_171237_mmlu_llama" / "data" / "sample_results.json",
        "num_layers": 32,
        "last_layer": 31,
        "model": "Llama-3-8B-Instruct",
    },
    "mmlu_mistral": {
        "path": MMLU_DIR / "EXP_20260220_000610_mmlu_mistral" / "data" / "sample_results.json",
        "num_layers": 32,
        "last_layer": 31,
        "model": "Mistral-7B-Instruct-v0.3",
    },
}

SEED = 42
N_BOOTSTRAP = 1000
N_PERMUTATION = 1000
N_FOLDS = 5
ALPHA = 0.05
METRICS = ['unnormed_entropy', 'normed_entropy', 'h_norm', 'wh_norm']
COVERAGE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

np.random.seed(SEED)

# ============================================================================
# 유틸리티
# ============================================================================

def log(tag, msg):
    print(f"[{tag}] {msg}", flush=True)


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_std = np.sqrt(((n1-1)*np.var(group1, ddof=1) +
                          (n2-1)*np.var(group2, ddof=1)) / (n1+n2-2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def safe_auroc(y, scores):
    try:
        return roc_auc_score(y, -scores)
    except ValueError:
        return 0.5


def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return ece / len(y_true)


def optimal_threshold_ba(y_true, y_scores):
    thresholds = np.percentile(y_scores, np.arange(1, 100))
    best = 0.5
    for t in thresholds:
        ba = balanced_accuracy_score(y_true, (y_scores >= t).astype(int))
        if ba > best:
            best = ba
    return best


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ============================================================================
# 데이터 로드
# ============================================================================

def load_full_data(path, num_layers):
    """sample_results.json → all metric arrays + labels + lengths"""
    with open(path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    valid = [s for s in samples if 'error' not in s and 'layer_data' in s]
    labels = np.array([1 if s['is_correct'] else 0 for s in valid])
    lengths = np.array([s.get('num_tokens', 0) for s in valid], dtype=float)

    avail_metrics = [m for m in METRICS if m in valid[0]['layer_data']['0']]

    data = {}
    for metric in avail_metrics:
        mat = np.zeros((len(valid), num_layers))
        for i, s in enumerate(valid):
            for l in range(num_layers):
                key = str(l)
                if key in s['layer_data'] and metric in s['layer_data'][key]:
                    val = s['layer_data'][key][metric]
                    if isinstance(val, (int, float)) and not (np.isinf(val) or np.isnan(val)):
                        mat[i, l] = val
        data[metric] = mat

    return data, labels, lengths, avail_metrics


# ============================================================================
# 1. AUROC / ECE / AUPRC
# ============================================================================

def analysis_auroc_ece(data, labels, num_layers, avail_metrics, cond_name):
    log("AUROC", f"{cond_name} 시작")
    all_metric_results = []

    for metric in avail_metrics:
        metric_results = []
        best_auroc = (0, -1)

        for l in range(num_layers):
            x = data[metric][:, l]
            try:
                auroc = roc_auc_score(labels, -x)
            except ValueError:
                auroc = 0.5
            try:
                auprc = average_precision_score(labels, -x)
            except ValueError:
                auprc = labels.mean()

            x_min, x_max = x.min(), x.max()
            if x_max > x_min:
                x_prob = 1.0 - (x - x_min) / (x_max - x_min)
            else:
                x_prob = np.full_like(x, 0.5)
            ece = compute_ece(labels, x_prob)
            ba = optimal_threshold_ba(labels, -x)

            correct_vals = x[labels == 1]
            incorrect_vals = x[labels == 0]
            d = cohens_d(correct_vals, incorrect_vals)

            metric_results.append({
                "layer": l,
                "auroc": round(float(auroc), 6),
                "auprc": round(float(auprc), 6),
                "ece": round(float(ece), 6),
                "balanced_acc": round(float(ba), 6),
                "cohens_d": round(float(d), 6),
            })

            if auroc > best_auroc[0]:
                best_auroc = (auroc, l)

        all_metric_results.append({
            "metric": metric,
            "per_layer": metric_results,
            "best_layer": best_auroc[1],
            "best_auroc": round(best_auroc[0], 6),
        })

    log("AUROC", f"{cond_name} 완료 - "
        + ", ".join(f"{m['metric']}:L{m['best_layer']}({m['best_auroc']:.4f})"
                    for m in all_metric_results))
    return {
        "n_samples": len(labels),
        "n_correct": int(labels.sum()),
        "n_incorrect": int(len(labels) - labels.sum()),
        "accuracy": round(labels.mean() * 100, 2),
        "metrics": all_metric_results,
    }


# ============================================================================
# 2. Statistical Validation
# ============================================================================

def analysis_statistical_validation(data, labels, num_layers, cond_name):
    log("STAT", f"{cond_name} 시작")
    results = {}

    for metric_name, metric_key in [("unnormed", "unnormed_entropy"), ("normed", "normed_entropy")]:
        if metric_key not in data:
            continue

        metric_data = data[metric_key]
        correct_mask = labels == 1
        incorrect_mask = labels == 0

        raw_pvalues = []
        raw_d = []
        for l in range(num_layers):
            _, p = stats.mannwhitneyu(metric_data[correct_mask, l],
                                       metric_data[incorrect_mask, l],
                                       alternative='two-sided')
            d = cohens_d(metric_data[correct_mask, l], metric_data[incorrect_mask, l])
            raw_pvalues.append(p)
            raw_d.append(d)

        raw_pvalues = np.array(raw_pvalues)
        raw_d = np.array(raw_d)
        best_layer = int(np.argmax(np.abs(raw_d)))
        best_d = raw_d[best_layer]

        # 1. Bonferroni
        bonferroni_alpha = ALPHA / num_layers
        best_bonf = bool(raw_pvalues[best_layer] < bonferroni_alpha)
        bonferroni_sig = int(np.sum(raw_pvalues < bonferroni_alpha))

        # 2. BH-FDR
        sorted_idx = np.argsort(raw_pvalues)
        bh_threshold = np.array([(i+1) / num_layers * ALPHA for i in range(num_layers)])
        sorted_p = raw_pvalues[sorted_idx]
        bh_sig_mask = sorted_p <= bh_threshold
        if bh_sig_mask.any():
            bh_cutoff = np.max(np.where(bh_sig_mask)[0])
            bh_significant = np.zeros(num_layers, dtype=bool)
            bh_significant[sorted_idx[:bh_cutoff+1]] = True
            bh_sig_count = int(bh_significant.sum())
        else:
            bh_significant = np.zeros(num_layers, dtype=bool)
            bh_sig_count = 0
        best_bh = bool(bh_significant[best_layer]) if bh_sig_count > 0 else False

        # 3. Bootstrap CI
        correct_vals = metric_data[correct_mask, best_layer]
        incorrect_vals = metric_data[incorrect_mask, best_layer]
        bootstrap_d = []
        for _ in range(N_BOOTSTRAP):
            bc = np.random.choice(correct_vals, size=len(correct_vals), replace=True)
            bi = np.random.choice(incorrect_vals, size=len(incorrect_vals), replace=True)
            bootstrap_d.append(cohens_d(bc, bi))
        bootstrap_d = np.array(bootstrap_d)
        ci_lower = float(np.percentile(bootstrap_d, 2.5))
        ci_upper = float(np.percentile(bootstrap_d, 97.5))
        ci_excludes_zero = (ci_lower > 0) or (ci_upper < 0)

        # 4. 5-fold CV
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_best_layers = []
        for _, test_idx in skf.split(metric_data, labels):
            fold_data = metric_data[test_idx]
            fold_labels = labels[test_idx]
            fold_correct = fold_labels == 1
            fold_incorrect = fold_labels == 0
            if fold_correct.sum() < 2 or fold_incorrect.sum() < 2:
                continue
            fold_d = [abs(cohens_d(fold_data[fold_correct, l], fold_data[fold_incorrect, l]))
                      for l in range(num_layers)]
            fold_best_layers.append(int(np.argmax(fold_d)))

        cv_best_counts = {}
        for l in fold_best_layers:
            cv_best_counts[l] = cv_best_counts.get(l, 0) + 1
        cv_consistency = cv_best_counts.get(best_layer, 0)

        # 5. Permutation
        observed_d = abs(raw_d[best_layer])
        perm_count = 0
        for _ in range(N_PERMUTATION):
            perm_labels = np.random.permutation(labels)
            perm_d = abs(cohens_d(metric_data[perm_labels == 1, best_layer],
                                   metric_data[perm_labels == 0, best_layer]))
            if perm_d >= observed_d:
                perm_count += 1
        perm_p = perm_count / N_PERMUTATION

        tests_passed = sum([best_bonf, best_bh, ci_excludes_zero, perm_p < ALPHA])
        verdict = "ROBUST" if tests_passed >= 3 else "MODERATE" if tests_passed >= 2 else "WEAK"

        results[metric_name] = {
            "best_layer": best_layer,
            "best_d": round(float(best_d), 6),
            "raw_p": float(raw_pvalues[best_layer]),
            "bonferroni": {"pass": best_bonf, "sig_layers": bonferroni_sig},
            "bh_fdr": {"pass": best_bh, "sig_layers": bh_sig_count},
            "bootstrap": {"ci_lower": ci_lower, "ci_upper": ci_upper, "excludes_zero": ci_excludes_zero},
            "cv": {"best_in_folds": cv_consistency, "total_folds": len(fold_best_layers),
                   "fold_best_layers": fold_best_layers,
                   "distribution": {str(k): v for k, v in cv_best_counts.items()}},
            "permutation": {"p_value": perm_p, "pass": perm_p < ALPHA},
            "tests_passed": tests_passed,
            "verdict": verdict,
        }

        log("STAT", f"{cond_name} {metric_name}: L{best_layer} d={best_d:.4f} → {verdict} ({tests_passed}/4)")

    return results


# ============================================================================
# 3. Length-Controlled Analysis
# ============================================================================

def analysis_length_controlled(data, labels, lengths, num_layers, cond_name):
    log("LENGTH", f"{cond_name} 시작")

    # Skip if no valid lengths
    if np.all(lengths == 0):
        log("LENGTH", f"{cond_name} - num_tokens 없음, 스킵")
        return {"skipped": True, "reason": "no num_tokens"}

    result = {}

    # Length-correctness correlation
    corr, p_value = stats.pointbiserialr(labels, lengths)
    correct_lengths = lengths[labels == 1]
    incorrect_lengths = lengths[labels == 0]
    result["length_correctness"] = {
        "correlation": round(float(corr), 6),
        "p_value": float(p_value),
        "correct_mean_length": round(float(np.mean(correct_lengths)), 1),
        "incorrect_mean_length": round(float(np.mean(incorrect_lengths)), 1),
    }

    for metric_key in ['unnormed_entropy', 'normed_entropy']:
        if metric_key not in data:
            continue

        metric_data = data[metric_key]

        # Residualize
        residuals = np.zeros_like(metric_data)
        for l in range(num_layers):
            y = metric_data[:, l]
            if np.std(lengths) > 1e-12 and np.std(y) > 1e-12:
                slope, intercept = np.polyfit(lengths, y, 1)
                residuals[:, l] = y - (slope * lengths + intercept)
            else:
                residuals[:, l] = y

        # Compare original vs residual
        comparison = []
        for l in range(num_layers):
            try:
                orig_auroc = roc_auc_score(labels, -metric_data[:, l])
            except ValueError:
                orig_auroc = 0.5
            try:
                resid_auroc = roc_auc_score(labels, -residuals[:, l])
            except ValueError:
                resid_auroc = 0.5

            retention = (resid_auroc / orig_auroc * 100) if orig_auroc > 0.5 else None
            comparison.append({
                "layer": l,
                "original_auroc": round(float(orig_auroc), 6),
                "residual_auroc": round(float(resid_auroc), 6),
                "retention_pct": round(float(retention), 1) if retention is not None else None,
            })

        best_orig = max(comparison, key=lambda x: x['original_auroc'])
        best_layer = best_orig['layer']

        # Bootstrap CI for residual at best layer
        np.random.seed(SEED)
        n = len(labels)
        boot_aurocs = []
        for _ in range(N_BOOTSTRAP):
            idx = np.random.choice(n, n, replace=True)
            y = labels[idx]
            x = residuals[idx, best_layer]
            if len(np.unique(y)) < 2:
                continue
            try:
                boot_aurocs.append(roc_auc_score(y, -x))
            except ValueError:
                continue

        boot_aurocs = np.array(boot_aurocs)
        boot_ci = {
            "mean": round(float(np.mean(boot_aurocs)), 6),
            "ci_lower": round(float(np.percentile(boot_aurocs, 2.5)), 6),
            "ci_upper": round(float(np.percentile(boot_aurocs, 97.5)), 6),
            "above_chance": bool(np.percentile(boot_aurocs, 2.5) > 0.5),
        }

        result[metric_key] = {
            "best_original_layer": best_layer,
            "original_auroc": best_orig['original_auroc'],
            "residual_auroc": best_orig['residual_auroc'],
            "retention_pct": best_orig['retention_pct'],
            "bootstrap_residual": boot_ci,
            "per_layer": comparison,
        }

        log("LENGTH", f"{cond_name} {metric_key}: L{best_layer} "
            f"orig={best_orig['original_auroc']:.4f} → resid={best_orig['residual_auroc']:.4f} "
            f"retain={best_orig['retention_pct']}% above_chance={boot_ci['above_chance']}")

    return result


# ============================================================================
# 4. Nested CV
# ============================================================================

def analysis_nested_cv(data, labels, num_layers, cond_name):
    log("NESTED", f"{cond_name} 시작")
    results = {}

    for metric_key in ['unnormed_entropy', 'normed_entropy']:
        if metric_key not in data:
            continue

        X = data[metric_key]

        # Oracle
        oracle_aurocs = [safe_auroc(labels, X[:, l]) for l in range(num_layers)]
        oracle_best_layer = int(np.argmax(oracle_aurocs))
        oracle_best_auroc = oracle_aurocs[oracle_best_layer]

        # Nested CV
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        nested_aurocs = []
        selected_layers = []

        for train_idx, test_idx in skf.split(X, labels):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            train_aurocs = [safe_auroc(y_train, X_train[:, l]) for l in range(num_layers)]
            selected_layer = int(np.argmax(train_aurocs))
            selected_layers.append(selected_layer)
            nested_aurocs.append(safe_auroc(y_test, X_test[:, selected_layer]))

        nested_mean = float(np.mean(nested_aurocs))
        optimism = oracle_best_auroc - nested_mean

        layer_counts = {}
        for sl in selected_layers:
            layer_counts[sl] = layer_counts.get(sl, 0) + 1
        most_common = max(layer_counts, key=layer_counts.get)
        consistency = layer_counts[most_common] / N_FOLDS

        results[metric_key] = {
            "oracle_best_layer": oracle_best_layer,
            "oracle_auroc": round(oracle_best_auroc, 6),
            "nested_mean_auroc": round(nested_mean, 6),
            "optimism": round(optimism, 6),
            "selected_layers": selected_layers,
            "most_common_layer": most_common,
            "consistency": round(consistency, 4),
        }

        log("NESTED", f"{cond_name} {metric_key}: oracle L{oracle_best_layer}({oracle_best_auroc:.4f}) "
            f"nested={nested_mean:.4f} optimism={optimism:+.4f} consist={consistency*100:.0f}%")

    return results


# ============================================================================
# 5. Mediation Bootstrap
# ============================================================================

def analysis_mediation_bootstrap(data, labels, num_layers, cond_name):
    log("MEDIATION", f"{cond_name} 시작")

    if 'unnormed_entropy' not in data or 'h_norm' not in data:
        return {"skipped": True}

    unnormed = data['unnormed_entropy']
    h_norm = data['h_norm']
    n = len(labels)

    aurocs = [safe_auroc(labels, unnormed[:, l]) for l in range(num_layers)]
    best_layer = int(np.argmax(aurocs))

    unn_l = unnormed[:, best_layer]
    h_l = h_norm[:, best_layer]

    def compute_med(u, h, y):
        orig = safe_auroc(y, u)
        if np.std(h) > 0:
            slope, intercept = np.polyfit(h, u, 1)
            residual = u - (slope * h + intercept)
            resid = safe_auroc(y, residual)
        else:
            resid = orig
        med = (orig - resid) / (orig - 0.5) * 100 if orig > 0.5 else 0.0
        return orig, resid, med

    orig_auroc, res_auroc, med_pct = compute_med(unn_l, h_l, labels)

    boot_med = []
    for _ in range(N_BOOTSTRAP):
        idx = np.random.choice(n, size=n, replace=True)
        if len(np.unique(labels[idx])) < 2:
            continue
        _, _, m = compute_med(unn_l[idx], h_l[idx], labels[idx])
        boot_med.append(m)

    boot_med = np.array(boot_med)
    ci_med = (float(np.percentile(boot_med, 2.5)), float(np.percentile(boot_med, 97.5)))
    med_sig = ci_med[0] > 0

    log("MEDIATION", f"{cond_name}: L{best_layer} med={med_pct:.1f}% "
        f"CI=[{ci_med[0]:.1f}%, {ci_med[1]:.1f}%] {'SIG' if med_sig else 'n.s.'}")

    return {
        "best_layer": best_layer,
        "original_auroc": round(orig_auroc, 6),
        "residual_auroc": round(res_auroc, 6),
        "mediation_pct": round(med_pct, 2),
        "bootstrap_ci": {"lower": round(ci_med[0], 2), "upper": round(ci_med[1], 2)},
        "mediation_significant": med_sig,
    }


# ============================================================================
# 6. Risk-Coverage
# ============================================================================

def analysis_risk_coverage(data, labels, num_layers, cond_name):
    log("RISK-COV", f"{cond_name} 시작")

    def rc_curve(labels, confidence, coverages):
        n = len(labels)
        sorted_idx = np.argsort(-confidence)
        results = []
        for cov in coverages:
            k = max(1, int(n * cov))
            selected = sorted_idx[:k]
            acc = labels[selected].mean()
            results.append({"coverage": round(float(cov), 2), "accuracy": round(float(acc), 6)})
        return results

    def auc_rc(rc_data):
        covs = [d["coverage"] for d in rc_data]
        accs = [d["accuracy"] for d in rc_data]
        return float(np.trapz(accs, covs))

    unnormed = data.get('unnormed_entropy')
    normed = data.get('normed_entropy')
    h_norm_data = data.get('h_norm')

    if unnormed is None:
        return {"skipped": True}

    aurocs = [safe_auroc(labels, unnormed[:, l]) for l in range(num_layers)]
    best_unn_layer = int(np.argmax(aurocs))

    methods = {}

    # Unnormed best
    rc = rc_curve(labels, -unnormed[:, best_unn_layer], COVERAGE_LEVELS)
    methods[f"unnormed_L{best_unn_layer}"] = {"risk_coverage": rc, "auc_rc": round(auc_rc(rc), 6)}

    # Normed best
    if normed is not None:
        normed_aurocs = [safe_auroc(labels, normed[:, l]) for l in range(num_layers)]
        best_nor_layer = int(np.argmax(normed_aurocs))
        rc = rc_curve(labels, -normed[:, best_nor_layer], COVERAGE_LEVELS)
        methods[f"normed_L{best_nor_layer}"] = {"risk_coverage": rc, "auc_rc": round(auc_rc(rc), 6)}

    # Combined
    if normed is not None and h_norm_data is not None:
        X_comb = np.column_stack([normed[:, best_unn_layer], h_norm_data[:, best_unn_layer]])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        conf_comb = np.zeros(len(labels))
        for tr, te in skf.split(X_comb, labels):
            sc = StandardScaler()
            lr = LogisticRegression(random_state=SEED, max_iter=1000)
            lr.fit(sc.fit_transform(X_comb[tr]), labels[tr])
            conf_comb[te] = lr.predict_proba(sc.transform(X_comb[te]))[:, 1]
        rc = rc_curve(labels, conf_comb, COVERAGE_LEVELS)
        methods[f"combined_L{best_unn_layer}"] = {"risk_coverage": rc, "auc_rc": round(auc_rc(rc), 6)}

    # Random
    np.random.seed(SEED)
    rc = rc_curve(labels, np.random.rand(len(labels)), COVERAGE_LEVELS)
    methods["random"] = {"risk_coverage": rc, "auc_rc": round(auc_rc(rc), 6)}

    best_method = max(methods, key=lambda k: methods[k]["auc_rc"])
    log("RISK-COV", f"{cond_name}: best={best_method} AUC-RC={methods[best_method]['auc_rc']:.4f}")

    return {"baseline_accuracy": round(float(labels.mean()), 4), "methods": methods}


# ============================================================================
# 7. Output-Proxy Baseline
# ============================================================================

def analysis_output_proxy(data, labels, num_layers, last_layer, cond_name):
    log("OUTPUT-PROXY", f"{cond_name} 시작")

    unnormed = data.get('unnormed_entropy')
    if unnormed is None:
        return {"skipped": True}

    aurocs = [safe_auroc(labels, unnormed[:, l]) for l in range(num_layers)]
    best_layer = int(np.argmax(aurocs))

    auroc_best = aurocs[best_layer]
    auroc_last = aurocs[last_layer]

    log("OUTPUT-PROXY", f"{cond_name}: best L{best_layer} AUROC={auroc_best:.4f} vs "
        f"output L{last_layer} AUROC={auroc_last:.4f} Δ={auroc_best-auroc_last:+.4f}")

    return {
        "best_layer": best_layer,
        "last_layer": last_layer,
        "auroc_best": round(auroc_best, 6),
        "auroc_last": round(auroc_last, 6),
        "auroc_delta": round(auroc_best - auroc_last, 6),
    }


# ============================================================================
# 8. Decomposition (Q1/Q2)
# ============================================================================

def analysis_decomposition(data, labels, num_layers, cond_name):
    log("DECOMP", f"{cond_name} 시작")

    normed = data.get('normed_entropy')
    unnormed = data.get('unnormed_entropy')
    h_norm_data = data.get('h_norm')

    if normed is None or unnormed is None or h_norm_data is None:
        return {"skipped": True}

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Q1: per-layer combined vs unnormed
    best_comb = (0, 0)
    best_unn = (0, 0)
    layer_results = []

    for l in range(num_layers):
        unn_auroc = safe_auroc(labels, unnormed[:, l])

        X_comb = np.column_stack([normed[:, l], h_norm_data[:, l]])
        y_scores = np.zeros(len(labels))
        for tr, te in skf.split(X_comb, labels):
            sc = StandardScaler()
            lr = LogisticRegression(random_state=SEED, max_iter=1000)
            lr.fit(sc.fit_transform(X_comb[tr]), labels[tr])
            y_scores[te] = lr.predict_proba(sc.transform(X_comb[te]))[:, 1]

        comb_auroc = roc_auc_score(labels, y_scores)
        delta = comb_auroc - unn_auroc

        layer_results.append({
            "layer": l,
            "unnormed_auroc": round(unn_auroc, 6),
            "combined_auroc": round(comb_auroc, 6),
            "delta": round(delta, 6),
        })

        if comb_auroc > best_comb[0]:
            best_comb = (comb_auroc, l)
        if unn_auroc > best_unn[0]:
            best_unn = (unn_auroc, l)

    q1_delta = best_comb[0] - best_unn[0]
    q1_answer = "YES" if q1_delta > 0.005 else "NO"

    # Q2: multi-layer decomposed vs single best
    X_multi = np.column_stack([normed, h_norm_data])
    y_scores = np.zeros(len(labels))
    for tr, te in skf.split(X_multi, labels):
        sc = StandardScaler()
        lr = LogisticRegression(penalty='l1', solver='saga', C=0.1,
                                random_state=SEED, max_iter=2000)
        lr.fit(sc.fit_transform(X_multi[tr]), labels[tr])
        y_scores[te] = lr.predict_proba(sc.transform(X_multi[te]))[:, 1]

    multi_auroc = roc_auc_score(labels, y_scores)
    q2_delta = multi_auroc - best_unn[0]
    q2_answer = "YES" if q2_delta > 0.005 else "NO"

    log("DECOMP", f"{cond_name}: Q1={q1_answer}(Δ={q1_delta:+.4f}) "
        f"Q2={q2_answer}(Δ={q2_delta:+.4f})")

    return {
        "best_combined": {"auroc": round(best_comb[0], 6), "layer": best_comb[1]},
        "best_unnormed": {"auroc": round(best_unn[0], 6), "layer": best_unn[1]},
        "multi_layer_auroc": round(multi_auroc, 6),
        "Q1_delta": round(q1_delta, 6),
        "Q1_answer": q1_answer,
        "Q2_delta": round(q2_delta, 6),
        "Q2_answer": q2_answer,
        "per_layer": layer_results,
    }


# ============================================================================
# 메인
# ============================================================================

def main():
    start = datetime.now()

    print(f"\n{'#'*100}")
    print(f"  MMLU 3-Model Post-Processing Pipeline")
    print(f"  Models: Qwen2.5-7B / Llama-3-8B / Mistral-7B")
    print(f"  Analyses: AUROC/ECE + StatVal + Length + NestedCV + Mediation + RiskCov + OutputProxy + Decomp")
    print(f"  {start.isoformat()}")
    print(f"{'#'*100}\n")

    all_results = {
        "timestamp": start.isoformat(),
        "seed": SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "n_permutation": N_PERMUTATION,
        "n_folds": N_FOLDS,
        "experiments": {},
    }

    for cond_name, info in EXPERIMENTS.items():
        path = info["path"]
        num_layers = info["num_layers"]
        last_layer = info["last_layer"]
        model = info["model"]

        if not path.exists():
            log("SKIP", f"{cond_name}: 파일 없음 - {path}")
            continue

        print(f"\n\n{'='*100}")
        print(f"  {cond_name.upper()} | {model} | {num_layers} layers")
        print(f"{'='*100}")

        log("LOAD", f"{cond_name} 데이터 로드 중...")
        data, labels, lengths, avail_metrics = load_full_data(path, num_layers)
        n = len(labels)
        n_correct = int(labels.sum())
        log("LOAD", f"{n} samples, {n_correct} correct ({n_correct/n*100:.1f}%), "
            f"metrics={avail_metrics}")

        cond_result = {
            "model": model,
            "num_layers": num_layers,
            "n_samples": n,
            "n_correct": n_correct,
            "accuracy": round(n_correct/n*100, 2),
        }

        # 1. AUROC/ECE
        cond_result["auroc_ece"] = analysis_auroc_ece(
            data, labels, num_layers, avail_metrics, cond_name)

        # 2. Statistical Validation
        cond_result["statistical_validation"] = analysis_statistical_validation(
            data, labels, num_layers, cond_name)

        # 3. Length-Controlled
        cond_result["length_controlled"] = analysis_length_controlled(
            data, labels, lengths, num_layers, cond_name)

        # 4. Nested CV
        cond_result["nested_cv"] = analysis_nested_cv(
            data, labels, num_layers, cond_name)

        # 5. Mediation Bootstrap
        cond_result["mediation_bootstrap"] = analysis_mediation_bootstrap(
            data, labels, num_layers, cond_name)

        # 6. Risk-Coverage
        cond_result["risk_coverage"] = analysis_risk_coverage(
            data, labels, num_layers, cond_name)

        # 7. Output-Proxy
        cond_result["output_proxy"] = analysis_output_proxy(
            data, labels, num_layers, last_layer, cond_name)

        # 8. Decomposition
        cond_result["decomposition"] = analysis_decomposition(
            data, labels, num_layers, cond_name)

        all_results["experiments"][cond_name] = cond_result

    # ========================================================================
    # Cross-Model Comparison Summary
    # ========================================================================
    print(f"\n\n{'#'*100}")
    print(f"  CROSS-MODEL SUMMARY")
    print(f"{'#'*100}")

    print(f"\n{'Model':>20} | {'Acc':>6} | {'Unn Best L':>10} | {'Unn AUROC':>9} | "
          f"{'Nor Best L':>10} | {'Nor AUROC':>9} | {'StatVal':>10}")
    print("-" * 95)

    for cond_name, cond_data in all_results["experiments"].items():
        acc = cond_data["accuracy"]

        # AUROC best layers
        auroc_data = cond_data["auroc_ece"]["metrics"]
        unn_m = next((m for m in auroc_data if m["metric"] == "unnormed_entropy"), None)
        nor_m = next((m for m in auroc_data if m["metric"] == "normed_entropy"), None)

        unn_str = f"L{unn_m['best_layer']}({unn_m['best_auroc']:.4f})" if unn_m else "N/A"
        nor_str = f"L{nor_m['best_layer']}({nor_m['best_auroc']:.4f})" if nor_m else "N/A"

        # StatVal verdict
        sv = cond_data["statistical_validation"]
        unn_verdict = sv.get("unnormed", {}).get("verdict", "N/A")
        nor_verdict = sv.get("normed", {}).get("verdict", "N/A")
        stat_str = f"U:{unn_verdict}/N:{nor_verdict}"

        print(f"{cond_data['model']:>20} | {acc:>5.1f}% | {unn_str:>10} | "
              f"{'':>9} | {nor_str:>10} | {'':>9} | {stat_str:>10}")

    print(f"\n{'Model':>20} | {'Med%':>8} | {'Med CI':>20} | {'Med Sig?':>8} | "
          f"{'Q1':>4} | {'Q2':>4} | {'Output Δ':>10}")
    print("-" * 90)

    for cond_name, cond_data in all_results["experiments"].items():
        med = cond_data["mediation_bootstrap"]
        decomp = cond_data["decomposition"]
        proxy = cond_data["output_proxy"]

        if "skipped" not in med:
            med_str = f"{med['mediation_pct']:>7.1f}%"
            ci_str = f"[{med['bootstrap_ci']['lower']:.1f}%, {med['bootstrap_ci']['upper']:.1f}%]"
            sig_str = "YES" if med['mediation_significant'] else "NO"
        else:
            med_str = ci_str = sig_str = "N/A"

        q1 = decomp.get("Q1_answer", "N/A") if "skipped" not in decomp else "N/A"
        q2 = decomp.get("Q2_answer", "N/A") if "skipped" not in decomp else "N/A"
        delta = f"{proxy.get('auroc_delta', 0):+.4f}" if "skipped" not in proxy else "N/A"

        print(f"{cond_data['model']:>20} | {med_str:>8} | {ci_str:>20} | {sig_str:>8} | "
              f"{q1:>4} | {q2:>4} | {delta:>10}")

    # ========================================================================
    # Save
    # ========================================================================
    MMLU_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MMLU_DIR / "mmlu_postprocessing_results.json"

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    elapsed = datetime.now() - start
    print(f"\n{'#'*100}")
    print(f"  완료: {elapsed.total_seconds():.1f}초")
    print(f"  결과: {out_path}")
    print(f"{'#'*100}")
    log("DONE", f"MMLU 후처리 완료 ({elapsed.total_seconds():.1f}초) → {out_path}")


if __name__ == "__main__":
    main()
