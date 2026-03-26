"""
Phase 2a: Entropy-Lens Faithful Baseline
=========================================
GPT 12차 4개 응답 전면 수렴:
  "H_post 단일 스칼라를 Entropy-Lens 직접 비교라고 부르면 안 됨.
   최소한 H_post profile + aggregate + k-NN 한 줄은 그대로 가져와야 함."

Entropy-Lens 원 논문 파이프라인:
  1. 각 생성 토큰에서 intermediate prediction (ln_f + lm_head) → entropy 계산
  2. 레이어별 entropy를 profile로 구성 (L-dim vector)
  3. 토큰 축으로 aggregate (mean or concat)
  4. k-NN classifier로 correctness 분류
  5. ROC-AUC 평가

우리 데이터 대응:
  - H_post (normed_entropy) = ln_f 후 lm_head 경로 → EL의 entropy 정의와 대응
  - 기존 MMLU 데이터: 토큰 평균된 레이어별 H_post = mean-aggregated profile
  - Phase 1 데이터: per-token 저장 → 더 정밀한 재현 가능 (Qwen만)

비교 구조 (GPT 12 수렴):
  A. EL-original: H_post profile + k-NN (faithful reproduction)
  B. EL-matched: H_post profile + LogisticRegression (classifier 통제)
  C. 우리 방법: H_pre single-layer (held-out best layer)
  D. 우리 profile: H_pre profile + same classifier
  → feature 차이와 classifier 차이를 분리

원칙: 실측값만. 추정/할루시네이션 절대 없음.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

SEED = 42
BASE_DIR = Path(__file__).parent.parent / "PoT_Experiment_Entropy_Attention_Extraction_Experiment"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
OUTPUT_DIR = EXPERIMENTS_DIR / "34_Phase2_Entropy_Lens_Baseline"

# ============================================================
# Data sources
# ============================================================
MMLU_CONFIGS = {
    "mmlu_qwen": {
        "path": "31_MMLU_Domain_Extension/EXP_20260219_053638_mmlu_qwen/data/sample_results.json",
        "num_layers": 28,
        "model": "Qwen2.5-7B-Instruct",
    },
    "mmlu_llama": {
        "path": "31_MMLU_Domain_Extension/EXP_20260219_171237_mmlu_llama/data/sample_results.json",
        "num_layers": 32,
        "model": "Llama-3-8B-Instruct",
    },
    "mmlu_mistral": {
        "path": "31_MMLU_Domain_Extension/EXP_20260220_000610_mmlu_mistral/data/sample_results.json",
        "num_layers": 32,
        "model": "Mistral-7B-Instruct-v0.3",
    },
}

# Also include Qwen Hard for cross-task comparison
EXTRA_CONFIGS = {
    "qwen_hard": {
        "path": "23_Normed_Difficulty_Analysis/EXP_20260213_113717_normed_hard/data/sample_results.json",
        "num_layers": 28,
        "model": "Qwen2.5-7B-Instruct",
        "dataset": "Math Hard (L4-5)",
    },
}

PROFILE_METRICS = ["normed_entropy", "unnormed_entropy", "h_norm", "logit_std"]


def safe_auroc(y, scores):
    if len(np.unique(y)) < 2 or len(y) < 10:
        return 0.5
    try:
        return float(roc_auc_score(y, scores))
    except ValueError:
        return 0.5


def load_profiles(config):
    """Load per-sample, per-layer metric profiles from existing data."""
    filepath = EXPERIMENTS_DIR / config["path"]
    with open(filepath, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if isinstance(samples, dict) and "results" in samples:
        samples = samples["results"]

    num_layers = config["num_layers"]
    labels = []
    profiles = {m: [] for m in PROFILE_METRICS}

    for s in samples:
        if "layer_data" not in s or "is_correct" not in s:
            continue
        ld = s["layer_data"]
        layer_keys = [str(l) for l in range(num_layers)]
        if not all(k in ld for k in layer_keys):
            continue

        valid = True
        row = {m: [] for m in PROFILE_METRICS}
        for lk in layer_keys:
            d = ld[lk]
            for m in PROFILE_METRICS:
                if m not in d:
                    valid = False
                    break
                v = d[m]
                if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                    v = np.nan
                row[m].append(v)
            if not valid:
                break
        if not valid:
            continue

        labels.append(1 if s["is_correct"] else 0)
        for m in PROFILE_METRICS:
            profiles[m].append(row[m])

    labels = np.array(labels)
    for m in PROFILE_METRICS:
        profiles[m] = np.array(profiles[m], dtype=np.float64)

    return labels, profiles, num_layers


def load_phase0_split(condition_name):
    """Load Phase 0 split indices for consistency."""
    split_file = EXPERIMENTS_DIR / "32_Phase0_CalTest_Baselines" / f"{condition_name}_baselines.json"
    if not split_file.exists():
        return None, None
    with open(split_file, "r") as f:
        data = json.load(f)
    return np.array(data["split"]["cal_indices"]), np.array(data["split"]["test_indices"])


# ============================================================
# Classifiers
# ============================================================
def knn_auroc(X_train, y_train, X_test, y_test, k=3):
    """k-NN classifier AUROC (EL faithful)."""
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return 0.5
    # Handle NaN
    mask_train = ~np.any(np.isnan(X_train), axis=1)
    mask_test = ~np.any(np.isnan(X_test), axis=1)
    Xtr, ytr = X_train[mask_train], y_train[mask_train]
    Xte, yte = X_test[mask_test], y_test[mask_test]
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2 or len(ytr) < k:
        return 0.5

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    clf = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    clf.fit(Xtr_s, ytr)
    proba = clf.predict_proba(Xte_s)
    if proba.shape[1] < 2:
        return 0.5
    return safe_auroc(yte, proba[:, 1])


def lr_auroc(X_train, y_train, X_test, y_test):
    """Logistic Regression AUROC (matched classifier)."""
    mask_train = ~np.any(np.isnan(X_train), axis=1)
    mask_test = ~np.any(np.isnan(X_test), axis=1)
    Xtr, ytr = X_train[mask_train], y_train[mask_train]
    Xte, yte = X_test[mask_test], y_test[mask_test]
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2 or len(ytr) < 10:
        return 0.5

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)),
    ])
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)
    if proba.shape[1] < 2:
        return 0.5
    return safe_auroc(yte, proba[:, 1])


def cv_auroc(X, y, classifier="knn", k=3, n_folds=10):
    """Cross-validated AUROC (for comparison with EL paper's 10-fold CV)."""
    mask = ~np.any(np.isnan(X), axis=1)
    X_clean, y_clean = X[mask], y[mask]
    if len(np.unique(y_clean)) < 2 or len(y_clean) < 20:
        return 0.5, []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_aurocs = []

    for train_idx, test_idx in skf.split(X_clean, y_clean):
        Xtr, ytr = X_clean[train_idx], y_clean[train_idx]
        Xte, yte = X_clean[test_idx], y_clean[test_idx]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        if classifier == "knn":
            clf = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        else:
            clf = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)

        clf.fit(Xtr_s, ytr)
        proba = clf.predict_proba(Xte_s)
        if proba.shape[1] < 2:
            continue
        fold_aurocs.append(safe_auroc(yte, proba[:, 1]))

    if not fold_aurocs:
        return 0.5, []
    return round(float(np.mean(fold_aurocs)), 6), [round(a, 6) for a in fold_aurocs]


def single_layer_auroc(y_cal, X_cal, y_test, X_test, num_layers):
    """Best single-layer AUROC with cal/test protocol."""
    best_layer, best_sign, best_cal = 0, 1, -1
    for l in range(num_layers):
        col = X_cal[:, l]
        mask = ~np.isnan(col)
        if mask.sum() < 10:
            continue
        a_pos = safe_auroc(y_cal[mask], col[mask])
        a_neg = safe_auroc(y_cal[mask], -col[mask])
        if a_pos >= a_neg and a_pos > best_cal:
            best_layer, best_sign, best_cal = l, 1, a_pos
        elif a_neg > a_pos and a_neg > best_cal:
            best_layer, best_sign, best_cal = l, -1, a_neg

    test_scores = best_sign * X_test[:, best_layer]
    test_auroc = safe_auroc(y_test, test_scores)
    return {
        "layer": int(best_layer),
        "sign": int(best_sign),
        "cal_auroc": round(best_cal, 6),
        "test_auroc": round(test_auroc, 6),
    }


# ============================================================
# Main analysis
# ============================================================
def analyze_condition(name, config):
    print(f"\n{'='*70}")
    print(f"Condition: {name} ({config['model']})")
    print(f"{'='*70}")

    labels, profiles, num_layers = load_profiles(config)
    print(f"  Loaded: {len(labels)} samples, {labels.sum()} correct ({labels.mean()*100:.1f}%), {num_layers} layers")

    # Use Phase 0 split if available, else create new
    cal_idx, test_idx = load_phase0_split(name)
    if cal_idx is None:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
        cal_idx, test_idx = next(splitter.split(np.zeros(len(labels)), labels))
        print(f"  New split: cal={len(cal_idx)}, test={len(test_idx)}")
    else:
        print(f"  Phase 0 split: cal={len(cal_idx)}, test={len(test_idx)}")

    y_cal, y_test = labels[cal_idx], labels[test_idx]

    results = {
        "condition": name,
        "model": config["model"],
        "n_samples": len(labels),
        "accuracy": round(float(labels.mean()), 6),
        "num_layers": num_layers,
        "split": {"cal": len(cal_idx), "test": len(test_idx)},
        "methods": {},
        "cv_results": {},
        "timestamp": datetime.now().isoformat(),
    }

    # ============================================================
    # A. EL-original: H_post profile + k-NN (faithful reproduction)
    # ============================================================
    print(f"\n  --- A. EL-original (H_post profile + k-NN) ---")

    X_hpost = profiles["normed_entropy"]
    X_cal_hpost, X_test_hpost = X_hpost[cal_idx], X_hpost[test_idx]

    for k in [3, 5, 7]:
        auroc = knn_auroc(X_cal_hpost, y_cal, X_test_hpost, y_test, k=k)
        key = f"EL_original_knn_k{k}"
        results["methods"][key] = {
            "feature": "H_post (normed_entropy) profile",
            "classifier": f"k-NN (k={k}, Euclidean, StandardScaler)",
            "type": "profile",
            "n_features": num_layers,
            "test_auroc": round(auroc, 6),
        }
        print(f"    k-NN k={k}: AUROC={auroc:.4f}")

    # 10-fold CV (for comparison with EL paper)
    cv_mean, cv_folds = cv_auroc(X_hpost, labels, classifier="knn", k=3, n_folds=10)
    results["cv_results"]["EL_original_knn_k3_10cv"] = {
        "mean_auroc": cv_mean,
        "fold_aurocs": cv_folds,
        "n_folds": len(cv_folds),
    }
    print(f"    10-fold CV k=3: mean AUROC={cv_mean:.4f} (folds: {len(cv_folds)})")

    # ============================================================
    # B. EL-matched: H_post profile + LogisticRegression
    # ============================================================
    print(f"\n  --- B. EL-matched (H_post profile + LR) ---")

    auroc_lr = lr_auroc(X_cal_hpost, y_cal, X_test_hpost, y_test)
    results["methods"]["EL_matched_LR"] = {
        "feature": "H_post (normed_entropy) profile",
        "classifier": "LogisticRegression (C=1.0, StandardScaler)",
        "type": "profile",
        "n_features": num_layers,
        "test_auroc": round(auroc_lr, 6),
    }
    print(f"    LR: AUROC={auroc_lr:.4f}")

    cv_mean_lr, cv_folds_lr = cv_auroc(X_hpost, labels, classifier="lr", n_folds=10)
    results["cv_results"]["EL_matched_LR_10cv"] = {
        "mean_auroc": cv_mean_lr,
        "fold_aurocs": cv_folds_lr,
    }
    print(f"    10-fold CV LR: mean AUROC={cv_mean_lr:.4f}")

    # ============================================================
    # C. Our method: H_pre single-layer (held-out best layer)
    # ============================================================
    print(f"\n  --- C. Our H_pre single-layer ---")

    X_hpre = profiles["unnormed_entropy"]
    X_cal_hpre, X_test_hpre = X_hpre[cal_idx], X_hpre[test_idx]
    sl_result = single_layer_auroc(y_cal, X_cal_hpre, y_test, X_test_hpre, num_layers)
    results["methods"]["our_H_pre_single_layer"] = {
        "feature": "H_pre (unnormed_entropy) best single layer",
        "classifier": "raw scalar (no model)",
        "type": "single_layer",
        **sl_result,
    }
    print(f"    Layer {sl_result['layer']} ({'+'if sl_result['sign']==1 else'-'}): "
          f"AUROC={sl_result['test_auroc']:.4f}")

    # H_post single-layer for comparison
    sl_hpost = single_layer_auroc(y_cal, X_cal_hpost, y_test, X_test_hpost, num_layers)
    results["methods"]["H_post_single_layer"] = {
        "feature": "H_post (normed_entropy) best single layer",
        "classifier": "raw scalar",
        "type": "single_layer",
        **sl_hpost,
    }
    print(f"    H_post single L{sl_hpost['layer']}: AUROC={sl_hpost['test_auroc']:.4f}")

    # ============================================================
    # D. Our H_pre profile + same classifiers
    # ============================================================
    print(f"\n  --- D. Our H_pre profile ---")

    auroc_hpre_knn = knn_auroc(X_cal_hpre, y_cal, X_test_hpre, y_test, k=3)
    results["methods"]["our_H_pre_profile_knn"] = {
        "feature": "H_pre (unnormed_entropy) profile",
        "classifier": "k-NN (k=3)",
        "type": "profile",
        "n_features": num_layers,
        "test_auroc": round(auroc_hpre_knn, 6),
    }
    print(f"    k-NN k=3: AUROC={auroc_hpre_knn:.4f}")

    auroc_hpre_lr = lr_auroc(X_cal_hpre, y_cal, X_test_hpre, y_test)
    results["methods"]["our_H_pre_profile_LR"] = {
        "feature": "H_pre (unnormed_entropy) profile",
        "classifier": "LogisticRegression",
        "type": "profile",
        "n_features": num_layers,
        "test_auroc": round(auroc_hpre_lr, 6),
    }
    print(f"    LR: AUROC={auroc_hpre_lr:.4f}")

    # ============================================================
    # E. Additional profile baselines (h_norm, logit_std)
    # ============================================================
    print(f"\n  --- E. Scale proxy profiles ---")

    for metric_name in ["h_norm", "logit_std"]:
        X = profiles[metric_name]
        X_cal_m, X_test_m = X[cal_idx], X[test_idx]

        auroc_knn = knn_auroc(X_cal_m, y_cal, X_test_m, y_test, k=3)
        auroc_lr_m = lr_auroc(X_cal_m, y_cal, X_test_m, y_test)
        sl = single_layer_auroc(y_cal, X_cal_m, y_test, X_test_m, num_layers)

        results["methods"][f"{metric_name}_profile_knn"] = {
            "feature": f"{metric_name} profile",
            "classifier": "k-NN (k=3)",
            "type": "profile",
            "test_auroc": round(auroc_knn, 6),
        }
        results["methods"][f"{metric_name}_profile_LR"] = {
            "feature": f"{metric_name} profile",
            "classifier": "LR",
            "type": "profile",
            "test_auroc": round(auroc_lr_m, 6),
        }
        results["methods"][f"{metric_name}_single_layer"] = {
            "feature": f"{metric_name} best single layer",
            "classifier": "raw scalar",
            "type": "single_layer",
            **sl,
        }

        print(f"    {metric_name:12s} | single L{sl['layer']}: {sl['test_auroc']:.4f} | "
              f"kNN: {auroc_knn:.4f} | LR: {auroc_lr_m:.4f}")

    # ============================================================
    # Summary table
    # ============================================================
    print(f"\n  {'─'*75}")
    print(f"  SUMMARY (sorted by Test AUROC)")
    print(f"  {'─'*75}")
    print(f"  {'Method':45s} {'Type':>10s} {'AUROC':>8s}")
    print(f"  {'─'*75}")

    sorted_methods = sorted(results["methods"].items(), key=lambda x: x[1]["test_auroc"], reverse=True)
    for mname, mdata in sorted_methods:
        print(f"  {mname:45s} {mdata['type']:>10s} {mdata['test_auroc']:8.4f}")
    print(f"  {'─'*75}")

    return results


def main():
    print("=" * 70)
    print("Phase 2a: Entropy-Lens Faithful Baseline")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = {}

    # MMLU conditions (EL paper anchor)
    for name, config in MMLU_CONFIGS.items():
        config["dataset"] = "MMLU"
        result = analyze_condition(name, config)
        all_results[name] = result

    # Extra conditions
    for name, config in EXTRA_CONFIGS.items():
        result = analyze_condition(name, config)
        all_results[name] = result

    # Save individual results
    for name, result in all_results.items():
        outpath = OUTPUT_DIR / f"{name}_el_baseline.json"
        with open(outpath, "w") as f:
            json.dump(result, f, indent=2)

    # ============================================================
    # Cross-condition comparison
    # ============================================================
    print(f"\n\n{'='*90}")
    print("CROSS-CONDITION: EL-original vs Our Method")
    print(f"{'='*90}")
    print(f"{'Condition':15s} {'EL-orig(k3)':>12s} {'EL-LR':>8s} {'H_pre-1L':>10s} {'H_pre-kNN':>10s} {'H_pre-LR':>9s} {'h_norm-LR':>10s} {'logit_std-LR':>13s}")
    print("-" * 90)

    summary = {}
    for name, result in all_results.items():
        m = result["methods"]
        el_k3 = m.get("EL_original_knn_k3", {}).get("test_auroc", "N/A")
        el_lr = m.get("EL_matched_LR", {}).get("test_auroc", "N/A")
        hpre_1l = m.get("our_H_pre_single_layer", {}).get("test_auroc", "N/A")
        hpre_knn = m.get("our_H_pre_profile_knn", {}).get("test_auroc", "N/A")
        hpre_lr = m.get("our_H_pre_profile_LR", {}).get("test_auroc", "N/A")
        hn_lr = m.get("h_norm_profile_LR", {}).get("test_auroc", "N/A")
        ls_lr = m.get("logit_std_profile_LR", {}).get("test_auroc", "N/A")

        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"{name:15s} {fmt(el_k3):>12s} {fmt(el_lr):>8s} {fmt(hpre_1l):>10s} {fmt(hpre_knn):>10s} {fmt(hpre_lr):>9s} {fmt(hn_lr):>10s} {fmt(ls_lr):>13s}")

        summary[name] = {
            "EL_original_knn_k3": el_k3,
            "EL_matched_LR": el_lr,
            "H_pre_single_layer": hpre_1l,
            "H_pre_profile_knn": hpre_knn,
            "H_pre_profile_LR": hpre_lr,
            "h_norm_profile_LR": hn_lr,
            "logit_std_profile_LR": ls_lr,
        }

    summary_path = OUTPUT_DIR / "el_baseline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved: {summary_path}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
