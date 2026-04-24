"""
Experiment 49 Extras: All remaining analyses not in Phase 2/3

Covers:
  A. Table 2 analog (multi-layer profile, 70/30 held-out)
  B. Table 7 analog (decoder comparison: LL vs TL vs H_post, Step 0, greedy labels)
  C. Table 12 analog (TL Step 0, greedy labels)
  D. Agreement separate file
  E. Token-position comparison (Step 0 + Full avg from greedy, greedy labels)
  F. Appendix J length confound (greedy data)

Input:
  - Greedy gen-avg: phase1_greedy/{model}/data/sample_results.json
  - Step 0 features: exp44 phase2_gpu_tokenpos_v2_aligned/{model}_mmlu_aligned/data/checkpoint.json
  - TL Step 0: exp46/{model}/step0_checkpoint.json
  - Sampled: exp31 sample_results.json

Usage:
  python scripts/run_exp49_extras.py
  python scripts/run_exp49_extras.py --smoke_test
"""

import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from _paths import POT_DIR

SEED = 42
BASE = Path(__file__).resolve().parent.parent
EXP31 = POT_DIR / "experiments" / "31_MMLU_Domain_Extension"
EXP44_TP = (POT_DIR
            / "experiments" / "44_GPT25_Experiments" / "phase2_gpu_tokenpos_v2_aligned")
EXP46 = POT_DIR / "experiments" / "46_Tuned_Lens_Control"
EXP49 = BASE / "experiments" / "49_Deterministic_Label_Robustness"

MODELS = {
    "qwen": {"exp31_dir": "EXP_20260219_053638_mmlu_qwen", "n_layers": 28,
             "tp_dir": "qwen_mmlu_aligned"},
    "llama": {"exp31_dir": "EXP_20260219_171237_mmlu_llama", "n_layers": 32,
              "tp_dir": "llama_mmlu_aligned"},
    "mistral": {"exp31_dir": "EXP_20260220_000610_mmlu_mistral", "n_layers": 32,
                "tp_dir": "mistral_mmlu_aligned"},
}
METRICS = ["unnormed_entropy", "normed_entropy", "logit_std", "h_norm"]
METRIC_LABELS = {"unnormed_entropy": "H_pre", "normed_entropy": "H_post",
                 "logit_std": "logit_std", "h_norm": "h_norm"}
N_BOOT = 1000


def log(msg):
    print(f"[EXP49-EX] {msg}", flush=True)


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def get_greedy_labels(model_key):
    p = EXP49 / "phase1_greedy" / model_key / "data" / "sample_results.json"
    if not p.exists():
        return None
    data = load_json(p)
    return {d["idx"]: d["is_correct"] for d in data}


def bootstrap_auroc_ci(y, scores, n_boot=N_BOOT):
    rng = np.random.RandomState(SEED)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y[idx], scores[idx]))
        except:
            continue
    if not aucs:
        return 0.5, 0.5
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


# ============================================================================
# A. Table 2 analog (multi-layer profile, 70/30 held-out)
# ============================================================================

def compute_table2(greedy_data, n_layers):
    """Profile AUROC with 70/30 held-out split."""
    n = len(greedy_data)
    y = np.array([1 if s["is_correct"] else 0 for s in greedy_data])

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
    cal_idx, test_idx = next(sss.split(np.zeros(n), y))

    results = {}
    for metric_key, metric_name in METRIC_LABELS.items():
        X = np.zeros((n, n_layers))
        for i, s in enumerate(greedy_data):
            if "layer_data" not in s:
                continue
            for layer in range(n_layers):
                if str(layer) in s["layer_data"] and metric_key in s["layer_data"][str(layer)]:
                    X[i, layer] = s["layer_data"][str(layer)][metric_key]

        sc = StandardScaler()
        lr = LogisticRegression(max_iter=1000)
        lr.fit(sc.fit_transform(X[cal_idx]), y[cal_idx])
        proba = lr.predict_proba(sc.transform(X[test_idx]))[:, 1]
        auc = roc_auc_score(y[test_idx], proba)
        ci_lo, ci_hi = bootstrap_auroc_ci(y[test_idx], proba)
        results[f"{metric_name}_profile"] = {
            "auroc": round(auc, 4), "ci": [round(ci_lo, 4), round(ci_hi, 4)]
        }

    # h_norm profile (dimension-matched L)
    # Already in individual metrics above

    return results


# ============================================================================
# B/C. Tables 7, 12 (TL relabeling with greedy labels)
# ============================================================================

def compute_tl_tables(model_key, greedy_labels, smoke_n=None):
    """TL Step 0 with greedy labels → Tables 7 and 12 analog."""
    cfg = MODELS[model_key]
    n_layers = cfg["n_layers"]

    # Load TL Step 0 per-sample data
    tl_path = EXP46 / model_key / "step0_checkpoint.json"
    if not tl_path.exists():
        return None

    tl_data = load_json(tl_path)
    if smoke_n:
        tl_data = tl_data[:smoke_n]

    n = len(tl_data)

    # Labels: greedy if available, else original
    if greedy_labels:
        y = np.array([1 if greedy_labels.get(i, False) else 0 for i in range(n)])
    else:
        y = np.array([1 if s["is_correct"] else 0 for s in tl_data])

    # Extract features: tl_entropy, ll_entropy (H_pre), post_entropy (H_post), logit_std, h_norm
    tl_metrics = ["tl_entropy", "ll_entropy", "post_entropy", "logit_std", "h_norm"]
    features = {m: np.zeros((n, n_layers)) for m in tl_metrics}

    for i, s in enumerate(tl_data):
        ld = s.get("layer_data", {})
        for layer in range(n_layers):
            if str(layer) in ld:
                for m in tl_metrics:
                    if m in ld[str(layer)]:
                        features[m][i, layer] = ld[str(layer)][m]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
    cal_idx, test_idx = next(sss.split(np.zeros(n), y))

    results = {}
    for metric in tl_metrics:
        X = features[metric]
        best_auc, best_layer, best_sign = 0, 0, 1
        for layer in range(n_layers):
            for sign in [+1, -1]:
                scores = X[cal_idx, layer] * sign
                try:
                    auc = roc_auc_score(y[cal_idx], scores)
                except:
                    auc = 0.5
                if auc > best_auc:
                    best_auc, best_layer, best_sign = auc, layer, sign

        test_scores = X[test_idx, best_layer] * best_sign
        test_auc = roc_auc_score(y[test_idx], test_scores)
        name_map = {"tl_entropy": "H_tl", "ll_entropy": "H_pre(LL)",
                     "post_entropy": "H_post", "logit_std": "logit_std", "h_norm": "h_norm"}
        results[name_map[metric]] = {
            "auroc": round(test_auc, 4), "layer": best_layer, "sign": best_sign
        }

    return results


# ============================================================================
# D. Agreement separate file
# ============================================================================

def save_agreement_file(model_key, greedy_data, sampled_data):
    """Save agreement as separate file in agreement/ directory."""
    out_dir = EXP49 / "agreement"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = min(len(greedy_data), len(sampled_data))
    g_correct = [greedy_data[i]["is_correct"] for i in range(n)]
    s_correct = [sampled_data[i]["is_correct"] for i in range(n)]
    g_pred = [greedy_data[i].get("predicted", "?") for i in range(n)]
    s_pred = [sampled_data[i].get("predicted", "?") for i in range(n)]

    agree = sum(1 for g, s in zip(g_correct, s_correct) if g == s)
    g_acc = sum(g_correct) / n
    s_acc = sum(s_correct) / n
    p_o = agree / n
    p_e = g_acc * s_acc + (1 - g_acc) * (1 - s_acc)
    kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 0

    result = {
        "model": model_key,
        "n_samples": n,
        "greedy_accuracy": round(g_acc, 4),
        "sampled_accuracy": round(s_acc, 4),
        "agreement_rate": round(p_o, 4),
        "cohens_kappa": round(kappa, 4),
        "greedy_correct_sampled_incorrect": sum(1 for g, s in zip(g_correct, s_correct) if g and not s),
        "greedy_incorrect_sampled_correct": sum(1 for g, s in zip(g_correct, s_correct) if not g and s),
        "both_correct": sum(1 for g, s in zip(g_correct, s_correct) if g and s),
        "both_incorrect": sum(1 for g, s in zip(g_correct, s_correct) if not g and not s),
        "per_sample": [
            {"idx": i, "greedy_correct": g_correct[i], "sampled_correct": s_correct[i],
             "greedy_pred": g_pred[i], "sampled_pred": s_pred[i], "agree": g_correct[i] == s_correct[i]}
            for i in range(n)
        ],
    }

    out_file = out_dir / f"sampled_vs_greedy_{model_key}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    log(f"  Agreement saved: {out_file}")
    return result


# ============================================================================
# E. Token-position comparison (Step 0 from exp44 + Full avg from greedy)
# ============================================================================

def compute_tokenpos(model_key, greedy_data, greedy_labels, smoke_n=None):
    """Token-position comparison: Step 0 vs Full avg, both with greedy labels."""
    cfg = MODELS[model_key]
    n_layers = cfg["n_layers"]

    # Full avg features from greedy generation
    n = len(greedy_data)
    y = np.array([1 if s["is_correct"] else 0 for s in greedy_data])

    # Step 0 features from exp44
    tp_path = EXP44_TP / cfg["tp_dir"] / "data" / "checkpoint.json"
    if not tp_path.exists():
        log(f"  Token-pos Step 0 data not found for {model_key}")
        return None

    tp_data = load_json(tp_path)
    if smoke_n:
        tp_data = tp_data[:smoke_n]

    n_tp = min(n, len(tp_data))

    # Step 0 labels = greedy labels
    if greedy_labels:
        y_s0 = np.array([1 if greedy_labels.get(i, False) else 0 for i in range(n_tp)])
    else:
        y_s0 = y[:n_tp]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)

    results = {}
    for metric_key, metric_name in METRIC_LABELS.items():
        # Full avg (from greedy)
        X_full = np.zeros((n, n_layers))
        for i, s in enumerate(greedy_data):
            if "layer_data" not in s:
                continue
            for layer in range(n_layers):
                if str(layer) in s["layer_data"] and metric_key in s["layer_data"][str(layer)]:
                    X_full[i, layer] = s["layer_data"][str(layer)][metric_key]

        cal_idx, test_idx = next(sss.split(np.zeros(n), y))
        best_auc_full, bl_full, bs_full = 0, 0, 1
        for layer in range(n_layers):
            for sign in [+1, -1]:
                try:
                    auc = roc_auc_score(y[cal_idx], X_full[cal_idx, layer] * sign)
                except:
                    auc = 0.5
                if auc > best_auc_full:
                    best_auc_full, bl_full, bs_full = auc, layer, sign
        test_auc_full = roc_auc_score(y[test_idx], X_full[test_idx, bl_full] * bs_full)

        # Step 0 (from exp44, with greedy labels)
        X_s0 = np.zeros((n_tp, n_layers))
        for i in range(n_tp):
            s0 = tp_data[i].get("position_data", {}).get("step0_prompt_last", {})
            for layer in range(n_layers):
                if str(layer) in s0 and metric_key in s0[str(layer)]:
                    X_s0[i, layer] = s0[str(layer)][metric_key]

        cal_idx_s0, test_idx_s0 = next(StratifiedShuffleSplit(
            n_splits=1, test_size=0.3, random_state=SEED).split(np.zeros(n_tp), y_s0))
        best_auc_s0, bl_s0, bs_s0 = 0, 0, 1
        for layer in range(n_layers):
            for sign in [+1, -1]:
                try:
                    auc = roc_auc_score(y_s0[cal_idx_s0], X_s0[cal_idx_s0, layer] * sign)
                except:
                    auc = 0.5
                if auc > best_auc_s0:
                    best_auc_s0, bl_s0, bs_s0 = auc, layer, sign
        test_auc_s0 = roc_auc_score(y_s0[test_idx_s0], X_s0[test_idx_s0, bl_s0] * bs_s0)

        results[metric_name] = {
            "step0": {"auroc": round(test_auc_s0, 4), "layer": bl_s0, "sign": bs_s0},
            "full_avg": {"auroc": round(test_auc_full, 4), "layer": bl_full, "sign": bs_full},
        }

    return results


# ============================================================================
# F. Appendix J length confound (greedy data)
# ============================================================================

def compute_length_confound(greedy_data, n_layers):
    """Length confound analysis using greedy generation data."""
    n = len(greedy_data)
    y = np.array([1 if s["is_correct"] else 0 for s in greedy_data])
    lengths = np.array([s.get("num_tokens", 0) for s in greedy_data], dtype=float)

    if lengths.std() < 1e-6:
        log("  Length has no variance, skipping confound analysis")
        return None

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
    cal_idx, test_idx = next(sss.split(np.zeros(n), y))

    results = {}
    for metric_key, metric_name in METRIC_LABELS.items():
        # Find best layer
        X = np.zeros((n, n_layers))
        for i, s in enumerate(greedy_data):
            if "layer_data" not in s:
                continue
            for layer in range(n_layers):
                if str(layer) in s["layer_data"] and metric_key in s["layer_data"][str(layer)]:
                    X[i, layer] = s["layer_data"][str(layer)][metric_key]

        best_auc, bl, bs = 0, 0, 1
        for layer in range(n_layers):
            for sign in [+1, -1]:
                try:
                    auc = roc_auc_score(y[cal_idx], X[cal_idx, layer] * sign)
                except:
                    auc = 0.5
                if auc > best_auc:
                    best_auc, bl, bs = auc, layer, sign

        scores = X[:, bl] * bs
        original_auc = roc_auc_score(y[test_idx], scores[test_idx])

        # Residualize against length
        from numpy.polynomial.polynomial import polyfit
        coeffs = polyfit(lengths[cal_idx], scores[cal_idx], 1)
        residuals = scores - (coeffs[0] + coeffs[1] * lengths)
        resid_auc = roc_auc_score(y[test_idx], residuals[test_idx])

        retain = ((resid_auc - 0.5) / (original_auc - 0.5) * 100) if original_auc != 0.5 else 100.0

        # Length-only baseline
        try:
            length_auc = max(roc_auc_score(y[test_idx], lengths[test_idx]),
                           roc_auc_score(y[test_idx], -lengths[test_idx]))
        except:
            length_auc = 0.5

        # Length + metric
        X_lm = np.column_stack([lengths, scores])
        sc = StandardScaler()
        lr = LogisticRegression(max_iter=1000)
        lr.fit(sc.fit_transform(X_lm[cal_idx]), y[cal_idx])
        lm_auc = roc_auc_score(y[test_idx], lr.predict_proba(sc.transform(X_lm[test_idx]))[:, 1])

        results[metric_name] = {
            "original_auroc": round(original_auc, 4),
            "residualized_auroc": round(resid_auc, 4),
            "retain_pct": round(retain, 1),
            "length_only_auroc": round(length_auc, 4),
            "length_plus_metric_auroc": round(lm_auc, 4),
            "incremental_over_length": round(lm_auc - length_auc, 4),
        }

    # Length stats
    correct_lengths = lengths[y == 1]
    incorrect_lengths = lengths[y == 0]
    results["length_stats"] = {
        "correct_mean": round(float(correct_lengths.mean()), 1) if len(correct_lengths) > 0 else 0,
        "incorrect_mean": round(float(incorrect_lengths.mean()), 1) if len(incorrect_lengths) > 0 else 0,
    }

    return results


# ============================================================================
# Main
# ============================================================================

def run_model(model_key, smoke_test=False):
    cfg = MODELS[model_key]
    n_layers = cfg["n_layers"]
    smoke_n = 100 if smoke_test else None

    # Load greedy data
    greedy_path = EXP49 / "phase1_greedy" / model_key / "data" / "sample_results.json"
    if not greedy_path.exists():
        if smoke_test:
            greedy_path = EXP31 / cfg["exp31_dir"] / "data" / "sample_results.json"
            log(f"  SMOKE: using exp31 as proxy for {model_key}")
        else:
            log(f"  SKIP {model_key}: greedy data not found")
            return None

    greedy_data = load_json(greedy_path)
    if smoke_n:
        greedy_data = greedy_data[:smoke_n]

    greedy_labels = get_greedy_labels(model_key)

    # Load sampled data
    sampled_path = EXP31 / cfg["exp31_dir"] / "data" / "sample_results.json"
    sampled_data = load_json(sampled_path) if sampled_path.exists() else None

    log(f"  Loaded {len(greedy_data)} greedy samples")

    result = {"model": model_key, "n_samples": len(greedy_data)}

    # A. Table 2
    log(f"  Computing Table 2 analog...")
    result["table2"] = compute_table2(greedy_data, n_layers)

    # B/C. TL tables
    log(f"  Computing TL tables (7, 12)...")
    tl = compute_tl_tables(model_key, greedy_labels, smoke_n=smoke_n)
    result["tl_step0"] = tl

    # D. Agreement
    if sampled_data and not smoke_test:
        log(f"  Saving agreement...")
        save_agreement_file(model_key, greedy_data, sampled_data)
    elif sampled_data and smoke_test:
        result["agreement_note"] = "smoke test: greedy=sampled proxy, agreement=100%"

    # E. Token-position
    log(f"  Computing token-position comparison...")
    result["tokenpos"] = compute_tokenpos(model_key, greedy_data, greedy_labels, smoke_n=smoke_n)

    # F. Length confound
    log(f"  Computing length confound...")
    result["length_confound"] = compute_length_confound(greedy_data, n_layers)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    log(f"Extras: {'SMOKE TEST' if args.smoke_test else 'FULL RUN'}")

    all_results = {}
    for model_key in ["qwen", "llama", "mistral"]:
        log(f"Processing {model_key}...")
        result = run_model(model_key, smoke_test=args.smoke_test)
        if result:
            all_results[model_key] = result

    suffix = "_smoke" if args.smoke_test else ""
    out_file = EXP49 / "phase2_recompute" / f"extras_results{suffix}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
