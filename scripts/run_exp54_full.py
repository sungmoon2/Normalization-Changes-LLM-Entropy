"""
Experiment 54 Full Run v3: Practical Consequence Demonstration
CPU only — uses existing MMLU sample_results.json (sampled + greedy)

Fixes v3:
  - Scenario A: nested bootstrap (re-split + re-select layer each iteration)
  - Scenario B: 20-split ranking stability test
  - Scenario C: per-layer sign re-optimization (v2 fix retained)

Usage:
  python scripts/run_exp54_full.py
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score
from _paths import POT_DIR

PROJECT_ROOT = Path(__file__).parent.parent
SEED = 42
N_BOOTSTRAP = 1000
N_RANKING_SPLITS = 20

DATA_SOURCES = {
    "sampled": {
        "qwen": POT_DIR / "experiments" / "31_MMLU_Domain_Extension" / "EXP_20260219_053638_mmlu_qwen" / "data" / "sample_results.json",
        "llama": POT_DIR / "experiments" / "31_MMLU_Domain_Extension" / "EXP_20260219_171237_mmlu_llama" / "data" / "sample_results.json",
        "mistral": POT_DIR / "experiments" / "31_MMLU_Domain_Extension" / "EXP_20260220_000610_mmlu_mistral" / "data" / "sample_results.json",
    },
    "greedy": {
        "qwen": PROJECT_ROOT / "experiments" / "49_Deterministic_Label_Robustness" / "phase1_greedy" / "qwen" / "data" / "sample_results.json",
        "llama": PROJECT_ROOT / "experiments" / "49_Deterministic_Label_Robustness" / "phase1_greedy" / "llama" / "data" / "sample_results.json",
        "mistral": PROJECT_ROOT / "experiments" / "49_Deterministic_Label_Robustness" / "phase1_greedy" / "mistral" / "data" / "sample_results.json",
    },
}

EXP_DIR = PROJECT_ROOT / "experiments" / "54_Practical_Consequence"
EXP_DIR.mkdir(parents=True, exist_ok=True)


def log(tag, msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}][{tag}] {msg}", flush=True)


def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [r for r in data if "layer_data" in r and "error" not in r]


def extract_features(data, num_layers):
    labels = np.array([1 if r["is_correct"] else 0 for r in data])
    h_pre = np.zeros((len(data), num_layers))
    h_post = np.zeros((len(data), num_layers))

    for i, r in enumerate(data):
        ld = r["layer_data"]
        for li in range(num_layers):
            li_str = str(li)
            if li_str in ld:
                h_pre[i, li] = ld[li_str]["unnormed_entropy"]
                h_post[i, li] = ld[li_str]["normed_entropy"]

    return labels, h_pre, h_post


def cal_test_split(n, rng, cal_ratio=0.7):
    indices = rng.permutation(n)
    n_cal = int(n * cal_ratio)
    return indices[:n_cal], indices[n_cal:]


def best_layer_sign(features, labels, cal_idx):
    n_layers = features.shape[1]
    best_auroc = -1
    best_layer = 0
    best_sign = 1
    for li in range(n_layers):
        vals = features[cal_idx, li]
        y = labels[cal_idx]
        for sign in [1, -1]:
            try:
                auroc = roc_auc_score(y, sign * vals)
            except:
                auroc = 0.5
            if auroc > best_auroc:
                best_auroc = auroc
                best_layer = li
                best_sign = sign
    return best_layer, best_sign, best_auroc


def best_sign_at_layer(features, labels, cal_idx, layer):
    vals = features[cal_idx, layer]
    y = labels[cal_idx]
    best_sign, best_auroc = 1, 0.5
    for sign in [1, -1]:
        try:
            a = roc_auc_score(y, sign * vals)
        except:
            a = 0.5
        if a > best_auroc:
            best_auroc = a
            best_sign = sign
    return best_sign


def selective_accuracy(scores, labels, coverage):
    n = len(scores)
    k = max(1, int(n * coverage))
    top_idx = np.argsort(-scores)[:k]
    return float(np.mean(labels[top_idx]))


# ============================================================================
# Scenario A: Nested Bootstrap (re-split + re-select layer each iteration)
# ============================================================================

def run_scenario_a_nested(h_pre, h_post, labels, n_boot=N_BOOTSTRAP):
    """Nested bootstrap: each iteration re-splits, re-selects layer, evaluates on test."""
    n = len(labels)

    # Point estimate with seed=42 split
    rng0 = np.random.RandomState(SEED)
    cal0, test0 = cal_test_split(n, rng0)
    l_pre, s_pre, _ = best_layer_sign(h_pre, labels, cal0)
    l_post, s_post, _ = best_layer_sign(h_post, labels, cal0)

    point_results = {}
    for cov in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        scores_pre = s_pre * h_pre[test0, l_pre]
        scores_post = s_post * h_post[test0, l_post]
        y_test = labels[test0]
        acc_pre = selective_accuracy(scores_pre, y_test, cov)
        acc_post = selective_accuracy(scores_post, y_test, cov)
        point_results[str(int(cov * 100))] = {
            "acc_H_pre": round(acc_pre, 4),
            "acc_H_post": round(acc_post, 4),
            "gap": round(acc_pre - acc_post, 4),
        }

    # Nested bootstrap: re-split each time
    rng_boot = np.random.RandomState(SEED + 1000)
    boot_gaps = {str(int(c * 100)): [] for c in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
    boot_layers_pre = []
    boot_layers_post = []

    for b in range(n_boot):
        cal_b, test_b = cal_test_split(n, np.random.RandomState(b))
        lb_pre, sb_pre, _ = best_layer_sign(h_pre, labels, cal_b)
        lb_post, sb_post, _ = best_layer_sign(h_post, labels, cal_b)
        boot_layers_pre.append(lb_pre)
        boot_layers_post.append(lb_post)

        y_b = labels[test_b]
        for cov in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            sc_pre = sb_pre * h_pre[test_b, lb_pre]
            sc_post = sb_post * h_post[test_b, lb_post]
            a_pre = selective_accuracy(sc_pre, y_b, cov)
            a_post = selective_accuracy(sc_post, y_b, cov)
            boot_gaps[str(int(cov * 100))].append(a_pre - a_post)

    results = {
        "point_estimate": {
            "best_layer_H_pre": int(l_pre),
            "best_sign_H_pre": int(s_pre),
            "best_layer_H_post": int(l_post),
            "best_sign_H_post": int(s_post),
        },
        "layer_selection_stability": {
            "H_pre_mode_layer": int(np.bincount(boot_layers_pre).argmax()),
            "H_pre_unique_layers": len(set(boot_layers_pre)),
            "H_pre_mode_fraction": round(float(np.max(np.bincount(boot_layers_pre)) / n_boot), 3),
            "H_post_mode_layer": int(np.bincount(boot_layers_post).argmax()),
            "H_post_unique_layers": len(set(boot_layers_post)),
            "H_post_mode_fraction": round(float(np.max(np.bincount(boot_layers_post)) / n_boot), 3),
        },
        "coverages": {},
    }

    for cov_key, gaps in boot_gaps.items():
        gaps = np.array(gaps)
        mean_gap = float(np.mean(gaps))
        ci_lo = float(np.percentile(gaps, 2.5))
        ci_hi = float(np.percentile(gaps, 97.5))
        p_val = float(min(np.mean(gaps <= 0), np.mean(gaps >= 0)) * 2)
        results["coverages"][cov_key] = {
            **point_results[cov_key],
            "nested_bootstrap_mean_gap": round(mean_gap, 4),
            "nested_bootstrap_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
            "nested_bootstrap_p": round(p_val, 4),
            "significant_005": p_val < 0.05,
        }

    return results


# ============================================================================
# Scenario B: Repeated-split ranking stability
# ============================================================================

def run_scenario_b_repeated(all_h_pre, all_h_post, all_labels, models, n_splits=N_RANKING_SPLITS):
    """Test ranking stability across multiple splits."""
    rankings_pre = []
    rankings_post = []

    for s in range(n_splits):
        rng = np.random.RandomState(s)
        aurocs_pre = {}
        aurocs_post = {}
        for m in models:
            n = len(all_labels[m])
            cal, test = cal_test_split(n, rng)
            lp, sp, _ = best_layer_sign(all_h_pre[m], all_labels[m], cal)
            lo, so, _ = best_layer_sign(all_h_post[m], all_labels[m], cal)
            y = all_labels[m][test]
            try:
                aurocs_pre[m] = roc_auc_score(y, sp * all_h_pre[m][test, lp])
            except:
                aurocs_pre[m] = 0.5
            try:
                aurocs_post[m] = roc_auc_score(y, so * all_h_post[m][test, lo])
            except:
                aurocs_post[m] = 0.5

        rank_pre = sorted(models, key=lambda m: aurocs_pre[m], reverse=True)
        rank_post = sorted(models, key=lambda m: aurocs_post[m], reverse=True)
        rankings_pre.append(rank_pre)
        rankings_post.append(rank_post)

    # Count how often ranking differs
    n_differ = sum(1 for rp, ro in zip(rankings_pre, rankings_post) if rp != ro)

    # Count most common ranking
    from collections import Counter
    pre_counts = Counter(tuple(r) for r in rankings_pre)
    post_counts = Counter(tuple(r) for r in rankings_post)

    return {
        "n_splits": n_splits,
        "n_ranking_differs": n_differ,
        "fraction_differs": round(n_differ / n_splits, 3),
        "most_common_ranking_H_pre": list(pre_counts.most_common(1)[0][0]),
        "H_pre_mode_fraction": round(pre_counts.most_common(1)[0][1] / n_splits, 3),
        "most_common_ranking_H_post": list(post_counts.most_common(1)[0][0]),
        "H_post_mode_fraction": round(post_counts.most_common(1)[0][1] / n_splits, 3),
        "all_rankings_H_pre": [list(r) for r in rankings_pre],
        "all_rankings_H_post": [list(r) for r in rankings_post],
    }


# ============================================================================
# Scenario C: Cross-normalization layer penalty (v2 sign fix retained)
# ============================================================================

def run_scenario_c(h_pre, h_post, labels, n_boot=N_BOOTSTRAP):
    rng0 = np.random.RandomState(SEED)
    cal_idx, test_idx = cal_test_split(len(labels), rng0)
    y_test = labels[test_idx]

    l_pre, s_pre, _ = best_layer_sign(h_pre, labels, cal_idx)
    l_post, s_post, _ = best_layer_sign(h_post, labels, cal_idx)
    s_pre_at_lpost = best_sign_at_layer(h_pre, labels, cal_idx, l_post)
    s_post_at_lpre = best_sign_at_layer(h_post, labels, cal_idx, l_pre)

    def safe_auroc(y, s):
        try:
            return roc_auc_score(y, s)
        except:
            return 0.5

    scores_pre_opt = s_pre * h_pre[test_idx, l_pre]
    scores_post_opt = s_post * h_post[test_idx, l_post]
    scores_pre_wrong = s_pre_at_lpost * h_pre[test_idx, l_post]
    scores_post_wrong = s_post_at_lpre * h_post[test_idx, l_pre]

    auroc_pre_opt = safe_auroc(y_test, scores_pre_opt)
    auroc_pre_wrong = safe_auroc(y_test, scores_pre_wrong)
    auroc_post_opt = safe_auroc(y_test, scores_post_opt)
    auroc_post_wrong = safe_auroc(y_test, scores_post_wrong)

    # Bootstrap CI
    rng_b = np.random.RandomState(SEED)
    n_test = len(test_idx)
    diffs_pre, diffs_post = [], []
    for _ in range(n_boot):
        idx = rng_b.choice(n_test, size=n_test, replace=True)
        a_po = safe_auroc(y_test[idx], scores_pre_opt[idx])
        a_pw = safe_auroc(y_test[idx], scores_pre_wrong[idx])
        a_oo = safe_auroc(y_test[idx], scores_post_opt[idx])
        a_ow = safe_auroc(y_test[idx], scores_post_wrong[idx])
        diffs_pre.append(a_po - a_pw)
        diffs_post.append(a_oo - a_ow)

    diffs_pre = np.array(diffs_pre)
    diffs_post = np.array(diffs_post)

    return {
        "H_pre_optimal": {"layer": int(l_pre), "sign": int(s_pre), "auroc": round(auroc_pre_opt, 4)},
        "H_pre_at_H_post_layer": {"layer": int(l_post), "sign": int(s_pre_at_lpost), "auroc": round(auroc_pre_wrong, 4)},
        "H_pre_penalty": {
            "delta": round(auroc_pre_opt - auroc_pre_wrong, 4),
            "bootstrap_ci_95": [round(float(np.percentile(diffs_pre, 2.5)), 4),
                                round(float(np.percentile(diffs_pre, 97.5)), 4)],
        },
        "H_post_optimal": {"layer": int(l_post), "sign": int(s_post), "auroc": round(auroc_post_opt, 4)},
        "H_post_at_H_pre_layer": {"layer": int(l_pre), "sign": int(s_post_at_lpre), "auroc": round(auroc_post_wrong, 4)},
        "H_post_penalty": {
            "delta": round(auroc_post_opt - auroc_post_wrong, 4),
            "bootstrap_ci_95": [round(float(np.percentile(diffs_post, 2.5)), 4),
                                round(float(np.percentile(diffs_post, 97.5)), 4)],
        },
    }


# ============================================================================
# Main
# ============================================================================

def main():
    start = datetime.now()
    log("START", "Exp54 Full Run v3: Practical Consequence (CPU)")

    all_results = {
        "metadata": {
            "experiment": "54_Practical_Consequence",
            "version": 3,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S KST"),
            "seed": SEED,
            "n_bootstrap": N_BOOTSTRAP,
            "n_ranking_splits": N_RANKING_SPLITS,
            "protocols": ["sampled (SET A)", "greedy (SET F)"],
            "models": ["qwen", "llama", "mistral"],
            "dataset": "MMLU 1000 samples",
            "fixes_v3": [
                "Scenario A: nested bootstrap (re-split + re-select layer per iteration)",
                "Scenario B: 20-split ranking stability test",
                "Scenario C: per-layer sign re-optimization (from v2)",
            ],
        },
        "per_protocol": {},
    }

    for protocol in ["sampled", "greedy"]:
        log("PROTOCOL", f"=== {protocol.upper()} ===")
        protocol_results = {}
        all_h_pre, all_h_post, all_labels_dict = {}, {}, {}

        for model_key in ["qwen", "llama", "mistral"]:
            path = DATA_SOURCES[protocol][model_key]
            if not path.exists():
                log("SKIP", f"{protocol}/{model_key}: not found")
                continue

            data = load_data(path)
            num_layers = len(data[0]["layer_data"])
            labels, h_pre, h_post = extract_features(data, num_layers)
            n = len(labels)

            all_h_pre[model_key] = h_pre
            all_h_post[model_key] = h_post
            all_labels_dict[model_key] = labels

            log("DATA", f"{model_key}: n={n}, layers={num_layers}, acc={np.mean(labels):.3f}")

            # Scenario A: nested bootstrap
            scenario_a = run_scenario_a_nested(h_pre, h_post, labels)
            c80 = scenario_a["coverages"]["80"]
            stab = scenario_a["layer_selection_stability"]
            log("A", f"{model_key}: gap@80%={c80['gap']:+.4f} "
                f"nested_CI=[{c80['nested_bootstrap_ci_95'][0]:+.4f}, {c80['nested_bootstrap_ci_95'][1]:+.4f}] "
                f"p={c80['nested_bootstrap_p']:.3f} "
                f"L_pre stability={stab['H_pre_mode_fraction']:.0%} "
                f"L_post stability={stab['H_post_mode_fraction']:.0%}")

            # Scenario C
            scenario_c = run_scenario_c(h_pre, h_post, labels)
            log("C", f"{model_key}: H_pre_pen={scenario_c['H_pre_penalty']['delta']:+.4f} "
                f"H_post_pen={scenario_c['H_post_penalty']['delta']:+.4f}")

            protocol_results[model_key] = {
                "n_samples": n,
                "num_layers": num_layers,
                "accuracy": round(float(np.mean(labels)), 4),
                "scenario_a_nested": scenario_a,
                "scenario_c": scenario_c,
            }

        # Scenario B: repeated-split ranking
        models = list(all_h_pre.keys())
        if len(models) >= 2:
            scenario_b = run_scenario_b_repeated(
                all_h_pre, all_h_post, all_labels_dict, models)
            log("B", f"{protocol}: differs {scenario_b['n_ranking_differs']}/{scenario_b['n_splits']} "
                f"({scenario_b['fraction_differs']:.0%}), "
                f"H_pre mode={scenario_b['most_common_ranking_H_pre']} ({scenario_b['H_pre_mode_fraction']:.0%}), "
                f"H_post mode={scenario_b['most_common_ranking_H_post']} ({scenario_b['H_post_mode_fraction']:.0%})")
        else:
            scenario_b = {"error": "insufficient models"}

        all_results["per_protocol"][protocol] = {
            "models": protocol_results,
            "scenario_b_ranking": scenario_b,
        }

    elapsed = (datetime.now() - start).total_seconds()
    all_results["metadata"]["elapsed_seconds"] = round(elapsed, 1)

    out_file = EXP_DIR / "practical_consequence_results.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    log("DONE", f"Elapsed: {elapsed:.1f}s. Saved to {out_file}")


if __name__ == "__main__":
    main()
