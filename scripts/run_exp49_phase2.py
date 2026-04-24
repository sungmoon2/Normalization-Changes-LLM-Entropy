"""
Experiment 49 Phase 2: Recompute Tables with Greedy Data

Input: phase1_greedy/{model}/data/sample_results.json (greedy generation-average)
       exp31 sample_results.json (sampled, for agreement comparison)

Output: phase2_recompute/*.json

Computes:
  1. Agreement table (sampled vs greedy)
  2. Table 1 analog (single-layer AUROC, 70/30 held-out)
  3. Table 3 analog (incremental utility)
  4. Table 5 analog (best layer/sign)
  5. Table 9 analog (profile 5-fold CV + absorption)
  6. Table 9 paired fold-wise test
  7. Per-layer landscape (bootstrap CI)
  8. Appendix I analog (20 repeated splits)

Usage:
  python scripts/run_exp49_phase2.py
  python scripts/run_exp49_phase2.py --smoke_test
"""

import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
from _paths import POT_DIR

SEED = 42
BASE = Path(__file__).resolve().parent.parent
EXP31 = POT_DIR / "experiments" / "31_MMLU_Domain_Extension"
EXP49 = BASE / "experiments" / "49_Deterministic_Label_Robustness"
OUT = EXP49 / "phase2_recompute"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "qwen": {"exp31_dir": "EXP_20260219_053638_mmlu_qwen", "n_layers": 28},
    "llama": {"exp31_dir": "EXP_20260219_171237_mmlu_llama", "n_layers": 32},
    "mistral": {"exp31_dir": "EXP_20260220_000610_mmlu_mistral", "n_layers": 32},
}
METRICS = ["unnormed_entropy", "normed_entropy", "logit_std", "h_norm"]
METRIC_LABELS = {"unnormed_entropy": "H_pre", "normed_entropy": "H_post",
                 "logit_std": "logit_std", "h_norm": "h_norm"}
N_BOOT = 1000


def log(msg):
    print(f"[EXP49-P2] {msg}", flush=True)


def load_results(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def extract_features(samples, n_layers):
    """Extract per-sample per-layer feature matrix + labels."""
    n = len(samples)
    y = np.array([1 if s["is_correct"] else 0 for s in samples])
    features = {}
    for metric in METRICS:
        X = np.zeros((n, n_layers))
        for i, s in enumerate(samples):
            if "layer_data" not in s:
                continue
            for layer in range(n_layers):
                lk = str(layer)
                if lk in s["layer_data"] and metric in s["layer_data"][lk]:
                    X[i, layer] = s["layer_data"][lk][metric]
        features[metric] = X
    return features, y


# ============================================================================
# 1. Agreement
# ============================================================================

def compute_agreement(greedy, sampled):
    """Compute agreement between greedy and sampled labels."""
    n = min(len(greedy), len(sampled))
    g_labels = [greedy[i]["is_correct"] for i in range(n)]
    s_labels = [sampled[i]["is_correct"] for i in range(n)]

    agree = sum(1 for g, s in zip(g_labels, s_labels) if g == s)
    g_acc = sum(g_labels) / n
    s_acc = sum(s_labels) / n
    p_o = agree / n
    p_e = g_acc * s_acc + (1 - g_acc) * (1 - s_acc)
    kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 0

    return {
        "n": n,
        "greedy_accuracy": round(g_acc, 4),
        "sampled_accuracy": round(s_acc, 4),
        "agreement_rate": round(p_o, 4),
        "cohens_kappa": round(kappa, 4),
        "greedy_correct_sampled_incorrect": sum(1 for g, s in zip(g_labels, s_labels) if g and not s),
        "greedy_incorrect_sampled_correct": sum(1 for g, s in zip(g_labels, s_labels) if not g and s),
    }


# ============================================================================
# 2-4. Tables 1, 3, 5 (single-layer, 70/30 held-out)
# ============================================================================

def compute_single_layer_auroc(X, y, layer, sign):
    scores = X[:, layer] * sign
    try:
        return roc_auc_score(y, scores)
    except ValueError:
        return 0.5


def find_best_layer_sign(X, y_cal, cal_idx):
    """Find best layer and sign on calibration set."""
    n_layers = X.shape[1]
    best_auc, best_layer, best_sign = 0, 0, 1
    for layer in range(n_layers):
        for sign in [+1, -1]:
            auc = compute_single_layer_auroc(X[cal_idx], y_cal, layer, sign)
            if auc > best_auc:
                best_auc, best_layer, best_sign = auc, layer, sign
    return best_layer, best_sign, best_auc


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
        return 0.5, 0.5, 0.0
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5), np.std(aucs)


def compute_tables_1_3_5(features, y, n_layers):
    """Compute Tables 1, 3, 5 analogs with 70/30 held-out."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
    cal_idx, test_idx = next(sss.split(np.zeros(len(y)), y))

    # Table 1 + Table 5: best layer/sign on cal, AUROC on test
    table1 = {}
    table5 = {}
    for metric in METRICS:
        X = features[metric]
        bl, bs, cal_auc = find_best_layer_sign(X, y[cal_idx], cal_idx)
        test_auc = compute_single_layer_auroc(X[test_idx], y[test_idx], bl, bs)
        ci_lo, ci_hi, _ = bootstrap_auroc_ci(y[test_idx], X[test_idx, bl] * bs)
        table1[METRIC_LABELS[metric]] = {
            "auroc": round(test_auc, 4), "layer": bl, "sign": bs,
            "cal_auroc": round(cal_auc, 4),
            "ci": [round(ci_lo, 4), round(ci_hi, 4)],
        }
        table5[METRIC_LABELS[metric]] = {"layer": bl, "sign": bs}

    # Output entropy (from final layer H_post)
    X_out = features["normed_entropy"]
    final_layer = n_layers - 1
    out_auc = compute_single_layer_auroc(X_out[test_idx], y[test_idx], final_layer, -1)
    table1["output_entropy"] = {"auroc": round(out_auc, 4), "layer": final_layer, "sign": -1}

    # Table 3: incremental utility
    table3 = {}
    # logit_std only
    bl_std = table5["logit_std"]["layer"]
    bs_std = table5["logit_std"]["sign"]

    scaler = StandardScaler()
    X_std_cal = scaler.fit_transform(features["logit_std"][cal_idx, bl_std:bl_std+1] * bs_std)
    X_std_test = scaler.transform(features["logit_std"][test_idx, bl_std:bl_std+1] * bs_std)

    bl_pre = table5["H_pre"]["layer"]
    bs_pre = table5["H_pre"]["sign"]
    X_pre_cal = features["unnormed_entropy"][cal_idx, bl_pre:bl_pre+1] * bs_pre
    X_pre_test = features["unnormed_entropy"][test_idx, bl_pre:bl_pre+1] * bs_pre

    # logit_std only (sign applied to match Table 1 convention)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(scaler.fit_transform(features["logit_std"][cal_idx, bl_std:bl_std+1] * bs_std), y[cal_idx])
    auc_std = roc_auc_score(y[test_idx], lr.predict_proba(
        scaler.transform(features["logit_std"][test_idx, bl_std:bl_std+1] * bs_std))[:, 1])

    # logit_std + H_pre
    X_both_cal = np.column_stack([features["logit_std"][cal_idx, bl_std] * bs_std,
                                   features["unnormed_entropy"][cal_idx, bl_pre] * bs_pre])
    X_both_test = np.column_stack([features["logit_std"][test_idx, bl_std] * bs_std,
                                    features["unnormed_entropy"][test_idx, bl_pre] * bs_pre])
    sc2 = StandardScaler()
    lr2 = LogisticRegression(max_iter=1000)
    lr2.fit(sc2.fit_transform(X_both_cal), y[cal_idx])
    auc_both = roc_auc_score(y[test_idx], lr2.predict_proba(sc2.transform(X_both_test))[:, 1])

    delta = auc_both - auc_std

    # Bootstrap CI for delta
    rng = np.random.RandomState(SEED)
    deltas = []
    for _ in range(N_BOOT):
        idx = rng.choice(len(test_idx), len(test_idx), replace=True)
        y_b = y[test_idx[idx]]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            a1 = roc_auc_score(y_b, lr.predict_proba(
                scaler.transform(features["logit_std"][test_idx[idx], bl_std:bl_std+1] * bs_std))[:, 1])
            a2 = roc_auc_score(y_b, lr2.predict_proba(
                sc2.transform(X_both_test[idx]))[:, 1])
            deltas.append(a2 - a1)
        except:
            continue

    table3 = {
        "logit_std_only": round(auc_std, 4),
        "logit_std_plus_H_pre": round(auc_both, 4),
        "delta": round(delta, 4),
        "delta_ci": [round(np.percentile(deltas, 2.5), 4), round(np.percentile(deltas, 97.5), 4)] if deltas else [0, 0],
        "ci_includes_zero": bool(np.percentile(deltas, 2.5) <= 0 <= np.percentile(deltas, 97.5)) if deltas else True,
    }

    return table1, table3, table5


# ============================================================================
# 5-6. Table 9 (profile 5-fold CV + paired tests)
# ============================================================================

def compute_table9(features, y, n_layers):
    """Profile-based 5-fold CV with absorption and paired tests."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    profile_results = {m: [] for m in list(METRIC_LABELS.values()) + ["scale_combined"]}
    knn_results = {m: [] for m in ["H_pre", "H_post"]}
    absorption_results = {"scale_only": [], "scale_plus_Hpre": [], "scale_plus_Hpost": []}

    # Collect OOF predictions for paired tests
    oof_preds = {m: np.zeros(len(y)) for m in list(METRIC_LABELS.values()) + ["scale_combined"]}

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        for metric_key, metric_name in METRIC_LABELS.items():
            X = features[metric_key]
            sc = StandardScaler()
            lr = LogisticRegression(max_iter=1000)
            lr.fit(sc.fit_transform(X[train_idx]), y[train_idx])
            proba = lr.predict_proba(sc.transform(X[test_idx]))[:, 1]
            auc = roc_auc_score(y[test_idx], proba)
            profile_results[metric_name].append(auc)
            oof_preds[metric_name][test_idx] = proba

        # k-NN for H_pre and H_post
        for metric_key, metric_name in [("unnormed_entropy", "H_pre"), ("normed_entropy", "H_post")]:
            X = features[metric_key]
            sc = StandardScaler()
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(sc.fit_transform(X[train_idx]), y[train_idx])
            proba = knn.predict_proba(sc.transform(X[test_idx]))[:, 1]
            auc = roc_auc_score(y[test_idx], proba)
            knn_results[metric_name].append(auc)

        # Scale combined (logit_std + h_norm)
        X_scale = np.column_stack([features["logit_std"], features["h_norm"]])
        sc = StandardScaler()
        lr = LogisticRegression(max_iter=1000)
        lr.fit(sc.fit_transform(X_scale[train_idx]), y[train_idx])
        proba = lr.predict_proba(sc.transform(X_scale[test_idx]))[:, 1]
        auc = roc_auc_score(y[test_idx], proba)
        profile_results["scale_combined"].append(auc)
        oof_preds["scale_combined"][test_idx] = proba
        absorption_results["scale_only"].append(auc)

        # Scale + H_pre
        X_sp = np.column_stack([features["logit_std"], features["h_norm"], features["unnormed_entropy"]])
        sc2 = StandardScaler()
        lr2 = LogisticRegression(max_iter=1000)
        lr2.fit(sc2.fit_transform(X_sp[train_idx]), y[train_idx])
        auc_sp = roc_auc_score(y[test_idx], lr2.predict_proba(sc2.transform(X_sp[test_idx]))[:, 1])
        absorption_results["scale_plus_Hpre"].append(auc_sp)

        # Scale + H_post
        X_spo = np.column_stack([features["logit_std"], features["h_norm"], features["normed_entropy"]])
        sc3 = StandardScaler()
        lr3 = LogisticRegression(max_iter=1000)
        lr3.fit(sc3.fit_transform(X_spo[train_idx]), y[train_idx])
        auc_spo = roc_auc_score(y[test_idx], lr3.predict_proba(sc3.transform(X_spo[test_idx]))[:, 1])
        absorption_results["scale_plus_Hpost"].append(auc_spo)

    # Compile results
    table9 = {}
    for name, aucs in profile_results.items():
        table9[name] = {"mean": round(np.mean(aucs), 4), "std": round(np.std(aucs), 4), "folds": [round(a, 4) for a in aucs]}
    for name, aucs in knn_results.items():
        table9[f"{name}_knn"] = {"mean": round(np.mean(aucs), 4), "std": round(np.std(aucs), 4)}

    # Absorption
    absorption = {}
    for key in absorption_results:
        aucs = absorption_results[key]
        absorption[key] = {"mean": round(np.mean(aucs), 4), "std": round(np.std(aucs), 4)}
    absorption["delta_Hpre"] = round(np.mean(absorption_results["scale_plus_Hpre"]) - np.mean(absorption_results["scale_only"]), 4)
    absorption["delta_Hpost"] = round(np.mean(absorption_results["scale_plus_Hpost"]) - np.mean(absorption_results["scale_only"]), 4)

    # Paired tests (bootstrap on OOF predictions)
    paired_tests = {}
    for pair_name, (m1, m2) in [("h_norm_vs_H_pre", ("h_norm", "H_pre")),
                                  ("logit_std_vs_H_pre", ("logit_std", "H_pre")),
                                  ("scale_vs_H_pre", ("scale_combined", "H_pre"))]:
        rng = np.random.RandomState(SEED)
        diffs = []
        for _ in range(N_BOOT):
            idx = rng.choice(len(y), len(y), replace=True)
            if len(np.unique(y[idx])) < 2:
                continue
            try:
                a1 = roc_auc_score(y[idx], oof_preds[m1][idx])
                a2 = roc_auc_score(y[idx], oof_preds[m2][idx])
                diffs.append(a1 - a2)
            except:
                continue
        if diffs:
            paired_tests[pair_name] = {
                "mean_diff": round(np.mean(diffs), 4),
                "ci": [round(np.percentile(diffs, 2.5), 4), round(np.percentile(diffs, 97.5), 4)],
                "ci_excludes_zero": bool(np.percentile(diffs, 2.5) > 0 or np.percentile(diffs, 97.5) < 0),
            }

    return table9, absorption, paired_tests


# ============================================================================
# 7. Per-layer landscape (bootstrap CI)
# ============================================================================

def compute_landscape(features, y, n_layers):
    """Per-layer AUROC with bootstrap CI."""
    landscape = {}
    for metric_key, metric_name in METRIC_LABELS.items():
        X = features[metric_key]
        layers = []
        for layer in range(n_layers):
            auc_pos = roc_auc_score(y, X[:, layer]) if len(np.unique(y)) > 1 else 0.5
            auc_neg = roc_auc_score(y, -X[:, layer]) if len(np.unique(y)) > 1 else 0.5
            if auc_neg > auc_pos:
                auc, sign = auc_neg, -1
            else:
                auc, sign = auc_pos, +1
            ci_lo, ci_hi, std = bootstrap_auroc_ci(y, X[:, layer] * sign)
            layers.append({"layer": layer, "auroc": round(auc, 4), "sign": sign,
                           "ci_lo": round(ci_lo, 4), "ci_hi": round(ci_hi, 4)})

        best_idx = max(range(n_layers), key=lambda i: layers[i]["auroc"])
        best_ci_lo = layers[best_idx]["ci_lo"]
        indisting = [l["layer"] for l in layers if l["ci_hi"] >= best_ci_lo]

        landscape[metric_name] = {
            "best_layer": layers[best_idx]["layer"],
            "best_auroc": layers[best_idx]["auroc"],
            "n_indistinguishable": len(indisting),
            "indistinguishable_layers": indisting,
            "per_layer": layers,
        }
    return landscape


# ============================================================================
# 8. Repeated splits (20 splits)
# ============================================================================

def compute_repeated_splits(features, y, n_layers, n_splits=20):
    """20 repeated 70/30 splits."""
    results = {}
    for metric_key, metric_name in METRIC_LABELS.items():
        X = features[metric_key]
        splits = []
        for seed in range(n_splits):
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
            cal_idx, test_idx = next(sss.split(np.zeros(len(y)), y))
            bl, bs, _ = find_best_layer_sign(X, y[cal_idx], np.arange(len(cal_idx)))
            test_auc = compute_single_layer_auroc(X[test_idx], y[test_idx], bl, bs)
            splits.append({"seed": seed, "layer": bl, "sign": bs, "auroc": round(test_auc, 4)})

        layers = [s["layer"] for s in splits]
        signs = [s["sign"] for s in splits]
        aucs = [s["auroc"] for s in splits]
        from collections import Counter
        layer_counts = Counter(layers)
        modal_layer = layer_counts.most_common(1)[0]
        sign_counts = Counter(signs)

        results[metric_name] = {
            "mean_auroc": round(np.mean(aucs), 4),
            "std_auroc": round(np.std(aucs), 4),
            "modal_layer": modal_layer[0],
            "modal_layer_freq": f"{modal_layer[1]}/{n_splits}",
            "sign_stability": f"{sign_counts.most_common(1)[0][1]}/{n_splits}",
            "dominant_sign": sign_counts.most_common(1)[0][0],
            "splits": splits,
        }
    return results


# ============================================================================
# Main
# ============================================================================

def run_model(model_key, smoke_test=False):
    cfg = MODELS[model_key]
    n_layers = cfg["n_layers"]

    # Load greedy data
    greedy_path = EXP49 / "phase1_greedy" / model_key / "data" / "sample_results.json"
    if not greedy_path.exists():
        # Smoke test: use exp31 sampled data as proxy
        if smoke_test:
            greedy_path = EXP31 / cfg["exp31_dir"] / "data" / "sample_results.json"
            log(f"  SMOKE: using exp31 data as proxy for {model_key}")
        else:
            log(f"  SKIP {model_key}: greedy data not found at {greedy_path}")
            return None

    greedy = load_results(greedy_path)
    if smoke_test:
        greedy = greedy[:100]

    # Load sampled data for agreement
    sampled_path = EXP31 / cfg["exp31_dir"] / "data" / "sample_results.json"
    sampled = load_results(sampled_path) if sampled_path.exists() else None

    log(f"  Loaded {len(greedy)} greedy samples")

    # Extract features
    features, y = extract_features(greedy, n_layers)
    acc = np.mean(y)
    log(f"  Accuracy: {acc:.1%}")

    # 1. Agreement
    agreement = compute_agreement(greedy, sampled) if sampled else {}
    if agreement:
        log(f"  Agreement: {agreement['agreement_rate']:.1%}, kappa={agreement['cohens_kappa']:.3f}")

    # 2-4. Tables 1, 3, 5
    table1, table3, table5 = compute_tables_1_3_5(features, y, n_layers)
    log(f"  Table 1: logit_std={table1['logit_std']['auroc']}, H_pre={table1['H_pre']['auroc']}")
    log(f"  Table 3: delta={table3['delta']}, CI={table3['delta_ci']}")

    # 5-6. Table 9
    table9, absorption, paired_tests = compute_table9(features, y, n_layers)
    log(f"  Table 9 scale: {table9['scale_combined']['mean']}, H_pre: {table9['H_pre']['mean']}")
    log(f"  Absorption: delta_Hpre={absorption['delta_Hpre']}, delta_Hpost={absorption['delta_Hpost']}")

    # 7. Landscape
    landscape = compute_landscape(features, y, n_layers)
    log(f"  Landscape: H_pre best=L{landscape['H_pre']['best_layer']} "
        f"({landscape['H_pre']['n_indistinguishable']}/{n_layers} indist), "
        f"H_post best=L{landscape['H_post']['best_layer']} "
        f"({landscape['H_post']['n_indistinguishable']}/{n_layers} indist)")

    # 8. Repeated splits
    repeated = compute_repeated_splits(features, y, n_layers)
    for m in ["logit_std", "H_pre"]:
        log(f"  Repeated {m}: mean={repeated[m]['mean_auroc']}, "
            f"modal_layer={repeated[m]['modal_layer']} ({repeated[m]['modal_layer_freq']})")

    return {
        "model": model_key,
        "n_samples": len(greedy),
        "accuracy": round(acc, 4),
        "agreement": agreement,
        "table1": table1,
        "table3": table3,
        "table5": table5,
        "table9": table9,
        "absorption": absorption,
        "paired_tests": paired_tests,
        "landscape": landscape,
        "repeated_splits": repeated,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    log(f"Phase 2: {'SMOKE TEST' if args.smoke_test else 'FULL RUN'}")

    all_results = {}
    for model_key in ["qwen", "llama", "mistral"]:
        log(f"Processing {model_key}...")
        result = run_model(model_key, smoke_test=args.smoke_test)
        if result:
            all_results[model_key] = result

    if not all_results:
        log("No results. Check greedy data availability.")
        return

    # Save
    suffix = "_smoke" if args.smoke_test else ""
    out_file = OUT / f"phase2_results{suffix}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"Saved: {out_file}")

    # Success criteria check
    log("\n" + "=" * 60)
    log("SUCCESS CRITERIA CHECK")
    log("=" * 60)
    for model_key, r in all_results.items():
        log(f"\n{model_key}:")
        t1 = r["table1"]
        std_gt_pre = t1["logit_std"]["auroc"] >= t1["H_pre"]["auroc"]
        log(f"  [{'PASS' if std_gt_pre else 'FAIL'}] logit_std >= H_pre: "
            f"{t1['logit_std']['auroc']} vs {t1['H_pre']['auroc']}")
        log(f"  [{'PASS' if r['table3']['ci_includes_zero'] else 'FAIL'}] "
            f"Incremental CI includes 0: {r['table3']['delta_ci']}")
        log(f"  [INFO] Absorption delta_Hpre: {r['absorption']['delta_Hpre']}")
        ls = r["landscape"]
        log(f"  [INFO] H_pre indist: {ls['H_pre']['n_indistinguishable']}/{MODELS[model_key]['n_layers']}, "
            f"H_post indist: {ls['H_post']['n_indistinguishable']}/{MODELS[model_key]['n_layers']}")


if __name__ == "__main__":
    main()
