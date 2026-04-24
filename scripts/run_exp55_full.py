"""
Experiment 55 Full Run v3: Radial/Angular AUROC Decomposition
CPU only — uses existing MMLU sample_results.json (sampled + greedy)

Fixes v3:
  - Reports linear R² + Spearman rho (NOT rho², labeled correctly as "rank correlation")
  - Profile AUROC: dimension-matched comparison (PCA to min dim)
  - Step 3 per-layer AUROC uses cal/test split (no leakage)
  - Explicit note on R² vs exp43 distinction

Usage:
  python scripts/run_exp55_full.py
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, r2_score
from scipy.stats import spearmanr
from _paths import POT_DIR

PROJECT_ROOT = Path(__file__).parent.parent
SEED = 42
N_BOOTSTRAP = 1000

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

EXP_DIR = PROJECT_ROOT / "experiments" / "55_Radial_Angular_Decomposition"
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
    h_norm = np.zeros((len(data), num_layers))
    logit_std = np.zeros((len(data), num_layers))

    for i, r in enumerate(data):
        ld = r["layer_data"]
        for li in range(num_layers):
            li_str = str(li)
            if li_str in ld:
                h_pre[i, li] = ld[li_str]["unnormed_entropy"]
                h_post[i, li] = ld[li_str]["normed_entropy"]
                h_norm[i, li] = ld[li_str]["h_norm"]
                logit_std[i, li] = ld[li_str]["logit_std"]

    return labels, h_pre, h_post, h_norm, logit_std


def cal_test_split(n, seed=SEED, cal_ratio=0.7):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    n_cal = int(n * cal_ratio)
    return indices[:n_cal], indices[n_cal:]


def profile_auroc_with_ci(features, labels, cal_idx, test_idx, C=1.0, n_boot=N_BOOTSTRAP):
    scaler = StandardScaler()
    X_cal = scaler.fit_transform(features[cal_idx])
    X_test = scaler.transform(features[test_idx])
    y_cal = labels[cal_idx]
    y_test = labels[test_idx]

    lr = LogisticRegression(max_iter=1000, solver='lbfgs', C=C)
    lr.fit(X_cal, y_cal)
    probs = lr.predict_proba(X_test)[:, 1]

    try:
        point_auroc = roc_auc_score(y_test, probs)
    except:
        point_auroc = 0.5

    rng = np.random.RandomState(SEED)
    n_test = len(test_idx)
    boot_aurocs = []
    for _ in range(n_boot):
        idx = rng.choice(n_test, size=n_test, replace=True)
        try:
            boot_aurocs.append(roc_auc_score(y_test[idx], probs[idx]))
        except:
            boot_aurocs.append(0.5)

    return {
        "auroc": round(point_auroc, 4),
        "ci_95": [round(float(np.percentile(boot_aurocs, 2.5)), 4),
                  round(float(np.percentile(boot_aurocs, 97.5)), 4)],
        "n_features": features.shape[1],
        "C": C,
    }


def profile_auroc_dim_matched(features_a, features_b, labels, cal_idx, test_idx,
                               target_dim=None, C=1.0, n_boot=N_BOOTSTRAP):
    """Compare two feature sets at matched dimensionality via PCA."""
    if target_dim is None:
        target_dim = min(features_a.shape[1], features_b.shape[1])

    results = {}
    for name, feat in [("a", features_a), ("b", features_b)]:
        scaler = StandardScaler()
        X_cal = scaler.fit_transform(feat[cal_idx])
        X_test = scaler.transform(feat[test_idx])

        if feat.shape[1] > target_dim:
            pca = PCA(n_components=target_dim)
            X_cal = pca.fit_transform(X_cal)
            X_test = pca.transform(X_test)

        y_cal = labels[cal_idx]
        y_test = labels[test_idx]

        lr = LogisticRegression(max_iter=1000, solver='lbfgs', C=C)
        lr.fit(X_cal, y_cal)
        probs = lr.predict_proba(X_test)[:, 1]

        try:
            auroc = roc_auc_score(y_test, probs)
        except:
            auroc = 0.5

        rng = np.random.RandomState(SEED)
        boot = []
        for _ in range(n_boot):
            idx = rng.choice(len(y_test), size=len(y_test), replace=True)
            try:
                boot.append(roc_auc_score(y_test[idx], probs[idx]))
            except:
                boot.append(0.5)

        results[name] = {
            "auroc": round(auroc, 4),
            "ci_95": [round(float(np.percentile(boot, 2.5)), 4),
                      round(float(np.percentile(boot, 97.5)), 4)],
            "dim_after_pca": target_dim,
        }

    return results


def per_layer_decomposition(h_pre, h_post, h_norm, num_layers):
    results = []
    for li in range(num_layers):
        y = h_pre[:, li]
        x_s = h_norm[:, li]
        x_d = h_post[:, li]

        # Linear R²
        try:
            r2_s = max(0, r2_score(y, LinearRegression().fit(x_s.reshape(-1, 1), y).predict(x_s.reshape(-1, 1))))
        except:
            r2_s = 0.0
        try:
            r2_d = max(0, r2_score(y, LinearRegression().fit(x_d.reshape(-1, 1), y).predict(x_d.reshape(-1, 1))))
        except:
            r2_d = 0.0
        try:
            x_both = np.column_stack([x_s, x_d])
            r2_b = max(0, r2_score(y, LinearRegression().fit(x_both, y).predict(x_both)))
        except:
            r2_b = 0.0

        # Spearman rank correlation (NOT rho² — report rho directly)
        rho_s, p_s = spearmanr(x_s, y)
        rho_d, p_d = spearmanr(x_d, y)

        results.append({
            "layer": li,
            "linear_R2_scale": round(r2_s, 4),
            "linear_R2_direction": round(r2_d, 4),
            "linear_R2_both": round(r2_b, 4),
            "unique_scale": round(max(0, r2_b - r2_d), 4),
            "unique_direction": round(max(0, r2_b - r2_s), 4),
            "spearman_rho_scale": round(float(rho_s), 4),
            "spearman_p_scale": round(float(p_s), 6),
            "spearman_rho_direction": round(float(rho_d), 4),
            "spearman_p_direction": round(float(p_d), 6),
        })
    return results


def per_layer_auroc_on_test(features, labels, cal_idx, test_idx):
    """Per-layer AUROC using cal for sign selection, test for evaluation."""
    n_layers = features.shape[1]
    y_cal = labels[cal_idx]
    y_test = labels[test_idx]
    results = []
    for li in range(n_layers):
        # Select sign on cal
        best_sign, best_cal = 1, 0.5
        for sign in [1, -1]:
            try:
                a = roc_auc_score(y_cal, sign * features[cal_idx, li])
            except:
                a = 0.5
            if a > best_cal:
                best_cal = a
                best_sign = sign
        # Evaluate on test
        try:
            test_auroc = roc_auc_score(y_test, best_sign * features[test_idx, li])
        except:
            test_auroc = 0.5
        results.append({"layer": li, "auroc": round(test_auroc, 4), "sign": best_sign})
    return results


def main():
    start = datetime.now()
    log("START", "Exp55 Full Run v3: Radial/Angular Decomposition (CPU)")
    np.random.seed(SEED)

    all_results = {
        "metadata": {
            "experiment": "55_Radial_Angular_Decomposition",
            "version": 3,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S KST"),
            "seed": SEED,
            "n_bootstrap": N_BOOTSTRAP,
            "cal_test_split": "70/30, seed=42",
            "protocols": ["sampled (SET A)", "greedy (SET F)"],
            "models": ["qwen", "llama", "mistral"],
            "dataset": "MMLU 1000 samples",
            "fixes_v3": [
                "Spearman reported as rho (rank correlation), NOT rho^2 (not variance explained)",
                "Combined profile: dimension-matched via PCA to min(dim_a, dim_b)",
                "Per-layer AUROC: sign on cal, evaluate on test (no leakage)",
                "Note: low R^2 != low causal effect (exp43 shows causal; R^2 measures cross-sample covariation)",
            ],
        },
        "per_protocol": {},
    }

    for protocol in ["sampled", "greedy"]:
        log("PROTOCOL", f"=== {protocol.upper()} ===")
        protocol_results = {}

        for model_key in ["qwen", "llama", "mistral"]:
            path = DATA_SOURCES[protocol][model_key]
            if not path.exists():
                continue

            data = load_data(path)
            num_layers = len(data[0]["layer_data"])
            labels, h_pre, h_post, h_norm, logit_std_feat = extract_features(data, num_layers)
            n = len(labels)
            cal_idx, test_idx = cal_test_split(n)

            log("DATA", f"{model_key}: n={n}, L={num_layers}, acc={np.mean(labels):.3f}")

            # Step 1: Per-layer decomposition
            decomp = per_layer_decomposition(h_pre, h_post, h_norm, num_layers)
            lin_r2 = [x["linear_R2_scale"] for x in decomp]
            sp_rho = [abs(x["spearman_rho_scale"]) for x in decomp]
            n_lin_dom = sum(1 for x in lin_r2 if x > 0.5)
            n_sp_strong = sum(1 for x in sp_rho if x > 0.7)

            log("R2", f"{model_key}: lin_R2_mean={np.mean(lin_r2):.4f}, "
                f"|rho|_mean={np.mean(sp_rho):.4f}, "
                f"lin_dom(>0.5)={n_lin_dom}/{num_layers}, "
                f"|rho|_strong(>0.7)={n_sp_strong}/{num_layers}")

            # Step 2: Profile-level AUROC (standard, same dim)
            auroc_radial = profile_auroc_with_ci(h_norm, labels, cal_idx, test_idx)
            auroc_angular = profile_auroc_with_ci(h_post, labels, cal_idx, test_idx)
            auroc_h_pre = profile_auroc_with_ci(h_pre, labels, cal_idx, test_idx)
            auroc_logit_std = profile_auroc_with_ci(logit_std_feat, labels, cal_idx, test_idx)

            # Dimension-matched combined comparison
            dim_matched = profile_auroc_dim_matched(
                np.column_stack([h_norm, h_post]),  # combined
                h_pre,                               # vs H_pre
                labels, cal_idx, test_idx,
                target_dim=num_layers, C=1.0)

            log("AUROC", f"{model_key}: rad={auroc_radial['auroc']:.4f} "
                f"ang={auroc_angular['auroc']:.4f} "
                f"H_pre={auroc_h_pre['auroc']:.4f} "
                f"lstd={auroc_logit_std['auroc']:.4f} "
                f"comb_matched={dim_matched['a']['auroc']:.4f}")

            # Step 3: Per-layer AUROC (cal/test split, no leakage)
            pla_hnorm = per_layer_auroc_on_test(h_norm, labels, cal_idx, test_idx)
            pla_hpre = per_layer_auroc_on_test(h_pre, labels, cal_idx, test_idx)
            pla_hpost = per_layer_auroc_on_test(h_post, labels, cal_idx, test_idx)
            pla_lstd = per_layer_auroc_on_test(logit_std_feat, labels, cal_idx, test_idx)

            a_hn = [x["auroc"] for x in pla_hnorm]
            a_hp = [x["auroc"] for x in pla_hpre]
            a_ho = [x["auroc"] for x in pla_hpost]
            a_ls = [x["auroc"] for x in pla_lstd]

            corr_hn_hp = float(np.corrcoef(a_hn, a_hp)[0, 1]) if np.std(a_hn) > 0 and np.std(a_hp) > 0 else 0.0
            corr_ls_hp = float(np.corrcoef(a_ls, a_hp)[0, 1]) if np.std(a_ls) > 0 and np.std(a_hp) > 0 else 0.0
            corr_hn_ho = float(np.corrcoef(a_hn, a_ho)[0, 1]) if np.std(a_hn) > 0 and np.std(a_ho) > 0 else 0.0

            log("CORR", f"{model_key}: corr(h_norm,H_pre)={corr_hn_hp:.4f} "
                f"corr(lstd,H_pre)={corr_ls_hp:.4f}")

            protocol_results[model_key] = {
                "n_samples": n,
                "num_layers": num_layers,
                "accuracy": round(float(np.mean(labels)), 4),
                "step1_decomposition": {
                    "per_layer": decomp,
                    "summary": {
                        "linear_R2_scale_mean": round(float(np.mean(lin_r2)), 4),
                        "abs_spearman_rho_scale_mean": round(float(np.mean(sp_rho)), 4),
                        "n_linear_R2_gt05": n_lin_dom,
                        "n_abs_rho_gt07": n_sp_strong,
                        "n_layers": num_layers,
                        "interpretation": (
                            "scale_dominant" if n_sp_strong > num_layers * 0.7
                            else "mixed" if n_sp_strong > num_layers * 0.3
                            else "direction_substantial"
                        ),
                        "note": "Low R^2 != low causal effect. R^2 measures cross-sample covariation. "
                                "Exp43 shows removing scale causally destroys H_pre for all models including Qwen.",
                    },
                },
                "step2_profile_auroc": {
                    "radial_h_norm": auroc_radial,
                    "angular_H_post": auroc_angular,
                    "H_pre": auroc_h_pre,
                    "logit_std_profile": auroc_logit_std,
                    "combined_dim_matched": {
                        "radial_plus_angular": dim_matched["a"],
                        "H_pre_at_same_dim": dim_matched["b"],
                        "target_dim": num_layers,
                    },
                    "delta_radial_vs_H_pre": round(auroc_radial["auroc"] - auroc_h_pre["auroc"], 4),
                    "delta_logit_std_vs_H_pre": round(auroc_logit_std["auroc"] - auroc_h_pre["auroc"], 4),
                },
                "step3_per_layer_auroc": {
                    "note": "Sign selected on cal, AUROC evaluated on test (no leakage)",
                    "h_norm": pla_hnorm,
                    "H_pre": pla_hpre,
                    "H_post": pla_hpost,
                    "logit_std": pla_lstd,
                },
                "step3_correlations": {
                    "corr_h_norm_H_pre": round(corr_hn_hp, 4),
                    "corr_logit_std_H_pre": round(corr_ls_hp, 4),
                    "corr_h_norm_H_post": round(corr_hn_ho, 4),
                },
            }

        all_results["per_protocol"][protocol] = protocol_results

    elapsed = (datetime.now() - start).total_seconds()
    all_results["metadata"]["elapsed_seconds"] = round(elapsed, 1)

    out_file = EXP_DIR / "decomposition_results.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Summary
    log("SUMMARY", "=" * 70)
    for protocol in ["sampled", "greedy"]:
        for m in ["qwen", "llama", "mistral"]:
            r = all_results["per_protocol"].get(protocol, {}).get(m, {})
            if r:
                s1 = r["step1_decomposition"]["summary"]
                s2 = r["step2_profile_auroc"]
                log("SUMMARY", f"  {protocol}/{m}: R2={s1['linear_R2_scale_mean']:.3f} "
                    f"|rho|={s1['abs_spearman_rho_scale_mean']:.3f} "
                    f"[{s1['interpretation']}] | "
                    f"rad={s2['radial_h_norm']['auroc']:.3f} "
                    f"ang={s2['angular_H_post']['auroc']:.3f} "
                    f"Hpre={s2['H_pre']['auroc']:.3f} "
                    f"lstd={s2['logit_std_profile']['auroc']:.3f}")

    log("DONE", f"Elapsed: {elapsed:.1f}s. Saved to {out_file}")


if __name__ == "__main__":
    main()
