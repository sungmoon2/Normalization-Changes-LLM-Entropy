"""
B2-B3-B4 CPU chain: Bootstrap CIs for Exp55 and Exp54
B2: Exp55 R²/rho bootstrap CIs
B3: Exp54 coverage sweep (60/70/80/90%)
B4: Exp54 Scenario C bootstrap CIs

CPU only, ~30min total.
"""

import json, numpy as np, time
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from _paths import POT_DIR

PROJECT_ROOT = Path(__file__).parent.parent
SEED = 42
N_BOOT = 2000

DATA_SOURCES = {
    "sampled": {
        "qwen": ("qwen", 28, POT_DIR / "experiments" / "31_MMLU_Domain_Extension" / "EXP_20260219_053638_mmlu_qwen" / "data" / "sample_results.json"),
        "llama": ("llama", 32, POT_DIR / "experiments" / "31_MMLU_Domain_Extension" / "EXP_20260219_171237_mmlu_llama" / "data" / "sample_results.json"),
        "mistral": ("mistral", 32, POT_DIR / "experiments" / "31_MMLU_Domain_Extension" / "EXP_20260220_000610_mmlu_mistral" / "data" / "sample_results.json"),
    },
    "greedy": {
        "qwen": ("qwen", 28, PROJECT_ROOT / "experiments" / "49_Deterministic_Label_Robustness" / "phase1_greedy" / "qwen" / "data" / "sample_results.json"),
        "llama": ("llama", 32, PROJECT_ROOT / "experiments" / "49_Deterministic_Label_Robustness" / "phase1_greedy" / "llama" / "data" / "sample_results.json"),
        "mistral": ("mistral", 32, PROJECT_ROOT / "experiments" / "49_Deterministic_Label_Robustness" / "phase1_greedy" / "mistral" / "data" / "sample_results.json"),
    },
}

OUT_DIR = PROJECT_ROOT / "experiments" / "56_B2B3B4_CI_Chain"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [r for r in data if "layer_data" in r and "error" not in r]


def extract_features(data, num_layers):
    n = len(data)
    labels = np.array([1 if r["is_correct"] else 0 for r in data])
    h_pre = np.zeros((n, num_layers))
    h_norm = np.zeros((n, num_layers))
    h_post = np.zeros((n, num_layers))
    logit_std = np.zeros((n, num_layers))
    for i, r in enumerate(data):
        ld = r["layer_data"]
        for li in range(num_layers):
            lk = str(li)
            if lk in ld:
                h_pre[i, li] = ld[lk].get("unnormed_entropy", 0)
                h_norm[i, li] = ld[lk].get("h_norm", 0)
                h_post[i, li] = ld[lk].get("normed_entropy", 0)
                logit_std[i, li] = ld[lk].get("logit_std", 0)
    return labels, h_pre, h_post, h_norm, logit_std


# ============================================================
# B2: Exp55 Bootstrap CIs for R² and rho
# ============================================================

def run_b2():
    log("=== B2: Exp55 Bootstrap CIs ===")
    results = {}

    for protocol in ["sampled", "greedy"]:
        for model_key, (name, n_layers, path) in DATA_SOURCES[protocol].items():
            data = load_data(path)
            labels, h_pre, h_post, h_norm, logit_std_arr = extract_features(data, n_layers)
            n = len(data)

            # Per-layer R² and rho with bootstrap
            r2_per_layer = []
            rho_per_layer = []
            r2_boot_ci = []
            rho_boot_ci = []

            for layer in range(n_layers):
                x = h_norm[:, layer]
                y = h_pre[:, layer]

                # Point estimates
                if np.std(x) > 0 and np.std(y) > 0:
                    lr = LinearRegression()
                    lr.fit(x.reshape(-1, 1), y)
                    r2 = lr.score(x.reshape(-1, 1), y)
                    rho, _ = spearmanr(x, y)
                else:
                    r2, rho = 0.0, 0.0

                r2_per_layer.append(float(r2))
                rho_per_layer.append(float(abs(rho)))

                # Bootstrap
                rng = np.random.RandomState(SEED + layer)
                r2_boots = []
                rho_boots = []
                for _ in range(N_BOOT):
                    idx = rng.choice(n, n, replace=True)
                    xb, yb = x[idx], y[idx]
                    if np.std(xb) > 0 and np.std(yb) > 0:
                        lr_b = LinearRegression()
                        lr_b.fit(xb.reshape(-1, 1), yb)
                        r2_boots.append(lr_b.score(xb.reshape(-1, 1), yb))
                        rho_b, _ = spearmanr(xb, yb)
                        rho_boots.append(abs(rho_b))

                if r2_boots:
                    r2_boot_ci.append([float(np.percentile(r2_boots, 2.5)), float(np.percentile(r2_boots, 97.5))])
                    rho_boot_ci.append([float(np.percentile(rho_boots, 2.5)), float(np.percentile(rho_boots, 97.5))])
                else:
                    r2_boot_ci.append([0, 0])
                    rho_boot_ci.append([0, 0])

            mean_r2 = float(np.mean(r2_per_layer))
            mean_rho = float(np.mean(rho_per_layer))

            # Mean-level bootstrap (mean across layers per bootstrap)
            rng2 = np.random.RandomState(SEED)
            mean_r2_boots = []
            mean_rho_boots = []
            for _ in range(N_BOOT):
                idx = rng2.choice(n, n, replace=True)
                r2s = []
                rhos = []
                for layer in range(n_layers):
                    xb = h_norm[idx, layer]
                    yb = h_pre[idx, layer]
                    if np.std(xb) > 0 and np.std(yb) > 0:
                        lr_b = LinearRegression()
                        lr_b.fit(xb.reshape(-1, 1), yb)
                        r2s.append(lr_b.score(xb.reshape(-1, 1), yb))
                        rho_b, _ = spearmanr(xb, yb)
                        rhos.append(abs(rho_b))
                if r2s:
                    mean_r2_boots.append(np.mean(r2s))
                    mean_rho_boots.append(np.mean(rhos))

            key = f"{name}_{protocol}"
            results[key] = {
                "model": name,
                "protocol": protocol,
                "n_samples": n,
                "n_layers": n_layers,
                "mean_r2": round(mean_r2, 4),
                "mean_r2_ci": [round(np.percentile(mean_r2_boots, 2.5), 4), round(np.percentile(mean_r2_boots, 97.5), 4)],
                "mean_abs_rho": round(mean_rho, 4),
                "mean_rho_ci": [round(np.percentile(mean_rho_boots, 2.5), 4), round(np.percentile(mean_rho_boots, 97.5), 4)],
            }

            log(f"  {key}: R²={mean_r2:.4f} CI={results[key]['mean_r2_ci']}, |rho|={mean_rho:.4f} CI={results[key]['mean_rho_ci']}")

    with open(OUT_DIR / "b2_exp55_bootstrap_ci.json", 'w') as f:
        json.dump(results, f, indent=2)
    log("B2 saved.")
    return results


# ============================================================
# B3: Exp54 Coverage Sweep
# ============================================================

def run_b3():
    log("=== B3: Exp54 Coverage Sweep ===")
    results = {}
    coverages = [0.6, 0.7, 0.8, 0.9]

    for protocol in ["sampled", "greedy"]:
        for model_key, (name, n_layers, path) in DATA_SOURCES[protocol].items():
            data = load_data(path)
            labels, h_pre, h_post, h_norm, logit_std_arr = extract_features(data, n_layers)
            n = len(data)

            key = f"{name}_{protocol}"
            results[key] = {"model": name, "protocol": protocol, "coverages": {}}

            for cov in coverages:
                # Nested bootstrap: for each bootstrap, re-split, re-select layer, evaluate at coverage
                rng = np.random.RandomState(SEED)
                gaps = []

                for _ in range(N_BOOT):
                    # Bootstrap resample
                    idx = rng.choice(n, n, replace=True)
                    y_b = labels[idx]
                    if len(np.unique(y_b)) < 2:
                        continue

                    # Split 70/30
                    n_cal = int(0.7 * len(idx))
                    cal, test = idx[:n_cal], idx[n_cal:]
                    y_cal, y_test = labels[cal], labels[test]
                    if len(np.unique(y_cal)) < 2 or len(np.unique(y_test)) < 2:
                        continue

                    # Select best layer for H_pre and H_post on cal
                    best_pre_auc, best_pre_l, best_pre_s = 0, 0, 1
                    best_post_auc, best_post_l, best_post_s = 0, 0, 1
                    for layer in range(n_layers):
                        for sign in [+1, -1]:
                            try:
                                a_pre = roc_auc_score(y_cal, h_pre[cal, layer] * sign)
                                if a_pre > best_pre_auc:
                                    best_pre_auc, best_pre_l, best_pre_s = a_pre, layer, sign
                                a_post = roc_auc_score(y_cal, h_post[cal, layer] * sign)
                                if a_post > best_post_auc:
                                    best_post_auc, best_post_l, best_post_s = a_post, layer, sign
                            except ValueError:
                                continue

                    # Scores on test
                    scores_pre = h_pre[test, best_pre_l] * best_pre_s
                    scores_post = h_post[test, best_post_l] * best_post_s

                    # Selective accuracy at coverage
                    k = max(1, int(cov * len(test)))

                    top_pre = np.argsort(-scores_pre)[:k]
                    top_post = np.argsort(-scores_post)[:k]

                    acc_pre = np.mean(y_test[top_pre]) if len(top_pre) > 0 else 0
                    acc_post = np.mean(y_test[top_post]) if len(top_post) > 0 else 0

                    gaps.append(acc_pre - acc_post)

                if gaps:
                    mean_gap = float(np.mean(gaps))
                    ci = [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))]
                    p_val = float(np.mean([1 if g <= 0 else 0 for g in gaps]))
                else:
                    mean_gap, ci, p_val = 0, [0, 0], 1.0

                results[key]["coverages"][str(cov)] = {
                    "mean_gap": round(mean_gap, 4),
                    "ci": [round(ci[0], 4), round(ci[1], 4)],
                    "p_value": round(p_val, 4),
                    "n_valid": len(gaps),
                }

                log(f"  {key} @{int(cov*100)}%: gap={mean_gap:+.4f} CI={ci} p={p_val:.4f}")

    with open(OUT_DIR / "b3_exp54_coverage_sweep.json", 'w') as f:
        json.dump(results, f, indent=2)
    log("B3 saved.")
    return results


# ============================================================
# B4: Exp54 Scenario C Bootstrap CIs
# ============================================================

def run_b4():
    log("=== B4: Exp54 Scenario C Bootstrap CIs ===")
    results = {}

    for protocol in ["sampled", "greedy"]:
        for model_key, (name, n_layers, path) in DATA_SOURCES[protocol].items():
            data = load_data(path)
            labels, h_pre, h_post, h_norm, logit_std_arr = extract_features(data, n_layers)
            n = len(data)

            key = f"{name}_{protocol}"

            # For each bootstrap: split, select best layers independently, then cross-evaluate
            rng = np.random.RandomState(SEED)
            penalties_pre = []  # AUROC(H_pre, H_post's layer) - AUROC(H_pre, H_pre's layer)
            penalties_post = []

            for _ in range(N_BOOT):
                idx = rng.choice(n, n, replace=True)
                n_cal = int(0.7 * len(idx))
                cal, test = idx[:n_cal], idx[n_cal:]
                y_cal, y_test = labels[cal], labels[test]
                if len(np.unique(y_cal)) < 2 or len(np.unique(y_test)) < 2:
                    continue

                # Select best layer/sign for each metric on cal
                best = {}
                for metric_name, metric_data in [("H_pre", h_pre), ("H_post", h_post)]:
                    ba, bl, bs = 0, 0, 1
                    for layer in range(n_layers):
                        for sign in [+1, -1]:
                            try:
                                a = roc_auc_score(y_cal, metric_data[cal, layer] * sign)
                                if a > ba:
                                    ba, bl, bs = a, layer, sign
                            except ValueError:
                                continue
                    best[metric_name] = (bl, bs)

                # Evaluate on test: own layer vs other's layer
                try:
                    # H_pre with its own layer
                    a_pre_own = roc_auc_score(y_test, h_pre[test, best["H_pre"][0]] * best["H_pre"][1])
                    # H_pre with H_post's layer (re-optimize sign on cal)
                    forced_layer = best["H_post"][0]
                    ba_forced, bs_forced = 0, 1
                    for sign in [+1, -1]:
                        a = roc_auc_score(y_cal, h_pre[cal, forced_layer] * sign)
                        if a > ba_forced:
                            ba_forced, bs_forced = a, sign
                    a_pre_forced = roc_auc_score(y_test, h_pre[test, forced_layer] * bs_forced)
                    penalties_pre.append(a_pre_forced - a_pre_own)

                    # H_post with its own layer
                    a_post_own = roc_auc_score(y_test, h_post[test, best["H_post"][0]] * best["H_post"][1])
                    # H_post with H_pre's layer (re-optimize sign on cal)
                    forced_layer2 = best["H_pre"][0]
                    ba_forced2, bs_forced2 = 0, 1
                    for sign in [+1, -1]:
                        a = roc_auc_score(y_cal, h_post[cal, forced_layer2] * sign)
                        if a > ba_forced2:
                            ba_forced2, bs_forced2 = a, sign
                    a_post_forced = roc_auc_score(y_test, h_post[test, forced_layer2] * bs_forced2)
                    penalties_post.append(a_post_forced - a_post_own)
                except ValueError:
                    continue

            results[key] = {
                "model": name,
                "protocol": protocol,
                "H_pre_penalty": {
                    "mean": round(float(np.mean(penalties_pre)), 4) if penalties_pre else 0,
                    "ci": [round(float(np.percentile(penalties_pre, 2.5)), 4), round(float(np.percentile(penalties_pre, 97.5)), 4)] if penalties_pre else [0, 0],
                    "n_valid": len(penalties_pre),
                },
                "H_post_penalty": {
                    "mean": round(float(np.mean(penalties_post)), 4) if penalties_post else 0,
                    "ci": [round(float(np.percentile(penalties_post, 2.5)), 4), round(float(np.percentile(penalties_post, 97.5)), 4)] if penalties_post else [0, 0],
                    "n_valid": len(penalties_post),
                },
            }

            log(f"  {key}: H_pre penalty={results[key]['H_pre_penalty']['mean']:+.4f} CI={results[key]['H_pre_penalty']['ci']}, H_post penalty={results[key]['H_post_penalty']['mean']:+.4f} CI={results[key]['H_post_penalty']['ci']}")

    with open(OUT_DIR / "b4_exp54_scenario_c_ci.json", 'w') as f:
        json.dump(results, f, indent=2)
    log("B4 saved.")
    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    start = time.time()
    log("Starting B2-B3-B4 chain")

    b2 = run_b2()
    b3 = run_b3()
    b4 = run_b4()

    elapsed = time.time() - start
    log(f"All done in {elapsed:.1f}s")

    summary = {
        "elapsed_seconds": round(elapsed, 1),
        "b2_models": len(b2),
        "b3_models": len(b3),
        "b4_models": len(b4),
    }
    with open(OUT_DIR / "chain_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
