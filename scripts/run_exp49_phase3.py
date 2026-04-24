"""
Experiment 49 Phase 3: Step 0 Anchor (Fully Deterministic)

Uses EXISTING Step 0 features (prompt-last, no generation dependency)
with GREEDY correctness labels → fully deterministic evaluation.

Input:
  - Step 0 features: phase2_gpu_tokenpos_v2_aligned/{model}_mmlu_aligned/data/checkpoint.json
  - Greedy labels: phase1_greedy/{model}/data/sample_results.json

Output: phase3_step0/*.json

Usage:
  python scripts/run_exp49_phase3.py
  python scripts/run_exp49_phase3.py --smoke_test
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
EXP44_TOKENPOS = (POT_DIR
                  / "experiments" / "44_GPT25_Experiments" / "phase2_gpu_tokenpos_v2_aligned")
EXP49 = BASE / "experiments" / "49_Deterministic_Label_Robustness"
OUT = EXP49 / "phase3_step0"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "qwen": {"tokenpos_dir": "qwen_mmlu_aligned", "n_layers": 28},
    "llama": {"tokenpos_dir": "llama_mmlu_aligned", "n_layers": 32},
    "mistral": {"tokenpos_dir": "mistral_mmlu_aligned", "n_layers": 32},
}
METRICS = ["unnormed_entropy", "normed_entropy", "logit_std", "h_norm"]
METRIC_LABELS = {"unnormed_entropy": "H_pre", "normed_entropy": "H_post",
                 "logit_std": "logit_std", "h_norm": "h_norm"}
N_BOOT = 1000


def log(msg):
    print(f"[EXP49-P3] {msg}", flush=True)


def load_step0_features(model_key):
    """Load Step 0 (prompt-last) features from token-position data."""
    cfg = MODELS[model_key]
    tp_path = EXP44_TOKENPOS / cfg["tokenpos_dir"] / "data" / "checkpoint.json"
    if not tp_path.exists():
        log(f"  Step 0 data not found: {tp_path}")
        return None
    with open(tp_path, encoding='utf-8') as f:
        data = json.load(f)
    log(f"  Loaded {len(data)} samples from token-pos checkpoint")
    return data


def load_greedy_labels(model_key):
    """Load greedy correctness labels."""
    path = EXP49 / "phase1_greedy" / model_key / "data" / "sample_results.json"
    if not path.exists():
        return None
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return {d["idx"]: d["is_correct"] for d in data}


def extract_step0_matrix(tokenpos_data, n_layers):
    """Extract Step 0 feature matrix from token-position data."""
    n = len(tokenpos_data)
    features = {m: np.zeros((n, n_layers)) for m in METRICS}
    original_labels = np.zeros(n, dtype=int)

    for i, sample in enumerate(tokenpos_data):
        original_labels[i] = 1 if sample["is_correct"] else 0
        s0 = sample.get("position_data", {}).get("step0_prompt_last", {})
        for layer in range(n_layers):
            lk = str(layer)
            if lk in s0:
                for metric in METRICS:
                    if metric in s0[lk]:
                        features[metric][i, layer] = s0[lk][metric]

    return features, original_labels


def compute_single_layer_auroc(X, y, layer, sign):
    scores = X[:, layer] * sign
    try:
        return roc_auc_score(y, scores)
    except ValueError:
        return 0.5


def find_best_layer_sign(X, y_cal, cal_idx):
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
        return 0.5, 0.5
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def run_model(model_key, smoke_test=False):
    cfg = MODELS[model_key]
    n_layers = cfg["n_layers"]

    # Load Step 0 features
    tokenpos_data = load_step0_features(model_key)
    if tokenpos_data is None:
        return None

    if smoke_test:
        tokenpos_data = tokenpos_data[:100]

    features, original_labels = extract_step0_matrix(tokenpos_data, n_layers)

    # Load greedy labels
    greedy_labels_map = load_greedy_labels(model_key)
    if greedy_labels_map is None:
        if smoke_test:
            log(f"  SMOKE: using original labels (greedy not available)")
            greedy_y = original_labels
        else:
            log(f"  SKIP: greedy labels not found")
            return None
    else:
        greedy_y = np.array([1 if greedy_labels_map.get(i, False) else 0
                             for i in range(len(tokenpos_data))])

    log(f"  Step 0 samples: {len(tokenpos_data)}")
    log(f"  Original acc: {np.mean(original_labels):.1%}, Greedy acc: {np.mean(greedy_y):.1%}")

    # Table 10 analog: incremental utility at Step 0
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
    cal_idx, test_idx = next(sss.split(np.zeros(len(greedy_y)), greedy_y))

    table10 = {}
    for metric_key, metric_name in METRIC_LABELS.items():
        X = features[metric_key]
        bl, bs, cal_auc = find_best_layer_sign(X, greedy_y[cal_idx], np.arange(len(cal_idx)))
        test_auc = compute_single_layer_auroc(X[test_idx], greedy_y[test_idx], bl, bs)
        ci_lo, ci_hi = bootstrap_auroc_ci(greedy_y[test_idx], X[test_idx, bl] * bs)
        table10[metric_name] = {
            "auroc": round(test_auc, 4), "layer": bl, "sign": bs,
            "ci": [round(ci_lo, 4), round(ci_hi, 4)],
        }

    # Incremental: logit_std + H_pre
    bl_std = table10["logit_std"]["layer"]
    bl_pre = table10["H_pre"]["layer"]
    bs_std = table10["logit_std"]["sign"]
    bs_pre = table10["H_pre"]["sign"]

    # logit_std only
    sc1 = StandardScaler()
    lr1 = LogisticRegression(max_iter=1000)
    lr1.fit(sc1.fit_transform(features["logit_std"][cal_idx, bl_std:bl_std+1]), greedy_y[cal_idx])
    auc_std = roc_auc_score(greedy_y[test_idx],
        lr1.predict_proba(sc1.transform(features["logit_std"][test_idx, bl_std:bl_std+1]))[:, 1])

    # logit_std + H_pre
    X_both_cal = np.column_stack([features["logit_std"][cal_idx, bl_std] * bs_std,
                                   features["unnormed_entropy"][cal_idx, bl_pre] * bs_pre])
    X_both_test = np.column_stack([features["logit_std"][test_idx, bl_std] * bs_std,
                                    features["unnormed_entropy"][test_idx, bl_pre] * bs_pre])
    sc2 = StandardScaler()
    lr2 = LogisticRegression(max_iter=1000)
    lr2.fit(sc2.fit_transform(X_both_cal), greedy_y[cal_idx])
    auc_both = roc_auc_score(greedy_y[test_idx], lr2.predict_proba(sc2.transform(X_both_test))[:, 1])

    # logit_std + h_norm
    bl_hn = table10["h_norm"]["layer"]
    bs_hn = table10["h_norm"]["sign"]
    X_full_cal = np.column_stack([features["logit_std"][cal_idx, bl_std] * bs_std,
                                   features["h_norm"][cal_idx, bl_hn] * bs_hn])
    X_full_test = np.column_stack([features["logit_std"][test_idx, bl_std] * bs_std,
                                    features["h_norm"][test_idx, bl_hn] * bs_hn])
    sc3 = StandardScaler()
    lr3 = LogisticRegression(max_iter=1000)
    lr3.fit(sc3.fit_transform(X_full_cal), greedy_y[cal_idx])
    auc_full = roc_auc_score(greedy_y[test_idx], lr3.predict_proba(sc3.transform(X_full_test))[:, 1])

    # logit_std + h_norm + H_pre
    X_fullp_cal = np.column_stack([X_full_cal, features["unnormed_entropy"][cal_idx, bl_pre] * bs_pre])
    X_fullp_test = np.column_stack([X_full_test, features["unnormed_entropy"][test_idx, bl_pre] * bs_pre])
    sc4 = StandardScaler()
    lr4 = LogisticRegression(max_iter=1000)
    lr4.fit(sc4.fit_transform(X_fullp_cal), greedy_y[cal_idx])
    auc_fullp = roc_auc_score(greedy_y[test_idx], lr4.predict_proba(sc4.transform(X_fullp_test))[:, 1])

    incremental = {
        "logit_std_only": round(auc_std, 4),
        "logit_std_plus_Hpre": round(auc_both, 4),
        "delta_std_Hpre": round(auc_both - auc_std, 4),
        "logit_std_plus_hnorm": round(auc_full, 4),
        "logit_std_plus_hnorm_plus_Hpre": round(auc_fullp, 4),
        "delta_full_Hpre": round(auc_fullp - auc_full, 4),
    }

    log(f"  Table 10: logit_std={table10['logit_std']['auroc']}, "
        f"H_pre={table10['H_pre']['auroc']}, h_norm={table10['h_norm']['auroc']}")
    log(f"  Incremental: delta_std_Hpre={incremental['delta_std_Hpre']}, "
        f"delta_full_Hpre={incremental['delta_full_Hpre']}")

    return {
        "model": model_key,
        "n_samples": len(tokenpos_data),
        "original_accuracy": round(float(np.mean(original_labels)), 4),
        "greedy_accuracy": round(float(np.mean(greedy_y)), 4),
        "protocol": "Step 0 (prompt-last) features + greedy labels = fully deterministic",
        "table10_analog": table10,
        "incremental": incremental,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    log(f"Phase 3 Step 0 Anchor: {'SMOKE TEST' if args.smoke_test else 'FULL RUN'}")

    all_results = {}
    for model_key in ["qwen", "llama", "mistral"]:
        log(f"Processing {model_key}...")
        result = run_model(model_key, smoke_test=args.smoke_test)
        if result:
            all_results[model_key] = result

    suffix = "_smoke" if args.smoke_test else ""
    out_file = OUT / f"phase3_results{suffix}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
