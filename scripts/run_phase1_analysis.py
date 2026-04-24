"""
Phase 1 Analysis: Token Position Comparison
=============================================
3종 비교: Step 0 (prompt-last) / Step 1 (first-gen) / Full average

GPT 12차 수렴 핵심 질문:
  1. 생성 전에 이미 신호가 있는가? (Step 0)
  2. 첫 토큰에서 가장 강한가? (Step 1)
  3. 전체 평균이 필요한가? (Full avg)
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from _paths import POT_DIR

SEED = 42
BASE_DIR = POT_DIR
EXP_DIR = BASE_DIR / "experiments" / "33_Phase1_Token_Position" / "EXP_20260320_052607_hard_token_position"
OUTPUT_DIR = BASE_DIR / "experiments" / "33_Phase1_Token_Position"

METRICS = ["unnormed_entropy", "normed_entropy", "h_norm", "wh_norm",
           "wh_rms", "logit_std", "logit_max", "logit_margin"]
POSITIONS = ["step0_prompt_last", "step1_first_gen", "full_gen_avg"]
NUM_LAYERS = 28


def safe_auroc(y, scores):
    mask = ~np.isnan(scores)
    if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y[mask], scores[mask]))
    except ValueError:
        return 0.5


def load_data():
    with open(EXP_DIR / "data" / "checkpoint.json", "r") as f:
        samples = json.load(f)

    labels = []
    position_metrics = {pos: {m: [] for m in METRICS} for pos in POSITIONS}
    num_tokens_list = []

    for s in samples:
        if "error" in s or "position_data" not in s:
            continue
        pd = s["position_data"]

        # Verify all positions have data
        valid = True
        for pos in POSITIONS:
            if pd.get(pos) is None:
                valid = False
                break
        if not valid:
            continue

        labels.append(1 if s["is_correct"] else 0)
        num_tokens_list.append(s.get("num_tokens", 0))

        for pos in POSITIONS:
            pos_data = pd[pos]
            for m in METRICS:
                vals = []
                for li in range(NUM_LAYERS):
                    lk = str(li)
                    if lk in pos_data and m in pos_data[lk]:
                        vals.append(pos_data[lk][m])
                    else:
                        vals.append(np.nan)
                position_metrics[pos][m].append(vals)

    labels = np.array(labels)
    for pos in POSITIONS:
        for m in METRICS:
            position_metrics[pos][m] = np.array(position_metrics[pos][m], dtype=np.float64)

    num_tokens = np.array(num_tokens_list, dtype=np.float64)
    print(f"Loaded: {len(labels)} valid samples, {labels.sum()} correct ({labels.mean()*100:.1f}%)")
    return labels, position_metrics, num_tokens


def analyze():
    print("=" * 70)
    print("Phase 1 Analysis: Token Position Comparison (Qwen Hard)")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    labels, position_metrics, num_tokens = load_data()
    n = len(labels)

    # 70/30 split (same protocol as Phase 0)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
    cal_idx, test_idx = next(splitter.split(np.zeros(n), labels))
    y_cal, y_test = labels[cal_idx], labels[test_idx]

    print(f"Split: cal={len(cal_idx)} ({y_cal.mean()*100:.1f}%), test={len(test_idx)} ({y_test.mean()*100:.1f}%)")

    results = {
        "experiment": "Phase 1 Token Position Analysis",
        "dataset": "Qwen Hard (competition_math L4-5)",
        "n_samples": n,
        "accuracy": round(float(labels.mean()), 6),
        "split": {"cal": len(cal_idx), "test": len(test_idx)},
        "timestamp": datetime.now().isoformat(),
        "positions": {},
    }

    # ============================================================
    # For each position × metric: select best layer on cal, eval on test
    # ============================================================
    print(f"\n{'Position':25s} {'Metric':25s} {'Layer':>6s} {'Sign':>5s} {'Cal AUROC':>10s} {'Test AUROC':>11s}")
    print("-" * 85)

    for pos in POSITIONS:
        results["positions"][pos] = {"baselines": {}}

        for m in METRICS:
            X = position_metrics[pos][m]
            X_cal, X_test = X[cal_idx], X[test_idx]

            # Select best layer + sign on cal
            best_layer, best_sign, best_cal = 0, 1, -1
            for l in range(NUM_LAYERS):
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

            # Eval on test
            test_scores = best_sign * X_test[:, best_layer]
            test_auroc = safe_auroc(y_test, test_scores)

            sign_str = "+" if best_sign == 1 else "-"
            print(f"{pos:25s} {m:25s} {best_layer:6d} {sign_str:>5s} {best_cal:10.4f} {test_auroc:11.4f}")

            results["positions"][pos]["baselines"][m] = {
                "cal_best_layer": int(best_layer),
                "cal_best_sign": int(best_sign),
                "cal_auroc": round(best_cal, 6),
                "test_auroc": round(test_auroc, 6),
            }

    # ============================================================
    # Summary: Compare positions for key metrics
    # ============================================================
    print(f"\n{'='*70}")
    print("POSITION COMPARISON SUMMARY (Test AUROC)")
    print(f"{'='*70}")
    print(f"{'Metric':25s} {'Step0':>10s} {'Step1':>10s} {'FullAvg':>10s} {'Best':>12s}")
    print("-" * 70)

    comparison = {}
    for m in METRICS:
        aurocs = {}
        for pos in POSITIONS:
            aurocs[pos] = results["positions"][pos]["baselines"][m]["test_auroc"]

        best_pos = max(aurocs, key=aurocs.get)
        best_val = aurocs[best_pos]
        marker = {pos: " *" if pos == best_pos else "" for pos in POSITIONS}

        print(f"{m:25s} {aurocs['step0_prompt_last']:10.4f}{marker['step0_prompt_last']:2s}"
              f" {aurocs['step1_first_gen']:10.4f}{marker['step1_first_gen']:2s}"
              f" {aurocs['full_gen_avg']:10.4f}{marker['full_gen_avg']:2s}"
              f" {best_pos.replace('_', ' '):>12s}")

        comparison[m] = {
            "step0": aurocs["step0_prompt_last"],
            "step1": aurocs["step1_first_gen"],
            "full_avg": aurocs["full_gen_avg"],
            "best_position": best_pos,
        }

    results["comparison"] = comparison

    # ============================================================
    # Key finding: H_pre at each position
    # ============================================================
    print(f"\n{'='*70}")
    print("H_PRE (unnormed_entropy) DETAIL BY POSITION")
    print(f"{'='*70}")

    for pos in POSITIONS:
        bl = results["positions"][pos]["baselines"]["unnormed_entropy"]
        print(f"  {pos:25s} | Layer {bl['cal_best_layer']:2d} "
              f"({'+'if bl['cal_best_sign']==1 else '-'}) | "
              f"Cal {bl['cal_auroc']:.4f} | Test {bl['test_auroc']:.4f}")

    # Save
    outpath = OUTPUT_DIR / "phase1_position_analysis.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {outpath}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    analyze()
