"""
Token-pos 재실행 완료 후 자동 생성 스크립트
==========================================
생성 항목:
  1. Table A2 (3모델 × 3 position AUROC) → JSON + 마크다운
  2. Fig 10 (3모델 token-position grouped bar chart) → PNG
  3. EXP-2c 재실행 (Step 0 incremental, 새 데이터) → JSON
  4. 검증 보고서 (exp31 대비 GT 분포 일치 확인)

사용법:
  python scripts/generate_post_tokenpos.py

전제:
  - token-pos 재실행 완료 (qwen/llama/mistral _mmlu_aligned/data/checkpoint.json 존재)
"""

import json
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter
from _paths import POT_DIR

BASE = Path(__file__).parent.parent
EXP_BASE = POT_DIR / "experiments"
TP_BASE = EXP_BASE / "44_GPT25_Experiments" / "phase2_gpu_tokenpos_v2_aligned"
OUTPUT = EXP_BASE / "44_GPT25_Experiments" / "phase4_analysis"

MODEL_CONFIGS = {
    "qwen": {"n_layers": 28, "exp31_path": EXP_BASE / "31_MMLU_Domain_Extension" / "EXP_20260219_053638_mmlu_qwen" / "data" / "sample_results.json"},
    "llama": {"n_layers": 32, "exp31_path": EXP_BASE / "31_MMLU_Domain_Extension" / "EXP_20260219_171237_mmlu_llama" / "data" / "sample_results.json"},
    "mistral": {"n_layers": 32, "exp31_path": EXP_BASE / "31_MMLU_Domain_Extension" / "EXP_20260220_000610_mmlu_mistral" / "data" / "sample_results.json"},
}


def load_checkpoint(model_key):
    path = TP_BASE / f"{model_key}_mmlu_aligned" / "data" / "checkpoint.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def verify_alignment(model_key, data):
    """Verify GT distribution matches exp31 exactly."""
    exp31_path = MODEL_CONFIGS[model_key]["exp31_path"]
    if not exp31_path.exists():
        return {"status": "SKIP", "reason": "exp31 not found"}

    with open(exp31_path) as f:
        exp31 = json.load(f)

    if len(data) != len(exp31):
        return {"status": "FAIL", "reason": f"length mismatch: {len(data)} vs {len(exp31)}"}

    gt_new = Counter(s["ground_truth"] for s in data)
    gt_31 = Counter(s["ground_truth"] for s in exp31)

    mismatches = sum(1 for i in range(len(data))
                     if data[i]["ground_truth"] != exp31[i]["ground_truth"])

    subj_new = Counter(s.get("subject", "?") for s in data)
    subj_31 = Counter(s.get("subject", "?") for s in exp31)
    subj_diffs = sum(1 for k in set(subj_new) | set(subj_31) if subj_new.get(k, 0) != subj_31.get(k, 0))

    corr_new = sum(1 for s in data if s.get("is_correct", False))
    corr_31 = sum(1 for s in exp31 if s.get("is_correct", False))

    return {
        "status": "OK" if mismatches == 0 and gt_new == gt_31 else "FAIL",
        "n_samples": len(data),
        "gt_match": gt_new == gt_31,
        "order_mismatches": mismatches,
        "subject_diffs": subj_diffs,
        "accuracy_new": round(corr_new / len(data), 4),
        "accuracy_exp31": round(corr_31 / len(exp31), 4),
    }


def compute_position_auroc(data, n_layers):
    """Compute held-out AUROC for each position × metric."""
    from sklearn.metrics import roc_auc_score

    n = len(data)
    y = np.array([1 if s.get("is_correct", False) else 0 for s in data])

    positions = ["step0_prompt_last", "step1_first_gen", "full_gen_avg"]
    metrics = ["unnormed_entropy", "normed_entropy", "logit_std", "h_norm"]

    # 70/30 split
    np.random.seed(42)
    n_cal = int(0.7 * n)
    perm = np.random.permutation(n)
    cal_idx, test_idx = perm[:n_cal], perm[n_cal:]

    results = {}
    for pos in positions:
        results[pos] = {}
        for metric in metrics:
            # Extract features
            feats = np.zeros((n, n_layers))
            for i, s in enumerate(data):
                pd = s.get("position_data", {})
                pos_data = pd.get(pos, {})
                for l in range(n_layers):
                    feats[i, l] = pos_data.get(str(l), {}).get(metric, 0)

            # Best layer+sign on cal
            best_layer, best_sign, best_cal = -1, 1, 0.5
            for l in range(n_layers):
                for sign in [1, -1]:
                    try:
                        auc = roc_auc_score(y[cal_idx], sign * feats[cal_idx, l])
                        if auc > best_cal:
                            best_cal = auc
                            best_layer = l
                            best_sign = sign
                    except:
                        pass

            test_auroc = roc_auc_score(y[test_idx], best_sign * feats[test_idx, best_layer])

            results[pos][metric] = {
                "best_layer": int(best_layer),
                "best_sign": int(best_sign),
                "cal_auroc": round(float(best_cal), 6),
                "test_auroc": round(float(test_auroc), 6),
            }

    return results


def generate_table_a2(all_results):
    """Print Table A2 as markdown."""
    positions = ["step0_prompt_last", "step1_first_gen", "full_gen_avg"]
    pos_labels = {"step0_prompt_last": "Step 0", "step1_first_gen": "Step 1", "full_gen_avg": "Full Avg"}
    metrics = ["unnormed_entropy", "normed_entropy", "logit_std", "h_norm"]
    metric_labels = {"unnormed_entropy": "H_pre", "normed_entropy": "H_post", "logit_std": "logit_std", "h_norm": "h_norm"}

    print("\n**Table A2. Held-out test AUROC by token position (3 models, MMLU 1000, 70/30 split, aligned samples)**\n")
    print("| Model | Metric | Step 0 | Step 1 | Full Avg | Best |")
    print("|:------|:-------|:------:|:------:|:--------:|:----:|")

    for model in ["qwen", "llama", "mistral"]:
        if model not in all_results:
            continue
        r = all_results[model]["position_auroc"]
        for metric in metrics:
            vals = {pos: r[pos][metric]["test_auroc"] for pos in positions}
            best_pos = max(vals, key=vals.get)
            row = f"| {model.capitalize()} | {metric_labels[metric]} |"
            for pos in positions:
                v = vals[pos]
                bold = "**" if pos == best_pos else ""
                row += f" {bold}{v:.4f}{bold} |"
            row += f" {pos_labels[best_pos]} |"
            print(row)


def generate_fig10(all_results):
    """Generate Fig 4 v4 (replaces old Fig 4): 3-model token-position grouped bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = ["step0_prompt_last", "step1_first_gen", "full_gen_avg"]
    pos_labels = ["Step 0", "Step 1", "Full Avg"]
    pos_colors = ["#2E86C1", "#E67E22", "#27AE60"]
    metrics = ["unnormed_entropy", "normed_entropy", "logit_std", "h_norm"]
    metric_labels = ["H_pre", "H_post", "logit_std", "h_norm"]
    models = ["qwen", "llama", "mistral"]
    model_labels = ["Qwen", "Llama", "Mistral"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax_idx, (model, model_label) in enumerate(zip(models, model_labels)):
        ax = axes[ax_idx]
        if model not in all_results:
            ax.set_title(f"{model_label} (no data)")
            continue

        r = all_results[model]["position_auroc"]
        x = np.arange(len(metrics))
        width = 0.25

        for j, (pos, pos_label, color) in enumerate(zip(positions, pos_labels, pos_colors)):
            vals = [r[pos][m]["test_auroc"] for m in metrics]
            bars = ax.bar(x + (j - 1) * width, vals, width, label=pos_label, color=color, alpha=0.85, edgecolor="white")
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=0)

        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Metric", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=10)
        ax.set_title(model_label, fontsize=13, fontweight="bold")
        ax.set_ylim(0.45, 0.85)
        ax.grid(True, axis="y", alpha=0.3)
        if ax_idx == 0:
            ax.set_ylabel("Held-out AUROC", fontsize=11)

    # Unified legend at bottom
    import matplotlib.patches as mpatches
    legend_elements = [
        mpatches.Patch(color=pos_colors[0], alpha=0.85, label="Step 0 (prompt-last)"),
        mpatches.Patch(color=pos_colors[1], alpha=0.85, label="Step 1 (first-gen)"),
        mpatches.Patch(color=pos_colors[2], alpha=0.85, label="Full Avg (all gen)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.06), frameon=True, edgecolor="lightgray")

    plt.suptitle("Token-Position Ablation: Step 0 vs Step 1 vs Full Average (MMLU 1000)", fontsize=14, y=1.02)
    plt.tight_layout()

    # Save as fig4 v4 (replacing old fig4)
    out_dir = BASE / "paper" / "figures" / "fig4"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "fig4_v4.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Fig 4 v4 saved: {out_dir / 'fig4_v4.png'} (replaces old Qwen-only fig4)")


def rerun_exp2c(all_results):
    """Re-run EXP-2c Step 0 incremental with aligned data.

    Fix (2026-03-28): Remove sign pre-application on LR features.
    LR learns direction internally; pre-multiplying by sign and feeding
    narrow-range features (e.g. Mistral L4 logit_std range=0.0008) caused
    LR to learn near-zero or inverted coefficients, producing complementary
    AUROC (0.37 instead of 0.63).  StandardScaler on raw values resolves this.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    out_dir = OUTPUT / "EXP2c_step0_incremental_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for model in ["qwen", "llama", "mistral"]:
        if model not in all_results:
            continue
        data = all_results[model]["data"]
        n_layers = MODEL_CONFIGS[model]["n_layers"]
        n = len(data)

        y = np.array([1 if s.get("is_correct", False) else 0 for s in data])

        # Extract Step 0 features
        h_pre = np.zeros((n, n_layers))
        logit_std_arr = np.zeros((n, n_layers))

        for i, s in enumerate(data):
            step0 = s.get("position_data", {}).get("step0_prompt_last", {})
            for l in range(n_layers):
                ld = step0.get(str(l), {})
                h_pre[i, l] = ld.get("unnormed_entropy", 0)
                logit_std_arr[i, l] = ld.get("logit_std", 0)

        # 70/30 split
        np.random.seed(42)
        n_cal = int(0.7 * n)
        perm = np.random.permutation(n)
        cal_idx, test_idx = perm[:n_cal], perm[n_cal:]

        # Best single-layer logit_std (sign-aware raw AUROC for layer selection only)
        best_l, best_s, best_cal = -1, 1, 0.5
        for l in range(n_layers):
            for sign in [1, -1]:
                try:
                    auc = roc_auc_score(y[cal_idx], sign * logit_std_arr[cal_idx, l])
                    if auc > best_cal:
                        best_cal = auc
                        best_l, best_s = l, sign
                except ValueError:
                    pass

        # LR on raw values with StandardScaler (no sign pre-application)
        logit_feat = logit_std_arr[:, best_l].reshape(-1, 1)
        scaler1 = StandardScaler()
        logit_feat_cal = scaler1.fit_transform(logit_feat[cal_idx])
        logit_feat_test = scaler1.transform(logit_feat[test_idx])

        lr1 = LogisticRegression(max_iter=1000)
        lr1.fit(logit_feat_cal, y[cal_idx])
        auc_logit = roc_auc_score(y[test_idx], lr1.predict_proba(logit_feat_test)[:, 1])

        # Best H_pre layer (sign-aware raw AUROC for layer selection only)
        best_l2, best_s2, best_cal2 = -1, 1, 0.5
        for l in range(n_layers):
            for sign in [1, -1]:
                try:
                    auc = roc_auc_score(y[cal_idx], sign * h_pre[cal_idx, l])
                    if auc > best_cal2:
                        best_cal2 = auc
                        best_l2, best_s2 = l, sign
                except ValueError:
                    pass

        hpre_feat = h_pre[:, best_l2].reshape(-1, 1)
        scaler2 = StandardScaler()
        hpre_feat_cal = scaler2.fit_transform(hpre_feat[cal_idx])
        hpre_feat_test = scaler2.transform(hpre_feat[test_idx])

        # logit_std + H_pre (both standardized independently)
        X_cal = np.column_stack([logit_feat_cal, hpre_feat_cal])
        X_test = np.column_stack([logit_feat_test, hpre_feat_test])
        lr2 = LogisticRegression(max_iter=1000)
        lr2.fit(X_cal, y[cal_idx])
        auc_both = roc_auc_score(y[test_idx], lr2.predict_proba(X_test)[:, 1])

        delta = auc_both - auc_logit

        results[f"{model}_mmlu"] = {
            "position": "step0_prompt_last",
            "n_samples": n,
            "accuracy": round(float(y.mean()), 4),
            "logit_std_auroc": round(float(auc_logit), 6),
            "logit_std_plus_hpre_auroc": round(float(auc_both), 6),
            "delta": round(float(delta), 6),
            "logit_std_layer": int(best_l),
            "logit_std_sign": int(best_s),
            "hpre_layer": int(best_l2),
            "hpre_sign": int(best_s2),
        }
        print(f"  {model}_mmlu Step0: logit_std={auc_logit:.4f} (L{best_l}), +H_pre={auc_both:.4f} (L{best_l2}), delta={delta:+.4f}")

    results["_metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "EXP2c_step0_incremental_v2_aligned",
    }

    with open(out_dir / "step0_incremental_v2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_dir / 'step0_incremental_v2_results.json'}")


def main():
    print("=" * 60)
    print("POST TOKEN-POS RERUN: Auto-generation")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)

    # Check completion
    all_results = {}
    all_ok = True
    for model in ["qwen", "llama", "mistral"]:
        data = load_checkpoint(model)
        if data is None:
            print(f"  {model}: NOT FOUND — rerun incomplete?")
            all_ok = False
            continue
        if len(data) < 1000:
            print(f"  {model}: {len(data)}/1000 — still running")
            all_ok = False
            continue
        print(f"  {model}: {len(data)} samples loaded")
        all_results[model] = {"data": data}

    if not all_ok:
        print("\nNot all models complete. Exiting.")
        sys.exit(1)

    # 1. Verify alignment
    print("\n--- Verification ---")
    for model in all_results:
        v = verify_alignment(model, all_results[model]["data"])
        all_results[model]["verification"] = v
        status = v["status"]
        print(f"  {model}: {status} | acc_new={v['accuracy_new']:.1%} acc_exp31={v['accuracy_exp31']:.1%} | order_mismatches={v['order_mismatches']} | subj_diffs={v['subject_diffs']}")
        if status != "OK":
            print(f"    WARNING: Alignment verification failed!")

    # 2. Compute position AUROC
    print("\n--- Position AUROC ---")
    for model in all_results:
        n_layers = MODEL_CONFIGS[model]["n_layers"]
        all_results[model]["position_auroc"] = compute_position_auroc(
            all_results[model]["data"], n_layers
        )
        # Show summary
        r = all_results[model]["position_auroc"]
        for pos in ["step0_prompt_last", "full_gen_avg"]:
            hpre = r[pos]["unnormed_entropy"]["test_auroc"]
            hpost = r[pos]["normed_entropy"]["test_auroc"]
            lstd = r[pos]["logit_std"]["test_auroc"]
            print(f"  {model} {pos[:6]}: H_pre={hpre:.4f} H_post={hpost:.4f} logit_std={lstd:.4f}")

    # 3. Table A2
    print("\n--- Table A2 ---")
    generate_table_a2(all_results)

    # 4. Fig 10
    print("\n--- Fig 10 ---")
    generate_fig10(all_results)

    # 5. EXP-2c rerun
    print("\n--- EXP-2c Rerun (Step 0 Incremental) ---")
    rerun_exp2c(all_results)

    # 6. Save verification report
    report = {
        model: all_results[model]["verification"]
        for model in all_results
    }
    report_path = TP_BASE / "alignment_verification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nVerification report: {report_path}")

    print("\n" + "=" * 60)
    print("ALL DONE. Update draft_v17 with new실측값.")
    print("=" * 60)


if __name__ == "__main__":
    main()
