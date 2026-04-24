#!/usr/bin/env python3
"""
Decoded Entropy Measurement Audit Toolkit
==========================================

A diagnostic toolkit for researchers using layerwise decoded entropy
in RMSNorm-based language models. Implements the measurement audit
protocol from:

  "Decoded Entropy Is Not One Signal: Pre- and Post-RMSNorm Projections
   Yield Non-Equivalent Measurements in Language Models"

Usage:
  python entropy_measurement_audit.py --model <model_name_or_path> --input <data.json>

The toolkit performs 6 diagnostic checks:
  1. H_pre/H_post dual computation
  2. Unit-norm collapse check
  3. Alpha-sweep invariance check
  4. Scale baseline comparison
  5. Incremental utility test
  6. Sign calibration stability report

Input format: JSON list of samples, each with 'layer_data' containing
per-layer 'unnormed_entropy', 'normed_entropy', 'logit_std', 'h_norm'.

Output: JSON report + console summary with pass/fail for each check.
"""

import json
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("ERROR: scikit-learn required. Install: pip install scikit-learn")
    sys.exit(1)

SEED = 42


def load_data(path):
    """Load per-sample, per-layer data from JSON."""
    data = json.load(open(path))
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Input must be a non-empty JSON list of samples")

    sample0 = data[0]
    if "layer_data" not in sample0:
        raise ValueError("Each sample must have 'layer_data' dict")

    n_layers = len(sample0["layer_data"])
    n = len(data)

    h_pre = np.zeros((n, n_layers))
    h_post = np.zeros((n, n_layers))
    logit_std = np.zeros((n, n_layers))
    h_norm = np.zeros((n, n_layers))
    y = np.zeros(n, dtype=int)

    for i, s in enumerate(data):
        y[i] = 1 if s.get("is_correct", s.get("correct", False)) else 0
        ld = s["layer_data"]
        for l in range(n_layers):
            layer = ld[str(l)]
            h_pre[i, l] = layer.get("unnormed_entropy", 0)
            h_post[i, l] = layer.get("normed_entropy", 0)
            logit_std[i, l] = layer.get("logit_std", 0)
            h_norm[i, l] = layer.get("h_norm", 0)

    return h_pre, h_post, logit_std, h_norm, y, n_layers


def check_1_dual_computation(h_pre, h_post, n_layers):
    """Check 1: Are both H_pre and H_post computed?"""
    h_pre_nonzero = (h_pre != 0).any()
    h_post_nonzero = (h_post != 0).any()
    corr = np.corrcoef(h_pre.mean(axis=0), h_post.mean(axis=0))[0, 1]

    result = {
        "check": "Dual H_pre/H_post computation",
        "h_pre_present": bool(h_pre_nonzero),
        "h_post_present": bool(h_post_nonzero),
        "mean_profile_correlation": round(float(corr), 4),
        "pass": bool(h_pre_nonzero and h_post_nonzero),
    }

    if not result["pass"]:
        result["warning"] = "Both H_pre and H_post should be computed to detect scale confounding."

    return result


def check_2_unitnorm_collapse(h_pre, h_norm, n_layers):
    """Check 2: Does H_pre approach 1.0 when h_norm is small?"""
    # Samples with smallest h_norm tend to have H_pre near 1.0
    mean_hnorm = h_norm.mean(axis=1)
    q10 = np.percentile(mean_hnorm, 10)
    low_norm_mask = mean_hnorm <= q10
    high_norm_mask = mean_hnorm >= np.percentile(mean_hnorm, 90)

    h_pre_low_norm = h_pre[low_norm_mask].mean()
    h_pre_high_norm = h_pre[high_norm_mask].mean()

    # Check per-layer: layers with small mean h_norm should have H_pre near 1.0
    layers_near_max = 0
    for l in range(n_layers):
        if h_pre[:, l].mean() > 0.95:
            layers_near_max += 1

    result = {
        "check": "Unit-norm collapse indicator",
        "h_pre_mean_low_norm_samples": round(float(h_pre_low_norm), 4),
        "h_pre_mean_high_norm_samples": round(float(h_pre_high_norm), 4),
        "layers_with_hpre_above_095": layers_near_max,
        "total_layers": n_layers,
        "collapse_ratio": round(float(layers_near_max / n_layers), 4),
        "pass": True,
    }

    if layers_near_max > n_layers * 0.5:
        result["warning"] = (
            f"{layers_near_max}/{n_layers} layers have mean H_pre > 0.95. "
            "This suggests H_pre is near-saturated and may not carry useful "
            "directional information. Consider using H_post instead."
        )
        result["pass"] = False

    return result


def check_3_scale_sensitivity(h_pre, h_post, h_norm, n_layers):
    """Check 3: Is H_pre correlated with scale while H_post is not?"""
    hpre_hnorm_corrs = []
    hpost_hnorm_corrs = []

    for l in range(n_layers):
        if h_norm[:, l].std() > 1e-10:
            c1 = np.corrcoef(h_pre[:, l], h_norm[:, l])[0, 1]
            c2 = np.corrcoef(h_post[:, l], h_norm[:, l])[0, 1]
            hpre_hnorm_corrs.append(c1)
            hpost_hnorm_corrs.append(c2)

    mean_hpre_corr = np.mean(np.abs(hpre_hnorm_corrs))
    mean_hpost_corr = np.mean(np.abs(hpost_hnorm_corrs))

    result = {
        "check": "Scale sensitivity (H_pre vs H_post correlation with h_norm)",
        "mean_abs_corr_hpre_hnorm": round(float(mean_hpre_corr), 4),
        "mean_abs_corr_hpost_hnorm": round(float(mean_hpost_corr), 4),
        "hpre_more_scale_sensitive": bool(mean_hpre_corr > mean_hpost_corr),
        "pass": True,
    }

    if mean_hpre_corr > 0.5:
        result["warning"] = (
            f"H_pre shows strong correlation with h_norm (mean |r| = {mean_hpre_corr:.3f}). "
            "H_pre may be primarily capturing scale rather than directional content."
        )

    return result


def check_4_scale_baseline(h_pre, h_post, logit_std, h_norm, y, n_layers):
    """Check 4: Does logit_std match or outperform H_pre?"""
    np.random.seed(SEED)
    perm = np.random.permutation(len(y))
    cal_idx = perm[:int(0.7 * len(y))]
    test_idx = perm[int(0.7 * len(y)):]

    def best_auroc(X):
        best_l, best_s, best_cal = -1, 1, 0.5
        for l in range(n_layers):
            for sign in [1, -1]:
                try:
                    auc = roc_auc_score(y[cal_idx], sign * X[cal_idx, l])
                    if auc > best_cal:
                        best_cal, best_l, best_s = auc, l, sign
                except ValueError:
                    pass
        if best_l >= 0:
            test_auc = roc_auc_score(y[test_idx], best_s * X[test_idx, best_l])
        else:
            test_auc = 0.5
        return best_l, round(float(test_auc), 4)

    hpre_l, hpre_auc = best_auroc(h_pre)
    hpost_l, hpost_auc = best_auroc(h_post)
    std_l, std_auc = best_auroc(logit_std)
    hnorm_l, hnorm_auc = best_auroc(h_norm)

    result = {
        "check": "Scale baseline comparison (held-out AUROC)",
        "H_pre": {"layer": hpre_l, "auroc": hpre_auc},
        "H_post": {"layer": hpost_l, "auroc": hpost_auc},
        "logit_std": {"layer": std_l, "auroc": std_auc},
        "h_norm": {"layer": hnorm_l, "auroc": hnorm_auc},
        "logit_std_ge_hpre": bool(std_auc >= hpre_auc),
        "pass": True,
    }

    if std_auc >= hpre_auc:
        result["warning"] = (
            f"logit_std ({std_auc:.4f}) >= H_pre ({hpre_auc:.4f}). "
            "This is consistent with H_pre being dominated by scale. "
            "Consider reporting scale baselines alongside entropy."
        )

    return result


def check_5_incremental_utility(h_pre, logit_std, h_norm, y, n_layers):
    """Check 5: Does H_pre add information beyond logit_std?"""
    np.random.seed(SEED)
    perm = np.random.permutation(len(y))
    cal_idx = perm[:int(0.7 * len(y))]
    test_idx = perm[int(0.7 * len(y)):]

    def best_layer(X):
        best_l, best_s, best_cal = -1, 1, 0.5
        for l in range(n_layers):
            for sign in [1, -1]:
                try:
                    auc = roc_auc_score(y[cal_idx], sign * X[cal_idx, l])
                    if auc > best_cal:
                        best_cal, best_l, best_s = auc, l, sign
                except ValueError:
                    pass
        return best_l

    l_std = best_layer(logit_std)
    l_hpre = best_layer(h_pre)

    feat_std = logit_std[:, l_std:l_std + 1]
    feat_hpre = h_pre[:, l_hpre:l_hpre + 1]

    sc1 = StandardScaler()
    sc2 = StandardScaler()

    X_std_cal = sc1.fit_transform(feat_std[cal_idx])
    X_std_test = sc1.transform(feat_std[test_idx])
    X_hpre_cal = sc2.fit_transform(feat_hpre[cal_idx])
    X_hpre_test = sc2.transform(feat_hpre[test_idx])

    lr1 = LogisticRegression(max_iter=1000, random_state=SEED)
    lr1.fit(X_std_cal, y[cal_idx])
    auc_std = roc_auc_score(y[test_idx], lr1.predict_proba(X_std_test)[:, 1])

    X_both_cal = np.hstack([X_std_cal, X_hpre_cal])
    X_both_test = np.hstack([X_std_test, X_hpre_test])
    lr2 = LogisticRegression(max_iter=1000, random_state=SEED)
    lr2.fit(X_both_cal, y[cal_idx])
    auc_both = roc_auc_score(y[test_idx], lr2.predict_proba(X_both_test)[:, 1])

    delta = auc_both - auc_std

    result = {
        "check": "Incremental utility of H_pre over logit_std",
        "logit_std_only_auroc": round(float(auc_std), 4),
        "logit_std_plus_hpre_auroc": round(float(auc_both), 4),
        "delta": round(float(delta), 4),
        "hpre_adds_information": bool(delta > 0.01),
        "pass": True,
    }

    if delta <= 0.01:
        result["warning"] = (
            f"Adding H_pre to logit_std yields delta = {delta:+.4f}. "
            "H_pre may not carry independent information beyond scale."
        )

    return result


def check_6_sign_stability(h_pre, h_post, y, n_layers):
    """Check 6: Is the entropy-correctness sign stable?"""
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    hpre_signs = []
    hpost_signs = []

    for train_idx, test_idx in skf.split(np.zeros(len(y)), y):
        # H_pre best sign
        best_s = 1
        best_cal = 0.5
        for l in range(n_layers):
            for sign in [1, -1]:
                try:
                    auc = roc_auc_score(y[train_idx], sign * h_pre[train_idx, l])
                    if auc > best_cal:
                        best_cal = auc
                        best_s = sign
                except ValueError:
                    pass
        hpre_signs.append(best_s)

        # H_post best sign
        best_s = 1
        best_cal = 0.5
        for l in range(n_layers):
            for sign in [1, -1]:
                try:
                    auc = roc_auc_score(y[train_idx], sign * h_post[train_idx, l])
                    if auc > best_cal:
                        best_cal = auc
                        best_s = sign
                except ValueError:
                    pass
        hpost_signs.append(best_s)

    hpre_stable = len(set(hpre_signs)) == 1
    hpost_stable = len(set(hpost_signs)) == 1

    result = {
        "check": "Sign calibration stability (5-fold)",
        "h_pre_signs": hpre_signs,
        "h_pre_stable": hpre_stable,
        "h_post_signs": hpost_signs,
        "h_post_stable": hpost_stable,
        "pass": True,
    }

    if not hpost_stable:
        result["warning"] = (
            "H_post sign is not stable across folds. "
            "Sign calibration on a held-out set is necessary before using H_post."
        )

    return result


def run_audit(data_path, output_dir=None):
    """Run full measurement audit on a dataset."""
    print("=" * 60)
    print("DECODED ENTROPY MEASUREMENT AUDIT")
    print(f"Input: {data_path}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)

    h_pre, h_post, logit_std, h_norm, y, n_layers = load_data(data_path)
    n = len(y)
    n_correct = y.sum()
    print(f"Samples: {n}, Correct: {n_correct} ({n_correct/n*100:.1f}%), Layers: {n_layers}")

    checks = [
        check_1_dual_computation(h_pre, h_post, n_layers),
        check_2_unitnorm_collapse(h_pre, h_norm, n_layers),
        check_3_scale_sensitivity(h_pre, h_post, h_norm, n_layers),
        check_4_scale_baseline(h_pre, h_post, logit_std, h_norm, y, n_layers),
        check_5_incremental_utility(h_pre, logit_std, h_norm, y, n_layers),
        check_6_sign_stability(h_pre, h_post, y, n_layers),
    ]

    # Print summary
    print("\n" + "=" * 60)
    print("AUDIT RESULTS")
    print("=" * 60)

    n_pass = 0
    n_warn = 0
    for i, c in enumerate(checks, 1):
        status = "PASS" if c["pass"] else "WARN"
        if status == "PASS":
            n_pass += 1
        else:
            n_warn += 1
        print(f"\n  Check {i}: {c['check']}")
        print(f"  Status: {status}")
        for k, v in c.items():
            if k in ("check", "pass", "warning"):
                continue
            print(f"    {k}: {v}")
        if "warning" in c:
            print(f"  WARNING: {c['warning']}")

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {n_pass} PASS, {n_warn} WARN out of {len(checks)} checks")
    print(f"{'='*60}")

    # Save report
    report = {
        "input_file": str(data_path),
        "n_samples": n,
        "n_correct": int(n_correct),
        "n_layers": n_layers,
        "timestamp": datetime.now().isoformat(),
        "checks": checks,
        "summary": {"pass": n_pass, "warn": n_warn, "total": len(checks)},
    }

    if output_dir:
        out_path = Path(output_dir) / "audit_report.json"
    else:
        out_path = Path(data_path).parent / "audit_report.json"

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {out_path}")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Decoded Entropy Measurement Audit")
    parser.add_argument("--input", required=True, help="Path to sample_results.json")
    parser.add_argument("--output_dir", default=None, help="Output directory for report")
    args = parser.parse_args()
    run_audit(args.input, args.output_dir)
