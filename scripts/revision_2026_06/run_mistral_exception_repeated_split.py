"""Mistral full-scale + Hpre exception repeated-split stability analysis.

This public copy writes outputs under results/revision_2026_06/mistral_exception_stability.
It reads the existing Mistral MMLU scalar-feature artifact and does not require
new model downloads, GPU inference, or hidden-state extraction.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


EXPECTED_EXP42 = {
    "logit_base": 0.7314779635258359,
    "logit_aug": 0.7365596504559271,
    "full_base": 0.7267287234042554,
    "full_aug": 0.7430186170212766,
    "delta_logit": 0.005081686930091145,
    "delta_full": 0.016289893617021156,
    "hpre_layer": 30,
    "hpre_sign": -1,
    "logit_layer": 13,
    "logit_sign": 1,
    "hnorm_layer": 13,
    "hnorm_sign": 1,
}


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    layer: int
    sign: int


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "PoT_Experiment_Entropy_Attention_Extraction_Experiment").exists():
            return parent
    raise RuntimeError("Could not locate repository root from script path.")


def load_records(root: Path) -> list[dict]:
    data_path = root / (
        "PoT_Experiment_Entropy_Attention_Extraction_Experiment/experiments/"
        "31_MMLU_Domain_Extension/EXP_20260220_000610_mmlu_mistral/"
        "data/sample_results.json"
    )
    with data_path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if len(records) != 1000:
        raise ValueError(f"Expected 1000 Mistral MMLU records, found {len(records)}")
    return records


def find_best_layer_sign(
    records: list[dict], metric: str, n_layers: int, cal_idx: np.ndarray
) -> tuple[int, int, float]:
    y_cal = np.array([1 if records[i]["is_correct"] else 0 for i in cal_idx])
    best_auc = -np.inf
    best_layer = 0
    best_sign = 1

    for layer in range(n_layers):
        values = np.array(
            [records[i]["layer_data"][str(layer)][metric] for i in cal_idx],
            dtype=float,
        )
        for sign in (1, -1):
            try:
                auc = roc_auc_score(y_cal, sign * values)
            except ValueError:
                continue
            if auc > best_auc:
                best_auc = auc
                best_layer = layer
                best_sign = sign
    return best_layer, best_sign, float(best_auc)


def build_matrix(records: list[dict], indices: np.ndarray, specs: list[FeatureSpec]) -> np.ndarray:
    rows: list[list[float]] = []
    for i in indices:
        row: list[float] = []
        for spec in specs:
            if spec.name == "length":
                value = records[i]["num_tokens"]
            else:
                value = records[i]["layer_data"][str(spec.layer)][spec.name]
            row.append(float(spec.sign) * float(value))
        rows.append(row)
    return np.asarray(rows, dtype=float)


def fit_score_auc(
    records: list[dict],
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    specs: list[FeatureSpec],
    y_cal: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, np.ndarray]:
    x_cal = build_matrix(records, cal_idx, specs)
    x_test = build_matrix(records, test_idx, specs)
    scaler = StandardScaler()
    x_cal_s = scaler.fit_transform(x_cal)
    x_test_s = scaler.transform(x_test)
    lr = LogisticRegression(max_iter=1000, solver="lbfgs")
    lr.fit(x_cal_s, y_cal)
    scores = lr.predict_proba(x_test_s)[:, 1]
    return float(roc_auc_score(y_test, scores)), scores


def run_one_split(records: list[dict], seed: int, fixed_layers: bool) -> dict:
    n_layers = len(records[0]["layer_data"])
    y_all = np.array([1 if row["is_correct"] else 0 for row in records])
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    cal_idx, test_idx = next(splitter.split(np.zeros(len(records)), y_all))
    y_cal = y_all[cal_idx]
    y_test = y_all[test_idx]

    if fixed_layers:
        hpre_layer, hpre_sign, hpre_cal_auc = (
            EXPECTED_EXP42["hpre_layer"],
            EXPECTED_EXP42["hpre_sign"],
            None,
        )
        logit_layer, logit_sign, logit_cal_auc = (
            EXPECTED_EXP42["logit_layer"],
            EXPECTED_EXP42["logit_sign"],
            None,
        )
        hnorm_layer, hnorm_sign, hnorm_cal_auc = (
            EXPECTED_EXP42["hnorm_layer"],
            EXPECTED_EXP42["hnorm_sign"],
            None,
        )
        mode = "fixed_seed42_layers"
    else:
        hpre_layer, hpre_sign, hpre_cal_auc = find_best_layer_sign(
            records, "unnormed_entropy", n_layers, cal_idx
        )
        logit_layer, logit_sign, logit_cal_auc = find_best_layer_sign(
            records, "logit_std", n_layers, cal_idx
        )
        hnorm_layer, hnorm_sign, hnorm_cal_auc = find_best_layer_sign(
            records, "h_norm", n_layers, cal_idx
        )
        mode = "dynamic_cal_selected_layers"

    logit_only = [FeatureSpec("logit_std", logit_layer, logit_sign)]
    logit_plus_hpre = [
        FeatureSpec("logit_std", logit_layer, logit_sign),
        FeatureSpec("unnormed_entropy", hpre_layer, hpre_sign),
    ]
    full_scale = [
        FeatureSpec("logit_std", logit_layer, logit_sign),
        FeatureSpec("h_norm", hnorm_layer, hnorm_sign),
        FeatureSpec("length", 0, 1),
    ]
    full_plus_hpre = [
        FeatureSpec("logit_std", logit_layer, logit_sign),
        FeatureSpec("h_norm", hnorm_layer, hnorm_sign),
        FeatureSpec("length", 0, 1),
        FeatureSpec("unnormed_entropy", hpre_layer, hpre_sign),
    ]

    logit_auc, _ = fit_score_auc(records, cal_idx, test_idx, logit_only, y_cal, y_test)
    logit_hpre_auc, _ = fit_score_auc(
        records, cal_idx, test_idx, logit_plus_hpre, y_cal, y_test
    )
    full_auc, _ = fit_score_auc(records, cal_idx, test_idx, full_scale, y_cal, y_test)
    full_hpre_auc, _ = fit_score_auc(
        records, cal_idx, test_idx, full_plus_hpre, y_cal, y_test
    )

    return {
        "mode": mode,
        "seed": seed,
        "n_cal": int(len(cal_idx)),
        "n_test": int(len(test_idx)),
        "test_accuracy": float(y_test.mean()),
        "hpre_layer": int(hpre_layer),
        "hpre_sign": int(hpre_sign),
        "hpre_cal_auc": hpre_cal_auc,
        "logit_layer": int(logit_layer),
        "logit_sign": int(logit_sign),
        "logit_cal_auc": logit_cal_auc,
        "hnorm_layer": int(hnorm_layer),
        "hnorm_sign": int(hnorm_sign),
        "hnorm_cal_auc": hnorm_cal_auc,
        "logit_base_auroc": logit_auc,
        "logit_plus_hpre_auroc": logit_hpre_auc,
        "logit_delta": logit_hpre_auc - logit_auc,
        "full_base_auroc": full_auc,
        "full_plus_hpre_auroc": full_hpre_auc,
        "full_delta": full_hpre_auc - full_auc,
    }


def summarize_rows(rows: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for mode in sorted(set(row["mode"] for row in rows)):
        mode_rows = [row for row in rows if row["mode"] == mode]
        for delta_key in ("logit_delta", "full_delta"):
            values = np.array([row[delta_key] for row in mode_rows], dtype=float)
            summary[f"{mode}:{delta_key}"] = {
                "n_splits": int(len(values)),
                "mean": float(mean(values)),
                "median": float(median(values)),
                "sd_population": float(pstdev(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "p05": float(np.percentile(values, 5)),
                "p25": float(np.percentile(values, 25)),
                "p75": float(np.percentile(values, 75)),
                "p95": float(np.percentile(values, 95)),
                "positive_count": int(np.sum(values > 0)),
                "positive_rate": float(np.mean(values > 0)),
                "gt_0p005_count": int(np.sum(values > 0.005)),
                "gt_0p005_rate": float(np.mean(values > 0.005)),
                "gt_0p010_count": int(np.sum(values > 0.010)),
                "gt_0p010_rate": float(np.mean(values > 0.010)),
                "lt_minus_0p005_count": int(np.sum(values < -0.005)),
                "lt_minus_0p005_rate": float(np.mean(values < -0.005)),
            }

        if mode == "dynamic_cal_selected_layers":
            for feature in ("hpre", "logit", "hnorm"):
                layers = [row[f"{feature}_layer"] for row in mode_rows]
                signs = [row[f"{feature}_sign"] for row in mode_rows]
                summary[f"{mode}:{feature}_layer_sign"] = {
                    "layer_counts": dict(Counter(layers).most_common()),
                    "sign_counts": dict(Counter(signs).most_common()),
                }
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "mode",
        "seed",
        "n_cal",
        "n_test",
        "test_accuracy",
        "hpre_layer",
        "hpre_sign",
        "hpre_cal_auc",
        "logit_layer",
        "logit_sign",
        "logit_cal_auc",
        "hnorm_layer",
        "hnorm_sign",
        "hnorm_cal_auc",
        "logit_base_auroc",
        "logit_plus_hpre_auroc",
        "logit_delta",
        "full_base_auroc",
        "full_plus_hpre_auroc",
        "full_delta",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def make_markdown(summary: dict, seed42_rows: list[dict], n_splits: int) -> str:
    def s(key: str) -> dict:
        return summary[key]

    dyn_full = s("dynamic_cal_selected_layers:full_delta")
    dyn_logit = s("dynamic_cal_selected_layers:logit_delta")
    fix_full = s("fixed_seed42_layers:full_delta")
    fix_logit = s("fixed_seed42_layers:logit_delta")

    lines = [
        "# Mistral Exception Repeated-Split Summary v1",
        "",
        f"작성일: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Purpose",
        "",
        "Check whether the E10 Mistral MMLU exception is a one-split artifact or a repeated-split stable pattern.",
        "",
        "Primary question:",
        "",
        "> Does adding Hpre to the full scale baseline repeatedly improve AUROC for Mistral MMLU sampled evaluation?",
        "",
        "## Protocol",
        "",
        f"- n repeated stratified 70/30 splits: {n_splits}",
        "- dataset: existing Mistral MMLU `sample_results.json` only",
        "- no new model download, no GPU inference, no hidden-state extraction",
        "- classifier: `StandardScaler` fit on calibration only + `LogisticRegression(max_iter=1000, solver='lbfgs')`",
        "- dynamic mode: select best layer/sign on each calibration split, matching exp42 protocol",
        "- fixed mode: reuse seed=42 layers/signs to isolate split variation from layer-selection variation",
        "",
        "## Seed 42 Reproduction Check",
        "",
        "| Mode | logit delta | full delta | hpre layer/sign | logit layer/sign | hnorm layer/sign |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in seed42_rows:
        lines.append(
            "| {mode} | {ld:+.6f} | {fd:+.6f} | L{hl}/{hs:+d} | L{ll}/{ls:+d} | L{nl}/{ns:+d} |".format(
                mode=row["mode"],
                ld=row["logit_delta"],
                fd=row["full_delta"],
                hl=row["hpre_layer"],
                hs=row["hpre_sign"],
                ll=row["logit_layer"],
                ls=row["logit_sign"],
                nl=row["hnorm_layer"],
                ns=row["hnorm_sign"],
            )
        )

    lines += [
        "",
        "Expected exp42 seed=42 full delta: +0.016290.",
        "",
        "## Repeated-Split Results",
        "",
        "| Mode | Comparison | Mean delta | Median | SD | 5-95% range | Positive rate | > +0.005 | > +0.010 |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|",
        "| dynamic | logit -> logit+Hpre | {mean:+.6f} | {median:+.6f} | {sd:.6f} | [{p05:+.6f}, {p95:+.6f}] | {pos:.1%} | {g5:.1%} | {g10:.1%} |".format(
            mean=dyn_logit["mean"],
            median=dyn_logit["median"],
            sd=dyn_logit["sd_population"],
            p05=dyn_logit["p05"],
            p95=dyn_logit["p95"],
            pos=dyn_logit["positive_rate"],
            g5=dyn_logit["gt_0p005_rate"],
            g10=dyn_logit["gt_0p010_rate"],
        ),
        "| dynamic | full scale -> full scale+Hpre | {mean:+.6f} | {median:+.6f} | {sd:.6f} | [{p05:+.6f}, {p95:+.6f}] | {pos:.1%} | {g5:.1%} | {g10:.1%} |".format(
            mean=dyn_full["mean"],
            median=dyn_full["median"],
            sd=dyn_full["sd_population"],
            p05=dyn_full["p05"],
            p95=dyn_full["p95"],
            pos=dyn_full["positive_rate"],
            g5=dyn_full["gt_0p005_rate"],
            g10=dyn_full["gt_0p010_rate"],
        ),
        "| fixed | logit -> logit+Hpre | {mean:+.6f} | {median:+.6f} | {sd:.6f} | [{p05:+.6f}, {p95:+.6f}] | {pos:.1%} | {g5:.1%} | {g10:.1%} |".format(
            mean=fix_logit["mean"],
            median=fix_logit["median"],
            sd=fix_logit["sd_population"],
            p05=fix_logit["p05"],
            p95=fix_logit["p95"],
            pos=fix_logit["positive_rate"],
            g5=fix_logit["gt_0p005_rate"],
            g10=fix_logit["gt_0p010_rate"],
        ),
        "| fixed | full scale -> full scale+Hpre | {mean:+.6f} | {median:+.6f} | {sd:.6f} | [{p05:+.6f}, {p95:+.6f}] | {pos:.1%} | {g5:.1%} | {g10:.1%} |".format(
            mean=fix_full["mean"],
            median=fix_full["median"],
            sd=fix_full["sd_population"],
            p05=fix_full["p05"],
            p95=fix_full["p95"],
            pos=fix_full["positive_rate"],
            g5=fix_full["gt_0p005_rate"],
            g10=fix_full["gt_0p010_rate"],
        ),
        "",
        "## Dynamic Layer/Sign Stability",
        "",
    ]

    for feature in ("hpre", "logit", "hnorm"):
        key = f"dynamic_cal_selected_layers:{feature}_layer_sign"
        entry = summary[key]
        lines += [
            f"### {feature}",
            "",
            f"- layer counts: `{entry['layer_counts']}`",
            f"- sign counts: `{entry['sign_counts']}`",
            "",
        ]

    lines += [
        "## Interpretation Guardrails",
        "",
        "- These repeated splits reuse the same 1000 Mistral MMLU items; they test split stability, not new-sample generalization.",
        "- Positive repeated-split stability supports a more cautious statement that the Mistral exception is not merely the seed=42 split.",
        "- It does not replace a true targeted sample expansion if reviewers require new-sample replication.",
        "- Do not convert repeated-split positive rates into independent-sample p-values; splits are correlated because they reuse the same items.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-splits", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()

    root = find_repo_root()
    records = load_records(root)
    output_dir = root / "results" / "revision_2026_06" / "mistral_exception_stability"
    docs_dir = root / "results" / "revision_2026_06" / "mistral_exception_stability" / "docs"
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seed_start, args.seed_start + args.n_splits))
    if 42 not in seeds:
        seeds.append(42)
        seeds = sorted(seeds)

    rows: list[dict] = []
    for seed in seeds:
        rows.append(run_one_split(records, seed, fixed_layers=False))
        rows.append(run_one_split(records, seed, fixed_layers=True))

    summary = summarize_rows(rows)
    seed42_rows = [row for row in rows if row["seed"] == 42]

    artifact = {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "script": str(Path(__file__).resolve()),
            "n_records": len(records),
            "accuracy": float(np.mean([1 if r["is_correct"] else 0 for r in records])),
            "n_splits_requested": args.n_splits,
            "seed_start": args.seed_start,
            "seeds": seeds,
            "source_data": (
                "PoT_Experiment_Entropy_Attention_Extraction_Experiment/experiments/"
                "31_MMLU_Domain_Extension/EXP_20260220_000610_mmlu_mistral/"
                "data/sample_results.json"
            ),
            "protocol": {
                "split": "70/30 stratified",
                "classifier": "StandardScaler(cal only) + LogisticRegression(max_iter=1000, lbfgs)",
                "dynamic_layer_selection": "best layer+sign on calibration set per split",
                "fixed_layer_selection": "seed=42 exp42 layers/signs",
            },
            "expected_exp42": EXPECTED_EXP42,
        },
        "summary": summary,
        "seed42_rows": seed42_rows,
        "rows": rows,
    }

    json_path = output_dir / "mistral_exception_repeated_split_results.json"
    csv_path = output_dir / "mistral_exception_repeated_split_rows.csv"
    md_path = docs_dir / "mistral_exception_repeated_split_summary_v1.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
    write_csv(csv_path, rows)
    md_path.write_text(make_markdown(summary, seed42_rows, len(seeds)), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print()
    print(make_markdown(summary, seed42_rows, len(seeds)))


if __name__ == "__main__":
    main()
