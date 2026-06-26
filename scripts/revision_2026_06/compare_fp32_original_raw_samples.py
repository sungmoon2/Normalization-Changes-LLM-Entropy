#!/usr/bin/env python3
"""Compare original Phase 3 raw samples with EXP-03 FP32 raw samples.

This script is read-only with respect to experiment results. It writes only
comparison artifacts under the requested output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]

ORIGINAL_ROOT = (
    REPO_ROOT
    / "PoT_Experiment_Entropy_Attention_Extraction_Experiment"
    / "experiments"
    / "43_Phase3_Unified_Scale_Intervention"
)
FP32_ROOT = REPO_ROOT / "results" / "revision_2026_06" / "fp32_precision_control" / "full"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "revision_2026_06" / "fp32_precision_control" / "raw_sample_comparison"


RAW_FIELDS = [
    ("original.h_pre", ("original", None, "h_pre")),
    ("original.h_post", ("original", None, "h_post")),
    ("original.h_norm", ("original", None, "h_norm")),
    ("unit_norm.h_pre", ("unit_norm", None, "h_pre")),
    ("unit_norm.h_post", ("unit_norm", None, "h_post")),
    ("alpha.h_pre", ("alpha_sweep", "alpha", "h_pre")),
    ("alpha.h_post", ("alpha_sweep", "alpha", "h_post")),
]

SUMMARY_FIELDS = [
    "h_pre_unit_mean_all_layers",
    "h_post_worst_layer",
    "h_post_worst_variation",
]

UNIT_SUMMARY_FIELDS = [
    "h_pre_orig_mean",
    "h_pre_unit_mean",
    "h_pre_change",
    "h_post_orig_mean",
    "h_post_unit_mean",
    "h_post_change",
    "h_norm_mean",
    "h_norm_std",
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sorted_int_keys(mapping: Dict[str, Any]) -> List[str]:
    return [str(item) for item in sorted(int(key) for key in mapping.keys())]


def sorted_alpha_keys(mapping: Dict[str, Any]) -> List[str]:
    return sorted(mapping.keys(), key=lambda value: float(value))


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def diff_record(
    field: str,
    original_value: float,
    fp32_value: float,
    sample_idx: Optional[int] = None,
    layer: Optional[str] = None,
    alpha: Optional[str] = None,
    location_key: Optional[str] = None,
) -> Dict[str, Any]:
    signed = float(fp32_value) - float(original_value)
    return {
        "field": field,
        "max_abs_diff": abs(signed),
        "signed_diff_fp32_minus_original": signed,
        "original_value": float(original_value),
        "fp32_value": float(fp32_value),
        "sample_idx": sample_idx,
        "layer": layer,
        "alpha": alpha,
        "location_key": location_key,
    }


def update_max(
    current: Optional[Dict[str, Any]],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    if current is None or candidate["max_abs_diff"] > current["max_abs_diff"]:
        return candidate
    return current


def compare_layer_summary_dict(
    field_prefix: str,
    original_map: Dict[str, Any],
    fp32_map: Dict[str, Any],
    fields: Iterable[str],
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    results: Dict[str, Optional[Dict[str, Any]]] = {
        f"{field_prefix}.{field}": None for field in fields
    }
    problems: List[str] = []
    original_layers = sorted_int_keys(original_map)
    fp32_layers = sorted_int_keys(fp32_map)
    if original_layers != fp32_layers:
        problems.append(f"{field_prefix}: layer keys differ")
    for layer in original_layers:
        if layer not in fp32_map:
            continue
        for field in fields:
            if field not in original_map[layer] or field not in fp32_map[layer]:
                problems.append(f"{field_prefix}.{field}: missing at layer {layer}")
                continue
            left = original_map[layer][field]
            right = fp32_map[layer][field]
            if not is_finite_number(left) or not is_finite_number(right):
                problems.append(f"{field_prefix}.{field}: non-finite at layer {layer}")
                continue
            key = f"{field_prefix}.{field}"
            results[key] = update_max(
                results[key],
                diff_record(key, left, right, layer=layer),
            )
    return {key: value for key, value in results.items() if value is not None}, problems


def compare_alpha_summary(
    original_map: Dict[str, Any],
    fp32_map: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    results: Dict[str, Optional[Dict[str, Any]]] = {
        "alpha_summary.h_pre": None,
        "alpha_summary.h_post": None,
    }
    problems: List[str] = []
    original_alphas = sorted_alpha_keys(original_map)
    fp32_alphas = sorted_alpha_keys(fp32_map)
    if original_alphas != fp32_alphas:
        problems.append("alpha_summary: alpha keys differ")
    for alpha in original_alphas:
        if alpha not in fp32_map:
            continue
        original_keys = sorted(original_map[alpha].keys())
        fp32_keys = sorted(fp32_map[alpha].keys())
        if original_keys != fp32_keys:
            problems.append(f"alpha_summary: metric keys differ at alpha {alpha}")
        for metric_key in original_keys:
            if metric_key not in fp32_map[alpha]:
                continue
            left = original_map[alpha][metric_key]
            right = fp32_map[alpha][metric_key]
            if not is_finite_number(left) or not is_finite_number(right):
                problems.append(
                    f"alpha_summary: non-finite at alpha {alpha}, key {metric_key}"
                )
                continue
            if metric_key.endswith("_h_pre"):
                field = "alpha_summary.h_pre"
            elif metric_key.endswith("_h_post"):
                field = "alpha_summary.h_post"
            else:
                field = "alpha_summary.other"
                results.setdefault(field, None)
            results[field] = update_max(
                results[field],
                diff_record(
                    field,
                    left,
                    right,
                    alpha=alpha,
                    location_key=metric_key,
                ),
            )
    return {key: value for key, value in results.items() if value is not None}, problems


def compare_raw_entries(
    original_raw: List[Dict[str, Any]],
    fp32_raw: List[Dict[str, Any]],
    layers: List[str],
    alphas: List[str],
) -> Tuple[Dict[str, Dict[str, Any]], List[str], int]:
    results: Dict[str, Optional[Dict[str, Any]]] = {
        field_name: None for field_name, _ in RAW_FIELDS
    }
    problems: List[str] = []
    non_finite_count = 0

    for sample_pos, (original_entry, fp32_entry) in enumerate(zip(original_raw, fp32_raw)):
        original_idx = original_entry.get("idx")
        fp32_idx = fp32_entry.get("idx")
        if original_idx != fp32_idx:
            problems.append(
                f"idx mismatch at sample_pos {sample_pos}: {original_idx} vs {fp32_idx}"
            )
            continue
        for field_name, (section, mode, metric) in RAW_FIELDS:
            if mode == "alpha":
                for alpha in alphas:
                    for layer in layers:
                        try:
                            left = original_entry[section][alpha][layer][metric]
                            right = fp32_entry[section][alpha][layer][metric]
                        except KeyError as exc:
                            problems.append(
                                f"{field_name}: missing {exc} at sample {original_idx}, "
                                f"alpha {alpha}, layer {layer}"
                            )
                            continue
                        if not is_finite_number(left) or not is_finite_number(right):
                            non_finite_count += 1
                            continue
                        results[field_name] = update_max(
                            results[field_name],
                            diff_record(
                                field_name,
                                left,
                                right,
                                sample_idx=int(original_idx),
                                layer=layer,
                                alpha=alpha,
                            ),
                        )
            else:
                for layer in layers:
                    try:
                        left = original_entry[section][layer][metric]
                        right = fp32_entry[section][layer][metric]
                    except KeyError as exc:
                        problems.append(
                            f"{field_name}: missing {exc} at sample {original_idx}, "
                            f"layer {layer}"
                        )
                        continue
                    if not is_finite_number(left) or not is_finite_number(right):
                        non_finite_count += 1
                        continue
                    results[field_name] = update_max(
                        results[field_name],
                        diff_record(
                            field_name,
                            left,
                            right,
                            sample_idx=int(original_idx),
                            layer=layer,
                        ),
                    )
    return {key: value for key, value in results.items() if value is not None}, problems, non_finite_count


def compare_model(model: str) -> Dict[str, Any]:
    original_dir = ORIGINAL_ROOT / model
    fp32_dir = FP32_ROOT / model
    original_raw_path = original_dir / "intervention_raw_data.json"
    fp32_raw_path = fp32_dir / "intervention_raw_data.json"
    original_analysis_path = original_dir / "intervention_analysis.json"
    fp32_analysis_path = fp32_dir / "intervention_analysis.json"

    for path in [
        original_raw_path,
        fp32_raw_path,
        original_analysis_path,
        fp32_analysis_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    original_raw = read_json(original_raw_path)
    fp32_raw = read_json(fp32_raw_path)
    original_analysis = read_json(original_analysis_path)
    fp32_analysis = read_json(fp32_analysis_path)

    problems: List[str] = []
    if not isinstance(original_raw, list) or not isinstance(fp32_raw, list):
        raise TypeError(f"{model}: raw data must be JSON lists")

    raw_entry_count = {
        "original": len(original_raw),
        "fp32": len(fp32_raw),
    }
    if len(original_raw) != len(fp32_raw):
        problems.append("raw entry counts differ")

    first_idx = {
        "original": original_raw[0].get("idx") if original_raw else None,
        "fp32": fp32_raw[0].get("idx") if fp32_raw else None,
    }
    last_idx = {
        "original": original_raw[-1].get("idx") if original_raw else None,
        "fp32": fp32_raw[-1].get("idx") if fp32_raw else None,
    }
    original_indices = [entry.get("idx") for entry in original_raw]
    fp32_indices = [entry.get("idx") for entry in fp32_raw]
    indices_match = original_indices == fp32_indices
    if not indices_match:
        problems.append("raw idx sequences differ")

    layers_original = sorted_int_keys(original_raw[0]["original"])
    layers_fp32 = sorted_int_keys(fp32_raw[0]["original"])
    layers_match = layers_original == layers_fp32
    if not layers_match:
        problems.append("layer keys differ")

    alphas_original = sorted_alpha_keys(original_raw[0]["alpha_sweep"])
    alphas_fp32 = sorted_alpha_keys(fp32_raw[0]["alpha_sweep"])
    alphas_match = alphas_original == alphas_fp32
    if not alphas_match:
        problems.append("alpha keys differ")

    layers = layers_original
    alphas = alphas_original

    raw_max_diffs, raw_problems, non_finite_count = compare_raw_entries(
        original_raw,
        fp32_raw,
        layers,
        alphas,
    )
    problems.extend(raw_problems)

    hpost_map, hpost_problems = compare_layer_summary_dict(
        "h_post_max_variation",
        {k: {"value": v} for k, v in original_analysis["h_post_max_variation"].items()},
        {k: {"value": v} for k, v in fp32_analysis["h_post_max_variation"].items()},
        ["value"],
    )
    unit_map, unit_problems = compare_layer_summary_dict(
        "unit_summary",
        original_analysis["unit_norm_summary"],
        fp32_analysis["unit_norm_summary"],
        UNIT_SUMMARY_FIELDS,
    )
    alpha_map, alpha_problems = compare_alpha_summary(
        original_analysis["alpha_summary"],
        fp32_analysis["alpha_summary"],
    )
    problems.extend(hpost_problems)
    problems.extend(unit_problems)
    problems.extend(alpha_problems)

    summary_values = {}
    for field in SUMMARY_FIELDS:
        original_value = original_analysis.get(field)
        fp32_value = fp32_analysis.get(field)
        if not is_finite_number(original_value) or not is_finite_number(fp32_value):
            problems.append(f"summary field non-finite or missing: {field}")
            continue
        summary_values[field] = {
            "original": original_value,
            "fp32": fp32_value,
            "signed_diff_fp32_minus_original": fp32_value - original_value,
            "abs_diff": abs(fp32_value - original_value),
        }

    sample_results_dir = fp32_dir / "sample_results"
    sample_result_files = (
        sorted(sample_results_dir.glob("sample_*.json"))
        if sample_results_dir.exists()
        else []
    )

    return {
        "model": model,
        "source_files": {
            "original_raw": str(original_raw_path.relative_to(REPO_ROOT)),
            "fp32_raw": str(fp32_raw_path.relative_to(REPO_ROOT)),
            "original_analysis": str(original_analysis_path.relative_to(REPO_ROOT)),
            "fp32_analysis": str(fp32_analysis_path.relative_to(REPO_ROOT)),
        },
        "sha256": {
            "original_raw": sha256_file(original_raw_path),
            "fp32_raw": sha256_file(fp32_raw_path),
            "original_analysis": sha256_file(original_analysis_path),
            "fp32_analysis": sha256_file(fp32_analysis_path),
        },
        "matched_structure": {
            "raw_entry_count": raw_entry_count,
            "first_idx": first_idx,
            "last_idx": last_idx,
            "indices_match": indices_match,
            "layers_original": len(layers_original),
            "layers_fp32": len(layers_fp32),
            "layers_match": layers_match,
            "alphas_original": alphas_original,
            "alphas_fp32": alphas_fp32,
            "alphas_match": alphas_match,
            "fp32_sample_result_file_count": len(sample_result_files),
        },
        "summary_level_comparison": summary_values,
        "layer_summary_max_abs_differences": {
            **hpost_map,
            **unit_map,
            **alpha_map,
        },
        "raw_sample_level_max_abs_differences": raw_max_diffs,
        "non_finite_raw_value_pair_count": non_finite_count,
        "problem_count": len(problems),
        "problems": problems,
    }


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "field",
        "max_abs_diff",
        "signed_diff_fp32_minus_original",
        "original_value",
        "fp32_value",
        "sample_idx",
        "layer",
        "alpha",
        "location_key",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen", "llama"],
        choices=["qwen", "llama", "mistral"],
        help="Model keys to compare.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for comparison artifacts.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_results = {}
    for model in args.models:
        result = compare_model(model)
        model_results[model] = result
        write_json(output_dir / f"{model}_raw_comparison.json", result)
        write_csv(
            output_dir / f"{model}_raw_sample_max_abs_differences.csv",
            result["raw_sample_level_max_abs_differences"].values(),
        )
        write_csv(
            output_dir / f"{model}_layer_summary_max_abs_differences.csv",
            result["layer_summary_max_abs_differences"].values(),
        )

    summary = {
        "script": str(SCRIPT_PATH.relative_to(REPO_ROOT)),
        "repo_root": str(REPO_ROOT),
        "models": args.models,
        "output_dir": str(output_dir.relative_to(REPO_ROOT)),
        "results": {
            model: {
                "problem_count": result["problem_count"],
                "matched_structure": result["matched_structure"],
                "summary_level_comparison": result["summary_level_comparison"],
                "raw_sample_level_max_abs_differences": result[
                    "raw_sample_level_max_abs_differences"
                ],
                "non_finite_raw_value_pair_count": result[
                    "non_finite_raw_value_pair_count"
                ],
            }
            for model, result in model_results.items()
        },
    }
    write_json(output_dir / "raw_comparison_summary.json", summary)

    for model, result in model_results.items():
        print(
            f"{model}: problems={result['problem_count']}, "
            f"raw_entries={result['matched_structure']['raw_entry_count']}, "
            f"layers={result['matched_structure']['layers_original']}, "
            f"sample_results={result['matched_structure']['fp32_sample_result_file_count']}"
        )
    print(f"wrote: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
