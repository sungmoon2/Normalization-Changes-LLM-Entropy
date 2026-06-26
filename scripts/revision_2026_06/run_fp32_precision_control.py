# -*- coding: utf-8 -*-
"""
FP16/FP32 precision control for IEEE Access resubmission.

Source-of-truth script:
    scripts/run_phase3_unified.py

Critical constraint:
    The manuscript-grade run must keep the original Phase 3 unified design fixed.
    The only intended experimental variable is torch_dtype in from_pretrained:
        fp16 -> torch.float16
        fp32 -> torch.float32

Smoke mode loads the exact MMLU-500 manifest and then runs only the first K
samples. Smoke outputs are feasibility/provenance checks, not paper evidence.
"""

import argparse
import gc
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch


SEED = 42
NUM_SAMPLES = 500
ALPHAS = [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _paths import POT_DIR  # noqa: E402


OUT_ROOT = PROJECT_ROOT / "results" / "revision_2026_06" / "fp32_precision_control" / "full"
SOURCE_RUN_DIR = POT_DIR / "experiments" / "43_Phase3_Unified_Scale_Intervention"
SOURCE_SCRIPT = PROJECT_ROOT / "scripts" / "run_phase3_unified.py"
CONTROL_SCRIPT = Path(__file__).resolve()

MODEL_CONFIGS = {
    "qwen": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "short": "qwen",
    },
    "llama": {
        "name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "short": "llama",
    },
    "mistral": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "short": "mistral",
    },
}


def set_seed(seed=SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sha256_file(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_dumps_strict(obj, indent=2):
    return json.dumps(obj, indent=indent, ensure_ascii=False, allow_nan=False)


def atomic_write_json(path, obj, indent=2):
    """Write JSON through a temporary file, then atomically replace the target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json_dumps_strict(obj, indent=indent), encoding="utf-8")
    os.replace(tmp_path, path)


def sha256_json_obj(obj):
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def package_version(package_name):
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def nvidia_driver_version():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[0] if lines else None


def environment_metadata():
    cuda_device = None
    if torch.cuda.is_available():
        try:
            cuda_device = torch.cuda.get_device_name(0)
        except Exception:
            cuda_device = None

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device_0": cuda_device,
        "nvidia_driver_version": nvidia_driver_version(),
        "transformers_version": package_version("transformers"),
        "datasets_version": package_version("datasets"),
        "accelerate_version": package_version("accelerate"),
    }


def assert_finite_scalar(name, value, sample_idx, layer_idx=None, alpha=None):
    try:
        value_f = float(value)
    except Exception as exc:
        raise ValueError(
            f"non-numeric value: {name}={value!r}, sample={sample_idx}, layer={layer_idx}, alpha={alpha}"
        ) from exc
    if not np.isfinite(value_f):
        raise ValueError(
            f"non-finite value: {name}={value!r}, sample={sample_idx}, layer={layer_idx}, alpha={alpha}"
        )


def sample_result_path(sample_dir, idx):
    return sample_dir / f"sample_{idx:04d}.json"


def load_sample_results(sample_dir):
    sample_dir = Path(sample_dir)
    results = {}
    unreadable = []
    if not sample_dir.exists():
        return results, unreadable

    for path in sorted(sample_dir.glob("sample_*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            idx = int(item["idx"])
        except Exception as exc:
            unreadable.append({"path": str(path), "error": repr(exc)})
            continue
        results[idx] = item
    return results, unreadable


def validate_sample_result(sample_data, num_layers):
    sample_idx = sample_data.get("idx")
    if not isinstance(sample_idx, int):
        raise ValueError(f"sample result has invalid idx: {sample_idx!r}")

    required_top = {"idx", "original", "unit_norm", "alpha_sweep"}
    missing_top = sorted(required_top - set(sample_data))
    if missing_top:
        raise ValueError(f"sample {sample_idx} missing top-level fields: {missing_top}")

    expected_layers = {str(i) for i in range(num_layers)}
    for block_name in ["original", "unit_norm"]:
        block = sample_data[block_name]
        if set(block) != expected_layers:
            missing = sorted(expected_layers - set(block))
            extra = sorted(set(block) - expected_layers)
            raise ValueError(f"sample {sample_idx} {block_name} layer mismatch: missing={missing}, extra={extra}")

    expected_alphas = {str(a) for a in ALPHAS}
    alpha_sweep = sample_data["alpha_sweep"]
    if set(alpha_sweep) != expected_alphas:
        missing = sorted(expected_alphas - set(alpha_sweep))
        extra = sorted(set(alpha_sweep) - expected_alphas)
        raise ValueError(f"sample {sample_idx} alpha mismatch: missing={missing}, extra={extra}")

    for layer_idx in range(num_layers):
        lk = str(layer_idx)
        original = sample_data["original"][lk]
        for field in ["h_pre", "h_post", "h_norm"]:
            if field not in original:
                raise ValueError(f"sample {sample_idx} original L{layer_idx} missing {field}")
            assert_finite_scalar(f"original.{field}", original[field], sample_idx, layer_idx=layer_idx)

        unit_norm = sample_data["unit_norm"][lk]
        for field in ["h_pre", "h_post"]:
            if field not in unit_norm:
                raise ValueError(f"sample {sample_idx} unit_norm L{layer_idx} missing {field}")
            assert_finite_scalar(f"unit_norm.{field}", unit_norm[field], sample_idx, layer_idx=layer_idx)

        for alpha in ALPHAS:
            ak = str(alpha)
            alpha_block = alpha_sweep[ak]
            if set(alpha_block) != expected_layers:
                missing = sorted(expected_layers - set(alpha_block))
                extra = sorted(set(alpha_block) - expected_layers)
                raise ValueError(
                    f"sample {sample_idx} alpha {alpha} layer mismatch: missing={missing}, extra={extra}"
                )
            fields = alpha_block[lk]
            for field in ["h_pre", "h_post"]:
                if field not in fields:
                    raise ValueError(f"sample {sample_idx} alpha {alpha} L{layer_idx} missing {field}")
                assert_finite_scalar(f"alpha_sweep.{field}", fields[field], sample_idx, layer_idx=layer_idx, alpha=alpha)


def write_resume_state(path, model_key, precision, total_samples, results_by_idx, status, unreadable=None):
    completed = sorted(idx for idx, item in results_by_idx.items() if "error" not in item)
    errored = sorted(idx for idx, item in results_by_idx.items() if "error" in item)
    pending = [idx for idx in range(total_samples) if idx not in results_by_idx]
    state = {
        "model_key": model_key,
        "precision": precision,
        "status": status,
        "total_samples": total_samples,
        "completed_count": len(completed),
        "error_count": len(errored),
        "pending_count": len(pending),
        "completed_indices": completed,
        "error_indices": errored,
        "next_pending_index": pending[0] if pending else None,
        "unreadable_sample_files": unreadable or [],
        "updated_at": datetime.now().isoformat(),
    }
    atomic_write_json(path, state)


def build_model_run_signature(model_key, precision, samples, manifest, run_context):
    return {
        "run_id": run_context["run_id"],
        "mode": run_context["mode"],
        "model_key": model_key,
        "precision": precision,
        "num_samples_run": len(samples),
        "num_samples_manifest": run_context["num_samples_manifest"],
        "seed": SEED,
        "alphas": ALPHAS,
        "source_script_sha256": run_context["source_script_sha256"],
        "control_script_sha256": run_context["control_script_sha256"],
        "locked_manifest_sha256": run_context["locked_manifest_sha256"],
        "model_manifest_sha256": sha256_json_obj(manifest[: len(samples)]),
    }


def ensure_resume_compatible(model_output_dir, sample_dir, signature, resume_enabled):
    signature_path = Path(model_output_dir) / "model_run_signature.json"
    sample_files = list(Path(sample_dir).glob("sample_*.json")) if Path(sample_dir).exists() else []

    if not resume_enabled:
        atomic_write_json(signature_path, signature)
        return

    if not signature_path.exists():
        if sample_files:
            raise ValueError(
                "Cannot resume: existing sample files were found but model_run_signature.json is missing. "
                f"Use a new run_id or intentionally restart with --no_resume. Directory: {sample_dir}"
            )
        atomic_write_json(signature_path, signature)
        return

    previous = json.loads(signature_path.read_text(encoding="utf-8"))
    checked_fields = [
        "run_id",
        "mode",
        "model_key",
        "precision",
        "num_samples_run",
        "num_samples_manifest",
        "seed",
        "alphas",
        "source_script_sha256",
        "control_script_sha256",
        "locked_manifest_sha256",
        "model_manifest_sha256",
    ]
    mismatches = []
    for field in checked_fields:
        if previous.get(field) != signature.get(field):
            mismatches.append({"field": field, "previous": previous.get(field), "current": signature.get(field)})

    if mismatches:
        raise ValueError(
            "Cannot resume: model_run_signature.json is incompatible with the current invocation. "
            f"Mismatches: {mismatches}"
        )

    atomic_write_json(signature_path, signature)


def collect_existing_analyses(output_dir, precision):
    precision_dir = Path(output_dir) / precision
    analyses = {}
    if not precision_dir.exists():
        return analyses

    for path in sorted(precision_dir.glob("*/intervention_analysis.json")):
        try:
            analysis = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        key = analysis.get("model_short") or path.parent.name
        analyses[key] = analysis
    return analyses


def append_invocation_log(output_dir, run_metadata):
    log_path = Path(output_dir) / "run_invocations.json"
    invocations = []
    if log_path.exists():
        try:
            loaded = json.loads(log_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                invocations = loaded
        except Exception:
            invocations = []

    entry = dict(run_metadata)
    entry["invocation_index"] = len(invocations) + 1
    invocations.append(entry)
    atomic_write_json(log_path, invocations)


def compute_entropy_from_logits(logits_f32):
    """Compute normalized Shannon entropy from logits (FP32), matching source."""
    probs = torch.softmax(logits_f32, dim=-1).clamp(min=1e-10)
    ent = (-probs * torch.log(probs)).sum(dim=-1)
    max_ent = float(np.log(logits_f32.shape[-1]))
    return float(ent.cpu().item()) / max_ent


def load_mmlu_samples(num_samples):
    """Load MMLU samples exactly as scripts/run_phase3_unified.py."""
    from datasets import load_dataset

    dataset = load_dataset("cais/mmlu", "all", split="test")
    np.random.seed(SEED)
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    indices.sort()

    samples = []
    manifest = []
    for order, idx in enumerate(indices):
        item = dataset[int(idx)]
        labels = ["A", "B", "C", "D"]
        answer_idx = item["answer"]
        choice_str = "\n".join(
            f"{label}. {text}" for label, text in zip(labels[: len(item["choices"])], item["choices"])
        )
        sample = {
            "question": item["question"],
            "choices_str": choice_str,
            "answer_key": labels[answer_idx] if answer_idx < 4 else "A",
            "subject": item.get("subject", "unknown"),
        }
        samples.append(sample)
        manifest.append(
            {
                "pre_shuffle_order": order,
                "dataset_index": int(idx),
                "subject": sample["subject"],
                "answer_key": sample["answer_key"],
                "question_sha256": hashlib.sha256(sample["question"].encode("utf-8")).hexdigest(),
            }
        )

    paired = list(zip(samples, manifest))
    random.seed(SEED)
    random.shuffle(paired)
    samples = [p[0] for p in paired]
    manifest = [dict(p[1], post_shuffle_order=i) for i, p in enumerate(paired)]
    return samples, manifest


def make_prompt(tokenizer, sample, model_key):
    """Create chat prompt, matching scripts/run_phase3_unified.py."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers multiple choice questions step by step.",
        },
        {
            "role": "user",
            "content": (
                f"Answer the following multiple choice question step by step.\n\n"
                f"Question: {sample['question']}\n\n{sample['choices_str']}\n\n"
                f"Think through each option carefully, then end with "
                f"\"The answer is [LETTER]\" where LETTER is A, B, C, or D."
            ),
        },
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def resolve_dtype(precision):
    if precision == "fp16":
        return torch.float16
    if precision == "fp32":
        return torch.float32
    raise ValueError(f"unsupported precision: {precision}")


def dtype_counts(model):
    counts = {}
    for p in model.parameters():
        key = str(p.dtype)
        counts[key] = counts.get(key, 0) + p.numel()
    return counts


def compare_values(new_obj, src_obj, path, stats):
    if isinstance(new_obj, dict) and isinstance(src_obj, dict):
        if set(new_obj.keys()) != set(src_obj.keys()):
            stats["schema_mismatches"].append(
                {"path": path, "new_keys": sorted(new_obj.keys()), "source_keys": sorted(src_obj.keys())}
            )
            return
        for key in new_obj:
            compare_values(new_obj[key], src_obj[key], f"{path}.{key}", stats)
    elif isinstance(new_obj, list) and isinstance(src_obj, list):
        if len(new_obj) != len(src_obj):
            stats["schema_mismatches"].append({"path": path, "new_len": len(new_obj), "source_len": len(src_obj)})
            return
        for i, (n, s) in enumerate(zip(new_obj, src_obj)):
            compare_values(n, s, f"{path}[{i}]", stats)
    elif isinstance(new_obj, (int, float)) and isinstance(src_obj, (int, float)):
        diff = abs(float(new_obj) - float(src_obj))
        if diff > stats["max_abs_diff"]:
            stats["max_abs_diff"] = diff
            stats["max_abs_diff_path"] = path
    else:
        if new_obj != src_obj:
            stats["value_mismatches"].append({"path": path, "new": new_obj, "source": src_obj})


def compare_fp16_smoke_to_source(model_key, raw_data):
    source_raw = SOURCE_RUN_DIR / model_key / "intervention_raw_data.json"
    if not source_raw.exists():
        return {
            "source_found": False,
            "source_path": str(source_raw),
            "status": "source_missing",
        }

    source_data = json.loads(source_raw.read_text(encoding="utf-8"))
    source_subset = source_data[: len(raw_data)]
    stats = {
        "source_found": True,
        "source_path": str(source_raw),
        "n_compared": len(raw_data),
        "max_abs_diff": 0.0,
        "max_abs_diff_path": None,
        "schema_mismatches": [],
        "value_mismatches": [],
    }
    compare_values(raw_data, source_subset, "raw", stats)
    stats["status"] = (
        "pass"
        if not stats["schema_mismatches"] and not stats["value_mismatches"] and stats["max_abs_diff"] <= 1e-6
        else "check_required"
    )
    return stats


def run_intervention(model_key, precision, samples, manifest, output_dir, compare_source, run_context, resume_enabled=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = MODEL_CONFIGS[model_key]
    model_name = config["name"]
    model_short = config["short"]
    torch_dtype = resolve_dtype(precision)
    model_output_dir = output_dir / precision / model_short
    model_output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = model_output_dir / "sample_results"
    resume_state_path = model_output_dir / "resume_state.json"
    dtype_trace_path = model_output_dir / "dtype_trace.json"
    analysis_path = model_output_dir / "intervention_analysis.json"
    model_run_signature = build_model_run_signature(model_key, precision, samples, manifest, run_context)
    ensure_resume_compatible(model_output_dir, sample_dir, model_run_signature, resume_enabled)

    existing_results = {}
    unreadable_samples = []
    if resume_enabled:
        existing_results, unreadable_samples = load_sample_results(sample_dir)
        if unreadable_samples:
            print(f"  WARNING: {len(unreadable_samples)} unreadable sample file(s) found; they will be ignored.")

    all_results_by_idx = {}
    for idx, item in existing_results.items():
        if 0 <= idx < len(samples) and "error" not in item:
            all_results_by_idx[idx] = item

    if resume_enabled and all_results_by_idx:
        print(f"  Resume: found {len(all_results_by_idx)}/{len(samples)} completed sample result(s).")

    atomic_write_json(model_output_dir / "sample_manifest_used.json", manifest[: len(samples)])
    write_resume_state(
        resume_state_path,
        model_key,
        precision,
        len(samples),
        all_results_by_idx,
        status="running",
        unreadable=unreadable_samples,
    )

    if resume_enabled and len(all_results_by_idx) == len(samples) and analysis_path.exists():
        print(f"  Resume: all samples already complete; loading existing analysis: {analysis_path}")
        write_resume_state(
            resume_state_path,
            model_key,
            precision,
            len(samples),
            all_results_by_idx,
            status="complete",
            unreadable=unreadable_samples,
        )
        return json.loads(analysis_path.read_text(encoding="utf-8"))

    print("=" * 70)
    print(f"FP precision control: {model_name}")
    print(f"Precision: {precision} ({torch_dtype})")
    print(f"Samples: {len(samples)} from locked MMLU-500 manifest")
    print(f"Position: Step 0 (prompt-last)")
    print(f"Alphas: {ALPHAS}")
    print("=" * 70)

    print(f"\nLoading model: {model_name}")
    load_started = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    load_seconds = time.time() - load_started
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    lm_head = model.lm_head
    norm = model.model.norm
    hf_device_map = getattr(model, "hf_device_map", None)
    model_commit_hash = getattr(model.config, "_commit_hash", None)
    tokenizer_commit_hash = getattr(tokenizer, "_commit_hash", None)

    print(f"  Layers: {num_layers}, Hidden: {hidden_size}")
    print(f"  lm_head dtype: {lm_head.weight.dtype}")
    print(f"  norm dtype: {norm.weight.dtype}")
    print(f"  load time: {timedelta(seconds=int(load_seconds))}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    dtype_trace = []
    if dtype_trace_path.exists():
        try:
            dtype_trace = json.loads(dtype_trace_path.read_text(encoding="utf-8"))
        except Exception:
            dtype_trace = []

    for i, sample in enumerate(samples):
        if resume_enabled and i in all_results_by_idx:
            print(f"  [{i + 1}/{len(samples)}] resume-skip")
            continue

        sample_dtype_trace = []
        try:
            prompt = make_prompt(tokenizer, sample, model_key)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            last_pos = -1
            sample_data = {
                "idx": i,
                "original": {},
                "unit_norm": {},
                "alpha_sweep": {str(a): {} for a in ALPHAS},
            }

            for layer_idx in range(num_layers):
                hidden = outputs.hidden_states[layer_idx + 1][:, last_pos:, :]
                h = hidden.float()
                h_norm_val = float(torch.norm(h).cpu().item())

                if i == 0 and layer_idx in {0, num_layers // 2, num_layers - 1}:
                    sample_dtype_trace.append(
                        {
                            "sample_idx": i,
                            "layer_idx": layer_idx,
                            "hidden_dtype": str(hidden.dtype),
                            "h_after_float_dtype": str(h.dtype),
                            "lm_head_weight_dtype": str(lm_head.weight.dtype),
                            "norm_weight_dtype": str(norm.weight.dtype),
                            "hidden_device": str(hidden.device),
                            "model_device": str(model.device),
                            "lm_head_device": str(lm_head.weight.device),
                            "norm_device": str(norm.weight.device),
                        }
                    )

                logits_u = lm_head(h.to(lm_head.weight.dtype)).float()
                h_normed = norm(h.to(norm.weight.dtype)).float()
                logits_n = lm_head(h_normed.to(lm_head.weight.dtype)).float()

                ent_pre = compute_entropy_from_logits(logits_u.squeeze(0))
                ent_post = compute_entropy_from_logits(logits_n.squeeze(0))

                sample_data["original"][str(layer_idx)] = {
                    "h_pre": ent_pre,
                    "h_post": ent_post,
                    "h_norm": h_norm_val,
                }

                if h_norm_val > 1e-10:
                    h_unit = h / h_norm_val
                else:
                    h_unit = h

                logits_u_unit = lm_head(h_unit.to(lm_head.weight.dtype)).float()
                h_unit_normed = norm(h_unit.to(norm.weight.dtype)).float()
                logits_n_unit = lm_head(h_unit_normed.to(lm_head.weight.dtype)).float()

                ent_pre_unit = compute_entropy_from_logits(logits_u_unit.squeeze(0))
                ent_post_unit = compute_entropy_from_logits(logits_n_unit.squeeze(0))

                sample_data["unit_norm"][str(layer_idx)] = {
                    "h_pre": ent_pre_unit,
                    "h_post": ent_post_unit,
                }

                for alpha in ALPHAS:
                    h_scaled = alpha * h

                    logits_u_a = lm_head(h_scaled.to(lm_head.weight.dtype)).float()
                    h_a_normed = norm(h_scaled.to(norm.weight.dtype)).float()
                    logits_n_a = lm_head(h_a_normed.to(lm_head.weight.dtype)).float()

                    ent_pre_a = compute_entropy_from_logits(logits_u_a.squeeze(0))
                    ent_post_a = compute_entropy_from_logits(logits_n_a.squeeze(0))

                    sample_data["alpha_sweep"][str(alpha)][str(layer_idx)] = {
                        "h_pre": ent_pre_a,
                        "h_post": ent_post_a,
                    }

            validate_sample_result(sample_data, num_layers)
            atomic_write_json(sample_result_path(sample_dir, i), sample_data, indent=1)
            all_results_by_idx[i] = sample_data
            if sample_dtype_trace:
                dtype_trace = sample_dtype_trace
                atomic_write_json(dtype_trace_path, dtype_trace)
            write_resume_state(
                resume_state_path,
                model_key,
                precision,
                len(samples),
                all_results_by_idx,
                status="running",
                unreadable=unreadable_samples,
            )
            print(f"  [{i + 1}/{len(samples)}] ok + saved")

            del outputs, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except KeyboardInterrupt:
            write_resume_state(
                resume_state_path,
                model_key,
                precision,
                len(samples),
                all_results_by_idx,
                status="interrupted",
                unreadable=unreadable_samples,
            )
            raise

        except Exception as exc:
            print(f"  [{i + 1}/{len(samples)}] ERROR: {exc}")
            sample_error = {"idx": i, "error": str(exc)}
            atomic_write_json(sample_result_path(sample_dir, i), sample_error, indent=1)
            all_results_by_idx[i] = sample_error
            write_resume_state(
                resume_state_path,
                model_key,
                precision,
                len(samples),
                all_results_by_idx,
                status="running",
                unreadable=unreadable_samples,
            )

    total_time = time.time() - start_time
    all_results = [all_results_by_idx.get(i, {"idx": i, "error": "missing_result_after_run"}) for i in range(len(samples))]
    valid_results = [r for r in all_results if "error" not in r]
    n_valid = len(valid_results)
    errors = len(all_results) - n_valid
    print(f"\nExtraction complete: {timedelta(seconds=int(total_time))}, errors: {errors}/{len(samples)}")

    raw_path = model_output_dir / "intervention_raw_data.json"
    atomic_write_json(raw_path, all_results, indent=1)

    manifest_path = model_output_dir / "sample_manifest_used.json"
    atomic_write_json(manifest_path, manifest[: len(samples)])

    key_layers = list(range(0, num_layers, 4))
    if (num_layers - 1) % 4 != 0:
        key_layers.append(num_layers - 1)

    unit_norm_summary = {}
    alpha_summary = {}
    h_post_max_var = {}
    pre_unit_mean_all = None
    worst_layer = None
    worst_var = None

    if n_valid:
        for layer_idx in range(num_layers):
            lk = str(layer_idx)
            pre_orig = [r["original"][lk]["h_pre"] for r in valid_results]
            pre_unit = [r["unit_norm"][lk]["h_pre"] for r in valid_results]
            post_orig = [r["original"][lk]["h_post"] for r in valid_results]
            post_unit = [r["unit_norm"][lk]["h_post"] for r in valid_results]
            h_norms = [r["original"][lk]["h_norm"] for r in valid_results]

            unit_norm_summary[lk] = {
                "h_pre_orig_mean": round(float(np.mean(pre_orig)), 6),
                "h_pre_unit_mean": round(float(np.mean(pre_unit)), 6),
                "h_pre_change": round(float(np.mean(pre_unit) - np.mean(pre_orig)), 6),
                "h_post_orig_mean": round(float(np.mean(post_orig)), 6),
                "h_post_unit_mean": round(float(np.mean(post_unit)), 6),
                "h_post_change": round(float(np.mean(post_unit) - np.mean(post_orig)), 6),
                "h_norm_mean": round(float(np.mean(h_norms)), 4),
                "h_norm_std": round(float(np.std(h_norms)), 4),
            }

        repr_layers = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]
        repr_layers = sorted(set(repr_layers))
        for alpha in ALPHAS:
            ak = str(alpha)
            row = {}
            for layer_idx in repr_layers:
                lk = str(layer_idx)
                pre_vals = [
                    r["alpha_sweep"][ak][lk]["h_pre"]
                    for r in valid_results
                    if ak in r["alpha_sweep"] and lk in r["alpha_sweep"][ak]
                ]
                post_vals = [
                    r["alpha_sweep"][ak][lk]["h_post"]
                    for r in valid_results
                    if ak in r["alpha_sweep"] and lk in r["alpha_sweep"][ak]
                ]
                row[f"L{layer_idx}_h_pre"] = round(float(np.mean(pre_vals)), 6) if pre_vals else None
                row[f"L{layer_idx}_h_post"] = round(float(np.mean(post_vals)), 6) if post_vals else None
            alpha_summary[ak] = row

        for layer_idx in range(num_layers):
            lk = str(layer_idx)
            post_by_alpha = []
            for alpha in ALPHAS:
                ak = str(alpha)
                vals = [
                    r["alpha_sweep"][ak][lk]["h_post"]
                    for r in valid_results
                    if ak in r["alpha_sweep"] and lk in r["alpha_sweep"][ak]
                ]
                if vals:
                    post_by_alpha.append(np.mean(vals))
            if len(post_by_alpha) >= 2:
                h_post_max_var[lk] = round(float(max(post_by_alpha) - min(post_by_alpha)), 8)

        worst_layer = int(max(h_post_max_var, key=h_post_max_var.get))
        worst_var = h_post_max_var[str(worst_layer)]
        pre_unit_mean_all = round(
            float(np.mean([float(unit_norm_summary[str(l)]["h_pre_unit_mean"]) for l in range(num_layers)])),
            6,
        )

    compare_summary = None
    if compare_source and precision == "fp16":
        compare_summary = compare_fp16_smoke_to_source(model_short, all_results)

    cuda_summary = None
    if torch.cuda.is_available():
        cuda_summary = {
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }

    analysis = {
        "model": model_name,
        "model_short": model_short,
        "precision": precision,
        "torch_dtype_requested": str(torch_dtype),
        "n_samples": n_valid,
        "n_errors": errors,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "position": "step0_prompt_last",
        "dataset": "MMLU",
        "dataset_samples": NUM_SAMPLES,
        "alphas": ALPHAS,
        "unit_norm_summary": unit_norm_summary,
        "alpha_summary": alpha_summary,
        "h_post_max_variation": h_post_max_var,
        "h_post_worst_layer": worst_layer,
        "h_post_worst_variation": worst_var,
        "h_pre_unit_mean_all_layers": pre_unit_mean_all,
        "load_time_seconds": round(load_seconds, 1),
        "total_time_seconds": round(total_time, 1),
        "dtype_trace": dtype_trace,
        "parameter_dtype_counts": dtype_counts(model),
        "hf_device_map": hf_device_map,
        "cuda_summary": cuda_summary,
        "source_script": str(SOURCE_SCRIPT),
        "source_script_sha256": sha256_file(SOURCE_SCRIPT) if SOURCE_SCRIPT.exists() else None,
        "control_script": str(CONTROL_SCRIPT),
        "control_script_sha256": sha256_file(CONTROL_SCRIPT),
        "source_run_dir": str(SOURCE_RUN_DIR),
        "model_run_signature": model_run_signature,
        "environment": run_context["environment"],
        "model_commit_hash": model_commit_hash,
        "tokenizer_commit_hash": tokenizer_commit_hash,
        "fp16_source_compare": compare_summary,
        "timestamp": datetime.now().isoformat(),
    }

    atomic_write_json(analysis_path, analysis)
    write_resume_state(
        resume_state_path,
        model_key,
        precision,
        len(samples),
        all_results_by_idx,
        status="complete_with_errors" if errors else "complete",
        unreadable=unreadable_samples,
    )

    del model
    torch.cuda.empty_cache()
    gc.collect()

    print(f"Saved: {analysis_path}")
    return analysis


def main():
    parser = argparse.ArgumentParser(description="FP16/FP32 precision control for Phase 3 unified scale intervention")
    parser.add_argument("--model", choices=["qwen", "llama", "mistral", "all"], default="qwen")
    parser.add_argument("--precision", choices=["fp16", "fp32"], required=True)
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES)
    parser.add_argument(
        "--smoke_samples",
        type=int,
        default=1,
        help="Use first K samples from the locked manifest. Use 0 for full manuscript-grade run.",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--compare_source_fp16", action="store_true")
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Ignore existing per-sample result files in the same run directory.",
    )
    args = parser.parse_args()

    if args.num_samples != NUM_SAMPLES:
        raise ValueError("Manuscript-grade protocol is locked to num_samples=500.")

    set_seed()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "full" if args.smoke_samples == 0 else f"smoke{args.smoke_samples}"
    output_dir = OUT_ROOT / f"{run_id}_{mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, manifest = load_mmlu_samples(args.num_samples)
    if args.smoke_samples > 0:
        samples_to_run = samples[: args.smoke_samples]
        manifest_to_run = manifest[: args.smoke_samples]
    else:
        samples_to_run = samples
        manifest_to_run = manifest

    source_script_sha256 = sha256_file(SOURCE_SCRIPT) if SOURCE_SCRIPT.exists() else None
    control_script_sha256 = sha256_file(CONTROL_SCRIPT)
    locked_manifest_sha256 = sha256_json_obj(manifest)
    run_manifest_sha256 = sha256_json_obj(manifest_to_run)
    env_metadata = environment_metadata()

    run_metadata = {
        "run_id": run_id,
        "mode": mode,
        "model_arg": args.model,
        "precision": args.precision,
        "num_samples_manifest": args.num_samples,
        "num_samples_run": len(samples_to_run),
        "seed": SEED,
        "alphas": ALPHAS,
        "project_root": str(PROJECT_ROOT),
        "pot_dir": str(POT_DIR),
        "source_script": str(SOURCE_SCRIPT),
        "source_script_sha256": source_script_sha256,
        "control_script": str(CONTROL_SCRIPT),
        "control_script_sha256": control_script_sha256,
        "source_run_dir": str(SOURCE_RUN_DIR),
        "locked_manifest_sha256": locked_manifest_sha256,
        "run_manifest_sha256": run_manifest_sha256,
        "environment": env_metadata,
        "resume_enabled": not args.no_resume,
        "timestamp": datetime.now().isoformat(),
    }
    atomic_write_json(output_dir / "run_metadata.json", run_metadata)
    append_invocation_log(output_dir, run_metadata)
    atomic_write_json(output_dir / "sample_manifest_locked_mmlu500.json", manifest)

    models_to_run = ["qwen", "llama", "mistral"] if args.model == "all" else [args.model]
    run_context = {
        "run_id": run_id,
        "mode": mode,
        "num_samples_manifest": args.num_samples,
        "source_script_sha256": source_script_sha256,
        "control_script_sha256": control_script_sha256,
        "locked_manifest_sha256": locked_manifest_sha256,
        "environment": env_metadata,
    }
    analyses = {}
    for model_key in models_to_run:
        set_seed()
        analyses[model_key] = run_intervention(
            model_key=model_key,
            precision=args.precision,
            samples=samples_to_run,
            manifest=manifest_to_run,
            output_dir=output_dir,
            compare_source=args.compare_source_fp16,
            run_context=run_context,
            resume_enabled=not args.no_resume,
        )

    summary_analyses = collect_existing_analyses(output_dir, args.precision)
    summary_analyses.update(analyses)

    summary = {
        "run_id": run_id,
        "mode": mode,
        "precision": args.precision,
        "source_script_sha256": source_script_sha256,
        "control_script": str(CONTROL_SCRIPT),
        "control_script_sha256": control_script_sha256,
        "locked_manifest_sha256": locked_manifest_sha256,
        "run_manifest_sha256": run_manifest_sha256,
        "environment": env_metadata,
        "models": {
            key: {
                "n_samples": value["n_samples"],
                "n_errors": value["n_errors"],
                "h_pre_unit_mean": value["h_pre_unit_mean_all_layers"],
                "h_post_worst_var": value["h_post_worst_variation"],
                "h_post_worst_layer": value["h_post_worst_layer"],
                "total_time_seconds": value["total_time_seconds"],
                "load_time_seconds": value["load_time_seconds"],
                "control_script_sha256": value.get("control_script_sha256"),
                "model_manifest_sha256": value.get("model_run_signature", {}).get("model_manifest_sha256"),
                "fp16_source_compare": value["fp16_source_compare"],
            }
            for key, value in summary_analyses.items()
        },
        "timestamp": datetime.now().isoformat(),
    }
    atomic_write_json(output_dir / "run_summary.json", summary)
    print(f"\nRun summary saved: {output_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
