"""
Experiment 53: TruthfulQA mc1 Task Scope Extension (Full Run)
==============================================================
Run all 817 TruthfulQA mc1 items with layerwise entropy extraction.
Choice order shuffled per question to remove position bias.

Usage:
  python scripts/run_exp53_full.py --model qwen
  python scripts/run_exp53_full.py --model llama --resume

Designed to be called by run_exp52_53_chain.py via subprocess.
"""

import os
import sys
import json
import random
import re
import time
import argparse
import traceback
import numpy as np
import torch
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import StoppingCriteria, StoppingCriteriaList

PROJECT_ROOT = Path(__file__).parent.parent

# ============================================================================
# Constants
# ============================================================================

SEED = 42

MODEL_CONFIGS = {
    "qwen": {"name": "Qwen/Qwen2.5-7B-Instruct", "short": "qwen"},
    "llama": {"name": "meta-llama/Meta-Llama-3-8B-Instruct", "short": "llama"},
    "mistral": {"name": "mistralai/Mistral-7B-Instruct-v0.3", "short": "mistral"},
}

METRICS = ["normed_entropy", "unnormed_entropy", "h_norm", "logit_std",
           "wh_norm", "wh_rms", "logit_max", "logit_margin"]

CHECKPOINT_INTERVAL = 25


class MaxTimeCriteria(StoppingCriteria):
    def __init__(self, max_time_seconds: float = 300.0):
        self.max_time = max_time_seconds
        self.start_time = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.start_time is None:
            self.start_time = time.time()
        return time.time() - self.start_time > self.max_time


def log(tag, msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}][{tag}] {msg}", flush=True)


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================================
# TruthfulQA mc1 Data Loading
# ============================================================================

def load_truthfulqa_mc1() -> List[Dict]:
    """Load ALL 817 TruthfulQA mc1 items with per-question choice shuffling."""
    from datasets import load_dataset
    log("DATA", "Loading TruthfulQA mc1 (truthful_qa, multiple_choice, validation)...")
    dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
    log("DATA", f"TruthfulQA total: {len(dataset)} items")

    samples = []
    choice_count_stats = []

    for item_idx in range(len(dataset)):
        item = dataset[int(item_idx)]
        question = item['question']

        # mc1_targets: {"choices": [...], "labels": [0,1,0,...]}
        mc1 = item['mc1_targets']
        choices = list(mc1['choices'])
        labels = list(mc1['labels'])

        # CRITICAL: Shuffle per-question with deterministic seed
        # Original data always has correct answer at index 0
        # Use SEED + original_dataset_idx (NOT loop index) for determinism
        rng = np.random.RandomState(SEED + item_idx)
        perm = rng.permutation(len(choices))
        choices = [choices[p] for p in perm]
        labels = [labels[p] for p in perm]

        correct_idx = labels.index(1) if 1 in labels else 0
        num_choices = len(choices)
        choice_count_stats.append(num_choices)

        letters = [chr(65 + i) for i in range(num_choices)]
        answer_key = letters[correct_idx]

        samples.append({
            "type": "truthfulqa_mc1",
            "question": question,
            "choices": {"label": letters, "text": choices},
            "answer_key": answer_key,
            "num_choices": num_choices,
            "original_idx": item_idx,
        })

    log("DATA", f"Loaded {len(samples)} TruthfulQA mc1 samples")
    log("DATA", f"Choice counts: min={min(choice_count_stats)}, max={max(choice_count_stats)}, "
        f"mean={np.mean(choice_count_stats):.1f}, unique={len(set(choice_count_stats))}")

    return samples


# ============================================================================
# Answer Extraction (variable choice count)
# ============================================================================

def extract_mcq_answer(text: str, max_letter: str = 'M') -> Optional[str]:
    if not text:
        return None

    letter_range = f"A-{max_letter}a-{max_letter.lower()}"

    patterns = [
        rf"[Tt]he answer is\s*[:\s]*\(?([{letter_range}])\)?",
        rf"[Aa]nswer\s*[:\s]*\(?([{letter_range}])\)?",
        rf"[Tt]herefore,?\s*(?:the answer is\s*)?\(?([{letter_range}])\)?",
        rf"[Oo]ption\s+([{letter_range}])\s+is\s+correct",
        rf"[Cc]orrect\s+(?:answer|option)\s+is\s+\(?([{letter_range}])\)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()

    # Fallback: last standalone letter in range
    letters = re.findall(rf'\b([{letter_range}])\b', text)
    if letters:
        return letters[-1].upper()
    return None


def make_truthfulqa_prompt(tokenizer, sample: Dict) -> str:
    num_choices = sample['num_choices']
    choice_str = "\n".join(
        f"{label}. {text}"
        for label, text in zip(sample['choices']['label'], sample['choices']['text'])
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers multiple choice questions step by step."},
        {"role": "user", "content": (
            f"Answer the following multiple choice question step by step.\n\n"
            f"Question: {sample['question']}\n\n{choice_str}\n\n"
            f"Think through each option carefully, then end with "
            f"\"The answer is [LETTER]\" where LETTER is one of {', '.join(sample['choices']['label'])}."
        )}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ============================================================================
# Entropy extraction (identical to run_exp52_full.py / run_mmlu_entropy.py)
# ============================================================================

def compute_dual_entropy_with_scale(model, tokenizer, prompt, max_new_tokens=1024, temperature=0.3):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]

    time_limit = MaxTimeCriteria(max_time_seconds=300.0)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([time_limit]),
        )

    generated_ids = outputs.sequences[0][input_length:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    num_layers = model.config.num_hidden_layers
    lm_head = model.lm_head
    norm = model.model.norm

    layer_data = {m: {i: [] for i in range(num_layers)} for m in METRICS}

    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
        for token_idx, token_hidden_states in enumerate(outputs.hidden_states):
            with torch.no_grad():
                first_hidden = token_hidden_states[1] if len(token_hidden_states) > 1 else None
                if first_hidden is None:
                    continue
                is_multi_pos = first_hidden.shape[1] > 1

                if not is_multi_pos:
                    hiddens = []
                    for li in range(num_layers):
                        if li + 1 < len(token_hidden_states):
                            hiddens.append(token_hidden_states[li + 1])
                    if not hiddens:
                        continue
                    all_hidden = torch.cat(hiddens, dim=0)
                    all_hidden_fp = all_hidden.to(lm_head.weight.dtype)
                    actual_layers = all_hidden_fp.shape[0]

                    h_norms = torch.norm(all_hidden_fp.view(actual_layers, -1), p=2, dim=-1)
                    logits_u_all = lm_head(all_hidden_fp)
                    logits_f_all = logits_u_all.float().squeeze(1)

                    m_all = logits_f_all.abs().amax(dim=-1, keepdim=True)
                    m_all_safe = m_all.clamp(min=1e-12)
                    wh_norm_all = m_all.squeeze(-1) * torch.norm(logits_f_all / m_all_safe, p=2, dim=-1)
                    wh_rms_all = torch.sqrt(torch.mean(logits_f_all ** 2, dim=-1))
                    logit_std_all = torch.std(logits_f_all, dim=-1)
                    logit_max_all = logits_f_all.amax(dim=-1)
                    top2_all = torch.topk(logits_f_all, k=2, dim=-1)
                    logit_margin_all = top2_all.values[:, 0] - top2_all.values[:, 1]

                    probs_u_all = torch.softmax(logits_f_all, dim=-1).clamp(min=1e-10)
                    ent_u_all = (-probs_u_all * torch.log(probs_u_all)).sum(dim=-1)

                    all_normed = norm(all_hidden_fp)
                    logits_n_all = lm_head(all_normed).float().squeeze(1)
                    probs_n_all = torch.softmax(logits_n_all, dim=-1).clamp(min=1e-10)
                    ent_n_all = (-probs_n_all * torch.log(probs_n_all)).sum(dim=-1)

                    max_ent = float(np.log(logits_f_all.shape[-1]))

                    for li in range(actual_layers):
                        layer_data["unnormed_entropy"][li].append(float(ent_u_all[li].cpu().item()) / max_ent)
                        layer_data["normed_entropy"][li].append(float(ent_n_all[li].cpu().item()) / max_ent)
                        layer_data["h_norm"][li].append(float(h_norms[li].cpu().item()))
                        layer_data["wh_norm"][li].append(float(wh_norm_all[li].cpu().item()))
                        layer_data["wh_rms"][li].append(float(wh_rms_all[li].cpu().item()))
                        layer_data["logit_std"][li].append(float(logit_std_all[li].cpu().item()))
                        layer_data["logit_max"][li].append(float(logit_max_all[li].cpu().item()))
                        layer_data["logit_margin"][li].append(float(logit_margin_all[li].cpu().item()))

    result = {}
    for li in range(num_layers):
        if layer_data["normed_entropy"][li]:
            result[li] = {m: float(np.mean(layer_data[m][li])) for m in layer_data}

    return output_text, result, len(generated_ids)


# ============================================================================
# Analysis functions (same as run_exp52_full.py)
# ============================================================================

def compute_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return (m1 - m2) / pooled if pooled > 0 else 0.0


def compute_auroc(correct_vals, incorrect_vals):
    if len(correct_vals) < 5 or len(incorrect_vals) < 5:
        return 0.5
    labels = [1] * len(correct_vals) + [0] * len(incorrect_vals)
    scores = list(correct_vals) + list(incorrect_vals)
    try:
        auc = roc_auc_score(labels, scores)
        return max(auc, 1 - auc)
    except ValueError:
        return 0.5


def compute_profile_auroc(results, num_layers, metric_name):
    X, y = [], []
    for r in results:
        if "layer_data" not in r:
            continue
        feat = []
        for li in range(num_layers):
            val = r["layer_data"].get(str(li), {}).get(metric_name, 0.0)
            feat.append(val)
        if len(feat) == num_layers:
            X.append(feat)
            y.append(1 if r["is_correct"] else 0)

    if len(X) < 20 or len(set(y)) < 2:
        return 0.5

    X = np.array(X)
    y = np.array(y)

    rng = np.random.RandomState(42)
    n = len(y)
    perm = rng.permutation(n)
    cal_n = int(n * 0.7)
    cal_idx, test_idx = perm[:cal_n], perm[cal_n:]

    scaler = StandardScaler()
    X_cal = scaler.fit_transform(X[cal_idx])
    X_test = scaler.transform(X[test_idx])

    try:
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_cal, y[cal_idx])
        probs = lr.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y[test_idx], probs)
        return float(auc)
    except Exception:
        return 0.5


def compute_incremental_utility(results, num_layers):
    X_base, X_full, y = [], [], []
    for r in results:
        if "layer_data" not in r:
            continue
        base_feat, full_feat = [], []
        for li in range(num_layers):
            ld = r["layer_data"].get(str(li), {})
            base_feat.append(ld.get("logit_std", 0.0))
            full_feat.append(ld.get("logit_std", 0.0))
            full_feat.append(ld.get("unnormed_entropy", 0.0))
        if len(base_feat) == num_layers:
            X_base.append(base_feat)
            X_full.append(full_feat)
            y.append(1 if r["is_correct"] else 0)

    if len(X_base) < 20 or len(set(y)) < 2:
        return {"delta": 0.0, "base_auroc": 0.5, "full_auroc": 0.5}

    X_base = np.array(X_base)
    X_full = np.array(X_full)
    y = np.array(y)

    rng = np.random.RandomState(42)
    n = len(y)
    perm = rng.permutation(n)
    cal_n = int(n * 0.7)
    cal_idx, test_idx = perm[:cal_n], perm[cal_n:]

    def fit_eval(X_train, X_test_, y_train, y_test_):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test_)
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(Xtr, y_train)
        probs = lr.predict_proba(Xte)[:, 1]
        return roc_auc_score(y_test_, probs)

    try:
        base_auc = fit_eval(X_base[cal_idx], X_base[test_idx], y[cal_idx], y[test_idx])
        full_auc = fit_eval(X_full[cal_idx], X_full[test_idx], y[cal_idx], y[test_idx])
        return {"delta": float(full_auc - base_auc), "base_auroc": float(base_auc), "full_auroc": float(full_auc)}
    except Exception:
        return {"delta": 0.0, "base_auroc": 0.5, "full_auroc": 0.5}


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Exp53: TruthfulQA mc1 Task Extension")
    parser.add_argument("--model", type=str, required=True, choices=["qwen", "llama", "mistral"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    model_cfg = MODEL_CONFIGS[args.model]
    model_name = model_cfg["name"]
    model_short = model_cfg["short"]

    exp_dir = PROJECT_ROOT / "experiments" / "53_Task_Scope_Extension" / model_short
    data_dir = exp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()
    log("START", f"Exp53 Full: model={model_short}, seed={SEED}, dataset=TruthfulQA mc1")

    # Set seed
    set_seed(SEED)

    # Load ALL TruthfulQA mc1 items (817)
    samples = load_truthfulqa_mc1()

    # Verify first few samples match smoke test
    for i in range(min(3, len(samples))):
        s = samples[i]
        log("VERIFY", f"Q{i}: orig_idx={s['original_idx']}, choices={s['num_choices']}, "
            f"answer={s['answer_key']}, q='{s['question'][:60]}...'")

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log("MODEL", f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    num_layers = model.config.num_hidden_layers
    log("MODEL", f"Loaded: {num_layers} layers")

    # Aggregators
    correct_data = {m: {i: [] for i in range(num_layers)} for m in METRICS}
    incorrect_data = {m: {i: [] for i in range(num_layers)} for m in METRICS}
    results = []
    correct_count = 0
    start_idx = 0

    # Resume
    if args.resume:
        ckpt_file = data_dir / "checkpoint.json"
        sr_file = data_dir / "sample_results.json"
        if ckpt_file.exists() and sr_file.exists():
            with open(ckpt_file, 'r') as f:
                ckpt = json.load(f)
            with open(sr_file, 'r') as f:
                prev_results = json.load(f)

            start_idx = ckpt["checkpoint_idx"]
            correct_count = ckpt["correct_count"]
            results = prev_results

            for r in prev_results:
                if "error" in r or "layer_data" not in r:
                    continue
                is_correct = r["is_correct"]
                for li_str, layer_vals in r["layer_data"].items():
                    li = int(li_str)
                    for m in METRICS:
                        if m in layer_vals:
                            if is_correct:
                                correct_data[m][li].append(layer_vals[m])
                            else:
                                incorrect_data[m][li].append(layer_vals[m])

            log("RESUME", f"checkpoint_idx={start_idx}, correct_count={correct_count}, "
                f"prev_results={len(prev_results)}")
        else:
            log("RESUME", "No checkpoint found, starting from scratch")

    # Check if already completed
    completed_file = data_dir / "checkpoint_completed.json"
    if completed_file.exists():
        log("SKIP", f"Already completed: {completed_file}")
        return

    # Reset generation seed after data loading
    set_seed(SEED)

    inference_start = datetime.now()
    error_count = 0
    extraction_success = 0

    for idx in range(start_idx, len(samples)):
        sample = samples[idx]
        try:
            prompt = make_truthfulqa_prompt(tokenizer, sample)
            output_text, layer_result, n_tokens = compute_dual_entropy_with_scale(
                model, tokenizer, prompt
            )

            gt = sample["answer_key"]
            max_letter = chr(64 + sample["num_choices"])
            pred = extract_mcq_answer(output_text, max_letter=max_letter)

            if pred is not None:
                extraction_success += 1

            is_correct = (pred == gt) if pred else False
            if is_correct:
                correct_count += 1

            for li in range(num_layers):
                if li in layer_result:
                    for m in METRICS:
                        val = layer_result[li][m]
                        if is_correct:
                            correct_data[m][li].append(val)
                        else:
                            incorrect_data[m][li].append(val)

            results.append({
                "idx": idx,
                "original_idx": sample["original_idx"],
                "is_correct": is_correct,
                "predicted": pred,
                "ground_truth": gt,
                "question_preview": sample["question"][:100],
                "num_choices": sample["num_choices"],
                "layer_data": {str(k): v for k, v in layer_result.items()},
                "num_tokens": n_tokens,
                "output_text_preview": output_text[:300],
            })

        except Exception as e:
            log("ERROR", f"idx {idx}: {e}")
            traceback.print_exc()
            results.append({"idx": idx, "error": str(e), "is_correct": False})
            error_count += 1

        # Progress
        elapsed = datetime.now() - inference_start
        done = idx - start_idx + 1
        if done % 5 == 0 or idx == start_idx:
            rate = done / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
            remaining = (len(samples) - idx - 1) / rate if rate > 0 else 0
            acc = correct_count / (idx + 1) * 100
            log("PROGRESS", f"{idx+1}/{len(samples)} | Acc: {acc:.1f}% | "
                f"Errors: {error_count} | Extract: {extraction_success}/{done} | "
                f"ETA: {timedelta(seconds=int(remaining))}")

        # Checkpoint
        if (idx + 1) % CHECKPOINT_INTERVAL == 0:
            ckpt = {"checkpoint_idx": idx + 1, "correct_count": correct_count}
            with open(data_dir / "checkpoint.json", 'w', encoding='utf-8') as f:
                json.dump(ckpt, f, indent=2)
            with open(data_dir / "sample_results.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            log("CKPT", f"Saved checkpoint at idx={idx+1}")

    # ========================================================================
    # Analysis
    # ========================================================================
    total_time = datetime.now() - start_time
    accuracy = correct_count / len(samples) * 100
    n_incorrect = len(samples) - correct_count

    log("ANALYSIS", f"Accuracy: {accuracy:.1f}% ({correct_count}/{len(samples)})")
    log("ANALYSIS", f"Errors: {error_count}, Extraction: {extraction_success}/{len(samples)}")

    # Per-layer AUROC
    layer_analysis = []
    for li in range(num_layers):
        entry = {"layer": li}
        for m in METRICS:
            c_vals = correct_data[m][li]
            i_vals = incorrect_data[m][li]
            entry[f"{m}_auroc"] = compute_auroc(c_vals, i_vals)
            entry[f"{m}_d"] = compute_cohens_d(i_vals, c_vals)
        layer_analysis.append(entry)

    # Best layers
    best_layers = {}
    for m in METRICS:
        aurocs = [(la["layer"], la[f"{m}_auroc"]) for la in layer_analysis]
        best = max(aurocs, key=lambda x: x[1])
        best_layers[m] = {"layer": best[0], "auroc": best[1]}
        log("BEST", f"{m}: Layer {best[0]} AUROC={best[1]:.4f}")

    # Profile AUROCs
    profile_aurocs = {}
    for m in ["unnormed_entropy", "normed_entropy", "logit_std", "h_norm"]:
        pauc = compute_profile_auroc(results, num_layers, m)
        profile_aurocs[m] = pauc
        log("PROFILE", f"{m} profile AUROC: {pauc:.4f}")

    # Incremental utility
    incr = compute_incremental_utility(results, num_layers)
    log("INCR", f"H_pre incremental over logit_std: delta={incr['delta']:.4f}")

    # Choice count statistics
    choice_counts = [r.get("num_choices", 0) for r in results if "num_choices" in r]

    # Save final results
    final_results = {
        "experiment": "53_Task_Scope_Extension",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": model_short,
            "model_name": model_name,
            "dataset": "truthful_qa_mc1",
            "seed": SEED,
            "num_samples": len(results),
            "temperature": 0.3,
            "do_sample": True,
            "max_new_tokens": 1024,
        },
        "results": {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "incorrect_count": n_incorrect,
            "error_count": error_count,
            "extraction_rate": extraction_success / len(samples) * 100 if samples else 0,
            "execution_time_seconds": total_time.total_seconds(),
            "choice_count_stats": {
                "min": min(choice_counts) if choice_counts else 0,
                "max": max(choice_counts) if choice_counts else 0,
                "mean": float(np.mean(choice_counts)) if choice_counts else 0,
            },
        },
        "best_layers": best_layers,
        "profile_aurocs": profile_aurocs,
        "incremental_utility": incr,
        "layer_analysis": layer_analysis,
    }

    with open(exp_dir / "final_results.json", 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    with open(data_dir / "sample_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Completion marker
    with open(completed_file, 'w', encoding='utf-8') as f:
        json.dump({
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "accuracy": accuracy,
        }, f, indent=2)

    log("DONE", f"Exp53 complete: {model_short}, acc={accuracy:.1f}%, time={total_time}")


if __name__ == "__main__":
    main()
