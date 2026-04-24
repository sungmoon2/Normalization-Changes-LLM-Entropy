"""
Experiment 52: Multi-Seed Generation Robustness (Full Run)
==========================================================
Same 1000 MMLU questions (DATA_SEED=42), different generation seeds.
Measures whether AUROC/profile conclusions are stable across generation runs.

Usage:
  python scripts/run_exp52_full.py --model qwen --gen_seed 123
  python scripts/run_exp52_full.py --model qwen --gen_seed 456 --resume

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

DATA_SEED = 42  # FIXED: same questions as all existing experiments

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
# Data Loading — DATA_SEED fixed, sample selection deterministic
# ============================================================================

def load_mmlu_dataset(num_samples: int) -> List[Dict]:
    from datasets import load_dataset
    log("DATA", "Loading MMLU (cais/mmlu, all, test)...")
    dataset = load_dataset("cais/mmlu", "all", split="test")
    log("DATA", f"Total MMLU size: {len(dataset)}")

    # CRITICAL: use DATA_SEED for sample selection — same 1000 questions as seed=42 runs
    rng = np.random.RandomState(DATA_SEED)
    indices = rng.choice(len(dataset), size=min(1000, len(dataset)), replace=False)
    indices.sort()

    # Take requested number
    indices = indices[:num_samples]

    samples = []
    for idx in indices:
        item = dataset[int(idx)]
        choices = item['choices']
        answer_idx = item['answer']
        labels = ['A', 'B', 'C', 'D']
        answer_key = labels[answer_idx] if answer_idx < 4 else 'A'
        samples.append({
            "type": "mmlu",
            "question": item['question'],
            "choices": {"label": labels[:len(choices)], "text": choices},
            "answer_key": answer_key,
            "subject": item.get('subject', 'unknown'),
            "original_idx": int(idx),
        })

    log("DATA", f"Loaded {len(samples)} MMLU samples (data_seed={DATA_SEED})")
    return samples


def extract_mcq_answer(text: str) -> Optional[str]:
    if not text:
        return None
    patterns = [
        r"[Tt]he answer is\s*[:\s]*\(?([A-Da-d])\)?",
        r"[Aa]nswer\s*[:\s]*\(?([A-Da-d])\)?",
        r"[Tt]herefore,?\s*(?:the answer is\s*)?\(?([A-Da-d])\)?",
        r"\b([A-Da-d])\s*\.\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    letters = re.findall(r'\b([A-Da-d])\b', text)
    if letters:
        return letters[-1].upper()
    return None


def make_prompt(tokenizer, sample: Dict) -> str:
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
            f"\"The answer is [LETTER]\" where LETTER is A, B, C, or D."
        )}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ============================================================================
# Entropy extraction (identical to run_mmlu_entropy.py batched path)
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
# Analysis functions
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
    """AUROC for discriminating correct vs incorrect."""
    if len(correct_vals) < 5 or len(incorrect_vals) < 5:
        return 0.5
    labels = [1] * len(correct_vals) + [0] * len(incorrect_vals)
    scores = list(correct_vals) + list(incorrect_vals)
    try:
        auc = roc_auc_score(labels, scores)
        return max(auc, 1 - auc)  # sign-agnostic
    except ValueError:
        return 0.5


def compute_profile_auroc(results, num_layers, metric_name):
    """Profile AUROC: use all layers as feature vector."""
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

    # 70/30 cal/test split (same as main paper)
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
    """H_pre incremental utility over logit_std (profile level)."""
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

    def fit_eval(X_train, X_test, y_train, y_test):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(Xtr, y_train)
        probs = lr.predict_proba(Xte)[:, 1]
        return roc_auc_score(y_test, probs)

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
    parser = argparse.ArgumentParser(description="Exp52: Multi-Seed MMLU Generation Robustness")
    parser.add_argument("--model", type=str, required=True, choices=["qwen", "llama", "mistral"])
    parser.add_argument("--gen_seed", type=int, required=True, help="Generation seed (e.g. 123, 456)")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    model_cfg = MODEL_CONFIGS[args.model]
    model_name = model_cfg["name"]
    model_short = model_cfg["short"]

    exp_dir = PROJECT_ROOT / "experiments" / "52_Multi_Seed_Robustness" / model_short / f"seed_{args.gen_seed}"
    data_dir = exp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()
    log("START", f"Exp52 Full: model={model_short}, gen_seed={args.gen_seed}, "
        f"data_seed={DATA_SEED}, n={args.num_samples}")

    # Load data with DATA_SEED (deterministic sample selection)
    samples = load_mmlu_dataset(args.num_samples)

    # Verify first question matches seed=42 reference
    log("VERIFY", f"First question: '{samples[0]['question'][:80]}...'")
    log("VERIFY", f"First answer: {samples[0]['answer_key']}, subject: {samples[0]['subject']}")

    # Set GENERATION seed — after data loading
    set_seed(args.gen_seed)
    log("SEED", f"Generation seed set to {args.gen_seed}")

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

            # Rebuild aggregators
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

    inference_start = datetime.now()
    error_count = 0

    for idx in range(start_idx, len(samples)):
        sample = samples[idx]
        try:
            prompt = make_prompt(tokenizer, sample)
            output_text, layer_result, n_tokens = compute_dual_entropy_with_scale(
                model, tokenizer, prompt
            )

            gt = sample["answer_key"]
            pred = extract_mcq_answer(output_text)
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
                "is_correct": is_correct,
                "predicted": pred,
                "ground_truth": gt,
                "subject": sample.get("subject", "unknown"),
                "layer_data": {str(k): v for k, v in layer_result.items()},
                "num_tokens": n_tokens,
                "output_text_preview": output_text[:200],
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
                f"Errors: {error_count} | ETA: {timedelta(seconds=int(remaining))}")

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

    # Save final results
    final_results = {
        "experiment": "52_Multi_Seed_Robustness",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": model_short,
            "model_name": model_name,
            "data_seed": DATA_SEED,
            "gen_seed": args.gen_seed,
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
            "execution_time_seconds": total_time.total_seconds(),
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
            "gen_seed": args.gen_seed,
        }, f, indent=2)

    log("DONE", f"Exp52 complete: {model_short} seed={args.gen_seed}, "
        f"acc={accuracy:.1f}%, time={total_time}")


if __name__ == "__main__":
    main()
