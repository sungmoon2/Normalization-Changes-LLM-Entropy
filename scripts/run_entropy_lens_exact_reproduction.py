# -*- coding: utf-8 -*-
"""
Entropy-Lens Exact Reproduction (v2 — protocol-matched)
========================================================
Reproduces Entropy-Lens (Ali et al., 2025) protocol as closely as possible.

Protocol matching (all 3 gaps from v1 fixed):
  1. Answer: logit-based (max probability among A/B/C/D), NOT text extraction
  2. Tokens: ONLY 32 generated tokens averaged (prompt-last EXCLUDED)
  3. Prompt: raw 1-shot format WITHOUT chat template (matching EL base template)

Original Entropy-Lens protocol:
  - Models: Llama-3.2-1B/3B, Llama-3-8B, Gemma-2-2B/9B, Phi-3, etc.
  - MMLU: 1-shot evaluation
  - Entropy: over 32 generated tokens (averaged)
  - Classifier: k-NN k=3, 10-fold CV
  - Correctness: highest probability among {A,B,C,D}

Usage:
  python run_entropy_lens_exact_reproduction.py --model llama8b [--num_samples 1000]
  python run_entropy_lens_exact_reproduction.py --model llama32 [--num_samples 1000]
"""

import os
import sys
import json
import random
import time
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
import argparse
from _paths import POT_DIR

PROJECT_ROOT = Path(__file__).parent.parent
SEED = 42
N_GEN_TOKENS = 32
TEMPERATURE = 0.3

MODEL_CONFIGS = {
    "llama8b": {
        "id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "short": "llama3_8b_v3",
    },
    "llama32": {
        "id": "meta-llama/Llama-3.2-3B-Instruct",
        "short": "llama32_3b_v3",
    },
}

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)

def log(tag, msg):
    print(f"[{tag}] {datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)


# ============================================================================
# MMLU Data Loading
# ============================================================================

def load_mmlu_dataset(num_samples: int) -> List[Dict]:
    from datasets import load_dataset
    log("DATA", "Loading MMLU dataset...")
    dataset = load_dataset("cais/mmlu", "all", split="test")
    log("DATA", f"MMLU total size: {len(dataset)}")

    np.random.seed(SEED)
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    indices.sort()

    samples = []
    for idx in indices:
        item = dataset[int(idx)]
        choices = item['choices']
        answer_idx = item['answer']
        labels = ['A', 'B', 'C', 'D']
        answer_key = labels[answer_idx] if answer_idx < 4 else 'A'
        samples.append({
            "question": item['question'],
            "choices": choices,
            "answer_key": answer_key,
            "subject": item.get('subject', 'unknown'),
        })

    random.shuffle(samples)
    log("DATA", f"Loaded {len(samples)} MMLU samples")
    return samples


# ============================================================================
# FIX 3: Raw 1-shot prompt WITHOUT chat template (matching EL base format)
# ============================================================================

def make_1shot_prompt_raw(sample: Dict) -> str:
    """
    Raw 1-shot MMLU prompt matching Entropy-Lens 'base' template.
    NO chat template — just plain text, ending with "The answer is"
    so the model's first token is the answer letter.
    """
    # 1-shot example
    example = (
        "Question: What is the capital of France?\n"
        "A. Berlin\n"
        "B. Madrid\n"
        "C. Paris\n"
        "D. Rome\n"
        "The answer is C.\n\n"
    )

    # Test question
    choice_str = "\n".join(
        f"{chr(65+i)}. {c}" for i, c in enumerate(sample['choices'])
    )
    test = (
        f"Question: {sample['question']}\n"
        f"{choice_str}\n"
        f"The answer is"
    )

    return example + test


# ============================================================================
# FIX 1: Logit-based answer extraction (max prob among A/B/C/D)
# ============================================================================

def get_answer_token_ids(tokenizer):
    """Get token IDs for A, B, C, D in the tokenizer's vocabulary."""
    answer_ids = {}
    for letter in ['A', 'B', 'C', 'D']:
        # Try multiple encodings: " A", "A", etc.
        candidates = [
            tokenizer.encode(f" {letter}", add_special_tokens=False),
            tokenizer.encode(letter, add_special_tokens=False),
        ]
        for ids in candidates:
            if len(ids) == 1:
                answer_ids[letter] = ids[0]
                break
        if letter not in answer_ids:
            # Fallback: take last token
            answer_ids[letter] = tokenizer.encode(f" {letter}", add_special_tokens=False)[-1]
    return answer_ids


def logit_based_answer(first_token_scores, answer_token_ids):
    """
    Determine answer from first generated token's logits.
    Returns the letter (A/B/C/D) with highest probability.
    Matches Entropy-Lens: "highest probability token among {A,B,C,D}"
    """
    probs = torch.softmax(first_token_scores.float(), dim=-1)
    best_letter = None
    best_prob = -1.0
    letter_probs = {}

    for letter, tid in answer_token_ids.items():
        p = float(probs[0, tid].cpu().item()) if probs.dim() > 1 else float(probs[tid].cpu().item())
        letter_probs[letter] = round(p, 6)
        if p > best_prob:
            best_prob = p
            best_letter = letter

    return best_letter, letter_probs


# ============================================================================
# FIX 2: Entropy over ONLY generated tokens (prompt-last EXCLUDED)
# ============================================================================

def compute_entropy_generated_only(model, tokenizer, prompt, answer_token_ids):
    """
    Generate 32 tokens. Compute per-layer entropy ONLY over generated tokens.
    Also extract logit-based answer from first token scores.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=N_GEN_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs.sequences[0][input_length:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    num_gen_tokens = len(generated_ids)

    # FIX 1: Logit-based answer from first token scores
    pred_letter = None
    letter_probs = {}
    if hasattr(outputs, 'scores') and outputs.scores:
        first_scores = outputs.scores[0]  # [1, vocab] or [vocab]
        pred_letter, letter_probs = logit_based_answer(first_scores, answer_token_ids)

    num_layers = model.config.num_hidden_layers
    lm_head = model.lm_head
    norm = model.model.norm

    layer_h_pre = {i: [] for i in range(num_layers)}
    layer_h_post = {i: [] for i in range(num_layers)}
    layer_h_norm_val = {i: [] for i in range(num_layers)}
    layer_logit_std_val = {i: [] for i in range(num_layers)}

    if not hasattr(outputs, 'hidden_states') or not outputs.hidden_states:
        return output_text, None, num_gen_tokens, pred_letter, letter_probs

    for token_idx, token_hidden_states in enumerate(outputs.hidden_states):
        with torch.no_grad():
            first_hidden = token_hidden_states[1] if len(token_hidden_states) > 1 else None
            if first_hidden is None:
                continue
            is_prompt_entry = first_hidden.shape[1] > 1

            if is_prompt_entry:
                # Entry 0 contains [1, prompt_len, hidden].
                # The LAST position is the first generated token's hidden state.
                # We extract only that position — NOT the prompt positions.
                for layer_idx in range(num_layers):
                    if layer_idx + 1 < len(token_hidden_states):
                        hidden = token_hidden_states[layer_idx + 1][:, -1:, :]
                        hidden_fp = hidden.to(lm_head.weight.dtype)

                        h_n = float(torch.norm(hidden_fp, p=2).cpu().item())
                        logits_u = lm_head(hidden_fp)
                        logits_f = logits_u.float()
                        logit_std_v = float(torch.std(logits_f).cpu().item())
                        probs_u = torch.softmax(logits_f, dim=-1).clamp(min=1e-10)
                        ent_u = float((-probs_u * torch.log(probs_u)).sum(dim=-1).squeeze().cpu().item())

                        hidden_normed = norm(hidden_fp)
                        logits_n = lm_head(hidden_normed)
                        probs_n = torch.softmax(logits_n.float(), dim=-1).clamp(min=1e-10)
                        ent_n = float((-probs_n * torch.log(probs_n)).sum(dim=-1).squeeze().cpu().item())

                        max_ent = float(np.log(logits_f.shape[-1]))
                        layer_h_pre[layer_idx].append(ent_u / max_ent)
                        layer_h_post[layer_idx].append(ent_n / max_ent)
                        layer_h_norm_val[layer_idx].append(h_n)
                        layer_logit_std_val[layer_idx].append(logit_std_v)
            else:
                # Subsequent entries: [1, 1, hidden] — batch all layers
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

                logits_u_all = lm_head(all_hidden_fp).float().squeeze(1)
                logit_std_all = torch.std(logits_u_all, dim=-1)
                probs_u_all = torch.softmax(logits_u_all, dim=-1).clamp(min=1e-10)
                ent_u_all = (-probs_u_all * torch.log(probs_u_all)).sum(dim=-1)

                all_normed = norm(all_hidden_fp)
                logits_n_all = lm_head(all_normed).float().squeeze(1)
                probs_n_all = torch.softmax(logits_n_all, dim=-1).clamp(min=1e-10)
                ent_n_all = (-probs_n_all * torch.log(probs_n_all)).sum(dim=-1)

                max_ent = float(np.log(logits_u_all.shape[-1]))

                for li in range(actual_layers):
                    layer_h_pre[li].append(float(ent_u_all[li].cpu().item()) / max_ent)
                    layer_h_post[li].append(float(ent_n_all[li].cpu().item()) / max_ent)
                    layer_h_norm_val[li].append(float(h_norms[li].cpu().item()))
                    layer_logit_std_val[li].append(float(logit_std_all[li].cpu().item()))

    # Average across generated tokens ONLY
    result = {}
    for li in range(num_layers):
        if layer_h_pre[li]:
            result[li] = {
                "unnormed_entropy": float(np.mean(layer_h_pre[li])),
                "normed_entropy": float(np.mean(layer_h_post[li])),
                "h_norm": float(np.mean(layer_h_norm_val[li])),
                "logit_std": float(np.mean(layer_logit_std_val[li])),
                "n_tokens_averaged": len(layer_h_pre[li]),
            }

    return output_text, result, num_gen_tokens, pred_letter, letter_probs


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(model_key: str, num_samples: int, resume_path: str = None):
    start_time = datetime.now()
    config = MODEL_CONFIGS[model_key]
    model_id = config["id"]
    model_short = config["short"]

    exp_base = POT_DIR / "experiments" / "47_Entropy_Lens_Exact_Reproduction"
    if resume_path:
        exp_dir = Path(resume_path)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = exp_base / f"EXP_{ts}_{model_short}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    data_dir = exp_dir / "data"
    data_dir.mkdir(exist_ok=True)

    log("EXP", f"Experiment directory: {exp_dir}")
    log("EXP", f"Model: {model_id}")
    log("EXP", f"Protocol: 1-shot raw (no chat template), {N_GEN_TOKENS} gen tokens only, logit-based answer, temp={TEMPERATURE}")

    # Checkpoint
    checkpoint_path = data_dir / "checkpoint.json"
    completed_indices = set()
    sample_results = []
    if resume_path and checkpoint_path.exists():
        cp = json.load(open(checkpoint_path))
        completed_indices = set(cp.get("completed_indices", []))
        results_path = data_dir / "sample_results.json"
        if results_path.exists():
            sample_results = json.load(open(results_path))
        log("RESUME", f"Resuming from {len(completed_indices)} completed samples")

    # Load model
    log("MODEL", f"Loading {model_id}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    num_layers = model.config.num_hidden_layers
    log("MODEL", f"Loaded. Layers={num_layers}, Hidden={model.config.hidden_size}, Vocab={model.config.vocab_size}")

    # Get answer token IDs
    answer_token_ids = get_answer_token_ids(tokenizer)
    log("MODEL", f"Answer token IDs: {answer_token_ids}")

    # Load data
    samples = load_mmlu_dataset(num_samples)
    total = len(samples)

    # Run
    errors = 0
    correct_count = sum(1 for r in sample_results if r.get("is_correct"))

    for i, sample in enumerate(samples):
        if i in completed_indices:
            continue

        try:
            t0 = time.time()
            prompt = make_1shot_prompt_raw(sample)
            output_text, layer_data, num_tokens, pred_letter, letter_probs = \
                compute_entropy_generated_only(model, tokenizer, prompt, answer_token_ids)
            elapsed = time.time() - t0

            if layer_data is None:
                errors += 1
                log("ERR", f"Sample {i}: no hidden states")
                continue

            is_correct = (pred_letter == sample["answer_key"]) if pred_letter else False
            if is_correct:
                correct_count += 1

            # Verify token count
            n_tokens_in_profile = layer_data.get(0, {}).get("n_tokens_averaged", 0)

            result = {
                "index": i,
                "subject": sample["subject"],
                "question": sample["question"][:100],
                "answer_key": sample["answer_key"],
                "predicted_logit": pred_letter,
                "letter_probs": letter_probs,
                "is_correct": is_correct,
                "output_text": output_text[:200],
                "num_generated_tokens": num_tokens,
                "n_tokens_in_entropy_avg": n_tokens_in_profile,
                "elapsed": round(elapsed, 2),
                "layer_data": layer_data,
            }
            sample_results.append(result)
            completed_indices.add(i)

            done = len(completed_indices)
            acc = correct_count / done * 100 if done > 0 else 0

            if done % 50 == 0 or done <= 5:
                log("PROG", f"{done}/{total} ({acc:.1f}% acc, {elapsed:.1f}s, err={errors}, avg_tok={n_tokens_in_profile})")

            if done % 100 == 0:
                json.dump(sample_results, open(data_dir / "sample_results.json", "w"), indent=1)
                json.dump({
                    "completed_indices": sorted(completed_indices),
                    "total": total,
                    "correct": correct_count,
                    "errors": errors,
                }, open(checkpoint_path, "w"), indent=2)
                log("SAVE", f"Checkpoint at {done}/{total}")

        except Exception as e:
            errors += 1
            log("ERR", f"Sample {i}: {str(e)[:100]}")
            if errors > 50:
                log("ABORT", "Too many errors, stopping")
                break

    # Final save
    json.dump(sample_results, open(data_dir / "sample_results.json", "w"), indent=1)

    elapsed_total = datetime.now() - start_time
    final_acc = correct_count / len(completed_indices) * 100 if completed_indices else 0

    summary = {
        "model": model_id,
        "model_short": model_short,
        "protocol": "entropy_lens_exact_v3",
        "protocol_details": {
            "prompt": "raw 1-shot, no chat template",
            "answer_method": "logit-based (max prob among A/B/C/D)",
            "entropy_tokens": "generated only (prompt-last excluded)",
            "n_gen_tokens": N_GEN_TOKENS,
            "temperature": TEMPERATURE,
            "cv": "10-fold (in analysis script)",
            "classifier": "k-NN k=3 (in analysis script)",
        },
        "num_samples": total,
        "completed": len(completed_indices),
        "correct": correct_count,
        "accuracy": round(final_acc, 2),
        "errors": errors,
        "num_layers": num_layers,
        "elapsed": str(elapsed_total),
        "timestamp": datetime.now().isoformat(),
    }
    json.dump(summary, open(exp_dir / "summary.json", "w"), indent=2)
    log("DONE", f"Completed: {len(completed_indices)}/{total}, Accuracy: {final_acc:.1f}%, Errors: {errors}, Time: {elapsed_total}")

    return exp_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    exp_dir = run_experiment(args.model, args.num_samples, args.resume)
    print(f"\nExperiment saved to: {exp_dir}")
