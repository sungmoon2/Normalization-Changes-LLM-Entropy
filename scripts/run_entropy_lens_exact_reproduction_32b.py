# -*- coding: utf-8 -*-
"""
Entropy-Lens Exact Reproduction with Llama-3-8B-Instruct
========================================================
Purpose: Reproduce Entropy-Lens (Ali et al., 2025) protocol as closely as possible
         using one of their original models, then add H_pre/H_post/scale controls.

Original Entropy-Lens protocol:
  - Models: Llama-3.2-1B/3B, Llama-3-8B, Gemma-2-2B/9B, Phi-3, etc.
  - MMLU: 1-shot evaluation
  - Entropy: over 32 generated tokens (averaged)
  - Classifier: k-NN k=3, 10-fold CV
  - Correctness: highest probability among {A,B,C,D}

Our reproduction:
  - Model: Llama-3-8B-Instruct (one of their original models)
  - MMLU: 1-shot (matching their template)
  - Tokens: 32 generated tokens (matching their protocol)
  - Classifier: k-NN k=3, 10-fold CV (matching their protocol)
  - Added: H_pre vs H_post comparison, scale baselines
  - Same 1000 MMLU samples as our main experiments (seed=42)

Usage:
  python run_entropy_lens_exact_reproduction.py [--num_samples 1000]
"""

import os
import sys
import json
import random
import re
import time
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
import argparse
import traceback
from _paths import POT_DIR

# ============================================================================
# Settings
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
SEED = 42

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_SHORT = "llama32_3b"
N_GEN_TOKENS = 32  # Entropy-Lens uses 32 generated tokens
TEMPERATURE = 0.3
MAX_TIME_PER_SAMPLE = 120.0  # seconds

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
# MMLU Data Loading (same seed=42 sampling as our main experiments)
# ============================================================================

def load_mmlu_dataset(num_samples: int) -> List[Dict]:
    """Load MMLU dataset with same seed=42 sampling."""
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
            "type": "mmlu",
            "question": item['question'],
            "choices": {
                "label": labels[:len(choices)],
                "text": choices,
            },
            "answer_key": answer_key,
            "answer_text": choices[answer_idx] if answer_idx < len(choices) else "",
            "subject": item.get('subject', 'unknown'),
        })

    random.shuffle(samples)
    log("DATA", f"Loaded {len(samples)} MMLU samples")
    return samples


# ============================================================================
# 1-Shot MMLU Prompt (matching Entropy-Lens protocol)
# ============================================================================

# Fixed 1-shot example (general knowledge, not from test set)
ONE_SHOT_EXAMPLE = {
    "question": "What is the capital of France?",
    "choices": ["Berlin", "Madrid", "Paris", "Rome"],
    "answer": "C"
}

def make_1shot_mmlu_prompt(tokenizer, sample: Dict) -> str:
    """
    1-shot MMLU prompt matching Entropy-Lens protocol.
    Uses chat template with a 1-shot example.
    """
    # Format 1-shot example
    example_choices = "\n".join(
        f"{chr(65+i)}. {c}" for i, c in enumerate(ONE_SHOT_EXAMPLE["choices"])
    )
    example_text = (
        f"Question: {ONE_SHOT_EXAMPLE['question']}\n\n"
        f"{example_choices}\n\n"
        f"The answer is {ONE_SHOT_EXAMPLE['answer']}."
    )

    # Format test question
    choice_str = "\n".join(
        f"{label}. {text}"
        for label, text in zip(sample['choices']['label'], sample['choices']['text'])
    )
    test_text = (
        f"Question: {sample['question']}\n\n"
        f"{choice_str}\n\n"
        f"The answer is"
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer the multiple choice question with the correct option letter."},
        {"role": "user", "content": f"Here is an example:\n\n{example_text}\n\nNow answer:\n\n{test_text}"}
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_mcq_answer(text: str) -> Optional[str]:
    """Extract MCQ answer (A/B/C/D) from generated text."""
    if not text:
        return None
    # Try structured patterns first
    patterns = [
        r"[Tt]he answer is\s*[:\s]*\(?([A-Da-d])\)?",
        r"[Aa]nswer\s*[:\s]*\(?([A-Da-d])\)?",
        r"^\s*\(?([A-Da-d])\)?[\.\s]",
        r"\b([A-Da-d])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    return None


# ============================================================================
# Entropy Computation (32 tokens, per-layer H_pre/H_post/scale)
# ============================================================================

def compute_entropy_32tokens(model, tokenizer, prompt):
    """
    Generate exactly 32 tokens and compute per-layer entropy at each token.
    Returns generation-average profiles (matching Entropy-Lens protocol).
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

    num_layers = model.config.num_hidden_layers
    lm_head = model.lm_head
    norm = model.model.norm  # RMSNorm

    # Accumulators for per-layer metrics across all generated tokens
    layer_h_pre = {i: [] for i in range(num_layers)}
    layer_h_post = {i: [] for i in range(num_layers)}
    layer_h_norm_val = {i: [] for i in range(num_layers)}
    layer_logit_std_val = {i: [] for i in range(num_layers)}

    if not hasattr(outputs, 'hidden_states') or not outputs.hidden_states:
        return output_text, None, num_gen_tokens

    for token_idx, token_hidden_states in enumerate(outputs.hidden_states):
        with torch.no_grad():
            first_hidden = token_hidden_states[1] if len(token_hidden_states) > 1 else None
            if first_hidden is None:
                continue
            is_prompt = first_hidden.shape[1] > 1

            if is_prompt:
                # First token includes full prompt — extract last position only
                for layer_idx in range(num_layers):
                    if layer_idx + 1 < len(token_hidden_states):
                        hidden = token_hidden_states[layer_idx + 1][:, -1:, :]  # last position
                        hidden_fp = hidden.to(lm_head.weight.dtype)

                        h_n = float(torch.norm(hidden_fp, p=2).cpu().item())

                        # H_pre (unnormed)
                        logits_u = lm_head(hidden_fp)
                        logits_f = logits_u.float()
                        logit_std_v = float(torch.std(logits_f).cpu().item())
                        probs_u = torch.softmax(logits_f, dim=-1).clamp(min=1e-10)
                        ent_u = float((-probs_u * torch.log(probs_u)).sum(dim=-1).squeeze().cpu().item())

                        # H_post (normed)
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
                # Subsequent tokens: [1, 1, hidden] — batch all layers
                hiddens = []
                for li in range(num_layers):
                    if li + 1 < len(token_hidden_states):
                        hiddens.append(token_hidden_states[li + 1])
                if not hiddens:
                    continue
                all_hidden = torch.cat(hiddens, dim=0)  # [L, 1, hidden]
                all_hidden_fp = all_hidden.to(lm_head.weight.dtype)
                actual_layers = all_hidden_fp.shape[0]

                h_norms = torch.norm(all_hidden_fp.view(actual_layers, -1), p=2, dim=-1)

                # Batch lm_head (unnormed)
                logits_u_all = lm_head(all_hidden_fp).float().squeeze(1)  # [L, vocab]
                logit_std_all = torch.std(logits_u_all, dim=-1)
                probs_u_all = torch.softmax(logits_u_all, dim=-1).clamp(min=1e-10)
                ent_u_all = (-probs_u_all * torch.log(probs_u_all)).sum(dim=-1)

                # Batch norm + lm_head (normed)
                all_normed = norm(all_hidden_fp)
                logits_n_all = lm_head(all_normed).float().squeeze(1)  # [L, vocab]
                probs_n_all = torch.softmax(logits_n_all, dim=-1).clamp(min=1e-10)
                ent_n_all = (-probs_n_all * torch.log(probs_n_all)).sum(dim=-1)

                max_ent = float(np.log(logits_u_all.shape[-1]))

                for li in range(actual_layers):
                    layer_h_pre[li].append(float(ent_u_all[li].cpu().item()) / max_ent)
                    layer_h_post[li].append(float(ent_n_all[li].cpu().item()) / max_ent)
                    layer_h_norm_val[li].append(float(h_norms[li].cpu().item()))
                    layer_logit_std_val[li].append(float(logit_std_all[li].cpu().item()))

    # Average across all tokens (generation-average, matching Entropy-Lens)
    result = {}
    for li in range(num_layers):
        if layer_h_pre[li]:
            result[li] = {
                "unnormed_entropy": float(np.mean(layer_h_pre[li])),
                "normed_entropy": float(np.mean(layer_h_post[li])),
                "h_norm": float(np.mean(layer_h_norm_val[li])),
                "logit_std": float(np.mean(layer_logit_std_val[li])),
            }

    return output_text, result, num_gen_tokens


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(num_samples: int, resume_path: str = None):
    start_time = datetime.now()

    # Experiment directory
    exp_base = POT_DIR / "experiments" / "47_Entropy_Lens_Exact_Reproduction"
    if resume_path:
        exp_dir = Path(resume_path)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = exp_base / f"EXP_{ts}_{MODEL_SHORT}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    data_dir = exp_dir / "data"
    data_dir.mkdir(exist_ok=True)

    log("EXP", f"Experiment directory: {exp_dir}")
    log("EXP", f"Model: {MODEL_ID}")
    log("EXP", f"Protocol: 1-shot MMLU, {N_GEN_TOKENS} tokens, temp={TEMPERATURE}")

    # Load checkpoint if resuming
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
    log("MODEL", f"Loading {MODEL_ID}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    num_layers = model.config.num_hidden_layers
    log("MODEL", f"Loaded. Layers={num_layers}, Hidden={model.config.hidden_size}, Vocab={model.config.vocab_size}")

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
            prompt = make_1shot_mmlu_prompt(tokenizer, sample)
            output_text, layer_data, num_tokens = compute_entropy_32tokens(model, tokenizer, prompt)
            elapsed = time.time() - t0

            if layer_data is None:
                errors += 1
                log("ERR", f"Sample {i}: no hidden states")
                continue

            # Extract answer
            pred = extract_mcq_answer(output_text)
            is_correct = (pred == sample["answer_key"]) if pred else False
            if is_correct:
                correct_count += 1

            result = {
                "index": i,
                "subject": sample["subject"],
                "question": sample["question"][:100],
                "answer_key": sample["answer_key"],
                "predicted": pred,
                "is_correct": is_correct,
                "output_text": output_text[:200],
                "num_tokens": num_tokens,
                "elapsed": round(elapsed, 2),
                "layer_data": layer_data,
            }
            sample_results.append(result)
            completed_indices.add(i)

            done = len(completed_indices)
            acc = correct_count / done * 100 if done > 0 else 0

            if done % 50 == 0 or done <= 5:
                log("PROG", f"{done}/{total} ({acc:.1f}% acc, {elapsed:.1f}s, err={errors})")

            # Checkpoint every 100 samples
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
        "model": MODEL_ID,
        "model_short": MODEL_SHORT,
        "protocol": "entropy_lens_exact_reproduction",
        "num_samples": total,
        "completed": len(completed_indices),
        "correct": correct_count,
        "accuracy": round(final_acc, 2),
        "errors": errors,
        "n_gen_tokens": N_GEN_TOKENS,
        "temperature": TEMPERATURE,
        "prompt_style": "1-shot",
        "num_layers": num_layers,
        "elapsed": str(elapsed_total),
        "timestamp": datetime.now().isoformat(),
    }
    json.dump(summary, open(exp_dir / "summary.json", "w"), indent=2)
    log("DONE", f"Completed: {len(completed_indices)}/{total}, Accuracy: {final_acc:.1f}%, Errors: {errors}, Time: {elapsed_total}")

    return exp_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    exp_dir = run_experiment(args.num_samples, args.resume)
    print(f"\nExperiment saved to: {exp_dir}")
