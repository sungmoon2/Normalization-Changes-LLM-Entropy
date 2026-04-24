"""
Experiment 49 Phase 1: Deterministic MMLU Label Scoring

Computes deterministic correctness labels for the same 1000 MMLU items
using next-token logit scoring over {A, B, C, D} tokens.
Uses the EXACT same 0-shot chat prompt as the main experiments.

Usage:
  python scripts/run_deterministic_labels.py --model qwen --smoke_test
  python scripts/run_deterministic_labels.py --model qwen
  python scripts/run_deterministic_labels.py --model llama
  python scripts/run_deterministic_labels.py --model mistral
"""

import argparse
import json
import time
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from _paths import POT_DIR

SEED = 42
BASE = Path(__file__).resolve().parent.parent
EXP31 = POT_DIR / "experiments" / "31_MMLU_Domain_Extension"
EXP49 = BASE / "experiments" / "49_Deterministic_Label_Robustness"

MODEL_CONFIGS = {
    "qwen": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "exp31_dir": "EXP_20260219_053638_mmlu_qwen",
    },
    "llama": {
        "name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "exp31_dir": "EXP_20260219_171237_mmlu_llama",
    },
    "mistral": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "exp31_dir": "EXP_20260220_000610_mmlu_mistral",
    },
}


def log(tag, msg):
    print(f"[{tag}] {msg}", flush=True)


def load_mmlu_dataset(num_samples: int) -> List[Dict]:
    """Load MMLU dataset — EXACT same code as run_mmlu_entropy.py"""
    from datasets import load_dataset
    log("DATA", "Loading MMLU dataset...")
    dataset = load_dataset("cais/mmlu", "all", split="test")

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
            "subject": item.get('subject', 'unknown'),
            "original_index": int(idx),
        })

    random.seed(SEED)
    random.shuffle(samples)
    log("DATA", f"Loaded {len(samples)} MMLU samples")
    return samples


def make_mmlu_prompt_original(tokenizer, sample: Dict) -> str:
    """EXACT same prompt as run_mmlu_entropy.py (step-by-step, for reference)"""
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


def make_mmlu_prompt_direct(tokenizer, sample: Dict) -> str:
    """Direct-answer prompt for deterministic scoring.
    Same chat template, same question/choices, but asks for just the letter.
    This ensures the model's first token IS the answer letter."""
    choice_str = "\n".join(
        f"{label}. {text}"
        for label, text in zip(sample['choices']['label'], sample['choices']['text'])
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer with just the letter (A, B, C, or D)."},
        {"role": "user", "content": (
            f"Question: {sample['question']}\n\n{choice_str}\n\n"
            f"Answer:"
        )}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def get_answer_token_ids(tokenizer) -> Dict[str, int]:
    """Get token IDs for A, B, C, D — handles different tokenizers."""
    answer_ids = {}
    for letter in ['A', 'B', 'C', 'D']:
        # Try with space prefix first (most common), then without
        candidates = [
            tokenizer.encode(f" {letter}", add_special_tokens=False),
            tokenizer.encode(letter, add_special_tokens=False),
        ]
        found = False
        for ids in candidates:
            if len(ids) == 1:
                answer_ids[letter] = ids[0]
                found = True
                break
        if not found:
            # Fallback: use first token of the encoding
            ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
            answer_ids[letter] = ids[0]
            log("WARN", f"Letter {letter} encodes to {len(ids)} tokens, using first: {ids[0]}")

    log("TOKENS", f"Answer token IDs: {answer_ids}")
    # Verify they are distinct
    assert len(set(answer_ids.values())) == 4, f"Answer tokens not distinct: {answer_ids}"
    return answer_ids


def load_sampled_results(model_key: str) -> Dict[int, Dict]:
    """Load exp31 sampled results for agreement comparison."""
    exp31_dir = EXP31 / MODEL_CONFIGS[model_key]["exp31_dir"] / "data" / "sample_results.json"
    if not exp31_dir.exists():
        log("WARN", f"exp31 results not found: {exp31_dir}")
        return {}

    with open(exp31_dir, "r") as f:
        data = json.load(f)

    # Index by sample idx for fast lookup
    return {item["idx"]: item for item in data}


def score_deterministic(model, tokenizer, samples, answer_token_ids, device, n_items=None):
    """Score each MMLU item deterministically using next-token logits.
    Uses direct-answer prompt so model's first token is the answer letter."""
    if n_items:
        samples = samples[:n_items]

    results = []
    t0 = time.time()

    for i, sample in enumerate(samples):
        prompt = make_mmlu_prompt_direct(tokenizer, sample)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Get logits at the last position (where model would start generating)
        last_logits = outputs.logits[0, -1, :]  # shape: [vocab_size]

        # Extract log-probs for A/B/C/D
        log_probs = torch.log_softmax(last_logits.float(), dim=-1)
        probs = torch.softmax(last_logits.float(), dim=-1)

        letter_scores = {}
        letter_probs = {}
        for letter, tid in answer_token_ids.items():
            letter_scores[letter] = float(log_probs[tid].cpu().item())
            letter_probs[letter] = float(probs[tid].cpu().item())

        # Deterministic prediction = highest probability letter
        det_pred = max(letter_probs, key=letter_probs.get)
        det_correct = (det_pred == sample["answer_key"])

        # Margin and entropy over 4 options
        sorted_probs = sorted(letter_probs.values(), reverse=True)
        det_margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
        prob_arr = np.array(list(letter_probs.values()))
        prob_arr = prob_arr / prob_arr.sum()  # renormalize over 4 options
        det_entropy = float(-np.sum(prob_arr * np.log(prob_arr + 1e-12)))

        results.append({
            "idx": i,
            "subject": sample["subject"],
            "answer_key": sample["answer_key"],
            "det_pred": det_pred,
            "det_correct": det_correct,
            "letter_probs": {k: round(v, 6) for k, v in letter_probs.items()},
            "letter_log_probs": {k: round(v, 4) for k, v in letter_scores.items()},
            "det_margin": round(det_margin, 6),
            "det_entropy_4way": round(det_entropy, 4),
        })

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            acc = sum(r["det_correct"] for r in results) / len(results)
            log("PROG", f"{i+1}/{len(samples)} | acc={acc:.1%} | {elapsed:.0f}s")

    return results


def compute_agreement(det_results, sampled_results):
    """Compute agreement between deterministic and sampled labels."""
    agreements = []
    for det in det_results:
        idx = det["idx"]
        if idx in sampled_results:
            sam = sampled_results[idx]
            det["sampled_correct"] = sam["is_correct"]
            det["sampled_pred"] = sam.get("predicted", "?")
            det["agree"] = (det["det_correct"] == sam["is_correct"])
            agreements.append(det["agree"])
        else:
            det["sampled_correct"] = None
            det["sampled_pred"] = None
            det["agree"] = None

    if not agreements:
        return {"agreement_rate": None, "n_compared": 0}

    agree_rate = sum(agreements) / len(agreements)
    # Cohen's kappa
    det_labels = [det["det_correct"] for det in det_results if det["agree"] is not None]
    sam_labels = [det["sampled_correct"] for det in det_results if det["agree"] is not None]

    n = len(det_labels)
    if n == 0:
        return {"agreement_rate": None, "n_compared": 0}

    p_o = agree_rate
    p_det_pos = sum(det_labels) / n
    p_sam_pos = sum(sam_labels) / n
    p_e = p_det_pos * p_sam_pos + (1 - p_det_pos) * (1 - p_sam_pos)
    kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 0

    det_acc = sum(det_labels) / n
    sam_acc = sum(sam_labels) / n

    return {
        "n_compared": n,
        "det_accuracy": round(det_acc, 4),
        "sampled_accuracy": round(sam_acc, 4),
        "agreement_rate": round(agree_rate, 4),
        "cohens_kappa": round(kappa, 4),
        "n_agree": sum(agreements),
        "n_disagree": n - sum(agreements),
        "det_correct_sam_incorrect": sum(1 for d, s in zip(det_labels, sam_labels) if d and not s),
        "det_incorrect_sam_correct": sum(1 for d, s in zip(det_labels, sam_labels) if not d and s),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen", "llama", "mistral"])
    parser.add_argument("--smoke_test", action="store_true", help="Run on 50 items only")
    args = parser.parse_args()

    model_key = args.model
    cfg = MODEL_CONFIGS[model_key]
    n_items = 50 if args.smoke_test else None
    suffix = "_smoke" if args.smoke_test else ""

    out_dir = EXP49 / "phase1_labels" / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    log("START", f"Model: {cfg['name']}, smoke_test={args.smoke_test}")

    # Load dataset
    samples = load_mmlu_dataset(1000)
    log("DATA", f"Total samples: {len(samples)}")

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log("MODEL", f"Loading {cfg['name']}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["name"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["name"],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device
    log("MODEL", f"Loaded on {device}")

    # Get answer token IDs
    answer_token_ids = get_answer_token_ids(tokenizer)

    # Verify determinism: run first 5 items twice
    log("VERIFY", "Checking determinism (5 items x 2 runs)...")
    r1 = score_deterministic(model, tokenizer, samples[:5], answer_token_ids, device, n_items=5)
    r2 = score_deterministic(model, tokenizer, samples[:5], answer_token_ids, device, n_items=5)
    match = all(
        r1[i]["letter_probs"] == r2[i]["letter_probs"] and r1[i]["det_pred"] == r2[i]["det_pred"]
        for i in range(5)
    )
    log("VERIFY", f"Determinism check: {'PASS' if match else 'FAIL'}")
    if not match:
        for i in range(5):
            if r1[i]["det_pred"] != r2[i]["det_pred"]:
                log("VERIFY", f"  Item {i}: run1={r1[i]['det_pred']} run2={r2[i]['det_pred']}")
                log("VERIFY", f"    probs1={r1[i]['letter_probs']}")
                log("VERIFY", f"    probs2={r2[i]['letter_probs']}")

    # Score all items
    log("SCORE", f"Scoring {'50 (smoke)' if n_items else '1000'} items...")
    t_start = time.time()
    results = score_deterministic(model, tokenizer, samples, answer_token_ids, device, n_items=n_items)
    elapsed = time.time() - t_start

    det_acc = sum(r["det_correct"] for r in results) / len(results)
    log("DONE", f"Deterministic accuracy: {det_acc:.1%} ({sum(r['det_correct'] for r in results)}/{len(results)})")
    log("DONE", f"Elapsed: {elapsed:.0f}s ({elapsed/len(results):.1f}s/item)")

    # Load sampled results for agreement
    sampled = load_sampled_results(model_key)
    if sampled:
        agreement = compute_agreement(results, sampled)
        log("AGREE", f"Agreement: {agreement['agreement_rate']:.1%} "
            f"({agreement['n_agree']}/{agreement['n_compared']})")
        log("AGREE", f"Cohen's kappa: {agreement['cohens_kappa']:.3f}")
        log("AGREE", f"Det acc: {agreement['det_accuracy']:.1%}, "
            f"Sampled acc: {agreement['sampled_accuracy']:.1%}")

        # Escalation check
        if agreement['agreement_rate'] < 0.85:
            log("ESCALATION", "WARNING: Agreement < 85%. Greedy generation may be needed.")
        else:
            log("ESCALATION", "Agreement >= 85%. Label-only approach is sufficient.")
    else:
        agreement = {"n_compared": 0, "agreement_rate": None}

    # Save results
    output = {
        "config": {
            "model": cfg["name"],
            "model_key": model_key,
            "n_items": len(results),
            "smoke_test": args.smoke_test,
            "seed": SEED,
            "prompt": "0-shot chat template, direct-answer format (Answer with just the letter)",
            "scoring": "next-token logit softmax over {A,B,C,D} tokens",
            "answer_token_ids": answer_token_ids,
            "elapsed_seconds": round(elapsed, 1),
        },
        "determinism_check": match,
        "det_accuracy": round(det_acc, 4),
        "agreement": agreement,
        "results": results,
    }

    out_file = out_dir / f"det_labels{suffix}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log("SAVE", f"Saved: {out_file}")


if __name__ == "__main__":
    main()
