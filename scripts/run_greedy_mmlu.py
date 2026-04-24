"""
Experiment 49 Phase 1B: Greedy MMLU Generation + Feature Extraction

Same pipeline as run_mmlu_entropy.py but with greedy decoding:
  do_sample=False, temperature=1.0 (unused), no top_p/top_k

Checkpoint every 25 samples. Resume with --resume.

Usage:
  python scripts/run_greedy_mmlu.py --model qwen --smoke_test
  python scripts/run_greedy_mmlu.py --model qwen
  python scripts/run_greedy_mmlu.py --model qwen --resume

Run detached (survives session close):
  python scripts/run_detached.py "python scripts/run_greedy_mmlu.py --model qwen"
"""

import argparse
import json
import time
import random
import re
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import torch

SEED = 42
BASE = Path(__file__).resolve().parent.parent
EXP49 = BASE / "experiments" / "49_Deterministic_Label_Robustness"

MODEL_CONFIGS = {
    "qwen": {"name": "Qwen/Qwen2.5-7B-Instruct", "short": "qwen"},
    "llama": {"name": "meta-llama/Meta-Llama-3-8B-Instruct", "short": "llama"},
    "mistral": {"name": "mistralai/Mistral-7B-Instruct-v0.3", "short": "mistral"},
}


def log(tag, msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}][{tag}] {msg}", flush=True)


def load_mmlu_dataset(num_samples: int) -> List[Dict]:
    """EXACT same as run_mmlu_entropy.py"""
    from datasets import load_dataset
    log("DATA", "Loading MMLU...")
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
            "type": "mmlu", "question": item['question'],
            "choices": {"label": labels[:len(choices)], "text": choices},
            "answer_key": answer_key,
            "subject": item.get('subject', 'unknown'),
        })

    random.seed(SEED)
    random.shuffle(samples)
    log("DATA", f"Loaded {len(samples)} samples")
    return samples


def extract_mcq_answer(text: str) -> Optional[str]:
    """EXACT same as run_mmlu_entropy.py"""
    if not text:
        return None
    patterns = [
        r"[Tt]he answer is\s*[:\s]*\(?([A-Da-d])\)?",
        r"[Aa]nswer\s*[:\s]*\(?([A-Da-d])\)?",
        r"[Tt]herefore,?\s*(?:the answer is\s*)?\(?([A-Da-d])\)?",
        r"\b([A-Da-d])\s*\.\s*$",
        r"[Oo]ption\s+([A-Da-d])\s+is\s+correct",
        r"[Cc]orrect\s+(?:answer|option)\s+is\s+\(?([A-Da-d])\)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    letters = re.findall(r'\b([A-Da-d])\b', text)
    if letters:
        return letters[-1].upper()
    return None


def make_mmlu_prompt(tokenizer, sample: Dict) -> str:
    """EXACT same as run_mmlu_entropy.py"""
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


def compute_greedy_entropy(model, tokenizer, prompt, max_new_tokens=1024):
    """Same as compute_dual_entropy_with_scale but GREEDY decoding."""
    from transformers import MaxTimeCriteria, StoppingCriteriaList

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]

    time_limit = MaxTimeCriteria(max_time=300.0)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,           # GREEDY
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

    layer_data = {m: {i: [] for i in range(num_layers)} for m in
                  ["normed_entropy", "unnormed_entropy", "h_norm", "wh_norm",
                   "wh_rms", "logit_std", "logit_max", "logit_margin"]}

    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
        for token_idx, token_hidden_states in enumerate(outputs.hidden_states):
            with torch.no_grad():
                first_hidden = token_hidden_states[1] if len(token_hidden_states) > 1 else None
                if first_hidden is None:
                    continue
                is_multi_pos = first_hidden.shape[1] > 1

                if is_multi_pos:
                    for layer_idx in range(num_layers):
                        if layer_idx + 1 < len(token_hidden_states):
                            hidden = token_hidden_states[layer_idx + 1]
                            _process_layer(hidden, lm_head, norm, layer_data, layer_idx)
                else:
                    hiddens = []
                    for li in range(num_layers):
                        if li + 1 < len(token_hidden_states):
                            hiddens.append(token_hidden_states[li + 1])
                    if not hiddens:
                        continue
                    all_hidden = torch.cat(hiddens, dim=0)
                    _process_batch(all_hidden, lm_head, norm, layer_data)

    result = {}
    for li in range(num_layers):
        if layer_data["normed_entropy"][li]:
            result[li] = {m: float(np.mean(layer_data[m][li])) for m in layer_data}

    return output_text, result, len(generated_ids)


def _process_layer(hidden, lm_head, norm, layer_data, li):
    """Process single layer hidden state."""
    hidden_fp = hidden.to(lm_head.weight.dtype)
    h_n = float(torch.norm(hidden_fp, p=2).cpu().item())
    logits_u = lm_head(hidden_fp)
    logits_f = logits_u.float()
    m = logits_f.abs().max()
    wh_norm_scaled = float((m * torch.norm(logits_f / m, p=2)).cpu().item()) if m > 0 else 0.0
    wh_rms = float(torch.sqrt(torch.mean(logits_f ** 2)).cpu().item())
    logit_std_val = float(torch.std(logits_f).cpu().item())
    logit_max_val = float(logits_f.max().cpu().item())
    top2 = torch.topk(logits_f.view(-1), k=2)
    logit_margin_val = float((top2.values[0] - top2.values[1]).cpu().item())

    probs_u = torch.softmax(logits_u.float(), dim=-1).clamp(min=1e-10)
    ent_u = float((-probs_u * torch.log(probs_u)).sum(dim=-1)[0, 0].cpu().item())
    hidden_normed = norm(hidden_fp)
    logits_n = lm_head(hidden_normed)
    probs_n = torch.softmax(logits_n.float(), dim=-1).clamp(min=1e-10)
    ent_n = float((-probs_n * torch.log(probs_n)).sum(dim=-1)[0, 0].cpu().item())

    max_ent = float(np.log(logits_u.shape[-1]))
    layer_data["unnormed_entropy"][li].append(ent_u / max_ent)
    layer_data["normed_entropy"][li].append(ent_n / max_ent)
    layer_data["h_norm"][li].append(h_n)
    layer_data["wh_norm"][li].append(wh_norm_scaled)
    layer_data["wh_rms"][li].append(wh_rms)
    layer_data["logit_std"][li].append(logit_std_val)
    layer_data["logit_max"][li].append(logit_max_val)
    layer_data["logit_margin"][li].append(logit_margin_val)


def _process_batch(all_hidden, lm_head, norm, layer_data):
    """Process batch of layer hidden states (generated tokens)."""
    all_hidden_fp = all_hidden.to(lm_head.weight.dtype)
    actual_layers = all_hidden_fp.shape[0]

    h_norms = torch.norm(all_hidden_fp.view(actual_layers, -1), p=2, dim=-1)
    logits_u_all = lm_head(all_hidden_fp)
    logits_f_all = logits_u_all.float().squeeze(1)

    m_all = logits_f_all.abs().amax(dim=-1, keepdim=True)
    m_all_safe = m_all.clamp(min=1e-12)
    wh_norm_all = (m_all.squeeze(-1) * torch.norm(logits_f_all / m_all_safe, p=2, dim=-1))
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


def run_experiment(model_key: str, smoke_test: bool = False, resume: bool = False):
    cfg = MODEL_CONFIGS[model_key]
    out_dir = EXP49 / "phase1_greedy" / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"
    data_dir.mkdir(exist_ok=True)

    num_samples = 1000
    n_items = 10 if smoke_test else num_samples

    log("START", f"Greedy MMLU | Model: {cfg['name']} | items: {n_items}")
    log("START", f"Output: {out_dir}")

    samples = load_mmlu_dataset(num_samples)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    log("MODEL", f"Loading {cfg['name']}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["name"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["name"], torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    log("MODEL", f"Loaded on {next(model.parameters()).device}")

    # Determinism verification: 5 items x 2 runs (quick check before full run)
    log("VERIFY", "Checking greedy determinism (5 items x 2 runs)...")
    r1_texts = []
    r2_texts = []
    for s in samples[:5]:
        p = make_mmlu_prompt(tokenizer, s)
        t1, _, _ = compute_greedy_entropy(model, tokenizer, p, max_new_tokens=50)
        t2, _, _ = compute_greedy_entropy(model, tokenizer, p, max_new_tokens=50)
        r1_texts.append(t1)
        r2_texts.append(t2)
    match = all(a == b for a, b in zip(r1_texts, r2_texts))
    log("VERIFY", f"Determinism: {'PASS' if match else 'FAIL'}")
    if not match:
        for i, (a, b) in enumerate(zip(r1_texts, r2_texts)):
            if a != b:
                log("VERIFY", f"  Item {i} MISMATCH:")
                log("VERIFY", f"    run1: {a[:80]}")
                log("VERIFY", f"    run2: {b[:80]}")
        log("WARN", "Greedy not deterministic on this hardware. Proceeding anyway.")
    # Save verification result
    with open(data_dir / "determinism_check.json", "w") as f:
        json.dump({"n_items": 5, "all_match": match,
                    "texts_r1": [t[:200] for t in r1_texts],
                    "texts_r2": [t[:200] for t in r2_texts]}, f, indent=2)

    # Resume
    results = []
    correct_count = 0
    start_idx = 0
    if resume:
        ckpt_file = data_dir / "checkpoint.json"
        sr_file = data_dir / "sample_results.json"
        if ckpt_file.exists() and sr_file.exists():
            with open(ckpt_file) as f:
                ckpt = json.load(f)
            with open(sr_file, encoding='utf-8') as f:
                results = json.load(f)
            start_idx = ckpt["checkpoint_idx"]
            correct_count = ckpt["correct_count"]
            log("RESUME", f"Resuming from idx={start_idx}, correct={correct_count}")
        else:
            log("RESUME", "No checkpoint found, starting from scratch")

    t_start = time.time()

    for idx in range(start_idx, min(len(samples), n_items)):
        sample = samples[idx]
        try:
            prompt = make_mmlu_prompt(tokenizer, sample)
            output_text, layer_result, n_tokens = compute_greedy_entropy(
                model, tokenizer, prompt
            )

            gt = sample["answer_key"]
            pred = extract_mcq_answer(output_text)
            is_correct = (pred == gt) if pred else False
            if is_correct:
                correct_count += 1

            results.append({
                "idx": idx,
                "is_correct": is_correct,
                "predicted": pred,
                "ground_truth": gt,
                "subject": sample.get("subject", "unknown"),
                "output_text": output_text,
                "layer_data": {str(k): v for k, v in layer_result.items()},
                "num_tokens": n_tokens,
            })

        except Exception as e:
            log("ERROR", f"idx {idx}: {e}")
            traceback.print_exc()
            results.append({"idx": idx, "error": str(e), "is_correct": False})

        done = idx - start_idx + 1
        elapsed = time.time() - t_start
        if done % 5 == 0 or idx == start_idx:
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (min(len(samples), n_items) - idx - 1) / rate if rate > 0 else 0
            acc = correct_count / (idx + 1) * 100
            log("PROG", f"{idx+1}/{n_items} | acc={acc:.1f}% | "
                f"rate={rate:.2f}/s | ETA={timedelta(seconds=int(remaining))}")

        # Checkpoint every 25
        if (idx + 1) % 25 == 0:
            ckpt = {"checkpoint_idx": idx + 1, "correct_count": correct_count,
                    "timestamp": datetime.now().isoformat()}
            with open(data_dir / "checkpoint.json", 'w') as f:
                json.dump(ckpt, f, indent=2)
            with open(data_dir / "sample_results.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            log("CKPT", f"Saved checkpoint at idx={idx+1}")

    # Final save
    total_time = time.time() - t_start
    acc = correct_count / len(results) * 100 if results else 0

    with open(data_dir / "sample_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    summary = {
        "model": cfg["name"],
        "model_key": model_key,
        "n_items": len(results),
        "n_correct": correct_count,
        "accuracy": round(acc, 2),
        "decoding": "greedy (do_sample=False)",
        "prompt": "0-shot chat template (same as run_mmlu_entropy.py)",
        "max_new_tokens": 1024,
        "elapsed_seconds": round(total_time, 1),
        "smoke_test": smoke_test,
    }
    with open(data_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Clean up checkpoint on completion
    ckpt_file = data_dir / "checkpoint.json"
    if ckpt_file.exists() and not smoke_test:
        ckpt_file.rename(data_dir / "checkpoint_completed.json")

    log("DONE", f"Accuracy: {acc:.1f}% ({correct_count}/{len(results)}) | {total_time:.0f}s")
    log("DONE", f"Results: {data_dir / 'sample_results.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen", "llama", "mistral"])
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_experiment(args.model, smoke_test=args.smoke_test, resume=args.resume)


if __name__ == "__main__":
    main()
