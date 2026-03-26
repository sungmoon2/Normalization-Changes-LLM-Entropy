"""
Phase 1: Token Position Ablation
=================================
GPT 12차 4개 응답 전면 수렴: "정의가 안 서면 나머지 다 흔들림"

목적:
  기존 run_mmlu_entropy.py를 수정하여 per-token entropy를 보존.
  3종 비교: Step 0 (prompt-last) / Step 1 (first-gen) / Full average

변경점 (원본 대비):
  - per-token entropy/metrics를 리스트로 저장 (평균만 저장하던 것 변경)
  - prompt 마지막 토큰 (token_idx=0)의 마지막 position만 따로 저장
  - 각 생성 토큰의 metrics를 개별 저장

범위 (GPT 12 수렴):
  - Qwen Hard (competition_math L4-5, 500 samples)
  - Qwen MMLU (1000 samples)
  - 3모델 전부는 과함 (Qwen 중심 + 추후 1 cross-model)

사용법:
  python run_phase1_token_position.py --dataset hard --num_samples 500
  python run_phase1_token_position.py --dataset mmlu --num_samples 1000
  python run_phase1_token_position.py --dataset hard --num_samples 500 --resume EXP_DIR

NOTE: GPU 필요. RTX 3090 Ti에서 실행.
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
from scipy import stats
from typing import Optional, Dict, Any, List
import argparse
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

# ============================================================================
SEED = 42
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.3
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

BASE_DIR = Path(__file__).parent.parent / "PoT_Experiment_Entropy_Attention_Extraction_Experiment"
EXPERIMENTS_DIR = BASE_DIR / "experiments" / "33_Phase1_Token_Position"


class MaxTimeCriteria(StoppingCriteria):
    def __init__(self, max_time_seconds: float = 300.0):
        self.max_time = max_time_seconds
        self.start_time = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.start_time is None:
            self.start_time = time.time()
        return time.time() - self.start_time > self.max_time


def set_seed(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model():
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        # NOTE: attn_implementation='eager' removed - causes CUDA assertion
        # in transformers 4.57.0. Not needed for hidden state extraction.
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.eval()
    print(f"  Layers: {model.config.num_hidden_layers}, Hidden: {model.config.hidden_size}")
    return model, tokenizer


def load_dataset(dataset_type, num_samples):
    """Load dataset samples."""
    if dataset_type == "hard":
        from datasets import load_dataset as hf_load
        ds = hf_load("qwedsacf/competition_math", split="train")
        # Filter Level 4-5
        hard_samples = [s for s in ds if s.get("level", "") in ["Level 4", "Level 5"]]
        random.shuffle(hard_samples)
        samples = hard_samples[:num_samples]
        print(f"  Loaded: {len(samples)} Math Hard samples (from {len(hard_samples)} total L4-5)")

        def format_sample(s):
            prompt = f"Solve: {s['problem']}\n\nLet's think step by step."
            return prompt, str(s.get("solution", "")), "math"

        return samples, format_sample

    elif dataset_type == "mmlu":
        from datasets import load_dataset as hf_load
        ds = hf_load("cais/mmlu", "all", split="test")
        all_samples = list(ds)
        random.shuffle(all_samples)
        samples = all_samples[:num_samples]
        print(f"  Loaded: {len(samples)} MMLU samples")

        choices = ["A", "B", "C", "D"]

        def format_sample(s):
            q = s["question"]
            opts = "\n".join(f"{choices[i]}. {s['choices'][i]}" for i in range(len(s['choices'])))
            prompt = (f"Answer the following multiple choice question. "
                     f"Think step by step, then give your final answer as "
                     f"\"The answer is [LETTER]\" where LETTER is A, B, C, or D.\n\n"
                     f"Question: {q}\n{opts}")
            gt = choices[s["answer"]]
            return prompt, gt, "mmlu"

        return samples, format_sample

    else:
        raise ValueError(f"Unknown dataset: {dataset_type}")


def compute_per_token_entropy(model, tokenizer, prompt):
    """
    Core function: compute per-token, per-layer entropy and metrics.

    Returns:
        output_text: generated text
        per_token_data: list of dicts, one per generation step
            Each dict has: {
                "step": int (0=prompt-last, 1=first-gen, ...),
                "is_prompt": bool,
                "layer_data": {
                    str(layer_idx): {
                        "normed_entropy": float,
                        "unnormed_entropy": float,
                        "h_norm": float,
                        "wh_norm": float,
                        "wh_rms": float,
                        "logit_std": float,
                        "logit_max": float,
                        "logit_margin": float,
                    }
                }
            }
        num_tokens: total generated tokens
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]

    time_limit = MaxTimeCriteria(max_time_seconds=300.0)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
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

    per_token_data = []

    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
        for token_idx, token_hidden_states in enumerate(outputs.hidden_states):
            with torch.no_grad():
                first_hidden = token_hidden_states[1] if len(token_hidden_states) > 1 else None
                if first_hidden is None:
                    continue

                is_multi_pos = first_hidden.shape[1] > 1

                if is_multi_pos:
                    # token_idx=0: prompt. Save ONLY the last position (prompt-last = Step 0)
                    last_pos = first_hidden.shape[1] - 1
                    step_data = {"step": 0, "is_prompt": True, "layer_data": {}}

                    for layer_idx in range(num_layers):
                        if layer_idx + 1 < len(token_hidden_states):
                            hidden = token_hidden_states[layer_idx + 1][:, last_pos:last_pos+1, :]
                            hidden_fp = hidden.to(lm_head.weight.dtype)

                            metrics = _compute_metrics(hidden_fp, lm_head, norm)
                            step_data["layer_data"][str(layer_idx)] = metrics

                    per_token_data.append(step_data)

                else:
                    # Generated tokens: each is Step 1, 2, 3, ...
                    step_data = {"step": token_idx, "is_prompt": False, "layer_data": {}}

                    hiddens = []
                    for li in range(num_layers):
                        if li + 1 < len(token_hidden_states):
                            hiddens.append(token_hidden_states[li + 1])
                    if not hiddens:
                        continue

                    all_hidden = torch.cat(hiddens, dim=0)  # [L, 1, hidden]
                    all_hidden_fp = all_hidden.to(lm_head.weight.dtype)
                    actual_layers = all_hidden_fp.shape[0]

                    # Batch compute all metrics
                    h_norms = torch.norm(all_hidden_fp.view(actual_layers, -1), p=2, dim=-1)
                    logits_u_all = lm_head(all_hidden_fp).float().squeeze(1)  # [L, vocab]

                    m_all = logits_u_all.abs().amax(dim=-1, keepdim=True)
                    m_all_safe = m_all.clamp(min=1e-12)
                    wh_norm_all = m_all.squeeze(-1) * torch.norm(logits_u_all / m_all_safe, p=2, dim=-1)
                    wh_rms_all = torch.sqrt(torch.mean(logits_u_all ** 2, dim=-1))
                    logit_std_all = torch.std(logits_u_all, dim=-1)
                    logit_max_all = logits_u_all.amax(dim=-1)
                    top2_all = torch.topk(logits_u_all, k=2, dim=-1)
                    logit_margin_all = top2_all.values[:, 0] - top2_all.values[:, 1]

                    probs_u_all = torch.softmax(logits_u_all, dim=-1).clamp(min=1e-10)
                    ent_u_all = (-probs_u_all * torch.log(probs_u_all)).sum(dim=-1)

                    all_normed = norm(all_hidden_fp)
                    logits_n_all = lm_head(all_normed).float().squeeze(1)
                    probs_n_all = torch.softmax(logits_n_all, dim=-1).clamp(min=1e-10)
                    ent_n_all = (-probs_n_all * torch.log(probs_n_all)).sum(dim=-1)

                    max_ent = float(np.log(logits_u_all.shape[-1]))

                    for li in range(actual_layers):
                        step_data["layer_data"][str(li)] = {
                            "normed_entropy": float(ent_n_all[li].cpu().item()) / max_ent,
                            "unnormed_entropy": float(ent_u_all[li].cpu().item()) / max_ent,
                            "h_norm": float(h_norms[li].cpu().item()),
                            "wh_norm": float(wh_norm_all[li].cpu().item()),
                            "wh_rms": float(wh_rms_all[li].cpu().item()),
                            "logit_std": float(logit_std_all[li].cpu().item()),
                            "logit_max": float(logit_max_all[li].cpu().item()),
                            "logit_margin": float(logit_margin_all[li].cpu().item()),
                        }

                    per_token_data.append(step_data)

    return output_text, per_token_data, len(generated_ids)


def _compute_metrics(hidden_fp, lm_head, norm):
    """Compute all 8 metrics for a single position."""
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
    ent_u = float((-probs_u * torch.log(probs_u)).sum(dim=-1).squeeze().cpu().item())
    hidden_normed = norm(hidden_fp)
    logits_n = lm_head(hidden_normed)
    probs_n = torch.softmax(logits_n.float(), dim=-1).clamp(min=1e-10)
    ent_n = float((-probs_n * torch.log(probs_n)).sum(dim=-1).squeeze().cpu().item())

    max_ent = float(np.log(logits_u.shape[-1]))

    return {
        "normed_entropy": ent_n / max_ent,
        "unnormed_entropy": ent_u / max_ent,
        "h_norm": h_n,
        "wh_norm": wh_norm_scaled,
        "wh_rms": wh_rms,
        "logit_std": logit_std_val,
        "logit_max": logit_max_val,
        "logit_margin": logit_margin_val,
    }


def aggregate_per_token(per_token_data, num_layers):
    """
    Aggregate per-token data into 3 position variants:
      - step0: prompt-last token only
      - step1: first generated token only
      - full_avg: mean of all generated tokens (current method)
    """
    step0_data = None
    gen_steps = []

    for td in per_token_data:
        if td["is_prompt"]:
            step0_data = td["layer_data"]
        else:
            gen_steps.append(td["layer_data"])

    # Step 1: first generated token
    step1_data = gen_steps[0] if len(gen_steps) > 0 else None

    # Full average: mean of all generated tokens
    full_avg_data = {}
    if gen_steps:
        metrics = ["normed_entropy", "unnormed_entropy", "h_norm", "wh_norm",
                    "wh_rms", "logit_std", "logit_max", "logit_margin"]
        for li in range(num_layers):
            lk = str(li)
            avg = {}
            for m in metrics:
                vals = [gs[lk][m] for gs in gen_steps if lk in gs and m in gs[lk]]
                avg[m] = float(np.mean(vals)) if vals else 0.0
            full_avg_data[lk] = avg

    return {
        "step0_prompt_last": step0_data,
        "step1_first_gen": step1_data,
        "full_gen_avg": full_avg_data,
        "n_gen_tokens": len(gen_steps),
    }


def extract_boxed_content(text):
    start_marker = '\\boxed{'
    start = text.find(start_marker)
    if start == -1:
        return None
    content_start = start + len(start_marker)
    depth = 0
    for i, char in enumerate(text[content_start:], content_start):
        if char == '{':
            depth += 1
        elif char == '}':
            if depth == 0:
                return text[content_start:i]
            depth -= 1
    return None


def extract_math_answer(text):
    if not text:
        return None
    boxed = extract_boxed_content(text)
    if boxed:
        return boxed.strip()
    patterns = [
        r'[Tt]he (?:final )?answer is[:\s]*([^\n\.]+)',
        r'=\s*([^\n=]+)$',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            return match.group(1).strip()
    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
    if numbers:
        return numbers[-1]
    return None


def normalize_answer(answer):
    if not answer:
        return ""
    answer = answer.strip()
    while '\\frac' in answer or '\\dfrac' in answer:
        answer = answer.replace('\\dfrac', '\\frac')
        frac_match = re.search(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', answer)
        if frac_match:
            num, den = frac_match.group(1), frac_match.group(2)
            answer = answer[:frac_match.start()] + f"({num})/({den})" + answer[frac_match.end():]
        else:
            break
    answer = re.sub(r'\\[a-zA-Z]+', '', answer)
    answer = re.sub(r'[{}$\\]', '', answer)
    answer = answer.replace('(', '').replace(')', '').replace(' ', '').lower()
    frac_match = re.match(r'^(-?\d+)/(-?\d+)$', answer)
    if frac_match:
        try:
            num, den = int(frac_match.group(1)), int(frac_match.group(2))
            if den != 0:
                answer = f"{num / den:.10f}".rstrip('0').rstrip('.')
        except (ValueError, ZeroDivisionError):
            pass
    try:
        val = float(answer)
        answer = f"{val:.10f}".rstrip('0').rstrip('.')
    except ValueError:
        pass
    return answer


def compare_math_answers(pred, gt):
    if not pred or not gt:
        return False
    np1, np2 = normalize_answer(pred), normalize_answer(gt)
    if np1 == np2:
        return True
    try:
        return abs(float(np1) - float(np2)) < 1e-6
    except (ValueError, TypeError):
        return False


def check_answer(predicted, ground_truth, dataset_type):
    """Check if the predicted answer is correct."""
    if dataset_type == "mmlu":
        match = re.search(r'(?:answer is|Answer:?)\s*\(?([A-D])\)?', predicted, re.IGNORECASE)
        if match:
            return match.group(1).upper() == ground_truth.upper()
        letters = re.findall(r'\b([A-D])\b', predicted)
        if letters:
            return letters[-1].upper() == ground_truth.upper()
        return False
    else:
        # Math: use proper extraction and comparison
        pred = extract_math_answer(predicted)
        gt = extract_math_answer(ground_truth)
        return compare_math_answers(pred, gt) if pred and gt else False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["hard", "mmlu"])
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    set_seed(SEED)

    # Setup experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"EXP_{timestamp}_{args.dataset}_token_position"
    exp_dir = EXPERIMENTS_DIR / exp_name
    if args.resume:
        exp_dir = EXPERIMENTS_DIR / args.resume
    os.makedirs(exp_dir / "data", exist_ok=True)

    print(f"{'='*60}")
    print(f"Phase 1: Token Position Ablation")
    print(f"Dataset: {args.dataset}, Samples: {args.num_samples}")
    print(f"Experiment: {exp_dir}")
    print(f"{'='*60}")

    # Load model
    model, tokenizer = load_model()
    num_layers = model.config.num_hidden_layers

    # Load dataset
    samples, format_fn = load_dataset(args.dataset, args.num_samples)

    # Resume checkpoint
    checkpoint_path = exp_dir / "data" / "checkpoint.json"
    completed = []
    start_idx = 0
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            completed = json.load(f)
        start_idx = len(completed)
        print(f"  Resuming from sample {start_idx}")

    # Process
    start_time = time.time()
    for i in range(start_idx, len(samples)):
        sample = samples[i]
        prompt_text, ground_truth, ds_type = format_fn(sample)

        # Apply chat template (matching original scripts)
        if ds_type == "math":
            messages = [
                {"role": "system", "content": "You are a helpful math assistant. Solve problems step by step."},
                {"role": "user", "content": prompt_text},
            ]
        else:
            messages = [{"role": "user", "content": prompt_text}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        try:
            t0 = time.time()
            output_text, per_token_data, num_tokens = compute_per_token_entropy(model, tokenizer, formatted)
            elapsed = time.time() - t0

            # Aggregate into 3 positions
            aggregated = aggregate_per_token(per_token_data, num_layers)

            is_correct = check_answer(output_text, ground_truth, ds_type)

            result = {
                "idx": i,
                "is_correct": is_correct,
                "predicted": output_text[:500],  # Truncate for storage
                "ground_truth": ground_truth,
                "num_tokens": num_tokens,
                "time": round(elapsed, 2),
                "position_data": aggregated,
            }

            if args.dataset == "mmlu":
                result["subject"] = sample.get("subject", "unknown")

            completed.append(result)

            # Save checkpoint every 10 samples
            if (i + 1) % 10 == 0 or i == len(samples) - 1:
                with open(checkpoint_path, "w") as f:
                    json.dump(completed, f, indent=1)

            n_correct = sum(1 for r in completed if r["is_correct"])
            eta = (time.time() - start_time) / (i - start_idx + 1) * (len(samples) - i - 1) if i > start_idx else 0
            print(f"  [{i+1}/{len(samples)}] correct={is_correct} "
                  f"tokens={num_tokens} gen_steps={aggregated['n_gen_tokens']} "
                  f"time={elapsed:.1f}s acc={n_correct}/{len(completed)} "
                  f"({n_correct/len(completed)*100:.1f}%) ETA={timedelta(seconds=int(eta))}")

        except Exception as e:
            print(f"  [{i+1}/{len(samples)}] ERROR: {e}")
            traceback.print_exc()
            completed.append({
                "idx": i,
                "is_correct": False,
                "error": str(e),
                "ground_truth": ground_truth,
            })

    # Final save
    with open(checkpoint_path, "w") as f:
        json.dump(completed, f, indent=1)

    # Summary
    n_total = len(completed)
    n_correct = sum(1 for r in completed if r.get("is_correct", False))
    n_errors = sum(1 for r in completed if "error" in r)
    total_time = time.time() - start_time

    summary = {
        "experiment": exp_name,
        "dataset": args.dataset,
        "model": MODEL_NAME,
        "num_layers": num_layers,
        "n_total": n_total,
        "n_correct": n_correct,
        "n_errors": n_errors,
        "accuracy": round(n_correct / n_total, 6) if n_total > 0 else 0,
        "total_time_seconds": round(total_time, 1),
        "seed": SEED,
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "timestamp": datetime.now().isoformat(),
        "positions_saved": ["step0_prompt_last", "step1_first_gen", "full_gen_avg"],
    }

    with open(exp_dir / "final_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Phase 1 Complete: {args.dataset}")
    print(f"  Accuracy: {n_correct}/{n_total} ({summary['accuracy']*100:.1f}%)")
    print(f"  Errors: {n_errors}")
    print(f"  Time: {timedelta(seconds=int(total_time))}")
    print(f"  Saved: {exp_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
