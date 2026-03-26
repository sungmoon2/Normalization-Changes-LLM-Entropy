"""
Phase 4b: Fair Self-Consistency Baseline
=========================================
교수님 직접 지시: "셀프 이컨시스턴시에서 이게 지금 템팔차가 서로 달라요"
GPT 12: "same prompt, same answer extraction, same temperature, same token budget"

기존 문제:
  - 본 방법: temp=0.3
  - 기존 SC: temp=0.7, K=5
  → 불공정 비교

이 실험:
  - temp=0.3 (본 방법과 동일)
  - K=5 (5회 샘플링)
  - Qwen Hard 200 samples (기존 SC 실험과 동일 규모)
  - 다수결로 정답 결정
  - 일치율(agreement ratio)을 confidence score로 사용 → AUROC 비교

사용법:
  python run_phase4_fair_sc.py --num_samples 200 --K 5
"""

import os
import json
import random
import re
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
import argparse

SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
TEMPERATURE = 0.3  # SAME as main method
MAX_NEW_TOKENS = 1024
BASE_DIR = Path(__file__).parent.parent / "PoT_Experiment_Entropy_Attention_Extraction_Experiment"
OUTPUT_DIR = BASE_DIR / "experiments" / "38_Phase4_Fair_SC"


def set_seed(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    try:
        val = float(answer)
        answer = f"{val:.10f}".rstrip('0').rstrip('.')
    except ValueError:
        pass
    return answer


def compare_answers(a, b):
    if not a or not b:
        return False
    na, nb = normalize_answer(a), normalize_answer(b)
    if na == nb:
        return True
    try:
        return abs(float(na) - float(nb)) < 1e-6
    except (ValueError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--K", type=int, default=5)
    args = parser.parse_args()

    set_seed()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print(f"Phase 4b: Fair Self-Consistency (temp={TEMPERATURE}, K={args.K})")
    print(f"Samples: {args.num_samples}")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.eval()
    print(f"  Model loaded: {MODEL_NAME}")

    # Load dataset
    from datasets import load_dataset
    ds = load_dataset("qwedsacf/competition_math", split="train")
    hard = [s for s in ds if s.get("level", "") in ["Level 4", "Level 5"]]
    random.shuffle(hard)
    samples = hard[:args.num_samples]
    print(f"  Dataset: {len(samples)} Math Hard samples")

    start_time = time.time()
    results = []

    for i, sample in enumerate(samples):
        prompt = f"Solve: {sample['problem']}\n\nLet's think step by step."
        messages = [
            {"role": "system", "content": "You are a helpful math assistant. Solve problems step by step."},
            {"role": "user", "content": prompt},
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        input_length = inputs.input_ids.shape[1]
        gt = extract_math_answer(sample.get("solution", ""))

        # Generate K samples
        answers = []
        outputs_list = []

        for k in range(args.K):
            try:
                # Different seed for each sample to get diversity
                torch.manual_seed(SEED + i * args.K + k)
                torch.cuda.manual_seed(SEED + i * args.K + k)

                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE,
                        do_sample=True,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                gen_text = tokenizer.decode(out[0][input_length:], skip_special_tokens=True)
                pred = extract_math_answer(gen_text)
                answers.append(pred)
                outputs_list.append(gen_text[:200])

            except Exception as e:
                answers.append(None)
                outputs_list.append(f"ERROR: {e}")

        # Majority vote
        valid_answers = [a for a in answers if a is not None]
        if not valid_answers:
            sc_answer = None
            agreement = 0.0
        else:
            # Group by equivalence
            groups = []
            for a in valid_answers:
                found = False
                for g in groups:
                    if compare_answers(a, g[0]):
                        g.append(a)
                        found = True
                        break
                if not found:
                    groups.append([a])

            # Largest group = majority
            groups.sort(key=len, reverse=True)
            sc_answer = groups[0][0]
            agreement = len(groups[0]) / len(valid_answers)

        sc_correct = compare_answers(sc_answer, gt) if sc_answer and gt else False

        # Single-pass (k=0) correctness
        single_correct = compare_answers(answers[0], gt) if answers[0] and gt else False

        result = {
            "idx": i,
            "ground_truth": gt,
            "answers": answers,
            "sc_answer": sc_answer,
            "sc_correct": sc_correct,
            "single_correct": single_correct,
            "agreement": round(agreement, 4),
            "n_valid": len(valid_answers),
            "K": args.K,
        }
        results.append(result)

        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (len(samples) - i - 1)
            n_sc = sum(1 for r in results if r["sc_correct"])
            n_single = sum(1 for r in results if r["single_correct"])
            print(f"  [{i+1}/{len(samples)}] SC={n_sc}/{i+1} ({n_sc/(i+1)*100:.1f}%) "
                  f"Single={n_single}/{i+1} ({n_single/(i+1)*100:.1f}%) "
                  f"agree={np.mean([r['agreement'] for r in results]):.2f} "
                  f"{timedelta(seconds=int(elapsed))} ETA {timedelta(seconds=int(eta))}")

        # Checkpoint every 50
        if (i + 1) % 50 == 0:
            with open(OUTPUT_DIR / "checkpoint.json", "w") as f:
                json.dump(results, f, indent=1)

    total_time = time.time() - start_time

    # Save full results
    with open(OUTPUT_DIR / "fair_sc_results.json", "w") as f:
        json.dump(results, f, indent=1)

    # Analysis
    n = len(results)
    n_sc_correct = sum(1 for r in results if r["sc_correct"])
    n_single_correct = sum(1 for r in results if r["single_correct"])
    agreements = [r["agreement"] for r in results]

    # SC agreement as confidence → AUROC
    y = np.array([1 if r["sc_correct"] else 0 for r in results])
    sc_conf = np.array(agreements)
    sc_auroc = float(roc_auc_score(y, sc_conf)) if len(np.unique(y)) >= 2 else 0.5

    print(f"\n{'='*60}")
    print(f"RESULTS (temp={TEMPERATURE}, K={args.K})")
    print(f"{'='*60}")
    print(f"  SC accuracy:     {n_sc_correct}/{n} ({n_sc_correct/n*100:.1f}%)")
    print(f"  Single accuracy: {n_single_correct}/{n} ({n_single_correct/n*100:.1f}%)")
    print(f"  Mean agreement:  {np.mean(agreements):.4f}")
    print(f"  SC AUROC (agreement→correctness): {sc_auroc:.4f}")
    print(f"  Time: {timedelta(seconds=int(total_time))}")

    summary = {
        "temperature": TEMPERATURE,
        "K": args.K,
        "n_samples": n,
        "sc_accuracy": round(n_sc_correct / n, 6),
        "single_accuracy": round(n_single_correct / n, 6),
        "mean_agreement": round(float(np.mean(agreements)), 6),
        "sc_auroc": round(sc_auroc, 6),
        "total_time_seconds": round(total_time, 1),
        "timestamp": datetime.now().isoformat(),
        "note": f"Fair comparison: same temp={TEMPERATURE} as main method",
    }

    with open(OUTPUT_DIR / "fair_sc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
