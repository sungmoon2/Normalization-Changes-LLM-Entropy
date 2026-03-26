# -*- coding: utf-8 -*-
"""
Normed Entropy 난이도별 재실행 + Scale 가설 검증
================================================
GPT 5.2 PRO 3차 자문 우선순위 A + B 통합 실험

목적:
1. (A) normed entropy로 Easy/Hard/ARC 재실행 → "조건별 L_opt 이동"이 normed에서도 유지되는지
2. (B) ‖h‖, ‖Wh‖ 동시 저장 → unnormed entropy 신호가 scale인지 확정

모드:
  --mode easy   : competition_math Level 1-2 (500샘플)
  --mode hard   : competition_math Level 4-5 (500샘플)
  --mode arc    : ARC Easy+Hard (500샘플)
  --mode combo  : competition_math Level 1-5 (500샘플, 빠른 검증용)

설정: COMBO001과 동일 (max_tokens=1024, temp=0.3, seed=42)
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
from typing import Optional, Tuple, Dict, Any, List
import argparse
import traceback
from transformers import StoppingCriteria, StoppingCriteriaList

# ============================================================================
# Per-sample timeout (generate 무한 hang 방지)
# ============================================================================

class MaxTimeCriteria(StoppingCriteria):
    """max_time 초 초과 시 generation 중단"""
    def __init__(self, max_time_seconds: float = 300.0):
        self.max_time = max_time_seconds
        self.start_time = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.start_time is None:
            self.start_time = time.time()
        return time.time() - self.start_time > self.max_time

# ============================================================================
# 설정
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
SEED = 42

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

# ============================================================================
# 로깅
# ============================================================================

def log(tag, msg):
    print(f"[{tag}] {msg}", flush=True)

# ============================================================================
# Math 답변 추출/비교
# ============================================================================

def extract_boxed_content(text: str) -> Optional[str]:
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

def extract_math_answer(text: str) -> Optional[str]:
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

def normalize_answer(answer: str) -> str:
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
        except:
            pass
    return answer.strip()

def compare_math_answers(model_answer: str, ground_truth: str) -> bool:
    model_norm = normalize_answer(model_answer)
    gt_norm = normalize_answer(ground_truth)
    if model_norm == gt_norm:
        return True
    try:
        return abs(float(model_norm) - float(gt_norm)) < 1e-6
    except:
        return False

# ============================================================================
# ARC 답변 추출
# ============================================================================

def extract_arc_answer(text: str) -> Optional[str]:
    if not text:
        return None
    patterns = [
        r"[Tt]he answer is\s*[:\s]*\(?([A-Da-d])\)?",
        r"[Aa]nswer\s*[:\s]*\(?([A-Da-d])\)?",
        r"[Tt]herefore,?\s*(?:the answer is\s*)?\(?([A-Da-d])\)?",
        r"\b([A-Da-d])\s*\.\s*$",
        r"[Oo]ption\s+([A-Da-d])\s+is\s+correct",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    letters = re.findall(r'\b([A-Da-d])\b', text)
    if letters:
        return letters[-1].upper()
    return None

# ============================================================================
# 데이터 로드
# ============================================================================

def load_math_by_difficulty(difficulty: str, num_samples: int) -> List[Dict]:
    from datasets import load_dataset

    if difficulty == "easy":
        levels = [1, 2]
        samples_per_level = num_samples // 2
    elif difficulty == "hard":
        levels = [4, 5]
        samples_per_level = num_samples // 2
    elif difficulty == "combo":
        levels = [1, 2, 3, 4, 5]
        samples_per_level = num_samples // 5
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")

    np.random.seed(SEED)
    dataset = load_dataset("qwedsacf/competition_math", split="train")

    level_indices = {level: [] for level in levels}
    for idx, item in enumerate(dataset):
        level_str = item.get("level", "")
        for level in levels:
            if f"Level {level}" in level_str:
                level_indices[level].append(idx)
                break

    samples = []
    for level in levels:
        indices = level_indices[level]
        n = min(samples_per_level, len(indices))
        selected = np.random.choice(indices, size=n, replace=False)
        for idx in selected:
            item = dataset[int(idx)]
            samples.append({
                "type": "math",
                "problem": item["problem"],
                "solution": item["solution"],
                "level": item.get("level", "Unknown"),
            })

    np.random.shuffle(samples)
    log("DATA", f"Math ({difficulty}) {len(samples)}개 샘플 로드")
    return samples


def load_arc_dataset(num_samples: int) -> List[Dict]:
    from datasets import load_dataset

    samples_per_diff = num_samples // 2
    samples = []

    for difficulty in ["Easy", "Challenge"]:
        subset_name = f"ARC-{difficulty}"
        dataset = load_dataset("allenai/ai2_arc", subset_name, split="train")
        dataset = dataset.shuffle(seed=SEED)
        selected = dataset.select(range(min(samples_per_diff, len(dataset))))

        for item in selected:
            answer_key = item['answerKey']
            choices = item['choices']
            try:
                answer_idx = choices['label'].index(answer_key)
                answer_text = choices['text'][answer_idx]
            except ValueError:
                answer_text = answer_key

            samples.append({
                "type": "arc",
                "question": item['question'],
                "choices": choices,
                "answer_key": answer_key,
                "answer_text": answer_text,
                "difficulty": "Easy" if difficulty == "Easy" else "Hard",
            })

    random.shuffle(samples)
    log("DATA", f"ARC {len(samples)}개 샘플 로드")
    return samples

# ============================================================================
# 프롬프트 생성
# ============================================================================

def make_prompt(tokenizer, sample: Dict) -> str:
    if sample["type"] == "math":
        messages = [
            {"role": "system", "content": "You are a helpful math assistant. Solve problems step by step."},
            {"role": "user", "content": f"Solve: {sample['problem']}\n\nLet's think step by step."}
        ]
    else:  # arc
        choice_str = "\n".join(
            f"{label}. {text}"
            for label, text in zip(sample['choices']['label'], sample['choices']['text'])
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant that solves science questions step by step."},
            {"role": "user", "content": (
                f"Solve this multiple choice question step by step.\n\n"
                f"Question: {sample['question']}\n\n{choice_str}\n\n"
                f"Think through each option carefully, then end with "
                f"\"The answer is [LETTER]\" where LETTER is A, B, C, or D."
            )}
        ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def check_answer(sample: Dict, output_text: str) -> Tuple[bool, Optional[str], str]:
    if sample["type"] == "math":
        gt = extract_math_answer(sample["solution"])
        pred = extract_math_answer(output_text)
        correct = compare_math_answers(pred, gt) if pred and gt else False
        return correct, pred, gt
    else:  # arc
        gt = sample["answer_key"]
        pred = extract_arc_answer(output_text)
        correct = (pred == gt) if pred else False
        return correct, pred, gt

# ============================================================================
# 핵심: Normed/Unnormed Entropy + Scale 측정
# ============================================================================

def compute_dual_entropy_with_scale(model, tokenizer, prompt, max_new_tokens=1024, temperature=0.3):
    """
    생성 중 각 토큰의 각 레이어에서:
    1. normed entropy (RMSNorm 적용)
    2. unnormed entropy (RMSNorm 미적용)
    3. ‖h‖ (hidden state L2 norm)
    4. ‖Wh‖ (logit vector L2 norm, unnormed 기준)
    동시 계산.
    """
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

    num_layers = model.config.num_hidden_layers  # 28
    lm_head = model.lm_head
    norm = model.model.norm  # RMSNorm

    # 수집: 레이어별, 토큰별 값
    layer_normed_ent = {i: [] for i in range(num_layers)}
    layer_unnormed_ent = {i: [] for i in range(num_layers)}
    layer_h_norm = {i: [] for i in range(num_layers)}
    layer_wh_norm = {i: [] for i in range(num_layers)}
    layer_wh_rms = {i: [] for i in range(num_layers)}
    layer_logit_std = {i: [] for i in range(num_layers)}
    layer_logit_max = {i: [] for i in range(num_layers)}
    layer_logit_margin = {i: [] for i in range(num_layers)}

    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
        for token_idx, token_hidden_states in enumerate(outputs.hidden_states):
            for layer_idx in range(num_layers):
                if layer_idx + 1 < len(token_hidden_states):
                    hidden = token_hidden_states[layer_idx + 1]  # [1, 1, hidden_dim]

                    with torch.no_grad():
                        hidden_fp = hidden.to(lm_head.weight.dtype)

                        # ‖h‖: hidden state norm
                        h_norm = float(torch.norm(hidden_fp, p=2).cpu().item())

                        # Unnormed logits
                        logits_u = lm_head(hidden_fp)

                        # (a) Scaled L2 norm (overflow-safe, mathematically equivalent to L2)
                        logits_f = logits_u.float()  # upcast to float32 from float16
                        m = logits_f.abs().max()
                        if m > 0:
                            wh_norm_scaled = float((m * torch.norm(logits_f / m, p=2)).cpu().item())
                        else:
                            wh_norm_scaled = 0.0

                        # (b) RMS of logits (dimension-independent)
                        wh_rms = float(torch.sqrt(torch.mean(logits_f ** 2)).cpu().item())

                        # (c) Logit statistics
                        logit_std = float(torch.std(logits_f).cpu().item())
                        logit_max = float(logits_f.max().cpu().item())
                        top2 = torch.topk(logits_f.view(-1), k=2)
                        logit_margin = float((top2.values[0] - top2.values[1]).cpu().item())

                        probs_u = torch.softmax(logits_u.float(), dim=-1).clamp(min=1e-10)
                        ent_u = float((-probs_u * torch.log(probs_u)).sum(dim=-1)[0, 0].cpu().item())

                        # Normed logits
                        hidden_normed = norm(hidden_fp)
                        logits_n = lm_head(hidden_normed)
                        probs_n = torch.softmax(logits_n.float(), dim=-1).clamp(min=1e-10)
                        ent_n = float((-probs_n * torch.log(probs_n)).sum(dim=-1)[0, 0].cpu().item())

                        # 정규화
                        max_ent = float(np.log(logits_u.shape[-1]))
                        layer_unnormed_ent[layer_idx].append(ent_u / max_ent)
                        layer_normed_ent[layer_idx].append(ent_n / max_ent)
                        layer_h_norm[layer_idx].append(h_norm)
                        layer_wh_norm[layer_idx].append(wh_norm_scaled)
                        layer_wh_rms[layer_idx].append(wh_rms)
                        layer_logit_std[layer_idx].append(logit_std)
                        layer_logit_max[layer_idx].append(logit_max)
                        layer_logit_margin[layer_idx].append(logit_margin)

    # 레이어별 평균
    result = {}
    for li in range(num_layers):
        if layer_normed_ent[li]:
            result[li] = {
                "normed_entropy": float(np.mean(layer_normed_ent[li])),
                "unnormed_entropy": float(np.mean(layer_unnormed_ent[li])),
                "h_norm": float(np.mean(layer_h_norm[li])),
                "wh_norm": float(np.mean(layer_wh_norm[li])),
                "wh_rms": float(np.mean(layer_wh_rms[li])),
                "logit_std": float(np.mean(layer_logit_std[li])),
                "logit_max": float(np.mean(layer_logit_max[li])),
                "logit_margin": float(np.mean(layer_logit_margin[li])),
            }

    return output_text, result, len(generated_ids)

# ============================================================================
# Cohen's d (부호 통일: incorrect - correct, 양수 = 오답이 더 높음)
# ============================================================================

def compute_cohens_d(incorrect_vals, correct_vals):
    n1, n2 = len(incorrect_vals), len(correct_vals)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1 = np.var(incorrect_vals, ddof=1)
    var2 = np.var(correct_vals, ddof=1)
    pooled_std = np.sqrt((var1 + var2) / 2)
    if pooled_std < 1e-12:
        return 0.0
    return (np.mean(incorrect_vals) - np.mean(correct_vals)) / pooled_std

# ============================================================================
# 메인 실험
# ============================================================================

def run_experiment(mode: str, num_samples: int, resume_path: str = None):
    start_time = datetime.now()

    # Resume 또는 새 실험
    if resume_path:
        exp_dir = Path(resume_path)
        exp_id = exp_dir.name
        log("RESUME", f"기존 실험 재개: {exp_id}")
    else:
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        exp_id = f"EXP_{timestamp}_normed_{mode}"
        exp_dir = (PROJECT_ROOT / "PoT_Experiment_Entropy_Attention_Extraction_Experiment"
                   / "experiments" / "23_Normed_Difficulty_Analysis" / exp_id)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "data").mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print(f" Normed Entropy by Difficulty + Scale Verification")
    print(f" Mode: {mode} | Samples: {num_samples}")
    print(f" Experiment: {exp_id}")
    if resume_path:
        print(f" *** RESUMING from checkpoint ***")
    print(f"{'='*70}\n", flush=True)

    # 데이터 로드
    if mode == "arc":
        samples = load_arc_dataset(num_samples)
    else:
        samples = load_math_by_difficulty(mode, num_samples)

    # 모델 로드
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    log("MODEL", f"로딩 중: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    num_layers = model.config.num_hidden_layers
    log("MODEL", f"로드 완료: {num_layers} layers")

    # 수집
    metrics = ["normed_entropy", "unnormed_entropy", "h_norm", "wh_norm",
               "wh_rms", "logit_std", "logit_max", "logit_margin"]
    correct_data = {m: {i: [] for i in range(num_layers)} for m in metrics}
    incorrect_data = {m: {i: [] for i in range(num_layers)} for m in metrics}

    results = []
    correct_count = 0
    start_idx = 0

    # Resume: 기존 데이터 복원
    if resume_path:
        ckpt_file = exp_dir / "data" / "checkpoint.json"
        sr_file = exp_dir / "data" / "sample_results.json"
        if ckpt_file.exists() and sr_file.exists():
            with open(ckpt_file, 'r') as f:
                ckpt = json.load(f)
            with open(sr_file, 'r') as f:
                prev_results = json.load(f)

            start_idx = ckpt["checkpoint_idx"]
            correct_count = ckpt["correct_count"]
            results = prev_results

            # correct_data / incorrect_data 복원
            for r in prev_results:
                if "error" in r or "layer_data" not in r:
                    continue
                is_correct = r["is_correct"]
                for li_str, layer_vals in r["layer_data"].items():
                    li = int(li_str)
                    for m in metrics:
                        if m in layer_vals:
                            if is_correct:
                                correct_data[m][li].append(layer_vals[m])
                            else:
                                incorrect_data[m][li].append(layer_vals[m])

            log("RESUME", f"checkpoint_idx={start_idx}, correct_count={correct_count}, "
                f"prev_results={len(prev_results)}개 복원 완료")
        else:
            log("RESUME", "checkpoint 파일 없음, 처음부터 시작")

    inference_start = datetime.now()

    for idx in range(start_idx, len(samples)):
        sample = samples[idx]
        try:
            prompt = make_prompt(tokenizer, sample)
            output_text, layer_result, n_tokens = compute_dual_entropy_with_scale(
                model, tokenizer, prompt
            )
            is_correct, pred, gt = check_answer(sample, output_text)

            if is_correct:
                correct_count += 1

            for li in range(num_layers):
                if li in layer_result:
                    for m in metrics:
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
                "layer_data": {str(k): v for k, v in layer_result.items()},
                "num_tokens": n_tokens,
            })

        except Exception as e:
            log("ERROR", f"idx {idx}: {e}")
            traceback.print_exc()
            results.append({"idx": idx, "error": str(e), "is_correct": False})

        # 진행 상황
        elapsed = datetime.now() - inference_start
        if (idx + 1) % 5 == 0 or idx == 0:
            rate = (idx + 1) / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
            remaining = (len(samples) - idx - 1) / rate if rate > 0 else 0
            acc = correct_count / (idx + 1) * 100
            log("PROGRESS", f"{idx+1}/{len(samples)} | Acc: {acc:.1f}% | "
                f"ETA: {timedelta(seconds=int(remaining))}")

        # 체크포인트
        if (idx + 1) % 25 == 0:
            ckpt = {"checkpoint_idx": idx + 1, "correct_count": correct_count}
            with open(exp_dir / "data" / "checkpoint.json", 'w', encoding='utf-8') as f:
                json.dump(ckpt, f, indent=2)
            with open(exp_dir / "data" / "sample_results.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # ========================================================================
    # 분석
    # ========================================================================
    total_time = datetime.now() - start_time
    accuracy = correct_count / len(samples) * 100
    n_correct_total = correct_count
    n_incorrect_total = len(samples) - correct_count

    print(f"\n{'='*120}")
    print(f" 정답률: {accuracy:.1f}% ({correct_count}/{len(samples)}) | 오답: {n_incorrect_total}")
    print(f"{'='*120}")

    # 레이어별 분석
    header = (f"{'Layer':>6} | {'N_Ent d':>9} {'U_Ent d':>9} | "
              f"{'h_norm d':>9} {'Wh_norm d':>9} {'Wh_rms d':>9} "
              f"{'lgStd d':>9} {'lgMax d':>9} {'lgMarg d':>9} | "
              f"{'N_Ent p':>10} {'U_Ent p':>10} | "
              f"{'Corr(Ue,h)':>10} {'Corr(Ue,wh)':>11} {'Corr(Ue,rms)':>12} {'Corr(Ue,mrg)':>12}")
    print(header)
    print("-" * 180, flush=True)

    layer_analysis = []
    normed_best_d = 0
    normed_best_layer = 0
    unnormed_best_d = 0
    unnormed_best_layer = 0

    # Collect all raw p-values for BH-FDR correction
    all_raw_pvalues = []   # list of (layer_index_in_analysis, metric_key, p_value)

    for li in range(num_layers):
        cn = correct_data["normed_entropy"][li]
        cu = correct_data["unnormed_entropy"][li]
        ign = incorrect_data["normed_entropy"][li]
        igu = incorrect_data["unnormed_entropy"][li]
        ch = correct_data["h_norm"][li]
        ih = incorrect_data["h_norm"][li]
        cwh = correct_data["wh_norm"][li]
        iwh = incorrect_data["wh_norm"][li]
        c_wh_rms = correct_data["wh_rms"][li]
        i_wh_rms = incorrect_data["wh_rms"][li]
        c_logit_std = correct_data["logit_std"][li]
        i_logit_std = incorrect_data["logit_std"][li]
        c_logit_max = correct_data["logit_max"][li]
        i_logit_max = incorrect_data["logit_max"][li]
        c_logit_margin = correct_data["logit_margin"][li]
        i_logit_margin = incorrect_data["logit_margin"][li]

        if not cn or not ign:
            continue

        # Discriminability (incorrect - correct)
        n_d = compute_cohens_d(ign, cn)
        u_d = compute_cohens_d(igu, cu)
        h_d = compute_cohens_d(ih, ch)
        wh_d = compute_cohens_d(iwh, cwh)
        wh_rms_d = compute_cohens_d(i_wh_rms, c_wh_rms)
        logit_std_d = compute_cohens_d(i_logit_std, c_logit_std)
        logit_max_d = compute_cohens_d(i_logit_max, c_logit_max)
        logit_margin_d = compute_cohens_d(i_logit_margin, c_logit_margin)

        # p-values
        _, n_p = stats.ttest_ind(cn, ign)
        _, u_p = stats.ttest_ind(cu, igu)
        _, h_p = stats.ttest_ind(ch, ih)
        _, wh_p = stats.ttest_ind(cwh, iwh)
        _, wh_rms_p = stats.ttest_ind(c_wh_rms, i_wh_rms)
        _, logit_std_p = stats.ttest_ind(c_logit_std, i_logit_std)
        _, logit_max_p = stats.ttest_ind(c_logit_max, i_logit_max)
        _, logit_margin_p = stats.ttest_ind(c_logit_margin, i_logit_margin)

        # Store raw p-values for BH-FDR correction
        layer_entry_idx = len(layer_analysis)
        for pkey, pval in [("normed_p", n_p), ("unnormed_p", u_p),
                           ("h_norm_p", h_p), ("wh_norm_p", wh_p),
                           ("wh_rms_p", wh_rms_p), ("logit_std_p", logit_std_p),
                           ("logit_max_p", logit_max_p), ("logit_margin_p", logit_margin_p)]:
            all_raw_pvalues.append((layer_entry_idx, pkey, pval))

        # unnormed entropy vs h_norm 상관 (scale 가설 검증)
        all_u = cu + igu
        all_h = ch + ih
        if len(all_u) > 2 and np.std(all_u) > 0 and np.std(all_h) > 0:
            corr_ue_h, corr_ue_h_p = stats.pearsonr(all_u, all_h)
        else:
            corr_ue_h, corr_ue_h_p = 0.0, 1.0

        # unnormed entropy vs wh_norm 상관
        all_wh = cwh + iwh
        if len(all_u) > 2 and np.std(all_u) > 0 and np.std(all_wh) > 0:
            corr_ue_wh, corr_ue_wh_p = stats.pearsonr(all_u, all_wh)
        else:
            corr_ue_wh, corr_ue_wh_p = 0.0, 1.0

        # unnormed entropy vs wh_rms 상관
        all_wh_rms = c_wh_rms + i_wh_rms
        if len(all_u) > 2 and np.std(all_u) > 0 and np.std(all_wh_rms) > 0:
            corr_ue_rms, corr_ue_rms_p = stats.pearsonr(all_u, all_wh_rms)
        else:
            corr_ue_rms, corr_ue_rms_p = 0.0, 1.0

        # unnormed entropy vs logit_margin 상관
        all_margin = c_logit_margin + i_logit_margin
        if len(all_u) > 2 and np.std(all_u) > 0 and np.std(all_margin) > 0:
            corr_ue_margin, corr_ue_margin_p = stats.pearsonr(all_u, all_margin)
        else:
            corr_ue_margin, corr_ue_margin_p = 0.0, 1.0

        entry = {
            "layer": li,
            "normed_d": float(n_d),
            "unnormed_d": float(u_d),
            "h_norm_d": float(h_d),
            "wh_norm_d": float(wh_d),
            "wh_rms_d": float(wh_rms_d),
            "logit_std_d": float(logit_std_d),
            "logit_max_d": float(logit_max_d),
            "logit_margin_d": float(logit_margin_d),
            "normed_p": float(n_p),
            "unnormed_p": float(u_p),
            "h_norm_p": float(h_p),
            "wh_norm_p": float(wh_p),
            "wh_rms_p": float(wh_rms_p),
            "logit_std_p": float(logit_std_p),
            "logit_max_p": float(logit_max_p),
            "logit_margin_p": float(logit_margin_p),
            "corr_unnormed_h": float(corr_ue_h),
            "corr_unnormed_h_p": float(corr_ue_h_p),
            "corr_unnormed_wh": float(corr_ue_wh),
            "corr_unnormed_wh_p": float(corr_ue_wh_p),
            "corr_unnormed_rms": float(corr_ue_rms),
            "corr_unnormed_rms_p": float(corr_ue_rms_p),
            "corr_unnormed_margin": float(corr_ue_margin),
            "corr_unnormed_margin_p": float(corr_ue_margin_p),
            "normed_correct_mean": float(np.mean(cn)),
            "normed_incorrect_mean": float(np.mean(ign)),
            "unnormed_correct_mean": float(np.mean(cu)),
            "unnormed_incorrect_mean": float(np.mean(igu)),
            "h_norm_correct_mean": float(np.mean(ch)),
            "h_norm_incorrect_mean": float(np.mean(ih)),
            "wh_norm_correct_mean": float(np.mean(cwh)),
            "wh_norm_incorrect_mean": float(np.mean(iwh)),
            "wh_rms_correct_mean": float(np.mean(c_wh_rms)),
            "wh_rms_incorrect_mean": float(np.mean(i_wh_rms)),
            "logit_std_correct_mean": float(np.mean(c_logit_std)),
            "logit_std_incorrect_mean": float(np.mean(i_logit_std)),
            "logit_max_correct_mean": float(np.mean(c_logit_max)),
            "logit_max_incorrect_mean": float(np.mean(i_logit_max)),
            "logit_margin_correct_mean": float(np.mean(c_logit_margin)),
            "logit_margin_incorrect_mean": float(np.mean(i_logit_margin)),
            "n_correct": len(cn),
            "n_incorrect": len(ign),
        }
        layer_analysis.append(entry)

        if abs(n_d) > abs(normed_best_d):
            normed_best_d = n_d
            normed_best_layer = li
        if abs(u_d) > abs(unnormed_best_d):
            unnormed_best_d = u_d
            unnormed_best_layer = li

        n_sig = "**" if n_p < 0.05 else "  "
        u_sig = "**" if u_p < 0.05 else "  "

        print(f"{li:>6} | {n_d:>+9.4f} {u_d:>+9.4f} | "
              f"{h_d:>+9.4f} {wh_d:>+9.4f} {wh_rms_d:>+9.4f} "
              f"{logit_std_d:>+9.4f} {logit_max_d:>+9.4f} {logit_margin_d:>+9.4f} | "
              f"{n_p:>9.2e}{n_sig} {u_p:>9.2e}{u_sig} | "
              f"{corr_ue_h:>+10.4f} {corr_ue_wh:>+11.4f} {corr_ue_rms:>+12.4f} {corr_ue_margin:>+12.4f}",
              flush=True)

    # ---- BH-FDR correction across all layers and metrics ----
    if all_raw_pvalues:
        raw_ps = np.array([x[2] for x in all_raw_pvalues])
        n_tests = len(raw_ps)
        sorted_indices = np.argsort(raw_ps)
        adjusted = np.empty(n_tests)
        # Benjamini-Hochberg procedure
        for rank_pos, orig_idx in enumerate(sorted_indices):
            bh_rank = rank_pos + 1  # 1-based rank
            adjusted[orig_idx] = raw_ps[orig_idx] * n_tests / bh_rank
        # Enforce monotonicity (going from largest rank to smallest)
        reverse_sorted = sorted_indices[::-1]
        cummin = adjusted[reverse_sorted[0]]
        for orig_idx in reverse_sorted:
            if adjusted[orig_idx] < cummin:
                cummin = adjusted[orig_idx]
            else:
                adjusted[orig_idx] = cummin
        # Clip to [0, 1]
        adjusted = np.clip(adjusted, 0.0, 1.0)

        # Write adjusted p-values back into layer_analysis entries
        for (entry_idx, pkey, _), adj_p in zip(all_raw_pvalues, adjusted):
            adj_key = pkey.replace("_p", "_p_adj")
            layer_analysis[entry_idx][adj_key] = float(adj_p)

        log("FDR", f"BH-FDR correction applied: {n_tests} tests across {len(layer_analysis)} layers")

    # 요약
    optimal_match = normed_best_layer == unnormed_best_layer

    print(f"\n{'='*70}")
    print(f" RESULTS SUMMARY ({mode})")
    print(f"{'='*70}")
    print(f"  Accuracy: {accuracy:.1f}% ({correct_count}/{len(samples)})")
    print(f"  Normed optimal:   Layer {normed_best_layer} (d = {normed_best_d:+.6f})")
    print(f"  Unnormed optimal: Layer {unnormed_best_layer} (d = {unnormed_best_d:+.6f})")
    print(f"  Match: {'YES' if optimal_match else 'NO'}")
    print(f"  Time: {total_time}", flush=True)

    # 저장
    final_results = {
        "experiment_id": exp_id,
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "config": {
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "num_samples": len(results),
            "max_new_tokens": 1024,
            "temperature": 0.3,
            "seed": SEED,
        },
        "results": {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "incorrect_count": n_incorrect_total,
            "normed_optimal_layer": normed_best_layer,
            "normed_optimal_d": float(normed_best_d),
            "unnormed_optimal_layer": unnormed_best_layer,
            "unnormed_optimal_d": float(unnormed_best_d),
            "optimal_layer_match": optimal_match,
            "execution_time_seconds": total_time.total_seconds(),
        },
        "layer_analysis": layer_analysis,
    }

    with open(exp_dir / "final_results.json", 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    with open(exp_dir / "data" / "sample_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log("SAVE", f"저장 완료: {exp_dir}")
    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normed Entropy by Difficulty + Scale Verification")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["easy", "hard", "arc", "combo"],
                        help="easy: L1-2, hard: L4-5, arc: ARC, combo: L1-5")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of samples (default: 500)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to existing experiment directory to resume from checkpoint")
    args = parser.parse_args()

    try:
        run_experiment(mode=args.mode, num_samples=args.num_samples, resume_path=args.resume)
    except KeyboardInterrupt:
        print("\n[INTERRUPT] 중단됨", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"\n[FATAL] {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
