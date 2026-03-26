# -*- coding: utf-8 -*-
"""
MMLU Entropy 실험 (Qwen / Llama 지원)
======================================
GPT 10차 자문 패키지 S: 도메인 확장 + Llama 일반화 보강

목적:
1. MMLU(일반지식 MCQ)에서 레이어별 entropy 분석
2. Qwen/Llama 양 모델 지원
3. 기존 normed/unnormed + scale 측정 파이프라인 재사용

사용법:
  python run_mmlu_entropy.py --model qwen --num_samples 1000
  python run_mmlu_entropy.py --model llama --num_samples 1000
  python run_mmlu_entropy.py --model qwen --num_samples 1000 --resume EXP_DIR

설정: 기존 실험과 동일 (max_tokens=1024, temp=0.3, seed=42)
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
# Per-sample timeout
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
# MMLU 데이터 로드
# ============================================================================

def load_mmlu_dataset(num_samples: int) -> List[Dict]:
    """MMLU 데이터셋 로드 (cais/mmlu, all subjects)"""
    from datasets import load_dataset

    log("DATA", "MMLU 데이터셋 로딩 중...")
    dataset = load_dataset("cais/mmlu", "all", split="test")

    log("DATA", f"MMLU 전체 크기: {len(dataset)}")

    # seed 고정 샘플링
    np.random.seed(SEED)
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    indices.sort()

    samples = []
    for idx in indices:
        item = dataset[int(idx)]
        choices = item['choices']
        answer_idx = item['answer']  # 0-3 정수

        # A/B/C/D 라벨
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
    log("DATA", f"MMLU {len(samples)}개 샘플 로드 (전체에서 무작위 샘플링)")
    return samples

# ============================================================================
# MCQ 답변 추출
# ============================================================================

def extract_mcq_answer(text: str) -> Optional[str]:
    """MCQ(A/B/C/D) 답변 추출"""
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
    # fallback: 마지막 단독 A/B/C/D
    letters = re.findall(r'\b([A-Da-d])\b', text)
    if letters:
        return letters[-1].upper()
    return None

# ============================================================================
# 프롬프트 생성
# ============================================================================

def make_mmlu_prompt(tokenizer, sample: Dict) -> str:
    """MMLU용 프롬프트"""
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
# Normed/Unnormed Entropy + Scale 측정 (기존 파이프라인 동일)
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
    norm = model.model.norm  # RMSNorm

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
            with torch.no_grad():
                # token_idx=0: hidden shape [1, prompt_len, hidden] → 개별 처리 (메모리 안전)
                # token_idx>0: hidden shape [1, 1, hidden] → 배치 처리 (28 layers 한번에)
                first_hidden = token_hidden_states[1] if len(token_hidden_states) > 1 else None
                if first_hidden is None:
                    continue
                is_multi_pos = first_hidden.shape[1] > 1

                if is_multi_pos:
                    # 첫 토큰: 프롬프트 전체 포함, 개별 처리 (기존 방식 유지)
                    for layer_idx in range(num_layers):
                        if layer_idx + 1 < len(token_hidden_states):
                            hidden = token_hidden_states[layer_idx + 1]
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
                            layer_unnormed_ent[layer_idx].append(ent_u / max_ent)
                            layer_normed_ent[layer_idx].append(ent_n / max_ent)
                            layer_h_norm[layer_idx].append(h_n)
                            layer_wh_norm[layer_idx].append(wh_norm_scaled)
                            layer_wh_rms[layer_idx].append(wh_rms)
                            layer_logit_std[layer_idx].append(logit_std_val)
                            layer_logit_max[layer_idx].append(logit_max_val)
                            layer_logit_margin[layer_idx].append(logit_margin_val)
                else:
                    # 이후 토큰: [1, 1, hidden] → 28개 레이어 배치 처리
                    hiddens = []
                    for li in range(num_layers):
                        if li + 1 < len(token_hidden_states):
                            hiddens.append(token_hidden_states[li + 1])
                    if not hiddens:
                        continue
                    all_hidden = torch.cat(hiddens, dim=0)  # [L, 1, hidden]
                    all_hidden_fp = all_hidden.to(lm_head.weight.dtype)
                    actual_layers = all_hidden_fp.shape[0]

                    # h_norm: 각 레이어의 [1, 1, hidden] norm (기존과 동일)
                    h_norms = torch.norm(all_hidden_fp.view(actual_layers, -1), p=2, dim=-1)

                    # 배치 lm_head (unnormed)
                    logits_u_all = lm_head(all_hidden_fp)  # [L, 1, vocab]
                    logits_f_all = logits_u_all.float().squeeze(1)  # [L, vocab]

                    # wh_norm: 배치 (numerically stable)
                    m_all = logits_f_all.abs().amax(dim=-1, keepdim=True)  # [L, 1]
                    m_all_safe = m_all.clamp(min=1e-12)
                    wh_norm_all = (m_all.squeeze(-1) * torch.norm(logits_f_all / m_all_safe, p=2, dim=-1))

                    # wh_rms, logit_std, logit_max, logit_margin: 배치
                    wh_rms_all = torch.sqrt(torch.mean(logits_f_all ** 2, dim=-1))
                    logit_std_all = torch.std(logits_f_all, dim=-1)
                    logit_max_all = logits_f_all.amax(dim=-1)
                    top2_all = torch.topk(logits_f_all, k=2, dim=-1)
                    logit_margin_all = top2_all.values[:, 0] - top2_all.values[:, 1]

                    # entropy (unnormed): 배치
                    probs_u_all = torch.softmax(logits_f_all, dim=-1).clamp(min=1e-10)
                    ent_u_all = (-probs_u_all * torch.log(probs_u_all)).sum(dim=-1)

                    # 배치 norm + lm_head (normed)
                    all_normed = norm(all_hidden_fp)
                    logits_n_all = lm_head(all_normed).float().squeeze(1)  # [L, vocab]
                    probs_n_all = torch.softmax(logits_n_all, dim=-1).clamp(min=1e-10)
                    ent_n_all = (-probs_n_all * torch.log(probs_n_all)).sum(dim=-1)

                    max_ent = float(np.log(logits_f_all.shape[-1]))

                    for li in range(actual_layers):
                        layer_unnormed_ent[li].append(float(ent_u_all[li].cpu().item()) / max_ent)
                        layer_normed_ent[li].append(float(ent_n_all[li].cpu().item()) / max_ent)
                        layer_h_norm[li].append(float(h_norms[li].cpu().item()))
                        layer_wh_norm[li].append(float(wh_norm_all[li].cpu().item()))
                        layer_wh_rms[li].append(float(wh_rms_all[li].cpu().item()))
                        layer_logit_std[li].append(float(logit_std_all[li].cpu().item()))
                        layer_logit_max[li].append(float(logit_max_all[li].cpu().item()))
                        layer_logit_margin[li].append(float(logit_margin_all[li].cpu().item()))

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
# Cohen's d
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

def run_experiment(model_key: str, num_samples: int, resume_path: str = None):
    start_time = datetime.now()
    model_config = MODEL_CONFIGS[model_key]
    model_name = model_config["name"]
    model_short = model_config["short"]

    # 실험 디렉토리
    if resume_path:
        exp_dir = Path(resume_path)
        exp_id = exp_dir.name
        log("RESUME", f"기존 실험 재개: {exp_id}")
    else:
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        exp_id = f"EXP_{timestamp}_mmlu_{model_short}"
        exp_dir = (PROJECT_ROOT / "PoT_Experiment_Entropy_Attention_Extraction_Experiment"
                   / "experiments" / "31_MMLU_Domain_Extension" / exp_id)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "data").mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print(f" MMLU Entropy Experiment")
    print(f" Model: {model_name} | Samples: {num_samples}")
    print(f" Experiment: {exp_id}")
    if resume_path:
        print(f" *** RESUMING from checkpoint ***")
    print(f"{'='*70}\n", flush=True)

    # 데이터 로드
    samples = load_mmlu_dataset(num_samples)

    # 모델 로드
    from transformers import AutoModelForCausalLM, AutoTokenizer
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

    # Resume
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
            prompt = make_mmlu_prompt(tokenizer, sample)
            output_text, layer_result, n_tokens = compute_dual_entropy_with_scale(
                model, tokenizer, prompt
            )

            # 정답 비교
            gt = sample["answer_key"]
            pred = extract_mcq_answer(output_text)
            is_correct = (pred == gt) if pred else False

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
                "subject": sample.get("subject", "unknown"),
                "layer_data": {str(k): v for k, v in layer_result.items()},
                "num_tokens": n_tokens,
            })

        except Exception as e:
            log("ERROR", f"idx {idx}: {e}")
            traceback.print_exc()
            results.append({"idx": idx, "error": str(e), "is_correct": False})

        # 진행 상황
        elapsed = datetime.now() - inference_start
        done = idx - start_idx + 1
        if done % 5 == 0 or idx == start_idx:
            rate = done / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
            remaining = (len(samples) - idx - 1) / rate if rate > 0 else 0
            acc = correct_count / (idx + 1) * 100
            log("PROGRESS", f"{idx+1}/{len(samples)} | Acc: {acc:.1f}% | "
                f"ETA: {timedelta(seconds=int(remaining))}")

        # 체크포인트 (25개마다)
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
    n_incorrect_total = len(samples) - correct_count

    print(f"\n{'='*120}")
    print(f" 정답률: {accuracy:.1f}% ({correct_count}/{len(samples)}) | 오답: {n_incorrect_total}")
    print(f"{'='*120}")

    # 레이어별 분석
    header = (f"{'Layer':>6} | {'N_Ent d':>9} {'U_Ent d':>9} | "
              f"{'h_norm d':>9} {'Wh_rms d':>9} | "
              f"{'N_Ent p':>10} {'U_Ent p':>10}")
    print(header)
    print("-" * 100, flush=True)

    layer_analysis = []
    normed_best_d = 0
    normed_best_layer = 0
    unnormed_best_d = 0
    unnormed_best_layer = 0

    for li in range(num_layers):
        cn = correct_data["normed_entropy"][li]
        cu = correct_data["unnormed_entropy"][li]
        ign = incorrect_data["normed_entropy"][li]
        igu = incorrect_data["unnormed_entropy"][li]
        ch = correct_data["h_norm"][li]
        ih = incorrect_data["h_norm"][li]
        c_wh_rms = correct_data["wh_rms"][li]
        i_wh_rms = incorrect_data["wh_rms"][li]

        if not cn or not ign:
            continue

        n_d = compute_cohens_d(ign, cn)
        u_d = compute_cohens_d(igu, cu)
        h_d = compute_cohens_d(ih, ch)
        wh_rms_d = compute_cohens_d(i_wh_rms, c_wh_rms)

        _, n_p = stats.ttest_ind(cn, ign)
        _, u_p = stats.ttest_ind(cu, igu)
        _, h_p = stats.ttest_ind(ch, ih)
        _, wh_rms_p = stats.ttest_ind(c_wh_rms, i_wh_rms)

        # 상관
        all_u = cu + igu
        all_h = ch + ih
        if len(all_u) > 2 and np.std(all_u) > 0 and np.std(all_h) > 0:
            corr_ue_h, _ = stats.pearsonr(all_u, all_h)
        else:
            corr_ue_h = 0.0

        entry = {
            "layer": li,
            "normed_d": float(n_d),
            "unnormed_d": float(u_d),
            "h_norm_d": float(h_d),
            "wh_rms_d": float(wh_rms_d),
            "normed_p": float(n_p),
            "unnormed_p": float(u_p),
            "h_norm_p": float(h_p),
            "wh_rms_p": float(wh_rms_p),
            "corr_unnormed_h": float(corr_ue_h),
            "normed_correct_mean": float(np.mean(cn)),
            "normed_incorrect_mean": float(np.mean(ign)),
            "unnormed_correct_mean": float(np.mean(cu)),
            "unnormed_incorrect_mean": float(np.mean(igu)),
            "h_norm_correct_mean": float(np.mean(ch)),
            "h_norm_incorrect_mean": float(np.mean(ih)),
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
              f"{h_d:>+9.4f} {wh_rms_d:>+9.4f} | "
              f"{n_p:>9.2e}{n_sig} {u_p:>9.2e}{u_sig}",
              flush=True)

    # 요약
    optimal_match = normed_best_layer == unnormed_best_layer

    print(f"\n{'='*70}")
    print(f" RESULTS SUMMARY (MMLU, {model_short})")
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
        "dataset": "MMLU",
        "config": {
            "model_name": model_name,
            "model_short": model_short,
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
    parser = argparse.ArgumentParser(description="MMLU Entropy Experiment (Qwen / Llama)")
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen", "llama", "mistral"],
                        help="Model to use: qwen or llama")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples (default: 1000)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to existing experiment directory to resume")
    args = parser.parse_args()

    try:
        run_experiment(model_key=args.model, num_samples=args.num_samples,
                      resume_path=args.resume)
    except KeyboardInterrupt:
        print("\n[INTERRUPT] 중단됨", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"\n[FATAL] {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
