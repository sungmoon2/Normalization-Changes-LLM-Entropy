# -*- coding: utf-8 -*-
"""
Llama-3-8B-Instruct Normed Entropy 일반화 검증
===============================================
Qwen에서 발견한 패턴이 Llama에서도 재현되는지 검증

목적:
1. 모델 독립적 일반성 확인
2. Normed + ||h|| decomposition이 Llama에서도 작동하는지
3. 최적 레이어 이동 패턴 재현 여부

모드:
  --mode easy   : competition_math Level 1-2 (200샘플, 메모리 고려)
  --mode hard   : competition_math Level 4-5 (200샘플)
  --mode combo  : competition_math Level 1-5 (200샘플)

설정: max_tokens=1024, temp=0.3, seed=42 (기존과 동일)
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
import warnings
warnings.filterwarnings('ignore')

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
# Math 답변 추출/비교 (Qwen과 동일)
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
# 데이터 로드 (메모리 고려하여 샘플 수 축소)
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

# ============================================================================
# 프롬프트 생성 (Llama용 수정)
# ============================================================================

def make_prompt(tokenizer, sample: Dict) -> str:
    # Llama-3는 system role 지원
    messages = [
        {"role": "system", "content": "You are a helpful math assistant. Solve problems step by step."},
        {"role": "user", "content": f"Solve: {sample['problem']}\n\nLet's think step by step."}
    ]
    # Llama tokenizer의 chat template 사용
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def check_answer(sample: Dict, output_text: str) -> Tuple[bool, Optional[str], str]:
    gt = extract_math_answer(sample["solution"])
    pred = extract_math_answer(output_text)
    correct = compare_math_answers(pred, gt) if pred and gt else False
    return correct, pred, gt

# ============================================================================
# 핵심: Llama용 Normed/Unnormed Entropy + Scale 측정
# ============================================================================

def compute_dual_entropy_with_scale_llama(model, tokenizer, prompt, max_new_tokens=1024, temperature=0.3):
    """
    Llama-3-8B에서 entropy 계산
    주의: Llama-3는 32 layers
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
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
            pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id,
            stopping_criteria=StoppingCriteriaList([time_limit]),
        )

    generated_ids = outputs.sequences[0][input_length:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    num_layers = model.config.num_hidden_layers  # Llama-3: 32
    lm_head = model.lm_head

    # Llama-3도 RMSNorm 사용
    norm = model.model.norm

    # 수집: 레이어별, 토큰별 값
    layer_normed_ent = {i: [] for i in range(num_layers)}
    layer_unnormed_ent = {i: [] for i in range(num_layers)}
    layer_h_norm = {i: [] for i in range(num_layers)}
    layer_wh_norm = {i: [] for i in range(num_layers)}

    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
        for token_idx, token_hidden_states in enumerate(outputs.hidden_states):
            for layer_idx in range(num_layers):
                if layer_idx + 1 < len(token_hidden_states):
                    hidden = token_hidden_states[layer_idx + 1]  # [1, 1, hidden_dim]

                    with torch.no_grad():
                        hidden_fp = hidden.to(lm_head.weight.dtype)

                        # ||h||: hidden state norm
                        h_norm = float(torch.norm(hidden_fp, p=2).cpu().item())

                        # Unnormed logits
                        logits_u = lm_head(hidden_fp)

                        # Scaled L2 norm (overflow-safe)
                        logits_f = logits_u.float()
                        m = logits_f.abs().max()
                        if m > 0:
                            wh_norm_scaled = float((m * torch.norm(logits_f / m, p=2)).cpu().item())
                        else:
                            wh_norm_scaled = 0.0

                        probs_u = torch.softmax(logits_u.float(), dim=-1).clamp(min=1e-10)
                        ent_u = float((-probs_u * torch.log(probs_u)).sum(dim=-1).mean().cpu().item())

                        # Normed logits
                        hidden_normed = norm(hidden_fp)
                        logits_n = lm_head(hidden_normed)
                        probs_n = torch.softmax(logits_n.float(), dim=-1).clamp(min=1e-10)
                        ent_n = float((-probs_n * torch.log(probs_n)).sum(dim=-1).mean().cpu().item())

                        # 정규화
                        max_ent = float(np.log(logits_u.shape[-1]))
                        layer_unnormed_ent[layer_idx].append(ent_u / max_ent)
                        layer_normed_ent[layer_idx].append(ent_n / max_ent)
                        layer_h_norm[layer_idx].append(h_norm)
                        layer_wh_norm[layer_idx].append(wh_norm_scaled)

    # 레이어별 평균
    result = {}
    for li in range(num_layers):
        if layer_normed_ent[li]:
            result[li] = {
                "normed_entropy": float(np.mean(layer_normed_ent[li])),
                "unnormed_entropy": float(np.mean(layer_unnormed_ent[li])),
                "h_norm": float(np.mean(layer_h_norm[li])),
                "wh_norm": float(np.mean(layer_wh_norm[li])),
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

def run_llama_experiment(mode: str, num_samples: int, resume_path: str = None):
    start_time = datetime.now()

    if resume_path:
        exp_dir = Path(resume_path)
        exp_id = exp_dir.name
        log("RESUME", f"기존 실험 재개: {exp_id}")
    else:
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        exp_id = f"EXP_{timestamp}_llama_{mode}"
        exp_dir = (PROJECT_ROOT / "PoT_Experiment_Entropy_Attention_Extraction_Experiment"
                   / "experiments" / "26_Llama_Generalization" / exp_id)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "data").mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print(f" Llama-3-8B-Instruct Normed Entropy Generalization")
    print(f" Mode: {mode} | Samples: {num_samples}")
    print(f" Experiment: {exp_id}")
    print(f"{'='*70}\n", flush=True)

    # 데이터 로드
    samples = load_math_by_difficulty(mode, num_samples)

    # 모델 로드
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    log("MODEL", f"로딩 중: {model_name}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import os

        # HF token: 환경변수 있으면 사용, 없으면 캐시에서 로드 시도
        hf_token = os.environ.get("HF_TOKEN", None)
        token_arg = hf_token if hf_token else True

        # Llama 모델 로드
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=token_arg,
                padding_side='left',
            )
        except Exception:
            log("INFO", "token 인증 실패, local_files_only로 시도")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=True,
                padding_side='left',
            )

        # pad_token 설정 (Llama는 기본적으로 없음)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                token=token_arg,
                low_cpu_mem_usage=True,
            )
        except Exception:
            log("INFO", "token 인증 실패, local_files_only로 시도")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
        model.eval()

    except Exception as e:
        log("ERROR", f"모델 로드 실패: {e}")
        log("INFO", "HuggingFace token이 필요할 수 있습니다.")
        log("INFO", "huggingface-cli login 실행 또는 HF_TOKEN 환경변수 설정")
        sys.exit(1)

    num_layers = model.config.num_hidden_layers  # Llama-3: 32 layers
    log("MODEL", f"로드 완료: {num_layers} layers")

    # 수집
    metrics = ["normed_entropy", "unnormed_entropy", "h_norm", "wh_norm"]
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
            with open(ckpt_file, 'r', encoding='utf-8') as f:
                ckpt = json.load(f)
            with open(sr_file, 'r', encoding='utf-8') as f:
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
            output_text, layer_result, n_tokens = compute_dual_entropy_with_scale_llama(
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
              f"{'h_norm d':>9} {'Wh_norm d':>9} | "
              f"{'N_Ent p':>10} {'U_Ent p':>10} | "
              f"{'Corr(Ue,h)':>10} {'Corr(Ue,wh)':>11}")
    print(header)
    print("-" * 120, flush=True)

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
        cwh = correct_data["wh_norm"][li]
        iwh = incorrect_data["wh_norm"][li]

        if not cn or not ign:
            continue

        # Discriminability
        n_d = compute_cohens_d(ign, cn)
        u_d = compute_cohens_d(igu, cu)
        h_d = compute_cohens_d(ih, ch)
        wh_d = compute_cohens_d(iwh, cwh)

        # p-values
        _, n_p = stats.ttest_ind(cn, ign)
        _, u_p = stats.ttest_ind(cu, igu)
        _, h_p = stats.ttest_ind(ch, ih)
        _, wh_p = stats.ttest_ind(cwh, iwh)

        # 상관관계
        all_u = cu + igu
        all_h = ch + ih
        all_wh = cwh + iwh

        if len(all_u) > 2 and np.std(all_u) > 0 and np.std(all_h) > 0:
            corr_ue_h, _ = stats.pearsonr(all_u, all_h)
        else:
            corr_ue_h = 0.0

        if len(all_u) > 2 and np.std(all_u) > 0 and np.std(all_wh) > 0:
            corr_ue_wh, _ = stats.pearsonr(all_u, all_wh)
        else:
            corr_ue_wh = 0.0

        entry = {
            "layer": li,
            "normed_d": float(n_d),
            "unnormed_d": float(u_d),
            "h_norm_d": float(h_d),
            "wh_norm_d": float(wh_d),
            "normed_p": float(n_p),
            "unnormed_p": float(u_p),
            "h_norm_p": float(h_p),
            "wh_norm_p": float(wh_p),
            "corr_unnormed_h": float(corr_ue_h),
            "corr_unnormed_wh": float(corr_ue_wh),
            "normed_correct_mean": float(np.mean(cn)),
            "normed_incorrect_mean": float(np.mean(ign)),
            "unnormed_correct_mean": float(np.mean(cu)),
            "unnormed_incorrect_mean": float(np.mean(igu)),
            "h_norm_correct_mean": float(np.mean(ch)),
            "h_norm_incorrect_mean": float(np.mean(ih)),
            "wh_norm_correct_mean": float(np.mean(cwh)),
            "wh_norm_incorrect_mean": float(np.mean(iwh)),
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
              f"{h_d:>+9.4f} {wh_d:>+9.4f} | "
              f"{n_p:>9.2e}{n_sig} {u_p:>9.2e}{u_sig} | "
              f"{corr_ue_h:>+10.4f} {corr_ue_wh:>+11.4f}",
              flush=True)

    # 요약
    print(f"\n{'='*70}")
    print(f" RESULTS SUMMARY (Llama-3-8B {mode})")
    print(f"{'='*70}")
    print(f"  Model: {model_name}")
    print(f"  Layers: {num_layers}")
    print(f"  Accuracy: {accuracy:.1f}% ({correct_count}/{len(samples)})")
    print(f"  Normed optimal:   Layer {normed_best_layer} (d = {normed_best_d:+.6f})")
    print(f"  Unnormed optimal: Layer {unnormed_best_layer} (d = {unnormed_best_d:+.6f})")
    print(f"  Time: {total_time}", flush=True)

    # 저장
    final_results = {
        "experiment_id": exp_id,
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "num_layers": num_layers,
        "mode": mode,
        "config": {
            "model_name": model_name,
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
            "execution_time_seconds": total_time.total_seconds(),
        },
        "layer_analysis": layer_analysis,
    }

    with open(exp_dir / "final_results.json", 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    with open(exp_dir / "data" / "sample_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log("SAVE", f"저장 완료: {exp_dir}")

    # Qwen 결과와 비교
    print(f"\n{'='*70}")
    print(f" Qwen vs Llama 비교 ({mode})")
    print(f"{'='*70}")
    print(f"  이 결과를 Qwen2.5-7B 결과와 비교하여:")
    print(f"  - 최적 레이어 위치가 비슷한지")
    print(f"  - Normed vs Unnormed 패턴이 유지되는지")
    print(f"  - Decomposition 효과가 재현되는지")
    print(f"  확인이 필요합니다.")

    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama-3-8B Normed Entropy Generalization")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["easy", "hard", "combo"],
                        help="easy: L1-2, hard: L4-5, combo: L1-5")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Number of samples (default: 200, reduced for Llama)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to existing experiment directory to resume from checkpoint")
    args = parser.parse_args()

    try:
        run_llama_experiment(mode=args.mode, num_samples=args.num_samples, resume_path=args.resume)
    except KeyboardInterrupt:
        print("\n[INTERRUPT] 중단됨", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"\n[FATAL] {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)