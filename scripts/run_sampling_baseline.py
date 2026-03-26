# -*- coding: utf-8 -*-
"""
Sampling Baseline (Self-Consistency) 비교 실험
===============================================
GPT 9차 자문 Q5: "sampling-based UQ(self-consistency)랑 비교했냐?"

목적:
  Single-pass layer-wise entropy vs K-pass self-consistency AUROC 비교.
  동일 문항에 대해 K=5 응답을 생성하고 agreement rate를 UQ 신호로 사용.
  기존 실험의 per-sample entropy와 paired 비교.

모드:
  --mode hard  : competition_math Level 4-5 (200문항, K=5)
  --mode easy  : competition_math Level 1-2 (200문항, K=5)

설정:
  - 데이터셋 선택: seed=42 (기존 실험과 동일 문항 순서)
  - 생성: temp=0.7, do_sample=True, K=5 (SC 표준)
  - hidden_states: 불필요 (텍스트 출력만)
  - 기존 entropy: sample_results.json에서 로드 (paired)
"""

import os
import sys
import json
import random
import re
import time
import numpy as np
import torch
from collections import Counter
from datetime import datetime
from pathlib import Path
from scipy import stats as scipy_stats
from typing import Optional, Tuple, Dict, List
import argparse
import traceback
from transformers import StoppingCriteria, StoppingCriteriaList

# ============================================================================
# Per-sample timeout
# ============================================================================

class MaxTimeCriteria(StoppingCriteria):
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
EXP_BASE = PROJECT_ROOT / "PoT_Experiment_Entropy_Attention_Extraction_Experiment" / "experiments"
SEED = 42
K = 5           # self-consistency 샘플 수
SC_TEMP = 0.7   # SC 표준 temperature
NUM_QUESTIONS = 200  # 조건당 문항 수
CHECKPOINT_EVERY = 25

# 기존 실험 경로 (per-sample entropy 로드용)
EXISTING_EXPERIMENTS = {
    "hard": EXP_BASE / "23_Normed_Difficulty_Analysis" / "EXP_20260213_113717_normed_hard" / "data" / "sample_results.json",
    "easy": EXP_BASE / "23_Normed_Difficulty_Analysis" / "EXP_20260213_013643_normed_easy" / "data" / "sample_results.json",
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

def log(tag, msg):
    print(f"[{tag}] {msg}", flush=True)

# ============================================================================
# Math 답변 추출/비교 (기존 코드 재사용)
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
# 프롬프트 생성
# ============================================================================

def make_prompt(tokenizer, sample: Dict) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful math assistant. Solve problems step by step."},
        {"role": "user", "content": f"Solve: {sample['problem']}\n\nLet's think step by step."}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ============================================================================
# Self-Consistency 생성 (hidden_states 없음, 빠름)
# ============================================================================

def generate_k_responses(model, tokenizer, prompt, k=5, temperature=0.7, max_new_tokens=1024):
    """문항 1개에 대해 K개 응답 생성 (hidden_states 없이, 빠르게)"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    responses = []

    for sample_k in range(k):
        time_limit = MaxTimeCriteria(max_time_seconds=300.0)
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_hidden_states=False,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=StoppingCriteriaList([time_limit]),
                )
            generated_ids = outputs.sequences[0][input_length:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            num_tokens = len(generated_ids)
        except Exception as e:
            output_text = ""
            num_tokens = 0
            log("WARN", f"  k={sample_k} 생성 실패: {e}")

        responses.append({
            "k": sample_k,
            "text": output_text,
            "num_tokens": num_tokens,
        })

    return responses

# ============================================================================
# SC 메트릭 계산
# ============================================================================

def compute_sc_metrics(responses: List[Dict], ground_truth: str) -> Dict:
    """K개 응답에서 self-consistency 메트릭 계산"""
    answers = []
    for r in responses:
        pred = extract_math_answer(r["text"])
        normalized = normalize_answer(pred) if pred else "__NONE__"
        answers.append(normalized)

    gt_norm = normalize_answer(extract_math_answer(ground_truth) or "")

    # answer distribution
    counter = Counter(answers)
    k = len(answers)

    # agreement rate = max(counts) / K
    agreement_rate = max(counter.values()) / k

    # answer entropy (normalized by log(K))
    counts = np.array(list(counter.values()), dtype=float)
    probs = counts / counts.sum()
    raw_entropy = float(scipy_stats.entropy(probs))
    max_entropy = np.log(k)
    answer_entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0

    # majority vote
    majority_answer = counter.most_common(1)[0][0]

    # majority vote correctness
    if majority_answer == "__NONE__":
        sc_correct = False
    else:
        sc_correct = (majority_answer == gt_norm)
        if not sc_correct:
            try:
                sc_correct = abs(float(majority_answer) - float(gt_norm)) < 1e-6
            except:
                pass

    # per-response correctness
    per_k_correct = []
    for ans in answers:
        if ans == "__NONE__":
            per_k_correct.append(False)
        else:
            c = (ans == gt_norm)
            if not c:
                try:
                    c = abs(float(ans) - float(gt_norm)) < 1e-6
                except:
                    pass
            per_k_correct.append(c)

    return {
        "agreement_rate": agreement_rate,
        "answer_entropy": answer_entropy,
        "sc_correct": sc_correct,
        "majority_answer": majority_answer,
        "n_unique_answers": len(counter),
        "answer_distribution": dict(counter),
        "per_k_correct": per_k_correct,
        "per_k_answers": answers,
    }

# ============================================================================
# 기존 실험 per-sample entropy 로드
# ============================================================================

def load_existing_entropy(mode: str, num_questions: int) -> List[Dict]:
    """기존 sample_results.json에서 처음 num_questions개의 entropy 데이터 로드"""
    path = EXISTING_EXPERIMENTS.get(mode)
    if not path or not path.exists():
        log("WARN", f"기존 실험 결과 없음: {path}")
        return []

    with open(path, 'r', encoding='utf-8') as f:
        all_results = json.load(f)

    # idx 기준 정렬 후 처음 num_questions개
    valid = [r for r in all_results if 'error' not in r and 'layer_data' in r]
    valid.sort(key=lambda x: x.get('idx', 0))
    subset = valid[:num_questions]

    log("DATA", f"기존 entropy {len(subset)}개 로드 (from {path.name})")
    return subset

# ============================================================================
# 비교 분석 (후처리, GPU 불필요)
# ============================================================================

def run_comparison_analysis(sc_results: List[Dict], existing_entropy: List[Dict], num_layers: int) -> Dict:
    """SC 메트릭 vs layer-wise entropy AUROC 비교"""
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold

    # 매칭: idx 기준
    n = min(len(sc_results), len(existing_entropy))
    if n == 0:
        return {"error": "no matched data"}

    # labels: 기존 single-pass(temp=0.3) 결과의 is_correct
    labels = np.array([existing_entropy[i]["is_correct"] for i in range(n)], dtype=int)
    n_correct = int(labels.sum())
    n_incorrect = n - n_correct

    if n_correct < 2 or n_incorrect < 2:
        return {"error": f"insufficient class balance: correct={n_correct}, incorrect={n_incorrect}"}

    # SC confidence scores
    agreement_rates = np.array([sc_results[i]["sc_metrics"]["agreement_rate"] for i in range(n)])
    answer_entropies = np.array([sc_results[i]["sc_metrics"]["answer_entropy"] for i in range(n)])

    # Layer-wise entropy scores (기존 데이터)
    layer_unnormed = {}
    layer_normed = {}
    for li in range(num_layers):
        li_str = str(li)
        unnormed_vals = []
        normed_vals = []
        for i in range(n):
            ld = existing_entropy[i].get("layer_data", {})
            if li_str in ld:
                unnormed_vals.append(ld[li_str].get("unnormed_entropy", 0))
                normed_vals.append(ld[li_str].get("normed_entropy", 0))
            else:
                unnormed_vals.append(0)
                normed_vals.append(0)
        layer_unnormed[li] = np.array(unnormed_vals)
        layer_normed[li] = np.array(normed_vals)

    def safe_auroc(y, scores):
        try:
            return float(roc_auc_score(y, scores))
        except ValueError:
            return 0.5

    # --- 1. SC AUROC ---
    sc_auroc_agreement = safe_auroc(labels, agreement_rates)  # higher agreement → more likely correct
    sc_auroc_entropy = safe_auroc(labels, -answer_entropies)  # lower answer entropy → more likely correct

    # --- 2. Layer-wise entropy AUROC (best layer) ---
    unnormed_aurocs = {}
    normed_aurocs = {}
    for li in range(num_layers):
        unnormed_aurocs[li] = safe_auroc(labels, -layer_unnormed[li])
        normed_aurocs[li] = safe_auroc(labels, -layer_normed[li])

    best_unnormed_layer = max(unnormed_aurocs, key=unnormed_aurocs.get)
    best_normed_layer = max(normed_aurocs, key=normed_aurocs.get)

    # --- 3. Spearman correlation (SC agreement ↔ best layer entropy) ---
    spearman_unnormed = scipy_stats.spearmanr(agreement_rates, layer_unnormed[best_unnormed_layer])
    spearman_normed = scipy_stats.spearmanr(agreement_rates, layer_normed[best_normed_layer])

    # --- 4. Combined (SC + layer entropy) via cross-validated logistic regression ---
    combined_results = {}
    for combo_name, layer_entropy in [
        ("sc_plus_unnormed", layer_unnormed[best_unnormed_layer]),
        ("sc_plus_normed", layer_normed[best_normed_layer]),
    ]:
        X = np.column_stack([agreement_rates, layer_entropy])
        try:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            combined_scores = np.zeros(n)
            for tr, te in skf.split(X, labels):
                sc = StandardScaler()
                lr = LogisticRegression(random_state=SEED, max_iter=1000)
                lr.fit(sc.fit_transform(X[tr]), labels[tr])
                combined_scores[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
            combined_auroc = safe_auroc(labels, combined_scores)
        except Exception as e:
            combined_auroc = 0.5
            log("WARN", f"Combined {combo_name} 실패: {e}")

        combined_results[combo_name] = combined_auroc

    # --- 5. Risk-Coverage ---
    def compute_risk_coverage(confidence, labels, coverages=None):
        if coverages is None:
            coverages = np.arange(0.1, 1.01, 0.1)
        n_total = len(labels)
        sorted_idx = np.argsort(-confidence)
        results = []
        for cov in coverages:
            k_sel = max(1, int(n_total * cov))
            selected = sorted_idx[:k_sel]
            acc = float(labels[selected].mean())
            results.append({"coverage": float(cov), "accuracy": acc, "n_selected": k_sel})
        accs = np.array([r["accuracy"] for r in results])
        covs = np.array([r["coverage"] for r in results])
        auc_rc = float(np.trapz(accs, covs))
        return {"curve": results, "auc_rc": auc_rc}

    rc_sc = compute_risk_coverage(agreement_rates, labels)
    rc_unnormed = compute_risk_coverage(-layer_unnormed[best_unnormed_layer], labels)
    rc_normed = compute_risk_coverage(-layer_normed[best_normed_layer], labels)

    # --- 6. SC 정답률 통계 ---
    sc_correct_flags = np.array([sc_results[i]["sc_metrics"]["sc_correct"] for i in range(n)], dtype=int)
    sc_accuracy = float(sc_correct_flags.mean())

    # per-k accuracy (각 개별 응답의 정답률)
    per_k_accs = []
    for k_idx in range(K):
        k_correct = sum(1 for i in range(n)
                        if k_idx < len(sc_results[i]["sc_metrics"]["per_k_correct"])
                        and sc_results[i]["sc_metrics"]["per_k_correct"][k_idx])
        per_k_accs.append(k_correct / n)

    # original single-pass accuracy
    original_accuracy = float(labels.mean())

    return {
        "n_questions": n,
        "n_correct_original": n_correct,
        "n_incorrect_original": n_incorrect,
        "original_accuracy": original_accuracy,
        "sc_majority_vote_accuracy": sc_accuracy,
        "sc_per_k_accuracies": per_k_accs,

        "auroc_comparison": {
            "sc_agreement_rate": sc_auroc_agreement,
            "sc_answer_entropy": sc_auroc_entropy,
            "best_unnormed_layer": best_unnormed_layer,
            "best_unnormed_auroc": unnormed_aurocs[best_unnormed_layer],
            "best_normed_layer": best_normed_layer,
            "best_normed_auroc": normed_aurocs[best_normed_layer],
            "all_unnormed_aurocs": {str(k): v for k, v in unnormed_aurocs.items()},
            "all_normed_aurocs": {str(k): v for k, v in normed_aurocs.items()},
        },

        "combined_auroc": combined_results,

        "correlation": {
            "sc_vs_unnormed_spearman_r": float(spearman_unnormed.statistic),
            "sc_vs_unnormed_spearman_p": float(spearman_unnormed.pvalue),
            "sc_vs_normed_spearman_r": float(spearman_normed.statistic),
            "sc_vs_normed_spearman_p": float(spearman_normed.pvalue),
        },

        "risk_coverage": {
            "sc_agreement": rc_sc,
            f"unnormed_L{best_unnormed_layer}": rc_unnormed,
            f"normed_L{best_normed_layer}": rc_normed,
        },
    }

# ============================================================================
# 메인 실험 루프
# ============================================================================

def run_experiment(mode: str, resume_path: Optional[str] = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = EXP_BASE / "30_Sampling_Baseline" / f"EXP_{timestamp}_sampling_{mode}"
    if resume_path:
        exp_dir = Path(resume_path)
    exp_dir.mkdir(parents=True, exist_ok=True)
    data_dir = exp_dir / "data"
    data_dir.mkdir(exist_ok=True)

    log("EXP", f"실험 디렉토리: {exp_dir}")
    log("EXP", f"모드: {mode}, K={K}, temp={SC_TEMP}, 문항수={NUM_QUESTIONS}")

    # --- 데이터 로드 ---
    set_seed(SEED)  # 기존 실험과 동일 문항 순서 보장
    # 기존 실험은 500샘플을 로드했으므로 동일하게 500 로드 후 처음 200만 사용
    all_samples = load_math_by_difficulty(mode, 500)
    samples = all_samples[:NUM_QUESTIONS]
    log("DATA", f"실험 대상: {len(samples)}문항 (전체 {len(all_samples)}에서 처음 {NUM_QUESTIONS}개)")

    # 기존 entropy 로드
    existing_entropy = load_existing_entropy(mode, NUM_QUESTIONS)
    log("DATA", f"기존 entropy 매칭: {len(existing_entropy)}개")

    # --- Resume 처리 ---
    sc_results = []
    start_idx = 0
    ckpt_file = data_dir / "checkpoint.json"
    sr_file = data_dir / "sample_results.json"

    if resume_path and ckpt_file.exists() and sr_file.exists():
        with open(ckpt_file, 'r', encoding='utf-8') as f:
            ckpt = json.load(f)
        with open(sr_file, 'r', encoding='utf-8') as f:
            sc_results = json.load(f)
        start_idx = ckpt["checkpoint_idx"]
        log("RESUME", f"체크포인트에서 재개: idx={start_idx}, 완료={len(sc_results)}")

    # --- 모델 로드 ---
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    log("MODEL", f"로드 중: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    num_layers = model.config.num_hidden_layers
    log("MODEL", f"로드 완료. {num_layers} layers, device={model.device}")

    # --- 생성 루프 ---
    t_start = time.time()

    for q_idx in range(start_idx, len(samples)):
        sample = samples[q_idx]
        prompt = make_prompt(tokenizer, sample)
        gt = extract_math_answer(sample["solution"])

        log("GEN", f"[{q_idx+1}/{len(samples)}] 문항 생성 중 (K={K}, temp={SC_TEMP})...")
        t_q = time.time()

        # K개 응답 생성
        responses = generate_k_responses(
            model, tokenizer, prompt,
            k=K, temperature=SC_TEMP, max_new_tokens=1024
        )

        # SC 메트릭 계산
        sc_metrics = compute_sc_metrics(responses, sample["solution"])

        # 기존 single-pass 정답 여부 (paired comparison label)
        original_correct = existing_entropy[q_idx]["is_correct"] if q_idx < len(existing_entropy) else None

        result = {
            "idx": q_idx,
            "ground_truth": gt,
            "original_is_correct": original_correct,
            "sc_metrics": sc_metrics,
            "responses": [{"k": r["k"], "num_tokens": r["num_tokens"],
                           "answer": normalize_answer(extract_math_answer(r["text"]) or "")}
                          for r in responses],
            "generation_time": time.time() - t_q,
        }
        sc_results.append(result)

        elapsed = time.time() - t_start
        log("GEN", f"  agreement={sc_metrics['agreement_rate']:.2f}, "
                    f"sc_correct={sc_metrics['sc_correct']}, "
                    f"unique={sc_metrics['n_unique_answers']}, "
                    f"time={result['generation_time']:.1f}s, "
                    f"elapsed={elapsed:.0f}s")

        # 체크포인트 저장
        if (q_idx + 1) % CHECKPOINT_EVERY == 0 or (q_idx + 1) == len(samples):
            with open(sr_file, 'w', encoding='utf-8') as f:
                json.dump(sc_results, f, ensure_ascii=False, indent=2)
            with open(ckpt_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "checkpoint_idx": q_idx + 1,
                    "completed": q_idx + 1,
                    "total": len(samples),
                }, f, indent=2)
            log("CKPT", f"체크포인트 저장: {q_idx+1}/{len(samples)}")

    total_time = time.time() - t_start
    log("DONE", f"생성 완료. {len(sc_results)}문항, {total_time:.1f}초")

    # --- 비교 분석 ---
    log("ANALYSIS", "비교 분석 시작...")
    comparison = run_comparison_analysis(sc_results, existing_entropy, num_layers)

    # --- 최종 결과 저장 ---
    final_results = {
        "experiment_id": exp_dir.name,
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "config": {
            "model_name": model_name,
            "num_questions": len(samples),
            "K": K,
            "sc_temperature": SC_TEMP,
            "original_temperature": 0.3,
            "max_new_tokens": 1024,
            "seed": SEED,
        },
        "summary": {
            "original_accuracy": comparison.get("original_accuracy"),
            "sc_majority_vote_accuracy": comparison.get("sc_majority_vote_accuracy"),
            "sc_per_k_accuracies": comparison.get("sc_per_k_accuracies"),
            "sc_auroc_agreement": comparison.get("auroc_comparison", {}).get("sc_agreement_rate"),
            "sc_auroc_entropy": comparison.get("auroc_comparison", {}).get("sc_answer_entropy"),
            "best_unnormed_layer": comparison.get("auroc_comparison", {}).get("best_unnormed_layer"),
            "best_unnormed_auroc": comparison.get("auroc_comparison", {}).get("best_unnormed_auroc"),
            "best_normed_layer": comparison.get("auroc_comparison", {}).get("best_normed_layer"),
            "best_normed_auroc": comparison.get("auroc_comparison", {}).get("best_normed_auroc"),
            "combined_aurocs": comparison.get("combined_auroc"),
            "execution_time_seconds": total_time,
        },
        "comparison": comparison,
    }

    final_path = exp_dir / "final_results.json"
    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    log("SAVE", f"최종 결과 저장: {final_path}")

    # --- 요약 출력 ---
    log("="*60, "")
    log("SUMMARY", f"Mode: {mode}")
    log("SUMMARY", f"Original accuracy (temp=0.3): {comparison.get('original_accuracy', 0)*100:.1f}%")
    log("SUMMARY", f"SC majority-vote accuracy (temp=0.7, K=5): {comparison.get('sc_majority_vote_accuracy', 0)*100:.1f}%")
    log("SUMMARY", "")
    log("SUMMARY", "AUROC 비교:")
    auroc_comp = comparison.get("auroc_comparison", {})
    log("SUMMARY", f"  SC agreement_rate:    {auroc_comp.get('sc_agreement_rate', 0):.4f}")
    log("SUMMARY", f"  SC answer_entropy:    {auroc_comp.get('sc_answer_entropy', 0):.4f}")
    log("SUMMARY", f"  Best unnormed (L{auroc_comp.get('best_unnormed_layer', '?')}): {auroc_comp.get('best_unnormed_auroc', 0):.4f}")
    log("SUMMARY", f"  Best normed (L{auroc_comp.get('best_normed_layer', '?')}):   {auroc_comp.get('best_normed_auroc', 0):.4f}")
    log("SUMMARY", "")
    log("SUMMARY", "Combined AUROC (SC + layer entropy):")
    for k, v in comparison.get("combined_auroc", {}).items():
        log("SUMMARY", f"  {k}: {v:.4f}")
    log("SUMMARY", "")
    corr = comparison.get("correlation", {})
    log("SUMMARY", f"Spearman r (SC ↔ unnormed): {corr.get('sc_vs_unnormed_spearman_r', 0):.4f} (p={corr.get('sc_vs_unnormed_spearman_p', 1):.4e})")
    log("SUMMARY", f"Spearman r (SC ↔ normed):   {corr.get('sc_vs_normed_spearman_r', 0):.4f} (p={corr.get('sc_vs_normed_spearman_p', 1):.4e})")
    log("SUMMARY", "")
    log("SUMMARY", "Risk-Coverage AUC-RC:")
    rc = comparison.get("risk_coverage", {})
    for k, v in rc.items():
        log("SUMMARY", f"  {k}: {v.get('auc_rc', 0):.4f}")
    log("SUMMARY", f"총 실행시간: {total_time:.1f}초 ({total_time/3600:.1f}시간)")
    log("="*60, "")

    return final_results


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sampling Baseline (Self-Consistency) 비교 실험")
    parser.add_argument("--mode", choices=["easy", "hard"], required=True,
                        help="데이터셋 난이도")
    parser.add_argument("--resume", type=str, default=None,
                        help="체크포인트에서 재개 (실험 디렉토리 경로)")
    parser.add_argument("--num_questions", type=int, default=NUM_QUESTIONS,
                        help=f"문항 수 (기본: {NUM_QUESTIONS})")
    parser.add_argument("--k", type=int, default=K,
                        help=f"샘플 수 per question (기본: {K})")
    parser.add_argument("--temperature", type=float, default=SC_TEMP,
                        help=f"SC 생성 temperature (기본: {SC_TEMP})")

    args = parser.parse_args()
    NUM_QUESTIONS = args.num_questions
    K = args.k
    SC_TEMP = args.temperature

    run_experiment(args.mode, resume_path=args.resume)
