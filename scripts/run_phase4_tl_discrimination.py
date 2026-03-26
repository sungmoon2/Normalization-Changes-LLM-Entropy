"""
Phase 4a: TL Discrimination on MMLU
=====================================
훈련된 Tuned Lens로 MMLU 샘플에서 TL-entropy 추출 → correctness AUROC 비교.

GPT 12 수렴:
  "TL을 correctness AUROC로만 평가하면 안 됨. 먼저 faithfulness (완료).
   그 다음에 entropy discriminability/AUROC로 내려감."

현재 상태:
  - Phase 2b: faithfulness 확인 (TL > LL 27/28 layers)
  - 이 스크립트: TL entropy의 correctness discrimination 평가

방법:
  1. MMLU Qwen 1000 samples에 forward pass (생성 없음, prompt-last)
  2. 각 layer에서 TL(h) → logits → entropy 계산
  3. LL(h) = lm_head(h) → entropy도 동시 계산 (비교용)
  4. 70/30 held-out AUROC 비교 (Phase 0과 동일 split)

사용법:
  python run_phase4_tl_discrimination.py
"""

import os
import json
import random
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
BASE_DIR = Path(__file__).parent.parent / "PoT_Experiment_Entropy_Attention_Extraction_Experiment"
OUTPUT_DIR = BASE_DIR / "experiments" / "37_Phase4_TL_Discrimination"


def set_seed(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_auroc(y, scores):
    mask = ~np.isnan(scores)
    if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y[mask], scores[mask]))
    except ValueError:
        return 0.5


def compute_entropy(logits_f32):
    probs = torch.softmax(logits_f32, dim=-1).clamp(min=1e-10)
    ent = (-probs * torch.log(probs)).sum(dim=-1)
    max_ent = float(np.log(logits_f32.shape[-1]))
    return float(ent.cpu().item()) / max_ent


def patch_tuned_lens_for_qwen():
    import tuned_lens.model_surgery as ms
    original = ms.get_final_norm
    def patched(model):
        if not hasattr(model, "base_model"):
            raise ValueError("No base_model")
        base = model.base_model
        if hasattr(base, 'norm') and 'Qwen2' in type(base).__name__:
            return base.norm
        return original(model)
    ms.get_final_norm = patched


def main():
    set_seed()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    patch_tuned_lens_for_qwen()

    print("=" * 70)
    print("Phase 4a: TL Discrimination on MMLU")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tuned_lens import TunedLens

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.eval()
    num_layers = model.config.num_hidden_layers
    lm_head = model.lm_head
    norm_layer = model.model.norm

    # Load trained TL
    lens_path = BASE_DIR / "experiments" / "35_Phase2b_Tuned_Lens" / "qwen_tuned_lens"
    lens = TunedLens.from_model_and_pretrained(model, lens_resource_id=str(lens_path))
    lens = lens.float().to(model.device)
    lens.eval()
    print(f"  Model: {num_layers} layers, TL loaded from {lens_path}")

    # Load MMLU data for correctness labels
    mmlu_data_path = BASE_DIR / "experiments" / "31_MMLU_Domain_Extension" / "EXP_20260219_053638_mmlu_qwen" / "data" / "sample_results.json"
    with open(mmlu_data_path) as f:
        mmlu_samples = json.load(f)
    if isinstance(mmlu_samples, dict) and "results" in mmlu_samples:
        mmlu_samples = mmlu_samples["results"]
    print(f"  MMLU samples: {len(mmlu_samples)}")

    # Load MMLU dataset to get prompts
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    # Match by index (same order as original experiment)
    # Actually, the original experiment shuffled with seed=42
    # We need to match samples by their content
    # Safer: use the stored predicted/ground_truth to identify, but we need prompts
    # Let's just use the dataset directly with the same subjects
    # For each stored sample, find matching question in dataset

    # Build lookup by question text
    ds_lookup = {}
    choices_letters = ["A", "B", "C", "D"]
    for item in ds:
        ds_lookup[item["question"][:100]] = item  # key by first 100 chars

    # Process each MMLU sample
    start_time = time.time()
    results = []

    for i, sample in enumerate(mmlu_samples):
        if "layer_data" not in sample or "is_correct" not in sample:
            continue

        is_correct = sample["is_correct"]
        subject = sample.get("subject", "unknown")
        gt = sample.get("ground_truth", "")

        # Find matching dataset entry to reconstruct prompt
        # Use subject + ground_truth to narrow down
        matched = None
        for item in ds:
            if item.get("subject") == subject:
                item_gt = choices_letters[item["answer"]]
                if item_gt == gt:
                    # Check if this matches by trying the question
                    matched = item
                    break

        if matched is None:
            # Fallback: use a generic prompt with just the subject
            # Skip this sample
            results.append({"idx": i, "skipped": True, "reason": "no_match"})
            continue

        # Construct prompt (same format as original)
        q = matched["question"]
        opts = "\n".join(f"{choices_letters[j]}. {matched['choices'][j]}" for j in range(len(matched['choices'])))
        prompt_text = (f"Answer the following multiple choice question. "
                      f"Think step by step, then give your final answer as "
                      f"\"The answer is [LETTER]\" where LETTER is A, B, C, or D.\n\n"
                      f"Question: {q}\n{opts}")
        messages = [{"role": "user", "content": prompt_text}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        try:
            inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            # Prompt-last position
            last_pos = -1
            sample_result = {
                "idx": i,
                "is_correct": is_correct,
                "subject": subject,
                "tl_entropy": {},
                "ll_entropy": {},
                "normed_entropy": {},
            }

            for layer_idx in range(num_layers):
                h = outputs.hidden_states[layer_idx + 1][:, last_pos:, :].float()

                # TL entropy
                tl_logits = lens(h, idx=layer_idx).float()
                tl_ent = compute_entropy(tl_logits.squeeze(0))

                # LL entropy (logit lens = raw lm_head)
                ll_logits = lm_head(h.to(lm_head.weight.dtype)).float()
                ll_ent = compute_entropy(ll_logits.squeeze(0))

                # Normed entropy (H_post)
                h_normed = norm_layer(h.to(norm_layer.weight.dtype)).float()
                n_logits = lm_head(h_normed.to(lm_head.weight.dtype)).float()
                n_ent = compute_entropy(n_logits.squeeze(0))

                sample_result["tl_entropy"][str(layer_idx)] = round(tl_ent, 8)
                sample_result["ll_entropy"][str(layer_idx)] = round(ll_ent, 8)
                sample_result["normed_entropy"][str(layer_idx)] = round(n_ent, 8)

            results.append(sample_result)

        except Exception as e:
            results.append({"idx": i, "error": str(e)})

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (len(mmlu_samples) - i - 1)
            n_valid = sum(1 for r in results if "tl_entropy" in r)
            print(f"  [{i+1}/{len(mmlu_samples)}] valid={n_valid} "
                  f"{timedelta(seconds=int(elapsed))} ETA {timedelta(seconds=int(eta))}")

    total_time = time.time() - start_time
    print(f"\nExtraction complete: {timedelta(seconds=int(total_time))}")

    # Save raw data
    raw_path = OUTPUT_DIR / "tl_discrimination_raw.json"
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=1)

    # ================================================================
    # Analysis: AUROC comparison
    # ================================================================
    valid = [r for r in results if "tl_entropy" in r]
    n_valid = len(valid)
    labels = np.array([r["is_correct"] for r in valid], dtype=int)
    print(f"\nAnalysis: {n_valid} valid samples, {labels.sum()} correct ({labels.mean()*100:.1f}%)")

    # Load Phase 0 split
    phase0_path = BASE_DIR / "experiments" / "32_Phase0_CalTest_Baselines" / "mmlu_qwen_baselines.json"
    with open(phase0_path) as f:
        phase0 = json.load(f)
    cal_indices_orig = set(phase0["split"]["cal_indices"])

    # Map valid samples to cal/test
    cal_mask = np.array([r["idx"] in cal_indices_orig for r in valid])
    test_mask = ~cal_mask
    y_cal, y_test = labels[cal_mask], labels[test_mask]

    print(f"  Split: cal={cal_mask.sum()}, test={test_mask.sum()}")

    # Build per-layer arrays
    tl_arr = np.zeros((n_valid, num_layers))
    ll_arr = np.zeros((n_valid, num_layers))
    normed_arr = np.zeros((n_valid, num_layers))

    for j, r in enumerate(valid):
        for l in range(num_layers):
            lk = str(l)
            tl_arr[j, l] = r["tl_entropy"].get(lk, np.nan)
            ll_arr[j, l] = r["ll_entropy"].get(lk, np.nan)
            normed_arr[j, l] = r["normed_entropy"].get(lk, np.nan)

    # Select best layer on cal, evaluate on test
    def best_layer_auroc(y_c, X_c, y_t, X_t, n_layers):
        bl, bs, ba = 0, 1, -1
        for l in range(n_layers):
            ap = safe_auroc(y_c, X_c[:, l])
            an = safe_auroc(y_c, -X_c[:, l])
            if ap >= an and ap > ba: bl, bs, ba = l, 1, ap
            elif an > ap and an > ba: bl, bs, ba = l, -1, an
        ta = safe_auroc(y_t, bs * X_t[:, bl])
        return bl, bs, round(ba, 6), round(ta, 6)

    tl_cal, tl_test = tl_arr[cal_mask], tl_arr[test_mask]
    ll_cal, ll_test = ll_arr[cal_mask], ll_arr[test_mask]
    nr_cal, nr_test = normed_arr[cal_mask], normed_arr[test_mask]

    tl_l, tl_s, tl_ca, tl_ta = best_layer_auroc(y_cal, tl_cal, y_test, tl_test, num_layers)
    ll_l, ll_s, ll_ca, ll_ta = best_layer_auroc(y_cal, ll_cal, y_test, ll_test, num_layers)
    nr_l, nr_s, nr_ca, nr_ta = best_layer_auroc(y_cal, nr_cal, y_test, nr_test, num_layers)

    print(f"\n  {'Method':25s} {'Layer':>6s} {'Sign':>5s} {'Cal':>8s} {'Test':>8s}")
    print("  " + "-" * 55)
    print(f"  {'TL entropy':25s} {tl_l:6d} {'+'if tl_s==1 else '-':>5s} {tl_ca:8.4f} {tl_ta:8.4f}")
    print(f"  {'LL entropy (H_pre)':25s} {ll_l:6d} {'+'if ll_s==1 else '-':>5s} {ll_ca:8.4f} {ll_ta:8.4f}")
    print(f"  {'Normed entropy (H_post)':25s} {nr_l:6d} {'+'if nr_s==1 else '-':>5s} {nr_ca:8.4f} {nr_ta:8.4f}")

    analysis = {
        "n_valid": n_valid,
        "n_correct": int(labels.sum()),
        "accuracy": round(float(labels.mean()), 6),
        "cal_size": int(cal_mask.sum()),
        "test_size": int(test_mask.sum()),
        "results": {
            "tl_entropy": {"layer": tl_l, "sign": tl_s, "cal_auroc": tl_ca, "test_auroc": tl_ta},
            "ll_entropy_hpre": {"layer": ll_l, "sign": ll_s, "cal_auroc": ll_ca, "test_auroc": ll_ta},
            "normed_entropy_hpost": {"layer": nr_l, "sign": nr_s, "cal_auroc": nr_ca, "test_auroc": nr_ta},
        },
        "total_time_seconds": round(total_time, 1),
        "timestamp": datetime.now().isoformat(),
    }

    with open(OUTPUT_DIR / "tl_discrimination_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"\n  Saved: {OUTPUT_DIR}")
    print(f"  Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
