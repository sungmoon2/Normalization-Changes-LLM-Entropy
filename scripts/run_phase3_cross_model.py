"""
Phase 3 Cross-Model: Scale Intervention on Llama & Mistral
==========================================================
GPT 13차 양쪽 만장일치 최우선 추가 실험:
  "Llama 또는 Mistral에 Phase 3 축소 복제 (2-3 representative layers, unit-norm, alpha-sweep)"
  "이 한 개가 venue odds를 가장 많이 올림"

목적:
  Phase 3 (Qwen)의 핵심 결과를 cross-model로 검증:
  1. unit-norm removal → H_pre가 ~1.0으로 가는지 (scale 소멸)
  2. alpha-sweep → H_pre 민감 / H_post 불변인지
  3. Mistral에서 Phase 0b retention 100%였는데, intervention에서도 다른 패턴?

비판적 체크포인트:
  - "pure scale" 금지. "strongly scale-sensitive" 확인 or 반증
  - Mistral retention 100%: unit-norm에서도 H_pre→1.0이면 "retention != invariance" 구분 필요
  - 수학적으로 z_pre = W(ru) = rWu이므로 direction u도 여전히 기여. "pure"는 과한 claim
  - alpha-sweep에서 H_post가 정말 4자리 동일한지, Llama/Mistral에서도 확인

설계:
  - Llama: MMLU 300 samples (기존 MMLU와 동일 seed → 동일 샘플 subset)
  - Mistral: MMLU 300 samples (동일)
  - Position: Step 0 (prompt-last) — Phase 1에서 검증된 위치
  - Layers: 전체 (32 layers) — 축소 아닌 전수 (forward pass만이라 빠름)
  - Alphas: [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0]
  - 생성 없음 → forward pass만 → 예상 5-10분/모델

사용법:
  python run_phase3_cross_model.py --model llama --num_samples 300
  python run_phase3_cross_model.py --model mistral --num_samples 300
  python run_phase3_cross_model.py --model all --num_samples 300

NOTE: GPU 필요. 생성 없이 forward pass만 수행.
"""

import os
import sys
import json
import random
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timedelta
import argparse

SEED = 42
PROJECT_ROOT = Path(__file__).parent.parent
BASE_DIR = PROJECT_ROOT / "PoT_Experiment_Entropy_Attention_Extraction_Experiment"
OUTPUT_DIR = BASE_DIR / "experiments" / "39_Phase3_Cross_Model_Scale_Intervention"

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

ALPHAS = [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0]


def set_seed(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_entropy_from_logits(logits_f32):
    """Compute normalized Shannon entropy from logits (FP32)."""
    probs = torch.softmax(logits_f32, dim=-1).clamp(min=1e-10)
    ent = (-probs * torch.log(probs)).sum(dim=-1)
    max_ent = float(np.log(logits_f32.shape[-1]))
    return float(ent.cpu().item()) / max_ent


def load_mmlu_samples(num_samples):
    """Load MMLU samples with same seed as existing experiments."""
    from datasets import load_dataset
    dataset = load_dataset("cais/mmlu", "all", split="test")
    np.random.seed(SEED)
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    indices.sort()

    samples = []
    for idx in indices:
        item = dataset[int(idx)]
        labels = ['A', 'B', 'C', 'D']
        answer_idx = item['answer']
        choice_str = "\n".join(f"{l}. {t}" for l, t in zip(labels[:len(item['choices'])], item['choices']))
        samples.append({
            "question": item['question'],
            "choices_str": choice_str,
            "answer_key": labels[answer_idx] if answer_idx < 4 else 'A',
            "subject": item.get('subject', 'unknown'),
        })

    random.seed(SEED)
    random.shuffle(samples)
    return samples


def make_prompt(tokenizer, sample, model_key):
    """Create chat prompt for MMLU sample."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers multiple choice questions step by step."},
        {"role": "user", "content": (
            f"Answer the following multiple choice question step by step.\n\n"
            f"Question: {sample['question']}\n\n{sample['choices_str']}\n\n"
            f"Think through each option carefully, then end with "
            f"\"The answer is [LETTER]\" where LETTER is A, B, C, or D."
        )}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_intervention(model_key, num_samples):
    """Run scale intervention for a single model."""
    config = MODEL_CONFIGS[model_key]
    model_name = config["name"]
    model_short = config["short"]

    model_output_dir = OUTPUT_DIR / model_short
    os.makedirs(model_output_dir, exist_ok=True)

    print("=" * 70)
    print(f"Phase 3 Cross-Model: Scale Intervention on {model_name}")
    print(f"Samples: {num_samples}, Alphas: {ALPHAS}")
    print(f"Position: Step 0 (prompt-last)")
    print("=" * 70)

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\nLoading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    lm_head = model.lm_head
    norm = model.model.norm  # RMSNorm for all three model families
    print(f"  Layers: {num_layers}, Hidden: {hidden_size}")

    # Load dataset
    print(f"\nLoading MMLU dataset...")
    samples = load_mmlu_samples(num_samples)
    print(f"  Loaded: {len(samples)} MMLU samples")

    # Process
    start_time = time.time()
    all_results = []
    errors = 0

    for i, sample in enumerate(samples):
        try:
            prompt = make_prompt(tokenizer, sample, model_key)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            last_pos = -1
            sample_data = {
                "idx": i,
                "original": {},
                "unit_norm": {},
                "alpha_sweep": {str(a): {} for a in ALPHAS},
            }

            for layer_idx in range(num_layers):
                hidden = outputs.hidden_states[layer_idx + 1][:, last_pos:, :]
                h = hidden.float()
                h_norm_val = float(torch.norm(h).cpu().item())

                # --- Original ---
                logits_u = lm_head(h.to(lm_head.weight.dtype)).float()
                h_normed = norm(h.to(norm.weight.dtype)).float()
                logits_n = lm_head(h_normed.to(lm_head.weight.dtype)).float()

                ent_pre = compute_entropy_from_logits(logits_u.squeeze(0))
                ent_post = compute_entropy_from_logits(logits_n.squeeze(0))

                sample_data["original"][str(layer_idx)] = {
                    "h_pre": ent_pre,
                    "h_post": ent_post,
                    "h_norm": h_norm_val,
                }

                # --- Unit-norm removal ---
                if h_norm_val > 1e-10:
                    h_unit = h / h_norm_val
                else:
                    h_unit = h

                logits_u_unit = lm_head(h_unit.to(lm_head.weight.dtype)).float()
                h_unit_normed = norm(h_unit.to(norm.weight.dtype)).float()
                logits_n_unit = lm_head(h_unit_normed.to(lm_head.weight.dtype)).float()

                ent_pre_unit = compute_entropy_from_logits(logits_u_unit.squeeze(0))
                ent_post_unit = compute_entropy_from_logits(logits_n_unit.squeeze(0))

                sample_data["unit_norm"][str(layer_idx)] = {
                    "h_pre": ent_pre_unit,
                    "h_post": ent_post_unit,
                }

                # --- Alpha-sweep ---
                for alpha in ALPHAS:
                    h_scaled = alpha * h

                    logits_u_a = lm_head(h_scaled.to(lm_head.weight.dtype)).float()
                    h_a_normed = norm(h_scaled.to(norm.weight.dtype)).float()
                    logits_n_a = lm_head(h_a_normed.to(lm_head.weight.dtype)).float()

                    ent_pre_a = compute_entropy_from_logits(logits_u_a.squeeze(0))
                    ent_post_a = compute_entropy_from_logits(logits_n_a.squeeze(0))

                    sample_data["alpha_sweep"][str(alpha)][str(layer_idx)] = {
                        "h_pre": ent_pre_a,
                        "h_post": ent_post_a,
                    }

            all_results.append(sample_data)

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (len(samples) - i - 1)
                print(f"  [{i+1}/{len(samples)}] {timedelta(seconds=int(elapsed))} "
                      f"ETA {timedelta(seconds=int(eta))}")

        except Exception as e:
            errors += 1
            print(f"  [{i+1}/{len(samples)}] ERROR: {e}")
            all_results.append({"idx": i, "error": str(e)})

    total_time = time.time() - start_time
    print(f"\nExtraction complete: {timedelta(seconds=int(total_time))}, errors: {errors}/{len(samples)}")

    # Save raw data
    raw_path = model_output_dir / "intervention_raw_data.json"
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=1)
    print(f"  Saved: {raw_path}")

    # ================================================================
    # Analysis
    # ================================================================
    valid_results = [r for r in all_results if "error" not in r]
    n_valid = len(valid_results)
    print(f"\n--- {model_short.upper()} Intervention Analysis ({n_valid} valid samples) ---")

    # Key layers: every 4th + final
    key_layers = list(range(0, num_layers, 4))
    if (num_layers - 1) % 4 != 0:
        key_layers.append(num_layers - 1)

    # Unit-norm analysis
    print(f"\n  Unit-norm removal effect (mean across {n_valid} samples):")
    print(f"  {'Layer':>6s} {'H_pre orig':>11s} {'H_pre unit':>11s} {'Change':>8s} "
          f"{'H_post orig':>12s} {'H_post unit':>12s} {'Change':>8s}")
    print("  " + "-" * 72)

    unit_norm_summary = {}
    for l in range(num_layers):
        lk = str(l)
        pre_orig = [r["original"][lk]["h_pre"] for r in valid_results]
        pre_unit = [r["unit_norm"][lk]["h_pre"] for r in valid_results]
        post_orig = [r["original"][lk]["h_post"] for r in valid_results]
        post_unit = [r["unit_norm"][lk]["h_post"] for r in valid_results]
        h_norms = [r["original"][lk]["h_norm"] for r in valid_results]

        pre_change = np.mean(pre_unit) - np.mean(pre_orig)
        post_change = np.mean(post_unit) - np.mean(post_orig)

        unit_norm_summary[lk] = {
            "h_pre_orig_mean": round(float(np.mean(pre_orig)), 6),
            "h_pre_unit_mean": round(float(np.mean(pre_unit)), 6),
            "h_pre_change": round(float(pre_change), 6),
            "h_post_orig_mean": round(float(np.mean(post_orig)), 6),
            "h_post_unit_mean": round(float(np.mean(post_unit)), 6),
            "h_post_change": round(float(post_change), 6),
            "h_norm_mean": round(float(np.mean(h_norms)), 4),
            "h_norm_std": round(float(np.std(h_norms)), 4),
        }

        if l in key_layers:
            print(f"  {l:6d} {np.mean(pre_orig):11.4f} {np.mean(pre_unit):11.4f} {pre_change:+8.4f} "
                  f"{np.mean(post_orig):12.4f} {np.mean(post_unit):12.4f} {post_change:+8.4f}")

    # Alpha-sweep at representative layers (early, mid, late)
    repr_layers = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]
    repr_layers = sorted(set(repr_layers))

    print(f"\n  Alpha-sweep at representative layers: {repr_layers}")

    alpha_summary = {}
    for alpha in ALPHAS:
        ak = str(alpha)
        row = {}
        for l in repr_layers:
            lk = str(l)
            pre_vals = [r["alpha_sweep"][ak][lk]["h_pre"] for r in valid_results
                        if ak in r["alpha_sweep"] and lk in r["alpha_sweep"][ak]]
            post_vals = [r["alpha_sweep"][ak][lk]["h_post"] for r in valid_results
                         if ak in r["alpha_sweep"] and lk in r["alpha_sweep"][ak]]
            row[f"L{l}_h_pre"] = round(float(np.mean(pre_vals)), 6) if pre_vals else None
            row[f"L{l}_h_post"] = round(float(np.mean(post_vals)), 6) if post_vals else None
        alpha_summary[ak] = row

    # Print alpha-sweep table for two key layers
    mid_layer = num_layers // 2
    late_layer = 3 * num_layers // 4
    print(f"\n  Alpha-sweep detail (L{mid_layer} and L{late_layer}):")
    print(f"  {'Alpha':>7s} {'L'+str(mid_layer)+' H_pre':>12s} {'L'+str(mid_layer)+' H_post':>13s} "
          f"{'L'+str(late_layer)+' H_pre':>13s} {'L'+str(late_layer)+' H_post':>14s}")
    print("  " + "-" * 60)

    for alpha in ALPHAS:
        ak = str(alpha)
        row = alpha_summary[ak]
        mp = row.get(f"L{mid_layer}_h_pre", 0)
        mq = row.get(f"L{mid_layer}_h_post", 0)
        lp = row.get(f"L{late_layer}_h_pre", 0)
        lq = row.get(f"L{late_layer}_h_post", 0)
        print(f"  {alpha:7.2f} {mp:12.4f} {mq:13.4f} {lp:13.4f} {lq:14.4f}")

    # Compute H_post max variation across alphas (critical sanity check)
    print(f"\n  H_post alpha-invariance check (max variation across alphas per layer):")
    h_post_max_var = {}
    for l in range(num_layers):
        lk = str(l)
        post_by_alpha = []
        for alpha in ALPHAS:
            ak = str(alpha)
            vals = [r["alpha_sweep"][ak][lk]["h_post"] for r in valid_results
                    if ak in r["alpha_sweep"] and lk in r["alpha_sweep"][ak]]
            if vals:
                post_by_alpha.append(np.mean(vals))
        if len(post_by_alpha) >= 2:
            max_var = max(post_by_alpha) - min(post_by_alpha)
            h_post_max_var[lk] = round(float(max_var), 8)

    worst_layer = max(h_post_max_var, key=h_post_max_var.get)
    worst_var = h_post_max_var[worst_layer]
    print(f"  Worst-case layer: L{worst_layer}, max H_post variation: {worst_var:.6f}")
    if worst_var < 0.001:
        print(f"  PASS: H_post is alpha-invariant (variation < 0.001)")
    elif worst_var < 0.01:
        print(f"  MARGINAL: H_post variation {worst_var:.6f} (< 0.01 but > 0.001)")
    else:
        print(f"  FAIL: H_post is NOT alpha-invariant (variation {worst_var:.6f} >= 0.01)")
        print(f"  --> This would CHALLENGE 'H_post = pure direction signal' claim")

    # Compute H_pre unit-norm saturation check
    print(f"\n  H_pre unit-norm saturation check:")
    pre_unit_mean_all = np.mean([float(unit_norm_summary[str(l)]["h_pre_unit_mean"]) for l in range(num_layers)])
    print(f"  Mean H_pre after unit-norm (all layers): {pre_unit_mean_all:.6f}")
    if pre_unit_mean_all > 0.99:
        print(f"  CONFIRMED: H_pre → ~1.0 after unit-norm (scale signal dominates)")
    elif pre_unit_mean_all > 0.95:
        print(f"  MOSTLY CONFIRMED: H_pre → ~{pre_unit_mean_all:.4f} (strong but not complete)")
    else:
        print(f"  PARTIAL: H_pre → ~{pre_unit_mean_all:.4f} after unit-norm")
        print(f"  --> Direction component contributes non-trivially to H_pre on {model_short}")

    # Save analysis
    analysis = {
        "model": model_name,
        "model_short": model_short,
        "n_samples": n_valid,
        "n_errors": errors,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "position": "step0_prompt_last",
        "alphas": ALPHAS,
        "repr_layers": repr_layers,
        "unit_norm_summary": unit_norm_summary,
        "alpha_summary": alpha_summary,
        "h_post_max_variation": h_post_max_var,
        "h_post_worst_layer": int(worst_layer),
        "h_post_worst_variation": worst_var,
        "h_pre_unit_mean_all_layers": round(pre_unit_mean_all, 6),
        "total_time_seconds": round(total_time, 1),
        "timestamp": datetime.now().isoformat(),
    }

    analysis_path = model_output_dir / "intervention_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n  Saved: {analysis_path}")

    # Cleanup GPU
    del model
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    print(f"\n{model_short.upper()} Phase 3 complete: {timedelta(seconds=int(total_time))}")
    return analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all", choices=["qwen", "llama", "mistral", "all"])
    parser.add_argument("--num_samples", type=int, default=300)
    args = parser.parse_args()

    set_seed()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    models_to_run = ["llama", "mistral"] if args.model == "all" else [args.model]
    all_analyses = {}

    for model_key in models_to_run:
        set_seed()  # Reset seed for each model for reproducibility
        analysis = run_intervention(model_key, args.num_samples)
        all_analyses[model_key] = analysis

    # Cross-model comparison if both ran
    if len(all_analyses) == 2:
        print("\n" + "=" * 70)
        print("Cross-Model Comparison: Llama vs Mistral")
        print("=" * 70)

        for model_key, a in all_analyses.items():
            print(f"\n  {model_key.upper()}: {a['num_layers']} layers, {a['hidden_size']} hidden")
            print(f"    H_pre unit-norm mean: {a['h_pre_unit_mean_all_layers']:.6f}")
            print(f"    H_post worst alpha-variation: {a['h_post_worst_variation']:.6f} (L{a['h_post_worst_layer']})")

        # Compare with Qwen Phase 3 results
        qwen_analysis_path = BASE_DIR / "experiments" / "36_Phase3_Scale_Intervention" / "intervention_analysis.json"
        if qwen_analysis_path.exists():
            with open(qwen_analysis_path) as f:
                qwen_a = json.load(f)

            print(f"\n  Qwen (original Phase 3):")
            if "h_pre_unit_mean_all_layers" in qwen_a:
                print(f"    H_pre unit-norm mean: {qwen_a['h_pre_unit_mean_all_layers']:.6f}")
            # Check H_post variation from Qwen
            if "h_post_max_variation" in qwen_a:
                qwen_worst = max(qwen_a["h_post_max_variation"].values())
                print(f"    H_post worst alpha-variation: {qwen_worst:.6f}")

        # Save cross-model summary
        summary = {
            "models": {k: {
                "h_pre_unit_mean": a["h_pre_unit_mean_all_layers"],
                "h_post_worst_var": a["h_post_worst_variation"],
                "n_samples": a["n_samples"],
            } for k, a in all_analyses.items()},
            "timestamp": datetime.now().isoformat(),
        }

        summary_path = OUTPUT_DIR / "cross_model_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Saved: {summary_path}")

    print(f"\nAll done.")


if __name__ == "__main__":
    main()
