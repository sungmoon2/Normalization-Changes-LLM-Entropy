# -*- coding: utf-8 -*-
"""
Phase 2: Tuned Lens Training and Evaluation
=============================================
논문: "Eliciting Latent Predictions from Transformers with the Tuned Lens"
      (Belrose et al., 2023, arXiv:2303.08112)

방법: 각 레이어 l에 대해 affine probe (A_l, b_l) 학습
  TunedLens_l(h_l) = lm_head(norm(A_l @ h_l + b_l))

  A_l: identity로 초기화 (bias-only 모드에서는 A_l = I 고정)
  b_l: zero로 초기화

학습: cross-entropy loss (예측 토큰 vs 실제 다음 토큰)
평가: 레이어별 perplexity, top-1 accuracy, entropy discriminability

비교: Tuned Lens vs Normed Logit Lens vs Unnormed Logit Lens
"""

import os
import sys
import json
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta
from pathlib import Path
from scipy import stats
import argparse
from _paths import POT_DIR


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# ============================================================================
# 설정
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# Tuned Lens Probe
# ============================================================================

class TunedLensProbe(nn.Module):
    """
    Affine probe for one layer.
    mode='bias': A=I (fixed), train b only (3584 params)
    mode='lowrank': A = I + U @ V^T, train U, V, b (2*3584*rank + 3584 params)
    """

    def __init__(self, hidden_dim, mode='bias', rank=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

        if mode == 'lowrank':
            self.U = nn.Parameter(torch.randn(hidden_dim, rank) * 0.001)
            self.V = nn.Parameter(torch.randn(hidden_dim, rank) * 0.001)
        else:
            self.U = None
            self.V = None

    def forward(self, h):
        # h: [*, hidden_dim]
        if self.mode == 'lowrank' and self.U is not None:
            # A @ h = h + U @ V^T @ h
            h = h + (h @ self.V) @ self.U.t()
        return h + self.bias


# ============================================================================
# 유틸리티
# ============================================================================

def log(tag, msg):
    print(f"[{tag}] {msg}", flush=True)

def compute_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt((var1 + var2) / 2)
    if pooled_std < 1e-12:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std

# ============================================================================
# 데이터 로드
# ============================================================================

def load_combo_labels():
    combo_path = (POT_DIR
                  / "experiments" / "EXP_20260205_104051_combo_1024tok_temp03"
                  / "data" / "sample_results.json")
    with open(combo_path, 'r', encoding='utf-8') as f:
        combo_results = json.load(f)
    labels = [r['is_correct'] for r in combo_results]
    log("DATA", f"COMBO001 라벨: {len(labels)}개 (정답 {sum(labels)}, 오답 {len(labels)-sum(labels)})")
    return labels

def load_math_dataset():
    from datasets import load_dataset

    np.random.seed(SEED)
    dataset = load_dataset("qwedsacf/competition_math", split="train")

    level_indices = {level: [] for level in range(1, 6)}
    for idx, item in enumerate(dataset):
        level_str = item.get("level", "")
        for level in range(1, 6):
            if f"Level {level}" in level_str:
                level_indices[level].append(idx)
                break

    samples = []
    for level in range(1, 6):
        indices = level_indices[level]
        selected = np.random.choice(indices, size=200, replace=False)
        for idx in selected:
            item = dataset[int(idx)]
            samples.append({"problem": item["problem"], "level": item.get("level", "Unknown")})

    np.random.shuffle(samples)
    log("DATA", f"데이터셋: {len(samples)}개")
    return samples

def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    log("MODEL", f"로딩: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    # Freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False

    log("MODEL", f"로드 완료: {model.config.num_hidden_layers} layers, frozen")
    return model, tokenizer

# ============================================================================
# 학습
# ============================================================================

def train_probes(model, tokenizer, samples, n_train, n_epochs=5, lr=1e-3, weight_decay=0.01,
                 probe_mode='bias', probe_rank=16):
    """
    COMBO001 프롬프트로 Tuned Lens probe 학습.
    각 프롬프트의 forward pass에서 (hidden_state, next_token) 쌍 추출 후 학습.
    """
    num_layers = model.config.num_hidden_layers  # 28
    hidden_dim = model.config.hidden_size  # 3584
    device = next(model.parameters()).device

    norm = model.model.norm
    lm_head = model.lm_head

    # Create probes
    probes = nn.ModuleList([
        TunedLensProbe(hidden_dim, mode=probe_mode, rank=probe_rank)
        for _ in range(num_layers)
    ]).to(device)

    # Cast probes to FP32 for training stability
    probes = probes.float()

    optimizer = torch.optim.Adam(probes.parameters(), lr=lr, weight_decay=weight_decay)

    log("TRAIN", f"Probes: {probe_mode} mode, {sum(p.numel() for p in probes.parameters())} total params")
    log("TRAIN", f"Training on {n_train} samples, {n_epochs} epochs")

    train_samples = samples[:n_train]

    for epoch in range(n_epochs):
        epoch_losses = {li: [] for li in range(num_layers)}
        epoch_start = time.time()

        for si, sample in enumerate(train_samples):
            # Create prompt
            messages = [
                {"role": "system", "content": "You are a helpful math assistant. Solve problems step by step."},
                {"role": "user", "content": f"Solve: {sample['problem']}\n\nLet's think step by step."}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
            input_ids = inputs.input_ids  # [1, seq_len]

            if input_ids.shape[1] < 3:
                continue

            # Forward pass (frozen)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)

            # Targets: next tokens (shift by 1)
            targets = input_ids[0, 1:].detach()  # [seq_len - 1]
            seq_len = targets.shape[0]

            # Train each layer's probe
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

            for li in range(num_layers):
                # Hidden state at this layer, all positions except last
                h = outputs.hidden_states[li + 1][0, :-1, :].detach()  # [seq_len-1, hidden_dim]

                # Apply probe (FP32)
                h_transformed = probes[li](h.float())  # [seq_len-1, hidden_dim]

                # Norm and lm_head (FP16 for norm/lm_head, compute loss in FP32)
                h_normed = norm(h_transformed.to(lm_head.weight.dtype))
                logits = lm_head(h_normed).float()  # [seq_len-1, vocab_size]

                loss = F.cross_entropy(logits, targets)
                total_loss = total_loss + loss / num_layers

                epoch_losses[li].append(loss.item())

            total_loss.backward()
            optimizer.step()

            if (si + 1) % 50 == 0:
                avg_loss = np.mean([np.mean(epoch_losses[li][-50:]) for li in range(num_layers)])
                log("TRAIN", f"  Epoch {epoch+1}/{n_epochs}, Sample {si+1}/{n_train}, Avg Loss: {avg_loss:.4f}")

        epoch_time = time.time() - epoch_start
        avg_losses = {li: np.mean(epoch_losses[li]) for li in range(num_layers)}
        overall_avg = np.mean(list(avg_losses.values()))
        log("TRAIN", f"Epoch {epoch+1}/{n_epochs} done in {epoch_time:.0f}s | Avg Loss: {overall_avg:.4f}")

    return probes

# ============================================================================
# 평가
# ============================================================================

def evaluate_probes(model, tokenizer, probes, samples, labels, n_train):
    """
    Tuned Lens probe 평가:
    1. 레이어별 perplexity (next-token prediction)
    2. 레이어별 top-1 accuracy
    3. 정답/오답 discriminability (entropy 기반)
    """
    num_layers = model.config.num_hidden_layers
    device = next(model.parameters()).device
    norm = model.model.norm
    lm_head = model.lm_head

    eval_samples = samples[n_train:]
    eval_labels = labels[n_train:]

    log("EVAL", f"Evaluating on {len(eval_samples)} samples")

    # Per-layer collectors
    layer_losses = {li: [] for li in range(num_layers)}
    layer_top1_correct = {li: [] for li in range(num_layers)}

    # Per-layer, per-correctness entropy (for discriminability)
    layer_entropy_correct = {li: [] for li in range(num_layers)}
    layer_entropy_incorrect = {li: [] for li in range(num_layers)}

    # Also compute Normed Logit Lens for comparison
    normed_ll_entropy_correct = {li: [] for li in range(num_layers)}
    normed_ll_entropy_incorrect = {li: [] for li in range(num_layers)}

    probes.eval()

    for si, sample in enumerate(eval_samples):
        is_correct = eval_labels[si] if si < len(eval_labels) else False

        messages = [
            {"role": "system", "content": "You are a helpful math assistant. Solve problems step by step."},
            {"role": "user", "content": f"Solve: {sample['problem']}\n\nLet's think step by step."}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        input_ids = inputs.input_ids

        if input_ids.shape[1] < 3:
            continue

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        targets = input_ids[0, 1:]
        seq_len = targets.shape[0]

        for li in range(num_layers):
            h = outputs.hidden_states[li + 1][0, :-1, :].detach()

            with torch.no_grad():
                # --- Tuned Lens ---
                h_t = probes[li](h.float())
                h_tn = norm(h_t.to(lm_head.weight.dtype))
                logits_t = lm_head(h_tn).float()

                loss_t = F.cross_entropy(logits_t, targets)
                layer_losses[li].append(loss_t.item())

                # Top-1 accuracy
                preds_t = logits_t.argmax(dim=-1)
                acc_t = (preds_t == targets).float().mean().item()
                layer_top1_correct[li].append(acc_t)

                # Entropy (average over positions in this sample)
                probs_t = F.softmax(logits_t, dim=-1).clamp(min=1e-10)
                ent_t = (-probs_t * torch.log(probs_t)).sum(dim=-1).mean().item()
                max_ent = np.log(logits_t.shape[-1])
                ent_t_norm = ent_t / max_ent

                if is_correct:
                    layer_entropy_correct[li].append(ent_t_norm)
                else:
                    layer_entropy_incorrect[li].append(ent_t_norm)

                # --- Normed Logit Lens (comparison, no probe) ---
                h_n = norm(h.to(lm_head.weight.dtype))
                logits_n = lm_head(h_n).float()

                probs_n = F.softmax(logits_n, dim=-1).clamp(min=1e-10)
                ent_n = (-probs_n * torch.log(probs_n)).sum(dim=-1).mean().item()
                ent_n_norm = ent_n / max_ent

                if is_correct:
                    normed_ll_entropy_correct[li].append(ent_n_norm)
                else:
                    normed_ll_entropy_incorrect[li].append(ent_n_norm)

        if (si + 1) % 100 == 0:
            log("EVAL", f"  {si+1}/{len(eval_samples)} processed")

    # ========================================================================
    # 결과 집계
    # ========================================================================
    print(f"\n{'='*110}")
    print(f"{'Layer':>6} | {'TL PPL':>8} {'TL Top1%':>9} | "
          f"{'TL d':>8} {'TL p':>10} | "
          f"{'NLL d':>8} {'NLL p':>10} | {'TL>NLL?':>7}")
    print(f"{'='*110}", flush=True)

    layer_analysis = []
    tl_best_d = 0
    tl_best_layer = 0
    nll_best_d = 0
    nll_best_layer = 0

    for li in range(num_layers):
        if not layer_losses[li] or not layer_entropy_correct[li] or not layer_entropy_incorrect[li]:
            continue

        # Perplexity
        avg_loss = np.mean(layer_losses[li])
        ppl = np.exp(avg_loss) if avg_loss < 100 else float('inf')

        # Top-1 accuracy
        top1 = np.mean(layer_top1_correct[li]) * 100

        # Tuned Lens discriminability
        tl_d = compute_cohens_d(layer_entropy_incorrect[li], layer_entropy_correct[li])
        _, tl_p = stats.ttest_ind(layer_entropy_correct[li], layer_entropy_incorrect[li])

        # Normed Logit Lens discriminability
        nll_d = compute_cohens_d(normed_ll_entropy_incorrect[li], normed_ll_entropy_correct[li])
        _, nll_p = stats.ttest_ind(normed_ll_entropy_correct[li], normed_ll_entropy_incorrect[li])

        tl_better = abs(tl_d) > abs(nll_d)

        if abs(tl_d) > abs(tl_best_d):
            tl_best_d = tl_d
            tl_best_layer = li
        if abs(nll_d) > abs(nll_best_d):
            nll_best_d = nll_d
            nll_best_layer = li

        entry = {
            "layer": li,
            "tuned_lens_perplexity": float(ppl),
            "tuned_lens_loss": float(avg_loss),
            "tuned_lens_top1_accuracy": float(top1),
            "tuned_lens_discriminability": float(tl_d),
            "tuned_lens_p_value": float(tl_p),
            "normed_logit_lens_discriminability": float(nll_d),
            "normed_logit_lens_p_value": float(nll_p),
            "tuned_lens_better": tl_better,
            "tuned_lens_entropy_correct_mean": float(np.mean(layer_entropy_correct[li])),
            "tuned_lens_entropy_incorrect_mean": float(np.mean(layer_entropy_incorrect[li])),
            "normed_ll_entropy_correct_mean": float(np.mean(normed_ll_entropy_correct[li])),
            "normed_ll_entropy_incorrect_mean": float(np.mean(normed_ll_entropy_incorrect[li])),
            "n_correct": len(layer_entropy_correct[li]),
            "n_incorrect": len(layer_entropy_incorrect[li]),
        }
        layer_analysis.append(entry)

        tl_sig = "**" if tl_p < 0.05 else "  "
        nll_sig = "**" if nll_p < 0.05 else "  "
        better = "YES" if tl_better else "no"

        print(f"{li:>6} | {ppl:>8.1f} {top1:>8.1f}% | "
              f"{tl_d:>+8.4f} {tl_p:>9.2e}{tl_sig} | "
              f"{nll_d:>+8.4f} {nll_p:>9.2e}{nll_sig} | {better:>7}", flush=True)

    return layer_analysis, tl_best_layer, tl_best_d, nll_best_layer, nll_best_d

# ============================================================================
# 메인
# ============================================================================

def run_experiment(n_train=500, n_epochs=5, probe_mode='bias', probe_rank=16, lr=1e-3):
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    exp_id = f"EXP_{timestamp}_tuned_lens_{probe_mode}"
    exp_dir = (POT_DIR
               / "experiments" / "22_Tuned_Lens_Analysis" / exp_id)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "probes").mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print(f" Phase 2: Tuned Lens Training and Evaluation")
    print(f" Experiment: {exp_id}")
    print(f" Mode: {probe_mode}, Train: {n_train}, Epochs: {n_epochs}")
    print(f"{'='*70}\n", flush=True)

    # Load
    labels = load_combo_labels()
    samples = load_math_dataset()
    model, tokenizer = load_model()

    # Train
    log("PHASE", "=== Training Phase ===")
    probes = train_probes(
        model, tokenizer, samples, n_train,
        n_epochs=n_epochs, lr=lr, probe_mode=probe_mode, probe_rank=probe_rank
    )

    # Save probes
    probe_path = exp_dir / "probes" / "tuned_lens_probes.pt"
    torch.save(probes.state_dict(), probe_path)
    log("SAVE", f"Probes saved: {probe_path}")

    # Evaluate
    log("PHASE", "=== Evaluation Phase ===")
    layer_analysis, tl_best, tl_best_d, nll_best, nll_best_d = evaluate_probes(
        model, tokenizer, probes, samples, labels, n_train
    )

    total_time = datetime.now() - start_time

    # Summary
    print(f"\n{'='*70}")
    print(f" RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  Tuned Lens best layer:       Layer {tl_best} (d = {tl_best_d:+.4f})")
    print(f"  Normed Logit Lens best layer: Layer {nll_best} (d = {nll_best_d:+.4f})")

    # Count how many layers TL > NLL
    tl_wins = sum(1 for la in layer_analysis if la.get("tuned_lens_better", False))
    print(f"  Tuned Lens > Normed LL:       {tl_wins}/{len(layer_analysis)} layers")
    print(f"  Time: {total_time}", flush=True)

    # Save
    final_results = {
        "experiment_id": exp_id,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "probe_mode": probe_mode,
            "probe_rank": probe_rank if probe_mode == 'lowrank' else None,
            "n_train": n_train,
            "n_eval": len(samples) - n_train,
            "n_epochs": n_epochs,
            "lr": lr,
            "weight_decay": 0.01,
            "seed": SEED,
        },
        "results": {
            "tuned_lens_best_layer": tl_best,
            "tuned_lens_best_discriminability": float(tl_best_d),
            "normed_ll_best_layer": nll_best,
            "normed_ll_best_discriminability": float(nll_best_d),
            "tuned_lens_wins": tl_wins,
            "total_layers": len(layer_analysis),
            "execution_time_seconds": total_time.total_seconds(),
        },
        "layer_analysis": layer_analysis,
    }

    with open(exp_dir / "tuned_lens_results.json", 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    log("SAVE", f"저장 완료: {exp_dir}")
    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Tuned Lens")
    parser.add_argument("--n_train", type=int, default=500)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--mode", type=str, default="bias", choices=["bias", "lowrank"])
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    try:
        run_experiment(
            n_train=args.n_train, n_epochs=args.n_epochs,
            probe_mode=args.mode, probe_rank=args.rank, lr=args.lr
        )
    except KeyboardInterrupt:
        print("\n[INTERRUPT] 중단됨", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"\n[FATAL] {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
