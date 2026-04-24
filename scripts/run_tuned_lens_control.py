"""
Tuned Lens Control: Does scale-dominance hold under a faithful decoder?

Purpose: Verify that the scale-dominance finding (logit_std >= H_pre,
         scale absorption) is not an artifact of raw Logit Lens unfaithfulness.
         If it holds under Tuned Lens, the finding is robust to decoder choice.

Protocol:
  1. Load model + trained Tuned Lens for each model
  2. For each MMLU sample (exp31, 1000 samples):
     - Extract hidden states at all layers (generation-average)
     - Compute TL-decoded entropy (H_tl) = H(softmax(TL(h)))
     - Also compute raw LL entropy (H_pre) and scale metrics for cross-check
  3. Compare AUROC: H_tl vs logit_std vs h_norm
  4. Test incremental utility: logit_std + H_tl vs logit_std only
  5. Save per-sample results with checkpoint/resume

GPU required. Detach-safe. Checkpoint every 50 samples.
"""

import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from _paths import POT_DIR

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE = POT_DIR / "experiments"
OUTPUT = BASE / "46_Tuned_Lens_Control"
OUTPUT.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = {
    "qwen": {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "n_layers": 28,
        "lens_path": str(BASE / "35_Phase2b_Tuned_Lens/qwen_tuned_lens"),
        "data_path": str(BASE / "31_MMLU_Domain_Extension/EXP_20260219_053638_mmlu_qwen/data/sample_results.json"),
    },
    "llama": {
        "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "n_layers": 32,
        "lens_path": str(BASE / "44_GPT25_Experiments/phase3_gpu_tunedlens/EXP3a_llama_tuned_lens/llama_tuned_lens"),
        "data_path": str(BASE / "31_MMLU_Domain_Extension/EXP_20260219_171237_mmlu_llama/data/sample_results.json"),
    },
    "mistral": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "n_layers": 32,
        "lens_path": str(BASE / "44_GPT25_Experiments/phase3_gpu_tunedlens/EXP3b_mistral_tuned_lens/mistral_tuned_lens"),
        "data_path": str(BASE / "31_MMLU_Domain_Extension/EXP_20260220_000610_mmlu_mistral/data/sample_results.json"),
    },
}

CHECKPOINT_INTERVAL = 50


def set_seed():
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def patch_tuned_lens():
    """Monkey-patch tuned-lens to support Qwen2 and Mistral architectures."""
    import tuned_lens.model_surgery as ms
    original_get_final_norm = ms.get_final_norm

    PATCHED_MODELS = ('Qwen2Model', 'Qwen2ForCausalLM', 'MistralModel', 'MistralForCausalLM')

    def patched_get_final_norm(model):
        if not hasattr(model, "base_model"):
            raise ValueError("Model does not have a `base_model` attribute.")
        base_model = model.base_model
        if hasattr(base_model, 'norm') and type(base_model).__name__ in PATCHED_MODELS:
            return base_model.norm
        return original_get_final_norm(model)

    ms.get_final_norm = patched_get_final_norm
    print("  Patched tuned-lens for Qwen2/Mistral support")


def compute_entropy(logits):
    """Compute normalized Shannon entropy from logits."""
    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log(probs + 1e-30)
    entropy = -(probs * log_probs).sum(dim=-1)
    max_entropy = np.log(probs.shape[-1])
    return float((entropy / max_entropy).mean().cpu().item())


def run_model(model_key, config):
    """Run TL control experiment for one model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tuned_lens import TunedLens

    model_dir = OUTPUT / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_dir / "checkpoint.json"
    final_path = model_dir / "tl_control_results.json"

    if final_path.exists():
        print(f"  {model_key}: Already completed, skipping.")
        return json.load(open(final_path))

    # Load checkpoint
    completed = []
    if ckpt_path.exists():
        completed = json.load(open(ckpt_path))
        print(f"  Resuming from checkpoint: {len(completed)} samples done")

    # Load exp31 data (need ground truth + layer_data for cross-check)
    exp31_data = json.load(open(config["data_path"]))
    n_total = len(exp31_data)
    n_done = len(completed)

    if n_done >= n_total:
        print(f"  {model_key}: All {n_total} samples done.")
        # Save final
        results = _build_final(completed, config, model_key)
        with open(final_path, "w") as f:
            json.dump(results, f, indent=2)
        return results

    # Load model
    if model_key in ("qwen", "mistral"):
        patch_tuned_lens()

    print(f"\n{'='*60}")
    print(f"  {model_key.upper()} Tuned Lens Control")
    print(f"  Model: {config['model_name']}")
    print(f"  Lens: {config['lens_path']}")
    print(f"  Samples: {n_done}/{n_total}")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"], torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    lens = TunedLens.from_model_and_pretrained(model, lens_resource_id=config["lens_path"])
    lens = lens.to(model.device)
    lens.eval()

    n_layers = config["n_layers"]
    vocab_size = model.config.vocab_size

    # Process samples
    for idx in range(n_done, n_total):
        sample = exp31_data[idx]
        t0 = time.time()

        # Reconstruct the prompt + response
        # exp31 stores generation results; we need to re-run inference
        # But we can use the stored response text to get hidden states
        subject = sample.get("subject", "unknown")
        gt = sample.get("ground_truth", "")
        predicted = sample.get("predicted", "")
        is_correct = sample.get("is_correct", False)

        # Build prompt (MMLU format)
        question_text = f"Answer the following multiple choice question about {subject}.\n\n"
        # We don't have the original question text in exp31 sample_results
        # Instead, use a simpler approach: encode the model's generated response
        # and extract hidden states from it

        # Actually, for TL control we need hidden states during generation.
        # But exp31 only stores per-layer aggregated metrics, not raw hidden states.
        # We need to re-generate or use a proxy.

        # Approach: Use the same MMLU sampling to get the question, generate a response,
        # and compute TL entropy on the generation-average hidden states.

        # Load MMLU dataset
        if idx == n_done:  # Load once
            from datasets import load_dataset
            mmlu = load_dataset("cais/mmlu", "all", split="test")
            # Same sampling as exp31
            mmlu_indices = np.random.RandomState(42).choice(len(mmlu), 1000, replace=False)
            print(f"  MMLU loaded: {len(mmlu)} total, using {len(mmlu_indices)} samples")

        mmlu_idx = mmlu_indices[idx]
        item = mmlu[int(mmlu_idx)]
        question = item["question"]
        choices = item["choices"]
        answer_idx = item["answer"]
        answer_letter = "ABCD"[answer_idx]

        # Format prompt
        choices_text = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        prompt = f"Answer the following multiple choice question about {subject}.\n\n{question}\n{choices_text}\n\nAnswer:"

        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        # Collect hidden states from generated tokens
        gen_token_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        n_gen = len(gen_token_ids)

        # For each generated token step, compute TL entropy and LL entropy
        tl_entropies = {l: [] for l in range(n_layers)}
        ll_entropies = {l: [] for l in range(n_layers)}
        logit_stds = {l: [] for l in range(n_layers)}
        h_norms = {l: [] for l in range(n_layers)}

        # outputs.hidden_states is a tuple of (n_gen_steps,) each containing
        # (n_layers+1,) tensors of shape (1, seq_len_at_step, hidden_dim)
        for step_idx in range(min(n_gen, len(outputs.hidden_states))):
            step_hidden = outputs.hidden_states[step_idx]
            # step_hidden is tuple of (n_layers+1) tensors
            # Each tensor shape: (1, seq_len, hidden_dim) for step 0,
            # or (1, 1, hidden_dim) for subsequent steps

            for layer_idx in range(n_layers):
                h = step_hidden[layer_idx + 1]  # +1 because index 0 is embedding
                # Take last token
                h_last = h[:, -1:, :]  # (1, 1, hidden_dim)

                # h_norm
                h_norm_val = float(h_last.float().norm().cpu().item())
                h_norms[layer_idx].append(h_norm_val)

                # LL entropy (raw lm_head)
                ll_logits = model.lm_head(h_last).float().squeeze(0)  # (1, vocab)
                ll_ent = compute_entropy(ll_logits)
                ll_entropies[layer_idx].append(ll_ent)

                # logit_std
                logit_std_val = float(ll_logits.std().cpu().item())
                logit_stds[layer_idx].append(logit_std_val)

                # TL entropy
                tl_logits = lens(h_last.squeeze(0), idx=layer_idx).float()  # (1, vocab)
                tl_ent = compute_entropy(tl_logits)
                tl_entropies[layer_idx].append(tl_ent)

        # Average over generated tokens
        sample_result = {
            "idx": idx,
            "is_correct": bool(is_correct),
            "subject": subject,
            "n_gen_tokens": n_gen,
            "time": round(time.time() - t0, 2),
            "layer_data": {},
        }

        for l in range(n_layers):
            sample_result["layer_data"][str(l)] = {
                "tl_entropy": round(float(np.mean(tl_entropies[l])) if tl_entropies[l] else 0, 6),
                "ll_entropy": round(float(np.mean(ll_entropies[l])) if ll_entropies[l] else 0, 6),
                "logit_std": round(float(np.mean(logit_stds[l])) if logit_stds[l] else 0, 6),
                "h_norm": round(float(np.mean(h_norms[l])) if h_norms[l] else 0, 6),
            }

        completed.append(sample_result)

        # Checkpoint
        if len(completed) % CHECKPOINT_INTERVAL == 0 or len(completed) == n_total:
            with open(ckpt_path, "w") as f:
                json.dump(completed, f)
            elapsed = sum(s["time"] for s in completed)
            acc = sum(1 for s in completed if s["is_correct"]) / len(completed)
            eta = (n_total - len(completed)) * (elapsed / len(completed)) / 3600
            print(f"  [{len(completed)}/{n_total}] acc={acc:.3f} elapsed={elapsed:.0f}s ETA={eta:.1f}h")

    # Build final results
    results = _build_final(completed, config, model_key)
    with open(final_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {final_path}")

    # Cleanup GPU
    del model, lens
    torch.cuda.empty_cache()

    return results


def _build_final(completed, config, model_key):
    """Build final results with AUROC analysis."""
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    n = len(completed)
    n_layers = config["n_layers"]
    y = np.array([1 if s["is_correct"] else 0 for s in completed])

    tl_ent = np.zeros((n, n_layers))
    ll_ent = np.zeros((n, n_layers))
    logit_std = np.zeros((n, n_layers))
    h_norm = np.zeros((n, n_layers))

    for i, s in enumerate(completed):
        for l in range(n_layers):
            ld = s["layer_data"][str(l)]
            tl_ent[i, l] = ld["tl_entropy"]
            ll_ent[i, l] = ld["ll_entropy"]
            logit_std[i, l] = ld["logit_std"]
            h_norm[i, l] = ld["h_norm"]

    # 70/30 split
    np.random.seed(SEED)
    perm = np.random.permutation(n)
    cal_idx, test_idx = perm[:int(0.7 * n)], perm[int(0.7 * n):]

    # Best single-layer AUROC
    def best_single(X):
        best_l, best_s, best_cal = -1, 1, 0.5
        for l in range(n_layers):
            for sign in [1, -1]:
                try:
                    auc = roc_auc_score(y[cal_idx], sign * X[cal_idx, l])
                    if auc > best_cal:
                        best_cal, best_l, best_s = auc, l, sign
                except:
                    pass
        test_auc = roc_auc_score(y[test_idx], best_s * X[test_idx, best_l])
        return best_l, best_s, round(float(best_cal), 4), round(float(test_auc), 4)

    tl_l, tl_s, tl_cal, tl_test = best_single(tl_ent)
    ll_l, ll_s, ll_cal, ll_test = best_single(ll_ent)
    std_l, std_s, std_cal, std_test = best_single(logit_std)
    hn_l, hn_s, hn_cal, hn_test = best_single(h_norm)

    # Incremental utility: logit_std + H_tl
    feat_std = logit_std[:, std_l:std_l+1]
    feat_tl = tl_ent[:, tl_l:tl_l+1]

    sc1 = StandardScaler()
    sc2 = StandardScaler()
    X_std_cal = sc1.fit_transform(feat_std[cal_idx])
    X_std_test = sc1.transform(feat_std[test_idx])
    X_tl_cal = sc2.fit_transform(feat_tl[cal_idx])
    X_tl_test = sc2.transform(feat_tl[test_idx])

    lr1 = LogisticRegression(max_iter=1000, random_state=SEED)
    lr1.fit(X_std_cal, y[cal_idx])
    auc_std_only = roc_auc_score(y[test_idx], lr1.predict_proba(X_std_test)[:, 1])

    X_both_cal = np.hstack([X_std_cal, X_tl_cal])
    X_both_test = np.hstack([X_std_test, X_tl_test])
    lr2 = LogisticRegression(max_iter=1000, random_state=SEED)
    lr2.fit(X_both_cal, y[cal_idx])
    auc_both = roc_auc_score(y[test_idx], lr2.predict_proba(X_both_test)[:, 1])

    delta = auc_both - auc_std_only

    n_correct = int(y.sum())
    results = {
        "model": model_key,
        "n_samples": n,
        "n_correct": n_correct,
        "accuracy": round(float(n_correct / n), 4),
        "single_layer_auroc": {
            "H_tl": {"layer": tl_l, "sign": tl_s, "cal": tl_cal, "test": tl_test},
            "H_pre": {"layer": ll_l, "sign": ll_s, "cal": ll_cal, "test": ll_test},
            "logit_std": {"layer": std_l, "sign": std_s, "cal": std_cal, "test": std_test},
            "h_norm": {"layer": hn_l, "sign": hn_s, "cal": hn_cal, "test": hn_test},
        },
        "incremental_utility": {
            "logit_std_only": round(float(auc_std_only), 4),
            "logit_std_plus_H_tl": round(float(auc_both), 4),
            "delta": round(float(delta), 4),
        },
        "scale_dominance_holds": bool(std_test >= tl_test),
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n  --- {model_key.upper()} Results ---")
    print(f"  H_tl:      L{tl_l} sign={tl_s:+d} test={tl_test:.4f}")
    print(f"  H_pre(LL): L{ll_l} sign={ll_s:+d} test={ll_test:.4f}")
    print(f"  logit_std: L{std_l} sign={std_s:+d} test={std_test:.4f}")
    print(f"  h_norm:    L{hn_l} sign={hn_s:+d} test={hn_test:.4f}")
    print(f"  Incremental: logit_std={auc_std_only:.4f}, +H_tl={auc_both:.4f}, delta={delta:+.4f}")
    print(f"  Scale dominance holds: {results['scale_dominance_holds']}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", choices=["qwen", "llama", "mistral", "all"])
    args = parser.parse_args()

    set_seed()

    print("=" * 60)
    print("TUNED LENS CONTROL EXPERIMENT")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)

    models = [args.model] if args.model != "all" else ["qwen", "llama", "mistral"]
    all_results = {}

    for model_key in models:
        config = MODEL_CONFIGS[model_key]
        result = run_model(model_key, config)
        all_results[model_key] = result

    # Summary
    print("\n" + "=" * 60)
    print("CROSS-MODEL SUMMARY")
    print("=" * 60)
    print(f"  {'Model':8s} | {'H_tl':>8s} | {'H_pre':>8s} | {'logit_std':>10s} | {'delta':>8s} | {'Scale dom.':>10s}")
    for m in models:
        r = all_results[m]
        sl = r["single_layer_auroc"]
        inc = r["incremental_utility"]
        print(f"  {m:8s} | {sl['H_tl']['test']:8.4f} | {sl['H_pre']['test']:8.4f} | {sl['logit_std']['test']:10.4f} | {inc['delta']:+8.4f} | {'YES' if r['scale_dominance_holds'] else 'NO':>10s}")

    # Save summary
    summary_path = OUTPUT / "tl_control_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
