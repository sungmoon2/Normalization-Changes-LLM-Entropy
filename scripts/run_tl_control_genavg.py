#!/usr/bin/env python3
"""
Tuned Lens Control - Generation-Average (200 samples per model)

Verifies scale-dominance under Tuned Lens at generation-average position,
where signal is strong (AUROC 0.6-0.8), not just Step 0 (AUROC ~0.5).

Approach (2-pass for efficiency):
  Pass 1: Generate response (normal, no hidden states) → save tokens
  Pass 2: Forward pass on prompt+response → extract hidden states at
           generated positions → apply TL → compute TL entropy

Why 2-pass is valid:
  Causal attention ensures hidden state at position t is identical whether
  computed during autoregressive generation or in a single forward pass,
  because position t only attends to tokens 0..t in both cases.

200 samples, ~25-40 min per model. Checkpoint every 20 samples. Resume-safe.
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
N_SAMPLES = 1000
MAX_NEW_TOKENS = 512
CHECKPOINT_INTERVAL = 20

BASE = POT_DIR / "experiments"
OUTPUT = BASE / "46_Tuned_Lens_Control" / "genavg"
OUTPUT.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = {
    "qwen": {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "n_layers": 28,
        "lens_path": str(BASE / "35_Phase2b_Tuned_Lens/qwen_tuned_lens"),
    },
    "llama": {
        "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "n_layers": 32,
        "lens_path": str(BASE / "44_GPT25_Experiments/phase3_gpu_tunedlens/EXP3a_llama_tuned_lens/llama_tuned_lens"),
    },
    "mistral": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "n_layers": 32,
        "lens_path": str(BASE / "44_GPT25_Experiments/phase3_gpu_tunedlens/EXP3b_mistral_tuned_lens/mistral_tuned_lens"),
    },
}

EXP31_PATHS = {
    "qwen": BASE / "31_MMLU_Domain_Extension/EXP_20260219_053638_mmlu_qwen/data/sample_results.json",
    "llama": BASE / "31_MMLU_Domain_Extension/EXP_20260219_171237_mmlu_llama/data/sample_results.json",
    "mistral": BASE / "31_MMLU_Domain_Extension/EXP_20260220_000610_mmlu_mistral/data/sample_results.json",
}


def set_seed():
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def patch_tuned_lens():
    import tuned_lens.model_surgery as ms
    original = ms.get_final_norm
    PATCHED = ('Qwen2Model', 'Qwen2ForCausalLM', 'MistralModel', 'MistralForCausalLM')
    def patched(model):
        base = model.base_model
        if hasattr(base, 'norm') and type(base).__name__ in PATCHED:
            return base.norm
        return original(model)
    ms.get_final_norm = patched


def build_prompt(subject, question, choices):
    choices_text = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
    return f"Answer the following multiple choice question about {subject}.\n\n{question}\n{choices_text}\n\nAnswer:"


def process_sample(model, tokenizer, lens, prompt, n_layers, log_vocab):
    """Two-pass: generate then forward pass for hidden states."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    # Pass 1: Generate response
    with torch.no_grad():
        gen_output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.3,
            do_sample=True,
        )
    full_ids = gen_output[0]  # (total_len,)
    gen_len = len(full_ids) - prompt_len
    if gen_len <= 0:
        return None, 0

    # Pass 2: Forward pass on full sequence (prompt + response)
    with torch.no_grad():
        outputs = model(full_ids.unsqueeze(0), output_hidden_states=True)

    # Extract metrics at generated token positions (prompt_len to end)
    layer_data = {}
    for l in range(n_layers):
        h_gen = outputs.hidden_states[l + 1][0, prompt_len:, :]  # (gen_len, hidden)

        # Generation-average metrics
        h_mean_norm = float(h_gen.float().norm(dim=-1).mean().cpu().item())

        # LL entropy (H_pre) - generation average
        ll_logits = model.lm_head(h_gen).float()  # (gen_len, vocab)
        ll_probs = torch.softmax(ll_logits, dim=-1)
        ll_ent_per_token = -(ll_probs * torch.log(ll_probs + 1e-30)).sum(dim=-1) / log_vocab
        ll_ent = float(ll_ent_per_token.mean().cpu().item())

        # logit_std - generation average
        logit_std_val = float(ll_logits.std(dim=-1).mean().cpu().item())

        # H_post - generation average
        h_normed = model.model.norm(h_gen)
        post_logits = model.lm_head(h_normed).float()
        post_probs = torch.softmax(post_logits, dim=-1)
        post_ent_per_token = -(post_probs * torch.log(post_probs + 1e-30)).sum(dim=-1) / log_vocab
        post_ent = float(post_ent_per_token.mean().cpu().item())

        # TL entropy - generation average
        tl_ent_accum = 0.0
        # Process in chunks to avoid OOM on lens
        chunk_size = 64
        for start in range(0, gen_len, chunk_size):
            end = min(start + chunk_size, gen_len)
            h_chunk = h_gen[start:end]
            tl_logits = lens(h_chunk, idx=l).float()  # (chunk, vocab)
            tl_probs = torch.softmax(tl_logits, dim=-1)
            tl_ent_chunk = -(tl_probs * torch.log(tl_probs + 1e-30)).sum(dim=-1) / log_vocab
            tl_ent_accum += float(tl_ent_chunk.sum().cpu().item())
        tl_ent = tl_ent_accum / gen_len

        layer_data[str(l)] = {
            "tl_entropy": round(tl_ent, 6),
            "ll_entropy": round(ll_ent, 6),
            "post_entropy": round(post_ent, 6),
            "logit_std": round(logit_std_val, 6),
            "h_norm": round(h_mean_norm, 6),
        }

    return layer_data, gen_len


def run_model(model_key, test_mode=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tuned_lens import TunedLens
    from datasets import load_dataset

    config = MODEL_CONFIGS[model_key]
    n_layers = config["n_layers"]
    model_dir = OUTPUT / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_dir / "checkpoint.json"
    final_path = model_dir / "tl_genavg_results.json"

    n_target = 2 if test_mode else N_SAMPLES

    if final_path.exists() and not test_mode:
        print(f"  {model_key}: Already completed.")
        sys.stdout.flush()
        return json.load(open(final_path))

    # Resume
    completed = []
    if ckpt_path.exists() and not test_mode:
        completed = json.load(open(ckpt_path))
        print(f"  Resuming: {len(completed)} done")
        sys.stdout.flush()

    # Load exp31 for correctness labels
    exp31 = json.load(open(EXP31_PATHS[model_key]))

    # Load MMLU
    print(f"  Loading MMLU...")
    sys.stdout.flush()
    mmlu = load_dataset("cais/mmlu", "all", split="test")
    mmlu_indices = np.random.RandomState(SEED).choice(len(mmlu), 1000, replace=False)

    n_done = len(completed)
    if n_done >= n_target:
        if test_mode:
            return None
        return _finalize(completed, config, model_key, final_path)

    # Patch + Load model
    if model_key in ("qwen", "mistral"):
        patch_tuned_lens()

    print(f"\n{'='*60}")
    print(f"  {model_key.upper()} TL Control (Generation-Average)")
    print(f"  Samples: {n_done}/{n_target} ({'TEST' if test_mode else 'FULL'})")
    print(f"{'='*60}")
    sys.stdout.flush()

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

    log_vocab = float(np.log(model.config.vocab_size))

    print(f"  Model + TL loaded. Starting...")
    sys.stdout.flush()

    for idx in range(n_done, n_target):
        t0 = time.time()
        set_seed()  # Reset seed each sample for reproducibility

        mmlu_idx = int(mmlu_indices[idx])
        item = mmlu[mmlu_idx]
        subject = item["subject"]
        prompt = build_prompt(subject, item["question"], item["choices"])

        layer_data, gen_len = process_sample(model, tokenizer, lens, prompt, n_layers, log_vocab)
        elapsed = round(time.time() - t0, 2)

        if layer_data is None:
            print(f"  [{idx}] WARNING: empty generation, skipping")
            sys.stdout.flush()
            continue

        completed.append({
            "idx": idx,
            "is_correct": bool(exp31[idx]["is_correct"]),
            "subject": subject,
            "n_gen_tokens": gen_len,
            "time": elapsed,
            "layer_data": layer_data,
        })

        if test_mode:
            print(f"  [TEST {idx}] gen_len={gen_len}, time={elapsed}s")
            print(f"    L0 tl_ent={layer_data['0']['tl_entropy']:.4f}, ll_ent={layer_data['0']['ll_entropy']:.4f}, logit_std={layer_data['0']['logit_std']:.4f}")
            sys.stdout.flush()
            continue

        if len(completed) % CHECKPOINT_INTERVAL == 0 or len(completed) == n_target:
            with open(ckpt_path, "w") as f:
                json.dump(completed, f)
            avg_time = sum(s["time"] for s in completed) / len(completed)
            eta = (n_target - len(completed)) * avg_time / 60
            print(f"  [{len(completed)}/{n_target}] avg={avg_time:.1f}s/sample, ETA={eta:.1f}min")
            sys.stdout.flush()

    # Cleanup
    del model, lens
    torch.cuda.empty_cache()

    if test_mode:
        # Clean test artifacts
        if ckpt_path.exists():
            ckpt_path.unlink()
        print(f"  [TEST {model_key}] PASSED. Test artifacts cleaned.")
        sys.stdout.flush()
        return None

    result = _finalize(completed, config, model_key, final_path)
    return result


def _finalize(completed, config, model_key, final_path):
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    n = len(completed)
    n_layers = config["n_layers"]
    y = np.array([1 if s["is_correct"] else 0 for s in completed])

    tl = np.zeros((n, n_layers))
    ll = np.zeros((n, n_layers))
    hp = np.zeros((n, n_layers))
    ls = np.zeros((n, n_layers))
    hn = np.zeros((n, n_layers))

    for i, s in enumerate(completed):
        for l in range(n_layers):
            ld = s["layer_data"][str(l)]
            tl[i, l] = ld["tl_entropy"]
            ll[i, l] = ld["ll_entropy"]
            hp[i, l] = ld["post_entropy"]
            ls[i, l] = ld["logit_std"]
            hn[i, l] = ld["h_norm"]

    np.random.seed(SEED)
    perm = np.random.permutation(n)
    cal, test = perm[:int(0.7*n)], perm[int(0.7*n):]

    def best_single(X):
        bl, bs, bc = -1, 1, 0.5
        for l in range(n_layers):
            for sign in [1, -1]:
                try:
                    a = roc_auc_score(y[cal], sign * X[cal, l])
                    if a > bc: bc, bl, bs = a, l, sign
                except: pass
        ta = roc_auc_score(y[test], bs * X[test, bl]) if bl >= 0 else 0.5
        return bl, bs, round(float(bc), 4), round(float(ta), 4)

    metrics = {}
    print(f"\n  --- {model_key.upper()} Generation-Average Results ---")
    for name, X in [("H_tl", tl), ("H_pre", ll), ("H_post", hp), ("logit_std", ls), ("h_norm", hn)]:
        bl, bs, bc, ta = best_single(X)
        metrics[name] = {"layer": bl, "sign": bs, "cal": bc, "test": ta}
        print(f"  {name:10s}: L{bl} sign={bs:+d} cal={bc:.4f} test={ta:.4f}")

    # Incremental utility
    feat_ls = ls[:, metrics["logit_std"]["layer"]:metrics["logit_std"]["layer"]+1]
    feat_tl = tl[:, metrics["H_tl"]["layer"]:metrics["H_tl"]["layer"]+1]
    sc1, sc2 = StandardScaler(), StandardScaler()
    Xls_c = sc1.fit_transform(feat_ls[cal]); Xls_t = sc1.transform(feat_ls[test])
    Xtl_c = sc2.fit_transform(feat_tl[cal]); Xtl_t = sc2.transform(feat_tl[test])

    lr1 = LogisticRegression(max_iter=1000, random_state=SEED)
    lr1.fit(Xls_c, y[cal])
    auc_ls = roc_auc_score(y[test], lr1.predict_proba(Xls_t)[:,1])

    lr2 = LogisticRegression(max_iter=1000, random_state=SEED)
    lr2.fit(np.hstack([Xls_c, Xtl_c]), y[cal])
    auc_both = roc_auc_score(y[test], lr2.predict_proba(np.hstack([Xls_t, Xtl_t]))[:,1])

    delta = auc_both - auc_ls
    dominance = metrics["logit_std"]["test"] >= metrics["H_tl"]["test"]

    print(f"\n  Incremental: logit_std={auc_ls:.4f}, +H_tl={auc_both:.4f}, delta={delta:+.4f}")
    print(f"  Scale dominance (logit_std >= H_tl): {dominance}")
    sys.stdout.flush()

    result = {
        "model": model_key,
        "n_samples": n,
        "n_correct": int(y.sum()),
        "accuracy": round(float(y.mean()), 4),
        "position": "generation_average",
        "single_layer": metrics,
        "incremental": {
            "logit_std_only": round(float(auc_ls), 4),
            "logit_std_plus_H_tl": round(float(auc_both), 4),
            "delta": round(float(delta), 4),
        },
        "scale_dominance_holds": dominance,
        "timestamp": datetime.now().isoformat(),
    }

    with open(final_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {final_path}")
    sys.stdout.flush()
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="full", choices=["test", "full"])
    parser.add_argument("--model", default="all", choices=["qwen", "llama", "mistral", "all"])
    args = parser.parse_args()

    set_seed()
    print("=" * 60)
    print(f"TUNED LENS CONTROL - Generation-Average ({args.mode.upper()})")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Samples per model: {2 if args.mode == 'test' else N_SAMPLES}")
    print("=" * 60)
    sys.stdout.flush()

    models = [args.model] if args.model != "all" else ["qwen", "llama", "mistral"]

    if args.mode == "test":
        for m in models:
            print(f"\n--- Testing {m} ---")
            sys.stdout.flush()
            run_model(m, test_mode=True)
        print("\nALL TESTS PASSED.")
        sys.stdout.flush()
        return

    # Full run
    results = {}
    for m in models:
        results[m] = run_model(m)

    if len(results) > 1:
        print("\n" + "=" * 60)
        print("CROSS-MODEL SUMMARY (Generation-Average)")
        print(f"  {'Model':8s} | {'H_tl':>8s} | {'H_pre':>8s} | {'logit_std':>10s} | {'delta':>8s} | {'Dominance':>10s}")
        for m in models:
            r = results[m]
            sl = r["single_layer"]
            inc = r["incremental"]
            print(f"  {m:8s} | {sl['H_tl']['test']:8.4f} | {sl['H_pre']['test']:8.4f} | {sl['logit_std']['test']:10.4f} | {inc['delta']:+8.4f} | {'YES' if r['scale_dominance_holds'] else 'NO':>10s}")

        with open(OUTPUT / "tl_genavg_summary.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSummary: {OUTPUT / 'tl_genavg_summary.json'}")

    print("\nALL DONE.")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
