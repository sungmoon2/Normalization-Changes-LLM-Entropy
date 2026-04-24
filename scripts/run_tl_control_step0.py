"""
Tuned Lens Control — Step 0 (Prompt-Only Forward Pass)

Purpose: Verify that scale-dominance holds under Tuned Lens decoder.
         Uses prompt-only forward pass (no generation) for efficiency.

Protocol:
  1. Load MMLU items (same np.random.choice as exp31)
  2. Forward pass on prompt only → extract hidden states at all layers
  3. Compute LL entropy (H_pre), TL entropy (H_tl), logit_std, h_norm
  4. Compare AUROC: H_tl vs logit_std vs H_pre
  5. Incremental utility: logit_std + H_tl vs logit_std only

Why Step 0 is rigorous:
  - Decoder-only causal attention: prompt-last hidden state is identical
    whether extracted during generation or via prompt-only forward pass
  - No generation-length confounding
  - Same scientific question: "does logit_std dominate entropy under TL?"

Estimated time: ~15-30 min per model (vs 16-33h for generation approach)
GPU required. Checkpoint every 100 samples. Resume-safe.
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

BASE = POT_DIR / "experiments"
OUTPUT = BASE / "46_Tuned_Lens_Control"
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

# Same MMLU sampling as exp31
EXP31_PATHS = {
    "qwen": BASE / "31_MMLU_Domain_Extension/EXP_20260219_053638_mmlu_qwen/data/sample_results.json",
    "llama": BASE / "31_MMLU_Domain_Extension/EXP_20260219_171237_mmlu_llama/data/sample_results.json",
    "mistral": BASE / "31_MMLU_Domain_Extension/EXP_20260220_000610_mmlu_mistral/data/sample_results.json",
}

CHECKPOINT_INTERVAL = 100


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


def run_model(model_key):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tuned_lens import TunedLens
    from datasets import load_dataset

    config = MODEL_CONFIGS[model_key]
    model_dir = OUTPUT / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_dir / "step0_checkpoint.json"
    final_path = model_dir / "step0_tl_control.json"

    if final_path.exists():
        print(f"  {model_key}: Already completed, skipping.")
        return json.load(open(final_path))

    # Resume
    completed = []
    if ckpt_path.exists():
        completed = json.load(open(ckpt_path))
        print(f"  Resuming: {len(completed)} done")

    # Load exp31 ground truth for correctness labels
    exp31 = json.load(open(EXP31_PATHS[model_key]))

    # Load MMLU (same sampling)
    print(f"  Loading MMLU dataset...")
    sys.stdout.flush()
    mmlu = load_dataset("cais/mmlu", "all", split="test")
    mmlu_indices = np.random.RandomState(SEED).choice(len(mmlu), 1000, replace=False)

    n_total = 1000
    n_done = len(completed)
    if n_done >= n_total:
        return _finalize(completed, config, model_key, final_path)

    # Patch + load model
    if model_key in ("qwen", "mistral"):
        patch_tuned_lens()

    n_layers = config["n_layers"]
    print(f"\n{'='*60}")
    print(f"  {model_key.upper()} TL Control (Step 0, prompt-only)")
    print(f"  Samples: {n_done}/{n_total}")
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

    vocab_size = model.config.vocab_size
    log_vocab = float(np.log(vocab_size))

    print(f"  Model + TL loaded. Starting inference...")
    sys.stdout.flush()

    for idx in range(n_done, n_total):
        t0 = time.time()

        mmlu_idx = int(mmlu_indices[idx])
        item = mmlu[mmlu_idx]
        subject = item["subject"]
        question = item["question"]
        choices = item["choices"]
        answer_idx = item["answer"]

        # Same prompt format as exp31
        choices_text = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        prompt = f"Answer the following multiple choice question about {subject}.\n\n{question}\n{choices_text}\n\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Extract prompt-last hidden state at each layer
        layer_data = {}
        for l in range(n_layers):
            h = outputs.hidden_states[l + 1][:, -1, :]  # (1, hidden_dim)

            # h_norm
            h_norm_val = float(h.float().norm().cpu().item())

            # LL entropy (H_pre)
            ll_logits = model.lm_head(h).float().squeeze(0)  # (vocab,)
            ll_probs = torch.softmax(ll_logits, dim=-1)
            ll_ent = float(-(ll_probs * torch.log(ll_probs + 1e-30)).sum().cpu().item()) / log_vocab

            # logit_std
            logit_std_val = float(ll_logits.std().cpu().item())

            # TL entropy
            tl_logits = lens(h, idx=l).float().squeeze(0)  # (vocab,)
            tl_probs = torch.softmax(tl_logits, dim=-1)
            tl_ent = float(-(tl_probs * torch.log(tl_probs + 1e-30)).sum().cpu().item()) / log_vocab

            # H_post (normed LL)
            h_normed = model.model.norm(h)
            post_logits = model.lm_head(h_normed).float().squeeze(0)
            post_probs = torch.softmax(post_logits, dim=-1)
            post_ent = float(-(post_probs * torch.log(post_probs + 1e-30)).sum().cpu().item()) / log_vocab

            layer_data[str(l)] = {
                "tl_entropy": round(tl_ent, 6),
                "ll_entropy": round(ll_ent, 6),
                "post_entropy": round(post_ent, 6),
                "logit_std": round(logit_std_val, 6),
                "h_norm": round(h_norm_val, 6),
            }

        # Use exp31 correctness
        is_correct = exp31[idx]["is_correct"]

        completed.append({
            "idx": idx,
            "is_correct": bool(is_correct),
            "subject": subject,
            "time": round(time.time() - t0, 3),
            "layer_data": layer_data,
        })

        if len(completed) % CHECKPOINT_INTERVAL == 0 or len(completed) == n_total:
            with open(ckpt_path, "w") as f:
                json.dump(completed, f)
            elapsed = sum(s["time"] for s in completed)
            avg = elapsed / len(completed)
            eta = (n_total - len(completed)) * avg / 60
            acc = sum(1 for s in completed if s["is_correct"]) / len(completed)
            print(f"  [{len(completed)}/{n_total}] avg={avg:.2f}s/sample, ETA={eta:.1f}min")
            sys.stdout.flush()

    result = _finalize(completed, config, model_key, final_path)

    del model, lens
    torch.cuda.empty_cache()
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
    for name, X in [("H_tl", tl), ("H_pre", ll), ("H_post", hp), ("logit_std", ls), ("h_norm", hn)]:
        bl, bs, bc, ta = best_single(X)
        metrics[name] = {"layer": bl, "sign": bs, "cal": bc, "test": ta}
        print(f"  {name:10s}: L{bl} sign={bs:+d} cal={bc:.4f} test={ta:.4f}")

    # Incremental: logit_std + H_tl
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
        "accuracy": round(float(y.mean()), 4),
        "position": "step0_prompt_last",
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
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", choices=["qwen", "llama", "mistral", "all"])
    args = parser.parse_args()

    set_seed()
    print("=" * 60)
    print("TUNED LENS CONTROL (Step 0, prompt-only)")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    sys.stdout.flush()

    models = [args.model] if args.model != "all" else ["qwen", "llama", "mistral"]
    results = {}
    for m in models:
        results[m] = run_model(m)

    if len(results) > 1:
        print("\n" + "=" * 60)
        print("CROSS-MODEL SUMMARY")
        print(f"  {'Model':8s} | {'H_tl':>8s} | {'H_pre':>8s} | {'logit_std':>10s} | {'delta':>8s} | {'Dominance':>10s}")
        for m in models:
            r = results[m]
            sl = r["single_layer"]
            inc = r["incremental"]
            print(f"  {m:8s} | {sl['H_tl']['test']:8.4f} | {sl['H_pre']['test']:8.4f} | {sl['logit_std']['test']:10.4f} | {inc['delta']:+8.4f} | {'YES' if r['scale_dominance_holds'] else 'NO':>10s}")

        with open(OUTPUT / "tl_control_summary.json", "w") as f:
            json.dump(results, f, indent=2)

    print("\nDONE.")


if __name__ == "__main__":
    main()
