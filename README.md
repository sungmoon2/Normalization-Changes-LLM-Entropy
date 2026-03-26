# Normalization-Changes-LLM-Entropy

Code and data for:

> **In RMSNorm-Based Language Models, Pre- vs. Post-Normalization Changes What Layerwise Decoded Entropy Measures**
>
> Sungmoon Park, Jinhong Yang
>
> Department of Healthcare IT, Inje University

## Overview

Layerwise decoded entropy is increasingly used as an internal uncertainty proxy in language models. We show that in RMSNorm-based architectures, entropy computed before versus after normalization captures qualitatively different information: pre-normalization entropy (H_pre) is structurally scale-sensitive, while post-normalization entropy (H_post) is largely scale-invariant. Existing work is inconsistent on this implementation choice, and none reports scale baselines alongside entropy.

**Models**: Qwen2.5-7B-Instruct, Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3

**Benchmarks**: MATH (competition_math), MMLU

## Repository Structure

```
.
├── scripts/                     # Core experiment scripts
│   ├── run_phase3_scale_intervention.py    # Section 4.2: Scale intervention (Qwen)
│   ├── run_phase3_cross_model.py           # Section 4.3: Cross-model verification
│   ├── run_phase0_caltest_baselines.py     # Section 5: Held-out baselines (Table 1-3)
│   ├── run_phase0b_norm_binned_control.py  # Appendix C: Norm-binned controls
│   ├── run_phase1_token_position.py        # Section 6.1: Token position ablation
│   ├── run_phase2_entropy_lens_baseline.py # Section 7.2: Classifier comparison
│   ├── run_phase2b_tuned_lens.py           # Section 7.1: Tuned Lens faithfulness
│   ├── run_phase4_tl_discrimination.py     # Section 7.1: TL discrimination
│   ├── run_phase4_fair_sc.py               # Section 5: Self-consistency comparison
│   ├── run_incremental_utility_test.py     # Section 5.2: Incremental utility (Table 3)
│   ├── run_repeated_split.py               # Appendix I: Repeated-split robustness
│   ├── run_mmlu_entropy.py                 # MMLU experiments (3 models)
│   ├── run_normed_entropy_by_difficulty.py # Math difficulty experiments (Qwen)
│   ├── run_llama_normed_entropy.py         # Math experiments (Llama)
│   └── generate_figures_v3.py              # Figures 1-7
│
├── paper/
│   ├── draft_v12.md              # Paper (English)
│   ├── draft_v12_ko.md           # Paper (Korean)
│   └── figures/                  # Figures 1-7
│
├── results/                      # Key experiment results (JSON)
│   ├── phase0_caltest_baselines/ # Section 5 baselines
│   ├── phase3_scale_intervention/# Section 4 interventions
│   ├── phase3_cross_model/       # Section 4.3 cross-model
│   ├── mmlu_postprocessing/      # MMLU analysis
│   └── repeated_split/           # Appendix I robustness
│
├── LICENSE
└── README.md
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- 1x NVIDIA GPU with >= 24 GB VRAM (tested on RTX 3090 Ti)

```bash
pip install torch transformers scikit-learn numpy scipy tqdm matplotlib
```

## Reproducing Main Results

### Section 4: Scale Intervention (Core Finding)

```bash
# Qwen case study (Table in Section 4.2)
python scripts/run_phase3_scale_intervention.py

# Cross-model verification (Table 4)
python scripts/run_phase3_cross_model.py
```

### Section 5: Baseline Evaluation (Tables 1-3)

```bash
python scripts/run_phase0_caltest_baselines.py
```

### Section 5.2: Incremental Utility (Table 3)

```bash
python scripts/run_incremental_utility_test.py
```

### Appendix I: Repeated-Split Robustness (Table A10)

```bash
python scripts/run_repeated_split.py
```

## Hardware

All experiments were run on a single NVIDIA RTX 3090 Ti (24 GB VRAM) with FP16 inference. Seed = 42, `cudnn.deterministic = True`, `cudnn.benchmark = False`.

## Citation

```bibtex
@article{park2026normalization,
  title={In RMSNorm-Based Language Models, Pre- vs. Post-Normalization Changes What Layerwise Decoded Entropy Measures},
  author={Park, Sungmoon and Yang, Jinhong},
  year={2026}
}
```

## License

MIT
