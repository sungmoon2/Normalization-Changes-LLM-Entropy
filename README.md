# Normalization-Changes-LLM-Entropy

Code and data for:

> **Decoded Entropy Is Not One Signal: Pre- and Post-RMSNorm Projections Yield Non-Equivalent Measurements in RMSNorm-Based Decoder LMs**
>
> Sungmoon Park, Jinhong Yang
>
> Department of Healthcare IT, Inje University

*This paper is currently under review at IEEE Access.*

## Overview

Layerwise decoded entropy is increasingly used as an internal uncertainty proxy in language models. We show that in RMSNorm-based architectures, entropy computed before versus after normalization captures qualitatively different information: pre-normalization entropy (H\_pre) is structurally scale-sensitive, while post-normalization entropy (H\_post) is largely scale-invariant. Existing work is inconsistent on this implementation choice, and none reports scale baselines alongside entropy.

**Models**: Qwen2.5-7B-Instruct, Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3

**Benchmarks**: MMLU, competition\_math, ARC-Challenge, TruthfulQA mc1

## Repository Structure

```
.
├── scripts/                          # Experiment and figure generation scripts
│   ├── _paths.py                     # Shared path configuration
│   ├── run_phase3_scale_intervention.py    # Section IV-B: Qwen scale intervention
│   ├── run_phase3_cross_model.py           # Section IV-C: Cross-model verification
│   ├── run_phase3_unified.py               # Section IV-C: Unified 3-model intervention
│   ├── run_phase0_caltest_baselines.py     # Section V-A: Held-out baselines (Table 2)
│   ├── run_incremental_utility_test.py     # Section V-B: Incremental utility (Table 3)
│   ├── run_greedy_mmlu.py                  # Section V-D: Greedy decoding (Table 4)
│   ├── run_repeated_split.py               # Section V-C: 20-split robustness (Table 14)
│   ├── run_exp52_full.py                   # Section V-C: Multi-seed generation
│   ├── run_exp53_full.py                   # Appendix F-E: TruthfulQA extension
│   ├── run_tokenpos_rerun_aligned.py       # Section VI-A: Token position (Table 12)
│   ├── run_phase1_token_position.py        # Section VI-A: Token position ablation
│   ├── run_tuned_lens.py                   # Section VII-A: Tuned Lens training
│   ├── run_tuned_lens_control.py           # Section VII-E: TL control
│   ├── run_tl_control_genavg.py            # Section VII-E: TL control (gen-avg)
│   ├── run_tl_control_step0.py             # Section VII-E: TL control (Step 0)
│   ├── run_phase2_entropy_lens_baseline.py # Section VII-B: Classifier comparison
│   ├── run_entropy_lens_reeval.py          # Section VII-C: EL re-evaluation (Table 5)
│   ├── run_entropy_lens_exact_reproduction.py  # Section VII-F: EL reproduction (Table 15)
│   ├── run_entropy_lens_exact_reproduction_32b.py # Section VII-F: Llama-3.2-3B
│   ├── run_selective_prediction_v2.py      # Section VII-C: Selective prediction
│   ├── run_length_controlled_analysis.py   # Appendix F-B: Length confound control
│   ├── run_deterministic_labels.py         # Section V-D: Greedy labels
│   ├── run_phase0b_norm_binned_control.py  # Appendix C: Norm-binned controls (Table 11)
│   ├── run_mmlu_entropy.py                 # MMLU experiments (3 models)
│   ├── run_normed_entropy_by_difficulty.py  # Math difficulty experiments (Qwen)
│   ├── run_llama_normed_entropy.py          # Math experiments (Llama)
│   ├── run_phase4_fair_sc.py               # Self-consistency comparison
│   ├── run_phase4_tl_discrimination.py     # TL discrimination
│   ├── run_mmlu_postprocessing.py          # MMLU postprocessing (8 analyses)
│   ├── run_sampling_baseline.py            # Sampling baseline
│   ├── run_cpu_supplements.py              # CPU-based supplementary analyses
│   ├── run_phase1_analysis.py              # Signal analysis
│   ├── run_exp54_full.py                   # Practical consequence analysis
│   ├── run_exp55_full.py                   # Radial/angular decomposition
│   ├── run_b2b3b4_chain.py                 # CI chain analysis
│   ├── run_selective_prediction_divergence.py  # Selective prediction divergence
│   ├── run_exp49_extras.py                 # Label robustness extras
│   ├── run_exp49_phase2.py                 # Label robustness phase 2
│   ├── run_exp49_phase3.py                 # Label robustness phase 3
│   ├── entropy_measurement_audit.py        # Diagnostic measurement audit tool
│   ├── generate_figures_v4.py              # Main figure generation (Figs. 1-6)
│   ├── generate_fig1_v4.py                 # Figure 1: Conceptual overview
│   ├── generate_fig4_v5_3model.py          # Figure 4: Baseline comparison (3 models)
│   ├── generate_fig7_v3_final.py           # Figure 7: Token position
│   ├── generate_perlayer_auroc_curves.py   # Per-layer AUROC curves
│   └── generate_post_tokenpos.py           # Post token-position visualization
│
├── paper/
│   ├── manuscript.tex            # Paper source (IEEE Access format)
│   └── figures/                  # Figures 1-7 (final versions)
│
├── results/                      # Key experiment results (JSON)
│   ├── phase0_caltest_baselines/ # Section V baselines
│   ├── phase3_scale_intervention/# Section IV interventions
│   ├── phase3_cross_model/       # Section IV-C cross-model
│   ├── mmlu_postprocessing/      # MMLU analysis
│   ├── repeated_split/           # Section V-C robustness
│   ├── exp52_multi_seed/         # Section V-C multi-seed (seeds 123, 456)
│   ├── exp53_truthfulqa/         # Appendix F-E TruthfulQA extension
│   ├── exp54_practical/          # Practical consequence analysis
│   ├── exp55_decomposition/      # Section IV-F radial/angular decomposition
│   └── exp56_ci_chain/           # CI chain analysis
│
├── requirements.txt
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

### Section IV: Scale Intervention (Core Finding)

```bash
# Qwen case study (Section IV-B, Table 1)
python scripts/run_phase3_scale_intervention.py

# Cross-model verification (Section IV-C, Table 1)
python scripts/run_phase3_cross_model.py

# Unified 3-model intervention (Section IV-C)
python scripts/run_phase3_unified.py
```

### Section V: Baseline Evaluation (Tables 2-4)

```bash
# Held-out baselines (Table 2)
python scripts/run_phase0_caltest_baselines.py

# Incremental utility (Table 3)
python scripts/run_incremental_utility_test.py

# Greedy decoding (Table 4)
python scripts/run_greedy_mmlu.py

# 20-split robustness (Table 14)
python scripts/run_repeated_split.py

# Multi-seed generation (Section V-C)
python scripts/run_exp52_full.py
```

### Section VI: Token Position (Table 12, Figure 7)

```bash
python scripts/run_tokenpos_rerun_aligned.py
```

### Section VII: Decoder Faithfulness

```bash
# Tuned Lens training (Section VII-A, Table 9)
python scripts/run_tuned_lens.py

# TL control (Section VII-E)
python scripts/run_tl_control_genavg.py
python scripts/run_tl_control_step0.py

# Classifier comparison (Section VII-B, Figure 6b)
python scripts/run_phase2_entropy_lens_baseline.py

# EL re-evaluation (Section VII-C, Table 5)
python scripts/run_entropy_lens_reeval.py

# Protocol-matched reproduction (Section VII-F, Table 15)
python scripts/run_entropy_lens_exact_reproduction.py
```

### Appendix: Robustness Controls

```bash
# Norm-binned controls (Appendix C, Table 11)
python scripts/run_phase0b_norm_binned_control.py

# Length confound control (Appendix F-B)
python scripts/run_length_controlled_analysis.py
```

## Hardware

All experiments were run on a single NVIDIA RTX 3090 Ti (24 GB VRAM) with FP16 inference. Seed = 42, `cudnn.deterministic = True`, `cudnn.benchmark = False`.

## Citation

```bibtex
@article{park2026decoded,
  title={Decoded Entropy Is Not One Signal: Pre- and Post-RMSNorm Projections Yield Non-Equivalent Measurements in RMSNorm-Based Decoder LMs},
  author={Park, Sungmoon and Yang, Jinhong},
  journal={IEEE Access},
  year={2026},
  note={Under review}
}
```

## Acknowledgments

This work was supported by the Korea Institute for Advancement of Technology (KIAT) grant funded by the Korea government (Ministry of Trade, Industry and Energy) through the International Cooperation in Industrial Technology program (Project Number: P0026190) and Institute of Information & Communications Technology Planning & Evaluation (IITP)-Innovative Human Resource Development for Local Intellectualization program grant funded by the Korea government (MSIT) (IITP-2026-RS-2024-00436773).

## License

MIT
