# Normalization-Changes-LLM-Entropy

Code and data for:

> **Decoded Entropy Is Not One Signal: Pre- and Post-RMSNorm Projections Yield Non-Equivalent Measurements in RMSNorm-Based Decoder LMs**
>
> Sungmoon Park, Jinhong Yang
>
> Department of Healthcare IT, Inje University

*This paper is currently under review at IEEE Access.*

## Overview

Layerwise decoded entropy is increasingly used as an internal uncertainty proxy in language models. This repository supports the paper's analysis of entropy computed before versus after RMSNorm in decoder-only language models. The experiments compare pre-normalization entropy (H_pre), post-normalization entropy (H_post), scale-related baselines, profile-level predictors, robustness controls, and revision-time precision checks.

**Models:** Qwen2.5-7B-Instruct, Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3

**Benchmarks:** MMLU, competition_math, ARC-Challenge, TruthfulQA mc1

## Repository Structure

```text
.
|-- paper/
|   |-- manuscript.tex
|   `-- figures/
|-- results/
|   |-- phase0_caltest_baselines/
|   |-- phase3_scale_intervention/
|   |-- phase3_cross_model/
|   |-- mmlu_postprocessing/
|   |-- repeated_split/
|   |-- exp52_multi_seed/
|   |-- exp53_truthfulqa/
|   |-- exp54_practical/
|   |-- exp55_decomposition/
|   |-- exp56_ci_chain/
|   `-- revision_2026_06/
|       |-- fp32_precision_control/
|       |-- greedy_repeated_split/
|       `-- mistral_exception_stability/
|-- scripts/
|   |-- revision_2026_06/
|   |-- run_phase3_scale_intervention.py
|   |-- run_phase3_cross_model.py
|   |-- run_phase3_unified.py
|   |-- run_phase0_caltest_baselines.py
|   |-- run_incremental_utility_test.py
|   |-- run_greedy_mmlu.py
|   |-- run_repeated_split.py
|   |-- run_exp52_full.py
|   |-- run_exp53_full.py
|   |-- run_tokenpos_rerun_aligned.py
|   |-- run_tuned_lens.py
|   |-- run_tuned_lens_control.py
|   |-- run_tl_control_genavg.py
|   |-- run_tl_control_step0.py
|   |-- run_phase2_entropy_lens_baseline.py
|   |-- run_entropy_lens_reeval.py
|   |-- run_entropy_lens_exact_reproduction.py
|   `-- run_entropy_lens_exact_reproduction_32b.py
|-- requirements.txt
`-- LICENSE
```

## June 2026 Revision Artifacts

The `results/revision_2026_06/` directory contains the public result artifacts added for the IEEE Access revision:

- `fp32_precision_control/full/`: FP32 control outputs for Qwen, Llama, and Mistral, including model-level summaries, raw intervention data, manifests, and 500 sample-level JSON files per model.
- `fp32_precision_control/raw_sample_comparison/`: original-vs-FP32 raw sample comparison summaries for Qwen and Llama.
- `mistral_exception_stability/`: repeated-split audit outputs for the Mistral exception analysis.
- `greedy_repeated_split/`: greedy repeated-split sign-stability audit outputs.

Public, repository-relative script copies for these revision artifacts are provided in `scripts/revision_2026_06/`. Local reviewer-response paths were removed from public JSON copies; measured values and CSV rows were preserved.

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- NVIDIA GPU with at least 24 GB VRAM for the main GPU experiments

```bash
pip install torch transformers scikit-learn numpy scipy tqdm matplotlib
```

## Reproducing Main Results

### Section IV: Scale Intervention

```bash
python scripts/run_phase3_scale_intervention.py
python scripts/run_phase3_cross_model.py
python scripts/run_phase3_unified.py
```

### Section V: Baseline Evaluation

```bash
python scripts/run_phase0_caltest_baselines.py
python scripts/run_incremental_utility_test.py
python scripts/run_greedy_mmlu.py
python scripts/run_repeated_split.py
python scripts/run_exp52_full.py
```

### Section VI: Token Position

```bash
python scripts/run_tokenpos_rerun_aligned.py
```

### Section VII: Decoder Faithfulness

```bash
python scripts/run_tuned_lens.py
python scripts/run_tl_control_genavg.py
python scripts/run_tl_control_step0.py
python scripts/run_phase2_entropy_lens_baseline.py
python scripts/run_entropy_lens_reeval.py
python scripts/run_entropy_lens_exact_reproduction.py
```

### Revision Controls

```bash
python scripts/revision_2026_06/compare_fp32_original_raw_samples.py
python scripts/revision_2026_06/run_mistral_exception_repeated_split.py
```

`scripts/revision_2026_06/run_fp32_precision_control.py` reproduces the precision-control run design but requires the same model access and hardware assumptions as the original GPU experiments.

## Hardware

The main experiments were run on a single NVIDIA RTX 3090 Ti (24 GB VRAM). Revision-time FP32 control artifacts and their run metadata are provided under `results/revision_2026_06/fp32_precision_control/`.

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
