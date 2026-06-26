# June 2026 revision artifacts

This directory contains public, repository-relative artifacts added for the IEEE Access revision. Local execution paths from reviewer-response work areas were omitted from the public JSON copies; measured values, arrays, split summaries, and CSV rows were preserved.

## Contents

- `fp32_precision_control/full/`: FP32 control outputs for Qwen, Llama, and Mistral. Each model directory contains `intervention_analysis.json`, `intervention_raw_data.json`, `resume_state.json`, `sample_manifest_used.json`, and 500 `sample_results/sample_XXXX.json` files.
- `fp32_precision_control/raw_sample_comparison/`: Qwen/Llama original-vs-FP32 raw sample comparison outputs and max-absolute-difference summaries. `raw_comparison_summary_public.json` is a path-sanitized copy of the local audit summary.
- `mistral_exception_stability/`: repeated-split audit outputs for the Mistral exception analysis. The JSON is path-sanitized; the CSV rows are copied from the measured output.
- `greedy_repeated_split/`: greedy repeated-split sign-stability audit outputs. The JSON is path-sanitized; the comparison CSV is copied from the measured output.

Public script copies for these revision artifacts are in `scripts/revision_2026_06/`.
