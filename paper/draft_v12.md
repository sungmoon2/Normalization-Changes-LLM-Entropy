# In RMSNorm-Based Language Models, Pre- vs. Post-Normalization Changes What Layerwise Decoded Entropy Measures

**Sungmoon Park**
Department of Healthcare IT, Inje University, Gimhae, Republic of Korea
qkrtjdans55@oasis.inje.ac.kr

**Jinhong Yang** (Corresponding author)
Department of Healthcare IT, Inje University, Gimhae, Republic of Korea
jinhong@inje.ac.kr

---

## Abstract

Layerwise decoded entropy—the Shannon entropy of intermediate hidden states projected to vocabulary space—is increasingly used as an internal uncertainty proxy in language models. However, an implementation choice has not been consistently specified: in RMSNorm-based architectures, entropy computed before versus after normalization captures qualitatively different information, and failing to distinguish them conflates hidden-state scale with directional content. We make this precise by defining pre-RMSNorm entropy (H_pre) and post-RMSNorm entropy (H_post) and studying both across three open-weight instruction-tuned decoder LMs—Qwen2.5-7B-Instruct, Llama-3-8B-Instruct, and Mistral-7B-Instruct-v0.3—on mathematical reasoning and MMLU benchmarks, using calibration/test splits, scalar baselines, token-position ablations, and lens-faithfulness checks. Controlled rescaling reveals that H_pre is structurally scale-sensitive—unit-normalizing hidden states collapses H_pre toward maximal entropy across all three model families (mean > 0.9999)—whereas H_post is largely scale-invariant in middle and late layers (alpha-sweep variation < 0.003), with residual early-layer deviations consistent with epsilon-regularized RMSNorm. Crucially, this scale-sensitivity is not merely theoretical: as single-layer scalar baselines, simple scale proxies such as logit standard deviation match or outperform single-layer H_pre for correctness discrimination in all four primary conditions, and adding H_pre to logit_std yields no significant incremental utility in either Qwen condition tested (delta = +0.003 to +0.020, 95% CI including zero)—supporting the view that H_pre's discriminative signal is largely reducible to scale. Nevertheless, carefully chosen internal layers consistently outperform final-output entropy by +0.09 to +0.22 AUROC. These results expose a normalization-dependent measurement confound in layerwise decoded entropy: prior work that reports entropy from logit-lens projections without specifying the normalization point—and existing work is inconsistent on this choice (see Table 8)—may be inadvertently reporting scale statistics rather than distributional uncertainty. We provide diagnostic criteria, actionable recommendations, and concrete implications for reinterpreting prior findings.

**Keywords:** layerwise entropy, normalization, RMSNorm, scale sensitivity, correctness discrimination, selective prediction, measurement

---

## 1. Introduction

Layerwise decoded entropy—the Shannon entropy computed from intermediate hidden states projected to vocabulary space—has attracted growing attention as an internal uncertainty proxy for large language models (Ali et al., 2025). More broadly, probing internal representations for quality and uncertainty signals has become a productive research direction (Ghandeharioun et al., 2024; Kossen et al., 2024). By examining how the model's internal probability distribution evolves across layers, researchers have identified "critical layers" and constructed entropy profiles that correlate with correctness.

However, an implementation choice with measurement consequences has not been consistently specified. In transformer architectures employing RMSNorm, each layer's hidden state passes through normalization before downstream processing. Entropy computed before versus after this normalization step may measure qualitatively different quantities—yet existing work is inconsistent on which is used: some include final normalization, some explicitly omit it, and many do not specify (see literature audit in Section 8.1, Table 8). This is not a minor implementation detail: RMSNorm removes radial scale from hidden states, and if pre-normalization entropy is dominated by that scale, then researchers using logit-lens entropy without normalization may be reporting hidden-state magnitude rather than distributional uncertainty. Any downstream conclusions—about "critical layers," entropy profiles, or uncertainty estimation—would then require reinterpretation.

This paper asks: **what does layerwise decoded entropy actually measure, and how does the normalization point change the answer?**

We define two quantities from the same hidden state h_l at layer l: **H_pre**, the entropy of softmax(W h_l) where W is the language model head applied directly; and **H_post**, the entropy of softmax(W RMSNorm(h_l)). Both are diagnostic projections—applying the final-layer decoder to intermediate representations that were not designed to be decoded at that point. Our central finding is that these two "entropies" are not the same signal. Decomposing h_l = r_l u_l into radial magnitude r_l and direction u_l, H_pre is structurally scale-sensitive because pre-norm logits scale linearly with r_l, acting as an implicit temperature. H_post, computed after RMSNorm removes radial scale, is largely invariant to magnitude rescaling—except in low-norm early layers where the implementation's epsilon term breaks exact invariance. We verify lens faithfulness caveats for these diagnostic projections in Section 7.

This distinction has concrete consequences for how prior results should be interpreted and how future experiments should be designed:

1. **Among single-layer scalar baselines, H_pre's discriminative signal is largely reducible to scale.** Direct scale statistics (logit standard deviation, hidden-state norm) match or outperform single-layer H_pre for correctness discrimination in all four primary conditions. Adding H_pre to logit_std yields no significant incremental utility in either Qwen condition tested (delta-AUROC +0.003 to +0.020, 95% CI including zero), consistent with pre-normalization entropy measuring scale rather than distributional uncertainty.

2. **Internal layers still beat final-output entropy.** Across all conditions and model families, the best internal-layer signal outperforms final-output entropy by +0.09 to +0.22 AUROC, confirming that intermediate representations carry useful correctness information beyond the output distribution—provided the normalization point is specified.

3. **Under our protocol, no universal optimal layer is observed.** The best layer, sign, and even which entropy variant is more useful all vary across model family, task, and evaluation protocol—including H_post sign reversal on Llama (higher H_post = correct in 15/20 repeated splits, vs. lower = correct on Qwen and Mistral). H_pre sign is stable (lower = correct, 20/20 splits across all conditions).

4. **Decoder and classifier choices substantially affect conclusions.** Logit Lens shows near-zero top-1 agreement with the final-layer distribution in Qwen middle layers, and switching the Entropy-Lens classifier from k-NN to logistic regression changes AUROC by up to +0.17, indicating that decoder and classifier choices are sensitive design parameters.

Our contributions are:

- **C1.** We define and distinguish H_pre and H_post, and show that the normalization point qualitatively changes the decoded entropy signal.
- **C2.** We provide direct geometric evidence across three model families that H_pre is structurally scale-sensitive while H_post is largely scale-invariant, via unit-norm removal and alpha-sweep interventions.
- **C3.** We evaluate internal correctness signals under a fair held-out protocol with 12 scalar baselines and show that, among single-layer scalars, simple scale proxies often match or outperform entropy—while internal layers consistently outperform output-only entropy.
- **C4.** We observe that conclusions are strongly dependent on layer, token position, model family, and decoding lens—including absence of a universal optimal layer and H_post sign reversal on Llama. Repeated-split analysis (20 splits) confirms sign stability for H_pre but not for H_post across all families.
- **C5.** We identify concrete implications for prior work that uses logit-lens decoded entropy without specifying the normalization point, and provide actionable diagnostic criteria for future studies.

This paper is a measurement study in the cautionary tradition: we do not propose a new method, but expose a confound that affects how existing measurements should be interpreted. We do not claim mechanistic tracing, causal mediation, universal uncertainty estimation, or superiority over self-consistency. We study what layerwise entropy measures, not where reasoning happens.

---

## 2. Related Work

### 2.1 Internal Uncertainty Signals in LLMs

Several lines of work extract uncertainty signals from LLM internals. Semantic entropy (Kuhn et al., 2023; Farquhar et al., 2024) clusters multiple sampled generations by meaning and computes distributional entropy, achieving strong hallucination detection but requiring multiple forward passes. Semantic Entropy Probes (Kossen et al., 2024) approximate this with single-pass hidden states via supervised training. DRIFT (Bhatnagar et al., 2026) trains probes on intermediate layers to detect representational inconsistencies indicative of factual errors, reporting AUROC up to 0.94. FLUE (Gao et al., 2025) establishes that hidden-state entropy provides an upper bound on predictive entropy. Ghandeharioun et al. (2024) propose Patchscopes, a framework for inspecting hidden representations by patching them into natural-language prompts for interpretation. Chen et al. (2025) propose Internal Confidence, a training-free method that assesses query-level uncertainty from internal representations before generation begins, directly relevant to pre-generation signals. Our work differs from all of these in that we do not propose a new uncertainty method; instead, we ask what the widely-used layerwise decoded entropy is actually measuring under different normalization choices.

### 2.2 Layerwise Entropy and Lens-Based Decoding

The Logit Lens (nostalgebraist, 2020) projects intermediate hidden states to vocabulary space via the language model head, enabling layer-by-layer inspection of evolving predictions. The Tuned Lens (Belrose et al., 2023) improves upon this with learned affine translators, reducing representational drift; notably, Belrose et al. explicitly define the logit lens with LayerNorm included. Geva et al. (2022) analyze how feed-forward layers promote concepts in vocabulary space, explicitly omitting LayerNorm from their projection and checking in an appendix that this omission does not substantially affect top-token interpretations—an early acknowledgment that normalization is a design choice in intermediate decoding, though they do not examine its effect on entropy semantics. Entropy-Lens (Ali et al., 2025) studies entropy dynamics of logit-lens probabilities across layers to characterize expansion and pruning decision strategies; when k-NN is used, it is framed as a diagnostic probe rather than a proposed predictive model. DoLa (Chuang et al., 2024) contrasts logit distributions between early and late layers to improve factuality, projecting intermediate hidden states through the vocabulary head without specifying normalization. SimLens (Ma et al., 2025) improves intermediate-layer decoding accuracy by eliciting predictions with one additional token, addressing lens drift at early layers. Our work directly builds on this lineage but asks a different question: rather than treating the decoded entropy as a fixed quantity, we investigate how the choice of normalization point (pre- vs. post-RMSNorm) qualitatively changes what the entropy measures. We also show that conclusions about which features are most discriminative are sensitive to classifier choice (Section 7).

### 2.3 Normalization, Scale, and Confidence

RMSNorm (Zhang & Sennrich, 2019) removes radial scale from hidden states, a property we exploit to decompose entropy into scale-dependent and direction-dependent components. Brody et al. (2023) showed that normalization interacts with linear projections to change the effective expressivity of attention. Confidence Regulation Neurons (Stolfo et al., 2024) identified specific neurons that modulate output confidence by scaling hidden-state norms, providing a mechanistic link between scale and model certainty. Sun et al. (2024) documented rare, extremely large activation outliers in LLMs with disproportionate functional impact on model behavior. Katz and Belinkov (2023) further showed that the final LayerNorm acts as a "semantic filter" in intermediate-layer decoding, defining the logit lens as LL(x) = softmax(ln_f(x) D) with normalization explicitly included and demonstrating that LayerNorm application alters token-level probability assignments. Collectively, these findings motivate our central question: if scale carries confidence-relevant information, and normalization removes scale, then pre- and post-normalization entropy must be measuring different things. Our intervention experiments (Section 4) provide direct cross-model evidence for this hypothesis.

### 2.4 Selective Prediction

Selective prediction allows a system to abstain on uncertain inputs, improving precision on the accepted subset. Geifman and El-Yaniv (2017) established the risk-coverage framework. Recent work extends this to LLMs: REFRAIN (Sun et al., 2025) uses an SW-UCB-based adaptive early-stopping framework for chain-of-thought reasoning, and self-consistency (Wang et al., 2023) uses majority voting over multiple samples as a confidence proxy. We use selective prediction not as our primary contribution but as the evaluation lens through which we assess the practical utility of different internal signals (Section 5).

---

## 3. Definitions and Evaluation Protocol

### 3.1 Models and Data

We study three instruction-tuned decoder LMs, all employing RMSNorm as their final normalization: Qwen2.5-7B-Instruct (Qwen Team, 2024), Llama-3-8B-Instruct (Grattafiori et al., 2024), and Mistral-7B-Instruct-v0.3 (Jiang et al., 2023).

| Model | Family | Layers | Hidden | Vocab |
|:------|:------:|:------:|:------:|:-----:|
| Qwen2.5-7B-Instruct | Alibaba | 28 | 3584 | 152,064 |
| Llama-3-8B-Instruct | Meta | 32 | 4096 | 128,256 |
| Mistral-7B-Instruct-v0.3 | Mistral AI | 32 | 4096 | 32,768 |

All experiments use a single NVIDIA RTX 3090 Ti (24 GB), FP16 inference, and a fixed random seed (seed = 42). Three main analysis protocols are used; see the master experiment table below and the Protocol note for details. Our primary evaluation conditions are:

- **Qwen Hard**: competition_math Level 4–5 (Hendrycks et al., 2021b), 499 valid samples after NaN filtering (271 correct, accuracy 54.3%). An earlier run on the same items produced 55.4% accuracy (277 correct); the difference arises from FP16 GPU non-determinism across independent runs despite identical seed and deterministic settings.
- **Qwen MMLU**: MMLU test (Hendrycks et al., 2021a), 1000 samples (accuracy 74.9%)
- **Llama MMLU**: MMLU test, 1000 samples (accuracy 63.8%)
- **Mistral MMLU**: MMLU test, 1000 samples (accuracy 62.7%)

Additional conditions (Qwen Easy, Qwen ARC (Clark et al., 2018), Llama Hard) appear in Appendix A.

**Master experiment table.**

| Section | Protocol | Dataset | n | Token position | Split | Layer selection | Metric |
|:--------|:---------|:--------|:-:|:---------------|:------|:----------------|:-------|
| 4.2 | Scale intervention | Qwen Hard | 500 | Step 0 (prompt-last) | None | All layers | Mean H_pre/H_post |
| 4.3 | Scale intervention | MMLU (3 models) | 300 | Step 0 | None | All layers | Mean H_pre/H_post |
| 5.1–5.2 | Baseline evaluation | Qwen Hard / MMLU (3 models) | 499/1000 | Generation average | 70/30 cal/test | On calibration set | Held-out AUROC |
| 6.1 | Token-position ablation | Qwen Hard / MMLU | 500/1000 | Step 0, Step 1, Full avg | 70/30 cal/test | On calibration set | Held-out AUROC |
| 7.1 | Lens faithfulness | Qwen MMLU | 1000 | Step 0 | 70/30 cal/test | On calibration set | AUROC, KL, top-1 agreement |
| 7.2 | Classifier comparison | All 4 conditions | 499/1000 | Generation average | 70/30 cal/test | On calibration set | Held-out AUROC |

**Protocol note.** Three distinct analysis protocols are used in this paper, as summarized in the master experiment table above. The *baseline evaluation protocol* (Section 5, Tables 1–3) uses generation-average entropy with 499/1000 samples after filtering, and selects best layer/sign on a 70/30 calibration set. The *token-position ablation protocol* (Section 6.1, Appendix B) independently resampled 1000 MMLU items and reran the same 500 Hard items. For MMLU, the accuracy difference (70.8% vs. 74.9%) is due to different sampled items. For Hard, the same items were used but FP16 GPU non-determinism (despite fixed seed = 42 and cudnn.deterministic = True) caused 54/499 items to receive different responses, yielding 53.2% vs. 54.3% accuracy. The *scale intervention protocol* (Section 4) uses Step 0 (prompt-last) position on the original samples. All protocols use seed = 42 and stratified splits.

### 3.2 H_pre and H_post

Given the hidden state h_l at layer l, the language model head W, and the final RMSNorm N_eps:

$$H_{\text{pre},l} = H(\text{softmax}(W h_l))$$
$$H_{\text{post},l} = H(\text{softmax}(W \cdot N_\epsilon(h_l)))$$

where H denotes Shannon entropy normalized by log(|V|).

In all experiments, h_l denotes the residual stream output after transformer block l and before the model's final normalization module. H_post applies the model's learned final RMSNorm (including gain parameter gamma and epsilon) before the output head, whereas H_pre bypasses that module. Both are diagnostic projections: applying the final-layer decoder to intermediate representations that were not designed to be decoded at that point. All three models use epsilon = 1e-6 in their RMSNorm implementation, and gain parameters (gamma) are included. We examine lens faithfulness in Section 7.

### 3.3 Scale Decomposition

Writing h_l = r_l u_l where r_l = ||h_l|| and ||u_l|| = 1:

- **Pre-norm logits**: W h_l = r_l W u_l. The scalar r_l acts as an inverse temperature: larger r_l concentrates the softmax distribution and decreases H_pre.
- **Post-norm logits**: Because RMSNorm_eps(alpha * h) = gamma * h / sqrt(mean(h^2) + eps/alpha^2), scale invariance is approximate. When mean(h^2) >> eps (middle and late layers), alpha cancels and H_post is effectively scale-invariant. In low-norm early layers, the eps term introduces residual alpha-dependence.

### 3.4 Token Positions

We extract hidden states at three positions:

- **Step 0 (prompt-last)**: Last token of the input prompt before generation begins.
- **Step 1 (first-gen)**: First generated token.
- **Full average**: Mean entropy across all generated tokens.

Unless otherwise noted, main results use the generation-average position for correctness discrimination and Step 0 (prompt-last) for scale interventions.

### 3.5 Evaluation Protocol

- **Split**: 70/30 stratified calibration/test split (StratifiedShuffleSplit, seed = 42).
- **Layer and sign selection**: Best layer and sign direction are selected on the calibration set only. The test set is never used for selection.
- **Metric**: Held-out test AUROC.
- **Baselines**: Scalar baselines grouped into output-level (entropy, max-prob, margin), internal scalars (H_pre, H_post, logit_std, h_norm, and others), and length-only. Table 1 reports the eight most informative baselines for the four primary conditions. Appendix A reports supplementary evaluation conditions (Qwen Easy, Qwen ARC, Llama Hard) that were not included in the main held-out baseline protocol.
- **Significance**: Paired bootstrap over the held-out test set (1,000 resamples). We report delta-AUROC together with 95% percentile bootstrap confidence intervals. An interval including zero is treated as non-significant.

---

## 4. What the Two Entropies Measure: Scale, Epsilon, and Saturation

This section provides the paper's central measurement evidence: direct geometric interventions showing that H_pre depends on hidden-state magnitude while H_post does not. The predictive implications are evaluated in Section 5.

### 4.1 Mathematical Intuition

Recall from Section 3.3 that pre-norm logits are W h_l = r_l W u_l, where r_l acts as an inverse temperature. This can be made precise:

**Proposition.** For any fixed logit vector z in R^V, let H_raw(alpha) = -sum_i p_i log p_i where p_i = softmax(alpha z)_i. Then dH_raw/d(alpha) = -alpha Var_{p_alpha}(z) <= 0, where Var denotes the variance under p_alpha. That is, the unnormalized Shannon entropy is monotonically non-increasing in the scale factor alpha. Since our normalized entropy H = H_raw / log|V| differs only by a positive constant, the monotonicity carries over: dH/d(alpha) = -(alpha / log|V|) Var_{p_alpha}(z) <= 0.

This follows from the identity d(-sum_i p_i log p_i)/d(alpha) = -alpha sum_i p_i(z_i - z_bar)^2. Increasing r_l concentrates the softmax distribution and decreases H_pre. Post-norm logits pass through RMSNorm, which under the standard implementation satisfies:

$$N_\epsilon(\alpha h) = \gamma \odot \frac{h}{\sqrt{\text{mean}(h^2) + \epsilon / \alpha^2}}$$

When mean(h^2) >> epsilon (middle and late layers with large hidden-state norms), alpha cancels approximately, making H_post effectively scale-invariant. In low-norm early layers, the epsilon term becomes non-negligible, breaking exact invariance.

We test these predictions with two interventions: unit-norm removal and alpha-sweep.

### 4.2 Qwen Case Study: Scale Intervention

We apply interventions to Qwen2.5-7B-Instruct on competition_math Hard (500 samples, Step 0 position).

**Unit-norm removal** (h -> h/||h||): Removing radial magnitude collapses H_pre toward maximal entropy across all layers, while H_post changes by less than 0.003:

| Layer | H_pre orig | H_pre unit | Change | H_post orig | H_post unit | Change |
|:-----:|:----------:|:----------:|:------:|:-----------:|:-----------:|:------:|
| 0 | 0.9963 | 1.0000 | +0.004 | 0.4788 | 0.4812 | +0.002 |
| 4 | 0.9705 | 1.0000 | +0.030 | 0.0780 | 0.0786 | +0.001 |
| 12 | 0.8577 | 1.0000 | +0.142 | 0.0484 | 0.0491 | +0.001 |
| 16 | 0.6637 | 0.9999 | +0.336 | 0.1822 | 0.1843 | +0.002 |
| 24 | 0.0444 | 0.9999 | +0.956 | 0.0861 | 0.0864 | +0.000 |
| 27 | 0.0166 | 1.0000 | +0.983 | 0.0307 | 0.0308 | +0.000 |

**Alpha-sweep** (h -> alpha * h) at representative layers confirms that H_pre responds strongly to scale manipulation while H_post remains invariant to four decimal places:

| Alpha | L4 H_pre | L4 H_post | L16 H_pre | L16 H_post |
|:-----:|:--------:|:---------:|:---------:|:----------:|
| 0.25 | 0.9985 | 0.0780 | 0.9817 | 0.1822 |
| 1.00 | 0.9705 | 0.0780 | 0.6637 | 0.1822 |
| 4.00 | 0.5169 | 0.0780 | 0.0990 | 0.1822 |

On Qwen, H_pre spans the range 0.10–0.98 under alpha-sweep at L16, while H_post shows zero variation to four decimal places. This establishes H_pre as structurally scale-sensitive and H_post as effectively scale-invariant on this model.

### 4.3 Cross-Family Verification

To verify that these findings are not Qwen-specific, we repeat the intervention on Llama-3-8B and Mistral-7B using 300 MMLU samples under identical conditions (Step 0, seed = 42). For comparability, we also rerun Qwen on the same MMLU-300 setup (note: Section 4.2 used competition_math Hard 500).

**Table 4. Cross-model scale intervention summary (MMLU 300, Step 0, seed = 42)**

| Model | Layers | H_pre unit-norm mean (over layers) | Max Delta-H_post (alpha-sweep) | Worst layer | h_norm at worst | % layers original H_pre > 0.99 |
|:------|:------:|:--------------------:|:-----------------:|:-----------:|:---------------:|:---------------------------:|
| **Qwen** | 28 | **0.999947** | **0.000454** (L0) | L0 | 10.21 | 0% |
| **Llama** | 32 | **0.999989** | **0.407632** (L1) | L1 | 0.89 | 66% |
| **Mistral** | 32 | **0.999999** | **0.105224** (L0) | L0 | 0.25 | 91% |

Note: The last column reports the percentage of layers where the *original* (unmodified) H_pre already exceeds 0.99, indicating a ceiling effect. This is distinct from the unit-norm result (column 3), where all models converge to ~1.0 by construction.

Across all three model families, unit-normalization collapses H_pre toward maximal entropy (mean > 0.9999), confirming that H_pre is structurally scale-sensitive regardless of model architecture.

H_post is largely scale-invariant in middle and late layers: at L16, alpha-sweep variation is 0.0023 (Llama) and 0.0028 (Mistral). At L24, Llama achieves exact invariance (H_post = 0.0689 across all alpha values).

### 4.4 Low-Norm Early-Layer Exception

H_post alpha-invariance breaks down in early layers: Llama L1 shows variation of 0.41, Mistral L0 shows 0.11, while Qwen L0 shows only 0.0005. These deviations correlate with hidden-state norm: Mistral L0 has mean h_norm = 0.25, Llama L1 has 0.89, while Qwen L0 has 10.21.

This is consistent with the RMSNorm mathematics: because practical implementations use nonzero epsilon, scale invariance is approximate. When the layer RMS is very small, the epsilon term becomes non-negligible under alpha-scaling, so early low-norm layers can deviate from exact invariance.

We therefore treat H_post as largely scale-invariant in the operating regime of middle and late layers, with explicit low-norm early-layer deviations rather than exact global invariance.

### 4.5 Saturation-Limited Observability

While unit-norm removal establishes that scale sensitivity *exists* structurally, the alpha-sweep reveals that its *observable magnitude* varies dramatically across models:

| Model | Mid-layer H_pre range (alpha 0.25–4.0) | Dynamic range |
|:------|:---------------------------------------:|:-------------:|
| Qwen L14 | 0.058 – 0.982 | **0.924** (wide) |
| Llama L16 | 0.992 – 1.000 | **0.008** (ceiling) |
| Mistral L16 | 0.999 – 1.000 | **0.001** (extreme ceiling) |

In Llama and Mistral, H_pre is already near maximal entropy (~1.0) in early-to-middle layers because hidden-state norms are small (Mistral L0: 0.25 vs. Qwen L0: 10.21). This ceiling effect compresses the observable alpha-sweep response, making scale sensitivity practically invisible in intermediate layers despite being structurally present.

This is consistent with norm-binned control results (Appendix C): when H_pre is controlled for logit_std, Mistral retains 100.1% of its AUROC—not because H_pre is scale-invariant, but because H_pre is already at ceiling and carries little additional information beyond scale. In contrast, Mistral's H_pre retention after controlling for h_norm is 85.2% (Appendix C, Table A3).

H_pre is structurally scale-sensitive, but the observable magnitude of this sensitivity is saturation-limited and therefore model- and layer-dependent.

---

## 5. Predictive Utility of Internal Signals

Having established that H_pre is structurally scale-sensitive (Section 4), we now evaluate the predictive landscape—which internal signals best discriminate correct from incorrect responses—and test whether H_pre carries information beyond what simple scale proxies already capture.

### 5.1 Internal Layers Outperform Final-Output Entropy

Table 1 presents held-out test AUROC for single-pass scalar baselines across four primary evaluation conditions. The main empirical finding is that H_pre often does not outperform direct scale proxies.

**Table 1. Held-out test AUROC — single-pass scalar baselines (70/30 split, best layer selected on calibration set)**

| Method | Passes | Qwen Hard | Qwen MMLU | Llama MMLU | Mistral MMLU |
|:-------|:------:|:---------:|:---------:|:----------:|:------------:|
| Output entropy | 1 | 0.6316 | 0.5775 | 0.5354 | 0.5070 |
| Output max-prob | 1 | 0.8041 | 0.6177 | 0.5211 | 0.4968 |
| Output margin | 1 | 0.6762 | 0.5216 | 0.5141 | 0.4685 |
| Length-only | 1 | 0.7964 | 0.6428 | 0.5325 | 0.6733 |
| h_norm | 1 | 0.8256 | 0.5740 | 0.5763 | 0.6983 |
| logit_std | 1 | **0.8086** | **0.6674** | **0.6522** | **0.7315** |
| H_post | 1 | 0.6613 | 0.6376 | 0.5796 | 0.5850 |
| **H_pre** | **1** | **0.7672** | **0.6367** | **0.6021** | **0.6737** |

Three patterns emerge, the second of which is central to the measurement confound this paper identifies:

1. **Internal > final-output entropy**: In every condition, the best internal-layer signal exceeds output entropy by +0.09 to +0.22 AUROC. We note that on Qwen Hard, output max-prob (0.8041) is already a strong output-level baseline; the internal advantage is relative to output entropy specifically, not to the strongest output-level confidence signal in all conditions.
2. **Among single-layer scalars, logit_std consistently outperforms H_pre**: The strongest scale proxy, logit_std, outperforms single-layer H_pre in all four conditions (h_norm does not uniformly do so). This is the expected outcome if H_pre is primarily capturing hidden-state scale (Section 4): a direct scale statistic should be at least as informative as an entropy measure that indirectly reflects scale through the softmax temperature effect. Note that multi-layer H_pre profiles (Table 2) can outperform single-layer scale proxies, suggesting that the full entropy trajectory carries information beyond a single scalar.
3. **Length-only is surprisingly strong**: Length-only achieves 0.7964 on Qwen Hard and 0.6733 on Mistral MMLU, approaching or exceeding H_pre. Because our main results use generation-average token positions, length confounding is a concern. However, on Qwen, Step 0 (prompt-last, pre-generation) results in Section 6.1 show that internal signals already carry substantial discriminative information (AUROC 0.59–0.71) before any tokens are generated, where length is undefined.

**Table 2. Multi-layer profiles and self-consistency (held-out test AUROC)**

| Method | Passes | Qwen Hard | Qwen MMLU | Llama MMLU | Mistral MMLU |
|:-------|:------:|:---------:|:---------:|:----------:|:------------:|
| H_post profile (LR) | 1 | 0.8060 | 0.6412 | 0.6573 | 0.6388 |
| H_pre profile (LR) | 1 | 0.8467 | 0.7042 | 0.7006 | 0.6955 |
| h_norm profile (LR) | 1 | **0.8660** | 0.6757 | **0.7165** | **0.7594** |
| Matched self-consistency (K=5, temp=0.3) | 5 | 0.7905* | — | — | — |

*Self-consistency AUROC is agreement-to-correctness; available for Qwen Hard only under matched temperature.

The best multi-layer profile consistently outperforms the best single-layer scalar (+0.02 to +0.10). Among profiles, h_norm profile is strongest in 3/4 conditions, consistent with scale being an important driver of discrimination. Under matched temperature (0.3), self-consistency (K=5) achieves agreement AUROC of 0.7905 with 5x compute, while logit_std achieves 0.8086 with 1x compute.

### 5.2 H_pre Is Largely Redundant with Scale Proxies: Evidence for Scale-Sensitivity

The results in Table 1 show that logit_std outperforms H_pre in all four conditions. If our measurement claim is correct—that H_pre primarily captures hidden-state scale—then adding H_pre to logit_std should yield little incremental information, because both are measuring the same underlying quantity. We test this prediction directly. We fit logistic regression models with and without H_pre on the calibration set and evaluate on the held-out test set (Table 3). This analysis is currently limited to two Qwen conditions; whether similar results hold for Llama and Mistral remains untested.

**Table 3. Incremental utility of H_pre over scale proxies (held-out test AUROC)**

| Features | Qwen Hard | Qwen MMLU |
|:---------|:---------:|:---------:|
| logit_std only | 0.8086 | 0.6674 |
| H_pre only | 0.7672 | 0.6367 |
| logit_std + H_pre | 0.8284 | 0.6705 |
| logit_std + h_norm + length | 0.8120 | 0.6466 |
| logit_std + h_norm + length + H_pre | 0.8143 | 0.6427 |
| **Delta (logit_std -> +H_pre)** | **+0.020 [−0.014, +0.057]** | **+0.003 [−0.003, +0.009]** |
| **Delta (full -> +H_pre)** | **+0.002 [−0.022, +0.027]** | **-0.004 [−0.012, +0.003]** |

In neither condition does adding H_pre to logit_std yield a 95% bootstrap CI excluding zero (paired bootstrap, 1000 resamples). This result is consistent with the scale-sensitivity hypothesis: if H_pre is primarily a function of hidden-state magnitude, then logit_std—a more direct measure of that same scale—should already capture the information that H_pre provides, leaving no residual. Whether this extends to Llama and Mistral—where H_pre dynamic range is more limited (Section 4.5)—remains untested.

---

## 6. Where and When the Signal Appears

### 6.1 Token Position

We compare three extraction positions on Qwen Hard (500 samples) and Qwen MMLU (1000 samples):

| Metric | Step 0 | Step 1 | Full Avg | Best |
|:-------|:------:|:------:|:--------:|:-----|
| **H_pre (Hard)** | 0.7087 | **0.7486** | 0.7479 | Step 1 |
| **H_pre (MMLU)** | 0.5911 | **0.6192** | 0.5968 | Step 1 |

H_pre signal is strongest at the first generated token (Step 1) and saturates—the difference between Step 1 and full-generation average is 0.0007 on Hard and 0.022 on MMLU. Step 0 (prompt-last, pre-generation) also carries substantial signal (AUROC 0.59–0.71), enabling uncertainty estimation before generation begins.

Different metrics show different optimal positions: h_norm favors Step 0, logit_margin favors full average. Token position should always be reported explicitly.

### 6.2 No Universal Optimal Layer Under Our Protocol

Under our held-out protocol (single 70/30 split, seed = 42), the best-performing layer varies substantially across conditions and model families:

**Table 5. Best layer and sign selected on calibration set (held-out protocol)**

| Condition | H_pre best layer | H_pre sign | logit_std best layer | logit_std sign | H_post best layer | H_post sign |
|:----------|:----------------:|:----------:|:--------------------:|:--------------:|:-----------------:|:-----------:|
| Qwen Hard | L17 | - | L4 | + | L27 | - |
| Qwen MMLU | L18 | - | L18 | + | L27 | - |
| Llama MMLU | L0 | + | L13 | - | L10 | - |
| Mistral MMLU | L30 | - | L13 | + | L24 | - |

Under this protocol, no fixed optimal layer emerges for correctness discrimination. The best layer depends on the metric, model family, and task. We note that these selections are based on a single split; their stability across multiple independent splits has not been verified (see Section 8.4).

A supplementary full-sample per-layer analysis using Cohen's d (Appendix F, Figure 7) shows broadly consistent patterns but yields different optimal layers in some conditions (e.g., Mistral H_pre best = L0 under full-sample Cohen's d vs. L30 under held-out AUROC), illustrating that optimal-layer conclusions are sensitive to both metric choice and evaluation protocol.

### 6.3 Cross-Family Sign Reversal

Repeated-split analysis (20 independent 70/30 splits, seeds 0–19) reveals that H_pre sign is fully stable: lower H_pre is associated with correct responses (sign = −) across all four conditions and all 20 splits (20/20). However, H_post exhibits cross-family sign instability: on Llama MMLU, H_post sign is + in 15/20 splits (higher H_post = correct), reversing the − sign observed in all Qwen and Mistral conditions. This precludes a fixed-sign heuristic for H_post and necessitates sign calibration on a held-out set. The sign reversal is specific to H_post, not H_pre.

### 6.4 Multi-Layer Profiles

Using logistic regression over all-layer features, profiles consistently outperform single best-layer scalars:

| Condition | H_pre single | H_pre profile (LR) | Gain |
|:----------|:------------:|:-------------------:|:----:|
| Qwen Hard | 0.7672 | 0.8467 | +0.080 |
| Qwen MMLU | 0.6367 | 0.7042 | +0.068 |
| Llama MMLU | 0.6021 | 0.7006 | +0.099 |
| Mistral MMLU | 0.6737 | 0.6955 | +0.022 |

However, h_norm profile (LR) is strongest in 3/4 conditions (Table 2), consistent with scale being an important driver of discrimination.

---

## 7. Lens Dependence

### 7.1 Raw Logit Lens Poorly Matches the Final-Layer Distribution on Qwen

Both H_pre and H_post as defined in this paper are diagnostic projections: they apply the final-layer decoder to intermediate representations. To assess how faithful this projection is on Qwen, we train a Tuned Lens (affine translator, 3 epochs on wikitext-2) and compare:

**Table 6. Logit Lens vs. Tuned Lens faithfulness (Qwen2.5-7B). KL denotes KL divergence from the final-layer output distribution (lower = more faithful).**

| Layer | KL(TL, final) | KL(LL, final) | TL more faithful? | Top-1 agr. (TL) | Top-1 agr. (LL) |
|:-----:|:------:|:------:|:--------:|:----------:|:----------:|
| 4 | 6.568 | 11.318 | Yes | 19.5% | 0.0% |
| 12 | 6.299 | 13.506 | Yes | 23.7% | 0.0% |
| 20 | 5.578 | 18.903 | Yes | 30.5% | 0.0% |
| 24 | 3.765 | 21.001 | Yes | 46.7% | 0.0% |

On Qwen, Logit Lens achieves 0% top-1 agreement with the final-layer distribution in middle layers, meaning that H_pre computed via raw lm_head projection does not reflect what the model would "predict" at that layer. Tuned Lens substantially improves faithfulness (KL(TL, final) < KL(LL, final) in 27/28 layers). We additionally evaluate discrimination on Qwen MMLU using prompt-last (Step 0) hidden states:

**Table 7. Prompt-last discrimination under different decoders (Qwen MMLU, held-out test AUROC)**

| Decoder | AUROC |
|:--------|:-----:|
| Logit Lens (H_pre) | 0.5325 |
| Tuned Lens | 0.5629 |
| H_post | 0.5574 |

Note: Table 7 uses the same 1000 MMLU samples as the main baseline evaluation (accuracy 74.9%, Section 3.1), not the separate token-position sample (accuracy 70.8%, Appendix B). The H_pre AUROC here (0.5325, layer 23) differs from Appendix B's Step 0 H_pre (0.5911, layer 16) because the two experiments use different sample sets and therefore select different best layers.

Tuned Lens achieves the highest prompt-last discrimination, but gains over Logit Lens are modest (+0.030). This does not invalidate H_pre as a useful correctness discriminator—predictive utility and faithfulness are independent properties. However, it means that H_pre should not be interpreted as revealing what the model "believes" at a given layer. Whether similar faithfulness patterns hold for Llama and Mistral remains untested (see Section 8.4).

### 7.2 Classifier Choice Changes Discrimination

Reproducing the Entropy-Lens setup (H_post profile + k-NN, k=3) and comparing with the same features under logistic regression under a matched-classifier comparison reveals a large classifier effect:

| Method | MMLU Qwen | MMLU Llama | MMLU Mistral | Qwen Hard |
|:-------|:---------:|:----------:|:------------:|:---------:|
| Entropy-Lens original (k-NN, k=3) | 0.5474 | 0.5521 | 0.5748 | 0.6384 |
| Entropy-Lens matched (H_post, LR) | 0.6412 | 0.6573 | 0.6388 | 0.8060 |
| H_pre profile (LR) | **0.7042** | **0.7006** | 0.6955 | **0.8467** |

Switching from k-NN to LR on the same H_post features increases AUROC by +0.06 to +0.17. Conclusions about which features are most discriminative are sensitive to classifier choice. Under a common classifier (LR), H_pre profile outperforms H_post profile in all four conditions (+0.04 to +0.06), consistent with the additional scale information in H_pre.

---

## 8. Discussion, Limitations, and Reporting Recommendations

### 8.1 Implications for Prior Work

Our findings have direct consequences for existing work that computes entropy from logit-lens projections. To ground this claim, Table 8 audits representative prior work on layerwise entropy and internal uncertainty signals, checking whether the normalization point is specified and whether scale baselines are reported.

**Table 8. Literature audit: normalization specification in prior work on layerwise entropy / internal uncertainty**

Panel A lists works that directly project intermediate hidden states to vocabulary space (logit-lens family); Panel B lists related work where the normalization point is less directly relevant.

*Panel A. Intermediate vocabulary projection (logit-lens family)*

| Reference | Lens / method | Normalization point specified? | Scale baseline reported? |
|:----------|:--------------|:------------------------------:|:------------------------:|
| nostalgebraist (2020) | Logit Lens | Unspecified in original post | No |
| Belrose et al. (2023) | Tuned Lens | Yes (explicit: LL(h) = LN[h] W_U) | No |
| Ali et al. (2025) | Entropy-Lens | Text unspecified; appendix code applies final LN | No |
| Geva et al. (2022) | FF vocab projection | Explicitly omitted; appendix checks LN effect | No |
| Chuang et al. (2024) | DoLa | Not explicitly specified | No |
| Ma et al. (2025) | SimLens | Not explicitly specified | No |

*Panel B. Probe-based, output-level, or other internal methods*

| Reference | Lens / method | Normalization point specified? | Scale baseline reported? |
|:----------|:--------------|:------------------------------:|:------------------------:|
| Kossen et al. (2024) | SEPs | N/A (probe-based) | No |
| Bhatnagar et al. (2026) | DRIFT | N/A (probe-based) | No |
| Gao et al. (2025) | FLUE | Output-level entropy | N/A |
| Chen et al. (2025) | Internal Confidence | Pre-generation hidden states | No |
| Stolfo et al. (2024) | Confidence Regulation Neurons | Norm-based mechanism | Partial (norm is the signal) |

Among the six works in Panel A that directly project intermediate hidden states to vocabulary space, normalization handling is inconsistent: Belrose et al. (2023) explicitly include LayerNorm in their logit-lens definition, Geva et al. (2022) explicitly omit it and check the effect in an appendix, Ali et al. (2025) leave the main text unspecified while their appendix code applies final LayerNorm, and the remaining three do not specify the normalization point at all. None of these works reports direct scale baselines (logit_std, h_norm) alongside entropy. The works in Panel B use probe-based or output-level methods where the normalization point is less directly relevant, so we do not claim that the confound applies to them in the same way. This cross-paper inconsistency does not imply that any individual finding is invalid—but it does mean that, for work based on decoded intermediate entropy, the normalization-dependent measurement choice identified in this paper has not been consistently specified or controlled for. We discuss three categories of affected research below.

**Logit-lens entropy without normalization specification.** The logit lens (nostalgebraist, 2020) projects intermediate hidden states to vocabulary space via the language model head. When researchers compute entropy from these projections without specifying whether the hidden state is first normalized, our results indicate that the resulting entropy is dominated by hidden-state scale (Section 4). Specifically, in RMSNorm-based models, pre-normalization logit-lens entropy acts as an implicit temperature-scaled measure: layers with large hidden-state norms produce concentrated distributions (low entropy), while layers with small norms produce near-uniform distributions (high entropy). Any "critical layer" identified by such entropy may reflect a norm magnitude transition rather than a distributional uncertainty shift. Prior work that reports layerwise entropy patterns from logit-lens projections should specify whether normalization was applied before the projection, and readers should consider the possibility that reported entropy gradients partially reflect scale gradients.

**Entropy-Lens and related profiling methods.** Entropy-Lens (Ali et al., 2025) constructs entropy profiles across layers and uses them for decision-strategy classification. Our results show that the discriminative content of such profiles depends on the classifier (Section 7.2): switching from k-NN to logistic regression on the same H_post features changes AUROC by up to +0.17. Furthermore, under a common classifier (LR), H_pre profiles outperform H_post profiles in all four conditions (+0.04 to +0.06), consistent with the additional scale information in H_pre. This means that prior comparisons of entropy profiles that do not control for classifier choice may conflate feature quality with classifier suitability—and profiles that appear effective may be leveraging scale information rather than directional uncertainty.

**Internal uncertainty estimation more broadly.** Methods that extract uncertainty signals from intermediate hidden states—including probing-based approaches (Kossen et al., 2024; Bhatnagar et al., 2026)—should consider whether their signals are partially confounded with hidden-state scale. Our norm-binned control analysis (Appendix C) provides a concrete diagnostic: if a signal's AUROC drops substantially after binning by h_norm or logit_std, the signal is scale-dependent. We recommend that future internal-uncertainty work report such controlled baselines alongside raw performance.

These implications are bounded by our experimental scope: three 7–8B RMSNorm-based models on mathematical reasoning and MMLU. Whether analogous confounds arise in LayerNorm-based architectures, larger models, or open-ended generation tasks remains an open empirical question. Concurrent work by Marín (2026) independently distinguishes normalized from raw logit-lens projections and reports that applying final normalization stabilizes early-layer decoding; our work complements this by systematically characterizing how this choice changes what decoded entropy measures across three model families. We frame these as diagnostic considerations rather than definitive invalidations of prior findings.

### 8.2 What This Paper Claims

1. Normalization qualitatively changes what layerwise decoded entropy measures.
2. H_pre is structurally scale-sensitive; H_post is largely scale-invariant outside low-norm early layers.
3. Among single-layer scalar baselines, simple scale proxies often match or outperform entropy for correctness discrimination in our evaluated settings.
4. Internal-layer signals consistently outperform final-output entropy.
5. Under our held-out protocol, conclusions depend strongly on layer, token position, model family, and decoding lens.
6. Prior work that computes layerwise entropy from logit-lens projections without specifying the normalization point may be conflating scale statistics with distributional uncertainty.

### 8.3 What This Paper Does Not Claim

- We do not claim to identify reasoning layers or trace mechanistic reasoning pathways.
- We do not claim causal mediation of scale on correctness.
- We do not claim H_pre is a universal uncertainty estimator.
- We do not claim superiority over self-consistency under all conditions.
- We do not claim that entropy is useless—it remains an interpretable diagnostic that reveals normalization effects.
- We do not empirically generalize beyond RMSNorm-based decoder LMs. Whether similar considerations apply to LayerNorm-based architectures remains an open empirical question.

### 8.4 Limitations

- **Model coverage**: Our deepest analysis (scale interventions, token positions, Tuned Lens) is on Qwen2.5-7B. Llama and Mistral serve as cross-family sanity checks on MMLU, not full replications.
- **Split and seed robustness**: Main results (Tables 1–3) use a single 70/30 split (seed = 42). We additionally report repeated-split analysis (20 splits, seeds 0–19; Appendix I) which confirms sign stability for H_pre (20/20 in all conditions) and layer selection consistency for some metrics (e.g., h_norm: L18 20/20 on Qwen Hard; H_post: L27 19–20/20 on Qwen). However, other metrics show substantial layer variability across splits (e.g., logit_std on Qwen Hard: L4 9/20, L16 6/20), and AUROC standard deviations range from 0.02 to 0.06. Single-split results should therefore be interpreted with caution.
- **Diagnostic projection caveat**: Both H_pre and H_post apply the final-layer decoder to intermediate representations. As shown in Section 7.1, this projection is unfaithful in middle layers. The entropy values should not be interpreted as reflecting the model's latent prediction at a given layer.
- **Lens-faithfulness scope**: Tuned Lens analysis is Qwen-only. Whether similar faithfulness patterns hold for Llama and Mistral remains untested.
- **H_post early-layer deviation**: We observe H_post alpha-variation in low-norm early layers. While consistent with epsilon-regularized RMSNorm, we have not fully disentangled finite-precision effects from epsilon contributions.
- **Task scope**: We evaluate on mathematical reasoning and MMLU. Whether findings extend to open-ended generation, long-form QA, or code generation remains open.
- **Ceiling effect**: On Llama and Mistral, H_pre is near-saturated in early-to-middle layers, limiting the observable dynamic range of scale sensitivity. This does not invalidate the structural finding but limits practical utility in those regimes.
- **Exploratory analyses excluded**: Some exploratory causal and probing analyses were conducted but excluded from the main text because they were not methodologically stable enough for reliable interpretation.

### 8.5 Reporting Recommendations

Based on our findings, we recommend that future work on layerwise entropy adopt the following practices. The first three address the measurement confound identified in this paper; the remaining three address methodological pitfalls observed during our evaluation.

1. **Always specify the normalization point.** Report whether entropy is computed before or after the final normalization module, and recognize that the two measure qualitatively different quantities. Without this specification, it is impossible to determine whether reported entropy patterns reflect distributional uncertainty or hidden-state scale.
2. **Report direct scale baselines.** Include at least logit_std and h_norm alongside entropy. If entropy does not outperform these trivial baselines, the signal may be scale-dominated. The norm-binned control diagnostic (Appendix C) provides a concrete test: compute entropy AUROC within bins of equal h_norm or logit_std and check whether discriminative power survives.
3. **Do not assume entropy is interpretable without calibration.** The sign of the entropy-correctness relationship can reverse across model families (Section 6.3), and the best layer is condition-dependent (Section 6.2). Always calibrate on a held-out set rather than assuming a fixed layer or sign direction.
4. **Specify token position.** Report whether hidden states are extracted at prompt-last, first-generated, or generation-average positions. Different metrics favor different positions (Section 6.1).
5. **Separate decoder from classifier effects.** When comparing entropy profiles, ensure the decoding lens and downstream classifier are controlled. Classifier choice alone can shift AUROC by up to +0.17 (Section 7.2).
6. **Report the normalization implementation details.** Specify the epsilon value and whether gain parameters are included, as these affect early-layer behavior (Section 4.4).

### 8.6 Future Work

- Extending the analysis to LayerNorm-based architectures to test whether the pre/post distinction produces analogous effects.
- Repeated-split and multi-seed robustness analysis for layer and sign selection stability.
- Evaluation on open-ended QA, code generation, and long-form generation tasks.
- Investigation of more faithful intermediate decoders (e.g., Tuned Lens, SimLens) under matched classifiers.

---

## References

Ali, R., Caso, F., Irwin, C., & Liò, P. (2025). Entropy-Lens: Uncovering decision strategies in LLMs. *arXiv:2502.16570*.

Belrose, N., Ostrovsky, I., McKinney, L., Furman, Z., Smith, L., Halawi, D., Biderman, S., & Steinhardt, J. (2023). Eliciting latent predictions from transformers with the tuned lens. *arXiv:2303.08112*.

Bhatnagar, R., Sun, Y., Zhang, C. A., Wen, Y., & Yang, H. (2026). DRIFT: Detecting representational inconsistencies for factual truthfulness. *arXiv:2601.14210*.

Brody, S., Alon, U., & Yahav, E. (2023). On the expressivity role of LayerNorm in transformers' attention. *Findings of ACL 2023*.

Chen, L., de Melo, G., Suchanek, F. M., & Varoquaux, G. (2025). Query-level uncertainty in large language models. *ICLR 2026*. arXiv:2506.09669.

Chuang, Y.-S., Xie, Y., Luo, H., Kim, Y., Glass, J., & He, P. (2024). DoLa: Decoding by contrasting layers improves factuality in large language models. *ICLR 2024*. arXiv:2309.03883.

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., & Tafjord, O. (2018). Think you have solved question answering? Try ARC, the AI2 Reasoning Challenge. *arXiv:1803.05457*.

Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). Detecting hallucinations in large language models using semantic entropy. *Nature, 630*(8017), 625–630.

Gao, S., Gong, T., Lin, Z., Xu, R., Zhou, H., & Li, J. (2025). FLUE: Streamlined uncertainty estimation for large language models. *Proceedings of the AAAI Conference on Artificial Intelligence, 39*(16), 16745–16753.

Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. *NeurIPS 2017*.

Geva, M., Caciularu, A., Wang, K. R., & Goldberg, Y. (2022). Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space. *EMNLP 2022*. arXiv:2203.14680.

Ghandeharioun, A., Caciularu, A., Pearce, A., Dixon, L., & Geva, M. (2024). Patchscopes: A unifying framework for inspecting hidden representations of language models. *ICML 2024*.

Grattafiori, A., Dubey, A., Jauhri, A., et al. (2024). The Llama 3 herd of models. *arXiv:2407.21783*.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., & Steinhardt, J. (2021a). Measuring massive multitask language understanding. *ICLR 2021*.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., & Steinhardt, J. (2021b). Measuring mathematical problem solving with the MATH dataset. *NeurIPS 2021*.

Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., de las Casas, D., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., Lavaud, L. R., Lachaux, M.-A., Stock, P., Le Scao, T., Lavril, T., Wang, T., Lacroix, T., & El Sayed, W. (2023). Mistral 7B. *arXiv:2310.06825*.

Katz, S., & Belinkov, Y. (2023). VISIT: Visualizing and interpreting the semantic information flow of transformers. *Findings of EMNLP 2023*. arXiv:2305.13417.

Kossen, J., Han, J., Razzak, M., Schut, L., Malik, S., & Gal, Y. (2024). Semantic entropy probes: Robust and cheap hallucination detection in LLMs. *arXiv:2406.15927*.

Kuhn, L., Gal, Y., & Farquhar, S. (2023). Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. *ICLR 2023*.

Ma, M., Zheng, B., Lin, Z., & Yang, T. (2025). SimLens for early exit in large language models: Eliciting accurate latent predictions with one more token. *arXiv:2507.17618*.

Marín, J. (2026). How transformers reject wrong answers: Rotational dynamics of factual constraint processing. *arXiv:2603.13259*.

nostalgebraist. (2020). interpreting GPT: the logit lens. *LessWrong blog post*.

Qwen Team. (2024). Qwen2.5 technical report. *arXiv:2412.15115*.

Stolfo, A., Wu, B., Gurnee, W., Belinkov, Y., Song, X., Sachan, M., & Nanda, N. (2024). Confidence regulation neurons in language models. *NeurIPS 2024*.

Sun, M., Chen, X., Kolter, J. Z., & Liu, Z. (2024). Massive activations in large language models. *COLM 2024*.

Sun, R., Cheng, W., Li, D., Chen, H., & Wang, W. (2025). Stop when enough: Adaptive early-stopping for chain-of-thought reasoning. *arXiv:2510.10103*.

Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., & Zhou, D. (2023). Self-consistency improves chain of thought reasoning in language models. *ICLR 2023*.

Zhang, B., & Sennrich, R. (2019). Root mean square layer normalization. *NeurIPS 2019*.

---

## Acknowledgment

This work was supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) Innovative Human Resource Development for Local Intellectualization program grant funded by the Korea government (MSIT)(IITP-2025-RS-2024-00436773). This work was supported by the Korea Institute for Advancement of Technology (KIAT) grant funded by the Korea government (Ministry of Trade, Industry and Energy) through the International Cooperation in Industrial Technology program (Project Number: P0026190).

---

## Appendix A. Additional Evaluation Conditions

Three supplementary conditions were evaluated under separate experimental pipelines and are **not directly comparable** to the main held-out results in Tables 1–3. We report available metrics below for reference.

**Qwen Easy (competition_math Level 1–2, 500 samples, accuracy 85.25%).** Evaluated under an earlier pipeline (generation-average, full-sample Cohen's d). Available held-out AUROC values from that pipeline:

| Method | AUROC |
|:-------|:-----:|
| Output entropy | 0.4818 |
| H_pre | 0.7295 |
| H_post | 0.5757 |
| h_norm | 0.6435 |
| Length-only | 0.6395 |

logit_std was not computed under this pipeline.

**Qwen ARC (ARC-Challenge, 999 samples, accuracy 89.3%).** Evaluated in separate entropy experiments using full-sample Cohen's d (unnormalized optimal layer L27, d = 0.36) but was not included in the held-out baseline protocol. The high accuracy (89.3%) limits discriminative dynamic range.

**Llama Hard (competition_math Level 4–5, 500 samples, accuracy 9.8%).** Evaluated with full-sample Cohen's d (normed optimal L9, d = −0.42; unnormed optimal L1, d = +0.86). The extreme class imbalance (9.8% correct) makes held-out AUROC unreliable and was therefore not computed under the main protocol.

## Appendix B. Token Position Ablations

**Important**: The token-position ablation was run as a separate experiment from the main baseline evaluation (Section 5). For MMLU, a different 1000-item sample was drawn, yielding 70.8% accuracy vs. 74.9% in the main evaluation. For Hard, the same 500 items were used but FP16 GPU non-determinism produced different responses on 54/499 items, yielding 53.2% vs. 54.3% (the main baseline). The 70/30 split was also applied independently. AUROC values in this appendix are therefore not directly comparable to Tables 1–3 in the main text; they are reported to show *relative* patterns across token positions.

**Table A2. Held-out test AUROC by token position (Qwen, 70/30 split, separate sample)**

**Qwen Hard (n=500, accuracy=53.2%)**

| Metric | Step 0 (prompt-last) | Step 1 (first-gen) | Full Avg | Best |
|:-------|:--------------------:|:------------------:|:--------:|:-----|
| H_pre (unnormed) | 0.7087 | **0.7486** | 0.7479 | Step 1 |
| H_post (normed) | **0.6893** | 0.6859 | 0.6032 | Step 0 |
| logit_std | 0.6963 | 0.6789 | **0.7514** | Full Avg |
| h_norm | **0.7638** | 0.6567 | 0.7336 | Step 0 |
| logit_max | 0.7141 | **0.7474** | 0.7468 | Step 1 |
| logit_margin | 0.6217 | 0.7326 | **0.7779** | Full Avg |
| Wh_norm | 0.7082 | 0.7179 | **0.7529** | Full Avg |

**Qwen MMLU (n=1000, accuracy=70.8%)**

| Metric | Step 0 (prompt-last) | Step 1 (first-gen) | Full Avg | Best |
|:-------|:--------------------:|:------------------:|:--------:|:-----|
| H_pre (unnormed) | 0.5911 | **0.6192** | 0.5968 | Step 1 |
| H_post (normed) | 0.5424 | 0.5591 | **0.6235** | Full Avg |
| logit_std | 0.6052 | **0.6221** | 0.5958 | Step 1 |
| h_norm | 0.5581 | 0.5724 | **0.6127** | Full Avg |
| logit_max | **0.5996** | 0.5962 | 0.5812 | Step 0 |
| logit_margin | 0.5467 | 0.4794 | **0.5671** | Full Avg |
| Wh_norm | 0.5504 | **0.6128** | 0.5296 | Step 1 |

## Appendix C. Norm-Binned Controls

To test whether H_pre's discrimination is merely a proxy for hidden-state norm or logit variability, we bin samples into 5 equal-frequency groups by the control variable, compute H_pre AUROC within each bin, and measure weighted retention relative to the original (unbinned) H_pre AUROC. Retention is defined as (AUROC_binned - 0.5) / (AUROC_original - 0.5) x 100%, where 0.5 represents chance-level AUROC. A retention of 100% means the signal survives fully after controlling for the variable; values near 0% or negative indicate that the signal is largely or entirely explained by the control variable. In both tables below, Original AUROC is the H_pre AUROC at its best layer before binning.

**Table A3. H_pre AUROC retention after controlling for h_norm**

| Condition | H_pre Original AUROC | Binned H_pre AUROC (weighted) | Retention |
|:----------|:--------------------:|:-----------------------------:|:---------:|
| Qwen Hard | 0.7672 | 0.6121 | 42.0% |
| Qwen MMLU | 0.6367 | 0.5888 | 65.0% |
| Llama MMLU | 0.6021 | 0.4564 | -42.7% |
| Mistral MMLU | 0.6737 | 0.6480 | 85.2% |

**Table A4. H_pre AUROC retention after controlling for logit_std**

| Condition | H_pre Original AUROC | Binned H_pre AUROC (weighted) | Retention |
|:----------|:--------------------:|:-----------------------------:|:---------:|
| Qwen Hard | 0.7672 | 0.4617 | -14.3% |
| Qwen MMLU | 0.6367 | 0.5576 | 42.1% |
| Llama MMLU | 0.6021 | 0.5347 | 34.0% |
| Mistral MMLU | 0.6737 | 0.6739 | 100.1% |

Mistral H_pre retains 100.1% of its AUROC after controlling for logit_std, consistent with H_pre being at ceiling (Section 4.5): when H_pre is already saturated, controlling for logit_std does not reduce H_pre's signal because both carry the same (minimal) information. In contrast, Mistral's H_pre retention after controlling for h_norm is 85.2% (Table A3), indicating that h_norm captures a different portion of H_pre's variance than logit_std. Negative retention (Llama H_pre controlled by h_norm, Qwen Hard H_pre controlled by logit_std) indicates that within-bin H_pre performance falls below chance, suggesting strong collinearity between the control variable and H_pre in those conditions.

## Appendix D. Tuned Lens Training Details

- **Architecture**: Affine translator per layer (W_l, b_l)
- **Training data**: wikitext-2 validation set
- **Epochs**: 3
- **Loss**: KL divergence from final-layer distribution
- **Result**: Loss improved from 7008 to 2459. KL(TL, final) < KL(LL, final) in 27/28 layers (lower KL = more faithful).

## Appendix E. Cross-Model Alpha-Sweep Tables

**Table A5. Llama-3-8B alpha-sweep (MMLU 300, Step 0, seed=42)**

| Alpha | L0 H_pre | L0 H_post | L8 H_pre | L8 H_post | L16 H_pre | L16 H_post | L24 H_pre | L24 H_post |
|:-----:|:--------:|:---------:|:--------:|:---------:|:---------:|:----------:|:---------:|:----------:|
| 0.25 | 1.0000 | 0.9433 | 1.0000 | 0.8125 | 1.0000 | 0.7995 | 0.9997 | 0.0689 |
| 0.50 | 1.0000 | 0.8804 | 1.0000 | 0.8069 | 0.9999 | 0.7978 | 0.9987 | 0.0689 |
| 1.00 | 1.0000 | 0.8344 | 0.9999 | 0.8055 | 0.9995 | 0.7973 | 0.9947 | 0.0689 |
| 2.00 | 1.0000 | 0.8169 | 0.9995 | 0.8051 | 0.9980 | 0.7972 | 0.9770 | 0.0689 |
| 4.00 | 1.0000 | 0.8120 | 0.9978 | 0.8050 | 0.9920 | 0.7972 | 0.6077 | 0.0689 |

H_post at L24 is exactly invariant (0.0689) across all alpha values. L0 H_post shows variation of 0.13 (low-norm regime, h_norm=0.89).

**Table A6. Mistral-7B alpha-sweep (MMLU 300, Step 0, seed=42)**

| Alpha | L0 H_pre | L0 H_post | L8 H_pre | L8 H_post | L16 H_pre | L16 H_post | L24 H_pre | L24 H_post |
|:-----:|:--------:|:---------:|:--------:|:---------:|:---------:|:----------:|:---------:|:----------:|
| 0.25 | 1.0000 | 0.9927 | 1.0000 | 0.9275 | 1.0000 | 0.8739 | 1.0000 | 0.5077 |
| 0.50 | 1.0000 | 0.9761 | 1.0000 | 0.9207 | 1.0000 | 0.8718 | 1.0000 | 0.5057 |
| 1.00 | 1.0000 | 0.9416 | 1.0000 | 0.9188 | 1.0000 | 0.8712 | 0.9999 | 0.5053 |
| 2.00 | 1.0000 | 0.9054 | 1.0000 | 0.9183 | 0.9999 | 0.8711 | 0.9994 | 0.5052 |
| 4.00 | 1.0000 | 0.8875 | 0.9999 | 0.9182 | 0.9995 | 0.8711 | 0.9987 | 0.5051 |

Mistral H_pre remains near 1.0 across all layers and alpha values (ceiling effect). H_post L16 variation is 0.0028 (middle-layer invariance), L0 variation is 0.1052 (low-norm early-layer exception).

## Appendix F. Separate Full-Sample Per-Layer Pipeline (Not Directly Comparable to Main Held-Out Results)

**Important**: Tables A7–A8 and Figure 7 report results from a separate per-layer entropy analysis pipeline that evaluates unnormalized/normalized entropy AUROC and Cohen's d across all layers using the full sample without the held-out calibration/test protocol used in the main text (Tables 1–5). These values are **not directly comparable** to the main results. In particular, optimal layers may differ between this pipeline and the held-out protocol (e.g., Mistral H_pre best = L0 here vs. L30 in Table 5).

**Table A7. MMLU per-layer entropy AUROC (best layer, full-sample evaluation)**

| Model | H_pre best | H_pre AUROC | H_post best | H_post AUROC |
|:------|:------------:|:--------------:|:-----------:|:------------:|
| Qwen | L12 | 0.6292 | L27 | 0.6566 |
| Llama | L0 | 0.5958 | L10 | 0.5654 |
| Mistral | L0 | 0.6854 | L24 | 0.6394 |

**Table A8. MMLU Statistical Validation and Nested CV (per-layer pipeline)**

| Model | StatVal H_pre | StatVal H_post | Nested CV H_pre Optimism | Nested CV H_post Optimism |
|:------|:----------------:|:--------------:|:--------------------------:|:------------------------:|
| Qwen | ROBUST | ROBUST | 0.0217 | -0.0003 |
| Llama | ROBUST | ROBUST | -0.0000 | 0.0260 |
| Mistral | ROBUST | ROBUST | 0.0035 | 0.0341 |

All 6 conditions (3 models x 2 metrics) pass statistical validation (Bonferroni-corrected significance, bootstrap CI excluding 0.5, 5-fold CV stability, permutation test).

## Appendix G. Exploratory Analyses Excluded from Main Text

During the research process, some exploratory analyses were conducted but excluded from the main text:

- **Probing analysis**: Logistic regression probes on hidden-state representations achieved moderate accuracy but were confounded by pooling strategy and normalization choices.
- **Causal/mediation analysis**: Associational decompositions of scale and direction contributions were performed but did not support causal interpretation robustly enough to include in the main text.

These are documented here for transparency and to avoid the impression of selective reporting.

## Appendix H. Reproducibility Protocol Card

**Table A9. Experimental protocol card**

| Parameter | Value |
|:----------|:------|
| **Models** | Qwen2.5-7B-Instruct, Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3 |
| **Precision** | FP16 (torch.float16) |
| **Hardware** | 1x NVIDIA RTX 3090 Ti (24 GB VRAM) |
| **Random seed** | 42 (torch, numpy, python hash; cudnn.deterministic=True, benchmark=False). Note: FP16 GPU arithmetic is not fully deterministic across runs; see Section 3.1 for observed variation. |
| **Decoding** | temperature=0.3, top_p=1.0 (greedy for MMLU), max_new_tokens=1024 (math) / 32 (MMLU) |
| **MMLU format** | 5-shot, multiple-choice (A/B/C/D), answer extracted as first token matching [A-D] |
| **Math format** | 0-shot, free-form generation, answer extracted via \\boxed{} pattern matching |
| **Math dataset** | qwedsacf/competition_math (HuggingFace), Level 4-5 for Hard, Level 1-2 for Easy |
| **MMLU dataset** | cais/mmlu, test split, stratified sample of 1000 |
| **ARC dataset** | allenai/ai2_arc, ARC-Challenge train split |
| **Filtering** | Samples with NaN entropy or failed answer extraction excluded (499/500 for Qwen Hard generation-average) |
| **Entropy** | Shannon entropy normalized by log(vocab_size); epsilon=1e-6, gain (gamma) included in RMSNorm |
| **Split** | StratifiedShuffleSplit, test_size=0.3, random_state=42 |
| **Layer selection** | Best layer and sign selected on calibration (70%) set only |
| **Metric** | AUROC on held-out test (30%) set |
| **Bootstrap** | 1000 resamples, 95% percentile CI (no p-values reported; significance judged by CI excluding zero) |

**Scalar baselines (12 total, 8 reported in main Table 1):**

| Baseline | Definition |
|:---------|:-----------|
| Output entropy | H(softmax(final logits)), normalized by log(V) |
| Output max-prob | max(softmax(final logits)) |
| Output margin | max_prob - second_max_prob |
| Length-only | Number of generated tokens |
| h_norm | L2 norm of hidden state at best layer |
| logit_std | Standard deviation of logit vector W*h_l at best layer |
| H_pre | Shannon entropy of softmax(W*h_l) at best layer |
| H_post | Shannon entropy of softmax(W*RMSNorm(h_l)) at best layer |
| logit_max | max(W*h_l) at best layer |
| logit_margin | max(W*h_l) - second_max(W*h_l) at best layer |
| Wh_norm | L2 norm of W*h_l at best layer |
| logit_entropy_ratio | H_pre / logit_std at best layer |

**Logistic regression settings (Tables 2–3, Section 7.2):**

| Parameter | Value |
|:----------|:------|
| Implementation | sklearn.linear_model.LogisticRegression |
| Solver | lbfgs |
| Regularization | C=1.0 (default L2) |
| Feature standardization | StandardScaler (fit on calibration set only) |
| Class weighting | None (balanced not used) |
| Max iterations | 1000 |

**Tuned Lens training (Section 7.1, Appendix D):**

| Parameter | Value |
|:----------|:------|
| Architecture | Per-layer affine translator (W_l, b_l) |
| Training data | wikitext-2 validation split |
| Epochs | 3 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Loss | KL divergence from final-layer output distribution |
| Final loss | 7008 → 2459 |

## Appendix I. Repeated-Split Robustness Analysis

To assess the stability of layer selection, sign selection, and AUROC under different data splits, we repeat the main held-out evaluation protocol 20 times with independent 70/30 stratified splits (seeds 0–19) on all four primary conditions. No model retraining or regeneration is performed; only the calibration/test partition changes.

**Table A10. Repeated-split summary (20 splits, seeds 0–19)**

| Condition | Metric | AUROC mean | std | [min, max] | Modal layer (freq) | Sign consistency |
|:----------|:-------|:----------:|:---:|:----------:|:-------------------:|:----------------:|
| Qwen Hard | H_pre | 0.7364 | 0.047 | [0.624, 0.830] | L16 (13/20) | −1 (20/20) |
| Qwen Hard | H_post | 0.6678 | 0.057 | [0.475, 0.745] | L27 (19/20) | −1 (20/20) |
| Qwen Hard | logit_std | 0.7720 | 0.031 | [0.703, 0.833] | L4 (9/20) | +1 (20/20) |
| Qwen Hard | h_norm | 0.7938 | 0.031 | [0.751, 0.862] | L18 (20/20) | +1 (20/20) |
| Qwen MMLU | H_pre | 0.6149 | 0.031 | [0.566, 0.682] | L18 (7/20) | −1 (20/20) |
| Qwen MMLU | H_post | 0.6607 | 0.023 | [0.615, 0.715] | L27 (20/20) | −1 (20/20) |
| Qwen MMLU | logit_std | 0.6424 | 0.031 | [0.588, 0.710] | L18 (7/20) | +1 (20/20) |
| Qwen MMLU | h_norm | 0.6257 | 0.022 | [0.583, 0.677] | L27 (10/20) | −1 (10/20) |
| Llama MMLU | H_pre | 0.6054 | 0.039 | [0.507, 0.670] | L0 (20/20) | −1 (20/20) |
| Llama MMLU | H_post | 0.5630 | 0.029 | [0.508, 0.611] | L10 (11/20) | +1 (15/20) |
| Llama MMLU | logit_std | 0.6821 | 0.026 | [0.640, 0.732] | L13 (18/20) | +1 (20/20) |
| Llama MMLU | h_norm | 0.5995 | 0.037 | [0.492, 0.654] | L0 (18/20) | +1 (20/20) |
| Mistral MMLU | H_pre | 0.6857 | 0.028 | [0.624, 0.740] | L0 (13/20) | −1 (20/20) |
| Mistral MMLU | H_post | 0.5661 | 0.026 | [0.520, 0.628] | L5 (9/20) | −1 (20/20) |
| Mistral MMLU | logit_std | 0.7248 | 0.024 | [0.671, 0.766] | L13 (11/20) | +1 (20/20) |
| Mistral MMLU | h_norm | 0.7035 | 0.025 | [0.653, 0.748] | L13 (17/20) | +1 (20/20) |

**Key findings from repeated-split analysis:**

1. **H_pre sign is fully stable**: Lower H_pre = correct in all 80/80 condition-split combinations (sign = −1, 20/20 in all four conditions). The previously reported "cross-family sign reversal for H_pre" was incorrect.
2. **Sign instability is not limited to H_post**: H_post sign = +1 in 15/20 splits on Llama MMLU, vs. −1 on Qwen and Mistral. Additionally, h_norm on Qwen MMLU shows sign = −1 in only 10/20 splits (effectively 50/50), making it the only metric-condition pair with no dominant sign direction. Both cases necessitate sign calibration on a held-out set.
3. **Layer selection is metric-dependent**: Some metrics show high stability (h_norm on Qwen Hard: L18 20/20; H_post on Qwen: L27 19–20/20), while others show substantial variability (logit_std on Qwen Hard: L4 9/20, L16 6/20; H_pre on Qwen MMLU: L18 7/20, L12 7/20).
4. **AUROC ordering is stable**: logit_std > H_pre across all 20 splits in 3/4 conditions (Qwen Hard, Llama MMLU, Mistral MMLU). On Qwen MMLU, H_post > logit_std in the majority of splits.
5. **AUROC variability**: Standard deviations range from 0.022 to 0.057, with Qwen Hard showing the highest variability due to smaller sample size (n=499).
