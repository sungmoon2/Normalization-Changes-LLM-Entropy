# RMSNorm 기반 언어 모델에서 정규화 전후가 레이어별 디코딩 엔트로피의 측정 대상을 바꾼다

**박성문 (Sungmoon Park)**
인제대학교 헬스케어IT학과, 김해, 대한민국
qkrtjdans55@oasis.inje.ac.kr

**양진홍 (Jinhong Yang)** (교신저자)
인제대학교 헬스케어IT학과, 김해, 대한민국
jinhong@inje.ac.kr

---

## 초록

레이어별 디코딩 엔트로피—중간 hidden state를 어휘 공간으로 투영하여 계산한 Shannon 엔트로피—는 언어 모델의 내부 불확실성 프록시로 점점 더 활용되고 있다. 그러나 측정 결과에 영향을 미치는 구현 선택이 일관되게 명시되지 않았다: RMSNorm 기반 아키텍처에서 정규화 전후에 계산된 엔트로피는 질적으로 다른 정보를 포착하며, 이를 구분하지 않으면 hidden-state 스케일과 방향 정보를 혼동하게 된다. 본 연구는 pre-RMSNorm 엔트로피(H_pre)와 post-RMSNorm 엔트로피(H_post)를 정밀히 정의하고, Qwen2.5-7B-Instruct, Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3 세 가지 오픈소스 instruction-tuned 디코더 LM에서 수학 추론 및 MMLU 벤치마크를 통해 연구한다. 보정/평가 분할, 스칼라 baseline, 토큰 위치 소거, 렌즈 충실도 검사를 적용하였다.

통제된 스케일링 실험 결과, H_pre는 구조적으로 스케일 민감(scale-sensitive)하여 hidden state를 단위 정규화하면 세 모델 모두에서 H_pre가 최대 엔트로피로 붕괴(평균 > 0.9999)하는 반면, H_post는 중간·후반 레이어에서 대체로 스케일 불변(alpha-sweep 변동 < 0.003)이며, 초기 레이어 잔여 편차는 epsilon 정규화된 RMSNorm 구현과 일치한다. 결정적으로, 이 스케일 민감도는 이론적 수준에 그치지 않는다: 단일 레이어 스칼라 baseline 중에서, logit 표준편차와 같은 단순 스케일 프록시가 네 주요 조건 모두에서 단일 레이어 H_pre의 정답 판별 성능과 동등하거나 이를 능가하며, 테스트된 두 Qwen 조건에서 H_pre를 logit_std에 추가해도 유의미한 증분 효과가 없다(delta = +0.003~+0.020, 95% CI가 0을 포함)—이는 H_pre의 판별 신호가 대부분 스케일로 환원된다는 견해를 뒷받침한다. 그럼에도 적절히 선택된 내부 레이어는 최종 출력 엔트로피를 +0.09~+0.22 AUROC만큼 일관되게 능가한다.

이 결과는 레이어별 디코딩 엔트로피에서 정규화 의존적 측정 교란(measurement confound)을 노출한다: 정규화 지점을 명시하지 않고 logit-lens 투영에서 엔트로피를 보고한 기존 연구—그리고 기존 연구의 이 선택은 일관되지 않다(표 8 참조)—는 분포적 불확실성이 아닌 스케일 통계를 의도치 않게 보고하고 있을 수 있다. 본 연구는 진단 기준, 실행 가능한 권고안, 기존 연구 결과 재해석을 위한 구체적 함의를 제공한다.

**핵심어:** 레이어별 엔트로피, 정규화, RMSNorm, 스케일 민감도, 정답 판별, 선별적 예측, 측정

---

## 1. 서론

레이어별 디코딩 엔트로피—중간 hidden state를 어휘 공간으로 투영하여 계산한 Shannon 엔트로피—는 대규모 언어 모델의 내부 불확실성 프록시로 주목받고 있다(Ali et al., 2025). 더 넓게는, 내부 표현을 탐색하여 품질 및 불확실성 신호를 얻는 것이 생산적인 연구 방향으로 자리 잡았다(Ghandeharioun et al., 2024; Kossen et al., 2024). 모델의 내부 확률 분포가 레이어에 걸쳐 어떻게 변화하는지 분석함으로써, 연구자들은 "핵심 레이어"를 식별하고 정답과 상관하는 엔트로피 프로필을 구축해 왔다.

그러나 측정 결과에 영향을 미치는 구현 선택이 일관되게 명시되지 않았다. RMSNorm을 사용하는 트랜스포머 구조에서 각 레이어의 hidden state는 후속 처리 전에 정규화를 거치며, 정규화 전후에 계산된 엔트로피는 질적으로 다른 양을 측정할 수 있다—그런데 기존 연구는 이 선택에 일관성이 없다: 최종 정규화를 포함하는 연구, 명시적으로 생략하는 연구, 명시하지 않는 연구가 혼재한다(8.1절 표 8의 문헌 점검 참조). 이는 사소한 구현 세부사항이 아니다: RMSNorm은 hidden state에서 radial scale을 제거하므로, 정규화 전 엔트로피가 그 스케일에 지배된다면, logit-lens 엔트로피를 정규화 없이 사용하는 연구자들은 분포적 불확실성이 아닌 hidden-state 크기를 보고하고 있을 수 있다. "핵심 레이어", 엔트로피 프로필, 불확실성 추정에 대한 모든 후속 결론은 재해석을 필요로 하게 된다.

본 논문은 다음을 묻는다: **레이어별 디코딩 엔트로피는 실제로 무엇을 측정하며, 정규화 지점은 그 답을 어떻게 바꾸는가?**

동일한 hidden state h_l에서 두 가지 양을 정의한다: **H_pre**는 언어 모델 헤드 W를 직접 적용한 softmax(W h_l)의 엔트로피이고, **H_post**는 softmax(W RMSNorm(h_l))의 엔트로피이다. 둘 다 진단적 투영(diagnostic projection)으로, 해당 지점에서 디코딩되도록 설계되지 않은 중간 표현에 최종 레이어 디코더를 적용한 것이다. 핵심 발견은 이 두 "엔트로피"가 같은 신호가 아니라는 것이다. h_l = r_l u_l (반경 크기 r_l과 방향 u_l)로 분해하면, H_pre는 pre-norm logit이 r_l에 선형적으로 스케일되므로 구조적으로 스케일 민감하며, 이때 r_l은 암묵적 온도(implicit temperature)로 작용한다. RMSNorm 이후 계산되는 H_post는 반경 스케일이 제거되어 크기 재조정에 대해 대체로 불변이다—다만 구현의 epsilon 항이 정확한 불변성을 깨는 저-norm 초기 레이어는 예외이다. 이러한 진단적 투영에 대한 렌즈 충실도 검토는 7절에서 다룬다.

이 구분은 기존 결과의 해석과 향후 실험 설계 모두에 구체적 영향을 미친다:

1. **단일 레이어 스칼라 baseline 중에서, H_pre의 판별 신호는 대부분 스케일로 환원된다.** 직접적 스케일 통계(logit 표준편차, hidden-state norm)가 네 주요 조건 모두에서 단일 레이어 H_pre의 정답 판별 성능과 동등하거나 이를 능가한다. 테스트된 두 Qwen 조건에서 H_pre를 logit_std에 추가해도 유의미한 증분 효과가 없다(delta-AUROC +0.003~+0.020, 모두 95% CI가 0을 포함). 이는 정규화 전 엔트로피가 분포적 불확실성이 아닌 스케일을 측정하고 있다는 것과 일치한다.

2. **내부 레이어는 여전히 최종 출력 엔트로피를 능가한다.** 모든 조건과 모델에서 최적 내부 레이어 신호가 최종 출력 엔트로피를 +0.09~+0.22 AUROC만큼 초과하며, 정규화 지점이 명시되는 한 중간 표현이 출력 분포 너머의 유용한 정답 정보를 담고 있음을 확인한다.

3. **본 프로토콜 하에서 보편적 최적 레이어가 관찰되지 않는다.** 최적 레이어, 부호, 어떤 엔트로피 변형이 더 유용한지 모두 모델, 과제, 평가 프로토콜에 따라 달라지며, Llama에서의 H_post 부호 반전(높은 H_post = 정답, 반복 분할 15/20, Qwen과 Mistral에서는 낮을수록 정답)도 관찰된다. H_pre 부호는 안정적이다(낮을수록 정답, 모든 조건에서 20/20).

4. **디코더 및 분류기 선택이 결론에 실질적으로 영향을 미친다.** Logit Lens는 Qwen 중간 레이어에서 최종 레이어 분포와의 top-1 일치율 0%를 보이며, Entropy-Lens의 분류기를 k-NN에서 로지스틱 회귀로 변경하면 AUROC가 +0.17까지 변하여, 디코더 및 분류기 선택이 민감한 설계 매개변수임을 나타낸다.

기여:

- **C1.** H_pre와 H_post를 정의·구분하고, 정규화 지점이 디코딩 엔트로피 신호를 질적으로 바꿈을 실증한다.
- **C2.** 세 모델에 걸쳐 unit-norm 제거 및 alpha-sweep 개입을 통해 H_pre가 구조적으로 스케일 민감하고 H_post가 대체로 스케일 불변임을 직접 기하학적 증거로 실증한다.
- **C3.** 공정한 held-out 프로토콜과 12개 스칼라 baseline 하에서, 단일 레이어 스칼라 중 단순 스케일 프록시가 엔트로피와 동등하거나 능가함을 보이되, 내부 레이어가 출력 전용 엔트로피를 일관되게 능가함을 확인한다.
- **C4.** 본 held-out 프로토콜 하에서 결론이 레이어, 토큰 위치, 모델, 디코딩 렌즈에 강하게 의존함을 관찰한다—보편적 최적 레이어 부재 및 모델 간 부호 반전 포함.
- **C5.** 정규화 지점을 명시하지 않고 logit-lens 디코딩 엔트로피를 사용한 기존 연구에 대한 구체적 함의를 식별하고, 향후 연구를 위한 실행 가능한 진단 기준을 제공한다.

본 논문은 주의 환기형 측정 연구이다: 새로운 방법을 제안하는 것이 아니라, 기존 측정의 해석에 영향을 미치는 교란 요인을 노출한다. 추론의 메커니즘적 추적, 인과적 매개, 보편적 불확실성 추정, 또는 self-consistency에 대한 우월성을 주장하지 않는다. 레이어별 엔트로피가 **무엇을** 측정하는지 연구하지, 추론이 **어디에서** 일어나는지 연구하지 않는다.

---

## 2. 관련 연구

### 2.1 LLM의 내부 불확실성 신호

여러 연구가 LLM 내부에서 불확실성 신호를 추출한다. Semantic entropy (Kuhn et al., 2023; Farquhar et al., 2024)는 다중 샘플 생성을 의미 클러스터링하여 분포적 엔트로피를 계산하며, 강력한 할루시네이션 탐지를 달성하나 다중 forward pass가 필요하다. Semantic Entropy Probes (Kossen et al., 2024)는 단일 패스 hidden state로 이를 근사하되 지도 학습이 필요하다. DRIFT (Bhatnagar et al., 2026)는 중간 레이어에 프로브를 학습시켜 사실적 오류를 나타내는 표현적 비일관성을 탐지하며, AUROC 최대 0.94를 보고한다. FLUE (Gao et al., 2025)는 hidden-state 엔트로피가 예측 엔트로피의 상한을 제공함을 확립한다. Ghandeharioun et al. (2024)은 hidden representation을 자연어 프롬프트에 패칭하여 검사하는 Patchscopes 프레임워크를 제안한다. Chen et al. (2025)은 Internal Confidence를 제안하여, 생성 시작 전 내부 표현으로부터 질의 수준 불확실성을 학습 없이 평가하며, 사전 생성(pre-generation) 신호와 직접 관련된다. 본 연구는 이들과 달리 새로운 불확실성 방법을 제안하는 것이 아니라, 널리 사용되는 레이어별 디코딩 엔트로피가 서로 다른 정규화 선택 하에서 **실제로 무엇을 측정하는지**를 묻는다.

### 2.2 레이어별 엔트로피 및 렌즈 기반 디코딩

Logit Lens (nostalgebraist, 2020)는 중간 hidden state를 언어 모델 헤드를 통해 어휘 공간으로 투영하여 레이어별 예측 변화를 검사한다. Tuned Lens (Belrose et al., 2023)는 학습된 affine translator로 representational drift를 줄여 이를 개선하며, 특히 logit lens를 LayerNorm 포함으로 명시적으로 정의한다. Geva et al. (2022)은 feed-forward 레이어가 어휘 공간에서 개념을 촉진하는 방식을 분석하면서, 투영에서 LayerNorm을 명시적으로 생략하고 부록에서 이 생략이 상위 토큰 해석에 실질적 영향을 미치지 않음을 확인하였다—중간 디코딩에서 정규화가 설계 선택임을 일찍이 인정한 것이나, 엔트로피 의미론에 대한 영향은 검토하지 않았다. Entropy-Lens (Ali et al., 2025)는 레이어 간 logit-lens 확률의 엔트로피 역학을 연구하여 확장(expansion) 및 가지치기(pruning) 의사결정 전략을 특성화하며, k-NN이 사용될 때 이를 제안된 예측 모델이 아닌 진단적 프로브로 정의한다. DoLa (Chuang et al., 2024)는 초기 레이어와 후기 레이어의 logit 분포를 대비하여 사실성을 개선하되, 중간 hidden state를 어휘 헤드로 투영할 때 정규화를 명시하지 않는다. SimLens (Ma et al., 2025)는 추가 토큰 하나를 활용하여 중간 레이어 디코딩 정확도를 개선하며, 초기 레이어에서의 렌즈 드리프트 문제를 해결한다. 본 연구는 이 계보 위에 직접 구축되나 다른 질문을 던진다: 디코딩 엔트로피를 고정된 양으로 취급하는 대신, 정규화 지점(pre- vs. post-RMSNorm)의 선택이 엔트로피가 측정하는 대상을 어떻게 질적으로 바꾸는지를 조사한다. 또한 어떤 특징이 가장 판별적인지에 대한 결론이 분류기 선택에 민감함을 보인다 (7절).

### 2.3 정규화, 스케일, 신뢰도

RMSNorm (Zhang & Sennrich, 2019)은 hidden state에서 radial scale을 제거하며, 본 연구는 이 성질을 활용하여 엔트로피를 스케일 의존적 성분과 방향 의존적 성분으로 분해한다. Brody et al. (2023)은 정규화가 선형 투영과 상호작용하여 어텐션의 유효 표현력을 바꿈을 보였다. Confidence Regulation Neurons (Stolfo et al., 2024)는 특정 뉴런이 hidden-state norm을 스케일링하여 출력 신뢰도를 조절함을 식별하여, 스케일과 모델 확신 간의 메커니즘적 연결을 제공하였다. Sun et al. (2024)은 LLM에서 희소하지만 극단적으로 큰 activation 이상치가 모델 행동에 불균형적 기능적 영향을 미침을 문서화하였다. Katz and Belinkov (2023)은 나아가 최종 LayerNorm이 중간 레이어 디코딩에서 "의미적 필터(semantic filter)"로 작용함을 보이며, logit lens를 LL(x) = softmax(ln_f(x) D)로 정규화를 명시적으로 포함하여 정의하고, LayerNorm 적용이 토큰 수준 확률 할당을 변경함을 실증하였다. 이 발견들은 종합적으로 본 연구의 핵심 질문을 동기 부여한다: 스케일이 신뢰도 관련 정보를 담고 있고 정규화가 스케일을 제거한다면, 정규화 전후 엔트로피는 반드시 다른 것을 측정해야 한다. 본 연구의 개입 실험 (4절)은 이 가설에 대한 직접적 교차 모델 증거를 제공한다.

### 2.4 선별적 예측

선별적 예측은 시스템이 불확실한 입력에 대해 기권하여 수용된 부분집합의 정밀도를 개선하는 것을 허용한다. Geifman and El-Yaniv (2017)는 risk-coverage 프레임워크를 확립하였다. 최근 연구는 이를 LLM으로 확장한다: REFRAIN (Sun et al., 2025)은 연쇄 추론(chain-of-thought)을 위한 SW-UCB 기반 적응형 조기 중단 프레임워크를 사용하고, self-consistency (Wang et al., 2023)는 다중 샘플에 대한 다수결 투표를 신뢰도 프록시로 사용한다. 본 연구는 선별적 예측을 주요 기여가 아닌, 다양한 내부 신호의 실용적 유용성을 평가하는 **평가 렌즈**로 사용한다 (5절).

---

## 3. 정의 및 평가 프로토콜

### 3.1 모델 및 데이터

모두 최종 정규화로 RMSNorm을 사용하는 세 가지 instruction-tuned 디코더 LM을 연구한다: Qwen2.5-7B-Instruct (Qwen Team, 2024), Llama-3-8B-Instruct (Grattafiori et al., 2024), Mistral-7B-Instruct-v0.3 (Jiang et al., 2023).

| 모델 | Family | 레이어 | Hidden | Vocab |
|:-----|:------:|:------:|:------:|:-----:|
| Qwen2.5-7B-Instruct | Alibaba | 28 | 3584 | 152,064 |
| Llama-3-8B-Instruct | Meta | 32 | 4096 | 128,256 |
| Mistral-7B-Instruct-v0.3 | Mistral AI | 32 | 4096 | 32,768 |

모든 실험은 NVIDIA RTX 3090 Ti (24GB), FP16 추론, 고정 랜덤 시드(seed=42)를 사용한다. 세 가지 관련 분석 프로토콜을 사용하며, 아래 마스터 실험 표와 프로토콜 참고를 참조한다.

주요 평가 조건:
- **Qwen Hard**: competition_math Level 4-5 (Hendrycks et al., 2021b), NaN 필터링 후 499개 유효 샘플(271 정답, 정답률 54.3%). 동일 항목의 초기 실험은 55.4%(277 정답)를 산출하였으며, 차이는 동일 seed 및 deterministic 설정에도 FP16 GPU 비결정성에 의한 것이다.
- **Qwen MMLU**: MMLU test (Hendrycks et al., 2021a), 1000 samples (정답률 74.9%)
- **Llama MMLU**: MMLU test, 1000 samples (정답률 63.8%)
- **Mistral MMLU**: MMLU test, 1000 samples (정답률 62.7%)

추가 조건(Qwen Easy, Qwen ARC (Clark et al., 2018), Llama Hard)은 부록 A에 수록.

**마스터 실험 표.**

| 절 | 프로토콜 | 데이터셋 | n | 토큰 위치 | 분할 | 레이어 선택 | 지표 |
|:---|:---------|:---------|:-:|:----------|:-----|:------------|:-----|
| 4.2 | 스케일 개입 | Qwen Hard | 500 | Step 0 | 없음 | 전체 레이어 | 평균 H_pre/H_post |
| 4.3 | 스케일 개입 | MMLU (3모델) | 300 | Step 0 | 없음 | 전체 레이어 | 평균 H_pre/H_post |
| 5.1-5.2 | Baseline 평가 | Qwen Hard / MMLU (3모델) | 499/1000 | 생성 평균 | 70/30 보정/평가 | 보정 세트에서 | Held-out AUROC |
| 6.1 | 토큰 위치 소거 | Qwen Hard / MMLU | 500/1000 | Step 0, 1, Full avg | 70/30 보정/평가 | 보정 세트에서 | Held-out AUROC |
| 7.1 | 렌즈 충실도 | Qwen MMLU | 1000 | Step 0 | 70/30 보정/평가 | 보정 세트에서 | AUROC, KL, top-1 일치 |
| 7.2 | 분류기 비교 | 4개 조건 전체 | 499/1000 | 생성 평균 | 70/30 보정/평가 | 보정 세트에서 | Held-out AUROC |

**프로토콜 참고.** 위 마스터 표에 요약된 바와 같이, 본 논문에서는 세 가지 주요 분석 프로토콜을 사용한다. *baseline 평가 프로토콜* (5절, 표 1-3)은 필터링 후 499/1000 샘플에서 생성 평균 엔트로피를 사용하고, 70/30 보정 세트에서 최적 레이어/부호를 선택한다. *토큰 위치 소거 프로토콜* (6.1절, 부록 B)은 MMLU의 경우 1000개 항목을 독립적으로 재샘플링하고, Hard의 경우 동일 500개 항목을 재실행한다. MMLU 정답률 차이(70.8% vs. 74.9%)는 다른 샘플 항목 때문이고, Hard 정답률 차이(53.2% vs. 54.3%)는 동일 항목에서 FP16 GPU 비결정성으로 54/499개 항목이 다른 응답을 받았기 때문이다. *스케일 개입 프로토콜* (4절)은 원본 샘플에서 Step 0 (prompt-last) 위치를 사용한다. 모든 프로토콜은 seed=42, 계층화 분할을 사용한다.

### 3.2 H_pre와 H_post

레이어 l의 hidden state h_l, 언어 모델 헤드 W, 최종 RMSNorm N_eps에 대해:

$$H_{\text{pre},l} = H(\text{softmax}(W h_l))$$
$$H_{\text{post},l} = H(\text{softmax}(W \cdot N_\epsilon(h_l)))$$

H는 log(|V|)로 정규화된 Shannon 엔트로피이다.

모든 실험에서 h_l은 트랜스포머 블록 l 이후, 모델의 최종 정규화 모듈 이전의 잔차 스트림(residual stream) 출력을 나타낸다. H_post는 모델의 학습된 최종 RMSNorm(gain 파라미터 gamma 및 epsilon 포함)을 출력 헤드 이전에 적용하고, H_pre는 해당 모듈을 우회한다. 둘 다 진단적 투영(diagnostic projection)으로, 해당 지점에서 디코딩되도록 설계되지 않은 중간 표현에 최종 레이어 디코더를 적용한 것이다. 세 모델 모두 RMSNorm 구현에서 epsilon = 1e-6을 사용하며, gain 파라미터(gamma)가 포함된다. 렌즈 충실도는 7절에서 검토한다.

### 3.3 스케일 분해

h_l = r_l u_l (r_l = ||h_l||, ||u_l|| = 1)로 쓰면:

- **Pre-norm logit**: W h_l = r_l W u_l. 스칼라 r_l이 역온도(inverse temperature)로 작용: r_l이 클수록 softmax가 집중되어 H_pre가 감소.
- **Post-norm logit**: RMSNorm_eps(alpha · h) = gamma · h / sqrt(mean(h^2) + eps/alpha^2)이므로, mean(h^2) >> eps인 중간·후반 레이어에서 alpha가 소거되어 H_post가 스케일 불변. 저-norm 초기 레이어에서는 eps 항이 잔여 alpha-의존성을 유발.

### 3.4 토큰 위치

- **Step 0 (prompt-last)**: 생성 시작 전 입력 프롬프트의 마지막 토큰.
- **Step 1 (first-gen)**: 첫 번째 생성 토큰.
- **Full average**: 전체 생성 토큰의 평균 엔트로피.

별도 명시가 없으면, 정답 판별에는 생성 평균 위치, 스케일 개입에는 Step 0 (prompt-last)를 사용한다.

### 3.5 평가 프로토콜

- **분할**: 70/30 계층화 보정/평가 분할 (StratifiedShuffleSplit, seed=42).
- **레이어·부호 선택**: 보정 세트에서만 선택. 평가 세트는 선택에 사용하지 않음.
- **지표**: Held-out test AUROC.
- **Baseline**: 스칼라 baseline—출력 수준(엔트로피, max-prob, margin), 내부 스칼라(H_pre, H_post, logit_std, h_norm 등), length-only. 표 1은 네 주요 조건에 대해 가장 유익한 8개 baseline을 보고한다. 부록 A는 주요 held-out baseline 프로토콜에 포함되지 않은 보충 평가 조건(Qwen Easy, Qwen ARC, Llama Hard)을 보고한다.
- **유의성**: Held-out test 세트에서 쌍체 부트스트랩 (1,000 리샘플). delta-AUROC와 95% 백분위 부트스트랩 신뢰구간을 함께 보고하며, 0을 포함하는 구간은 비유의미로 처리한다. 0을 포함하는 구간은 비유의미로 판단한다.

---

## 4. 두 엔트로피가 측정하는 것: 스케일, Epsilon, 포화

이 절은 논문의 핵심 측정 증거를 제공한다: H_pre가 hidden-state 크기에 의존하고 H_post는 의존하지 않음을 보이는 직접적 기하학적 개입 실험. 예측적 함의는 5절에서 평가한다.

### 4.1 수학적 직관

3.3절에서 pre-norm logit이 W h_l = r_l W u_l이며 r_l이 역온도로 작용함을 상기한다. 이를 정형화하면:

**명제.** 임의의 고정 로짓 벡터 z in R^V에 대해, H_raw(alpha) = -sum_i p_i log p_i (여기서 p_i = softmax(alpha z)_i)로 두면 dH_raw/d(alpha) = -alpha Var_{p_alpha}(z) <= 0이다. 여기서 Var는 p_alpha 하의 분산이다. 즉 비정규화 Shannon 엔트로피는 스케일 인자 alpha에 대해 단조 비증가한다. 본 연구의 정규화 엔트로피 H = H_raw / log|V|는 양의 상수만 차이나므로 단조성이 보존된다: dH/d(alpha) = -(alpha / log|V|) Var_{p_alpha}(z) <= 0.

이는 d(-sum_i p_i log p_i)/d(alpha) = -alpha sum_i p_i(z_i - z_bar)^2 항등식에서 직접 따른다. r_l을 증가시키면 softmax 분포가 집중되어 H_pre가 감소한다. Post-norm logit은 RMSNorm을 거치며, 표준 구현 하에서:

$$N_\epsilon(\alpha h) = \gamma \odot \frac{h}{\sqrt{\text{mean}(h^2) + \epsilon / \alpha^2}}$$

mean(h^2) >> epsilon인 경우(큰 hidden-state norm을 가진 중간·후반 레이어), alpha가 근사적으로 소거되어 H_post가 효과적으로 스케일 불변이 된다. 저-norm 초기 레이어에서는 epsilon 항이 무시할 수 없게 되어 정확한 불변성이 깨진다.

두 가지 개입으로 이 예측을 검증한다: unit-norm 제거 및 alpha-sweep.

### 4.2 Qwen 사례 연구: 스케일 개입

Qwen2.5-7B-Instruct에 competition_math Hard (500 samples, Step 0 위치)에서 개입을 적용한다.

**Unit-norm 제거** (h → h/||h||): 반경 크기를 제거하면 모든 레이어에서 H_pre가 최대 엔트로피로 붕괴하는 반면, H_post 변화는 0.003 미만:

| 레이어 | H_pre 원본 | H_pre 단위 | 변화 | H_post 원본 | H_post 단위 | 변화 |
|:-----:|:----------:|:----------:|:----:|:-----------:|:-----------:|:----:|
| 0 | 0.9963 | 1.0000 | +0.004 | 0.4788 | 0.4812 | +0.002 |
| 4 | 0.9705 | 1.0000 | +0.030 | 0.0780 | 0.0786 | +0.001 |
| 12 | 0.8577 | 1.0000 | +0.142 | 0.0484 | 0.0491 | +0.001 |
| 16 | 0.6637 | 0.9999 | +0.336 | 0.1822 | 0.1843 | +0.002 |
| 24 | 0.0444 | 0.9999 | +0.956 | 0.0861 | 0.0864 | +0.000 |
| 27 | 0.0166 | 1.0000 | +0.983 | 0.0307 | 0.0308 | +0.000 |

**Alpha-sweep** (h → alpha · h)은 대표 레이어에서 H_pre가 스케일 조작에 강하게 반응하는 반면 H_post는 소수점 넷째 자리까지 불변임을 확인한다:

| Alpha | L4 H_pre | L4 H_post | L16 H_pre | L16 H_post |
|:-----:|:--------:|:---------:|:---------:|:----------:|
| 0.25 | 0.9985 | 0.0780 | 0.9817 | 0.1822 |
| 1.00 | 0.9705 | 0.0780 | 0.6637 | 0.1822 |
| 4.00 | 0.5169 | 0.0780 | 0.0990 | 0.1822 |

Qwen에서 L16의 alpha-sweep 하에 H_pre는 0.10~0.98 범위를 포괄하는 반면, H_post는 소수점 넷째 자리까지 변동이 없다. 이는 H_pre가 구조적으로 스케일 민감하고 H_post가 이 모델에서 효과적으로 스케일 불변임을 확립한다.

### 4.3 교차 모델 검증

이 발견이 Qwen에 한정되지 않음을 검증하기 위해, Llama-3-8B 및 Mistral-7B에서 동일 조건(MMLU 300 samples, Step 0, seed=42)으로 개입을 반복한다. 비교 가능성을 위해 Qwen도 동일한 MMLU-300 설정으로 재실행한다(참고: 4.2절은 competition_math Hard 500을 사용).

**표 4. 교차 모델 스케일 개입 요약 (MMLU 300, Step 0, 동일 시드)**

| 모델 | 레이어 | H_pre unit-norm 평균 | H_post 최대 변동 | 최악 레이어 | 최악 h_norm | 원본 H_pre > 0.99인 레이어 비율 |
|:-----|:------:|:--------------------:|:----------------:|:-----------:|:-----------:|:-------------------------------:|
| **Qwen** | 28 | **0.999947** | **0.000454** (L0) | L0 | 10.21 | 0% |
| **Llama** | 32 | **0.999989** | **0.407632** (L1) | L1 | 0.89 | 66% |
| **Mistral** | 32 | **0.999999** | **0.105224** (L0) | L0 | 0.25 | 91% |

참고: 마지막 열은 수정되지 않은 *원본* H_pre가 이미 0.99를 초과하는 레이어 비율로, 천장 효과(ceiling effect)를 나타낸다. 이는 모든 모델이 구조적으로 ~1.0에 수렴하는 unit-norm 결과(3열)와 구별된다.

세 모델 모두에서 unit-normalization이 H_pre를 최대 엔트로피로 붕괴시키며(평균 > 0.9999), 모델 아키텍처에 관계없이 H_pre가 구조적으로 스케일 민감함을 확인한다.

H_post는 중간·후반 레이어에서 대체로 스케일 불변이다: L16에서 alpha-sweep 변동은 0.0023(Llama) 및 0.0028(Mistral). L24에서 Llama는 정확한 불변성을 달성한다(H_post = 0.0689, 모든 alpha 값에서 동일).

### 4.4 저-Norm 초기 레이어 예외

H_post alpha-불변성은 초기 레이어에서 붕괴한다: Llama L1은 변동 0.41, Mistral L0은 0.11, Qwen L0은 0.0005. 이 편차는 hidden-state norm과 상관한다: Mistral L0의 평균 h_norm = 0.25, Llama L1 = 0.89, Qwen L0 = 10.21.

이는 RMSNorm 수학과 일치한다: 실제 구현은 0이 아닌 epsilon을 사용하므로 스케일 불변성은 근사적이다. 레이어 RMS가 매우 작으면 epsilon 항이 alpha-스케일링 하에서 무시할 수 없게 되어, 초기 저-norm 레이어는 정확한 불변성에서 이탈할 수 있다.

따라서 H_post를 중간·후반 레이어의 실질적 관심 구간(operating regime)에서 대체로 스케일 불변으로 취급하되, 정확한 전역 불변성이 아닌 명시적 저-norm 초기 레이어 편차를 수반하는 것으로 정리한다.

### 4.5 포화 제한 관측성

Unit-norm 제거는 스케일 민감도가 구조적으로 *존재*함을 확립하지만, alpha-sweep은 그 *관측 가능한 크기*가 모델에 따라 극적으로 다름을 드러낸다:

| 모델 | 중간 레이어 H_pre 범위 (alpha 0.25-4.0) | 동적 범위 |
|:-----|:---------------------------------------:|:---------:|
| Qwen L14 | 0.058 – 0.982 | **0.924** (광범위) |
| Llama L16 | 0.992 – 1.000 | **0.008** (천장) |
| Mistral L16 | 0.999 – 1.000 | **0.001** (극단적 천장) |

Llama와 Mistral에서 H_pre는 hidden-state norm이 작기 때문에(Mistral L0: 0.25 vs. Qwen L0: 10.21) 초기~중간 레이어에서 이미 최대 엔트로피(~1.0) 근처에 있다. 이 천장 효과(ceiling effect)는 관측 가능한 alpha-sweep 반응을 압축하여, 구조적으로 존재함에도 불구하고 중간 레이어에서 스케일 민감도를 실질적으로 보이지 않게 만든다.

이는 norm-binned 통제 결과(부록 C)와 일치한다: H_pre를 logit_std로 통제하면, Mistral은 AUROC의 100.1%를 유지한다—이는 H_pre가 스케일 불변이라서가 아니라, H_pre가 이미 천장에 있어 스케일 너머의 추가 정보를 거의 담지 않기 때문이다. 반면 Mistral의 H_pre를 h_norm으로 통제한 유지율은 85.2%이다(부록 C, 표 A3).

H_pre는 구조적으로 스케일 민감하나, 이 민감도의 관측 가능한 크기는 포화에 의해 제한되며 따라서 모델 및 레이어 의존적이다.

---

## 5. 내부 신호의 예측적 유용성

4절에서 H_pre가 구조적으로 스케일 민감함을 확립한 후, 이제 예측적 지형을 평가한다—어떤 내부 신호가 정답/오답 응답을 가장 잘 판별하는지—그리고 H_pre가 단순 스케일 프록시가 이미 포착한 정보 너머의 것을 담는지 검정한다.

### 5.1 내부 레이어가 최종 출력 엔트로피를 능가한다

표 1은 네 가지 주요 평가 조건에서 단일 패스 스칼라 baseline의 held-out test AUROC를 제시한다. 주된 실증적 발견은 H_pre가 직접적 스케일 프록시를 종종 능가하지 못한다는 점이다.

**표 1. Held-out test AUROC — 단일 패스 스칼라 baseline (70/30 분할, 보정 세트에서 최적 레이어 선택)**

| 방법 | 패스 | Qwen Hard | Qwen MMLU | Llama MMLU | Mistral MMLU |
|:-----|:----:|:---------:|:---------:|:----------:|:------------:|
| 출력 엔트로피 | 1 | 0.6316 | 0.5775 | 0.5354 | 0.5070 |
| 출력 max-prob | 1 | 0.8041 | 0.6177 | 0.5211 | 0.4968 |
| 출력 margin | 1 | 0.6762 | 0.5216 | 0.5141 | 0.4685 |
| Length-only | 1 | 0.7964 | 0.6428 | 0.5325 | 0.6733 |
| h_norm | 1 | 0.8256 | 0.5740 | 0.5763 | 0.6983 |
| logit_std | 1 | **0.8086** | **0.6674** | **0.6522** | **0.7315** |
| H_post | 1 | 0.6613 | 0.6376 | 0.5796 | 0.5850 |
| **H_pre** | **1** | **0.7672** | **0.6367** | **0.6021** | **0.6737** |

세 가지 패턴이 나타나며, 두 번째는 본 논문이 식별하는 측정 교란의 핵심이다:

1. **내부 > 최종 출력 엔트로피**: 모든 조건에서 최적 내부 레이어 신호가 출력 엔트로피를 +0.09~+0.22 AUROC만큼 초과한다. 다만 Qwen Hard에서는 출력 max-prob(0.8041)이 이미 강한 출력 수준 baseline이며, 내부 우위는 모든 조건에서 최강 출력 수준 신뢰도 신호를 이긴다는 것이 아니라 출력 엔트로피를 기준으로 한 것임을 유의한다.
2. **단일 레이어 스칼라 중 logit_std가 H_pre를 일관되게 능가**: 가장 강한 스케일 프록시인 logit_std가 네 조건 모두에서 단일 레이어 H_pre를 초과한다(h_norm은 모든 조건에서 균일하게 그렇지는 않음). 이는 H_pre가 주로 hidden-state 스케일을 포착하고 있다면 예상되는 결과이다(4절): 직접적 스케일 통계가 softmax 온도 효과를 통해 간접적으로 스케일을 반영하는 엔트로피 측정보다 최소한 동등한 정보를 제공해야 한다. 다만 다중 레이어 H_pre 프로필(표 2)은 단일 레이어 스케일 프록시를 능가할 수 있어, 전체 엔트로피 궤적이 단일 스칼라 너머의 정보를 담고 있음을 시사한다.
3. **Length-only가 의외로 강하다**: Length-only는 Qwen Hard에서 0.7964, Mistral MMLU에서 0.6733으로 H_pre에 근접하거나 초과한다. 주요 결과가 생성 평균 토큰 위치를 사용하므로 길이 교란(length confounding)이 우려된다. 그러나 Qwen에서의 6.1절 Step 0 (prompt-last, 생성 전) 결과는 길이가 정의되지 않는 상태에서도 내부 신호가 실질적 판별력(AUROC 0.59~0.71)을 가짐을 보여, 신호가 길이 교란만은 아님을 확인한다.

**표 2. 다중 레이어 프로필 및 self-consistency (held-out test AUROC)**

| 방법 | 패스 | Qwen Hard | Qwen MMLU | Llama MMLU | Mistral MMLU |
|:-----|:----:|:---------:|:---------:|:----------:|:------------:|
| H_post 프로필 (LR) | 1 | 0.8060 | 0.6412 | 0.6573 | 0.6388 |
| H_pre 프로필 (LR) | 1 | 0.8467 | 0.7042 | 0.7006 | 0.6955 |
| h_norm 프로필 (LR) | 1 | **0.8660** | 0.6757 | **0.7165** | **0.7594** |
| 동일 조건 SC (K=5, temp=0.3) | 5 | 0.7905* | — | — | — |

*SC AUROC는 합의-정답 상관; 동일 온도 하에서 Qwen Hard에서만 수행.

최적 다중 레이어 프로필은 최적 단일 레이어 스칼라를 일관되게 능가한다(+0.02~+0.10). 프로필 간에는 h_norm 프로필이 4개 조건 중 3개에서 최강이며, 스케일이 판별의 중요한 동인임과 일치한다. 동일 온도(0.3)에서 self-consistency(K=5)는 5배 계산 비용으로 합의 AUROC 0.7905를 달성하지만, logit_std는 1배 비용으로 0.8086을 달성한다.

### 5.2 H_pre는 스케일 프록시와 대부분 중복된다: 스케일 민감도의 증거

표 1의 결과는 logit_std가 네 조건 모두에서 H_pre를 능가함을 보인다. 본 연구의 측정 주장이 맞다면—H_pre가 주로 hidden-state 스케일을 포착한다면—H_pre를 logit_std에 추가해도 증분 정보가 거의 없어야 한다. 둘 다 같은 기저 양을 측정하고 있기 때문이다. 이 예측을 직접 검정한다. 보정 세트에서 H_pre 유무에 따른 로지스틱 회귀 모델을 학습하고 held-out test 세트에서 평가한다(표 3). 이 분석은 현재 두 Qwen 조건에 한정되며, Llama 및 Mistral에서 유사한 결과가 성립하는지는 미검증이다.

**표 3. H_pre의 스케일 프록시 대비 증분 효용 (held-out test AUROC)**

| Feature | Qwen Hard | Qwen MMLU |
|:--------|:---------:|:---------:|
| logit_std만 | 0.8086 | 0.6674 |
| H_pre만 | 0.7672 | 0.6367 |
| logit_std + H_pre | 0.8284 | 0.6705 |
| logit_std + h_norm + length | 0.8120 | 0.6466 |
| logit_std + h_norm + length + H_pre | 0.8143 | 0.6427 |
| **Delta (logit_std → +H_pre)** | **+0.020 [−0.014, +0.057]** | **+0.003 [−0.003, +0.009]** |
| **Delta (full → +H_pre)** | **+0.002 [−0.022, +0.027]** | **-0.004 [−0.012, +0.003]** |

두 조건 모두에서 H_pre 추가의 95% bootstrap CI가 0을 포함하여 유의미한 개선이 관찰되지 않는다(1000 리샘플). 이 결과는 스케일 민감도 가설과 일치한다: H_pre가 주로 hidden-state 크기의 함수라면, 동일한 스케일의 보다 직접적인 측정인 logit_std가 H_pre가 제공하는 정보를 이미 포착하고 있어야 하며, 잔여가 남지 않아야 한다. H_pre 동적 범위가 더 제한적인 Llama 및 Mistral(4.5절)에서 동일한 결과가 성립하는지는 미검증이다.

---

## 6. 신호가 나타나는 위치와 시점

### 6.1 토큰 위치

Qwen Hard (500 samples) 및 Qwen MMLU (1000 samples)에서 세 추출 위치를 비교한다:

| 지표 | Step 0 | Step 1 | Full Avg | 최적 |
|:-----|:------:|:------:|:--------:|:-----|
| **H_pre (Hard)** | 0.7087 | **0.7486** | 0.7479 | Step 1 |
| **H_pre (MMLU)** | 0.5911 | **0.6192** | 0.5968 | Step 1 |

H_pre 신호는 첫 번째 생성 토큰(Step 1)에서 가장 강하고 포화한다—Step 1과 전체 생성 평균의 차이는 Hard에서 0.0007, MMLU에서 0.022이다. Step 0 (prompt-last, 생성 전)도 상당한 신호를 담고 있어(AUROC 0.59~0.71), 생성 시작 전 불확실성 추정이 가능하다.

지표에 따라 최적 위치가 다르다: h_norm은 Step 0, logit_margin은 full average를 선호한다. 토큰 위치는 항상 명시적으로 보고되어야 한다.

### 6.2 본 프로토콜 하에서 보편적 최적 레이어의 부재

본 held-out 프로토콜(단일 70/30 분할, seed=42) 하에서, 최적 레이어는 조건 및 모델에 따라 실질적으로 달라진다:

**표 5. 보정 세트에서 선택된 최적 레이어 및 부호 (held-out 프로토콜)**

| 조건 | H_pre 레이어 | H_pre 부호 | logit_std 레이어 | logit_std 부호 | H_post 레이어 | H_post 부호 |
|:-----|:----------:|:--------:|:--------------:|:------------:|:-----------:|:---------:|
| Qwen Hard | L17 | − | L4 | + | L27 | − |
| Qwen MMLU | L18 | − | L18 | + | L27 | − |
| Llama MMLU | L0 | − | L13 | + | L10 | + |
| Mistral MMLU | L30 | − | L13 | + | L24 | − |

이 프로토콜 하에서 정답 판별을 위한 고정 최적 레이어는 나타나지 않는다. 최적 레이어는 지표, 모델, 과제에 의존한다. 이 선택이 단일 분할에 기반하며, 다중 독립 분할에 걸친 안정성은 검증하지 않았음을 유의한다(8.4절).

보충적으로, Cohen's d를 사용한 전체 샘플 레이어별 분석(부록 F, 그림 7)은 대체로 일관된 패턴을 보이나 일부 조건에서 다른 최적 레이어를 산출한다(예: Mistral H_pre 최적 = 이 파이프라인에서 L0 vs. held-out AUROC에서 L30). 이는 최적 레이어 결론이 지표 선택과 평가 프로토콜 모두에 민감함을 보여준다.

### 6.3 교차 모델 부호 반전

반복 분할 분석(20회 독립 70/30 분할, seed 0-19)에 따르면, H_pre 부호는 완전히 안정적이다: 모든 4개 조건과 20회 분할에서 낮은 H_pre가 정답과 연관된다(부호 = −, 20/20). 그러나 H_post는 교차 모델 부호 불안정성을 보인다: Llama MMLU에서 H_post 부호는 15/20 분할에서 +(높은 H_post = 정답)이며, 이는 모든 Qwen 및 Mistral 조건의 − 부호와 반전된다. 또한 h_norm은 Qwen MMLU에서 부호 = −1이 10/20 분할에서만 나타나(사실상 50/50), 우세한 부호 방향이 없는 유일한 지표-조건 쌍이다. 두 경우 모두 held-out 세트에서의 부호 보정을 필수로 한다.

### 6.4 다중 레이어 프로필

전 레이어 특징에 대한 로지스틱 회귀를 사용하면, 프로필이 단일 최적 레이어 스칼라를 일관되게 능가한다:

| 조건 | H_pre 단일 | H_pre 프로필 (LR) | 이득 |
|:-----|:----------:|:-----------------:|:----:|
| Qwen Hard | 0.7672 | 0.8467 | +0.080 |
| Qwen MMLU | 0.6367 | 0.7042 | +0.068 |
| Llama MMLU | 0.6021 | 0.7006 | +0.099 |
| Mistral MMLU | 0.6737 | 0.6955 | +0.022 |

다만 h_norm 프로필 (LR)이 4개 조건 중 3개에서 최강(표 2)이며, 스케일이 판별의 중요한 동인임과 일치한다.

---

## 7. 렌즈 의존성

### 7.1 Logit Lens는 Qwen 중간 레이어에서 비충실하다

본 논문에서 정의한 H_pre와 H_post는 모두 진단적 투영이다: 최종 레이어 디코더를 중간 표현에 적용한 것이다. 이 투영이 얼마나 충실한지 평가하기 위해, Tuned Lens (affine translator, wikitext-2에서 3 epochs 학습)를 훈련하고 비교한다:

| 레이어 | KL(TL) | KL(LL) | TL < LL? | Top-1 (TL) | Top-1 (LL) |
|:-----:|:------:|:------:|:--------:|:----------:|:----------:|
| 4 | 6.568 | 11.318 | Yes | 19.5% | 0.0% |
| 12 | 6.299 | 13.506 | Yes | 23.7% | 0.0% |
| 20 | 5.578 | 18.903 | Yes | 30.5% | 0.0% |
| 24 | 3.765 | 21.001 | Yes | 46.7% | 0.0% |

Qwen에서 Logit Lens는 중간 레이어에서 최종 분포와 0% top-1 일치를 보여, raw lm_head 투영을 통한 H_pre가 해당 레이어에서 모델이 "예측"할 것을 반영하지 않음을 의미한다. Tuned Lens는 충실도를 실질적으로 개선한다(27/28 레이어에서 KL(TL, final) < KL(LL, final); KL이 작을수록 충실). 추가로 Qwen MMLU에서 prompt-last (Step 0) hidden state를 사용한 판별력을 평가한다:

**표 7. 다른 디코더에서의 prompt-last 판별력 (Qwen MMLU, held-out test AUROC)**

| 디코더 | AUROC |
|:-------|:-----:|
| Logit Lens (H_pre) | 0.5325 |
| Tuned Lens | 0.5629 |
| H_post | 0.5574 |

참고: 표 7은 본문 baseline 평가와 동일한 1000개 MMLU 샘플(정답률 74.9%, 3.1절)을 사용하며, 부록 B의 별도 토큰 위치 샘플(정답률 70.8%)과는 다르다. H_pre AUROC(0.5325, layer 23)가 부록 B의 Step 0 H_pre(0.5911, layer 16)와 다른 이유는 두 실험이 다른 샘플 세트를 사용하여 다른 최적 레이어를 선택하기 때문이다.

Tuned Lens가 가장 높은 prompt-last 판별력을 달성하나, Logit Lens 대비 향상은 미미하다(+0.030). 이는 H_pre를 유용한 정답 판별 변수로서 무효화하지 않는다—예측적 유용성과 충실도는 독립적 속성이다. 그러나 H_pre가 주어진 레이어에서 모델이 "믿는 것"을 드러낸다고 해석해서는 안 된다. 유사한 충실도 패턴이 Llama 및 Mistral에서도 성립하는지는 미검증이다(8.4절).

### 7.2 분류기 선택이 판별력을 바꾼다

Entropy-Lens 설정(H_post 프로필 + k-NN, k=3)을 재현하고 동일 특징에 대해 로지스틱 회귀와 비교하면, 큰 분류기 효과가 드러난다:

| 방법 | MMLU Qwen | MMLU Llama | MMLU Mistral | Qwen Hard |
|:-----|:---------:|:----------:|:------------:|:---------:|
| EL-원본 (k-NN k=3) | 0.5474 | 0.5521 | 0.5748 | 0.6384 |
| EL-매칭 (H_post, LR) | 0.6412 | 0.6573 | 0.6388 | 0.8060 |
| H_pre 프로필 (LR) | **0.7042** | **0.7006** | 0.6955 | **0.8467** |

동일 H_post 특징에서 k-NN을 LR로 교체하면 AUROC가 +0.06~+0.17 증가한다. 어떤 특징이 가장 판별적인지에 대한 결론은 분류기 선택에 민감하다. 공통 분류기(LR) 하에서 H_pre 프로필은 네 조건 모두에서 H_post 프로필을 능가(+0.04~+0.06)하며, H_pre의 추가 스케일 정보와 일치한다.

---

## 8. 논의, 한계, 보고 권고

### 8.1 기존 연구에 대한 함의

본 연구의 발견은 logit-lens 투영에서 엔트로피를 계산하는 기존 연구에 직접적 영향을 미친다. 이 주장의 근거를 마련하기 위해, 표 8은 레이어별 엔트로피 및 내부 불확실성 신호에 관한 대표적 기존 연구를 감사하여, 정규화 지점 명시 여부와 스케일 baseline 보고 여부를 확인한다.

**표 8. 문헌 점검: 레이어별 엔트로피 / 내부 불확실성에 관한 기존 연구의 정규화 명시 여부**

패널 A는 중간 hidden state를 어휘 공간으로 직접 투영하는 연구(logit-lens 계열)를, 패널 B는 정규화 지점이 덜 직접적으로 관련되는 연구를 나열한다.

*패널 A. 중간 어휘 투영 (logit-lens 계열)*

| 참고문헌 | 렌즈 / 방법 | 정규화 지점 명시? | 스케일 baseline 보고? |
|:---------|:------------|:-----------------:|:---------------------:|
| nostalgebraist (2020) | Logit Lens | 원본 게시물에서 미명시 | 아니오 |
| Belrose et al. (2023) | Tuned Lens | 예 (명시적: LL(h) = LN[h] W_U) | 아니오 |
| Ali et al. (2025) | Entropy-Lens | 본문 미명시; 부록 코드는 최종 LN 적용 | 아니오 |
| Geva et al. (2022) | FF 어휘 투영 | 명시적 생략; 부록에서 LN 효과 확인 | 아니오 |
| Chuang et al. (2024) | DoLa | 명시적 기술 없음 | 아니오 |
| Ma et al. (2025) | SimLens | 명시적 기술 없음 | 아니오 |

*패널 B. 프로브 기반, 출력 수준, 기타 내부 방법*

| 참고문헌 | 렌즈 / 방법 | 정규화 지점 명시? | 스케일 baseline 보고? |
|:---------|:------------|:-----------------:|:---------------------:|
| Kossen et al. (2024) | SEPs | 해당 없음 (프로브 기반) | 아니오 |
| Bhatnagar et al. (2026) | DRIFT | 해당 없음 (프로브 기반) | 아니오 |
| Gao et al. (2025) | FLUE | 출력 수준 엔트로피 | 해당 없음 |
| Chen et al. (2025) | Internal Confidence | 생성 전 hidden states | 아니오 |
| Stolfo et al. (2024) | Confidence Regulation Neurons | Norm 기반 메커니즘 | 부분적 (norm이 신호) |

패널 A의 6편 중 중간 hidden state를 어휘 공간으로 직접 투영하는 연구에서 정규화 처리는 일관되지 않는다: Belrose et al. (2023)은 logit-lens 정의에 LayerNorm을 명시적으로 포함하고, Geva et al. (2022)은 이를 명시적으로 생략한 뒤 부록에서 효과를 확인하며, Ali et al. (2025)은 본문에서 미명시하나 부록 코드에서는 최종 LayerNorm을 적용하고, 나머지 셋은 정규화 지점을 전혀 명시하지 않는다. 이들 중 어느 연구도 엔트로피와 함께 직접적 스케일 baseline(logit_std, h_norm)을 보고하지 않았다. 패널 B의 연구는 프로브 기반 또는 출력 수준 방법을 사용하여 정규화 지점이 덜 직접적으로 관련되므로, 동일한 방식으로 교란이 적용된다고 주장하지 않는다. 이러한 논문 간 불일치는 개별 연구의 발견이 무효라는 의미가 아니라—디코딩된 중간 엔트로피에 기반한 연구에서 본 논문이 식별한 정규화 의존적 측정 선택이 일관되게 명시되거나 통제되지 않았음을 의미한다. 아래에서 세 가지 범주의 관련 연구를 논의한다.

**정규화 명시 없는 logit-lens 엔트로피.** Logit lens (nostalgebraist, 2020)는 중간 hidden state를 언어 모델 헤드를 통해 어휘 공간으로 투영한다. 연구자들이 이 투영에서 엔트로피를 계산할 때 hidden state가 먼저 정규화되었는지를 명시하지 않으면, 본 연구 결과에 따르면 결과 엔트로피는 hidden-state 스케일에 지배된다(4절). 구체적으로, RMSNorm 기반 모델에서 정규화 전 logit-lens 엔트로피는 암묵적 온도 조절(temperature-scaled) 측정으로 작용한다: hidden-state norm이 큰 레이어는 집중된 분포(낮은 엔트로피)를, norm이 작은 레이어는 거의 균일한 분포(높은 엔트로피)를 생성한다. 그러한 엔트로피로 식별된 "핵심 레이어"는 분포적 불확실성 전이가 아닌 norm 크기 전이를 반영할 수 있다. Logit-lens 투영에서 레이어별 엔트로피 패턴을 보고하는 기존 연구는 투영 전에 정규화가 적용되었는지를 명시해야 하며, 독자는 보고된 엔트로피 기울기가 부분적으로 스케일 기울기를 반영할 가능성을 고려해야 한다.

**Entropy-Lens 및 관련 프로파일링 방법.** Entropy-Lens (Ali et al., 2025)는 레이어에 걸친 엔트로피 프로필을 구축하여 의사결정 전략 분류에 사용한다. 본 연구 결과에 따르면 이러한 프로필의 판별 내용은 분류기에 의존한다(7.2절): 동일한 H_post 특징에서 k-NN을 로지스틱 회귀로 교체하면 AUROC가 +0.17까지 변한다. 나아가, 공통 분류기(LR) 하에서 H_pre 프로필이 네 조건 모두에서 H_post 프로필을 능가하며(+0.04~+0.06), 이는 H_pre의 추가 스케일 정보와 일치한다. 이는 분류기 선택을 통제하지 않는 엔트로피 프로필 비교가 특징 품질과 분류기 적합성을 혼동할 수 있음을 의미하며—효과적으로 보이는 프로필이 방향적 불확실성이 아닌 스케일 정보를 활용하고 있을 수 있다.

**내부 불확실성 추정 일반.** 중간 hidden state에서 불확실성 신호를 추출하는 방법—프로빙 기반 접근법(Kossen et al., 2024; Bhatnagar et al., 2026) 포함—은 자신의 신호가 hidden-state 스케일과 부분적으로 교란되는지를 고려해야 한다. 본 연구의 norm-binned 통제 분석(부록 C)은 구체적 진단법을 제공한다: h_norm 또는 logit_std의 동일 빈도 그룹 내에서 신호의 AUROC를 계산하고, 판별력이 유지되는지 확인한다. 향후 내부 불확실성 연구는 원시 성능과 함께 이러한 통제된 baseline을 보고할 것을 권고한다.

이 함의는 본 연구의 실험 범위—수학 추론 및 MMLU에서의 7-8B RMSNorm 기반 모델 3개—로 제한된다. LayerNorm 기반 아키텍처, 대형 모델, 개방형 생성 과제에서 유사한 교란이 발생하는지는 열린 경험적 질문이다. 동시기 연구인 Marín (2026)은 독립적으로 정규화된 logit-lens 투영과 원시(raw) logit-lens 투영을 구분하고, 최종 정규화 적용이 초기 레이어 디코딩을 안정화한다고 보고하였다; 본 연구는 이를 보완하여, 이 선택이 세 모델에 걸쳐 디코딩 엔트로피가 측정하는 대상을 어떻게 바꾸는지를 체계적으로 규명한다. 이를 기존 연구 결과의 확정적 무효화가 아닌 진단적 고려사항으로 제시한다.

### 8.2 본 논문이 주장하는 것

1. 정규화는 레이어별 디코딩 엔트로피가 측정하는 대상을 질적으로 바꾼다.
2. H_pre는 구조적으로 스케일 민감하며, H_post는 저-norm 초기 레이어를 제외하고 대체로 스케일 불변이다.
3. 단일 레이어 스칼라 baseline 중에서, 단순 스케일 프록시가 본 연구의 평가 환경에서 엔트로피와 동등하거나 정답 판별을 능가하는 경우가 빈번하다.
4. 내부 레이어 신호가 최종 출력 엔트로피를 일관되게 능가한다.
5. 본 held-out 프로토콜 하에서 결론은 레이어, 토큰 위치, 모델, 디코딩 렌즈에 강하게 의존한다.
6. 정규화 지점을 명시하지 않고 logit-lens 투영에서 레이어별 엔트로피를 계산하는 기존 연구는 분포적 불확실성이 아닌 스케일 통계를 혼동하고 있을 수 있다.

### 8.3 본 논문이 주장하지 않는 것

- 추론 레이어 식별이나 메커니즘적 추론 경로 추적을 주장하지 않는다.
- 스케일의 정답에 대한 인과적 매개를 주장하지 않는다.
- H_pre가 보편적 불확실성 추정기라고 주장하지 않는다.
- 모든 조건에서 self-consistency에 대한 우월성을 주장하지 않는다.
- 엔트로피가 무용하다고 주장하지 않는다—정규화 효과를 드러내는 해석 가능한 진단 도구로 남는다.
- RMSNorm 기반 디코더 LM을 넘어서 경험적으로 일반화하지 않는다. LayerNorm 기반 아키텍처에 유사한 고려사항이 적용되는지는 열린 경험적 질문이다.

### 8.4 한계

- **모델 범위**: 가장 심층적인 분석(스케일 개입, 토큰 위치, Tuned Lens)은 Qwen2.5-7B에 대해 수행되었다. Llama와 Mistral은 MMLU에서의 교차 모델 검증(sanity check)으로서 기능하며, 완전한 재현은 아니다.
- **분할 및 시드 강건성**: 모든 결과가 단일 70/30 계층화 분할(seed=42)을 사용한다. 부트스트랩 신뢰 구간이 delta-AUROC 불확실성 추정을 제공하지만, 다중 독립 분할에 걸친 레이어/부호 선택 안정성은 검증하지 않았다. 특히 6절의 "보편적 최적 레이어 부재" 및 "교차 모델 부호 반전" 관찰은 부분적으로 분할 특이적일 수 있으며, 이를 안정적 규칙성으로 확립하려면 반복 분할 분석이 필요하다.
- **진단적 투영 주의**: H_pre와 H_post 모두 최종 레이어 디코더를 중간 표현에 적용한다. 7.1절에서 보듯 이 투영은 중간 레이어에서 비충실하다. 엔트로피 값은 주어진 레이어에서 모델의 잠재적 예측을 반영하는 것으로 해석되어서는 안 된다.
- **렌즈 충실도 범위**: Tuned Lens 분석은 Qwen에 한정된다. 유사한 충실도 패턴이 Llama 및 Mistral에서도 성립하는지는 미검증이다.
- **H_post 초기 레이어 편차**: 저-norm 초기 레이어에서 H_post alpha-변동을 관찰한다. epsilon 정규화된 RMSNorm과 일치하지만, 유한 정밀도 효과와 epsilon 기여를 완전히 분리하지 못했다.
- **과제 범위**: 수학 추론 및 MMLU에서 평가한다. 개방형 생성, 장문 QA, 코드 생성으로 발견이 확장되는지는 열린 문제이다.
- **천장 효과**: Llama와 Mistral에서 H_pre가 초기~중간 레이어에서 거의 포화되어, 스케일 민감도의 관측 가능한 동적 범위를 제한한다. 구조적 발견을 무효화하지는 않으나 해당 영역에서의 실용적 유용성을 제한한다.
- **탐색적 분석 제외**: 일부 탐색적 인과 및 프로빙 분석이 수행되었으나, 신뢰할 수 있는 해석을 위해 방법론적으로 충분히 안정적이지 않아 본문에서 제외되었다.

### 8.5 보고 권고

본 연구의 발견에 기반하여, 레이어별 엔트로피에 관한 향후 연구에 다음 관행을 권고한다. 처음 세 가지는 본 논문에서 식별된 측정 교란을 다루며, 나머지 세 가지는 평가 과정에서 관찰된 방법론적 함정을 다룬다.

1. **항상 정규화 지점을 명시할 것.** 엔트로피가 최종 정규화 모듈 전후 중 어디에서 계산되는지 보고하고, 둘이 질적으로 다른 양을 측정함을 인식할 것. 이 명시 없이는 보고된 엔트로피 패턴이 분포적 불확실성을 반영하는지 hidden-state 스케일을 반영하는지 판단할 수 없다.
2. **직접 스케일 baseline을 보고할 것.** 엔트로피와 함께 최소한 logit_std와 h_norm을 포함할 것. 엔트로피가 이 간단한 baseline을 능가하지 못하면, 해당 신호는 스케일에 지배되고 있을 수 있다. Norm-binned 통제 진단(부록 C)은 구체적 검정을 제공한다: h_norm 또는 logit_std의 동일 빈도 그룹 내에서 엔트로피 AUROC를 계산하고 판별력이 유지되는지 확인할 것.
3. **보정 없이 엔트로피가 해석 가능하다고 가정하지 말 것.** 엔트로피-정답 관계의 부호가 모델 간에 반전될 수 있으며(6.3절), 최적 레이어는 조건에 따라 달라진다(6.2절). 고정 레이어나 부호 방향을 가정하지 말고, 항상 held-out 세트에서 보정할 것.
4. **토큰 위치를 명시할 것.** Hidden state가 prompt-last, first-generated, generation-average 중 어느 위치에서 추출되는지 보고할 것. 지표에 따라 최적 위치가 다르다(6.1절).
5. **디코더와 분류기 효과를 분리할 것.** 엔트로피 프로필을 비교할 때, 디코딩 렌즈와 하위 분류기가 통제되도록 할 것. 분류기 선택만으로도 AUROC가 +0.17까지 변할 수 있다(7.2절).
6. **정규화 구현 세부사항을 보고할 것.** epsilon 값과 gain 파라미터 포함 여부를 명시할 것—초기 레이어 행동에 영향을 미치므로(4.4절).

### 8.6 향후 연구

- pre/post 구분이 유사한 효과를 낳는지 검증하기 위한 LayerNorm 기반 아키텍처로의 분석 확장.
- 레이어 및 부호 선택 안정성을 위한 반복 분할 및 다중 시드 강건성 분석.
- 개방형 QA, 코드 생성, 장문 생성 과제에서의 평가.
- 매칭된 분류기 하에서 더 충실한 중간 디코더(Tuned Lens, SimLens 등) 탐구.

---

## 참고문헌

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

## 감사의 글

본 연구는 정보통신기획평가원(IITP) 지역 지능화 혁신 인재 양성 사업(IITP-2025-RS-2024-00436773)의 지원을 받았습니다. 본 연구는 한국산업기술진흥원(KIAT)이 산업통상자원부의 국제산업기술협력사업(과제번호: P0026190)으로 지원한 연구입니다.

---

## 부록 A. 추가 평가 조건

세 가지 보충 조건이 별도의 실험 파이프라인에서 평가되었으며, 본문 표 1-3의 주요 held-out 결과와 **직접 비교할 수 없다.** 참고용으로 이용 가능한 지표를 아래에 보고한다.

**Qwen Easy (competition_math Level 1-2, 500 샘플, 정답률 85.25%).** 이전 파이프라인(generation-average, 전체 샘플 Cohen's d)에서 평가. 해당 파이프라인의 held-out AUROC:

| 방법 | AUROC |
|:-----|:-----:|
| 출력 엔트로피 | 0.4818 |
| H_pre | 0.7295 |
| H_post | 0.5757 |
| h_norm | 0.6435 |
| Length-only | 0.6395 |

logit_std는 이 파이프라인에서 계산되지 않았다.

**Qwen ARC (ARC-Challenge, 999 샘플, 정답률 89.3%).** 별도 엔트로피 실험에서 전체 샘플 Cohen's d로 평가(unnormed 최적 레이어 L27, d=0.36). Held-out baseline 프로토콜에는 포함되지 않았다. 높은 정답률(89.3%)이 판별적 동적 범위를 제한한다.

**Llama Hard (competition_math Level 4-5, 500 샘플, 정답률 9.8%).** 전체 샘플 Cohen's d로 평가(normed 최적 L9, d=−0.42; unnormed 최적 L1, d=+0.86). 극단적 클래스 불균형(정답 9.8%)으로 held-out AUROC가 신뢰할 수 없어 주요 프로토콜에서 계산하지 않았다.

## 부록 B. 토큰 위치 소거

**중요**: 토큰 위치 소거 실험은 본문의 baseline 평가(5절)와 별도 실험으로 수행되었다. MMLU의 경우 다른 1000개 항목이 추출되어 70.8% 정답률(vs. 본문 74.9%). Hard의 경우 동일 500개 항목이지만 FP16 GPU 비결정성으로 54/499개 항목에서 다른 응답이 생성되어 53.2% 정답률(vs. 본문 baseline 54.3%). 70/30 분할도 독립적으로 적용되었다. 본 부록의 AUROC 값은 본문 표 1-3과 직접 비교 불가하며, 토큰 위치 간 *상대적* 패턴을 보이기 위해 보고한다.

**표 A2. 토큰 위치별 held-out test AUROC (Qwen, 70/30 분할, 별도 샘플)**

**Qwen Hard (n=500, 정답률=53.2%)**

| 지표 | Step 0 (prompt-last) | Step 1 (first-gen) | Full Avg | 최적 |
|:-----|:--------------------:|:------------------:|:--------:|:-----|
| H_pre | 0.7087 | **0.7486** | 0.7479 | Step 1 |
| H_post | **0.6893** | 0.6859 | 0.6032 | Step 0 |
| logit_std | 0.6963 | 0.6789 | **0.7514** | Full Avg |
| h_norm | **0.7638** | 0.6567 | 0.7336 | Step 0 |
| logit_max | 0.7141 | **0.7474** | 0.7468 | Step 1 |
| logit_margin | 0.6217 | 0.7326 | **0.7779** | Full Avg |
| Wh_norm | 0.7082 | 0.7179 | **0.7529** | Full Avg |

**Qwen MMLU (n=1000, 정답률=70.8%)**

| 지표 | Step 0 (prompt-last) | Step 1 (first-gen) | Full Avg | 최적 |
|:-----|:--------------------:|:------------------:|:--------:|:-----|
| H_pre | 0.5911 | **0.6192** | 0.5968 | Step 1 |
| H_post | 0.5424 | 0.5591 | **0.6235** | Full Avg |
| logit_std | 0.6052 | **0.6221** | 0.5958 | Step 1 |
| h_norm | 0.5581 | 0.5724 | **0.6127** | Full Avg |
| logit_max | **0.5996** | 0.5962 | 0.5812 | Step 0 |
| logit_margin | 0.5467 | 0.4794 | **0.5671** | Full Avg |
| Wh_norm | 0.5504 | **0.6128** | 0.5296 | Step 1 |

## 부록 C. Norm-Binned 통제

H_pre의 판별력이 단순히 hidden-state norm이나 logit 변동성의 프록시인지 검정하기 위해, 통제 변수 기준으로 샘플을 5개 동일 빈도 그룹으로 나누고, 빈 내 H_pre AUROC를 계산하여 원본 대비 유지율을 측정한다. 두 표 모두 원본 AUROC는 binning 전 H_pre의 최적 레이어 AUROC이다.

**표 A3. h_norm 통제 후 H_pre AUROC 유지율**

| 조건 | H_pre 원본 AUROC | 빈 내 H_pre AUROC (가중) | 유지율 |
|:-----|:----------------:|:------------------------:|:------:|
| Qwen Hard | 0.7672 | 0.6121 | 42.0% |
| Qwen MMLU | 0.6367 | 0.5888 | 65.0% |
| Llama MMLU | 0.6021 | 0.4564 | -42.7% |
| Mistral MMLU | 0.6737 | 0.6480 | 85.2% |

**표 A4. logit_std 통제 후 H_pre AUROC 유지율**

| 조건 | H_pre 원본 AUROC | 빈 내 H_pre AUROC (가중) | 유지율 |
|:-----|:----------------:|:------------------------:|:------:|
| Qwen Hard | 0.7672 | 0.4617 | -14.3% |
| Qwen MMLU | 0.6367 | 0.5576 | 42.1% |
| Llama MMLU | 0.6021 | 0.5347 | 34.0% |
| Mistral MMLU | 0.6737 | 0.6739 | 100.1% |

Mistral에서 logit_std 통제 후 H_pre AUROC가 100.1% 유지되는 것은 H_pre가 천장에 있을 때(4.5절) logit_std 통제가 H_pre의 신호를 감소시키지 않음과 일치한다—둘 다 동일한 (최소한의) 정보를 담기 때문이다. 반면 h_norm 통제 후 유지율은 85.2%로(표 A3), h_norm이 logit_std와 다른 부분의 H_pre 변동을 포착함을 보인다. 음수 유지율(Llama H_pre의 h_norm 통제, Qwen Hard의 logit_std 통제)은 빈 내 H_pre 성능이 chance 이하로 떨어짐을 나타내며, 해당 조건에서 통제 변수와 H_pre 간 강한 공선성을 시사한다.

## 부록 D. Tuned Lens 학습 세부사항

- **아키텍처**: 레이어별 affine translator (W_l, b_l)
- **학습 데이터**: wikitext-2 validation set
- **에포크**: 3
- **손실**: 최종 레이어 분포로부터의 KL divergence
- **결과**: 손실 7008 → 2459로 개선. 27/28 레이어에서 KL(TL, final) < KL(LL, final) (KL이 작을수록 충실).

## 부록 E. 교차 모델 Alpha-Sweep 표

**표 A5. Llama-3-8B alpha-sweep (MMLU 300, Step 0, seed=42)**

| Alpha | L0 H_pre | L0 H_post | L8 H_pre | L8 H_post | L16 H_pre | L16 H_post | L24 H_pre | L24 H_post |
|:-----:|:--------:|:---------:|:--------:|:---------:|:---------:|:----------:|:---------:|:----------:|
| 0.25 | 1.0000 | 0.9433 | 1.0000 | 0.8125 | 1.0000 | 0.7995 | 0.9997 | 0.0689 |
| 0.50 | 1.0000 | 0.8804 | 1.0000 | 0.8069 | 0.9999 | 0.7978 | 0.9987 | 0.0689 |
| 1.00 | 1.0000 | 0.8344 | 0.9999 | 0.8055 | 0.9995 | 0.7973 | 0.9947 | 0.0689 |
| 2.00 | 1.0000 | 0.8169 | 0.9995 | 0.8051 | 0.9980 | 0.7972 | 0.9770 | 0.0689 |
| 4.00 | 1.0000 | 0.8120 | 0.9978 | 0.8050 | 0.9920 | 0.7972 | 0.6077 | 0.0689 |

L24에서 H_post는 모든 alpha 값에서 정확히 불변(0.0689). L0 H_post는 0.13 변동(저-norm 영역, h_norm=0.89).

**표 A6. Mistral-7B alpha-sweep (MMLU 300, Step 0, seed=42)**

| Alpha | L0 H_pre | L0 H_post | L8 H_pre | L8 H_post | L16 H_pre | L16 H_post | L24 H_pre | L24 H_post |
|:-----:|:--------:|:---------:|:--------:|:---------:|:---------:|:----------:|:---------:|:----------:|
| 0.25 | 1.0000 | 0.9927 | 1.0000 | 0.9275 | 1.0000 | 0.8739 | 1.0000 | 0.5077 |
| 0.50 | 1.0000 | 0.9761 | 1.0000 | 0.9207 | 1.0000 | 0.8718 | 1.0000 | 0.5057 |
| 1.00 | 1.0000 | 0.9416 | 1.0000 | 0.9188 | 1.0000 | 0.8712 | 0.9999 | 0.5053 |
| 2.00 | 1.0000 | 0.9054 | 1.0000 | 0.9183 | 0.9999 | 0.8711 | 0.9994 | 0.5052 |
| 4.00 | 1.0000 | 0.8875 | 0.9999 | 0.9182 | 0.9995 | 0.8711 | 0.9987 | 0.5051 |

Mistral H_pre는 모든 레이어·alpha에서 ~1.0 유지(천장 효과). H_post L16 변동 0.0028(중간 레이어 불변성), L0 변동 0.1052(저-norm 초기 레이어 예외).

## 부록 F. 별도 전체 샘플 레이어별 파이프라인 (주요 Held-Out 결과와 직접 비교 불가)

**중요**: 표 A7-A8 및 그림 7은 본문(표 1-5)의 held-out 보정/평가 프로토콜을 사용하지 않고, 전체 샘플에서 모든 레이어의 unnormed/normed 엔트로피 AUROC 및 Cohen's d를 평가하는 별도 레이어별 분석 파이프라인의 결과이다. 주요 결과와 **직접 비교할 수 없다**. 특히 최적 레이어가 이 파이프라인과 held-out 프로토콜 간에 다를 수 있다(예: Mistral H_pre 최적 = 여기서 L0 vs. 표 5에서 L30).

**표 A7. MMLU 레이어별 엔트로피 AUROC (최적 레이어, 전체 샘플 평가)**

| 모델 | Unnormed 최적 | Unnormed AUROC | Normed 최적 | Normed AUROC |
|:-----|:------------:|:--------------:|:-----------:|:------------:|
| Qwen | L12 | 0.6292 | L27 | 0.6566 |
| Llama | L0 | 0.5958 | L10 | 0.5654 |
| Mistral | L0 | 0.6854 | L24 | 0.6394 |

**표 A8. MMLU 통계적 검증 및 Nested CV**

| 모델 | StatVal Unnormed | StatVal Normed | NestedCV Unnormed Optimism | NestedCV Normed Optimism |
|:-----|:----------------:|:--------------:|:--------------------------:|:------------------------:|
| Qwen | ROBUST | ROBUST | 0.0217 | -0.0003 |
| Llama | ROBUST | ROBUST | -0.0000 | 0.0260 |
| Mistral | ROBUST | ROBUST | 0.0035 | 0.0341 |

6개 조건(3모델 × 2지표) 전부 통계적 검증 통과 (Bonferroni 보정 유의성, 부트스트랩 CI 0.5 제외, 5-fold CV 안정성, 순열 검정).

## 부록 G. 과거 탐색적 분석

연구 과정에서 일부 탐색적 분석이 수행되었으나 본문에서 제외되었다:

- **프로빙 분석**: Hidden-state 표현에 대한 로지스틱 회귀 프로브가 중간 수준의 정확도를 보였으나, pooling 전략 및 정규화 선택에 의해 교란되었다.
- **인과/매개 분석**: 스케일 및 방향 기여의 연관적 분해가 수행되었으나, 인과적 해석이 본문에 포함할 만큼 충분히 근거하지 못했다.

투명성을 위해 여기에 문서화하며, 선별적 보고의 인상을 피하고자 한다.
