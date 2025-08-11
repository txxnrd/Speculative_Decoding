# Affine Alignment based Draft Tree Search for Speculative Decoding

이 프로젝트는 대규모 언어 모델의 추론 속도를 향상시키기 위한 혁신적인 speculative decoding 기법을 구현합니다.

## 📋 주요 특징

1. **Multi-candidate Tree Search**: Draft 모델이 트리 구조로 여러 토큰 경로를 생성
2. **Affine Alignment**: Draft hidden state를 Target hidden state로 효율적으로 변환
3. **Acceptance Probability Prediction**: MLP 기반으로 타겟 모델의 검증 통과 확률 예측
4. **Intelligent Tree Pruning**: 낮은 확률 경로를 사전에 제거하여 검증 오버헤드 최소화
5. **Adaptive Optimization**: 실시간 성능에 따라 pruning 전략 동적 조정

## 🚀 빠른 시작

### 설치

```bash
# Clone repository
git clone https://github.com/yourusername/speculative-decoding.git
cd speculative-decoding

# Install dependencies
pip install -r requirements.txt
```

### 기본 사용법

```python
from speculative_decoder import SpeculativeDecoder
from config import SpeculativeDecodingConfig

# Configuration
config = SpeculativeDecodingConfig()
config.model.draft_model_name = "meta-llama/Llama-3.1-8B"
config.model.target_model_name = "meta-llama/Llama-3.1-70B"

# Initialize decoder
decoder = SpeculativeDecoder(config)

# Generate text
prompt = "The future of AI is"
input_ids = decoder.tokenizer.encode(prompt, return_tensors="pt")
generated_ids, stats = decoder.generate(input_ids, max_new_tokens=100)

# Decode output
generated_text = decoder.tokenizer.decode(generated_ids[0])
print(generated_text)
```

## 🏗️ 아키텍처

### 핵심 구성 요소

1. **AffineAlignment** (`models/affine_alignment.py`)
   - Draft 모델의 hidden state를 Target 모델의 hidden state로 변환
   - 수식: `h_target = W * h_draft + b`
   - 학습된 가중치 저장/로드 지원

2. **DraftTreeSearch** (`models/draft_tree_search.py`)
   - Multi-candidate 토큰 생성
   - Tree 구조 관리 및 탐색
   - Top-k, Top-p sampling 지원

3. **AcceptanceProbabilityPredictor** (`models/acceptance_predictor.py`)
   - MLP 기반 확률 예측
   - Temperature calibration
   - Confidence 분석

4. **TreePruner** (`models/tree_pruner.py`)
   - Threshold 기반 pruning
   - Adaptive pruning 전략
   - Level-wise pruning 지원

## 📊 성능 벤치마킹

### 벤치마크 실행

```bash
cd tests
python benchmark.py
```

### 주요 메트릭

- **Acceptance Rate**: Draft 토큰이 Target 모델에 의해 수락되는 비율
- **Speedup**: 표준 생성 대비 속도 향상 배수
- **Pruning Effectiveness**: 제거된 경로 비율 vs 성능 영향
- **Latency Breakdown**: 각 컴포넌트별 시간 분석

## 🔧 고급 설정

### Tree Search 파라미터

```python
config.tree_search.max_candidates = 5  # 각 노드의 최대 후보 수
config.tree_search.max_depth = 4      # Tree 최대 깊이
config.tree_search.temperature = 0.8   # Sampling temperature
```

### Pruning 설정

```python
config.pruning.min_acceptance_prob = 0.1  # 최소 acceptance 확률
config.pruning.adaptive_pruning = True    # 적응적 pruning 사용
config.pruning.pruning_ratio = 0.5       # 목표 pruning 비율
```

### Affine Alignment

```python
# 사전 학습된 alignment 가중치 로드
config.affine_alignment.alignment_checkpoint = "path/to/checkpoint.pt"
```

## 📈 예상 성능 향상

- **Llama 3.1 8B → 70B**: 2-3x speedup (task dependent)
- **Acceptance Rate**: 40-60% (with proper alignment)
- **Memory Overhead**: ~10% additional for alignment/prediction modules

## 🤝 기여 방법

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🔗 참고 자료

- [Speculative Decoding Paper](https://arxiv.org/abs/2211.17192)
- [Multi-candidate Speculative Decoding](https://arxiv.org/abs/2401.06706)
- [Llama 3.1 Model Card](https://github.com/facebookresearch/llama)

## 📧 연락처

질문이나 제안사항이 있으시면 이슈를 열어주세요! 