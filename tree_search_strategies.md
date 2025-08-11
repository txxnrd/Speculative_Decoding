# Tree Search Expansion Strategies

## Overview
Tree-based speculative decoding에서 각 노드를 확장하는 전략은 성능과 품질에 큰 영향을 미칩니다.

## 1. Top-K Expansion (`top_k`)
- **설명**: 각 위치에서 확률이 가장 높은 K개 토큰을 선택
- **장점**: 간단하고 빠름, 품질이 보장됨
- **단점**: 다양성이 부족할 수 있음
- **사용 예시**:
  ```bash
  python test_tree_search.py --tree-strategy top_k --tree-top-k 10 --beam-width 3
  ```

## 2. Nucleus Sampling (`top_p`)
- **설명**: 누적 확률이 p를 초과할 때까지 토큰을 선택
- **장점**: 동적인 후보 개수, 더 자연스러운 텍스트
- **단점**: 때로는 품질이 낮은 토큰도 포함될 수 있음
- **사용 예시**:
  ```bash
  python test_tree_search.py --tree-strategy top_p --tree-top-p 0.9 --beam-width 5
  ```

## 3. Beam Search (`beam`)
- **설명**: 각 단계에서 전체 경로의 확률이 가장 높은 K개 유지
- **장점**: 최적 경로를 찾을 가능성이 높음
- **단점**: 결정론적이어서 다양성 부족
- **사용 예시**:
  ```bash
  python test_tree_search.py --tree-strategy beam --beam-width 3
  ```

## 4. Diverse Beam Search (`diverse_beam`)
- **설명**: 선택된 토큰과 유사한 토큰에 페널티를 부여하여 다양성 확보
- **장점**: 다양한 경로 탐색, 더 창의적인 결과
- **단점**: 계산 복잡도가 높음
- **사용 예시**:
  ```bash
  python test_tree_search.py --tree-strategy diverse_beam --beam-width 3 --tree-diversity 0.5
  ```

## 고급 설정

### Temperature 조절
트리 확장 시 temperature를 조절하여 확률 분포를 조정:
```bash
# 낮은 temperature (더 확실한 선택)
python test_tree_search.py --tree-temperature 0.7

# 높은 temperature (더 다양한 선택)
python test_tree_search.py --tree-temperature 1.3
```

### 혼합 전략
실제 구현에서는 depth에 따라 다른 전략을 사용할 수도 있습니다:
- 초반 (depth 0-2): `diverse_beam`으로 다양성 확보
- 중반 (depth 3-4): `top_p`로 자연스러운 확장
- 후반 (depth 5+): `top_k`로 안정적인 마무리

## 성능 고려사항

1. **Acceptance Rate vs Diversity Trade-off**:
   - 다양한 경로 → 더 높은 acceptance rate 가능성
   - 너무 많은 경로 → 계산 비용 증가

2. **Affine Pruning과의 상호작용**:
   - 다양한 경로를 생성하되, Affine verifier로 효율적으로 가지치기
   - Pruning threshold를 전략에 맞게 조정

3. **최적 설정 찾기**:
   - 작업 유형에 따라 다름 (코드 생성 vs 창의적 글쓰기)
   - A/B 테스트로 최적 조합 찾기 