"""
Speculative Decoding Models Package

이 패키지는 speculative decoding에 필요한 모든 모델 구성 요소를 포함합니다:
- AffineAlignment: Draft → Target hidden state 변환
- DraftTreeSearch: Multi-candidate tree generation
- AcceptanceProbabilityPredictor: MLP 기반 acceptance probability 예측
- TreePruner: Tree pruning based on predicted probabilities
- TreeMaskModelWrapper: 4D attention mask를 지원하는 모델 래퍼
"""

from .affine_alignment import AffineAlignment
from .draft_tree_search import DraftTreeSearch, TreeNode, TreePath
from .acceptance_predictor import AcceptanceProbabilityPredictor
from .tree_pruner import TreePruner, PruningStatistics
from .tree_mask_wrapper import TreeMaskModelWrapper

__all__ = [
    'AffineAlignment',
    'DraftTreeSearch',
    'TreeNode',
    'TreePath',
    'AcceptanceProbabilityPredictor',
    'TreePruner',
    'PruningStatistics',
    'TreeMaskModelWrapper'
] 