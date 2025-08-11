"""
Configuration for Affine Alignment based Draft Tree Search

이 모듈은 전체 시스템의 설정값들을 관리합니다:
- 모델 설정 (draft/target model paths)
- Tree search 파라미터 (beam width, depth)
- Affine alignment 설정
- Pruning thresholds
"""

from dataclasses import dataclass, field
from typing import Optional, List
import torch


@dataclass
class ModelConfig:
    """모델 관련 설정"""
    draft_model_name: str = "meta-llama/Llama-3.1-8B"  # Draft 모델
    target_model_name: str = "meta-llama/Llama-3.1-70B"  # Target 모델
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16
    max_length: int = 2048
    

@dataclass 
class TreeSearchConfig:
    """Tree Search 관련 설정"""
    max_candidates: int = 5  # 각 노드에서 생성할 최대 후보 수
    max_depth: int = 4  # Tree의 최대 깊이
    temperature: float = 0.7  # Draft 모델의 sampling temperature
    top_k: int = 50  # Top-k sampling
    top_p: float = 0.9  # Nucleus sampling
    max_paths_per_level: Optional[int] = 4  # 각 depth에서 유지할 최대 노드 수 (None이면 제한 없음)
    

@dataclass
class AffineAlignmentConfig:
    """Affine Alignment 설정"""
    hidden_size_draft: int = 4096  # Llama 3.1 8B hidden size
    hidden_size_target: int = 8192  # Llama 3.1 70B hidden size
    alignment_checkpoint: Optional[str] = None  # 학습된 alignment weights 경로
    use_bias: bool = True
    dropout_rate: float = 0.1
    

@dataclass
class PruningConfig:
    """Tree Pruning 관련 설정"""
    min_acceptance_prob: float = 0.1  # 최소 acceptance probability threshold
    adaptive_pruning: bool = True  # 적응적 pruning 사용 여부
    pruning_ratio: float = 0.5  # 각 레벨에서 pruning할 경로 비율
    top_k_paths: Optional[int] = 4  # 검증에 넘길 최대 경로 수 (None이면 제한 없음)
    

@dataclass
class MLPConfig:
    """Acceptance Probability Predictor MLP 설정"""
    hidden_dims: List[int] = field(default_factory=lambda: [2048, 1024, 512])
    activation: str = "gelu"
    dropout_rate: float = 0.1
    use_layer_norm: bool = True
    

@dataclass
class SpeculativeDecodingConfig:
    """전체 Speculative Decoding 시스템 설정"""
    model: ModelConfig = field(default_factory=ModelConfig)
    tree_search: TreeSearchConfig = field(default_factory=TreeSearchConfig)
    affine_alignment: AffineAlignmentConfig = field(default_factory=AffineAlignmentConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)
    
    # 추가 설정
    batch_size: int = 1
    seed: int = 42
    verbose: bool = True
    profile: bool = False  # 성능 프로파일링 여부 