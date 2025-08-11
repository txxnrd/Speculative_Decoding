"""
Tree Pruning Module

이 모듈은 예측된 acceptance probability를 기반으로 tree에서 낮은 확률의 
경로들을 제거합니다. 이를 통해 타겟 모델의 검증 오버헤드를 줄입니다.

주요 기능:
1. Threshold-based pruning
2. Adaptive pruning based on statistics
3. Top-k path selection
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import heapq


@dataclass 
class PruningStatistics:
    """Pruning 과정의 통계 정보"""
    original_paths: int
    pruned_paths: int  
    pruning_ratio: float
    min_prob_kept: float
    max_prob_pruned: float
    avg_prob_kept: float
    avg_prob_pruned: float


class TreePruner:
    """
    Tree pruning을 수행하는 클래스
    
    Args:
        min_acceptance_prob: 최소 acceptance probability threshold
        adaptive_pruning: 적응적 pruning 사용 여부
        pruning_ratio: 각 레벨에서 pruning할 경로 비율
        top_k_paths: 유지할 최대 경로 수
    """
    
    def __init__(
        self,
        min_acceptance_prob: float = 0.1,
        adaptive_pruning: bool = True,
        pruning_ratio: float = 0.5,
        top_k_paths: Optional[int] = None
    ):
        self.min_acceptance_prob = min_acceptance_prob
        self.adaptive_pruning = adaptive_pruning
        self.pruning_ratio = pruning_ratio
        self.top_k_paths = top_k_paths
        
        # Adaptive threshold history
        self.threshold_history = []
        self.performance_history = []
        
    def prune_paths(
        self,
        tree_paths: List,  # List[TreePath] from draft_tree_search
        acceptance_probs: Optional[List[float]] = None,
        return_stats: bool = True
    ) -> Tuple[List, Optional[PruningStatistics]]:
        """
        Tree paths를 pruning
        
        Args:
            tree_paths: 생성된 tree paths
            acceptance_probs: 각 경로의 acceptance probability (없으면 path 내부 값 사용)
            return_stats: 통계 정보 반환 여부
            
        Returns:
            pruned_paths: Pruning된 경로들
            stats: Pruning 통계 (optional)
        """
        if not tree_paths:
            return [], None
            
        # Get acceptance probabilities
        if acceptance_probs is None:
            acceptance_probs = [
                path.acceptance_prob if path.acceptance_prob is not None else 0.0
                for path in tree_paths
            ]
            
        # Ensure we have probabilities for all paths
        assert len(acceptance_probs) == len(tree_paths)
        
        # Determine pruning threshold
        if self.adaptive_pruning:
            threshold = self._compute_adaptive_threshold(acceptance_probs)
        else:
            threshold = self.min_acceptance_prob
            
        # Apply different pruning strategies
        if self.top_k_paths is not None:
            # Top-k selection
            pruned_paths, pruned_indices = self._top_k_pruning(
                tree_paths, acceptance_probs, self.top_k_paths
            )
        else:
            # Threshold-based pruning with ratio constraint
            pruned_paths, pruned_indices = self._threshold_pruning(
                tree_paths, acceptance_probs, threshold
            )
            
        # Compute statistics
        stats = None
        if return_stats:
            stats = self._compute_pruning_stats(
                tree_paths, pruned_paths, acceptance_probs, pruned_indices
            )
            
        return pruned_paths, stats
    
    def _compute_adaptive_threshold(self, acceptance_probs: List[float]) -> float:
        """
        적응적으로 pruning threshold 계산
        
        통계 정보를 기반으로 동적으로 threshold를 조정합니다.
        """
        probs = np.array(acceptance_probs)
        
        # Basic statistics
        mean_prob = np.mean(probs)
        std_prob = np.std(probs)
        median_prob = np.median(probs)
        
        # Compute percentile-based threshold
        percentile_threshold = np.percentile(probs, (1 - self.pruning_ratio) * 100)
        
        # Combine different criteria
        if len(self.threshold_history) > 0:
            # Use historical performance to adjust
            avg_historical = np.mean(self.threshold_history[-10:])
            threshold = 0.7 * percentile_threshold + 0.3 * avg_historical
        else:
            threshold = percentile_threshold
            
        # Ensure minimum threshold
        threshold = max(threshold, self.min_acceptance_prob)
        
        # Update history
        self.threshold_history.append(threshold)
        
        return threshold
    
    def _threshold_pruning(
        self,
        tree_paths: List,
        acceptance_probs: List[float],
        threshold: float
    ) -> Tuple[List, List[int]]:
        """
        Threshold 기반 pruning
        """
        pruned_paths = []
        pruned_indices = []
        
        # First pass: collect paths above threshold
        candidates = []
        for i, (path, prob) in enumerate(zip(tree_paths, acceptance_probs)):
            if prob >= threshold:
                candidates.append((i, path, prob))
                
        # If too many paths remain, apply ratio constraint
        if len(candidates) > len(tree_paths) * (1 - self.pruning_ratio):
            # Sort by probability and keep top paths
            candidates.sort(key=lambda x: x[2], reverse=True)
            max_keep = int(len(tree_paths) * (1 - self.pruning_ratio))
            candidates = candidates[:max_keep]
            
        # Extract pruned paths and indices
        for i, path, _ in candidates:
            pruned_paths.append(path)
            pruned_indices.append(i)
            
        return pruned_paths, pruned_indices
    
    def _top_k_pruning(
        self,
        tree_paths: List,
        acceptance_probs: List[float],
        k: int
    ) -> Tuple[List, List[int]]:
        """
        Top-k paths 선택
        """
        # Use heap to efficiently find top-k
        if len(tree_paths) <= k:
            return tree_paths, list(range(len(tree_paths)))
            
        # Create (prob, index, path) tuples
        items = [(prob, i, path) for i, (path, prob) in enumerate(zip(tree_paths, acceptance_probs))]
        
        # Get top-k using heap
        top_k_items = heapq.nlargest(k, items, key=lambda x: x[0])
        
        # Sort by original index to maintain order
        top_k_items.sort(key=lambda x: x[1])
        
        pruned_paths = [item[2] for item in top_k_items]
        pruned_indices = [item[1] for item in top_k_items]
        
        return pruned_paths, pruned_indices
    
    def _compute_pruning_stats(
        self,
        original_paths: List,
        pruned_paths: List,
        acceptance_probs: List[float],
        kept_indices: List[int]
    ) -> PruningStatistics:
        """
        Pruning 통계 계산
        """
        pruned_indices = set(range(len(original_paths))) - set(kept_indices)
        
        kept_probs = [acceptance_probs[i] for i in kept_indices]
        pruned_probs = [acceptance_probs[i] for i in pruned_indices]
        
        stats = PruningStatistics(
            original_paths=len(original_paths),
            pruned_paths=len(pruned_paths),
            pruning_ratio=1 - len(pruned_paths) / len(original_paths),
            min_prob_kept=min(kept_probs) if kept_probs else 0.0,
            max_prob_pruned=max(pruned_probs) if pruned_probs else 0.0,
            avg_prob_kept=np.mean(kept_probs) if kept_probs else 0.0,
            avg_prob_pruned=np.mean(pruned_probs) if pruned_probs else 0.0
        )
        
        return stats
    
    def update_performance(self, acceptance_rate: float):
        """
        실제 acceptance rate를 기반으로 pruning 성능 업데이트
        
        이 정보는 adaptive threshold 계산에 사용됩니다.
        """
        self.performance_history.append(acceptance_rate)
        
        # Adjust parameters based on performance
        if len(self.performance_history) >= 10:
            recent_performance = np.mean(self.performance_history[-10:])
            
            # If acceptance rate is too low, be less aggressive
            if recent_performance < 0.3:
                self.pruning_ratio = max(0.3, self.pruning_ratio * 0.9)
            # If acceptance rate is high, can be more aggressive  
            elif recent_performance > 0.7:
                self.pruning_ratio = min(0.7, self.pruning_ratio * 1.1)
                
    def prune_tree_level_wise(
        self,
        tree_paths: List,
        acceptance_predictor,
        affine_alignment,
        max_paths_per_level: Optional[Dict[int, int]] = None
    ) -> List:
        """
        Level-wise pruning: 각 depth에서 별도로 pruning 수행
        
        Args:
            tree_paths: 모든 tree paths
            acceptance_predictor: AcceptanceProbabilityPredictor 인스턴스
            affine_alignment: AffineAlignment 인스턴스  
            max_paths_per_level: 각 레벨별 최대 경로 수
            
        Returns:
            pruned_paths: Level-wise pruning된 경로들
        """
        # Group paths by depth
        paths_by_depth = {}
        for path in tree_paths:
            depth = len(path.token_ids)
            if depth not in paths_by_depth:
                paths_by_depth[depth] = []
            paths_by_depth[depth].append(path)
            
        # Predict probabilities if not already done
        for paths in paths_by_depth.values():
            for path in paths:
                if path.acceptance_prob is None and path.hidden_states is not None:
                    aligned_states = affine_alignment(path.hidden_states.unsqueeze(0))
                    probs = acceptance_predictor(aligned_states).squeeze(0)
                    path.acceptance_prob = probs.mean().item()
                    
        # Prune each level
        all_pruned_paths = []
        for depth, paths in sorted(paths_by_depth.items()):
            # Determine max paths for this level
            if max_paths_per_level and depth in max_paths_per_level:
                k = max_paths_per_level[depth]
                level_pruner = TreePruner(
                    min_acceptance_prob=self.min_acceptance_prob,
                    top_k_paths=k
                )
            else:
                level_pruner = self
                
            pruned_paths, _ = level_pruner.prune_paths(paths, return_stats=False)
            all_pruned_paths.extend(pruned_paths)
            
        return all_pruned_paths 