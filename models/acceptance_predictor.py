"""
Acceptance Probability Predictor Module

이 모듈은 Affine 변환된 hidden state를 받아서 타겟 모델의 검증을 통과할 
확률을 예측합니다. 이를 통해 낮은 확률의 경로를 미리 제거할 수 있습니다.

주요 기능:
1. MLP 기반 확률 예측
2. Multi-layer architecture with dropout
3. Confidence calibration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import numpy as np


class AcceptanceProbabilityPredictor(nn.Module):
    """
    타겟 모델의 acceptance probability를 예측하는 MLP
    
    Args:
        input_dim: 입력 차원 (aligned hidden state dimension)
        hidden_dims: MLP의 hidden layer dimensions
        activation: 활성화 함수 종류
        dropout_rate: Dropout 비율
        use_layer_norm: Layer normalization 사용 여부
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [2048, 1024, 512],
        activation: str = "gelu",
        dropout_rate: float = 0.1,
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.use_layer_norm = use_layer_norm
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Layer normalization
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            # Activation
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
                
            prev_dim = hidden_dim
        
        # Final output layer (probability)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Output probability [0, 1]
        
        self.mlp = nn.Sequential(*layers)
        
        # Calibration parameters (learned during training)
        self.temperature = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(
        self, 
        aligned_hidden_states: torch.Tensor,
        return_logits: bool = False
    ) -> torch.Tensor:
        """
        Acceptance probability 예측
        
        Args:
            aligned_hidden_states: Affine 변환된 hidden states
                Shape: [batch_size, seq_len, hidden_dim] or
                       [batch_size, num_candidates, seq_len, hidden_dim]
            return_logits: True면 sigmoid 적용 전 logits도 반환
            
        Returns:
            probabilities: 예측된 acceptance probabilities
        """
        original_shape = aligned_hidden_states.shape
        
        # Handle different input shapes
        if len(original_shape) == 4:
            # Multi-candidate case
            batch_size, num_candidates, seq_len, hidden_dim = original_shape
            # Reshape to [batch_size * num_candidates * seq_len, hidden_dim]
            aligned_hidden_states = aligned_hidden_states.reshape(-1, hidden_dim)
        elif len(original_shape) == 3:
            # Single sequence case
            batch_size, seq_len, hidden_dim = original_shape
            aligned_hidden_states = aligned_hidden_states.reshape(-1, hidden_dim)
        else:
            # Already flattened
            pass
        
        # Forward through MLP
        raw_probs = self.mlp[:-1](aligned_hidden_states)  # Before sigmoid
        
        # Temperature scaling for calibration
        calibrated_logits = (raw_probs - self.bias) / self.temperature
        probabilities = torch.sigmoid(calibrated_logits)
        
        # Reshape back to original shape (except last dimension)
        if len(original_shape) == 4:
            probabilities = probabilities.reshape(batch_size, num_candidates, seq_len)
            if return_logits:
                calibrated_logits = calibrated_logits.reshape(batch_size, num_candidates, seq_len)
        elif len(original_shape) == 3:
            probabilities = probabilities.reshape(batch_size, seq_len)
            if return_logits:
                calibrated_logits = calibrated_logits.reshape(batch_size, seq_len)
                
        if return_logits:
            return probabilities, calibrated_logits
        else:
            return probabilities
    
    def compute_loss(
        self,
        aligned_hidden_states: torch.Tensor,
        target_accepted: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        class_weights: Optional[Tuple[float, float]] = (1.0, 1.0)
    ) -> torch.Tensor:
        """
        학습을 위한 loss 계산
        
        Args:
            aligned_hidden_states: Affine 변환된 hidden states
            target_accepted: 실제 acceptance 여부 (0 or 1)
            mask: attention mask
            class_weights: (weight_for_rejected, weight_for_accepted)
            
        Returns:
            loss: Binary cross entropy loss
        """
        probabilities, logits = self.forward(aligned_hidden_states, return_logits=True)
        
        # Flatten for loss computation
        probabilities = probabilities.reshape(-1)
        logits = logits.reshape(-1)
        target_accepted = target_accepted.reshape(-1).float()
        
        if mask is not None:
            mask = mask.reshape(-1)
            # Apply mask
            probabilities = probabilities[mask]
            logits = logits[mask]
            target_accepted = target_accepted[mask]
        
        # Weighted BCE loss
        pos_weight = torch.tensor([class_weights[1] / class_weights[0]], device=logits.device)
        loss = F.binary_cross_entropy_with_logits(
            logits, target_accepted, pos_weight=pos_weight
        )
        
        return loss
    
    @torch.no_grad()
    def evaluate_calibration(
        self,
        aligned_hidden_states: torch.Tensor,
        target_accepted: torch.Tensor,
        num_bins: int = 10
    ) -> dict:
        """
        모델의 calibration 평가 (reliability diagram을 위한 데이터)
        
        Returns:
            dict: calibration 메트릭들
        """
        probabilities = self.forward(aligned_hidden_states).reshape(-1)
        target_accepted = target_accepted.reshape(-1).float()
        
        # Compute calibration error
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        accuracies = []
        confidences = []
        counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.float().mean().item()
            
            if prop_in_bin > 0:
                accuracy_in_bin = target_accepted[in_bin].float().mean().item()
                avg_confidence_in_bin = probabilities[in_bin].mean().item()
                count_in_bin = in_bin.sum().item()
                
                accuracies.append(accuracy_in_bin)
                confidences.append(avg_confidence_in_bin)
                counts.append(count_in_bin)
            else:
                accuracies.append(0)
                confidences.append((bin_lower + bin_upper) / 2)
                counts.append(0)
        
        # Expected Calibration Error (ECE)
        ece = 0
        total_count = sum(counts)
        for acc, conf, count in zip(accuracies, confidences, counts):
            if total_count > 0:
                ece += (count / total_count) * abs(acc - conf)
        
        return {
            'ece': ece,
            'accuracies': accuracies,
            'confidences': confidences,
            'counts': counts,
            'total_samples': total_count
        }
    
    def predict_path_probabilities(
        self,
        tree_paths: List,  # List[TreePath] from draft_tree_search
        affine_alignment: nn.Module
    ) -> List[float]:
        """
        Tree paths의 acceptance probability 예측
        
        Args:
            tree_paths: DraftTreeSearch에서 생성된 경로들
            affine_alignment: AffineAlignment 모듈
            
        Returns:
            probabilities: 각 경로의 평균 acceptance probability
        """
        path_probs = []
        
        for path in tree_paths:
            if path.hidden_states is not None:
                # Apply affine alignment
                aligned_states = affine_alignment(path.hidden_states.unsqueeze(0))
                
                # Predict acceptance probabilities
                probs = self.forward(aligned_states).squeeze(0)
                
                # Average probability across the sequence
                avg_prob = probs.mean().item()
                path.acceptance_prob = avg_prob
                path_probs.append(avg_prob)
            else:
                path_probs.append(0.0)
                
        return path_probs 