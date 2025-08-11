"""
Affine Alignment Module for Hidden State Transformation

이 모듈은 Draft 모델의 hidden state를 Target 모델의 hidden state로 변환합니다.
핵심 수식: h_target = W * h_draft + b

주요 기능:
1. Draft → Target hidden state 변환
2. 학습된 가중치 로드/저장
3. 배치 처리 지원
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import os
from einops import rearrange


class AffineAlignment(nn.Module):
    """
    Draft 모델의 hidden state를 Target 모델의 hidden state로 변환하는 Affine 변환 모듈
    
    Args:
        hidden_size_draft: Draft 모델의 hidden dimension
        hidden_size_target: Target 모델의 hidden dimension  
        use_bias: bias term 사용 여부
        dropout_rate: dropout 비율
    """
    
    def __init__(
        self,
        hidden_size_draft: int,
        hidden_size_target: int,
        use_bias: bool = True,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size_draft = hidden_size_draft
        self.hidden_size_target = hidden_size_target
        
        # Affine transformation: h_target = W * h_draft + b
        self.weight = nn.Parameter(
            torch.randn(hidden_size_target, hidden_size_draft) * 0.02
        )
        
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size_target))
        else:
            self.register_parameter('bias', None)
            
        self.dropout = nn.Dropout(dropout_rate)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_size_target)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """가중치 초기화 - Xavier uniform 사용"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            
    def forward(
        self, 
        draft_hidden_states: torch.Tensor,
        return_raw: bool = False
    ) -> torch.Tensor:
        """
        Draft hidden states를 Target hidden states로 변환
        
        Args:
            draft_hidden_states: [batch_size, seq_len, hidden_size_draft] 또는
                                [batch_size, num_candidates, seq_len, hidden_size_draft]
            return_raw: True면 layer norm 적용 전 값 반환
            
        Returns:
            target_hidden_states: 변환된 hidden states
        """
        original_shape = draft_hidden_states.shape
        
        # Handle multi-candidate case
        if len(original_shape) == 4:
            batch_size, num_candidates, seq_len, _ = original_shape
            # Flatten batch and candidates dimensions
            draft_hidden_states = rearrange(
                draft_hidden_states, 'b c s h -> (b c) s h'
            )
        
        # Affine transformation
        target_hidden_states = F.linear(
            draft_hidden_states, self.weight, self.bias
        )
        
        # Dropout for regularization
        target_hidden_states = self.dropout(target_hidden_states)
        
        if return_raw:
            raw_states = target_hidden_states.clone()
        
        # Layer normalization
        target_hidden_states = self.layer_norm(target_hidden_states)
        
        # Restore original shape if needed
        if len(original_shape) == 4:
            target_hidden_states = rearrange(
                target_hidden_states, 
                '(b c) s h -> b c s h',
                b=batch_size, 
                c=num_candidates
            )
            if return_raw:
                raw_states = rearrange(
                    raw_states,
                    '(b c) s h -> b c s h', 
                    b=batch_size,
                    c=num_candidates
                )
        
        return (target_hidden_states, raw_states) if return_raw else target_hidden_states
    
    def compute_alignment_loss(
        self,
        draft_hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Alignment 학습을 위한 loss 계산
        
        Args:
            draft_hidden_states: Draft 모델의 hidden states
            target_hidden_states: 실제 Target 모델의 hidden states (ground truth)
            mask: attention mask
            
        Returns:
            loss: MSE loss
        """
        predicted_states = self.forward(draft_hidden_states)
        
        if mask is not None:
            # Apply mask
            mask = mask.unsqueeze(-1)  # Add hidden dimension
            predicted_states = predicted_states * mask
            target_hidden_states = target_hidden_states * mask
            
            # Compute MSE loss only on non-masked positions
            loss = F.mse_loss(predicted_states, target_hidden_states, reduction='sum')
            loss = loss / mask.sum()
        else:
            loss = F.mse_loss(predicted_states, target_hidden_states)
            
        return loss
    
    def save_weights(self, path: str):
        """학습된 가중치 저장"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state_dict = {
            'weight': self.weight.data,
            'bias': self.bias.data if self.bias is not None else None,
            'hidden_size_draft': self.hidden_size_draft,
            'hidden_size_target': self.hidden_size_target,
        }
        torch.save(state_dict, path)
        
    def load_weights(self, path: str):
        """학습된 가중치 로드"""
        state_dict = torch.load(path, map_location='cpu')
        
        # Verify dimensions
        assert state_dict['hidden_size_draft'] == self.hidden_size_draft
        assert state_dict['hidden_size_target'] == self.hidden_size_target
        
        self.weight.data = state_dict['weight']
        if self.bias is not None and state_dict['bias'] is not None:
            self.bias.data = state_dict['bias']
            
    @torch.no_grad()
    def analyze_alignment_quality(
        self,
        draft_hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor
    ) -> dict:
        """
        Alignment 품질 분석을 위한 메트릭 계산
        
        Returns:
            dict: 다양한 alignment 메트릭들
        """
        predicted_states = self.forward(draft_hidden_states)
        
        # MSE
        mse = F.mse_loss(predicted_states, target_hidden_states).item()
        
        # Cosine similarity
        cos_sim = F.cosine_similarity(
            predicted_states.view(-1, self.hidden_size_target),
            target_hidden_states.view(-1, self.hidden_size_target)
        ).mean().item()
        
        # L1 distance
        l1_dist = F.l1_loss(predicted_states, target_hidden_states).item()
        
        return {
            'mse': mse,
            'cosine_similarity': cos_sim,
            'l1_distance': l1_dist,
        } 