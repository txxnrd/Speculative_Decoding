"""
Draft Tree Search Module

이 모듈은 Draft 모델을 사용하여 tree 구조로 여러 토큰 경로를 생성합니다.
각 노드에서 여러 후보를 생성하여 다양한 가능성을 탐색합니다.

주요 기능:
1. Multi-candidate token generation
2. Tree structure management 
3. Hidden state extraction for alignment
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from collections import defaultdict


class TreeNode:
    """
    Tree의 각 노드를 표현하는 클래스
    
    Attributes:
        token_id: 현재 노드의 토큰 ID
        hidden_state: 현재 노드에서의 hidden state
        logit: 이 토큰이 선택될 때의 logit 값
        parent: 부모 노드
        children: 자식 노드들
        depth: 트리에서의 깊이
        cumulative_score: 루트부터 현재 노드까지의 누적 점수
        topk_tokens: (프로파일링) 이 위치에서의 draft 상위 K 토큰
        topk_probs: (프로파일링) 상위 K 토큰 확률
    """
    
    def __init__(
        self,
        token_id: int,
        hidden_state: Optional[torch.Tensor] = None,
        logit: Optional[float] = None,
        parent: Optional['TreeNode'] = None,
        depth: int = 0,
        topk_tokens: Optional[List[int]] = None,
        topk_probs: Optional[List[float]] = None,
    ):
        self.token_id = token_id
        self.hidden_state = hidden_state
        self.logit = logit
        self.parent = parent
        self.children: List['TreeNode'] = []
        self.depth = depth
        self.cumulative_score = 0.0
        self.topk_tokens = topk_tokens
        self.topk_probs = topk_probs
        
        if parent is not None:
            parent.children.append(self)
            if logit is not None:
                self.cumulative_score = parent.cumulative_score + logit
                
    def get_path_to_root(self) -> List['TreeNode']:
        """루트까지의 경로를 반환"""
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))
    
    def get_token_sequence(self) -> List[int]:
        """루트부터 현재 노드까지의 토큰 시퀀스 반환"""
        path = self.get_path_to_root()
        return [node.token_id for node in path if node.token_id is not None]


@dataclass
class TreePath:
    """Tree에서의 경로를 표현하는 데이터 클래스"""
    nodes: List[TreeNode]
    token_ids: List[int]
    hidden_states: torch.Tensor  # [seq_len, hidden_size]
    cumulative_score: float
    acceptance_prob: Optional[float] = None  # Pruning을 위한 예측 확률
    # Diagnostics (optional)
    draft_topk_tokens: Optional[List[List[int]]] = None
    draft_topk_probs: Optional[List[List[float]]] = None


class DraftTreeSearch:
    """
    Draft 모델을 사용한 Tree Search 구현
    
    Args:
        draft_model: Draft 언어 모델
        tokenizer: 토크나이저
        max_candidates: 각 노드에서 생성할 최대 후보 수
        max_depth: Tree의 최대 깊이
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
    """
    
    def __init__(
        self,
        draft_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        max_candidates: int = 5,
        max_depth: int = 4,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        device: str = "cuda",
        do_sample: bool = True,
        max_paths_per_level: Optional[int] = None,
    ):
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.max_candidates = max_candidates
        self.max_depth = max_depth
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.device = device
        self.do_sample = do_sample
        self.max_paths_per_level = max_paths_per_level
        
        # Model is already loaded with device_map="auto" for multi-GPU
        self.draft_model.eval()
        
    def _get_model_device(self):
        """Get the primary device of the model"""
        try:
            return next(iter(self.draft_model.parameters())).device
        except:
            return torch.device(self.device)
        
    def _sample_top_k_top_p(
        self, 
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Top-k, Top-p (nucleus) sampling 또는 결정적 상위 후보 선택
        
        Returns:
            sampled_indices: 샘플링/선택된 토큰 인덱스들
            sampled_logits: 해당 토큰들의 logit 값
        """
        # Apply temperature
        logits = logits / temperature
        
        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.clone()
            logits[indices_to_remove] = float('-inf')
            
        # Top-p filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.clone()
            logits[indices_to_remove] = float('-inf')
            
        # Determine number of candidates
        probs = F.softmax(logits, dim=-1)
        num_available = (probs > 0).sum().item()
        num_samples = min(self.max_candidates, num_available) if num_available > 0 else 0
        
        if num_samples == 0:
            sampled_indices = torch.argmax(logits, dim=-1, keepdim=True)
            sampled_logits = logits.gather(-1, sampled_indices)
            return sampled_indices, sampled_logits
        
        if self.do_sample:
            sampled_indices = torch.multinomial(probs, num_samples, replacement=False)
            sampled_logits = logits.gather(-1, sampled_indices)
        else:
            # FIXED: For greedy decoding, select based on probabilities, not raw logits
            print("[DEBUG] USING FIXED GREEDY SELECTION CODE")
            probs_for_selection = F.softmax(logits, dim=-1)
            topk_probs = torch.topk(probs_for_selection, k=num_samples, dim=-1)
            sampled_indices = topk_probs.indices
            sampled_logits = logits.gather(-1, sampled_indices)
            print(f"[DEBUG] Selected indices from probs: {sampled_indices}")
            
            # Debug: print what we're selecting
            if hasattr(self, '_debug_step') and self._debug_step < 3:
                print(f"[DEBUG DraftTreeSearch] Selecting top-{num_samples} tokens:")
                print(f"  sampled_indices: {sampled_indices}")
                print(f"  sampled_logits: {sampled_logits}")
        
        return sampled_indices, sampled_logits
    
    @torch.no_grad()
    def generate_tree(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = True,
        collect_diagnostics: bool = False,
    ) -> List[TreePath]:
        """
        Tree 구조로 draft 토큰들을 생성
        
        Args:
            input_ids: 입력 토큰 시퀀스 [batch_size, seq_len]
            attention_mask: 어텐션 마스크
            return_hidden_states: hidden states 반환 여부
            
        Returns:
            tree_paths: 생성된 모든 경로들의 리스트
        """
        batch_size = input_ids.shape[0]
        assert batch_size == 1, "Currently only batch_size=1 is supported"
        
        # Ensure input is on the right device
        model_device = self._get_model_device()
        input_ids = input_ids.to(model_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model_device)
        
        # Initialize root node
        root = TreeNode(token_id=None, depth=-1)
        
        # Current frontier (nodes to expand)
        frontier = [root]
        all_paths = []
        
        # Initial model forward pass
        outputs = self.draft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=return_hidden_states,
            use_cache=True
        )
        
        past_key_values = outputs.past_key_values
        initial_hidden_state = outputs.hidden_states[-1][:, -1, :] if return_hidden_states else None
        
        # Store initial past_key_values for reuse
        base_past_key_values = past_key_values
        
        # Tree expansion
        for depth in range(self.max_depth):
            next_frontier = []
            
            # Optional: cap frontier size per level
            if hasattr(self, 'max_paths_per_level') and self.max_paths_per_level:
                if len(frontier) > self.max_paths_per_level:
                    # Keep highest score nodes
                    frontier = sorted(frontier, key=lambda n: n.cumulative_score if n.cumulative_score is not None else 0.0, reverse=True)[: self.max_paths_per_level]
            
            for parent_node in frontier:
                # Get the current sequence
                if parent_node.parent is None:  # Root node
                    current_input_ids = input_ids
                    current_past_key_values = past_key_values
                    current_position = input_ids.shape[1]
                else:
                    # Get tokens from root to current node
                    token_sequence = parent_node.get_token_sequence()
                    current_input_ids = torch.tensor(
                        [token_sequence[-1]], 
                        device=model_device
                    ).unsqueeze(0)
                    current_past_key_values = parent_node.past_key_values
                    current_position = input_ids.shape[1] + len(token_sequence) - 1
                
                # Generate candidates
                outputs = self.draft_model(
                    input_ids=current_input_ids,
                    past_key_values=current_past_key_values,
                    output_hidden_states=return_hidden_states,
                    use_cache=True
                )
                
                logits = outputs.logits[0, -1, :]  # [vocab_size]
                hidden_state = outputs.hidden_states[-1][0, -1, :] if return_hidden_states else None
                
                # Sample multiple candidates
                sampled_indices, sampled_logits = self._sample_top_k_top_p(
                    logits, self.temperature, self.top_k, self.top_p
                )
                
                # Diagnostics: compute draft top-k at this position if requested
                diag_topk_tokens = None
                diag_topk_probs = None
                if collect_diagnostics:
                    k_diag = 10
                    topk_vals, topk_idx = torch.topk(F.softmax(logits, dim=-1), k=min(k_diag, logits.shape[-1]))
                    diag_topk_tokens = topk_idx.tolist()
                    diag_topk_probs = topk_vals.tolist()
                
                # Debug: print what we're creating
                if depth == 0 and not hasattr(self, '_printed_debug'):
                    self._printed_debug = True
                    print(f"\n[DEBUG] First level token selection:")
                    print(f"  Top-5 from logits: {torch.topk(F.softmax(logits, dim=-1), 5)}")
                    print(f"  sampled_indices: {sampled_indices}")
                    print(f"  sampled_logits: {sampled_logits}")
                
                # Create child nodes
                for idx, logit in zip(sampled_indices.tolist(), sampled_logits.tolist()):
                    child_node = TreeNode(
                        token_id=idx,
                        hidden_state=hidden_state,
                        logit=logit,
                        parent=parent_node,
                        depth=depth,
                        topk_tokens=diag_topk_tokens,
                        topk_probs=diag_topk_probs,
                    )
                    child_node.past_key_values = outputs.past_key_values
                    next_frontier.append(child_node)
                    
                    # If this is the last depth, add to paths
                    if depth == self.max_depth - 1:
                        path_nodes = child_node.get_path_to_root()[1:]  # Exclude root
                        token_ids = [node.token_id for node in path_nodes]
                        
                        if return_hidden_states:
                            # Collect hidden states
                            hidden_states_list = []
                            for node in path_nodes:
                                if node.hidden_state is not None:
                                    hidden_states_list.append(node.hidden_state)
                            
                            if hidden_states_list:
                                hidden_states = torch.stack(hidden_states_list)
                            else:
                                hidden_states = None
                        else:
                            hidden_states = None
                        
                        # Collect diagnostics along the path
                        draft_topk_tokens = None
                        draft_topk_probs = None
                        if collect_diagnostics:
                            draft_topk_tokens = []
                            draft_topk_probs = []
                            for node in path_nodes:
                                draft_topk_tokens.append(node.topk_tokens or [])
                                draft_topk_probs.append(node.topk_probs or [])
                        
                        tree_path = TreePath(
                            nodes=path_nodes,
                            token_ids=token_ids,
                            hidden_states=hidden_states,
                            cumulative_score=child_node.cumulative_score,
                            draft_topk_tokens=draft_topk_tokens,
                            draft_topk_probs=draft_topk_probs,
                        )
                        all_paths.append(tree_path)
            
            frontier = next_frontier
            
            # Early stopping if no more candidates
            if not frontier:
                break
            
        return all_paths
    
    def get_tree_statistics(self, tree_paths: List[TreePath]) -> Dict:
        """Tree 생성 통계 반환"""
        if not tree_paths:
            return {}
            
        depths = [len(path.token_ids) for path in tree_paths]
        scores = [path.cumulative_score for path in tree_paths]
        
        return {
            'num_paths': len(tree_paths),
            'avg_depth': np.mean(depths),
            'max_depth': max(depths),
            'min_depth': min(depths),
            'avg_score': np.mean(scores),
            'max_score': max(scores),
            'min_score': min(scores),
        } 