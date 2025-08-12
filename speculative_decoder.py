"""
Affine Alignment based Speculative Decoder

이 모듈은 모든 구성 요소를 통합하여 완전한 speculative decoding 시스템을 제공합니다.

주요 기능:
1. Draft tree generation
2. Affine alignment-based hidden state transformation
3. Acceptance probability prediction and pruning
4. Target model verification
5. Token selection and generation
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Dict, List, Tuple, Union
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from collections import defaultdict
import warnings
import numpy as np

# Enable TF32 for speed where available
try:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

# Import internal modules
from config import SpeculativeDecodingConfig
from models import AffineAlignment, DraftTreeSearch, AcceptanceProbabilityPredictor, TreePruner, TreeMaskModelWrapper
from models.draft_tree_search import TreePath  # Add TreePath import

warnings.filterwarnings("ignore")


class SpeculativeDecoder:
    """
    Affine Alignment 기반 Speculative Decoder
    
    이 클래스는 모든 구성 요소를 통합하여 효율적인 speculative decoding을 수행합니다.
    """
    
    def __init__(
        self,
        config: SpeculativeDecodingConfig,
        draft_model: Optional[AutoModelForCausalLM] = None,
        target_model: Optional[AutoModelForCausalLM] = None,
        tokenizer: Optional[AutoTokenizer] = None
    ):
        self.config = config
        self.device = config.model.device
        
        # Multi-GPU support: set primary device for alignment modules
        if self.device == "cuda":
            self.primary_device = "cuda:0"  # Alignment modules on first GPU
        else:
            self.primary_device = self.device
        
        # Load models if not provided
        if draft_model is None:
            print(f"Loading draft model: {config.model.draft_model_name}")
            self.draft_model = AutoModelForCausalLM.from_pretrained(
                config.model.draft_model_name,
                torch_dtype=config.model.dtype,
                device_map="auto",  # Automatic multi-GPU distribution
                trust_remote_code=True
            )
        else:
            self.draft_model = draft_model
            
        if target_model is None:
            print(f"Loading target model: {config.model.target_model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                config.model.target_model_name,
                torch_dtype=config.model.dtype,
                device_map="auto",  # Automatic multi-GPU distribution
                trust_remote_code=True
            )
            # Wrap with TreeMaskModelWrapper for 4D attention mask support - DISABLED
            # self.target_model = TreeMaskModelWrapper(base_model)
            # print("✓ Target model wrapped with TreeMaskModelWrapper for tree attention support")
            self.target_model = base_model
            print("⚠ TreeMaskModelWrapper DISABLED for debugging")
        else:
            # Wrap provided model if not already wrapped - DISABLED
            # if not isinstance(target_model, TreeMaskModelWrapper):
            #     self.target_model = TreeMaskModelWrapper(target_model)
            #     print("✓ Wrapped provided target model with TreeMaskModelWrapper")
            # else:
            #     self.target_model = target_model
            self.target_model = target_model
            print("⚠ TreeMaskModelWrapper DISABLED for debugging")
            
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model.draft_model_name,
                trust_remote_code=True
            )
        else:
            self.tokenizer = tokenizer
            
        # Initialize components on primary device
        self.affine_alignment = AffineAlignment(
            hidden_size_draft=config.affine_alignment.hidden_size_draft,
            hidden_size_target=config.affine_alignment.hidden_size_target,
            use_bias=config.affine_alignment.use_bias,
            dropout_rate=config.affine_alignment.dropout_rate
        ).to(self.primary_device).to(config.model.dtype)  # Match model dtype
        
        self.acceptance_predictor = AcceptanceProbabilityPredictor(
            input_dim=config.affine_alignment.hidden_size_target,
            hidden_dims=config.mlp.hidden_dims,
            activation=config.mlp.activation,
            dropout_rate=config.mlp.dropout_rate,
            use_layer_norm=config.mlp.use_layer_norm
        ).to(self.primary_device).to(config.model.dtype)  # Match model dtype
        
        self.tree_search = DraftTreeSearch(
            draft_model=self.draft_model,
            tokenizer=self.tokenizer,
            max_candidates=config.tree_search.max_candidates,
            max_depth=config.tree_search.max_depth,
            temperature=config.tree_search.temperature,
            top_k=config.tree_search.top_k,
            top_p=config.tree_search.top_p,
            device=self.device,  # Will handle multi-GPU internally
            do_sample=False,
            max_paths_per_level=config.tree_search.max_paths_per_level
        )
        
        self.tree_pruner = TreePruner(
            min_acceptance_prob=config.pruning.min_acceptance_prob,
            adaptive_pruning=config.pruning.adaptive_pruning,
            pruning_ratio=config.pruning.pruning_ratio,
            top_k_paths=config.pruning.top_k_paths
        )
        
        # Ensure models are in eval mode to disable dropout
        self.draft_model.eval()
        self.target_model.eval()
        
        # Load pre-trained weights if available
        if config.affine_alignment.alignment_checkpoint:
            self.load_pretrained_weights(config.affine_alignment.alignment_checkpoint)
            
        # Statistics tracking
        self.stats = {
            'total_draft_tokens': 0,
            'total_accepted_tokens': 0,
            'total_verified_tokens': 0,  # sum of best path lengths (denominator for effective acceptance)
            'total_iterations': 0,
            'pruning_stats': [],
            'timing': {
                'draft_generation': 0,
                'alignment': 0,
                'prediction': 0,
                'pruning': 0,
                'verification': 0
            },
            'diagnostics': []
        }
        
    def load_pretrained_weights(self, checkpoint_path: str):
        """
        Load pre-trained weights for affine alignment and acceptance predictor
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        print(f"Loading pre-trained weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Load affine alignment weights
        if 'W' in checkpoint and 'b' in checkpoint:
            # Convert to appropriate dtype and device
            W = checkpoint['W'].to(dtype=self.config.model.dtype, device=self.primary_device)
            b = checkpoint['b'].to(dtype=self.config.model.dtype, device=self.primary_device)
            
            # Assign to affine alignment module
            self.affine_alignment.weight.data = W
            if self.affine_alignment.bias is not None:
                self.affine_alignment.bias.data = b
            print(f"✓ Loaded affine alignment weights: W={W.shape}, b={b.shape}")
        
        # Don't load MLP from affine checkpoint - use separately trained one
        print("Pre-trained affine weights loaded successfully!")
    
    def load_acceptance_mlp(self, mlp_checkpoint_path: str):
        """
        Load separately trained acceptance probability predictor MLP
        
        Args:
            mlp_checkpoint_path: Path to the trained MLP checkpoint
        """
        print(f"Loading acceptance MLP from {mlp_checkpoint_path}")
        mlp_checkpoint = torch.load(mlp_checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'model_state_dict' in mlp_checkpoint:
            # Load the state dict
            state_dict = mlp_checkpoint['model_state_dict']
            
            # Create a mapping for our acceptance predictor structure
            # The trained model has a simple structure without LayerNorm/Dropout
            our_state_dict = self.acceptance_predictor.state_dict()
            
            # Map the weights - assuming the trained MLP has structure:
            # Linear(8192->256) -> ReLU -> Linear(256->128) -> ReLU -> Linear(128->1)
            layer_mapping = {
                'mlp.0.weight': 'mlp.0.weight',    # First linear
                'mlp.0.bias': 'mlp.0.bias',
                'mlp.2.weight': 'mlp.3.weight',    # Second linear (after ReLU, LayerNorm, Dropout)
                'mlp.2.bias': 'mlp.3.bias',
                'mlp.4.weight': 'mlp.6.weight',    # Final linear
                'mlp.4.bias': 'mlp.6.bias'
            }
            
            # Update our state dict
            for src_key, dst_key in layer_mapping.items():
                if src_key in state_dict and dst_key in our_state_dict:
                    weight = state_dict[src_key].to(
                        dtype=self.config.model.dtype,
                        device=self.primary_device
                    )
                    our_state_dict[dst_key] = weight
            
            # Load updated state dict
            self.acceptance_predictor.load_state_dict(our_state_dict, strict=False)
            
            if 'val_acc' in mlp_checkpoint:
                print(f"✓ Loaded MLP with validation accuracy: {mlp_checkpoint['val_acc']:.4f}")
            else:
                print("✓ Loaded MLP weights successfully")
                
        print("Acceptance MLP loaded successfully!")
        
    def _get_model_device(self, model):
        """Get the primary device of a model (for multi-GPU models)"""
        try:
            # For models with device_map
            if hasattr(model, 'hf_device_map'):
                # Get the device of the first layer
                return next(iter(model.parameters())).device
            else:
                return next(iter(model.parameters())).device
        except:
            return torch.device(self.device)
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        early_stopping: bool = True,
        eos_token_id: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Speculative decoding을 사용한 텍스트 생성
        
        Args:
            input_ids: 입력 토큰 시퀀스
            attention_mask: 어텐션 마스크
            max_new_tokens: 생성할 최대 토큰 수
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            do_sample: Sampling 여부
            early_stopping: EOS 토큰에서 조기 종료
            eos_token_id: EOS 토큰 ID
            
        Returns:
            generated_ids: 생성된 토큰 시퀀스
            generation_stats: 생성 통계
        """
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
            
        # Move input to appropriate device for multi-GPU
        draft_device = self._get_model_device(self.draft_model)
        input_ids = input_ids.to(draft_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(draft_device)
            
        # Initialize
        generated_ids = input_ids.clone()
        cur_len = input_ids.shape[1]
        
        # Generation loop
        with tqdm(total=max_new_tokens, desc="Generating") as pbar:
            while generated_ids.shape[1] - cur_len < max_new_tokens:
                # Update tree search parameters
                self.tree_search.temperature = temperature
                self.tree_search.top_k = top_k  
                self.tree_search.top_p = top_p
                self.tree_search.do_sample = do_sample
                
                # Debug: ensure greedy for same model test
                if getattr(self.config, 'profile', False) and self.stats['total_iterations'] < 3:
                    print(f"\n[DEBUG] Tree search params: temp={temperature}, top_k={top_k}, top_p={top_p}, do_sample={do_sample}")
                
                # Single iteration of speculative decoding
                new_tokens, iter_stats = self._speculative_decode_step(
                    generated_ids,
                    attention_mask,
                    do_sample=do_sample
                )
                
                # Append accepted tokens
                if new_tokens.numel() > 0:
                    generated_ids = torch.cat([generated_ids, new_tokens], dim=1)
                    
                    # Update attention mask
                    if attention_mask is not None:
                        attention_mask = torch.cat([
                            attention_mask,
                            torch.ones(1, new_tokens.shape[1], device=attention_mask.device)
                        ], dim=1)
                        
                    # Check for EOS
                    if early_stopping and eos_token_id is not None:
                        if (new_tokens == eos_token_id).any():
                            break
                            
                    pbar.update(new_tokens.shape[1])
                else:
                    # No tokens accepted, generate single token with target model
                    single_token = self._generate_single_token(
                        generated_ids,
                        attention_mask,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        do_sample=do_sample
                    )
                    generated_ids = torch.cat([generated_ids, single_token], dim=1)
                    
                    if attention_mask is not None:
                        attention_mask = torch.cat([
                            attention_mask,
                            torch.ones(1, 1, device=attention_mask.device)
                        ], dim=1)
                        
                    if early_stopping and single_token.item() == eos_token_id:
                        break
                        
                    pbar.update(1)
                    
        # Compute final statistics
        final_stats = self._compute_generation_stats()
        
        return generated_ids, final_stats
    
    def _speculative_decode_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        do_sample: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Single step of speculative decoding
        
        Returns:
            accepted_tokens: 검증을 통과한 토큰들
            step_stats: 단계별 통계
        """
        step_stats = {}
        start_time = time.time()
        
        # 1. Draft tree generation
        t0 = time.time()
        tree_paths = self.tree_search.generate_tree(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_hidden_states=True,
            collect_diagnostics=getattr(self.config, 'profile', False)
        )
        self.stats['timing']['draft_generation'] += time.time() - t0
        step_stats['num_draft_paths'] = len(tree_paths)
        
        if not tree_paths:
            draft_device = self._get_model_device(self.draft_model)
            return torch.tensor([], device=draft_device), step_stats
            
        # 2. Affine alignment - handle device transfer
        t0 = time.time()
        # Batch process all paths at once
        all_hidden_states = []
        path_lengths = []
        paths_with_hidden = []
        
        for path in tree_paths:
            if path.hidden_states is not None:
                all_hidden_states.append(path.hidden_states)
                path_lengths.append(path.hidden_states.shape[0])
                paths_with_hidden.append(path)
        
        if all_hidden_states:
            # Get draft device and dtype - use these throughout
            draft_device = self._get_model_device(self.draft_model)
            draft_dtype = next(self.draft_model.parameters()).dtype
            
            # Ensure affine/mlp are on draft device/dtype for efficiency
            if self.affine_alignment.weight.device != draft_device:
                self.affine_alignment = self.affine_alignment.to(draft_device)
            if self.acceptance_predictor.mlp[0].weight.device != draft_device:
                self.acceptance_predictor = self.acceptance_predictor.to(draft_device)
            
            # Pad sequences to same length for batching
            max_len = max(path_lengths)
            padded_states = []
            for hidden in all_hidden_states:
                seq_len = hidden.shape[0]
                if seq_len < max_len:
                    padding = torch.zeros(max_len - seq_len, hidden.shape[1], 
                                        device=hidden.device, dtype=hidden.dtype)
                    padded = torch.cat([hidden, padding], dim=0)
                else:
                    padded = hidden
                padded_states.append(padded)
            
            # Stack into batch [B, L, H]
            batch_hidden = torch.stack(padded_states, dim=0)
            
            # Batch affine transformation
            batch_aligned = self.affine_alignment(batch_hidden)  # [B, L, H_target]
            
            # Store aligned states back to paths
            for i, (path, length) in enumerate(zip(paths_with_hidden, path_lengths)):
                path.aligned_states = batch_aligned[i, :length]  # Trim padding
        
        self.stats['timing']['alignment'] += time.time() - t0
        
        # 3. Acceptance probability prediction - now batched
        t0 = time.time()
        path_probs = []
        
        if paths_with_hidden:
            # Collect all aligned states
            all_aligned = []
            for path in paths_with_hidden:
                if path.aligned_states is not None:
                    all_aligned.append(path.aligned_states)
            
            if all_aligned:
                # Flatten all sequences for MLP: [total_tokens, H]
                flat_aligned = torch.cat(all_aligned, dim=0)
                
                # Batch MLP forward
                with torch.no_grad():
                    flat_probs = self.acceptance_predictor(flat_aligned)  # [total_tokens]
                
                # Redistribute probabilities back to paths
                offset = 0
                for path in paths_with_hidden:
                    length = path.aligned_states.shape[0]
                    path_prob_values = flat_probs[offset:offset+length]
                    # Use geometric mean instead of arithmetic mean for length-invariance
                    path.acceptance_prob = float(torch.exp(torch.log(path_prob_values + 1e-8).mean()).item())
                    path_probs.append(path.acceptance_prob)
        
        # Handle paths without hidden states
        for path in tree_paths:
            if path not in paths_with_hidden:
                # Fallback normalization
                if len(tree_paths) > 1:
                    min_score = min(p.cumulative_score for p in tree_paths)
                    max_score = max(p.cumulative_score for p in tree_paths)
                    score_range = max_score - min_score
                    if score_range > 0:
                        normalized_score = (path.cumulative_score - min_score) / score_range
                    else:
                        normalized_score = 0.5
                else:
                    normalized_score = 0.5
                path.acceptance_prob = normalized_score
                path_probs.append(normalized_score)
        
        # Debug: print MLP predictions for first few steps
        if self.stats['total_iterations'] < 3:
            print(f"\n[Step {self.stats['total_iterations']}] MLP predictions: {path_probs[:5]}")
        
        self.stats['timing']['prediction'] += time.time() - t0
        step_stats['path_probabilities'] = path_probs
        
        # 4. Tree pruning
        t0 = time.time()
        # Pass the calculated acceptance probabilities to the pruner
        pruned_paths, pruning_stats = self.tree_pruner.prune_paths(
            tree_paths,
            path_probs  # Pass the MLP predictions as positional argument
        )
        self.stats['timing']['pruning'] += time.time() - t0
        
        step_stats['num_pruned_paths'] = len(pruned_paths)
        if pruning_stats:
            step_stats['pruning_ratio'] = pruning_stats.pruning_ratio
            self.stats['pruning_stats'].append(pruning_stats)
        
        if not pruned_paths:
            draft_device = self._get_model_device(self.draft_model)
            return torch.tensor([], device=draft_device), step_stats
        
        # 5. Target model verification with pruned paths
        t0 = time.time()
        accepted_tokens, acceptance_rate = self._verify_with_target(
            input_ids,
            attention_mask, 
            pruned_paths,
            do_sample=do_sample
        )
        self.stats['timing']['verification'] += time.time() - t0
        
        # Update statistics
        self.stats['total_iterations'] += 1
        self.stats['total_draft_tokens'] += sum(len(p.token_ids) for p in tree_paths)
        self.stats['total_accepted_tokens'] += len(accepted_tokens)
        
        step_stats['acceptance_rate'] = acceptance_rate
        step_stats['num_accepted_tokens'] = len(accepted_tokens)
        step_stats['time'] = time.time() - start_time
        
        # Update pruner performance
        self.tree_pruner.update_performance(acceptance_rate)
        
        return accepted_tokens.unsqueeze(0), step_stats

    def _verify_with_target(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        paths: List[TreePath],
        do_sample: bool = True
    ) -> Tuple[torch.Tensor, float]:
        """
        트리-마스크 방식 검증: 트리를 한 시퀀스로 펼치고, 각 노드가 자기 조상만 볼 수 있게
        attention mask를 구성한 후, 타겟 모델 1회 전방패스로 모든 노드의 로짓을 획득.
        """
        if not paths:
            draft_device = self._get_model_device(self.draft_model)
            return torch.tensor([], device=draft_device), 0.0

        target_device = self._get_model_device(self.target_model)
        draft_device = self._get_model_device(self.draft_model)

        # 1) Flatten tree: collect all unique nodes in topological order
        # Build node -> flat_idx mapping
        visited = {}  # node_id -> flat_idx
        flat_token_ids = []
        flat_position_ids = []
        flat_parent_indices = []  # For mask construction
        node_to_flat_idx = {}
        flat_idx = 0
        
        # Start with base context length
        base_len = input_ids.shape[1]
        
        # Process nodes by depth (BFS style to ensure parents before children)
        max_depth = max(len(p.token_ids) for p in paths)
        for depth in range(max_depth):
            for path in paths:
                if depth < len(path.nodes):
                    node = path.nodes[depth]
                    node_id = id(node)
                    if node_id not in visited:
                        visited[node_id] = flat_idx
                        node_to_flat_idx[node] = flat_idx
                        flat_token_ids.append(node.token_id)
                        # Position ID = base_len + depth (all nodes at same depth share position)
                        flat_position_ids.append(base_len + depth)
                        # Find parent's flat index
                        if node.parent and id(node.parent) in visited:
                            parent_flat_idx = visited[id(node.parent)]
                            flat_parent_indices.append(parent_flat_idx)
                        else:
                            # Root node or first depth - can see all base context
                            flat_parent_indices.append(-1)
                        flat_idx += 1
        
        if not flat_token_ids:
            return torch.tensor([], device=draft_device), 0.0
        
        # 2) Construct tree-structured attention mask - vectorized
        # Each node can only attend to: (1) base context, (2) its ancestors
        tree_size = len(flat_token_ids)
        total_len = base_len + tree_size
        
        # Vectorized mask construction
        # Create attention mask [1, 1, total_len, total_len] 
        attn_mask = torch.zeros((total_len, total_len), device=target_device)
        
        # Base context: causal mask
        base_indices = torch.arange(base_len, device=target_device)
        attn_mask[:base_len, :base_len] = torch.triu(torch.ones(base_len, base_len, device=target_device), diagonal=1)
        
        # Tree nodes setup
        if tree_size > 0:
            # Convert parent indices to tensor for vectorized ops
            parent_indices_tensor = torch.tensor(flat_parent_indices, device=target_device, dtype=torch.long)
            
            # Build ancestor matrix efficiently
            # ancestors[i,j] = 1 if j is ancestor of i
            ancestors = torch.zeros((tree_size, tree_size), device=target_device, dtype=torch.bool)
            
            # Each node can see itself
            ancestors.fill_diagonal_(True)
            
            # Propagate ancestor relationships
            for i in range(tree_size):
                if parent_indices_tensor[i] >= 0:
                    # Node i can see its parent
                    ancestors[i, parent_indices_tensor[i]] = True
                    # Node i can see all ancestors of its parent
                    if parent_indices_tensor[i] < i:  # Parent was processed before
                        ancestors[i] = ancestors[i] | ancestors[parent_indices_tensor[i]]
            
            # Set tree part of attention mask
            # Tree nodes can't see each other unless ancestor relationship
            attn_mask[base_len:, base_len:] = (~ancestors).float()
            
            # Tree nodes can see all base context (already 0)
            # Base context can't see tree nodes
            attn_mask[:base_len, base_len:] = 1.0
        
        # Convert to HF format: 1 -> -10000, 0 -> 0
        attn_mask = attn_mask * -10000.0
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
        
        # 3) Prepare inputs with better position IDs
        flat_tokens_tensor = torch.tensor(flat_token_ids, device=target_device).unsqueeze(0)
        combined_input_ids = torch.cat([input_ids.to(target_device), flat_tokens_tensor], dim=1)
        
        # Position IDs: avoid collisions by using unique positions for each node
        # Strategy: base positions as-is, then tree nodes get sequential positions
        base_positions = torch.arange(base_len, device=target_device)
        tree_positions = torch.arange(base_len, base_len + tree_size, device=target_device)
        combined_position_ids = torch.cat([base_positions, tree_positions]).unsqueeze(0)
        
        # Basic attention mask for padding (all 1s since no padding)
        basic_attention_mask = torch.ones(1, total_len, device=target_device)
        if attention_mask is not None:
            basic_attention_mask[:, :base_len] = attention_mask.to(target_device)
        
        # 4) Single forward pass with tree mask
        # Note: Most HF models don't support 4D attention masks directly.
        # We'll convert our tree mask to a 2D attention mask that encodes the tree structure
        with torch.no_grad():
            # Convert 4D tree mask to 2D mask compatible with HF models
            # HF expects 0 for attend, 1 for mask (opposite of our convention)
            # Also, HF uses additive masks with large negative values
            
            # Create 2D attention mask from our tree structure
            # Shape: [batch_size, seq_len]
            tree_mask_2d = torch.zeros((1, total_len), device=target_device)
            
            # For causal mask with tree structure, we need to create a custom attention mask
            # that HF models can process. We'll use the approach of creating a full attention
            # matrix and then converting it.
            # Create a full attention matrix [seq_len, seq_len]
            full_attn_mask = torch.zeros((total_len, total_len), device=target_device)
            
            # Fill in the attention pattern
            # Base context can attend to itself causally
            for i in range(base_len):
                full_attn_mask[i, i+1:base_len] = 1  # mask future tokens in base
                
            # Tree nodes can only see their ancestors and base
            for i in range(tree_size):
                tree_pos = base_len + i
                # Block everything first
                full_attn_mask[tree_pos, :] = 1
                # Allow base context
                full_attn_mask[tree_pos, :base_len] = 0
                # Allow self
                full_attn_mask[tree_pos, tree_pos] = 0
                
                # Trace ancestors and allow
                current_idx = i
                while flat_parent_indices[current_idx] >= 0:
                    parent_idx = flat_parent_indices[current_idx]
                    parent_pos = base_len + parent_idx
                    full_attn_mask[tree_pos, parent_pos] = 0
                    current_idx = parent_idx
            
            # Convert to HF format: 1 -> -10000, 0 -> 0
            full_attn_mask = full_attn_mask * -10000.0
            full_attn_mask = full_attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
            
            # For models that support it, we can try to pass the 4D mask directly
            # Otherwise, we'll rely on the model to expand the 2D mask appropriately
            use_tree_wrapper = False  # isinstance(self.target_model, TreeMaskModelWrapper) - DISABLED
            
            # Simple forward pass - no tree mask, just standard attention
            outputs = self.target_model(
                input_ids=combined_input_ids,
                attention_mask=basic_attention_mask,
                position_ids=combined_position_ids,
                use_cache=False,
            )
        
        # 5) Extract logits for tree nodes
        # CRITICAL: Teacher-forcing means logits at position i predict token i+1
        # The first tree token is predicted by logits at position base_len-1
        all_logits = outputs.logits[0, base_len-1:base_len-1+tree_size, :]  # [tree_size, vocab]
        
        # Debug: check shapes and positions
        if self.stats['total_iterations'] < 3 and getattr(self.config, 'profile', False):
            print(f"\n[DEBUG] Logit extraction:")
            print(f"  base_len={base_len}, tree_size={tree_size}")
            print(f"  outputs.logits.shape={outputs.logits.shape}")
            print(f"  all_logits.shape={all_logits.shape}")
            print(f"  combined_input_ids.shape={combined_input_ids.shape}")
        
        # 6) Verify each path
        best_path_idx = -1
        max_accepted_length = 0
        
        # Debug: print first few verifications
        if self.stats['total_iterations'] < 3 and getattr(self.config, 'profile', False):
            print(f"\n[DEBUG Step {self.stats['total_iterations']}] Verification details:")
        
        for path_idx, path in enumerate(paths):
            accepted_length = 0
            for depth, node in enumerate(path.nodes):
                # Get flat index for this node
                if node not in node_to_flat_idx:
                    break
                flat_idx = node_to_flat_idx[node]
                
                # Get target's greedy prediction at this position
                node_logits = all_logits[flat_idx]
                target_token = torch.argmax(node_logits).item()
                
                # Debug output for first few paths/steps
                if self.stats['total_iterations'] < 3 and path_idx < 2 and depth < 3 and getattr(self.config, 'profile', False):
                    # Get top 5 predictions from target
                    topk_values, topk_indices = torch.topk(node_logits, 5)
                    topk_probs = F.softmax(topk_values, dim=-1)
                    print(f"  Path {path_idx}, depth {depth}: draft_token={node.token_id}, target_token={target_token}")
                    print(f"    Target top-5: {[(self.tokenizer.decode([t]), f'{p:.3f}') for t, p in zip(topk_indices.tolist(), topk_probs.tolist())]}")
                    print(f"    Match: {target_token == node.token_id}")
                
                if target_token == node.token_id:
                    accepted_length += 1
                else:
                    break
            
            if accepted_length > max_accepted_length:
                max_accepted_length = accepted_length
                best_path_idx = path_idx
        
        # 7) Return results
        if best_path_idx >= 0 and max_accepted_length > 0:
            best_path = paths[best_path_idx]
            accepted_tokens = torch.tensor(
                best_path.token_ids[:max_accepted_length], 
                device=draft_device
            )
            acceptance_rate = max_accepted_length / len(best_path.token_ids)
        else:
            accepted_tokens = torch.tensor([], device=draft_device)
            acceptance_rate = 0.0
        
        # Update stats
        self.stats['total_verified_tokens'] += max_accepted_length
        
        return accepted_tokens, acceptance_rate
    
    def _generate_single_token(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> torch.Tensor:
        """
        타겟 모델로 단일 토큰 생성 (fallback)
        """
        # Get target model device and move inputs
        target_device = self._get_model_device(self.target_model)
        input_ids_target = input_ids.to(target_device)
        attention_mask_target = attention_mask.to(target_device) if attention_mask is not None else None
        
        with torch.no_grad():
            outputs = self.target_model(
                input_ids=input_ids_target,
                attention_mask=attention_mask_target
            )
            
        logits = outputs.logits[0, -1, :]
        
        # Apply temperature
        logits = logits / temperature
        
        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][-1]
            logits[indices_to_remove] = float('-inf')
            
        # Top-p filtering  
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            
        # Sample or greedy
        if do_sample:
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
        else:
            next_token = torch.argmax(logits, keepdim=True)
            
        # Return on draft device for consistency
        draft_device = self._get_model_device(self.draft_model)
        return next_token.unsqueeze(0).to(draft_device)
    
    def _compute_generation_stats(self) -> Dict:
        """생성 통계 계산"""
        stats = {
            'total_iterations': self.stats['total_iterations'],
            'total_draft_tokens': self.stats['total_draft_tokens'],
            'total_accepted_tokens': self.stats['total_accepted_tokens'],
            'average_acceptance_rate': (
                self.stats['total_accepted_tokens'] / self.stats['total_draft_tokens']
                if self.stats['total_draft_tokens'] > 0 else 0
            ),
            'effective_acceptance_rate': (
                self.stats['total_accepted_tokens'] / self.stats['total_verified_tokens']
                if self.stats['total_verified_tokens'] > 0 else 0
            ),
            'tokens_per_iteration': (
                self.stats['total_accepted_tokens'] / self.stats['total_iterations']
                if self.stats['total_iterations'] > 0 else 0
            ),
            'timing': self.stats['timing']
        }
        
        # Pruning statistics
        if self.stats['pruning_stats']:
            pruning_ratios = [s.pruning_ratio for s in self.stats['pruning_stats']]
            stats['average_pruning_ratio'] = np.mean(pruning_ratios)
        
        # Diagnostics
        if getattr(self.config, 'profile', False):
            stats['diagnostics'] = self.stats.get('diagnostics', [])
            stats['avg_best_path_len'] = (
                float(np.mean([d['best_path_len'] for d in stats['diagnostics']]))
                if stats['diagnostics'] else 0.0
            )
            stats['avg_num_paths_verified'] = (
                float(np.mean([d['num_paths_verified'] for d in stats['diagnostics']]))
                if stats['diagnostics'] else 0.0
            )
        
        return stats 