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
from models import AffineAlignment, DraftTreeSearch, AcceptanceProbabilityPredictor, TreePruner
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
            self.target_model = AutoModelForCausalLM.from_pretrained(
                config.model.target_model_name,
                torch_dtype=config.model.dtype,
                device_map="auto",  # Automatic multi-GPU distribution
                trust_remote_code=True
            )
        else:
            self.target_model = target_model
            
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
            do_sample=False
        )
        
        self.tree_pruner = TreePruner(
            min_acceptance_prob=config.pruning.min_acceptance_prob,
            adaptive_pruning=config.pruning.adaptive_pruning,
            pruning_ratio=config.pruning.pruning_ratio,
            top_k_paths=config.pruning.top_k_paths
        )
        
        # Load pre-trained weights if available
        if config.affine_alignment.alignment_checkpoint:
            self.load_pretrained_weights(config.affine_alignment.alignment_checkpoint)
            
        # Statistics tracking
        self.stats = {
            'total_draft_tokens': 0,
            'total_accepted_tokens': 0,
            'total_iterations': 0,
            'pruning_stats': [],
            'timing': {
                'draft_generation': 0,
                'alignment': 0,
                'prediction': 0,
                'pruning': 0,
                'verification': 0
            }
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
            return_hidden_states=True
        )
        self.stats['timing']['draft_generation'] += time.time() - t0
        step_stats['num_draft_paths'] = len(tree_paths)
        
        if not tree_paths:
            draft_device = self._get_model_device(self.draft_model)
            return torch.tensor([], device=draft_device), step_stats
            
        # 2. Affine alignment - handle device transfer
        t0 = time.time()
        for path in tree_paths:
            if path.hidden_states is not None:
                # Move hidden states to alignment device
                hidden_states_aligned = path.hidden_states.to(self.primary_device)
                aligned_states = self.affine_alignment(hidden_states_aligned.unsqueeze(0))
                path.aligned_states = aligned_states
        self.stats['timing']['alignment'] += time.time() - t0
        
        # 3. Acceptance probability prediction
        t0 = time.time()
        path_probs = []
        for path in tree_paths:
            if path.aligned_states is not None:
                with torch.no_grad():
                    # aligned_states shape: [1, seq_len, hidden_size]
                    aligned_states = path.aligned_states
                    # Ensure correct device and dtype to match the MLP
                    aligned_states = aligned_states.to(self.primary_device)
                    target_dtype = self.acceptance_predictor.mlp[0].weight.dtype
                    if aligned_states.dtype != target_dtype:
                        aligned_states = aligned_states.to(dtype=target_dtype)
                    # Predict probabilities in batch and aggregate to path-level
                    probs = self.acceptance_predictor(aligned_states.unsqueeze(0))  # [1, seq_len]
                    acceptance_prob = float(probs.mean().item())
                    path.acceptance_prob = acceptance_prob
                    path_probs.append(acceptance_prob)
            else:
                # Fallback normalization logic
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
        타겟 모델로 드래프트 경로들을 배치로 한 번에 검증 (teacher forcing + greedy 비교)
        """
        if not paths:
            draft_device = self._get_model_device(self.draft_model)
            return torch.tensor([], device=draft_device), 0.0

        target_device = self._get_model_device(self.target_model)
        draft_device = self._get_model_device(self.draft_model)

        # Cap number of paths to verify for speed
        max_verify = 8
        if len(paths) > max_verify:
            paths = sorted(
                paths,
                key=lambda p: (
                    p.acceptance_prob if p.acceptance_prob is not None else -1.0,
                    p.cumulative_score,
                ),
                reverse=True,
            )[:max_verify]

        # Build batched sequences: base input + full path tokens (padded to max)
        base_len = input_ids.shape[1]
        max_path_len = max(len(p.token_ids) for p in paths)

        batch_input_ids = []
        batch_attention_mask = []
        valid_lengths = []

        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        for path in paths:
            path_tokens = torch.tensor(path.token_ids, device=draft_device).unsqueeze(0)  # [1, L]
            candidate_ids = torch.cat([input_ids.to(draft_device), path_tokens], dim=1)
            path_len = len(path.token_ids)
            padding_needed = max_path_len - path_len
            if padding_needed > 0:
                pad = torch.full((1, padding_needed), pad_token_id, device=draft_device)
                candidate_ids = torch.cat([candidate_ids, pad], dim=1)
            batch_input_ids.append(candidate_ids)
            valid_lengths.append(path_len)

            if attention_mask is not None:
                path_attn = torch.cat([
                    attention_mask.to(draft_device),
                    torch.ones(1, path_len, device=draft_device),
                    torch.zeros(1, padding_needed, device=draft_device),
                ], dim=1)
            else:
                path_attn = torch.cat([
                    torch.ones(1, base_len + path_len, device=draft_device),
                    torch.zeros(1, padding_needed, device=draft_device),
                ], dim=1)
            batch_attention_mask.append(path_attn)

        batch_input_ids = torch.cat(batch_input_ids, dim=0).to(target_device)  # [B, base+maxL]
        batch_attention_mask = torch.cat(batch_attention_mask, dim=0).to(target_device)

        # Teacher-forced forward for all paths at once (no cache for better throughput on long seq)
        with torch.no_grad():
            outputs = self.target_model(
                input_ids=batch_input_ids[:, :-1],
                attention_mask=batch_attention_mask[:, :-1],
                use_cache=False,
            )

        # Logits for positions where new tokens appear
        all_logits = outputs.logits[:, base_len - 1 : base_len - 1 + max_path_len, :]

        # Verify each path greedily
        best_path_idx = -1
        max_accepted_length = 0

        for i, (path, valid_len) in enumerate(zip(paths, valid_lengths)):
            path_logits = all_logits[i, :valid_len, :]
            accepted_length = 0
            for j, draft_token in enumerate(path.token_ids):
                target_token = torch.argmax(path_logits[j]).item()
                if target_token == draft_token:
                    accepted_length += 1
                else:
                    break
            if accepted_length > max_accepted_length:
                max_accepted_length = accepted_length
                best_path_idx = i

        if best_path_idx >= 0 and max_accepted_length > 0:
            best_path = paths[best_path_idx]
            accepted_tokens = torch.tensor(best_path.token_ids[:max_accepted_length], device=draft_device)
            acceptance_rate = max_accepted_length / len(best_path.token_ids)
        else:
            accepted_tokens = torch.tensor([], device=draft_device)
            acceptance_rate = 0.0

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
            
        return stats 