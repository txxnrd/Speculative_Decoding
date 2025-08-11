"""Optimized Speculative Decoding V2 - Based on HuggingFace implementation."""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import time
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass, field

from .sampling import sample_from_logits, top_k_top_p_filtering
from ..utils.config import Config


@dataclass
class SpeculativeDecodingConfig:
    """Configuration for speculative decoding."""
    num_assistant_tokens: int = 5
    dynamic_adjustment: bool = False
    confidence_decay: float = 0.9
    confidence_boost: float = 1.1
    use_cache: bool = True
    verbose: bool = False
    
    # Affine verification
    affine_verification: bool = False
    affine_model_path: Optional[str] = None
    affine_accept_threshold: float = 0.5
    
    # Tree search parameters
    tree_search: bool = False
    beam_width: int = 3  # Number of candidate paths per position
    tree_depth: int = 5  # Same as num_assistant_tokens by default
    pruning_threshold: float = 0.3  # Minimum acceptance prob to keep path
    
    # Tree expansion strategy
    tree_expansion_strategy: str = "top_k"  # "top_k", "top_p", "beam", "diverse_beam"
    tree_top_k: int = 10  # For top-k expansion
    tree_top_p: float = 0.9  # For nucleus sampling expansion
    tree_temperature: float = 1.0  # Temperature for tree expansion
    tree_diversity_penalty: float = 0.5  # For diverse beam search


class OptimizedSpeculativeDecoderV2:
    """Optimized speculative decoder based on HuggingFace implementation."""
    
    def __init__(
        self,
        draft_model,
        target_model,
        tokenizer,
        config: Config,
        spec_config: Optional[SpeculativeDecodingConfig] = None,
        logger=None
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.config = config
        self.spec_config = spec_config or SpeculativeDecodingConfig()
        self.logger = logger
        
        # Set models to eval mode
        self.draft_model.eval()
        self.target_model.eval()
        
        # Device setup
        self.device = self.draft_model.device if hasattr(self.draft_model, 'device') else 'cuda'
        
        # Affine verifier (optional)
        self.affine_verifier = None
        if self.spec_config.affine_verification and self.spec_config.affine_model_path:
            try:
                from .affine_alignment import AffineVerifier
                state = torch.load(self.spec_config.affine_model_path, map_location=self.device)
                self.affine_verifier = AffineVerifier.from_state_dict(state).to(self.device)
                # Ensure dtype matches draft model
                if hasattr(self.draft_model, 'dtype'):
                    self.affine_verifier = self.affine_verifier.to(self.draft_model.dtype)
                elif self.draft_model.parameters().__next__().dtype == torch.float16:
                    self.affine_verifier = self.affine_verifier.half()
                self.affine_verifier.eval()
                self._log(f"Loaded affine verifier from {self.spec_config.affine_model_path}")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to load affine verifier: {e}")
                self.affine_verifier = None

        # Generator for sampling
        if config.seed is not None:
            torch.manual_seed(config.seed)
            self.generator = torch.Generator(device=self.device).manual_seed(config.seed)
        else:
            self.generator = None
        
        # Dynamic confidence tracking
        self.current_confidence_threshold = 0.3  # Default value
        self.num_assistant_tokens = self.spec_config.num_assistant_tokens
        
    def _log(self, message: str, level: str = "info"):
        """Log a message."""
        if self.logger and self.spec_config.verbose:
            getattr(self.logger, level)(f"[SpecDecV2] {message}")
    
    def _get_draft_tokens(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple] = None,
        num_tokens: int = 5,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor, Optional[Tuple], Optional[torch.Tensor]]:
        """Generate draft tokens using the draft model."""
        # When not using cache or for the first token, use HF generate
        if not self.spec_config.use_cache or past_key_values is None:
             draft_outputs = self.draft_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=num_tokens,
                # ... (other generate args) ...
                pad_token_id=self.tokenizer.eos_token_id,
             )
             # ... (logic to extract tokens, logits, etc.)
             return draft_tokens, draft_logits, draft_outputs.past_key_values, draft_hiddens
        else:
            # Manual generation loop when cache is present
            return self._get_draft_tokens_manual(
                input_ids, attention_mask, past_key_values, num_tokens
            )

    def _get_draft_tokens_manual(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple] = None,
        num_tokens: int = 5,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor, Optional[Tuple], Optional[torch.Tensor]]:
        """Manual generation loop for when using KV cache."""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Start with the last token when using cache
        current_input_ids = input_ids[:, -1:]
        current_past = past_key_values
        
        draft_tokens = []
        draft_logits = []
        draft_hiddens_list = [] if self.spec_config.affine_verification else None
        
        for _ in range(num_tokens):
            with torch.no_grad():
                outputs = self.draft_model(
                    input_ids=current_input_ids,
                    past_key_values=current_past,
                    use_cache=True,
                    output_hidden_states=self.spec_config.affine_verification,
                )
            
            logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            
            # Sample next token
            if self.config.sampling.do_sample:
                probs = torch.nn.functional.softmax(logits / self.config.sampling.temperature, dim=-1)
                if self.config.sampling.top_k > 0:
                    probs, indices = torch.topk(probs, self.config.sampling.top_k, dim=-1)
                    next_token = indices.gather(-1, torch.multinomial(probs, 1))
                else:
                    next_token = torch.multinomial(probs, 1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            draft_tokens.append(next_token)
            draft_logits.append(logits)
            
            if self.spec_config.affine_verification and outputs.hidden_states is not None:
                # Get the last hidden state from the last layer
                last_hidden = outputs.hidden_states[-1][:, -1, :]
                draft_hiddens_list.append(last_hidden)
            
            # Update for next iteration
            current_input_ids = next_token
            current_past = outputs.past_key_values
        
        # Stack results
        draft_tokens = torch.cat(draft_tokens, dim=1)  # [batch_size, num_tokens]
        draft_logits = torch.stack(draft_logits, dim=1)  # [batch_size, num_tokens, vocab_size]
        
        draft_hiddens = None
        if draft_hiddens_list:
            draft_hiddens = torch.stack(draft_hiddens_list, dim=1)  # [batch_size, num_tokens, hidden_size]
        
        return draft_tokens, draft_logits, current_past, draft_hiddens
    
    def _get_draft_tokens_tree(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple] = None,
        tree_depth: int = 5,
        beam_width: int = 3,
    ) -> Tuple[List[torch.LongTensor], List[torch.FloatTensor], List[Optional[Tuple]], List[Optional[torch.Tensor]]]:
        """Generate draft tokens using tree search (multiple paths).
        
        Returns:
            Lists of (tokens, logits, past_key_values, hiddens) for each path
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Tree structure: List of paths, each path is a dict
        paths = [{
            'tokens': [],
            'logits': [],
            'hiddens': [],
            'past_key_values': past_key_values,
            'score': 0.0,
            'parent_idx': -1
        }]
        
        for depth in range(tree_depth):
            new_paths = []
            
            for path_idx, path in enumerate(paths):
                # Determine input for this iteration
                if depth == 0:
                    # First iteration
                    if past_key_values is None:
                        # No cache, use full input
                        current_input = input_ids
                    else:
                        # With cache, use only last token
                        current_input = input_ids[:, -1:]
                else:
                    # Subsequent iterations - use the last generated token
                    current_input = path['tokens'][-1]
                
                with torch.no_grad():
                    outputs = self.draft_model(
                        input_ids=current_input,
                        past_key_values=path['past_key_values'],
                        use_cache=True,
                        output_hidden_states=self.spec_config.affine_verification,
                    )
                
                logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
                
                # Apply tree-specific temperature
                if self.spec_config.tree_temperature != 1.0:
                    logits = logits / self.spec_config.tree_temperature
                
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Get candidates based on expansion strategy
                if self.spec_config.tree_expansion_strategy == "top_k":
                    # Top-k sampling
                    k = min(beam_width, self.spec_config.tree_top_k)
                    topk_probs, topk_indices = torch.topk(probs, k, dim=-1)
                    candidates = [(topk_indices[:, i:i+1], topk_probs[:, i]) for i in range(k)]
                    
                elif self.spec_config.tree_expansion_strategy == "top_p":
                    # Nucleus (top-p) sampling
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # Find where cumsum exceeds p
                    mask = cumsum_probs <= self.spec_config.tree_top_p
                    # Always include at least one token
                    mask[:, 0] = True
                    
                    # Get valid candidates
                    candidates = []
                    for i in range(min(beam_width, mask.sum().item())):
                        if mask[:, i].item():
                            candidates.append((sorted_indices[:, i:i+1], sorted_probs[:, i]))
                    
                elif self.spec_config.tree_expansion_strategy == "beam":
                    # Standard beam search (deterministic)
                    topk_probs, topk_indices = torch.topk(probs, beam_width, dim=-1)
                    candidates = [(topk_indices[:, i:i+1], topk_probs[:, i]) for i in range(beam_width)]
                    
                elif self.spec_config.tree_expansion_strategy == "diverse_beam":
                    # Diverse beam search - penalize similar tokens
                    candidates = []
                    remaining_probs = probs.clone()
                    selected_indices = []
                    
                    for i in range(beam_width):
                        # Get best from remaining
                        max_prob, max_idx = torch.max(remaining_probs, dim=-1)
                        candidates.append((max_idx.unsqueeze(-1), max_prob))
                        selected_indices.append(max_idx.item())
                        
                        # Penalize similar tokens (simple version - penalize nearby vocab indices)
                        if i < beam_width - 1:
                            penalty_range = 100  # Penalize tokens within this range
                            start_idx = max(0, max_idx.item() - penalty_range)
                            end_idx = min(remaining_probs.shape[-1], max_idx.item() + penalty_range)
                            remaining_probs[:, start_idx:end_idx] *= (1 - self.spec_config.tree_diversity_penalty)
                else:
                    raise ValueError(f"Unknown tree expansion strategy: {self.spec_config.tree_expansion_strategy}")
                
                # Create new paths for each candidate
                for next_token, token_prob in candidates:
                    new_path = {
                        'tokens': path['tokens'] + [next_token],
                        'logits': path['logits'] + [logits],
                        'hiddens': path['hiddens'] + [outputs.hidden_states[-1][:, -1, :]] if self.spec_config.affine_verification else [],
                        'past_key_values': outputs.past_key_values,
                        'score': path['score'] + torch.log(token_prob).item(),  # Log prob for numerical stability
                        'parent_idx': path_idx
                    }
                    new_paths.append(new_path)
            
            # Prune paths based on score (keep top beam_width paths)
            new_paths.sort(key=lambda x: x['score'], reverse=True)
            paths = new_paths[:beam_width]
        
        # Convert paths to output format
        all_tokens = []
        all_logits = []
        all_past = []
        all_hiddens = []
        
        # Get vocab size from the model
        vocab_size = self.draft_model.config.vocab_size
        
        for path in paths:
            # Stack tokens and logits
            tokens = torch.cat(path['tokens'], dim=1) if path['tokens'] else torch.empty(batch_size, 0, dtype=torch.long, device=device)
            logits = torch.stack(path['logits'], dim=1) if path['logits'] else torch.empty(batch_size, 0, vocab_size, device=device)
            hiddens = torch.stack(path['hiddens'], dim=1) if path['hiddens'] else None
            
            all_tokens.append(tokens)
            all_logits.append(logits)
            all_past.append(path['past_key_values'])
            all_hiddens.append(hiddens)
        
        return all_tokens, all_logits, all_past, all_hiddens
    
    def _pre_filter_tokens(
        self,
        draft_tokens: torch.LongTensor,
        draft_logits: torch.FloatTensor,
        draft_hiddens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor, Optional[torch.Tensor], int]:
        """Pre-filter draft tokens using affine verifier before target model verification.
        
        Returns:
            filtered_tokens, filtered_logits, filtered_hiddens, num_filtered
        """
        if not self.spec_config.affine_verification or self.affine_verifier is None or draft_hiddens is None:
            return draft_tokens, draft_logits, draft_hiddens, draft_tokens.size(1)

        B, K, H = draft_hiddens.shape
        
        # Vectorized forward pass
        hidden_flat = draft_hiddens.reshape(B * K, H)
        if self.affine_verifier.affine.weight.dtype != hidden_flat.dtype:
            hidden_flat = hidden_flat.to(self.affine_verifier.affine.weight.dtype)
        
        accept_probs = torch.sigmoid(self.affine_verifier(hidden_flat)).view(B, K)
        
        # Vectorized filtering logic
        mask = accept_probs >= self.spec_config.affine_accept_threshold
        
        # Find the length of the valid prefix for each batch item
        # `(~mask).cumsum(-1) == 0` creates a mask of the initial contiguous accepted block
        keep_lengths = ((~mask).cumsum(-1) == 0).sum(-1)
        
        # If all tokens are rejected for any item, forcefully keep the first one
        # to prevent infinite loops.
        keep_lengths = torch.clamp(keep_lengths, min=1)
        
        num_filtered = keep_lengths.max().item()
        
        # Create a mask for slicing the tensors
        # Shape: (B, K) -> (B, K, 1) -> (B, K, H) for hiddens
        range_tensor = torch.arange(K, device=draft_tokens.device).expand(B, -1)
        slicing_mask = range_tensor < keep_lengths.unsqueeze(-1)
        
        # Create padded tensors to hold the filtered results
        # We'll fill them with the valid tokens and the rest will be padding (or last valid token)
        filtered_tokens = torch.full_like(draft_tokens, self.tokenizer.pad_token_id or 0)[:, :num_filtered]
        filtered_logits = torch.zeros_like(draft_logits)[:, :num_filtered, :]
        filtered_hiddens = torch.zeros_like(draft_hiddens)[:, :num_filtered, :]

        # This part is tricky to fully vectorize without scatter operations, but we can do it per-batch item.
        for b in range(B):
            length = keep_lengths[b].item()
            filtered_tokens[b, :length] = draft_tokens[b, :length]
            filtered_logits[b, :length] = draft_logits[b, :length]
            filtered_hiddens[b, :length] = draft_hiddens[b, :length]
            
            # Pad the remainder of the slice with the last valid token if necessary
            if length < num_filtered:
                filtered_tokens[b, length:] = draft_tokens[b, length-1]
                filtered_logits[b, length:] = draft_logits[b, length-1]
                filtered_hiddens[b, length:] = draft_hiddens[b, length-1]

        return filtered_tokens, filtered_logits, filtered_hiddens, num_filtered
    
    def _prune_tree_paths(
        self,
        paths_tokens: List[torch.LongTensor],
        paths_logits: List[torch.FloatTensor], 
        paths_hiddens: List[Optional[torch.Tensor]],
    ) -> Tuple[List[torch.LongTensor], List[torch.FloatTensor], List[Optional[torch.Tensor]], List[float]]:
        """Prune tree paths using affine verifier.
        
        Returns:
            Filtered paths and their acceptance scores
        """
        if not self.spec_config.affine_verification or self.affine_verifier is None:
            # No pruning, return all paths with score 1.0
            return paths_tokens, paths_logits, paths_hiddens, [1.0] * len(paths_tokens)
        
        kept_tokens = []
        kept_logits = []
        kept_hiddens = []
        path_scores = []
        
        for tokens, logits, hiddens in zip(paths_tokens, paths_logits, paths_hiddens):
            if hiddens is None:
                # No hidden states, keep path with default score
                kept_tokens.append(tokens)
                kept_logits.append(logits)
                kept_hiddens.append(hiddens)
                path_scores.append(1.0)
                continue
            
            # Compute acceptance probability for each token in the path
            B, K, H = hiddens.shape
            hidden_flat = hiddens.reshape(B * K, H)
            if self.affine_verifier.affine.weight.dtype != hidden_flat.dtype:
                hidden_flat = hidden_flat.to(self.affine_verifier.affine.weight.dtype)
            
            accept_probs = torch.sigmoid(self.affine_verifier(hidden_flat)).view(B, K)
            
            # Path score is the minimum acceptance probability (weakest link)
            # or average probability along the path
            path_score = accept_probs.mean().item()  # Can also use min()
            
            # Keep path if score exceeds pruning threshold
            if path_score >= self.spec_config.pruning_threshold:
                kept_tokens.append(tokens)
                kept_logits.append(logits)
                kept_hiddens.append(hiddens)
                path_scores.append(path_score)
        
        # If no paths survive, keep the best one
        if not kept_tokens and paths_tokens:
            # Recompute scores and keep the best
            all_scores = []
            for tokens, logits, hiddens in zip(paths_tokens, paths_logits, paths_hiddens):
                if hiddens is not None:
                    B, K, H = hiddens.shape
                    hidden_flat = hiddens.reshape(B * K, H)
                    if self.affine_verifier.affine.weight.dtype != hidden_flat.dtype:
                        hidden_flat = hidden_flat.to(self.affine_verifier.affine.weight.dtype)
                    accept_probs = torch.sigmoid(self.affine_verifier(hidden_flat)).view(B, K)
                    score = accept_probs.mean().item()
                else:
                    score = 0.0
                all_scores.append(score)
            
            best_idx = max(range(len(all_scores)), key=lambda i: all_scores[i])
            kept_tokens.append(paths_tokens[best_idx])
            kept_logits.append(paths_logits[best_idx])
            kept_hiddens.append(paths_hiddens[best_idx])
            path_scores.append(all_scores[best_idx])
        
        return kept_tokens, kept_logits, kept_hiddens, path_scores
    
    def _verify_and_accept_tokens(
        self,
        draft_tokens: torch.LongTensor,
        draft_logits: torch.FloatTensor,
        target_logits: torch.FloatTensor,
        draft_hiddens: Optional[torch.Tensor] = None, # Keep for future use
        temperature: float = 1.0,
    ) -> Tuple[List[torch.LongTensor], int, int, int]:
        """
        Verify draft tokens against target model and accept valid ones using vectorized operations.
        This implementation avoids Python loops for performance.
        """
        batch_size, num_draft, vocab_size = draft_logits.shape
        
        # Apply temperature and get probabilities
        draft_probs = torch.softmax(draft_logits / temperature, dim=-1)
        target_probs = torch.softmax(target_logits / temperature, dim=-1)

        # Get probabilities of the drafted tokens
        # draft_tokens has shape (B, K) -> needs to be (B, K, 1) for gather
        draft_token_probs = torch.gather(draft_probs, 2, draft_tokens.unsqueeze(-1)).squeeze(-1)
        target_token_probs = torch.gather(target_probs, 2, draft_tokens.unsqueeze(-1)).squeeze(-1)

        # Rejection sampling (vectorized)
        # Add a small epsilon to avoid division by zero
        acceptance_probs = torch.clamp(target_token_probs / (draft_token_probs + 1e-10), max=1.0)
        
        # Sample random numbers for all tokens at once
        u = torch.rand_like(acceptance_probs)
        
        # Create a mask of accepted tokens
        accepted_mask = u < acceptance_probs

        # Note: For now, we are not implementing the affine rescue in this vectorized version
        # as the primary goal is to match the performance of HF's core rejection sampling.
        num_accepted_affine = 0
        num_accepted_semantic = 0

        # Process each item in the batch
        final_accepted_tokens = []
        total_accepted_count = 0

        for b in range(batch_size):
            # Find the first rejected token's index
            # `~accepted_mask[b]` gives True for rejected tokens.
            # `cumsum` will be 0 until the first True, then >= 1.
            # `(cumsum == 0).sum()` counts the number of initial accepted tokens.
            num_accepted = ((~accepted_mask[b]).cumsum(0) == 0).sum().item()
            total_accepted_count += num_accepted

            # Get the tokens that were accepted
            accepted_prefix = draft_tokens[b, :num_accepted]

            # If not all draft tokens were accepted, we need to sample one more token
            if num_accepted < num_draft:
                # Get the distributions for the first rejected position
                q = draft_probs[b, num_accepted]
                p = target_probs[b, num_accepted]
                
                # Resample from the adjusted distribution `max(0, p-q)`
                resample_dist = torch.clamp(p - q, min=0)
                resample_dist_sum = resample_dist.sum()

                if resample_dist_sum > 1e-6: # Check if the distribution is valid
                    resample_dist /= resample_dist_sum
                    resampled_token = torch.multinomial(resample_dist, num_samples=1)
                else:
                    # Fallback to sampling from the target distribution directly
                    resampled_token = torch.multinomial(p, num_samples=1)
                
                # Append the resampled token to the accepted ones
                final_tokens_for_batch = torch.cat([accepted_prefix, resampled_token])
            else:
                # All draft tokens were accepted, no resampling needed
                final_tokens_for_batch = accepted_prefix

            final_accepted_tokens.append(final_tokens_for_batch)
            
        return final_accepted_tokens, total_accepted_count, num_accepted_semantic, num_accepted_affine
    
    def _verify_tree_paths(
        self,
        paths_tokens: List[torch.LongTensor],
        paths_logits: List[torch.FloatTensor],
        target_logits: torch.FloatTensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.LongTensor, int, int]:
        """Verify multiple tree paths and select the best one.
        
        Returns:
            accepted_tokens: Best path tokens
            num_accepted: Number of accepted tokens
            path_idx: Index of selected path
        """
        best_path_idx = -1
        best_num_accepted = 0
        best_accepted_tokens = None
        
        # Try each path
        for path_idx, (draft_tokens, draft_logits) in enumerate(zip(paths_tokens, paths_logits)):
            # Verify this path using standard verification
            accepted_tokens_list, num_accepted, _, _ = self._verify_and_accept_tokens(
                draft_tokens, draft_logits, target_logits, temperature
            )
            
            # Track the best path (most accepted tokens)
            if num_accepted > best_num_accepted:
                best_num_accepted = num_accepted
                best_accepted_tokens = accepted_tokens_list[0]  # Assuming batch_size=1 for simplicity
                best_path_idx = path_idx
        
        # If no tokens accepted from any path, fallback to sampling from target
        if best_accepted_tokens is None:
            batch_size = target_logits.shape[0]
            if self.config.sampling.do_sample:
                last_probs = F.softmax(target_logits[:, 0] / temperature, dim=-1)
                best_accepted_tokens = torch.multinomial(last_probs, num_samples=1).squeeze(-1)
            else:
                best_accepted_tokens = torch.argmax(target_logits[:, 0], dim=-1)
            best_accepted_tokens = best_accepted_tokens.unsqueeze(-1)
            best_num_accepted = 1
            best_path_idx = 0
        
        return best_accepted_tokens, best_num_accepted, best_path_idx
    
    def _update_confidence_threshold(self, acceptance_rate: float):
        """Update confidence threshold based on acceptance rate."""
        if self.spec_config.dynamic_adjustment:
            return
        
        # Adjust threshold based on acceptance rate
        if acceptance_rate < 0.3:
            # Too many rejections, increase threshold (be more conservative)
            self.current_confidence_threshold = min(
                0.95,
                self.current_confidence_threshold * self.spec_config.confidence_boost
            )
            self.num_assistant_tokens = max(
                self.spec_config.min_assistant_tokens,
                self.num_assistant_tokens - 1
            )
        elif acceptance_rate > 0.8:
            # High acceptance, decrease threshold (be more aggressive)
            self.current_confidence_threshold = max(
                0.1,
                self.current_confidence_threshold * self.spec_config.confidence_decay
            )
            self.num_assistant_tokens = min(
                self.spec_config.max_assistant_tokens,
                self.num_assistant_tokens + 1
            )
    
    def _trim_kv_cache(self, past_key_values: Optional[Tuple], num_to_keep: int):
        """Trims a legacy tuple-based KV cache to a specified length."""
        if past_key_values is None:
            return None
        
        current_len = past_key_values[0][0].shape[2]
        if num_to_keep >= current_len:
            return past_key_values

        trimmed_cache = []
        for key, value in past_key_values:
            trimmed_cache.append((key[..., :num_to_keep, :], value[..., :num_to_keep, :]))
        return tuple(trimmed_cache)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        max_new_tokens: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text using optimized speculative decoding."""
        self._log(f"Starting generation with max_new_tokens={max_new_tokens}")
        
        # Ensure attention_mask is not None, create if it is
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize
        generated_ids = input_ids
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_token_id
        
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Initialize caches as None, they will be populated on the first model call
        draft_past_key_values = None
        target_past_key_values = None
        
        # Tracking
        total_drafted = 0
        total_accepted = 0
        total_affine = 0
        num_iterations = 0
        
        # Progress bar
        pbar = tqdm(total=max_new_tokens, desc="Generating", disable=not self.spec_config.verbose)
        
        start_time = time.time()
        draft_time = 0
        verify_time = 0
        
        while generated_ids.shape[1] - input_ids.shape[1] < max_new_tokens and not finished.all():
            num_iterations += 1
            
            # --- Start: Inlined _get_draft_tokens_manual ---
            draft_start = time.time()
            num_draft = self.spec_config.num_assistant_tokens
            
            draft_tokens = []
            draft_logits = []
            draft_hiddens_list = []
            
            current_input_ids = generated_ids if draft_past_key_values is None else generated_ids[:, -1:]
            current_attention_mask = attention_mask
            
            temp_draft_cache = draft_past_key_values

            for _ in range(num_draft):
                draft_outputs = self.draft_model(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=temp_draft_cache,
                    use_cache=True,
                    output_hidden_states=self.spec_config.affine_verification,
                )
                
                temp_draft_cache = draft_outputs.past_key_values # This is now a DynamicCache
                
                next_token_logits = draft_outputs.logits[:, -1, :]
                next_token = sample_from_logits(next_token_logits, **self.config.sampling.to_dict())
                
                draft_tokens.append(next_token)
                draft_logits.append(next_token_logits)
                if self.spec_config.affine_verification:
                    draft_hiddens_list.append(draft_outputs.hidden_states[-1][:, -1, :])

                current_input_ids = next_token.unsqueeze(-1)
                if current_attention_mask is not None:
                     current_attention_mask = torch.cat([current_attention_mask, torch.ones_like(current_input_ids)], dim=1)

            draft_tokens = torch.cat(draft_tokens, dim=1)
            draft_logits = torch.stack(draft_logits, dim=1)
            draft_hiddens = torch.stack(draft_hiddens_list, dim=1) if self.spec_config.affine_verification else None
            draft_time += time.time() - draft_start
            # --- End: Inlined _get_draft_tokens_manual ---

            # Pre-filter with affine verifier if enabled
            if self.spec_config.affine_verification and self.affine_verifier is not None and draft_hiddens is not None:
                filtered_indices = self._pre_filter_tokens(draft_hiddens, threshold=self.spec_config.affine_accept_threshold)
                if filtered_indices is not None and len(filtered_indices) > 0:
                    draft_tokens = draft_tokens[:, filtered_indices]
                    draft_logits = draft_logits[:, filtered_indices]
                    if draft_hiddens is not None:
                        draft_hiddens = draft_hiddens[:, filtered_indices]
                    num_draft = len(filtered_indices)
                else:
                    # No tokens passed pre-filtering
                    if self.spec_config.verbose:
                        self.logger.info("No tokens passed pre-filtering")
                    continue
            
            # When standard decoding with cache, get fresh target cache
            if self.spec_config.use_cache and target_past_key_values is not None:
                # For incremental generation, we need to prepare target model's cache
                # First, run target model on the current generated tokens to get proper cache
                with torch.no_grad():
                    target_prep = self.target_model(
                        input_ids=generated_ids[:, -1:],  # Just last token
                        past_key_values=target_past_key_values,
                        use_cache=True,
                    )
                    target_past_key_values = target_prep.past_key_values
            
            # Step 2: Get target model predictions
            verify_start = time.time()
            
            # Prepare input for target model
            if num_draft > 0:
                if target_past_key_values is not None:
                    # When using cache, only pass the draft tokens
                    target_input = draft_tokens
                else:
                    # Without cache, pass the full sequence
                    target_input = torch.cat([generated_ids, draft_tokens], dim=1)
                
                with torch.no_grad():
                    target_outputs = self.target_model(
                        input_ids=target_input,
                        attention_mask=torch.cat([attention_mask, torch.ones(batch_size, num_draft, device=device)], dim=1) if target_past_key_values is None else None,
                        past_key_values=target_past_key_values,
                        use_cache=self.spec_config.use_cache,
                    )
                
                target_logits = target_outputs.logits
                verify_time += time.time() - verify_start
                
                # Step 3: Verify and accept tokens
                accepted_tokens_list, num_accepted, num_accepted_semantic, num_accepted_affine = self._verify_and_accept_tokens(
                    draft_tokens, draft_logits, target_logits, self.config.sampling.temperature
                )
                accepted_tokens = accepted_tokens_list[0]  # Single batch for now
                
                # Update statistics
                total_drafted += num_draft
                total_accepted += num_accepted
                
                # Step 4: Update KV Caches
                if self.spec_config.use_cache:
                    if num_accepted > 0:
                        # The draft model's cache has seen `num_draft` tokens. We only keep the part corresponding to the accepted ones.
                        draft_past_key_values = self._trim_kv_cache(new_draft_past, num_accepted)

                        # The target model has seen the original sequence plus `num_draft` tokens.
                        # We need to trim its cache back to the length of the original sequence plus the `num_accepted` tokens.
                        original_len = generated_ids.shape[1]
                        target_past_key_values = self._trim_kv_cache(target_outputs.past_key_values, original_len + num_accepted)
                    else:
                        # If no tokens are accepted, the draft cache is discarded.
                        draft_past_key_values = None 
                        # The target cache should be what it was *before* this failed verification step.
                        # We don't update target_past_key_values, so it correctly preserves its previous state.
            
            # Update generated sequence
            if accepted_tokens.dim() == 1:
                accepted_tokens = accepted_tokens.unsqueeze(0)  # Make it 2D
            generated_ids = torch.cat([generated_ids, accepted_tokens], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones(batch_size, accepted_tokens.shape[1], device=device)], dim=1)
            
            # Check for EOS
            if eos_token_id is not None:
                finished |= (accepted_tokens == eos_token_id).any(dim=1)
            
            # Update progress
            pbar.update(accepted_tokens.shape[1])
            
            self._log(f"Iteration {num_iterations}: drafted {num_draft}, accepted {num_accepted}")
        
        pbar.close()
        
        # Final statistics
        total_time = time.time() - start_time
        total_tokens = generated_ids.shape[1] - input_ids.shape[1]
        
        if self.spec_config.verbose and self.logger:
            self.logger.info("\n" + "="*50)
            self.logger.info("Final Generation Statistics:")
            self.logger.info("="*50)
            self.logger.info(f"  Total tokens generated: {total_tokens}")
            self.logger.info(f"  Total drafted: {total_drafted}")
            self.logger.info(f"  Total accepted: {total_accepted}")
            if self.spec_config.affine_verification:
                self.logger.info(f"  Affine rescues: {total_affine}")
            
            # Core metrics
            avg_acceptance_rate = total_accepted/total_drafted if total_drafted > 0 else 0.0
            tokens_per_second = total_tokens/total_time if total_time > 0 else 0.0
            draft_efficiency = total_accepted/total_drafted if total_drafted > 0 else 0.0
            
            self.logger.info(f"  Average acceptance rate: {avg_acceptance_rate:.1%}")
            self.logger.info(f"  Draft efficiency: {draft_efficiency:.1%}")
            self.logger.info(f"  Tokens/second: {tokens_per_second:.1f}")
            self.logger.info(f"  Total time: {total_time:.2f}s")
            self.logger.info(f"  Draft time: {draft_time:.2f}s ({draft_time/total_time:.1%})")
            self.logger.info(f"  Verify time: {verify_time:.2f}s ({verify_time/total_time:.1%})")
            
            # Advanced metrics
            avg_drafted_per_iter = total_drafted/num_iterations if num_iterations > 0 else 0.0
            avg_accepted_per_iter = total_accepted/num_iterations if num_iterations > 0 else 0.0
            speedup_theoretical = 1 + avg_accepted_per_iter  # Theoretical speedup
            
            self.logger.info(f"  Total iterations: {num_iterations}")
            self.logger.info(f"  Avg drafted per iteration: {avg_drafted_per_iter:.1f}")
            self.logger.info(f"  Avg accepted per iteration: {avg_accepted_per_iter:.1f}")
            self.logger.info(f"  Theoretical speedup: {speedup_theoretical:.2f}x")
            
            # Efficiency metrics
            if self.spec_config.affine_verification:
                affine_rescue_rate = total_affine/total_drafted if total_drafted > 0 else 0.0
                self.logger.info(f"  Affine rescue rate: {affine_rescue_rate:.1%}")
            
            self.logger.info(f"  Final confidence threshold: {self.current_confidence_threshold:.3f}")
            self.logger.info(f"  Final assistant tokens: {self.num_assistant_tokens}")
        
        # Return comprehensive statistics
        stats = {
            "total_time": total_time,
            "draft_time": draft_time,
            "verify_time": verify_time,
            "total_tokens": total_tokens,
            "total_drafted": total_drafted,
            "total_accepted": total_accepted,
            "total_affine_rescues": total_affine,
            "num_iterations": num_iterations,
            "acceptance_rate": total_accepted/total_drafted if total_drafted > 0 else 0.0,
            "draft_efficiency": total_accepted/total_drafted if total_drafted > 0 else 0.0,
            "tokens_per_second": total_tokens/total_time if total_time > 0 else 0.0,
            "avg_drafted_per_iter": total_drafted/num_iterations if num_iterations > 0 else 0.0,
            "avg_accepted_per_iter": total_accepted/num_iterations if num_iterations > 0 else 0.0,
            "theoretical_speedup": 1 + (total_accepted/num_iterations if num_iterations > 0 else 0),
            "draft_time_ratio": draft_time/total_time if total_time > 0 else 0.0,
            "verify_time_ratio": verify_time/total_time if total_time > 0 else 0.0,
            "affine_rescue_rate": total_affine/total_drafted if total_drafted > 0 else 0.0,
            "final_confidence_threshold": self.current_confidence_threshold,
            "final_assistant_tokens": self.num_assistant_tokens,
        }
        
        return {
            "sequences": generated_ids,
            "stats": stats
        } 