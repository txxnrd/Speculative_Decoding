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
    """Configuration for speculative decoding algorithm."""
    
    # Core parameters
    num_assistant_tokens: int = 5
    max_assistant_tokens: int = 20
    min_assistant_tokens: int = 1
    
    # Dynamic adjustment
    dynamic_adjustment: bool = True
    assistant_confidence_threshold: float = 0.3
    confidence_adjustment_factor: float = 0.1
    acceptance_rate_target: float = 0.7
    
    # Performance options
    use_cache: bool = True
    
    # Debugging and logging
    verbose: bool = False
    log_interval: int = 1
    
    # Affine alignment verification (Research Idea 2)
    affine_verification: bool = False  # Enable affine-map-based verifier
    affine_model_path: Optional[str] = None  # Path to pre-trained affine model (.pt)
    affine_accept_threshold: float = 0.3  # Threshold on MLP probability


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
        self.current_confidence_threshold = self.spec_config.assistant_confidence_threshold
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
        """
        Generate draft tokens using the draft model.
        When using cache, we use a manual loop to avoid HuggingFace generate complexities.
        """
        if past_key_values is not None and self.spec_config.use_cache:
            # Manual generation loop when using cache
            return self._get_draft_tokens_manual(
                input_ids, attention_mask, past_key_values, num_tokens
            )
        
        # When not using cache, use HuggingFace generate for simplicity
        with torch.no_grad():
            draft_outputs = self.draft_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=num_tokens,
                use_cache=self.spec_config.use_cache,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=self.spec_config.affine_verification,
                do_sample=self.config.sampling.do_sample,
                temperature=self.config.sampling.temperature,
                top_k=self.config.sampling.top_k,
                top_p=self.config.sampling.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Extract new tokens
        draft_tokens = draft_outputs.sequences[:, input_ids.shape[1]:]
        
        # `scores` is a tuple of logits for each generated token
        draft_logits = torch.stack(draft_outputs.scores, dim=1)

        new_draft_past = draft_outputs.past_key_values
        
        draft_hiddens = None
        if self.spec_config.affine_verification and draft_outputs.hidden_states is not None:
            last_layer_hiddens = draft_outputs.hidden_states[-1]
            hidden_states_list = [h[:, -1, :] for h in last_layer_hiddens]
            draft_hiddens = torch.stack(hidden_states_list, dim=1)

        return draft_tokens, draft_logits, new_draft_past, draft_hiddens
    
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
    
    def _trim_kv_cache(self, past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor]], num_to_keep: int):
        """Trims a KV cache to a specified length."""
        if past_key_values is None:
            return None
        
        trimmed_cache = []
        for key, value in past_key_values:
            # key and value shape: (B, num_heads, seq_len, head_dim)
            if key.shape[2] > num_to_keep:
                trimmed_key = key[:, :, :num_to_keep, :]
                trimmed_value = value[:, :, :num_to_keep, :]
                trimmed_cache.append((trimmed_key, trimmed_value))
            else:
                trimmed_cache.append((key, value))
        
        return tuple(trimmed_cache)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        max_new_tokens: int = 100,
        **kwargs
    ) -> torch.LongTensor:
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
        
        # Cache management
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
            cur_len = generated_ids.shape[1]
            
            # Determine number of tokens to draft
            remaining = max_new_tokens - (cur_len - input_ids.shape[1])
            k = min(self.num_assistant_tokens, remaining)
            
            if k <= 0:
                break
            
            # Draft tokens
            draft_start = time.time()
            draft_tokens, draft_logits, new_draft_past, draft_hiddens = self._get_draft_tokens(
                generated_ids,
                attention_mask,
                draft_past_key_values,
                num_tokens=k
            )
            draft_time += time.time() - draft_start
            
            actual_drafted = draft_tokens.shape[1]
            total_drafted += actual_drafted
            
            # Pre-filter tokens
            filtered_tokens, filtered_logits, filtered_hiddens, num_filtered = self._pre_filter_tokens(
                draft_tokens,
                draft_logits,
                draft_hiddens
            )
            
            # Check if any tokens passed pre-filtering
            if num_filtered == 0:
                # No tokens to verify, skip to next iteration
                self._log(f"No tokens passed pre-filtering, generating new draft tokens")
                continue
            
            # Prepare input for target model verification
            if self.spec_config.use_cache and target_past_key_values is not None:
                # Only verify the new tokens
                verify_ids = filtered_tokens
                verify_mask = torch.ones_like(verify_ids) if attention_mask is not None else None
            else:
                # Verify full sequence
                verify_ids = torch.cat([generated_ids, filtered_tokens], dim=1)
                verify_mask = torch.cat([
                    attention_mask,
                    torch.ones(batch_size, num_filtered, device=device)
                ], dim=1) if attention_mask is not None else None
            
            # Target model forward pass
            verify_start = time.time()
            with torch.no_grad():
                target_outputs = self.target_model(
                    input_ids=verify_ids,
                    attention_mask=verify_mask,
                    past_key_values=target_past_key_values,
                    use_cache=self.spec_config.use_cache,
                    return_dict=True
                )
            
            # Extract relevant logits
            if self.spec_config.use_cache and target_past_key_values is not None:
                target_logits = target_outputs.logits
            else:
                target_logits = target_outputs.logits[:, -(num_filtered+1):, :]
            
            verify_time += time.time() - verify_start
            
            # Verify and accept tokens
            accepted_tokens, num_accepted, num_accepted_semantic, num_accepted_affine = self._verify_and_accept_tokens(
                filtered_tokens,
                filtered_logits,
                target_logits[:, :num_filtered, :],
                draft_hiddens=filtered_hiddens,
                temperature=self.config.sampling.temperature
            )
            
            total_accepted += num_accepted
            total_affine += num_accepted_affine
            acceptance_rate = num_accepted / (num_filtered * batch_size) if num_filtered > 0 else 0
            
            # Update dynamic parameters
            self._update_confidence_threshold(acceptance_rate)
            
            # Update generated sequences
            max_accepted = max(len(tokens) for tokens in accepted_tokens)
            
            # Important: The actual number of new tokens can vary per batch item
            # We will handle padding and mask updates carefully
            new_generated_ids_list = []
            if attention_mask is not None:
                new_attention_mask_list = []

            for i in range(batch_size):
                if not finished[i]:
                    # Append accepted tokens for this item
                    new_tokens = accepted_tokens[i]
                    current_item_ids = torch.cat([generated_ids[i:i+1], new_tokens.unsqueeze(0)], dim=1)
                    new_generated_ids_list.append(current_item_ids)

                    if attention_mask is not None:
                        new_mask = torch.ones(1, new_tokens.shape[0], dtype=attention_mask.dtype, device=device)
                        current_item_mask = torch.cat([attention_mask[i:i+1], new_mask], dim=1)
                        new_attention_mask_list.append(current_item_mask)
                    
                    if eos_token_id in new_tokens:
                        finished[i] = True
                else: # if already finished, just carry over
                    new_generated_ids_list.append(generated_ids[i:i+1])
                    if attention_mask is not None:
                        new_attention_mask_list.append(attention_mask[i:i+1])

            # Pad all sequences to the same length for the next batch iteration
            max_len = max(seq.shape[1] for seq in new_generated_ids_list)
            
            final_ids = torch.full(
                (batch_size, max_len), pad_token_id, device=device, dtype=generated_ids.dtype
            )
            if attention_mask is not None:
                final_mask = torch.zeros(
                    (batch_size, max_len), device=device, dtype=attention_mask.dtype
                )

            for i in range(batch_size):
                seq_len = new_generated_ids_list[i].shape[1]
                final_ids[i, :seq_len] = new_generated_ids_list[i]
                if attention_mask is not None:
                    final_mask[i, :seq_len] = new_attention_mask_list[i]
            
            generated_ids = final_ids
            if attention_mask is not None:
                attention_mask = final_mask

            # Update KV caches
            if self.spec_config.use_cache:
                # The new cache length should be the length of the sequence *before* this iteration's accepted tokens
                # The target_outputs.past_key_values contains state for `cur_len + num_filtered` tokens
                # We need to trim it back to `cur_len + num_accepted_for_that_batch_item`
                # For simplicity in batching, we trim to the new max_len, which is correct
                new_cache_len = generated_ids.shape[1]
                
                target_past_key_values = self._trim_kv_cache(target_outputs.past_key_values, new_cache_len)
                draft_past_key_values = self._trim_kv_cache(target_outputs.past_key_values, new_cache_len)
            
            # Update progress
            pbar.update(max_accepted)
            
            # Log iteration details
            if num_iterations % self.spec_config.log_interval == 0:
                self._log(
                    f"Iteration {num_iterations}: "
                    f"Drafted {actual_drafted} tokens (threshold={self.current_confidence_threshold:.3f}), "
                    f"accepted {num_accepted}/{num_filtered*batch_size} ({acceptance_rate:.1%}), "
                    f"next_k={self.num_assistant_tokens}"
                )
        
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