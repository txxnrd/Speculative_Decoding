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
        """Generate draft tokens using the draft model with KV cache support.
        Returns:
            draft_tokens (Tensor): (B, K)
            draft_logits (Tensor): (B, K, vocab)
            draft_past_key_values (tuple): cache
            draft_hiddens (Tensor|None): (B, K, hidden_size) if affine verifier enabled
        """
        draft_ids = []
        draft_logits = []
        draft_hiddens = []
        current_ids = input_ids
        current_mask = attention_mask
        draft_past_key_values = past_key_values
        
        for i in range(num_tokens):
            # Forward pass with cache
            with torch.no_grad():
                outputs = self.draft_model(
                    input_ids=current_ids if draft_past_key_values is None else current_ids[:, -1:],
                    attention_mask=current_mask,
                    past_key_values=draft_past_key_values,
                    use_cache=self.spec_config.use_cache,
                    return_dict=True,
                    output_hidden_states=self.spec_config.affine_verification
                )
            
            # Get logits and sample
            logits = outputs.logits[:, -1, :]
            draft_logits.append(logits)
            if self.spec_config.affine_verification and outputs.hidden_states is not None:
                last_hidden = outputs.hidden_states[-1][:, -1, :]
                draft_hiddens.append(last_hidden)
            
            # Apply sampling
            if self.config.sampling.do_sample:
                next_token = sample_from_logits(
                    logits,
                    temperature=self.config.sampling.temperature,
                    top_k=self.config.sampling.top_k,
                    top_p=self.config.sampling.top_p
                )
            else:
                next_token = torch.argmax(logits, dim=-1)
            
            draft_ids.append(next_token)
            
            # Update for next iteration
            current_ids = torch.cat([current_ids, next_token.unsqueeze(1)], dim=1)
            if current_mask is not None:
                current_mask = torch.cat([current_mask, torch.ones_like(next_token).unsqueeze(1)], dim=1)
            
            # Update cache
            if self.spec_config.use_cache:
                draft_past_key_values = outputs.past_key_values
            
            # Early stopping based on confidence
            if self.spec_config.dynamic_adjustment:
                confidence = torch.max(F.softmax(logits, dim=-1), dim=-1)[0].mean().item()
                if confidence < self.current_confidence_threshold:
                    break
        
        draft_tokens = torch.stack(draft_ids, dim=1)
        draft_logits = torch.stack(draft_logits, dim=1)
        if self.spec_config.affine_verification and draft_hiddens:
            draft_hiddens = torch.stack(draft_hiddens, dim=1)  # (B,K,H)
        else:
            draft_hiddens = None
        
        return draft_tokens, draft_logits, draft_past_key_values, draft_hiddens
    
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
            # No filtering - return all tokens
            return draft_tokens, draft_logits, draft_hiddens, draft_tokens.size(1)
        
        batch_size, num_draft = draft_tokens.shape
        filtered_tokens_list = []
        filtered_logits_list = []
        filtered_hiddens_list = []
        
        for b in range(batch_size):
            batch_filtered_tokens = []
            batch_filtered_logits = []
            batch_filtered_hiddens = []
            
            for i in range(num_draft):
                # Get accept probability from affine verifier
                draft_vec = draft_hiddens[b, i]
                # Ensure dtype compatibility
                if self.affine_verifier.affine.weight.dtype != draft_vec.dtype:
                    draft_vec = draft_vec.to(self.affine_verifier.affine.weight.dtype)
                accept_prob = torch.sigmoid(self.affine_verifier(draft_vec.unsqueeze(0))).item()
                
                # Debug: log the first few probabilities and hidden states
                if b == 0 and i < 3:
                    hidden_norm = torch.norm(draft_vec).item()
                    hidden_mean = torch.mean(draft_vec).item()
                    hidden_std = torch.std(draft_vec).item()
                    draft_token_id = draft_tokens[b, i].item()
                    self._log(f"Token {i}: accept_prob={accept_prob:.4f}, threshold={self.spec_config.affine_accept_threshold}")
                    self._log(f"  Hidden stats: norm={hidden_norm:.4f}, mean={hidden_mean:.6f}, std={hidden_std:.4f}")
                    self._log(f"  Draft token ID: {draft_token_id}")
                
                if accept_prob >= self.spec_config.affine_accept_threshold:
                    # Keep this token for target verification
                    batch_filtered_tokens.append(draft_tokens[b, i])
                    batch_filtered_logits.append(draft_logits[b, i])
                    batch_filtered_hiddens.append(draft_hiddens[b, i])
                else:
                    # Stop here - don't include this token or any subsequent ones
                    if b == 0 and i == 0:
                        self._log(f"First token rejected with prob {accept_prob:.4f} < {self.spec_config.affine_accept_threshold}")
                    break
            
            # Convert to tensors (pad if necessary)
            if len(batch_filtered_tokens) > 0:
                filtered_tokens_list.append(torch.stack(batch_filtered_tokens))
                filtered_logits_list.append(torch.stack(batch_filtered_logits))
                filtered_hiddens_list.append(torch.stack(batch_filtered_hiddens))
            else:
                # INFINITE LOOP-PREVENTION:
                # If all draft tokens are filtered out by the verifier, it can cause an infinite
                # loop. To prevent this, we forcefully include the *first* draft token so that
                # there is at least one token to verify, ensuring the generation progresses.
                self._log("All draft tokens were filtered by affine verifier. Forcing first token to prevent infinite loop.")
                filtered_tokens_list.append(draft_tokens[b, 0:1])
                filtered_logits_list.append(draft_logits[b, 0:1])
                filtered_hiddens_list.append(draft_hiddens[b, 0:1])
        
        # Pad to same length for batching
        if len(filtered_tokens_list) > 0:
            max_len = max(len(tokens) for tokens in filtered_tokens_list)
            if max_len > 0:
                # Pad tokens
                padded_tokens = []
                padded_logits = []
                padded_hiddens = []
                
                for i in range(batch_size):
                    tokens = filtered_tokens_list[i]
                    logits = filtered_logits_list[i]
                    hiddens = filtered_hiddens_list[i]
                    
                    if len(tokens) < max_len:
                        # Pad with the last token/logits/hiddens
                        pad_len = max_len - len(tokens)
                        if len(tokens) > 0:
                            last_token = tokens[-1:]
                            last_logits = logits[-1:]
                            last_hiddens = hiddens[-1:]
                        else:
                            # Use original first token as fallback
                            last_token = draft_tokens[i:i+1, 0]
                            last_logits = draft_logits[i:i+1, 0]
                            last_hiddens = draft_hiddens[i:i+1, 0]
                        
                        tokens = torch.cat([tokens] + [last_token] * pad_len)
                        logits = torch.cat([logits] + [last_logits] * pad_len)
                        hiddens = torch.cat([hiddens] + [last_hiddens] * pad_len)
                    
                    padded_tokens.append(tokens)
                    padded_logits.append(logits)
                    padded_hiddens.append(hiddens)
                
                filtered_tokens = torch.stack(padded_tokens)
                filtered_logits = torch.stack(padded_logits)
                filtered_hiddens = torch.stack(padded_hiddens)
                
                return filtered_tokens, filtered_logits, filtered_hiddens, max_len
            else:
                # All tokens were filtered out
                return (
                    torch.empty(batch_size, 0, dtype=draft_tokens.dtype, device=draft_tokens.device),
                    torch.empty(batch_size, 0, draft_logits.size(-1), dtype=draft_logits.dtype, device=draft_logits.device),
                    torch.empty(batch_size, 0, draft_hiddens.size(-1), dtype=draft_hiddens.dtype, device=draft_hiddens.device),
                    0
                )
        else:
            # No filtering applied
            return draft_tokens, draft_logits, draft_hiddens, draft_tokens.size(1)
    
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
            padded_accepted = torch.full(
                (batch_size, max_accepted),
                pad_token_id,
                device=device,
                dtype=generated_ids.dtype
            )
            
            for i in range(batch_size):
                if not finished[i]:
                    new_tokens = accepted_tokens[i]
                    padded_accepted[i, :len(new_tokens)] = new_tokens
                    if eos_token_id in new_tokens:
                        finished[i] = True
            
            generated_ids = torch.cat([generated_ids, padded_accepted], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(batch_size, max_accepted, device=device)
                ], dim=1)
            
            # Update KV caches
            if self.spec_config.use_cache:
                target_past_key_values = target_outputs.past_key_values
                # Reset draft cache since we accepted some tokens
                draft_past_key_values = None
            
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