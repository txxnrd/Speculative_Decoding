"""Sampling utilities for speculative decoding."""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float('Inf')
) -> torch.Tensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        top_k: keep only top k tokens with highest probability
        top_p: keep the top tokens with cumulative probability >= top_p
        filter_value: fill value for removed tokens
        
    Returns:
        Filtered logits
    """
    assert logits.dim() == 2  # batch size x vocabulary size
    
    top_k = min(top_k, logits.size(-1))  # Safety check
    
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
        
    return logits


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    do_sample: bool = True,
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """
    Sample tokens from logits.
    
    Args:
        logits: Logits tensor of shape (batch_size, vocab_size)
        temperature: Sampling temperature
        top_k: Top-k filtering parameter
        top_p: Top-p (nucleus) filtering parameter
        do_sample: Whether to sample or take argmax
        generator: Random generator for reproducibility
        
    Returns:
        Sampled token indices of shape (batch_size,)
    """
    if temperature != 1.0:
        logits = logits / temperature
        
    # Apply top-k and/or top-p filtering
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        
    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)
    
    if do_sample:
        # Sample from the distribution
        next_tokens = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)
    else:
        # Take the argmax
        next_tokens = torch.argmax(probs, dim=-1)
        
    return next_tokens


def sample_multiple_tokens(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    num_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    do_sample: bool = True,
    generator: Optional[torch.Generator] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Tuple] = None,
    use_cache: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample multiple tokens autoregressively.
    
    Args:
        model: The language model
        input_ids: Input token IDs
        num_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering parameter
        top_p: Top-p filtering parameter
        do_sample: Whether to sample or take argmax
        generator: Random generator
        attention_mask: Attention mask
        past_key_values: Past key values for caching
        use_cache: Whether to use KV cache
        
    Returns:
        Tuple of (generated_tokens, all_logits)
    """
    generated_tokens = []
    all_logits = []
    
    current_input_ids = input_ids
    current_past_key_values = past_key_values
    
    for _ in range(num_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=current_input_ids,
                attention_mask=attention_mask,
                past_key_values=current_past_key_values,
                use_cache=use_cache,
                return_dict=True
            )
            
        logits = outputs.logits[:, -1, :]  # Get last token logits
        all_logits.append(logits)
        
        # Sample next token
        next_token = sample_from_logits(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            generator=generator
        )
        
        generated_tokens.append(next_token)
        
        # Update inputs for next iteration
        current_input_ids = next_token.unsqueeze(-1)
        if attention_mask is not None:
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
            ], dim=1)
            
        if use_cache:
            current_past_key_values = outputs.past_key_values
            
    generated_tokens = torch.stack(generated_tokens, dim=1)
    all_logits = torch.stack(all_logits, dim=1)
    
    return generated_tokens, all_logits 