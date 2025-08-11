#!/usr/bin/env python3
"""Test different cache options in transformers 4.54."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache, DynamicCache

def test_static_cache():
    """Test using StaticCache which is compatible with both old and new formats."""
    print("Testing StaticCache...")
    
    # Load a small model
    model_name = "meta-llama/Llama-2-7b-hf"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create inputs
    text = "The future of AI is"
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
    
    # Option 1: Use StaticCache
    print("\n1. Testing with StaticCache...")
    
    # Get model config info
    config = model.config
    max_cache_len = 1024  # Maximum sequence length to cache
    
    # Create static cache
    past_key_values = StaticCache(
        config=config,
        max_batch_size=1,
        max_cache_len=max_cache_len,
        device=model.device,
        dtype=model.dtype
    )
    
    with torch.no_grad():
        # First pass - full input
        outputs = model(
            **inputs,
            past_key_values=past_key_values,
            use_cache=True
        )
        print(f"   First pass successful!")
        print(f"   Cache type: {type(outputs.past_key_values)}")
        
        # Second pass - incremental
        next_token = torch.tensor([[1234]]).to("cuda:0")
        outputs2 = model(
            input_ids=next_token,
            past_key_values=outputs.past_key_values,
            use_cache=True
        )
        print(f"   Incremental pass successful!")

if __name__ == "__main__":
    test_static_cache() 