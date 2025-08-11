#!/usr/bin/env python3
"""Test KV cache compatibility with current transformers version."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_cache():
    print("Testing KV cache with transformers...")
    
    # Load a small model for testing
    model_name = "meta-llama/Llama-2-7b-hf"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map="cuda:0"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Test input
    text = "The future of AI is"
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
    
    # Test 1: Without cache
    print("\n1. Testing without cache...")
    with torch.no_grad():
        outputs = model(**inputs, use_cache=False)
        print(f"   Output shape: {outputs.logits.shape}")
    
    # Test 2: With cache (first pass)
    print("\n2. Testing with cache (first pass)...")
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
        print(f"   Output shape: {outputs.logits.shape}")
        print(f"   Past key values type: {type(outputs.past_key_values)}")
        if hasattr(outputs.past_key_values, 'get_seq_length'):
            print(f"   Past key values has get_seq_length method")
        else:
            print(f"   Past key values is a tuple/list of length: {len(outputs.past_key_values)}")
    
    # Test 3: With cache (incremental)
    print("\n3. Testing incremental generation...")
    next_token = torch.tensor([[1234]]).to("cuda:0")  # Random token
    with torch.no_grad():
        outputs2 = model(
            input_ids=next_token,
            past_key_values=outputs.past_key_values,
            use_cache=True
        )
        print(f"   Output shape: {outputs2.logits.shape}")
    
    # Test 4: Using generate
    print("\n4. Testing generate with cache...")
    output_ids = model.generate(
        **inputs,
        max_new_tokens=5,
        use_cache=True,
        do_sample=False
    )
    print(f"   Generated shape: {output_ids.shape}")
    print(f"   Generated text: {tokenizer.decode(output_ids[0])}")

if __name__ == "__main__":
    test_cache() 