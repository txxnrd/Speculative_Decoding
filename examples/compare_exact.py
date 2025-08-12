#!/usr/bin/env python3
"""Compare exact outputs from draft and target models."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_path = "/hdd1/taeyun/Llama-3.1-8B-Instruct"
    
    print("Loading models...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Test with exact same input as basic_usage.py
    prompt = "The future of artificial intelligence is"
    input_ids = torch.tensor([[791, 3938, 315, 21075, 11478, 374]], device='cuda:0')
    
    print(f"\nPrompt: {prompt}")
    print(f"Input IDs: {input_ids}")
    print(f"Tokens: {[tokenizer.decode([t]) for t in input_ids[0].tolist()]}")
    
    with torch.no_grad():
        # Test 1: Direct forward
        outputs = model(input_ids, use_cache=False)
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        
        # Get top 10
        top10_probs, top10_indices = torch.topk(probs, 10)
        
        print("\nTop-10 predictions:")
        for i in range(10):
            token_id = top10_indices[i].item()
            prob = top10_probs[i].item()
            token = tokenizer.decode([token_id])
            print(f"  {i+1}. {token_id:6d}: '{token}' (p={prob:.4f})")
        
        # Test 2: With generation config
        print("\n=== Generation Config ===")
        print(model.generation_config)
        
        # Test 3: Greedy generation
        print("\n=== Greedy Generation ===")
        gen_output = model.generate(
            input_ids,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(gen_output[0])
        print(f"Generated: {generated_text}")

if __name__ == "__main__":
    main() 