#!/usr/bin/env python3
"""Debug script to compare draft and target model outputs directly."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # Load models
    model_path = "/hdd1/taeyun/Llama-3.1-8B-Instruct"
    
    print("Loading draft model...")
    draft_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True
    )
    draft_model.eval()
    
    print("Loading target model...")
    target_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cuda:1",
        trust_remote_code=True
    )
    target_model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Test input
    prompt = "The future of artificial intelligence is"
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    
    print(f"\nPrompt: {prompt}")
    print(f"Input IDs: {input_ids}")
    
    # Test 1: Direct forward pass
    print("\n=== Test 1: Direct Forward Pass ===")
    with torch.no_grad():
        # Draft model
        draft_input = input_ids.to("cuda:0")
        draft_output = draft_model(draft_input, use_cache=False)
        draft_logits = draft_output.logits[0, -1, :]
        draft_probs = torch.softmax(draft_logits, dim=-1)
        draft_top5 = torch.topk(draft_probs, 5)
        
        print("\nDraft model top-5:")
        for i in range(5):
            token_id = draft_top5.indices[i]
            prob = draft_top5.values[i]
            token = tokenizer.decode([token_id])
            print(f"  {token_id}: '{token}' ({prob:.3f})")
        
        # Target model
        target_input = input_ids.to("cuda:1")
        target_output = target_model(target_input, use_cache=False)
        target_logits = target_output.logits[0, -1, :]
        target_probs = torch.softmax(target_logits, dim=-1)
        target_top5 = torch.topk(target_probs, 5)
        
        print("\nTarget model top-5:")
        for i in range(5):
            token_id = target_top5.indices[i]
            prob = target_top5.values[i]
            token = tokenizer.decode([token_id])
            print(f"  {token_id}: '{token}' ({prob:.3f})")
    
    # Test 2: Generation config
    print("\n=== Test 2: Generation Config ===")
    print(f"Draft generation config: {draft_model.generation_config}")
    print(f"Target generation config: {target_model.generation_config}")
    
    # Test 3: Model config differences
    print("\n=== Test 3: Model Config ===")
    print(f"Draft model dtype: {draft_model.dtype}")
    print(f"Target model dtype: {target_model.dtype}")
    print(f"Draft model device: {draft_model.device}")
    print(f"Target model device: {target_model.device}")
    
    # Test 4: Generate with greedy
    print("\n=== Test 4: Greedy Generation ===")
    with torch.no_grad():
        draft_gen = draft_model.generate(
            draft_input,
            max_new_tokens=5,
            do_sample=False,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=torch.ones_like(draft_input)
        )
        
        target_gen = target_model.generate(
            target_input,
            max_new_tokens=5,
            do_sample=False,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=torch.ones_like(target_input)
        )
        
        print(f"Draft generated: {tokenizer.decode(draft_gen[0])}")
        print(f"Target generated: {tokenizer.decode(target_gen[0])}")

if __name__ == "__main__":
    main() 