#!/usr/bin/env python3
"""
Test HuggingFace's built-in speculative decoding implementation to compare its performance
against the baseline and our custom implementation.

Usage:
    CUDA_VISIBLE_DEVICES=6,7 python test_hf_speculative.py --num-samples 10
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import time
import torch
from datasets import load_dataset

from speculative_decoding.models import ModelLoader
from speculative_decoding.utils import Config, setup_logger
from speculative_decoding.utils.config import ModelConfig, SamplingConfig

def run_test(model, tokenizer, questions, max_new_tokens, logger, assistant_model=None):
    """Run generation test for either baseline or HF speculative decoding."""
    tag = "HF Speculative" if assistant_model else "Baseline"
    logger.info(f"Running {tag} generation...")
    
    stats = []
    
    for i, q in enumerate(questions):
        input_ids = tokenizer.encode(q, return_tensors="pt").to(model.device)
        
        start_time = time.time()
        with torch.no_grad():
            # Pass assistant_model if it's a speculative run
            generate_kwargs = {"assistant_model": assistant_model} if assistant_model else {}
            
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                **generate_kwargs
            )
        end_time = time.time()
        
        total_time = end_time - start_time
        total_tokens = outputs.size(1) - input_ids.size(1)
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        
        logger.info(f"  {tag} {i+1}: {total_tokens} tokens in {total_time:.2f}s ({tokens_per_second:.1f} tok/s)")
        stats.append({"tps": tokens_per_second})
        
    return stats

def avg(lst, key):
    return sum(d[key] for d in lst) / len(lst) if lst else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--max-new", type=int, default=100)
    args = parser.parse_args()

    logger = setup_logger()

    logger.info("Loading GSM8K dataset...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    questions = [ds[i]["question"].strip() + " Answer:" for i in range(min(args.num_samples, len(ds)))]

    cfg = Config(
        draft_model=ModelConfig(model_path="/raid/junha/models/Llama-3.1-8B-Instruct", device="auto"),
        target_model=ModelConfig(model_path="/raid/junha/models/Llama-3.1-70B-Instruct", device="auto"),
        sampling=SamplingConfig(temperature=0.7, top_k=50, top_p=0.9, do_sample=True),
    )

    logger.info("Loading models...")
    loader = ModelLoader(logger=logger)
    draft_model, target_model, tokenizer = loader.load_draft_and_target_models(cfg.draft_model, cfg.target_model)

    # Run Baseline
    base_stats = run_test(target_model, tokenizer, questions, args.max_new, logger)
    
    # Run HF Speculative Decoding
    hf_spec_stats = run_test(target_model, tokenizer, questions, args.max_new, logger, assistant_model=draft_model)

    logger.info("\n===== HuggingFace Speculative Decoding Results =====")
    logger.info(f"  - Baseline Average      : {avg(base_stats, 'tps'):.1f} tok/s")
    logger.info(f"  - HF Speculative Average: {avg(hf_spec_stats, 'tps'):.1f} tok/s")
    
    speedup = avg(hf_spec_stats, 'tps') / avg(base_stats, 'tps') if avg(base_stats, 'tps') > 0 else 0
    logger.info(f"  - Speedup vs Baseline   : {speedup:.2f}x")

if __name__ == "__main__":
    main() 