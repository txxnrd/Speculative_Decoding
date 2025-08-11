#!/usr/bin/env python3
"""Test tree search using the full generate function."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import time
from speculative_decoding.models import ModelLoader
from speculative_decoding.algorithms import OptimizedSpeculativeDecoderV2, SpeculativeDecodingConfig
from speculative_decoding.utils import Config, setup_logger
from speculative_decoding.utils.config import ModelConfig, SamplingConfig

def main():
    logger = setup_logger()
    
    # Configuration
    cfg = Config(
        draft_model=ModelConfig(
            model_path="/raid/junha/models/Llama-3.1-8B-Instruct",
            device="auto",
        ),
        target_model=ModelConfig(
            model_path="/raid/junha/models/Llama-3.1-70B-Instruct",
            device="auto",
        ),
        sampling=SamplingConfig(
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True,
        ),
    )
    
    logger.info("Loading models...")
    loader = ModelLoader(logger=logger)
    draft_model, target_model, tokenizer = loader.load_draft_and_target_models(
        cfg.draft_model, cfg.target_model
    )
    
    prompt = "The future of artificial intelligence"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")
    
    # Test with fewer tokens for faster results
    max_tokens = 10  # Reduced from 30
    
    # Test 1: Standard speculative decoding
    logger.info("\n=== Testing STANDARD speculative decoding ===")
    spec_config_standard = SpeculativeDecodingConfig(
        num_assistant_tokens=5,
        tree_search=False,
        use_cache=True,
        verbose=True,
    )
    
    decoder_standard = OptimizedSpeculativeDecoderV2(
        draft_model, target_model, tokenizer, cfg, spec_config_standard, logger
    )
    
    start = time.time()
    outputs_standard = decoder_standard.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_tokens,
    )
    standard_time = time.time() - start
    
    logger.info(f"Standard Time: {standard_time:.2f}s")
    logger.info(f"Generated tokens: {outputs_standard['sequences'].shape[1] - input_ids.shape[1]}")
    generated_text = tokenizer.decode(outputs_standard['sequences'][0], skip_special_tokens=True)
    logger.info(f"Standard Output: {generated_text}")
    logger.info(f"Tokens/s: {outputs_standard['stats']['tokens_per_second']:.1f}")
    logger.info(f"Acceptance rate: {outputs_standard['stats']['acceptance_rate']:.1%}")
    
    # Test 2: Tree search speculative decoding
    logger.info("\n=== Testing TREE SEARCH speculative decoding ===")
    spec_config_tree = SpeculativeDecodingConfig(
        num_assistant_tokens=5,
        tree_search=True,
        beam_width=3,
        tree_depth=5,
        tree_expansion_strategy="top_k",
        tree_top_k=10,
        use_cache=True,
        verbose=True,
    )
    
    decoder_tree = OptimizedSpeculativeDecoderV2(
        draft_model, target_model, tokenizer, cfg, spec_config_tree, logger
    )
    
    start = time.time()
    outputs_tree = decoder_tree.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=30,
    )
    tree_time = time.time() - start
    
    logger.info(f"Tree Time: {tree_time:.2f}s")
    logger.info(f"Generated tokens: {outputs_tree['sequences'].shape[1] - input_ids.shape[1]}")
    generated_text_tree = tokenizer.decode(outputs_tree['sequences'][0], skip_special_tokens=True)
    logger.info(f"Tree Output: {generated_text_tree}")
    logger.info(f"Tokens/s: {outputs_tree['stats']['tokens_per_second']:.1f}")
    logger.info(f"Acceptance rate: {outputs_tree['stats']['acceptance_rate']:.1%}")
    
    # Comparison
    logger.info("\n=== COMPARISON ===")
    logger.info(f"Speedup: {standard_time / tree_time:.2f}x")
    logger.info(f"Standard acceptance: {outputs_standard['stats']['acceptance_rate']:.1%}")
    logger.info(f"Tree acceptance: {outputs_tree['stats']['acceptance_rate']:.1%}")

if __name__ == "__main__":
    main() 