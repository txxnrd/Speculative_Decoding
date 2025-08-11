#!/usr/bin/env python3
"""Simple test for tree generation functions without full integration."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from speculative_decoding.models import ModelLoader
from speculative_decoding.algorithms import OptimizedSpeculativeDecoderV2, SpeculativeDecodingConfig
from speculative_decoding.utils import Config, setup_logger
from speculative_decoding.utils.config import ModelConfig, SamplingConfig

def test_tree_functions():
    logger = setup_logger()
    
    # Use existing models
    cfg = Config(
        draft_model=ModelConfig(
            model_path="/raid/junha/models/Llama-3.1-8B-Instruct",
            device="cuda:0",
        ),
        target_model=ModelConfig(
            model_path="/raid/junha/models/Llama-3.1-8B-Instruct",  # Same model for testing
            device="cuda:1",
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
    
    # Load draft model only for now
    draft_model, tokenizer = loader.load_model_and_tokenizer(cfg.draft_model)
    target_model = draft_model  # Use same model for testing
    
    # Create decoder with tree search
    spec_config = SpeculativeDecodingConfig(
        tree_search=True,
        beam_width=3,
        tree_depth=5,
        tree_expansion_strategy="top_k",
        tree_top_k=10,
        use_cache=False,  # Disable cache for simpler testing
        verbose=True,
    )
    
    decoder = OptimizedSpeculativeDecoderV2(
        draft_model, target_model, tokenizer, cfg, spec_config, logger
    )
    
    # Test input
    prompt = "The key to success is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")
    
    logger.info("Testing tree generation...")
    with torch.no_grad():
        # Test tree generation
        paths_tokens, paths_logits, paths_past, paths_hiddens = decoder._get_draft_tokens_tree(
            input_ids, attention_mask, None, tree_depth=3, beam_width=3
        )
        
        logger.info(f"Generated {len(paths_tokens)} paths")
        
        # Show generated paths
        for i, tokens in enumerate(paths_tokens):
            text = tokenizer.decode(tokens[0], skip_special_tokens=True)
            logger.info(f"Path {i+1}: '{text}'")
            logger.info(f"  Shape: {tokens.shape}")
            logger.info(f"  Tokens: {tokens[0].tolist()}")
        
        # Test verification (simplified)
        if len(paths_tokens) > 0:
            # Get target logits for first path
            target_input = torch.cat([input_ids, paths_tokens[0]], dim=1)
            target_out = target_model(target_input)
            target_logits = target_out.logits[:, input_ids.shape[1]:, :]
            
            logger.info(f"\nTarget logits shape: {target_logits.shape}")
            
            # Test path verification
            accepted_tokens, num_accepted, best_idx = decoder._verify_tree_paths(
                paths_tokens, paths_logits, target_logits
            )
            
            logger.info(f"Best path index: {best_idx}")
            logger.info(f"Accepted {num_accepted} tokens")
            logger.info(f"Accepted text: '{tokenizer.decode(accepted_tokens[0], skip_special_tokens=True)}'")

if __name__ == "__main__":
    test_tree_functions() 