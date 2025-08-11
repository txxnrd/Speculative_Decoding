#!/usr/bin/env python3
"""Test script for tree-based speculative decoding with affine pruning."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import torch
import time
from speculative_decoding.models import ModelLoader
from speculative_decoding.algorithms import OptimizedSpeculativeDecoderV2, SpeculativeDecodingConfig
from speculative_decoding.utils import Config, setup_logger
from speculative_decoding.utils.config import ModelConfig, SamplingConfig

def test_tree_search():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="The key to successful machine learning is")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--tree-depth", type=int, default=5)
    parser.add_argument("--pruning-threshold", type=float, default=0.3)
    parser.add_argument("--affine-model", type=str, default="affine_verifier_v4_regression.pt")
    parser.add_argument("--tree-strategy", type=str, default="top_k", 
                       choices=["top_k", "top_p", "beam", "diverse_beam"])
    parser.add_argument("--tree-top-k", type=int, default=10)
    parser.add_argument("--tree-top-p", type=float, default=0.9)
    parser.add_argument("--tree-temperature", type=float, default=1.0)
    parser.add_argument("--tree-diversity", type=float, default=0.5)
    args = parser.parse_args()
    
    logger = setup_logger()
    
    # Model configuration
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
    
    # Load models
    loader = ModelLoader(logger=logger)
    draft_model, target_model, tokenizer = loader.load_draft_and_target_models(
        cfg.draft_model, cfg.target_model
    )
    
    # Test standard speculative decoding
    logger.info("Testing standard speculative decoding...")
    spec_config_standard = SpeculativeDecodingConfig(
        num_assistant_tokens=args.tree_depth,
        tree_search=False,
        use_cache=True,
        verbose=True,
    )
    decoder_standard = OptimizedSpeculativeDecoderV2(
        draft_model, target_model, tokenizer, cfg, spec_config_standard, logger
    )
    
    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")
    
    start_time = time.time()
    outputs_standard = decoder_standard.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_tokens,
    )
    standard_time = time.time() - start_time
    
    logger.info(f"Standard mode - Time: {standard_time:.2f}s")
    logger.info(f"Standard mode - Tokens/s: {outputs_standard['stats']['tokens_per_second']:.1f}")
    logger.info(f"Standard mode - Acceptance rate: {outputs_standard['stats']['acceptance_rate']:.1%}")
    
    # Test tree search with affine pruning
    logger.info("\nTesting tree search with affine pruning...")
    logger.info(f"Tree expansion strategy: {args.tree_strategy}")
    
    spec_config_tree = SpeculativeDecodingConfig(
        num_assistant_tokens=args.tree_depth,
        tree_search=True,
        beam_width=args.beam_width,
        tree_depth=args.tree_depth,
        pruning_threshold=args.pruning_threshold,
        tree_expansion_strategy=args.tree_strategy,
        tree_top_k=args.tree_top_k,
        tree_top_p=args.tree_top_p,
        tree_temperature=args.tree_temperature,
        tree_diversity_penalty=args.tree_diversity,
        affine_verification=True,
        affine_model_path=args.affine_model,
        use_cache=True,
        verbose=True,
    )
    decoder_tree = OptimizedSpeculativeDecoderV2(
        draft_model, target_model, tokenizer, cfg, spec_config_tree, logger
    )
    
    # Note: The current implementation doesn't fully support tree search in generate()
    # This is a placeholder to show how it would be used
    logger.info("Tree search implementation is not yet integrated into the main generate loop.")
    logger.info("The infrastructure is in place with:")
    logger.info(f"  - _get_draft_tokens_tree() for multi-path generation")
    logger.info(f"  - _prune_tree_paths() for affine-based pruning")
    logger.info(f"  - _verify_tree_paths() for multi-path verification")
    
    # Demo the tree generation function
    logger.info("\nDemonstrating tree generation...")
    with torch.no_grad():
        # Get tree paths
        paths_tokens, paths_logits, paths_past, paths_hiddens = decoder_tree._get_draft_tokens_tree(
            input_ids, attention_mask, None, args.tree_depth, args.beam_width
        )
        
        logger.info(f"Generated {len(paths_tokens)} paths")
        
        # Prune paths
        if decoder_tree.affine_verifier:
            pruned_tokens, pruned_logits, pruned_hiddens, scores = decoder_tree._prune_tree_paths(
                paths_tokens, paths_logits, paths_hiddens
            )
            logger.info(f"After pruning: {len(pruned_tokens)} paths remain")
            logger.info(f"Path scores: {[f'{s:.3f}' for s in scores]}")
        
        # Show generated paths
        for i, tokens in enumerate(paths_tokens[:3]):  # Show first 3 paths
            text = tokenizer.decode(tokens[0], skip_special_tokens=True)
            logger.info(f"Path {i+1}: {text}")

if __name__ == "__main__":
    test_tree_search() 