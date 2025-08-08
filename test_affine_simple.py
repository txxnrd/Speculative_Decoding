#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from speculative_decoding.algorithms.optimized_speculative_decoding_v2 import OptimizedSpeculativeDecoderV2, SpeculativeDecodingConfig
from speculative_decoding.utils import Config, setup_logger
from speculative_decoding.utils.config import ModelConfig, SamplingConfig

def test_affine_simple():
    logger = setup_logger()
    
    # Create simple test configuration
    config = Config(
        draft_model=ModelConfig(
            model_path="/raid/junha/models/Llama-3.1-8B-Instruct",
            device="auto"
        ),
        target_model=ModelConfig(
            model_path="/raid/junha/models/Llama-3.1-8B-Instruct",  # Use same model for simplicity
            device="auto"
        ),
        sampling=SamplingConfig(
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
    )
    
    # Test just affine verification functionality
    spec_config = SpeculativeDecodingConfig(
        num_assistant_tokens=3,  # Smaller number for faster testing
        affine_verification=True,
        affine_model_path="affine_verifier_llama_v2.pt",
        affine_accept_threshold=0.05,  # Reasonable threshold based on observed probabilities
        use_cache=False,  # Disable KV cache to match training conditions
        verbose=True
    )
    
    logger.info("Loading models...")
    from speculative_decoding.models import ModelLoader
    loader = ModelLoader(logger=logger)
    draft_model, target_model, tokenizer = loader.load_draft_and_target_models(
        config.draft_model, config.target_model
    )
    
    decoder = OptimizedSpeculativeDecoderV2(
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        config=config,
        spec_config=spec_config,
        logger=logger
    )
    
    # Simple test prompt
    prompt = "The future of AI is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    
    logger.info(f"Testing with prompt: '{prompt}'")
    logger.info("Starting generation...")
    
    try:
        result = decoder.generate(
            input_ids=input_ids,
            max_new_tokens=20,  # Very small for quick test
            num_return_sequences=1
        )
        
        stats = result["stats"]
        logger.info(f"✓ Generation completed successfully!")
        logger.info(f"Generated {stats['total_tokens']} tokens in {stats['total_time']:.2f}s")
        logger.info(f"Acceptance rate: {stats['acceptance_rate']:.1%}")
        
        if 'generated_text' in result:
            logger.info(f"Generated text: {result['generated_text']}")
        
    except Exception as e:
        logger.error(f"✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_affine_simple() 