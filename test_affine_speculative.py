#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from speculative_decoding.models import ModelLoader
from speculative_decoding.algorithms import OptimizedSpeculativeDecoderV2, SpeculativeDecodingConfig
from speculative_decoding.utils import Config, setup_logger

def main():
    logger = setup_logger()
    
    # Configuration
    config = Config(
        model=Config.ModelConfig(
            model_path="/raid/junha/models/Llama-3.1-8B-Instruct",
            device="cuda"
        ),
        target_model=Config.ModelConfig(
            model_path="/raid/junha/models/Llama-3.1-70B-Instruct", 
            device="auto"
        ),
        sampling=Config.SamplingConfig(
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
    )
    
    # Speculative decoding configuration with affine verification
    spec_config = SpeculativeDecodingConfig(
        num_assistant_tokens=5,
        affine_verification=True,
        affine_model_path="affine_verifier_llama.pt",
        affine_accept_threshold=0.5,
        use_cache=True,
        verbose=True
    )
    
    # Load models
    logger.info("Loading models...")
    loader = ModelLoader(logger=logger)
    draft_model, target_model, tokenizer = loader.load_draft_and_target_models(
        config.model, config.target_model
    )
    
    # Create decoder
    decoder = OptimizedSpeculativeDecoderV2(
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        config=config,
        spec_config=spec_config,
        logger=logger
    )
    
    # Test prompts
    prompts = [
        "The future of artificial intelligence is",
        "In a world where technology advances rapidly,",
        "Explain quantum computing in simple terms:",
    ]
    
    logger.info("Testing affine verifier speculative decoding...")
    
    for i, prompt in enumerate(prompts):
        logger.info(f"\n{'='*60}")
        logger.info(f"Test {i+1}: {prompt}")
        logger.info('='*60)
        
        # Encode prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        
        # Generate with affine verification
        logger.info("Generating with affine verification...")
        result = decoder.generate(
            input_ids=input_ids,
            max_new_tokens=100,
            num_return_sequences=1
        )
        
        # Decode and display
        if isinstance(result, dict):
            generated_ids = result["sequences"]
            stats = result.get("stats", {})
        else:
            generated_ids = result
            stats = {}
            
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        logger.info(f"\nGenerated text:")
        print(generated_text)
        
        if stats:
            logger.info(f"\nStatistics:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")

if __name__ == "__main__":
    main() 