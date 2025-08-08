"""Main entry point for speculative decoding."""

import sys
from pathlib import Path

# Add parent directory to path if running directly
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import json
import yaml

from speculative_decoding.models import ModelLoader
from speculative_decoding.algorithms import OptimizedSpeculativeDecoderV2
from speculative_decoding.utils import Config, setup_logger


def main():
    """Main function to run speculative decoding."""
    parser = argparse.ArgumentParser(description="Speculative Decoding for Qwen3 models")
    
    # Configuration arguments
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file"
    )
    
    # Model arguments
    parser.add_argument(
        "--draft-model",
        type=str,
        default="/raid/taeyun/Qwen3-8B",
        help="Path to draft model"
    )
    parser.add_argument(
        "--target-model",
        type=str,
        default="/raid/taeyun/Qwen3-14B",
        help="Path to target model"
    )
    
    # Generation arguments
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt for generation"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--draft-k",
        type=int,
        default=4,
        help="Number of tokens to draft at each iteration"
    )
    
    # Sampling arguments
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        default=True,
        help="Whether to use sampling (vs greedy decoding)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )
    
    # Other arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Model dtype"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save output JSON"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run baseline generation with target model only for comparison"
    )
    
    args = parser.parse_args()
    
    # Set up logger
    logger = setup_logger(level=args.log_level)
    
    # Load or create configuration
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = Config.from_yaml(args.config)
    else:
        # Create config from command line arguments
        from speculative_decoding.utils.config import ModelConfig, SamplingConfig, SpeculativeConfig
        
        config = Config(
            draft_model=ModelConfig(
                model_path=args.draft_model,
                device=args.device,
                dtype=args.dtype
            ),
            target_model=ModelConfig(
                model_path=args.target_model,
                device=args.device,
                dtype=args.dtype
            ),
            sampling=SamplingConfig(
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                do_sample=args.do_sample
            ),
            speculative=SpeculativeConfig(
                draft_k=args.draft_k
            ),
            max_new_tokens=args.max_new_tokens,
            seed=args.seed
        )
    
    logger.info("Configuration:")
    logger.info(json.dumps(config.to_dict(), indent=2))
    
    # Load models
    model_loader = ModelLoader(logger=logger)
    draft_model, target_model, tokenizer = model_loader.load_draft_and_target_models(
        config.draft_model,
        config.target_model
    )
    
    # Tokenize input
    logger.info(f"Input prompt: {args.prompt}")
    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(args.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(args.device)
    
    # Run speculative decoding
    decoder = OptimizedSpeculativeDecoderV2(
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        config=config,
        logger=logger
    )
    
    logger.info("\n" + "="*50)
    logger.info("Running speculative decoding...")
    logger.info("="*50 + "\n")
    
    result = decoder.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens
    )

    # Decode output supporting both plain tensor and dict formats
    if isinstance(result, torch.Tensor):
        generated_ids = result
    else:
        generated_ids = result.get("sequences", result)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    logger.info("\n" + "="*50)
    logger.info("Generated text:")
    logger.info("="*50)
    print(generated_text)
    
    # Print statistics
    if isinstance(result, dict) and "stats" in result:
        stats = result["stats"]
        logger.info("\n" + "="*50)
        logger.info("Generation statistics:")
        logger.info("="*50)
        logger.info(f"Total iterations: {stats['total_iterations']}")
        logger.info(f"Total draft tokens: {stats['total_draft_tokens']}")
        logger.info(f"Total accepted tokens: {stats['total_accepted_tokens']}")
        logger.info(f"Average acceptance rate: {stats['average_acceptance_rate']:.1%}")
        logger.info(f"Total time: {stats['total_time']:.2f}s")
        logger.info(f"Tokens/second: {result['num_generated_tokens'] / stats['total_time']:.1f}")
    
    # Run baseline if requested
    if args.baseline:
        logger.info("\n" + "="*50)
        logger.info("Running baseline generation (target model only)...")
        logger.info("="*50 + "\n")
        
        import time
        start_time = time.time()
        
        with torch.no_grad():
            baseline_output = target_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                do_sample=args.do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        baseline_time = time.time() - start_time
        baseline_text = tokenizer.decode(baseline_output[0], skip_special_tokens=True)
        
        logger.info("Baseline generated text:")
        print(baseline_text)
        
        logger.info(f"\nBaseline time: {baseline_time:.2f}s")
        logger.info(f"Baseline tokens/second: {args.max_new_tokens / baseline_time:.1f}")
        
        if isinstance(result, dict) and "stats" in result:
            speedup = baseline_time / stats['total_time']
            logger.info(f"\nSpeculative decoding speedup: {speedup:.2f}x")
    
    # Save output if requested
    if args.output:
        output_data = {
            "prompt": args.prompt,
            "generated_text": generated_text,
            "num_generated_tokens": result["num_generated_tokens"],
            "config": config.to_dict()
        }
        
        if isinstance(result, dict) and "stats" in result:
            output_data["stats"] = result["stats"]
            
        if args.baseline:
            output_data["baseline"] = {
                "text": baseline_text,
                "time": baseline_time,
                "tokens_per_second": args.max_new_tokens / baseline_time
            }
            
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"\nOutput saved to {args.output}")


if __name__ == "__main__":
    main() 