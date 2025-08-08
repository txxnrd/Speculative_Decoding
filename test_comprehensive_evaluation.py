#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import time
from speculative_decoding.models import ModelLoader
from speculative_decoding.algorithms import OptimizedSpeculativeDecoderV2, SpeculativeDecodingConfig
from speculative_decoding.utils import Config, setup_logger
from speculative_decoding.utils.config import ModelConfig, SamplingConfig

def run_baseline_test(target_model, tokenizer, prompts, max_new_tokens, logger):
    """Run baseline (target model only) generation."""
    logger.info("Running baseline generation...")
    
    baseline_stats = []
    
    for i, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(target_model.device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = target_model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        end_time = time.time()
        
        total_time = end_time - start_time
        total_tokens = outputs.size(1) - input_ids.size(1)
        tokens_per_second = total_tokens / total_time
        
        baseline_stats.append({
            "prompt_idx": i,
            "total_time": total_time,
            "total_tokens": total_tokens,
            "tokens_per_second": tokens_per_second
        })
        
        logger.info(f"  Baseline {i+1}: {total_tokens} tokens in {total_time:.2f}s ({tokens_per_second:.1f} tok/s)")
    
    return baseline_stats

def run_speculative_test(decoder, prompts, max_new_tokens, logger, test_name="Speculative"):
    """Run speculative decoding generation."""
    logger.info(f"Running {test_name} generation...")
    
    spec_stats = []
    
    for i, prompt in enumerate(prompts):
        input_ids = decoder.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        
        result = decoder.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1
        )
        
        stats = result["stats"]
        stats["prompt_idx"] = i
        spec_stats.append(stats)
        
        logger.info(f"  {test_name} {i+1}: {stats['total_tokens']} tokens in {stats['total_time']:.2f}s ({stats['tokens_per_second']:.1f} tok/s)")
        logger.info(f"    Acceptance rate: {stats['acceptance_rate']:.1%}, Iterations: {stats['num_iterations']}")
    
    return spec_stats

def print_comparison_results(baseline_stats, spec_stats, affine_stats, logger):
    """Print detailed comparison results."""
    
    # Calculate averages
    def calc_avg(stats, key):
        return sum(s[key] for s in stats) / len(stats) if stats else 0.0
    
    baseline_avg = {
        "time": calc_avg(baseline_stats, "total_time"),
        "tokens": calc_avg(baseline_stats, "total_tokens"),
        "tok_s": calc_avg(baseline_stats, "tokens_per_second")
    }
    
    spec_avg = {
        "time": calc_avg(spec_stats, "total_time"),
        "tokens": calc_avg(spec_stats, "total_tokens"),
        "tok_s": calc_avg(spec_stats, "tokens_per_second"),
        "acceptance": calc_avg(spec_stats, "acceptance_rate"),
        "iterations": calc_avg(spec_stats, "num_iterations"),
        "theoretical_speedup": calc_avg(spec_stats, "theoretical_speedup")
    }
    
    affine_avg = {
        "time": calc_avg(affine_stats, "total_time"),
        "tokens": calc_avg(affine_stats, "total_tokens"),
        "tok_s": calc_avg(affine_stats, "tokens_per_second"),
        "acceptance": calc_avg(affine_stats, "acceptance_rate"),
        "iterations": calc_avg(affine_stats, "num_iterations"),
        "theoretical_speedup": calc_avg(affine_stats, "theoretical_speedup"),
        "affine_rescue": calc_avg(affine_stats, "affine_rescue_rate")
    }
    
    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE EVALUATION RESULTS")
    logger.info("="*80)
    
    logger.info(f"\n📊 AVERAGE PERFORMANCE:")
    logger.info(f"  Baseline (Target Only):")
    logger.info(f"    Time: {baseline_avg['time']:.2f}s | Tokens/s: {baseline_avg['tok_s']:.1f}")
    
    logger.info(f"  Speculative (No Affine):")
    logger.info(f"    Time: {spec_avg['time']:.2f}s | Tokens/s: {spec_avg['tok_s']:.1f}")
    logger.info(f"    Acceptance: {spec_avg['acceptance']:.1%} | Iterations: {spec_avg['iterations']:.1f}")
    
    logger.info(f"  Speculative + Affine:")
    logger.info(f"    Time: {affine_avg['time']:.2f}s | Tokens/s: {affine_avg['tok_s']:.1f}")
    logger.info(f"    Acceptance: {affine_avg['acceptance']:.1%} | Iterations: {affine_avg['iterations']:.1f}")
    logger.info(f"    Affine Rescue: {affine_avg['affine_rescue']:.1%}")
    
    # Calculate speedups
    spec_speedup = baseline_avg['tok_s'] / spec_avg['tok_s'] if spec_avg['tok_s'] > 0 else 0
    affine_speedup = baseline_avg['tok_s'] / affine_avg['tok_s'] if affine_avg['tok_s'] > 0 else 0
    
    logger.info(f"\n🚀 SPEEDUP ANALYSIS:")
    logger.info(f"  Speculative vs Baseline: {spec_speedup:.2f}x")
    logger.info(f"  Affine vs Baseline: {affine_speedup:.2f}x")
    logger.info(f"  Affine vs Speculative: {spec_avg['tok_s']/affine_avg['tok_s']:.2f}x" if affine_avg['tok_s'] > 0 else "N/A")
    
    logger.info(f"\n⚡ EFFICIENCY METRICS:")
    logger.info(f"  Spec Theoretical Speedup: {spec_avg['theoretical_speedup']:.2f}x")
    logger.info(f"  Affine Theoretical Speedup: {affine_avg['theoretical_speedup']:.2f}x")
    logger.info(f"  Affine Rescue Impact: +{affine_avg['affine_rescue']:.1%} tokens saved")

def main():
    logger = setup_logger()
    
    # Test prompts
    prompts = [
        "The future of artificial intelligence is",
        "In a world where technology advances rapidly,",
        "Explain quantum computing in simple terms:",
        "Write a short story about a robot discovering emotions.",
        "What are the benefits and challenges of renewable energy?"
    ]
    
    max_new_tokens = 150
    
    # Configuration
    config = Config(
        draft_model=ModelConfig(
            model_path="/raid/junha/models/Llama-3.1-8B-Instruct",
            device="auto"  # Change to auto for better GPU distribution
        ),
        target_model=ModelConfig(
            model_path="/raid/junha/models/Llama-3.1-70B-Instruct", 
            device="auto"
        ),
        sampling=SamplingConfig(    
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
    )
    
    # Load models
    logger.info("Loading models...")
    loader = ModelLoader(logger=logger)
    draft_model, target_model, tokenizer = loader.load_draft_and_target_models(
        config.draft_model, config.target_model
    )
    
    # 1. Baseline test
    baseline_stats = run_baseline_test(target_model, tokenizer, prompts, max_new_tokens, logger)
    
    # 2. Speculative decoding without affine verification
    spec_config_basic = SpeculativeDecodingConfig(
        num_assistant_tokens=5,
        affine_verification=False,
        use_cache=False,  # Disable cache for fair comparison
        verbose=True
    )
    
    decoder_basic = OptimizedSpeculativeDecoderV2(
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        config=config,
        spec_config=spec_config_basic,
        logger=logger
    )
    
    spec_stats = run_speculative_test(decoder_basic, prompts, max_new_tokens, logger, "Speculative (Basic)")
    
    # 3. Speculative decoding with affine verification
    spec_config_affine = SpeculativeDecodingConfig(
        num_assistant_tokens=5,
        affine_verification=True,
        affine_model_path="affine_verifier_llama_v2.pt",
        affine_accept_threshold=0.05,  # Reasonable threshold based on analysis
        use_cache=False,  # CRITICAL: Disable cache for proper hidden states
        verbose=True
    )
    
    decoder_affine = OptimizedSpeculativeDecoderV2(
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        config=config,
        spec_config=spec_config_affine,
        logger=logger
    )
    
    affine_stats = run_speculative_test(decoder_affine, prompts, max_new_tokens, logger, "Speculative + Affine")
    
    # Print comprehensive results
    print_comparison_results(baseline_stats, spec_stats, affine_stats, logger)

if __name__ == "__main__":
    main() 