#!/usr/bin/env python3
"""
Test vLLM's built-in speculative decoding implementation to compare its performance
against the baseline. vLLM is optimized for high-throughput serving and batching.

Usage:
    python test_vllm_speculative.py --num-samples 20 --max-new 100
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import time
import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams

from speculative_decoding.utils import setup_logger

# --- Model Configuration ---
TARGET_MODEL_PATH = "/raid/junha/models/Llama-3.1-70B-Instruct"
DRAFT_MODEL_PATH = "/raid/junha/models/Llama-3.1-8B-Instruct"

def run_vllm_test(
    questions: list[str],
    sampling_params: SamplingParams,
    logger,
    use_speculative: bool = False
):
    """
    Runs a generation test using vLLM, either in baseline or speculative mode.
    
    Returns a dictionary of performance stats.
    """
    tag = "vLLM Speculative" if use_speculative else "vLLM Baseline"
    logger.info(f"--- Running {tag} ---")

    # Detect available GPUs for tensor parallelism
    num_gpus = torch.cuda.device_count()
    logger.info(f"Using {num_gpus} GPUs for tensor parallelism.")

    # Initialize the LLM engine
    llm_kwargs = {
        "model": TARGET_MODEL_PATH,
        "tensor_parallel_size": num_gpus,
        "disable_log_stats": True,
        "disable_log_requests": True,
        "enforce_eager": False,
    }
    if use_speculative:
        llm_kwargs["speculative_model"] = DRAFT_MODEL_PATH
        llm_kwargs["num_speculative_tokens"] = 5 # Default K for vLLM
    
    llm = LLM(**llm_kwargs)
    
    # Run generation
    start_time = time.time()
    outputs = llm.generate(questions, sampling_params)
    end_time = time.time()
    
    total_time = end_time - start_time
    
    # Collect stats from outputs
    total_input_tokens = 0
    total_output_tokens = 0
    acceptance_rates = []
    
    for output in outputs:
        total_input_tokens += len(output.prompt_token_ids)
        total_output_tokens += len(output.outputs[0].token_ids)
        if use_speculative and hasattr(output.outputs[0], 'acceptance_rate'):
            acceptance_rates.append(output.outputs[0].acceptance_rate)

    tokens_per_second = total_output_tokens / total_time if total_time > 0 else 0
    avg_acceptance_rate = sum(acceptance_rates) / len(acceptance_rates) if acceptance_rates else 0.0

    logger.info(f"  - Completed {len(questions)} prompts in {total_time:.2f}s")
    logger.info(f"  - Generated {total_output_tokens} tokens.")
    logger.info(f"  - Throughput: {tokens_per_second:.1f} tok/s")
    if use_speculative:
        logger.info(f"  - Average Acceptance Rate: {avg_acceptance_rate:.1%}")
        
    # Clean up the model to free VRAM
    del llm
    torch.cuda.empty_cache()
    
    return {"tps": tokens_per_second, "acceptance_rate": avg_acceptance_rate}

def avg(stats_list, key):
    return sum(d[key] for d in stats_list) / len(stats_list) if stats_list else 0.0

def main():
    parser = argparse.ArgumentParser(description="Run vLLM Speculative Decoding Benchmark")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of GSM8K questions to sample")
    parser.add_argument("--max-new", type=int, default=100, help="Max new tokens to generate")
    args = parser.parse_args()

    logger = setup_logger()

    logger.info("Loading GSM8K dataset...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    questions = [ds[i]["question"].strip() + " Answer:" for i in range(min(args.num_samples, len(ds)))]

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_tokens=args.max_new,
    )

    # Run Baseline
    base_stats = run_vllm_test(questions, sampling_params, logger, use_speculative=False)
    
    # Run vLLM Speculative Decoding
    spec_stats = run_vllm_test(questions, sampling_params, logger, use_speculative=True)

    logger.info("\n" + "="*40 + " vLLM FINAL RESULTS " + "="*40)
    logger.info(f"  - Baseline Average      : {base_stats['tps']:.1f} tok/s")
    logger.info(f"  - vLLM Speculative Avg  : {spec_stats['tps']:.1f} tok/s")
    logger.info(f"  - Avg Acceptance Rate   : {spec_stats['acceptance_rate']:.1%}")
    
    speedup = spec_stats['tps'] / base_stats['tps'] if base_stats['tps'] > 0 else 0
    logger.info(f"  - Speedup vs Baseline   : {speedup:.2f}x")
    logger.info("="*94)

if __name__ == "__main__":
    main() 