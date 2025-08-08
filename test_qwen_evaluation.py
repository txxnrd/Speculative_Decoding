import argparse
import time
import torch
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
from transformers import set_seed

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from speculative_decoding.utils.model_loader import ModelLoader
from speculative_decoding.utils.config import LoraConfig, DeviceMap, OptimizationConfig, SpeculativeDecodingConfig, ModelConfig
from speculative_decoding.algorithms.optimized_speculative_decoding_v2 import OptimizedSpeculativeDecoderV2
from speculative_decoding.utils.kv_cache import KVCache
from speculative_decoding.utils.misc import get_system_ram_gb
from datasets import load_dataset

# --- Constants ---
DRAFT_MODEL_PATH = "/raid/taeyun/Qwen3-0.6B"
TARGET_MODEL_PATH = "/raid/taeyun/Qwen3-8B"
AFFINE_MODEL_PATH = "affine_verifier_v3_large.pt"
AFFINE_THRESHOLD = 0.5
DEFAULT_MAX_NEW_TOKENS = 100
DEFAULT_NUM_SAMPLES = 10

# --- Model and Tokenizer Loading ---
def load_models_and_tokenizer(args):
    """Loads the draft model, target model, and tokenizer."""
    set_seed(42)
    
    # Define model configurations
    draft_model_config = ModelConfig(
        model_path=DRAFT_MODEL_PATH,
        device_map=DeviceMap.AUTO,
        use_flash_attention_2=True
    )
    
    target_model_config = ModelConfig(
        model_path=TARGET_MODEL_PATH,
        device_map=DeviceMap.AUTO,
        use_flash_attention_2=True
    )

    # Load models and tokenizer
    model_loader = ModelLoader(
        draft_model_config=draft_model_config,
        target_model_config=target_model_config
    )
    draft_model, target_model, tokenizer = model_loader.load_models()
    
    return draft_model, target_model, tokenizer

# --- Evaluation Scenarios ---

def run_baseline(target_model, tokenizer, questions, max_new_tokens):
    """Runs the baseline generation without speculative decoding."""
    print("\n--- Running Baseline ---")
    total_time = 0
    total_tokens = 0
    
    for question in tqdm(questions, desc="Baseline"):
        input_ids = tokenizer(question, return_tensors="pt").input_ids.to(target_model.device)
        
        start_time = time.time()
        outputs = target_model.generate(input_ids, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
        end_time = time.time()
        
        total_time += end_time - start_time
        total_tokens += len(outputs[0]) - len(input_ids[0])
        
    avg_speed = total_tokens / total_time if total_time > 0 else 0
    print(f"Baseline Average Speed: {avg_speed:.2f} tokens/sec")
    return avg_speed

def run_our_basic_speculative(target_model, draft_model, tokenizer, questions, max_new_tokens):
    """Runs our basic speculative decoding implementation."""
    print("\n--- Running Our Basic Speculative Decoder ---")
    spec_config = SpeculativeDecodingConfig(
        draft_model_config=ModelConfig(model_path=DRAFT_MODEL_PATH),
        target_model_config=ModelConfig(model_path=TARGET_MODEL_PATH),
        num_assistant_tokens=5,
    )
    
    decoder = OptimizedSpeculativeDecoderV2(
        config=spec_config,
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer
    )

    total_time = 0
    total_tokens = 0
    acceptance_rates = []

    for question in tqdm(questions, desc="Our Basic Speculative"):
        input_ids = tokenizer(question, return_tensors="pt").input_ids
        
        start_time = time.time()
        output_ids, stats = decoder.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
        )
        end_time = time.time()

        total_time += end_time - start_time
        total_tokens += len(output_ids[0]) - len(input_ids[0])
        acceptance_rates.append(stats['acceptance_rate'])
        
    avg_speed = total_tokens / total_time if total_time > 0 else 0
    avg_acceptance_rate = np.mean(acceptance_rates) * 100
    print(f"Our Basic Speculative Average Speed: {avg_speed:.2f} tokens/sec")
    print(f"Our Basic Speculative Average Acceptance Rate: {avg_acceptance_rate:.2f}%")
    return avg_speed, avg_acceptance_rate

def run_our_affine_speculative(target_model, draft_model, tokenizer, questions, max_new_tokens, affine_model_path, affine_threshold):
    """Runs our speculative decoding with the Affine Verifier."""
    print("\n--- Running Our Affine Speculative Decoder ---")
    spec_config = SpeculativeDecodingConfig(
        draft_model_config=ModelConfig(model_path=DRAFT_MODEL_PATH),
        target_model_config=ModelConfig(model_path=TARGET_MODEL_PATH),
        num_assistant_tokens=5,
        affine_verification=True,
        affine_model_path=affine_model_path,
        affine_accept_threshold=affine_threshold
    )
    
    decoder = OptimizedSpeculativeDecoderV2(
        config=spec_config,
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer
    )

    total_time = 0
    total_tokens = 0
    acceptance_rates = []
    
    for question in tqdm(questions, desc="Our Affine Speculative"):
        input_ids = tokenizer(question, return_tensors="pt").input_ids
        
        start_time = time.time()
        output_ids, stats = decoder.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
        )
        end_time = time.time()

        total_time += end_time - start_time
        total_tokens += len(output_ids[0]) - len(input_ids[0])
        acceptance_rates.append(stats['acceptance_rate'])

    avg_speed = total_tokens / total_time if total_time > 0 else 0
    avg_acceptance_rate = np.mean(acceptance_rates) * 100
    print(f"Our Affine Speculative Average Speed: {avg_speed:.2f} tokens/sec")
    print(f"Our Affine Speculative Average Acceptance Rate: {avg_acceptance_rate:.2f}%")
    return avg_speed, avg_acceptance_rate
    
def run_hf_speculative(target_model, draft_model, tokenizer, questions, max_new_tokens):
    """Runs HuggingFace's native speculative decoding."""
    print("\n--- Running HuggingFace Native Speculative Decoder ---")
    total_time = 0
    total_tokens = 0

    for question in tqdm(questions, desc="HF Speculative"):
        input_ids = tokenizer(question, return_tensors="pt").input_ids.to(target_model.device)
        
        start_time = time.time()
        outputs = target_model.generate(
            input_ids,
            assistant_model=draft_model,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
        end_time = time.time()
        
        total_time += end_time - start_time
        total_tokens += len(outputs[0]) - len(input_ids[0])
        
    avg_speed = total_tokens / total_time if total_time > 0 else 0
    print(f"HF Speculative Average Speed: {avg_speed:.2f} tokens/sec")
    return avg_speed

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comprehensive evaluation for Qwen models.")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES, help="Number of samples to evaluate from the dataset.")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Maximum number of new tokens to generate.")
    parser.add_argument("--affine-model", type=str, default=AFFINE_MODEL_PATH, help="Path to the affine verifier model.")
    parser.add_argument("--affine-threshold", type=float, default=AFFINE_THRESHOLD, help="Acceptance threshold for the affine verifier.")
    args = parser.parse_args()

    # Load dataset
    gsm8k = load_dataset("gsm8k", "main")
    questions = [item['question'] for item in gsm8k['test']][:args.num_samples]

    # Load models
    draft_model, target_model, tokenizer = load_models_and_tokenizer(args)

    # Run evaluations
    baseline_speed = run_baseline(target_model, tokenizer, questions, args.max_new_tokens)
    our_basic_speed, our_basic_acc_rate = run_our_basic_speculative(target_model, draft_model, tokenizer, questions, args.max_new_tokens)
    our_affine_speed, our_affine_acc_rate = run_our_affine_speculative(target_model, draft_model, tokenizer, questions, args.max_new_tokens, args.affine_model, args.affine_threshold)
    hf_speed = run_hf_speculative(target_model, draft_model, tokenizer, questions, args.max_new_tokens)

    # Print summary
    print("\n--- Evaluation Summary ---")
    print(f"Baseline Speed: {baseline_speed:.2f} tokens/sec")
    print(f"Our Basic Speculative Speed: {our_basic_speed:.2f} tokens/sec ({(our_basic_speed/baseline_speed):.2f}x speedup), Acceptance Rate: {our_basic_acc_rate:.2f}%")
    print(f"Our Affine Speculative Speed: {our_affine_speed:.2f} tokens/sec ({(our_affine_speed/baseline_speed):.2f}x speedup), Acceptance Rate: {our_affine_acc_rate:.2f}%")
    print(f"HuggingFace Speculative Speed: {hf_speed:.2f} tokens/sec ({(hf_speed/baseline_speed):.2f}x speedup)") 