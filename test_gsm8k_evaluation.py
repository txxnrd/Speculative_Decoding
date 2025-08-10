#!/usr/bin/env python3
"""Comprehensive evaluation of baseline, basic speculative, and affine speculative decoding
   specifically on GSM8K math word-problem questions (same domain used for training the
   Affine Verifier).

   Usage:
       CUDA_VISIBLE_DEVICES=0,1 python test_gsm8k_evaluation.py --num-samples 20
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import time
import torch
from datasets import load_dataset

from speculative_decoding.models import ModelLoader
from speculative_decoding.algorithms import OptimizedSpeculativeDecoderV2, SpeculativeDecodingConfig
from speculative_decoding.utils import Config, setup_logger
from speculative_decoding.utils.config import ModelConfig, SamplingConfig

def run_baseline(target_model, tokenizer, questions, max_new_tokens, logger):
    """Run baseline generation (target model only)."""
    logger.info("Running baseline generation (70B only)…")
    stats = []
    for i, q in enumerate(questions):
        input_ids = tokenizer.encode(q, return_tensors="pt").to(target_model.device)
        attention_mask = torch.ones_like(input_ids)
        start = time.time()
        with torch.no_grad():
            _ = target_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        t = time.time() - start
        tok = max_new_tokens
        logger.info(f"  Baseline {i+1}: {tok} tokens in {t:.2f}s ({tok/t:.1f} tok/s)")
        stats.append({"time": t, "tokens": tok, "tps": tok / t})
    return stats

def run_spec(decoder, tokenizer, questions, max_new_tokens, logger, tag):
    logger.info(f"Running {tag} generation…")
    stats = []
    for i, q in enumerate(questions):
        inputs = tokenizer(q, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = inputs.input_ids.to("cuda")
        attention_mask = inputs.attention_mask.to("cuda")
        out = decoder.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens
        )
        s = out["stats"]
        logger.info(
            f"  {tag} {i+1}: {s['total_tokens']} tokens in {s['total_time']:.2f}s "
            f"({s['tokens_per_second']:.1f} tok/s, acceptance {s['acceptance_rate']:.1%})"
        )
        stats.append(s)
    return stats

def avg(lst, key):
    return sum(d[key] for d in lst) / len(lst) if lst else 0.0

def run_hf_speculative(draft_model, target_model, tokenizer, questions, max_new_tokens, logger):
    """Run generation using HuggingFace's built-in speculative decoding."""
    logger.info("Running HuggingFace Speculative generation…")
    stats = []
    for i, q in enumerate(questions):
        input_ids = tokenizer.encode(q, return_tensors="pt").to(target_model.device)
        attention_mask = torch.ones_like(input_ids)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = target_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                assistant_model=draft_model,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        end_time = time.time()
        
        total_time = end_time - start_time
        # In HF spec dec, output includes prompt, so subtract its length
        total_tokens = outputs.size(1) - input_ids.size(1)
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        
        logger.info(f"  HF Speculative {i+1}: {total_tokens} tokens in {total_time:.2f}s ({tokens_per_second:.1f} tok/s)")
        stats.append({"tps": tokens_per_second})
        
    return stats

def run_hf_affine(draft_model, target_model, verifier, tokenizer, questions, max_new_tokens, logger, threshold=0.5):
    """HF speculative decoding with affine pre-filtering via wrapped draft model."""
    from speculative_decoding.models.draft_with_affine import DraftModelWithAffine
    wrapped_draft = DraftModelWithAffine(draft_model, verifier, threshold=threshold)
    logger.info("Running HF Speculative + Affine generation…")
    stats = []
    for i, q in enumerate(questions):
        input_ids = tokenizer.encode(q, return_tensors="pt").to(target_model.device)
        attention_mask = torch.ones_like(input_ids)
        start = time.time()
        with torch.no_grad():
            out = target_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                assistant_model=wrapped_draft,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        t = time.time()-start
        tok = out.size(1)-input_ids.size(1)
        logger.info(f"  HF+Affine {i+1}: {tok} tokens in {t:.2f}s ({tok/t:.1f} tok/s)")
        stats.append({"tps": tok/t})
    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=10, help="number of GSM8K questions to sample")
    parser.add_argument("--max-new", type=int, default=100)
    parser.add_argument("--affine-model", type=str, default="affine_verifier_llama_v2_kvcache.pt", help="KV-cache trained verifier")
    parser.add_argument("--affine-threshold", type=float, default=0.5)
    args = parser.parse_args()

    logger = setup_logger()

    logger.info("Loading GSM8K dataset…")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    questions = [ds[i]["question"].strip() + " Answer:" for i in range(min(args.num_samples, len(ds)))]

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
            temperature=0.7, top_k=50, top_p=0.9, do_sample=True,
        ),
    )

    loader = ModelLoader(logger=logger)
    draft, target, tok = loader.load_draft_and_target_models(cfg.draft_model, cfg.target_model)
    
    # Baseline (KV cache enabled by default in HF)
    base_stats = run_baseline(target, tok, questions, args.max_new, logger)

    # Our implementation - Basic speculative (with KV cache)
    spec_cfg_basic = SpeculativeDecodingConfig(num_assistant_tokens=5, use_cache=True, verbose=False)
    dec_basic = OptimizedSpeculativeDecoderV2(draft, target, tok, cfg, spec_cfg_basic, logger)
    basic_stats = run_spec(dec_basic, tok, questions, args.max_new, logger, "Our Basic Speculative (KV)")

    # Our implementation - Affine speculative (with KV cache)
    spec_cfg_aff = SpeculativeDecodingConfig(
        num_assistant_tokens=5,
        affine_verification=True,
        affine_model_path=args.affine_model,
        affine_accept_threshold=0.5,
        use_cache=True,
        verbose=False,
    )
    dec_aff = OptimizedSpeculativeDecoderV2(draft, target, tok, cfg, spec_cfg_aff, logger)
    affine_stats = run_spec(dec_aff, tok, questions, args.max_new, logger, "Our Affine Speculative (KV)")

    # HuggingFace native speculative decoding
    hf_spec_stats = run_hf_speculative(draft, target, tok, questions, args.max_new, logger)

    # # Load verifier for HF+Affine
    # from speculative_decoding.algorithms.affine_alignment import AffineVerifier
    # state = torch.load(args.affine_model, map_location="cpu")
    # verifier = AffineVerifier.from_state_dict(state).to(draft.device)
    # verifier.eval()

    # # HF+Affine speculative decoding
    # hf_aff_stats = run_hf_affine(draft, target, verifier, tok, questions, args.max_new, logger, threshold=args.affine_threshold)

    # # Summary
    logger.info("\n" + "="*40 + " FINAL RESULTS " + "="*40)
    logger.info(f"Baseline                : {avg(base_stats,'tps'):.1f} tok/s")
    logger.info(f"HF Speculative          : {avg(hf_spec_stats,'tps'):.1f} tok/s")
    # logger.info(f"HF+Affine Speculative   : {avg(hf_aff_stats,'tps'):.1f} tok/s")
    logger.info("-"*95)
    logger.info(f"Our Basic Speculative   : {avg(basic_stats,'tokens_per_second'):.1f} tok/s (acc {avg(basic_stats,'acceptance_rate'):.1%})")
    logger.info(f"Our Affine Speculative  : {avg(affine_stats,'tokens_per_second'):.1f} tok/s (acc {avg(affine_stats,'acceptance_rate'):.1%})")
    logger.info("="*95)

if __name__ == "__main__":
    main() 