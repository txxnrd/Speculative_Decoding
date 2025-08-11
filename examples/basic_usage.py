"""
Basic usage example of Affine Alignment based Speculative Decoder

이 예시는 Speculative Decoder의 기본 사용법을 보여줍니다.
"""

import torch
import sys
import os

# 부모 디렉토리를 Python path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# 디버깅용 출력
print(f"Current directory: {current_dir}")
print(f"Parent directory: {parent_dir}")
print(f"Python path: {sys.path[:2]}")

from speculative_decoder import SpeculativeDecoder
from config import SpeculativeDecodingConfig
from transformers import AutoTokenizer


def main():
    # 1. Configuration 설정
    config = SpeculativeDecodingConfig()
    
    # 모델 설정 - 실제 로컬 Llama 모델 경로 사용
    config.model.draft_model_name = "/hdd1/taeyun/Llama-3.1-8B-Instruct"
    config.model.target_model_name = "/hdd1/taeyun/Llama-3.1-70B-Instruct"
    
    # Hidden size 설정 (Llama 3.1 모델에 맞게)
    config.affine_alignment.hidden_size_draft = 4096   # Llama 3.1 8B
    config.affine_alignment.hidden_size_target = 8192  # Llama 3.1 70B
    
    # Tree search 파라미터
    config.tree_search.max_candidates = 3
    config.tree_search.max_depth = 4
    config.tree_search.temperature = 0.8
    
    # Pruning 설정
    config.pruning.min_acceptance_prob = 0.1
    config.pruning.pruning_ratio = 0.5
    
    # 2. Speculative Decoder 초기화
    print("Initializing Speculative Decoder...")
    print(f"Draft model: {config.model.draft_model_name}")
    print(f"Target model: {config.model.target_model_name}")
    
    decoder = SpeculativeDecoder(config)
    
    # 3. 텍스트 생성 예시
    prompt = "The future of artificial intelligence is"
    
    # Tokenize input
    tokenizer = decoder.tokenizer
    
    # Llama 모델의 경우 padding token 설정이 필요할 수 있음
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(decoder.device)
    
    print(f"\nPrompt: {prompt}")
    print("Generating text with speculative decoding...")
    
    # Generate
    generated_ids, stats = decoder.generate(
        input_ids=input_ids,
        max_new_tokens=50,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        do_sample=True
    )
    
    # Decode output
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print(f"\nGenerated text: {generated_text}")
    
    # 4. 통계 출력
    print("\n=== Generation Statistics ===")
    print(f"Total iterations: {stats['total_iterations']}")
    print(f"Total draft tokens: {stats['total_draft_tokens']}")
    print(f"Total accepted tokens: {stats['total_accepted_tokens']}")
    print(f"Average acceptance rate: {stats['average_acceptance_rate']:.2%}")
    print(f"Tokens per iteration: {stats['tokens_per_iteration']:.2f}")
    
    if 'average_pruning_ratio' in stats:
        print(f"Average pruning ratio: {stats['average_pruning_ratio']:.2%}")
    
    print("\n=== Timing Breakdown ===")
    total_time = sum(stats['timing'].values())
    for component, time_spent in stats['timing'].items():
        percentage = (time_spent / total_time * 100) if total_time > 0 else 0
        print(f"{component}: {time_spent:.3f}s ({percentage:.1f}%)")
    print(f"Total: {total_time:.3f}s")
    
    # 5. 일반 생성과 비교 (옵션)
    print("\n\nComparing with standard generation...")
    
    # Standard generation with target model
    import time
    start_time = time.time()
    
    with torch.no_grad():
        standard_output = decoder.target_model.generate(
            input_ids=input_ids,
            max_new_tokens=50,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
    
    standard_time = time.time() - start_time
    standard_text = tokenizer.decode(standard_output[0], skip_special_tokens=True)
    
    print(f"Standard generation time: {standard_time:.3f}s")
    print(f"Speculative generation time: {total_time:.3f}s")
    print(f"Speedup: {standard_time / total_time:.2f}x")
    

if __name__ == "__main__":
    main() 