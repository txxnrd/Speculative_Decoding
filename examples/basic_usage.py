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
    
    # 모델 설정 - 디버깅을 위해 draft와 target 모두 8B로 설정
    config.model.draft_model_name = "/hdd1/taeyun/Llama-3.1-8B-Instruct"
    config.model.target_model_name = "/hdd1/taeyun/Llama-3.1-8B-Instruct"
    
    # Hidden size 설정 (둘 다 Llama 3.1 8B이므로 같은 크기)
    config.affine_alignment.hidden_size_draft = 4096   # 8B hidden size
    config.affine_alignment.hidden_size_target = 4096  # 8B hidden size
    
    # Load pre-trained alignment weights - 비활성화 (draft와 target이 같은 크기이므로 불필요)
    # config.affine_alignment.alignment_checkpoint = "/home/taeyun/Speculative_Decoding/affine_verifier_v4_regression.pt"
    config.affine_alignment.alignment_checkpoint = None  # Identity mapping으로 충분
    
    # MLP 설정 (8B hidden size에 맞게 조정)
    config.mlp.hidden_dims = [256, 128]  # 4096 -> 256 -> 128 -> 1
    
    # Tree search 파라미터 - 보수적으로 축소
    config.tree_search.max_candidates = 2  # 3->2: 분기 수 감소
    config.tree_search.max_depth = 2      # 4->2: 깊이 대폭 축소
    config.tree_search.temperature = 0.3   # 0.8->0.3: 더 확실한 토큰만 선택
    
    # Pruning 설정 - 더 공격적으로
    config.pruning.min_acceptance_prob = 0.3  # 0.1->0.3: 낮은 확률 경로 조기 제거
    config.pruning.pruning_ratio = 0.7       # 0.5->0.7: 더 많이 가지치기
    config.pruning.top_k_paths = 2           # 4->2: 검증할 경로 수 축소
    config.profile = True
    
    # 2. Speculative Decoder 초기화
    print("Initializing Speculative Decoder...")
    print(f"Draft model: {config.model.draft_model_name}")
    print(f"Target model: {config.model.target_model_name}")
    
    decoder = SpeculativeDecoder(config)
    
    # Load trained acceptance MLP - 비활성화 (디버깅 모드에서는 불필요)
    # mlp_path = "/home/taeyun/Speculative_Decoding/best_acceptance_mlp.pt"
    # if os.path.exists(mlp_path):
    #     decoder.load_acceptance_mlp(mlp_path)
    #     print("✓ Loaded trained acceptance MLP")
    # else:
    #     print("⚠ Warning: No trained acceptance MLP found, using random initialization")
    print("⚠ Debug mode: Using random MLP initialization (draft=target model)")
    
    # 3. 텍스트 생성 예시
    prompt = "The future of artificial intelligence is"
    
    # Tokenize input
    tokenizer = decoder.tokenizer
    
    # Llama 모델의 경우 padding token 설정이 필요할 수 있음
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(decoder.device)
    attention_mask = torch.ones_like(input_ids, device=decoder.device)
    
    print(f"\nPrompt: {prompt}")
    print("Generating text with speculative decoding...")
    
    # Generate
    generated_ids, stats = decoder.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=50,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        do_sample=False
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
    print(f"Effective acceptance rate (vs verified tokens): {stats.get('effective_acceptance_rate', 0):.2%}")
    
    print("\n=== Timing Breakdown ===")
    total_time = sum(stats['timing'].values())
    for component, time_spent in stats['timing'].items():
        percentage = (time_spent / total_time * 100) if total_time > 0 else 0
        print(f"{component}: {time_spent:.3f}s ({percentage:.1f}%)")
    print(f"Total: {total_time:.3f}s")
    
    # Diagnostics
    if 'diagnostics' in stats:
        print("\n=== Diagnostics (first 5 iterations) ===")
        for i, d in enumerate(stats['diagnostics'][:5]):
            print(f"iter {i}: best_len={d.get('best_path_len')}, num_paths={d.get('num_paths_verified')}, "
                  f"avg_kl={d.get('avg_kl_draft_target', 'na')}, overlap@10={d.get('avg_topk_overlap@10', 'na')}")
        print(f"avg_best_path_len={stats.get('avg_best_path_len', 0):.2f}, avg_num_paths_verified={stats.get('avg_num_paths_verified', 0):.2f}")
    
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