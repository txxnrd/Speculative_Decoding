"""
Benchmark script for Affine Alignment based Speculative Decoder

이 스크립트는 다양한 설정에서 Speculative Decoder의 성능을 측정합니다.
"""

import torch
import time
import json
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speculative_decoder import SpeculativeDecoder
from config import SpeculativeDecodingConfig


class SpeculativeDecodingBenchmark:
    """Speculative Decoding 성능 벤치마크"""
    
    def __init__(self, base_config: SpeculativeDecodingConfig):
        self.base_config = base_config
        self.results = []
        
    def benchmark_acceptance_rate(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        tree_depths: List[int] = [2, 3, 4, 5],
        max_candidates_list: List[int] = [3, 5, 7]
    ) -> pd.DataFrame:
        """
        다양한 tree configuration에서 acceptance rate 측정
        """
        results = []
        
        for depth in tree_depths:
            for max_candidates in max_candidates_list:
                # Update config
                self.base_config.tree_search.max_depth = depth
                self.base_config.tree_search.max_candidates = max_candidates
                
                # Initialize decoder
                decoder = SpeculativeDecoder(self.base_config)
                
                acceptance_rates = []
                speedups = []
                
                for prompt in prompts:
                    # Tokenize
                    input_ids = decoder.tokenizer.encode(
                        prompt, return_tensors="pt"
                    ).to(decoder.device)
                    
                    # Speculative generation
                    start_time = time.time()
                    _, stats = decoder.generate(
                        input_ids=input_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=0.8,
                        do_sample=True
                    )
                    spec_time = time.time() - start_time
                    
                    # Standard generation
                    start_time = time.time()
                    with torch.no_grad():
                        decoder.target_model.generate(
                            input_ids=input_ids,
                            max_new_tokens=max_new_tokens,
                            temperature=0.8,
                            do_sample=True
                        )
                    std_time = time.time() - start_time
                    
                    acceptance_rates.append(stats['average_acceptance_rate'])
                    speedups.append(std_time / spec_time)
                
                result = {
                    'depth': depth,
                    'max_candidates': max_candidates,
                    'avg_acceptance_rate': np.mean(acceptance_rates),
                    'std_acceptance_rate': np.std(acceptance_rates),
                    'avg_speedup': np.mean(speedups),
                    'std_speedup': np.std(speedups)
                }
                results.append(result)
                
                print(f"Depth={depth}, Candidates={max_candidates}: "
                      f"Acceptance={result['avg_acceptance_rate']:.2%}, "
                      f"Speedup={result['avg_speedup']:.2f}x")
                
        return pd.DataFrame(results)
    
    def benchmark_pruning_effectiveness(
        self,
        prompts: List[str],
        pruning_ratios: List[float] = [0.3, 0.5, 0.7],
        adaptive_settings: List[bool] = [True, False]
    ) -> pd.DataFrame:
        """
        Pruning 효과성 측정
        """
        results = []
        
        for pruning_ratio in pruning_ratios:
            for adaptive in adaptive_settings:
                # Update config
                self.base_config.pruning.pruning_ratio = pruning_ratio
                self.base_config.pruning.adaptive_pruning = adaptive
                
                # Initialize decoder
                decoder = SpeculativeDecoder(self.base_config)
                
                total_paths = []
                pruned_paths = []
                acceptance_rates = []
                
                for prompt in prompts:
                    input_ids = decoder.tokenizer.encode(
                        prompt, return_tensors="pt"
                    ).to(decoder.device)
                    
                    _, stats = decoder.generate(
                        input_ids=input_ids,
                        max_new_tokens=50,
                        temperature=0.8,
                        do_sample=True
                    )
                    
                    # Extract pruning statistics
                    if decoder.stats['pruning_stats']:
                        avg_original = np.mean([
                            s.original_paths for s in decoder.stats['pruning_stats']
                        ])
                        avg_pruned = np.mean([
                            s.pruned_paths for s in decoder.stats['pruning_stats']
                        ])
                        
                        total_paths.append(avg_original)
                        pruned_paths.append(avg_pruned)
                        acceptance_rates.append(stats['average_acceptance_rate'])
                
                result = {
                    'pruning_ratio': pruning_ratio,
                    'adaptive': adaptive,
                    'avg_total_paths': np.mean(total_paths),
                    'avg_pruned_paths': np.mean(pruned_paths),
                    'actual_pruning_ratio': 1 - np.mean(pruned_paths) / np.mean(total_paths),
                    'avg_acceptance_rate': np.mean(acceptance_rates)
                }
                results.append(result)
                
                print(f"Pruning={pruning_ratio}, Adaptive={adaptive}: "
                      f"Actual pruning={result['actual_pruning_ratio']:.2%}, "
                      f"Acceptance={result['avg_acceptance_rate']:.2%}")
                
        return pd.DataFrame(results)
    
    def benchmark_latency_breakdown(
        self,
        prompts: List[str],
        max_new_tokens: int = 100
    ) -> Dict:
        """
        각 컴포넌트의 latency 분석
        """
        decoder = SpeculativeDecoder(self.base_config)
        
        # Reset timing stats
        decoder.stats['timing'] = {k: 0 for k in decoder.stats['timing']}
        
        for prompt in prompts:
            input_ids = decoder.tokenizer.encode(
                prompt, return_tensors="pt"
            ).to(decoder.device)
            
            decoder.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                do_sample=True
            )
        
        # Compute average timings
        total_time = sum(decoder.stats['timing'].values())
        timing_breakdown = {
            component: {
                'time': time_spent,
                'percentage': (time_spent / total_time * 100) if total_time > 0 else 0
            }
            for component, time_spent in decoder.stats['timing'].items()
        }
        
        return timing_breakdown
    
    def plot_results(self, acceptance_df: pd.DataFrame, pruning_df: pd.DataFrame):
        """
        벤치마크 결과 시각화
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Acceptance rate heatmap
        pivot = acceptance_df.pivot(
            index='depth', 
            columns='max_candidates',
            values='avg_acceptance_rate'
        )
        sns.heatmap(pivot, annot=True, fmt='.2%', ax=axes[0, 0], cmap='YlOrRd')
        axes[0, 0].set_title('Acceptance Rate by Tree Configuration')
        
        # 2. Speedup heatmap
        pivot = acceptance_df.pivot(
            index='depth',
            columns='max_candidates', 
            values='avg_speedup'
        )
        sns.heatmap(pivot, annot=True, fmt='.2f', ax=axes[0, 1], cmap='Blues')
        axes[0, 1].set_title('Speedup by Tree Configuration')
        
        # 3. Pruning effectiveness
        pruning_adaptive = pruning_df[pruning_df['adaptive'] == True]
        pruning_static = pruning_df[pruning_df['adaptive'] == False]
        
        axes[1, 0].plot(
            pruning_adaptive['pruning_ratio'],
            pruning_adaptive['avg_acceptance_rate'],
            'o-', label='Adaptive', markersize=8
        )
        axes[1, 0].plot(
            pruning_static['pruning_ratio'],
            pruning_static['avg_acceptance_rate'],
            's-', label='Static', markersize=8
        )
        axes[1, 0].set_xlabel('Target Pruning Ratio')
        axes[1, 0].set_ylabel('Acceptance Rate')
        axes[1, 0].set_title('Pruning Strategy Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Actual vs target pruning ratio
        axes[1, 1].scatter(
            pruning_df['pruning_ratio'],
            pruning_df['actual_pruning_ratio'],
            s=100, alpha=0.6
        )
        axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1, 1].set_xlabel('Target Pruning Ratio')
        axes[1, 1].set_ylabel('Actual Pruning Ratio')
        axes[1, 1].set_title('Pruning Ratio: Target vs Actual')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    # Test prompts
    test_prompts = [
        "The key to successful machine learning is",
        "In the future, artificial intelligence will",
        "The most important scientific discovery of the 21st century",
        "Climate change is a global challenge that requires",
        "The role of technology in education has",
    ]
    
    # Base configuration
    config = SpeculativeDecodingConfig()
    # 실제 로컬 Llama 모델 경로 사용
    config.model.draft_model_name = "/hdd1/taeyun/Llama-3.1-8B-Instruct"
    config.model.target_model_name = "/hdd1/taeyun/Llama-3.1-70B-Instruct"
    
    # Hidden size 설정 (Llama 3.1 모델에 맞게)
    config.affine_alignment.hidden_size_draft = 4096   # Llama 3.1 8B
    config.affine_alignment.hidden_size_target = 8192  # Llama 3.1 70B
    
    # Run benchmarks
    benchmark = SpeculativeDecodingBenchmark(config)
    
    print("=== Running Acceptance Rate Benchmark ===")
    acceptance_results = benchmark.benchmark_acceptance_rate(
        test_prompts,
        max_new_tokens=50,
        tree_depths=[2, 3, 4],
        max_candidates_list=[3, 5, 7]
    )
    
    print("\n=== Running Pruning Effectiveness Benchmark ===")
    pruning_results = benchmark.benchmark_pruning_effectiveness(
        test_prompts,
        pruning_ratios=[0.3, 0.5, 0.7],
        adaptive_settings=[True, False]
    )
    
    print("\n=== Latency Breakdown ===")
    latency_breakdown = benchmark.benchmark_latency_breakdown(
        test_prompts[:3],  # Use fewer prompts for latency test
        max_new_tokens=50
    )
    
    for component, stats in latency_breakdown.items():
        print(f"{component}: {stats['time']:.3f}s ({stats['percentage']:.1f}%)")
    
    # Save results
    results = {
        'acceptance_results': acceptance_results.to_dict(),
        'pruning_results': pruning_results.to_dict(),
        'latency_breakdown': latency_breakdown
    }
    
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot results
    benchmark.plot_results(acceptance_results, pruning_results)
    
    print("\nBenchmark complete! Results saved to benchmark_results.json")


if __name__ == "__main__":
    main() 