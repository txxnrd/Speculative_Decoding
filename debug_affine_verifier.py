#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from speculative_decoding.algorithms.affine_alignment import AffineVerifier

def debug_affine_verifier():
    print("=== Affine Verifier Debug ===")
    
    # Load the affine verifier
    try:
        state = torch.load("affine_verifier_llama_v2.pt", map_location="cpu")
        print(f"✓ Loaded affine verifier state dict")
        print(f"State dict keys: {list(state.keys())}")
        
        # Check the state dict structure
        if 'affine.weight' in state:
            print(f"Affine weight shape: {state['affine.weight'].shape}")
        if 'affine.bias' in state:
            print(f"Affine bias shape: {state['affine.bias'].shape}")
        if 'mlp.0.weight' in state:
            print(f"MLP layer 0 weight shape: {state['mlp.0.weight'].shape}")
        
        verifier = AffineVerifier.from_state_dict(state)
        verifier.eval()
        print(f"✓ Created AffineVerifier instance")
        
        # Test with some dummy hidden states
        # Llama 8B has 4096 hidden size
        dummy_hidden = torch.randn(1, 4096)
        
        with torch.no_grad():
            logits = verifier(dummy_hidden)
            prob = torch.sigmoid(logits).item()
            
        print(f"✓ Test forward pass completed")
        print(f"Input shape: {dummy_hidden.shape}")
        print(f"Output logits: {logits.item():.4f}")
        print(f"Output probability: {prob:.4f}")
        
        # Test with multiple samples
        print("\n=== Testing with multiple samples ===")
        test_samples = torch.randn(10, 4096)
        
        with torch.no_grad():
            logits_batch = verifier(test_samples)
            probs_batch = torch.sigmoid(logits_batch)
        
        print(f"Batch logits range: [{logits_batch.min().item():.4f}, {logits_batch.max().item():.4f}]")
        print(f"Batch probs range: [{probs_batch.min().item():.4f}, {probs_batch.max().item():.4f}]")
        print(f"Mean probability: {probs_batch.mean().item():.4f}")
        
        # Test different thresholds
        print(f"\n=== Threshold Analysis ===")
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        for thresh in thresholds:
            passed = (probs_batch >= thresh).sum().item()
            print(f"Threshold {thresh}: {passed}/10 samples pass ({passed/10*100:.1f}%)")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_affine_verifier() 