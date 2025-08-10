#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

from speculative_decoding.algorithms.affine_alignment import learn_affine_map, AffineVerifier
from speculative_decoding.utils.config import ModelConfig
from speculative_decoding.models import ModelLoader
from speculative_decoding.utils import setup_logger

def collect_acceptance_data_kvcache(
    draft_model, target_model, tokenizer, texts, device, 
    max_pairs=50000, layer_index=-1, max_seq_len=100
):
    """
    Collect acceptance data using KV cache mode (incremental generation).
    This matches the actual inference conditions.
    """
    logger = setup_logger()
    logger.info("Collecting acceptance data with KV cache mode...")
    
    draft_hiddens = []
    target_hiddens = []
    acceptance_labels = []
    
    draft_model.eval()
    target_model.eval()
    
    total_pairs = 0
    
    with torch.no_grad():
        for text_idx, text in enumerate(tqdm(texts, desc="Processing texts")):
            if total_pairs >= max_pairs:
                break
                
            # Tokenize
            tokens = tokenizer.encode(text, add_special_tokens=True, max_length=max_seq_len, truncation=True)
            if len(tokens) < 3:  # Need at least some context
                continue
                
            input_ids = torch.tensor([tokens], device=device)
            seq_len = len(tokens)
            
            # Process incrementally (like real inference)
            draft_past_key_values = None
            target_past_key_values = None
            
            for pos in range(1, seq_len - 1):  # Start from 1, predict up to seq_len-1
                current_context = input_ids[:, :pos+1]  # Context including current position
                next_token = input_ids[:, pos+1]  # Ground truth next token
                
                # === DRAFT MODEL: Incremental forward pass ===
                if draft_past_key_values is None:
                    # First token: full context
                    draft_input = current_context
                else:
                    # Subsequent tokens: only the last token
                    draft_input = current_context[:, -1:]
                
                draft_out = draft_model(
                    input_ids=draft_input,
                    past_key_values=draft_past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True
                )
                draft_logits = draft_out.logits[:, -1, :]  # Last position logits
                draft_hidden = draft_out.hidden_states[layer_index][:, -1, :]  # Last position hidden
                draft_past_key_values = draft_out.past_key_values
                
                # === TARGET MODEL: Incremental forward pass ===
                if target_past_key_values is None:
                    # First token: full context
                    target_input = current_context
                else:
                    # Subsequent tokens: only the last token
                    target_input = current_context[:, -1:]
                
                target_out = target_model(
                    input_ids=target_input,
                    past_key_values=target_past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True
                )
                target_logits = target_out.logits[:, -1, :]  # Last position logits
                target_hidden = target_out.hidden_states[layer_index][:, -1, :]  # Last position hidden
                target_past_key_values = target_out.past_key_values
                
                # === REJECTION SAMPLING SIMULATION ===
                # Sample from draft model
                draft_probs = F.softmax(draft_logits, dim=-1)
                draft_token = torch.multinomial(draft_probs, num_samples=1).squeeze()
                
                # Check if draft token would be accepted by target model
                draft_prob = draft_probs[0, draft_token].item()
                target_probs = F.softmax(target_logits, dim=-1)
                target_prob = target_probs[0, draft_token].item()
                
                # Acceptance probability (rejection sampling)
                if draft_prob <= 0:
                    draft_prob = 1e-10
                acceptance_prob = min(1.0, target_prob / draft_prob)
                
                # CHANGED: Use actual acceptance probability as target (regression)
                # instead of binary classification
                acceptance_label = acceptance_prob
                
                # Store data
                draft_hiddens.append(draft_hidden.cpu())
                target_hiddens.append(target_hidden.cpu()) 
                acceptance_labels.append(acceptance_label)  # Now a float in [0, 1]
                
                total_pairs += 1
                
                if total_pairs >= max_pairs:
                    break
            
            if text_idx % 100 == 0:
                logger.info(f"Processed {text_idx} texts, collected {total_pairs} pairs")
    
    logger.info(f"Collected {total_pairs} total pairs")
    
    if total_pairs == 0:
        raise ValueError("No pairs collected!")
    
    # Convert to tensors - FIXED: use torch.cat to maintain 2D shape (N, H)
    draft_h = torch.cat(draft_hiddens, dim=0)
    target_h = torch.cat(target_hiddens, dim=0)
    labels = torch.tensor(acceptance_labels)  # (N,)
    
    acceptance_rate = labels.float().mean().item()
    logger.info(f"Overall acceptance rate: {acceptance_rate:.1%}")
    
    return draft_h, target_h, labels

def main():
    parser = argparse.ArgumentParser(description="Train Affine Verifier with KV Cache")
    parser.add_argument("--draft-model", type=str, required=True)
    parser.add_argument("--target-model", type=str, required=True)
    parser.add_argument("--output", type=str, default="affine_verifier.pt")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-pairs", type=int, default=50000, help="Max hidden state pairs to collect")
    parser.add_argument("--num-questions", type=int, default=1000, help="Number of questions to process from GSM8K")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs to train MLP")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mlp-hidden", type=int, default=512, help="MLP hidden layer size")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for MLP training")
    args = parser.parse_args()
    
    logger = setup_logger()
    logger.info("Training Affine Verifier with KV Cache mode")
    
    # Load models
    logger.info("Loading models...")
    loader = ModelLoader(logger=logger)
    
    draft_config = ModelConfig(model_path=args.draft_model, device=args.device)
    target_config = ModelConfig(model_path=args.target_model, device=args.device)
    
    draft_model, tokenizer = loader.load_model_and_tokenizer(draft_config)
    target_model, _ = loader.load_model_and_tokenizer(target_config)
    
    device = draft_model.device if hasattr(draft_model, 'device') else 'cuda'
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    texts = [item["question"] for item in dataset.select(range(min(len(dataset), args.num_questions)))]  # Limit for speed
    
    # Collect data
    draft_h, target_h, labels = collect_acceptance_data_kvcache(
        draft_model, target_model, tokenizer, texts, device, 
        max_pairs=args.max_pairs
    )
    
    logger.info(f"Draft hidden shape: {draft_h.shape}")
    logger.info(f"Target hidden shape: {target_h.shape}")
    logger.info(f"Labels shape: {labels.shape}")
    
    # Learn affine transformation
    logger.info("Learning affine transformation...")
    W, b = learn_affine_map(draft_h, target_h)
    
    draft_hidden_size = draft_h.shape[1]
    target_hidden_size = target_h.shape[1]
    
    logger.info(f"Learned affine map: W{W.shape}, b{b.shape}")
    
    # Create and train verifier
    logger.info("Training MLP verifier...")
    verifier = AffineVerifier(W, b, draft_hidden_size, target_hidden_size)
    verifier = verifier.to(device)
    
    # Convert data to float32 and move to device
    draft_h = draft_h.to(device).float()
    labels = labels.to(device)
    
    # Training setup
    # CHANGED: Use MSE loss for regression instead of BCE for classification
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(verifier.mlp.parameters(), lr=args.lr)
    
    # Create data loader
    dataset_tensor = TensorDataset(draft_h, labels)
    dataloader = DataLoader(dataset_tensor, batch_size=args.batch_size, shuffle=True)
    
    # Training loop
    verifier.train()
    for epoch in range(args.epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_draft, batch_labels in dataloader:
            logits = verifier(batch_draft)
            # Apply sigmoid to get probabilities in [0, 1]
            probs = torch.sigmoid(logits.squeeze())
            loss = criterion(probs, batch_labels.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy (predictions within threshold of true value)
            # For regression, we'll measure how many predictions are within 0.1 of true value
            error = torch.abs(probs - batch_labels)
            correct += (error < 0.1).sum().item()
            total += batch_labels.size(0)
        
        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.1%}")
    
    # Save model
    logger.info(f"Saving model to {args.output}")
    save_dict = {
        'W': W,
        'b': b, 
        'draft_hidden_size': draft_hidden_size,
        'target_hidden_size': target_hidden_size,
        'mlp_hidden': args.mlp_hidden,
        'mlp': verifier.mlp.state_dict()
    }
    torch.save(save_dict, args.output)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 