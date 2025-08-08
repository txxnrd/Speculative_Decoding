import argparse
from pathlib import Path
import sys
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from speculative_decoding.models import ModelLoader
from speculative_decoding.utils import Config, setup_logger
from speculative_decoding.utils.config import ModelConfig
from speculative_decoding.algorithms.affine_alignment import learn_affine_map, AffineVerifier


def collect_acceptance_data(draft_model, target_model, tokenizer, texts, device, max_pairs=50000, layer_index=-1):
    """Collect draft hiddens, target hiddens, and actual acceptance labels."""
    draft_hiddens = []
    target_hiddens = []
    acceptance_labels = []
    
    draft_model.eval()
    target_model.eval()
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Collecting acceptance data"):
            if len(draft_hiddens) >= max_pairs:
                break
                
            # Tokenize
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            input_ids = enc["input_ids"].to(device)
            seq_len = input_ids.size(1)
            
            if seq_len < 2:  # Need at least 2 tokens for prediction
                continue
            
            # Process each position (except last, since we need next token)
            for pos in range(seq_len - 1):
                context = input_ids[:, :pos+1]  # Context up to current position
                next_token = input_ids[:, pos+1]  # Ground truth next token
                
                # Get draft model's prediction and hidden state
                draft_out = draft_model(input_ids=context, output_hidden_states=True, return_dict=True)
                draft_logits = draft_out.logits[:, -1, :]  # Last position logits
                draft_hidden = draft_out.hidden_states[layer_index][:, -1, :]  # Last position hidden
                
                # Get target model's prediction and hidden state  
                target_out = target_model(input_ids=context, output_hidden_states=True, return_dict=True)
                target_logits = target_out.logits[:, -1, :]  # Last position logits
                target_hidden = target_out.hidden_states[layer_index][:, -1, :]  # Last position hidden
                
                # Sample from draft model
                draft_probs = F.softmax(draft_logits, dim=-1)
                draft_token = torch.multinomial(draft_probs, num_samples=1).squeeze()
                
                # Check if draft token would be accepted by target model (rejection sampling)
                draft_prob = draft_probs[0, draft_token].item()
                target_probs = F.softmax(target_logits, dim=-1)
                target_prob = target_probs[0, draft_token].item()
                
                if draft_prob <= 0:
                    draft_prob = 1e-10
                
                acceptance_prob = min(1.0, target_prob / draft_prob)
                
                # Determine acceptance (we use deterministic threshold instead of sampling for stable training)
                is_accepted = 1 if acceptance_prob >= 0.5 else 0
                
                # Store data
                draft_hiddens.append(draft_hidden.cpu())
                target_hiddens.append(target_hidden.cpu())
                acceptance_labels.append(is_accepted)
                
                if len(draft_hiddens) >= max_pairs:
                    break
    
    # Convert to tensors
    draft_h = torch.cat(draft_hiddens, dim=0)  # (N, H_draft)
    target_h = torch.cat(target_hiddens, dim=0)  # (N, H_target)
    labels = torch.tensor(acceptance_labels, dtype=torch.long)  # (N,)
    
    return draft_h, target_h, labels


def main():
    parser = argparse.ArgumentParser("Train affine verifier using GSM8K")
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--output", type=str, default="affine_verifier.pt")
    parser.add_argument("--max-pairs", type=int, default=10000)  # Reduced for faster testing
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    logger = setup_logger()

    # Load models
    loader = ModelLoader(logger=logger)
    draft_model, target_model, tokenizer = loader.load_draft_and_target_models(
        ModelConfig(model_path=args.draft_model, device=args.device),
        ModelConfig(model_path=args.target_model, device="auto")  # Use auto for 70B model
    )

    draft_model.eval()
    target_model.eval()

    # Load GSM8K dataset (train split)
    logger.info("Loading GSM8K dataset...")
    ds = load_dataset("gsm8k", "main", split="train[:1%]")  # Use smaller subset for testing
    texts = [ex["question"] for ex in ds]

    # Collect acceptance data
    logger.info("Collecting acceptance data...")
    draft_h, target_h, labels = collect_acceptance_data(
        draft_model, target_model, tokenizer, texts, args.device, max_pairs=args.max_pairs
    )
    logger.info(f"Collected {draft_h.size(0)} samples")
    logger.info(f"Acceptance rate: {labels.float().mean():.3f}")

    # Learn affine map
    logger.info("Learning affine transformation...")
    W, b = learn_affine_map(draft_h, target_h)
    logger.info(f"Learned affine map: W shape {W.shape}, b shape {b.shape}")

    # Train MLP classifier
    logger.info("Training MLP classifier...")
    draft_hidden_size = draft_h.size(1)
    target_hidden_size = target_h.size(1)
    verifier = AffineVerifier(W, b, draft_hidden_size, target_hidden_size)
    
    # Move to GPU for training
    verifier = verifier.to(args.device)
    draft_h = draft_h.to(args.device).float()  # Convert to float32 to match affine layer
    labels = labels.to(args.device)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(verifier.mlp.parameters(), lr=args.lr)
    
    for epoch in range(args.epochs):
        # Forward pass
        logits = verifier(draft_h)
        loss = criterion(logits, labels.float())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        preds = (torch.sigmoid(logits) > 0.5).long()
        accuracy = (preds == labels).float().mean()
        
        logger.info(f"Epoch {epoch+1}/{args.epochs}: Loss={loss.item():.4f}, Accuracy={accuracy.item():.3f}")

    # Save
    logger.info("Saving trained verifier...")
    torch.save(verifier.to_state_dict(), args.output)
    logger.info(f"Saved affine verifier to {args.output}")


if __name__ == "__main__":
    main() 