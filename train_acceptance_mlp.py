"""
Train Acceptance Probability Predictor MLP using GSM8K dataset

이 스크립트는 GSM8K 데이터셋을 사용해서:
1. Draft model과 Target model을 실행
2. Draft hidden states를 W,b로 변환
3. 변환된 hidden state와 실제 acceptance 여부를 수집
4. MLP를 학습
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
import os
from typing import List, Dict, Tuple
import argparse
from datetime import datetime

# Custom imports
from models.affine_alignment import AffineAlignment


class AcceptanceDataset(Dataset):
    """Acceptance prediction 학습을 위한 데이터셋"""
    def __init__(self, data_path: str):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        aligned_hidden = torch.tensor(item['aligned_hidden_state'], dtype=torch.float32)
        accepted = torch.tensor(item['accepted'], dtype=torch.float32)
        return aligned_hidden, accepted


class SimpleMLP(nn.Module):
    """Simple MLP for acceptance probability prediction"""
    def __init__(self, input_dim: int = 8192, hidden_dims: List[int] = [256, 128]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x).squeeze(-1)


def collect_training_data(
    draft_model,
    target_model,
    tokenizer,
    affine_alignment,
    dataset,
    num_samples: int = 1000,
    max_length: int = 512,
    device: str = "cuda"
) -> List[Dict]:
    """
    GSM8K 데이터셋에서 학습 데이터 수집
    
    Returns:
        List of {'aligned_hidden_state': tensor, 'accepted': bool}
    """
    training_data = []
    
    draft_model.eval()
    target_model.eval()
    affine_alignment.eval()
    
    # Get model devices for multi-GPU
    draft_device = next(draft_model.parameters()).device
    target_device = next(target_model.parameters()).device
    affine_device = next(affine_alignment.parameters()).device
    
    with torch.no_grad():
        for idx, item in enumerate(tqdm(dataset, total=num_samples, desc="Collecting data")):
            if idx >= num_samples:
                break
                
            # Prepare input
            question = item['question']
            answer = item['answer']
            full_text = f"Question: {question}\nAnswer: {answer}"
            
            # Tokenize
            inputs = tokenizer(full_text, return_tensors="pt", max_length=max_length, truncation=True)
            input_ids_draft = inputs['input_ids'].to(draft_device)
            input_ids_target = inputs['input_ids'].to(target_device)
            
            if input_ids_draft.shape[1] < 10:  # Skip too short sequences
                continue
            
            # Get draft model hidden states
            draft_outputs = draft_model(input_ids_draft, output_hidden_states=True)
            draft_hidden_states = draft_outputs.hidden_states[-1]  # Last layer
            
            # Get target model outputs for comparison
            target_outputs = target_model(input_ids_target)
            target_logits = target_outputs.logits
            
            # Process each position
            for pos in range(1, min(input_ids_draft.shape[1], 50)):  # Limit positions per sample
                # Draft model's prediction
                draft_logit = draft_outputs.logits[0, pos-1]
                draft_token = torch.argmax(draft_logit)
                
                # Target model's distribution
                target_logit = target_logits[0, pos-1]
                target_probs = torch.softmax(target_logit, dim=-1)
                
                # Check if draft token would be accepted
                # In real speculative decoding, this depends on sampling
                # For training, we use a threshold on target probability
                accepted = target_probs[draft_token] > 0.1  # Simple threshold
                
                # Get aligned hidden state
                draft_hidden = draft_hidden_states[0, pos-1].unsqueeze(0)
                # Move to affine alignment device and ensure correct dtype
                draft_hidden = draft_hidden.to(affine_device).to(torch.float16)
                aligned_hidden = affine_alignment(draft_hidden.unsqueeze(0)).squeeze(0)
                
                training_data.append({
                    'aligned_hidden_state': aligned_hidden.cpu().float().numpy().tolist(),  # Convert to float32 for storage
                    'accepted': int(accepted.item()),
                    'draft_token': draft_token.item(),
                    'target_prob': target_probs[draft_token].item()
                })
                
            # Clear GPU memory
            if idx % 10 == 0:
                torch.cuda.empty_cache()
    
    return training_data


def train_mlp(
    mlp: SimpleMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda"
) -> Dict:
    """Train the MLP"""
    mlp.to(device)
    optimizer = optim.Adam(mlp.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training
        mlp.train()
        train_losses = []
        
        for aligned_hidden, accepted in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            aligned_hidden = aligned_hidden.to(device)
            accepted = accepted.to(device)
            
            optimizer.zero_grad()
            predictions = mlp(aligned_hidden)
            loss = criterion(predictions, accepted)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        mlp.eval()
        val_losses = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for aligned_hidden, accepted in val_loader:
                aligned_hidden = aligned_hidden.to(device)
                accepted = accepted.to(device)
                
                predictions = mlp(aligned_hidden)
                loss = criterion(predictions, accepted)
                val_losses.append(loss.item())
                
                # Calculate accuracy
                predicted_class = (predictions > 0.5).float()
                correct += (predicted_class == accepted).sum().item()
                total += accepted.shape[0]
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        val_acc = correct / total
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
    
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to collect')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128], help='MLP hidden dimensions')
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load models
    print("Loading models...")
    draft_model_name = "/hdd1/taeyun/Llama-3.1-8B-Instruct"
    target_model_name = "/hdd1/taeyun/Llama-3.1-70B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(draft_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load affine alignment
    print("Loading affine alignment...")
    affine_alignment = AffineAlignment(4096, 8192).to(device).to(torch.float16)  # Match model dtype
    checkpoint = torch.load('affine_verifier_v4_regression.pt', map_location=device)
    affine_alignment.weight.data = checkpoint['W'].to(torch.float16)
    affine_alignment.bias.data = checkpoint['b'].to(torch.float16)
    
    # Load GSM8K dataset
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="train")
    
    # Collect training data
    data_path = f"acceptance_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    if not os.path.exists(data_path):
        print(f"Collecting {args.num_samples} training samples...")
        training_data = collect_training_data(
            draft_model, target_model, tokenizer, affine_alignment,
            dataset, num_samples=args.num_samples, device=device
        )
        
        # Save collected data
        with open(data_path, 'w') as f:
            json.dump(training_data, f)
        print(f"Saved {len(training_data)} samples to {data_path}")
    else:
        print(f"Loading existing data from {data_path}")
    
    # Create dataset and dataloaders
    full_dataset = AcceptanceDataset(data_path)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create and train MLP
    print("Training MLP...")
    mlp = SimpleMLP(input_dim=8192, hidden_dims=args.hidden_dims)
    history = train_mlp(mlp, train_loader, val_loader, num_epochs=args.num_epochs, lr=args.lr, device=device)
    
    # Save trained model
    model_path = f"acceptance_mlp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save({
        'model_state_dict': mlp.state_dict(),
        'hidden_dims': args.hidden_dims,
        'history': history,
        'args': vars(args)
    }, model_path)
    print(f"Saved trained model to {model_path}")
    
    print(f"\nFinal validation accuracy: {history['val_acc'][-1]:.4f}")


if __name__ == "__main__":
    main() 