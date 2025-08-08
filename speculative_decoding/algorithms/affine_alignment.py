from __future__ import annotations

"""Affine alignment trainer and verifier for Research Idea 2.

Offline stage learns an affine map W, b that projects draft hidden states to
(target) hidden space using least-squares. The online verifier applies the
pre-learned map followed by a small MLP to predict acceptance probability.

This file intentionally keeps the implementation lightweight so that the main
speculative decoding loop can optionally plug it in at runtime.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict


class AffineVerifier(nn.Module):
    """Frozen affine map (W, b) + 2-layer MLP for accept probability."""

    def __init__(self, W: torch.Tensor, b: torch.Tensor, draft_hidden_size: int, target_hidden_size: int, mlp_hidden: int = 256):
        super().__init__()
        # Affine projection layer (frozen)
        self.affine = nn.Linear(draft_hidden_size, target_hidden_size)
        with torch.no_grad():
            self.affine.weight.copy_(W)
            self.affine.bias.copy_(b)
        for p in self.affine.parameters():
            p.requires_grad = False

        # Small MLP classifier (trainable offline, frozen online)
        self.mlp = nn.Sequential(
            nn.Linear(target_hidden_size, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden // 2, 1)
        )

    def forward(self, draft_hidden: torch.Tensor) -> torch.Tensor:  # (B,H_draft) -> (B,1)
        h_tilde = self.affine(draft_hidden)
        logits = self.mlp(h_tilde)
        return logits.squeeze(-1)  # shape (B,)

    # ---------------------------------------------------------------------
    # Utilities for saving / loading
    # ---------------------------------------------------------------------
    def to_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "W": self.affine.weight.detach().cpu(),
            "b": self.affine.bias.detach().cpu(),
            "draft_hidden_size": self.affine.in_features,
            "target_hidden_size": self.affine.out_features,
            "mlp_hidden": self.mlp[0].out_features,
            "mlp": self.mlp.state_dict()
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, torch.Tensor]) -> "AffineVerifier":
        W = state["W"]
        b = state["b"]
        
        # Handle both tensor and int types for hidden sizes for robustness
        draft_hidden_size = state["draft_hidden_size"]
        if hasattr(draft_hidden_size, 'item'):
            draft_hidden_size = draft_hidden_size.item()
            
        target_hidden_size = state["target_hidden_size"]
        if hasattr(target_hidden_size, 'item'):
            target_hidden_size = target_hidden_size.item()
        
        # Build a temporary model first with any valid mlp_hidden (will rebuild below if needed)
        mlp_hidden_meta = state.get("mlp_hidden", 256)  # may be inconsistent with saved mlp weights
        if hasattr(mlp_hidden_meta, 'item'):
            mlp_hidden_meta = mlp_hidden_meta.item()
        
        model = cls(W, b, draft_hidden_size, target_hidden_size, mlp_hidden=mlp_hidden_meta)
        
        # If an MLP state dict is provided, reconstruct the MLP to MATCH its shapes
        mlp_state = state.get("mlp")
        if isinstance(mlp_state, dict):
            # Infer layer sizes from checkpoint
            # Expected keys like '0.weight', '2.weight', '4.weight'
            first_w = None
            second_w = None
            for k, v in mlp_state.items():
                if k.endswith("0.weight"):
                    first_w = v
                elif k.endswith("2.weight"):
                    second_w = v
            if first_w is not None:
                hidden1 = first_w.shape[0]
                # Fall back if second_w missing
                hidden2 = second_w.shape[0] if second_w is not None else max(1, hidden1 // 2)
                # Rebuild MLP to match checkpoint shapes
                model.mlp = nn.Sequential(
                    nn.Linear(target_hidden_size, hidden1),
                    nn.ReLU(),
                    nn.Linear(hidden1, hidden2),
                    nn.ReLU(),
                    nn.Linear(hidden2, 1)
                )
            # Finally, load weights
            model.mlp.load_state_dict(mlp_state)
        
        return model


# -------------------------------------------------------------------------
# Offline training utilities
# -------------------------------------------------------------------------

def learn_affine_map(
    draft_hidden: torch.Tensor,  # (N,H_draft)
    target_hidden: torch.Tensor  # (N,H_target)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve min_{W,b} || W h_draft + b - h_target ||^2 using least squares.

    Returns
    -------
    W : (H_target, H_draft) weight matrix
    b : (H_target,) bias vector
    """
    # Remove assertion - draft and target can have different hidden dimensions
    assert draft_hidden.size(0) == target_hidden.size(0), "Batch size must match"
    
    # Move to CPU for lstsq and ensure same dtype
    draft_hidden_cpu = draft_hidden.to("cpu").float()  # Convert to float32
    target_hidden_cpu = target_hidden.to("cpu").float()  # Convert to float32

    # Append ones for bias term: [h_draft; 1]
    ones = torch.ones(draft_hidden_cpu.size(0), 1, device="cpu")
    A = torch.cat([draft_hidden_cpu, ones], dim=1)  # (N, H_draft+1)
    B = target_hidden_cpu  # (N, H_target)

    # Solve A X = B in least-squares sense where X = [W^T, b]^T of shape (H_draft+1, H_target)
    result = torch.linalg.lstsq(A, B)  # Note: arguments are reversed compared to torch.lstsq
    X = result.solution  # Shape: (H_draft+1, H_target)
    W = X[:-1].T.contiguous()  # (H_target, H_draft)
    b = X[-1].contiguous()     # (H_target,)
    return W.to(draft_hidden.device), b.to(draft_hidden.device) 