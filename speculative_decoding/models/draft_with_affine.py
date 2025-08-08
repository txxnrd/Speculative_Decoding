from __future__ import annotations

"""Draft model wrapper that applies affine verifier pre-filtering to the sequences
returned by HuggingFace's internal speculative-decoding draft generation call.

This lets us plug our affine mapping strategy into HF's highly-optimized
`generate(assistant_model=…)` path **without** forking the whole GenerationMixin.
"""

from typing import Optional
import torch
from transformers import PreTrainedModel
try:
    from transformers import GenerationConfig
except Exception:  # older versions
    GenerationConfig = None  # type: ignore

# The HF generate with return_dict_in_generate=True yields a ModelOutput subclass
# that at least has `.sequences`, `.scores`, and (if requested) `.hidden_states`.

class DraftModelWithAffine(PreTrainedModel):
    """Thin wrapper around an existing draft model that trims its draft tokens
    using an affine verifier before they reach the target model."""

    def __init__(self, base_model: PreTrainedModel, affine_verifier, threshold: float = 0.5):
        # IMPORTANT: initialize parent before assigning Module attributes
        super().__init__(base_model.config)
        self.base_model = base_model
        self.affine_verifier = affine_verifier
        self.threshold = threshold

        # Ensure generation_config exists and carries assistant fields expected by HF
        gen_cfg = getattr(base_model, "generation_config", None)
        if gen_cfg is None and GenerationConfig is not None:
            # Create from model config if possible
            try:
                gen_cfg = GenerationConfig.from_model_config(base_model.config)
            except Exception:
                gen_cfg = GenerationConfig()
        self.generation_config = gen_cfg if gen_cfg is not None else getattr(self, "generation_config", None)
        
        # Inject defaults if missing
        if self.generation_config is not None:
            if not hasattr(self.generation_config, "num_assistant_tokens"):
                setattr(self.generation_config, "num_assistant_tokens", 5)
            if not hasattr(self.generation_config, "assistant_confidence_threshold"):
                setattr(self.generation_config, "assistant_confidence_threshold", 0.3)
        # tie parameters etc. not needed because we delegate

    # ------------------------------------------------------------------
    # Delegation helpers
    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs):  # noqa
        return self.base_model(*args, **kwargs)

    def __getattr__(self, name):
        # Delegate everything else to the underlying model
        if name == "base_model":
            return super().__getattribute__(name)
        return getattr(self.base_model, name)

    # ------------------------------------------------------------------
    # Overridden generate that applies filtering
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, *args, **kwargs):  # noqa: C901  complexity okay (wrapper)
        # We enforce hidden_state output so we can filter.
        kwargs["output_hidden_states"] = True
        kwargs["return_dict_in_generate"] = True

        out = self.base_model.generate(*args, **kwargs)

        # out.sequences: (B, prompt + K)
        # Determine draft portion length K from max_new_tokens argument
        max_new = kwargs.get("max_new_tokens") or kwargs.get("max_length")
        if max_new is None:
            # Fallback: assume last tokens except prompt (this matches HF usage)
            prompt_len = args[0].shape[1]
            draft_tokens = out.sequences[:, prompt_len:]
        else:
            draft_tokens = out.sequences[:, -max_new:]

        # hidden_states is a tuple(layer) of tensors (B, seq, H)
        # Take final layer hidden for draft part
        hidden = out.hidden_states[-1][:, -draft_tokens.size(1):, :]  # (B,K,H)

        # Affine verifier operates token-wise. Compute accept probs.
        B, K, _ = hidden.shape
        hidden_flat = hidden.reshape(B * K, -1)
        accept_p = torch.sigmoid(self.affine_verifier(hidden_flat)).view(B, K)
        mask = accept_p >= self.threshold
        # keep prefix until first reject (inclusive reject? we drop reject)
        keep_lengths = ( (~mask).cumsum(-1) == 0 ).sum(-1)  # (B,)
        max_keep = keep_lengths.max().item()
        if max_keep == 0:
            max_keep = 1  # always keep at least 1 token to avoid loops
            keep_lengths = torch.ones(B, dtype=torch.long, device=draft_tokens.device)

        # Build trimmed sequences per batch (pad with last kept token for shape)
        trimmed_tokens = []
        for b in range(B):
            k = keep_lengths[b].item()
            trimmed = draft_tokens[b, :k]
            if k < max_keep:
                pad = trimmed[-1:].repeat(max_keep - k)
                trimmed = torch.cat([trimmed, pad])
            trimmed_tokens.append(trimmed)
        trimmed_tokens = torch.stack(trimmed_tokens, dim=0)

        # Replace tokens in out.sequences
        seq_prefix = out.sequences[:, :-draft_tokens.size(1)]
        new_sequences = torch.cat([seq_prefix, trimmed_tokens], dim=1)
        out.sequences = new_sequences

        # Also slice scores and hidden_states if present
        if hasattr(out, "scores") and out.scores is not None:
            out.scores = [s[:, :max_keep, :] for s in out.scores]
        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            out.hidden_states = tuple(h[:, :seq_prefix.size(1) + max_keep, :] for h in out.hidden_states)

        return out 