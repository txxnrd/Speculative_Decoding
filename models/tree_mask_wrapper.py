"""
Tree Mask Model Wrapper

HuggingFace 모델을 감싸서 4D attention mask를 지원하도록 하는 래퍼.
트리 구조의 attention pattern을 구현하기 위해 필요.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from transformers.modeling_outputs import CausalLMOutputWithPast
import warnings


class TreeMaskModelWrapper(nn.Module):
    """
    HuggingFace 모델을 감싸서 custom 4D attention mask를 지원하는 래퍼
    
    트리 구조의 마스킹을 위해 attention 계산 시 custom mask를 주입합니다.
    """
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config
        
        # Hook into attention layers to inject custom masks
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks to inject custom attention masks"""
        self._custom_attn_mask = None
        
        def create_attention_hook(layer_idx):
            def attention_forward_hook(module, args):
                # PyTorch pre-forward hooks receive (module, args) tuple
                # We need to modify args which may contain kwargs
                # Unpack args - it might be (args, kwargs) or just args
                if len(args) == 2 and isinstance(args[1], dict):
                    # New style: (args_tuple, kwargs_dict)
                    args_tuple, kwargs = args
                else:
                    # Old style: just args_tuple
                    args_tuple = args
                    kwargs = {}
                
                # Check if we have a custom mask to inject
                if self._custom_attn_mask is not None:
                    # Try to inject the mask appropriately
                    if 'attention_mask' in kwargs:
                        # Replace or combine with existing mask
                        if kwargs['attention_mask'] is not None and kwargs['attention_mask'].dim() == 4:
                            # Combine with existing 4D mask
                            kwargs['attention_mask'] = kwargs['attention_mask'] + self._custom_attn_mask
                        else:
                            # Use our custom mask
                            kwargs['attention_mask'] = self._custom_attn_mask
                    elif len(args_tuple) > 1 and args_tuple[1] is not None:
                        # Attention mask might be positional arg
                        args_list = list(args_tuple)
                        if isinstance(args_list[1], torch.Tensor):
                            if args_list[1].dim() == 4:
                                args_list[1] = args_list[1] + self._custom_attn_mask
                            else:
                                args_list[1] = self._custom_attn_mask
                        args_tuple = tuple(args_list)
                
                # Return modified args in the format expected
                if kwargs:
                    return (args_tuple, kwargs)
                else:
                    return args_tuple
                
            return attention_forward_hook
        
        # Find attention modules and register hooks
        registered_count = 0
        for name, module in self.model.named_modules():
            # Look for LlamaAttention or similar attention modules specifically
            if ('attention' in name.lower() or 'self_attn' in name.lower()) and hasattr(module, 'forward'):
                # Skip projection layers and other non-attention modules
                if any(skip in name.lower() for skip in ['_proj', 'mlp', 'layer_norm', 'embed']):
                    continue
                # Only hook actual attention modules (check for q_proj as indicator)
                if hasattr(module, 'q_proj') or hasattr(module, 'query') or 'attention' in module.__class__.__name__.lower():
                    # Register pre-forward hook
                    module._forward_pre_hooks.clear()  # Clear existing hooks to avoid conflicts
                    module.register_forward_pre_hook(create_attention_hook(name))
                    registered_count += 1
        
        if registered_count == 0:
            warnings.warn(f"No attention modules found to hook in {self.model.__class__.__name__}. Tree mask may not be applied.")
        else:
            print(f"✓ Registered tree mask hooks on {registered_count} attention modules")
                    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        tree_attention_mask: Optional[torch.Tensor] = None,  # Our custom 4D mask
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass with optional tree-structured attention mask
        
        Args:
            tree_attention_mask: Optional 4D attention mask [batch, heads, seq, seq]
                                Values should be 0 (attend) or -inf (no attend)
        """
        # Set the custom mask for hooks to use
        self._custom_attn_mask = tree_attention_mask
        
        try:
            # Call the wrapped model
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs
            )
            
            return outputs
            
        finally:
            # Clear the custom mask after use
            self._custom_attn_mask = None
            
    def generate(self, *args, **kwargs):
        """Pass through to wrapped model's generate method"""
        return self.model.generate(*args, **kwargs)
        
    @property
    def device(self):
        """Get the device of the wrapped model"""
        return next(self.model.parameters()).device
        
    def __getattr__(self, name):
        """Delegate attribute access to wrapped model"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name) 