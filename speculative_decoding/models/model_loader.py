"""Model loader for Qwen3 models."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional
import logging
from ..utils.config import ModelConfig


class ModelLoader:
    """Handles loading and management of language models."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize model loader."""
        self.logger = logger or logging.getLogger(__name__)
        
    def load_model_and_tokenizer(
        self,
        config: ModelConfig
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load a model and its tokenizer.
        
        Args:
            config: Model configuration
            
        Returns:
            Tuple of (model, tokenizer)
        """
        self.logger.info(f"Loading model from {config.model_path}")
        
        # Convert dtype string to torch dtype
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(config.dtype, torch.float16)
        
        # Load tokenizer (try fast, fallback to slow if necessary)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_path,
                trust_remote_code=True,
                use_fast=True  # Prefer fast tokenizer for speed
            )
        except Exception as e:
            self.logger.warning(
                f"Fast tokenizer loading failed ({e}). Falling back to slow tokenizer. This may be slower but more robust."
            )
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_path,
                trust_remote_code=True,
                use_fast=False
            )
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Model loading kwargs
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": config.device,
            "trust_remote_code": True,
        }
        
        # Handle quantization
        if config.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"
        elif config.load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            model_kwargs["device_map"] = "auto"
            
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            **model_kwargs
        )
        
        # Move to device if not using device_map="auto"
        if not (config.load_in_8bit or config.load_in_4bit or config.device == "auto"):
            model = model.to(config.device)
            
        model.eval()
        
        device_info = "multiple GPUs" if config.device == "auto" else config.device
        self.logger.info(f"Model loaded successfully on {device_info}")
        self.logger.info(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
        
        return model, tokenizer
    
    def load_draft_and_target_models(
        self,
        draft_config: ModelConfig,
        target_config: ModelConfig
    ) -> Tuple[AutoModelForCausalLM, AutoModelForCausalLM, AutoTokenizer]:
        """
        Load both draft and target models.
        
        Args:
            draft_config: Configuration for draft model
            target_config: Configuration for target model
            
        Returns:
            Tuple of (draft_model, target_model, tokenizer)
        """
        self.logger.info("Loading draft model...")
        draft_model, tokenizer = self.load_model_and_tokenizer(draft_config)
        
        self.logger.info("Loading target model...")
        target_model, _ = self.load_model_and_tokenizer(target_config)
        
        return draft_model, target_model, tokenizer 