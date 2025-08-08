"""Configuration management for speculative decoding."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for a model."""
    model_path: str
    device: str = "cuda"
    dtype: str = "float16"
    load_in_8bit: bool = False
    load_in_4bit: bool = False


@dataclass
class SamplingConfig:
    """Configuration for sampling."""
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    do_sample: bool = True


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""
    draft_k: int = 4  # Number of tokens to draft
    max_draft_len: int = 100  # Maximum draft sequence length
    acceptance_threshold: float = 0.0  # Minimum acceptance probability


@dataclass
class Config:
    """Main configuration class."""
    # Model configurations
    draft_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        model_path="/raid/taeyun/Qwen3-8B"
    ))
    target_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        model_path="/raid/taeyun/Qwen3-14B"
    ))
    
    # Sampling configuration
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    
    # Speculative decoding configuration
    speculative: SpeculativeConfig = field(default_factory=SpeculativeConfig)
    
    # Generation settings
    max_new_tokens: int = 100
    seed: Optional[int] = None
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Parse model configs
        draft_model = ModelConfig(**data.get('draft_model', {}))
        target_model = ModelConfig(**data.get('target_model', {}))
        
        # Parse other configs
        sampling = SamplingConfig(**data.get('sampling', {}))
        speculative = SpeculativeConfig(**data.get('speculative', {}))
        
        return cls(
            draft_model=draft_model,
            target_model=target_model,
            sampling=sampling,
            speculative=speculative,
            max_new_tokens=data.get('max_new_tokens', 100),
            seed=data.get('seed')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'draft_model': self.draft_model.__dict__,
            'target_model': self.target_model.__dict__,
            'sampling': self.sampling.__dict__,
            'speculative': self.speculative.__dict__,
            'max_new_tokens': self.max_new_tokens,
            'seed': self.seed
        } 