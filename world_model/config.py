"""
Configuration for World Model.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class WorldModelConfig:
    """Configuration for the World Model."""
    
    # Model architecture
    model_type: str = "diffusion"  # "diffusion", "autoregressive", "hybrid"
    hidden_dim: int = 512
    num_layers: int = 8
    
    # Visual processing
    image_size: int = 256
    latent_dim: int = 128
    
    # Generation settings
    num_diffusion_steps: int = 50
    guidance_scale: float = 7.5
    
    # Training settings
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # Checkpoint settings
    pretrained_path: Optional[str] = None
    checkpoint_every: int = 1000
